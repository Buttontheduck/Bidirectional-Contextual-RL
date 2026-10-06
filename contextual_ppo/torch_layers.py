"""
Recurrent network builders for Contextual_PPO. The xLSTM builders and the context encoder are the ones of
recurrent SAC (``stable_baselines3.recurrent_sac.torch_layers``), imported rather than copied, so both
algorithms always run the same encoder code. Contextual_PPO only adds resets inside a sequence, an
optional output activation of the context head, a Gaussian context head (mean and variance) and an optional
auxiliary head trained with supervised targets.
"""

from collections.abc import Mapping
from typing import Any

import torch as th
from torch import nn
from torch.nn import functional as F
from xlstm import xLSTMBlockStackConfig

from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.recurrent_sac.torch_layers import DEFAULT_XLSTM_CONFIG, XLSTMContextEncoder, create_xlstm

__all__ = [
    "CONTEXT_ACTIVATIONS",
    "CONTEXT_HEADS",
    "DEFAULT_XLSTM_CONFIG",
    "GaussianContextOutput",
    "XLSTMContextEncoder",
    "XLSTMRolloutEncoder",
    "create_xlstm",
]

# Output activations of the context head, by config name
CONTEXT_ACTIVATIONS: dict[str, type[nn.Module]] = {"sigmoid": nn.Sigmoid}
# What the context head outputs: "point", the context itself; "gaussian", the mean and variance of the context
CONTEXT_HEADS = ("point", "gaussian")
# Lower bound of the Gaussian head's variance, which keeps the likelihood finite (torch's gaussian_nll_loss eps)
MIN_CONTEXT_VARIANCE = 1e-6


class GaussianContextOutput(nn.Module):
    """
    End of the Gaussian context head, without parameters: splits the ``2 * context_dim`` outputs of the last
    linear layer into the mean (through the optional mean activation) and the variance
    ``softplus(.) + min_variance``, and returns them concatenated, ``[mean, variance]``.

    :param context_dim: Dimension of the context, i.e. of the mean and of the variance
    :param mean_activation: Activation of the mean (a key of ``CONTEXT_ACTIVATIONS``), None for a linear mean
    :param min_variance: Lower bound of the variance
    """

    def __init__(self, context_dim: int, mean_activation: str | None = None, min_variance: float = MIN_CONTEXT_VARIANCE) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.mean_activation = nn.Identity() if mean_activation is None else CONTEXT_ACTIVATIONS[mean_activation]()
        self.min_variance = min_variance

    def forward(self, outputs: th.Tensor) -> th.Tensor:
        mean, raw_variance = outputs.split(self.context_dim, dim=-1)
        return th.cat([self.mean_activation(mean), F.softplus(raw_variance) + self.min_variance], dim=-1)


class XLSTMRolloutEncoder(XLSTMContextEncoder):
    """
    ``XLSTMContextEncoder`` for the complete rollout streams of Contextual_PPO, which can hold several episodes:
    ``unroll()`` also takes ``episode_starts`` and resets the memory inside the sequence.

    The context head (``context_projection``) maps the xLSTM embedding to ``context_dim`` through the hidden
    layers ``projection_net_arch``. ``context_activation`` optionally ends it with an activation, e.g.
    ``"sigmoid"`` with ``context_dim=1`` and ``projection_net_arch=[64]``: embedding -> 64 -> 1 -> sigmoid, so
    the context ``z_t`` is one number in (0, 1). The activation has no parameters: the state dict, state layout
    and ``STATE_BATCH_AXIS`` are those of the base class with the same arguments.

    With ``context_head="gaussian"``, the head estimates the context as a diagonal Gaussian: its last linear
    layer has ``2 * context_dim`` outputs, and ``GaussianContextOutput`` turns them into the mean ``mu_t``
    (through ``context_activation``) and the variance ``sigma2_t > 0``. The encoder then outputs
    ``z_t = [mu_t, sigma2_t]``, ``output_dim = 2 * context_dim`` wide (``split_context`` separates them). The state
    dict is that of the base class built with ``context_dim=output_dim``.

    ``auxiliary_dim`` adds a second head on the same xLSTM embedding, ``auxiliary_projection`` (hidden layers
    ``auxiliary_net_arch``, linear output), which predicts supervised auxiliary targets (BatteryPlane: the true
    speed ``vx_t / v_ref``). It is not part of ``z_t``: ``step`` and ``unroll`` never run it, the actor and critic
    never read it, and only ``unroll_with_auxiliary`` (the encoder's supervised training) computes it, so that its
    loss shapes the shared embedding. Its parameters are added to the base class's state dict.

    :param observation_dim: Dimension of the flattened observation
    :param action_dim: Dimension of the action
    :param context_dim: Dimension of the context ``z_t`` (with the Gaussian head: of its mean and of its variance)
    :param xlstm_config: Config of the block stack, see ``create_xlstm``
    :param projection_net_arch: Hidden layers of the context head, linear by default
    :param activation_fn: Activation function of the hidden layers of the context head
    :param context_activation: Output activation of the context head (a key of ``CONTEXT_ACTIVATIONS``),
        None for a linear output. With the Gaussian head, the activation of the mean only.
    :param context_head: ``"point"`` (the context) or ``"gaussian"`` (its mean and variance), see ``CONTEXT_HEADS``
    :param auxiliary_dim: Size of the auxiliary head's output, None for no auxiliary head
    :param auxiliary_net_arch: Hidden layers of the auxiliary head, linear by default
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        context_dim: int = 8,
        xlstm_config: xLSTMBlockStackConfig | Mapping[str, Any] | None = None,
        projection_net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        context_activation: str | None = None,
        context_head: str = "point",
        auxiliary_dim: int | None = None,
        auxiliary_net_arch: list[int] | None = None,
    ) -> None:
        if context_activation is not None and context_activation not in CONTEXT_ACTIVATIONS:
            raise ValueError(f"context_activation must be None or one of {sorted(CONTEXT_ACTIVATIONS)}, got {context_activation!r}")
        if context_head not in CONTEXT_HEADS:
            raise ValueError(f"context_head must be one of {CONTEXT_HEADS}, got {context_head!r}")
        if auxiliary_dim is not None and (isinstance(auxiliary_dim, bool) or not isinstance(auxiliary_dim, int) or auxiliary_dim < 1):
            raise ValueError(f"auxiliary_dim must be None or an integer >= 1, got {auxiliary_dim!r}")
        output_dim = 2 * context_dim if context_head == "gaussian" else context_dim
        super().__init__(observation_dim, action_dim, output_dim, xlstm_config, projection_net_arch, activation_fn)
        # The base class names the width of its head context_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.context_activation = context_activation
        self.context_head = context_head
        if context_head == "gaussian":
            self.context_projection.append(GaussianContextOutput(context_dim, context_activation))
        elif context_activation is not None:
            self.context_projection.append(CONTEXT_ACTIVATIONS[context_activation]())
        self.auxiliary_dim = auxiliary_dim
        self.auxiliary_net_arch = auxiliary_net_arch
        self.auxiliary_projection = (
            None
            if auxiliary_dim is None
            else nn.Sequential(*create_mlp(self.embedding_dim, auxiliary_dim, auxiliary_net_arch or [], activation_fn))
        )

    def embed_step(
        self, state: dict[str, Any], observations: th.Tensor, prev_actions: th.Tensor
    ) -> tuple[th.Tensor, dict[str, Any]]:
        """
        ``XLSTMContextEncoder.step`` without the context head: the xLSTM embedding of one input ``[obs_t, action_{t-1}]``
        per row, which both heads read. ``context_projection(embed_step(...)[0])`` is ``step(...)[0]``.

        :param state: ``state_before[t]``
        :param observations: ``obs_t`` [B, *obs_shape]
        :param prev_actions: ``action_{t-1}`` [B, action_dim], zero at episode start
        :return: The embedding [B, embedding_dim] and ``state_after[t]``
        """
        x = th.cat([observations.flatten(start_dim=1), prev_actions.flatten(start_dim=1)], dim=-1)
        if self.uses_cuda_backend and x.device.type != "cuda":
            raise RuntimeError(f"The sLSTM 'cuda' backend only runs on CUDA devices, not {x.device}: use backend: vanilla")
        x = self.input_projection(x).unsqueeze(1)
        # xlstm writes into the state dict and overwrites conv_state in place, so it gets a copy
        output, new_state = self.xlstm.step(x, self._clone_state(state))
        return output.squeeze(1), new_state

    def unroll_with_auxiliary(
        self,
        state: dict[str, Any],
        observations: th.Tensor,
        prev_actions: th.Tensor,
        mask: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, dict[str, Any]]:
        """
        ``unroll`` without ``episode_starts`` (e.g. complete episodes of the episode buffer), which also runs the
        auxiliary head on every embedding. Where ``mask`` is False, a row keeps its state and both outputs are zero.

        :param state: State before the first input
        :param observations: [B, L, *obs_shape]
        :param prev_actions: [B, L, action_dim]
        :param mask: [B, L] bool, True for valid inputs. All inputs are valid by default.
        :return: The contexts [B, L, output_dim], the auxiliary predictions [B, L, auxiliary_dim] and the state
            after each row's last valid input
        """
        if self.auxiliary_projection is None:
            raise ValueError("unroll_with_auxiliary needs an auxiliary head: set auxiliary_dim")
        batch_size, seq_len = observations.shape[:2]
        if seq_len == 0:
            return (
                observations.new_zeros(batch_size, 0, self.output_dim),
                observations.new_zeros(batch_size, 0, self.auxiliary_dim),
                state,
            )
        contexts, auxiliaries = [], []
        for i in range(seq_len):
            embedding, new_state = self.embed_step(state, observations[:, i], prev_actions[:, i])
            context, auxiliary = self.context_projection(embedding), self.auxiliary_projection(embedding)
            if mask is None:
                state = new_state
            else:
                valid = mask[:, i].bool()
                state = self._select_state(valid, new_state, state)
                context = th.where(valid.unsqueeze(-1), context, th.zeros_like(context))
                auxiliary = th.where(valid.unsqueeze(-1), auxiliary, th.zeros_like(auxiliary))
            contexts.append(context)
            auxiliaries.append(auxiliary)
        return th.stack(contexts, dim=1), th.stack(auxiliaries, dim=1), state

    def split_context(self, contexts: th.Tensor) -> tuple[th.Tensor, th.Tensor | None]:
        """
        Separate the encoder outputs into the context estimate and its variance.

        :param contexts: [..., output_dim] outputs of ``step`` or ``unroll``
        :return: The context (the mean with the Gaussian head) [..., context_dim], and the variance
            [..., context_dim] with the Gaussian head, None with the point head
        """
        if self.context_head == "gaussian":
            mean, variance = contexts.split(self.context_dim, dim=-1)
            return mean, variance
        return contexts, None

    def unroll(
        self,
        state: dict[str, Any],
        observations: th.Tensor,
        prev_actions: th.Tensor,
        mask: th.Tensor | None = None,
        episode_starts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, dict[str, Any]]:
        """
        Process a sequence step by step. Without ``episode_starts``, exactly ``XLSTMContextEncoder.unroll``
        (where ``mask`` marks padding). With ``episode_starts``, the rows where ``episode_starts[:, i]`` is
        set get the initial state before input ``i``: the memory and the gradient of the earlier inputs stop
        there, as at an episode start during collection. Streams have no padding, so ``mask`` must then be None.

        :param state: State before the first input (e.g. ``ContextualRolloutSamples.initial_state``)
        :param observations: [B, L, *obs_shape]
        :param prev_actions: [B, L, action_dim], zero where an episode starts
        :param mask: [B, L] bool padding mask, only without ``episode_starts``
        :param episode_starts: [B, L] bool, reset the row before input ``i``
        :return: The contexts [B, L, output_dim] and the state after the last input
        """
        if episode_starts is None:
            if observations.shape[1] == 0:
                # The base class would size the empty contexts by its context_dim, not by output_dim
                return observations.new_zeros(observations.shape[0], 0, self.output_dim), state
            return super().unroll(state, observations, prev_actions, mask)
        if mask is not None:
            raise ValueError("episode_starts and a padding mask can't be combined: rollout streams have no padding")

        batch_size, seq_len = observations.shape[:2]
        if tuple(episode_starts.shape) != (batch_size, seq_len):
            raise ValueError(f"episode_starts must have shape {(batch_size, seq_len)}, got {tuple(episode_starts.shape)}")
        if seq_len == 0:
            return observations.new_zeros(batch_size, 0, self.output_dim), state

        episode_starts = episode_starts.bool()
        contexts = []
        for i in range(seq_len):
            if episode_starts[:, i].any():
                state = self.reset_state(state, episode_starts[:, i])
            context, state = self.step(state, observations[:, i], prev_actions[:, i])
            contexts.append(context)
        return th.stack(contexts, dim=1), state

    def forward(
        self,
        state: dict[str, Any],
        observations: th.Tensor,
        prev_actions: th.Tensor,
        mask: th.Tensor | None = None,
        episode_starts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, dict[str, Any]]:
        return self.unroll(state, observations, prev_actions, mask, episode_starts)
