"""Step 0 of ENCODER_PLAN.md: Contextual_PPO with the true battery, or with no context at all, in place of the
learned context.

Oracle, cppo.context_source=oracle (config/step0/cppo_oracle.yaml, run 0c): the environment runs in observed mode,
whose observation is the four hidden-mode values followed by the true charge b_t (BatteryPlaneVec._obs). The xLSTM
encoder is replaced by TrueBatteryEncoder, which has no parameters and no memory, and the actor and critic read the
observation alone: the five inputs of plain PPO in observed mode.

Memoryless, cppo.context_source=none (config/step0/cppo_blind.yaml, run 0c_blind): hidden mode, and NoContextEncoder,
which outputs nothing about the battery or the history. The actor and critic read the four hidden-mode values alone,
the inputs of plain PPO in hidden mode: experiment A without its encoder, and 0c without the battery.

Both keep the rest of CPPO unchanged: the collector, the whole-stream minibatches and the PPO losses.

Project glue, like the training script: the learner (contextual_ppo/) stays environment-agnostic.
"""

import numpy as np
import torch as th
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from torch import nn

from contextual_ppo.policies import ContextualActorCriticPolicy

# Column of the true charge b_t in observed- and full-mode observations
BATTERY_COLUMN = -1


class TrueBatteryEncoder(nn.Module):
    """
    Stand-in for ``XLSTMRolloutEncoder`` with the interface the policy and the learner use: ``z_t = b_t``, read
    from the observation. It has no parameters, so no update can change it, and no state (None, which the rollout
    buffer accepts for an encoder without state).
    """

    context_head = "point"
    auxiliary_dim = None

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.context_dim = self.output_dim = 1

    def initial_state(self, batch_size: int) -> None:
        return None

    def reset_state(self, state: None, reset_mask: th.Tensor | np.ndarray) -> None:
        return None

    def step(self, state: None, observations: th.Tensor, prev_actions: th.Tensor) -> tuple[th.Tensor, None]:
        return observations[..., BATTERY_COLUMN].unsqueeze(-1), None

    def unroll(
        self,
        state: None,
        observations: th.Tensor,
        prev_actions: th.Tensor,
        mask: th.Tensor | None = None,
        episode_starts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, None]:
        contexts = observations[..., BATTERY_COLUMN].unsqueeze(-1)
        if mask is not None:
            # Padding gets a zero context, as with the xLSTM encoder
            contexts = contexts * mask.unsqueeze(-1).to(contexts.dtype)
        return contexts, None

    def split_context(self, contexts: th.Tensor) -> tuple[th.Tensor, None]:
        return contexts, None


class NoContextEncoder(TrueBatteryEncoder):
    """
    Stand-in for ``XLSTMRolloutEncoder`` that knows nothing: ``z_t = 0`` at every step, whatever the observation and
    the previous action. No parameters and no state, like ``TrueBatteryEncoder``.
    """

    def step(self, state: None, observations: th.Tensor, prev_actions: th.Tensor) -> tuple[th.Tensor, None]:
        return observations.new_zeros(*observations.shape[:-1], 1), None

    def unroll(
        self,
        state: None,
        observations: th.Tensor,
        prev_actions: th.Tensor,
        mask: th.Tensor | None = None,
        episode_starts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, None]:
        return observations.new_zeros(*observations.shape[:-1], 1), None


class ObservationOnlyPolicy(ContextualActorCriticPolicy):
    """
    ``ContextualActorCriticPolicy`` whose actor and critic read the observation only: ``z_t`` is computed and returned
    (rollout diagnostics) but never appended, so the branches have plain PPO's inputs and sizes. Subclasses choose the
    stand-in encoder. The xLSTM and context-head arguments are accepted and ignored.
    """

    def _build_mlp_extractor(self) -> None:
        # Nothing of z_t reaches the branches
        self.context_input_dim = 0
        super()._build_mlp_extractor()

    def context_inputs(self, latents: th.Tensor) -> th.Tensor:
        return latents[..., :0]


class TrueBatteryPolicy(ObservationOnlyPolicy):
    """
    Oracle: ``TrueBatteryEncoder``, ``z_t = b_t``. The observed-mode observation already ends with ``b_t``, so the
    branches get ``[y, vx, vy, tau, b]``, the input of plain PPO in observed mode.
    """

    def make_encoder(self) -> TrueBatteryEncoder:  # type: ignore[override]
        return TrueBatteryEncoder(get_flattened_obs_dim(self.observation_space), get_action_dim(self.action_space)).to(self.device)


class NoContextPolicy(ObservationOnlyPolicy):
    """
    Memoryless: ``NoContextEncoder``, ``z_t = 0``. In hidden mode the branches get ``[y, vx, vy, tau]``, the input
    of plain PPO in hidden mode, and the previous action has no effect.
    """

    def make_encoder(self) -> NoContextEncoder:  # type: ignore[override]
        return NoContextEncoder(get_flattened_obs_dim(self.observation_space), get_action_dim(self.action_space)).to(self.device)
