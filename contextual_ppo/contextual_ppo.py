import math
import time
from collections.abc import Sequence
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance

from .buffers import ContextualRolloutBuffer, _flatten_recurrent_state, _unflatten_recurrent_state
from .episode_buffers import SupervisedContextualEpisodeBuffer
from .on_policy_algorithm import ContextualOnPolicyAlgorithm
from .policies import ContextualActorCriticPolicy, MlpPolicy
from .type_aliases import ContextualEpisodeSamples, ContextualRolloutSamples

SelfContextualPPO = TypeVar("SelfContextualPPO", bound="ContextualPPO")


class ContextualPPO(ContextualOnPolicyAlgorithm):
    """
    Proximal Policy Optimization (clip version) with a recurrent context encoder: Contextual_PPO.

    The encoder reads ``[o_t, a_{t-1}]`` and gives the context ``z_t`` to the actor ``pi(a_t | o_t, z_t)`` and
    the critic ``V(o_t, z_t)`` (``ContextualActorCriticPolicy``). Collection carries the encoder memory across
    steps and rollouts (``ContextualOnPolicyAlgorithm``). The PPO objective, GAE, the clipping schedules,
    advantage normalization and KL early stopping are those of SB3's ``PPO``.

    Each epoch visits every environment stream once, in ``n_minibatches`` minibatches of complete streams
    (``ContextualRolloutBuffer.get``). A minibatch is re-encoded in time order with the current parameters,
    from the encoder state stored at the start of the rollout, resetting the memory at episode starts.
    ``bptt_len`` bounds the backpropagation: the stream is processed in consecutive segments of that length;
    the memory carries its values across segments but its gradient is cut, and every segment is backpropagated
    before the next one is computed, so only one segment's graph is in memory at a time. The losses are means
    over the whole minibatch, so each segment counts by its number of transitions. Advantages are normalized
    over the whole minibatch, and the KL early stopping uses the whole minibatch: when it stops, the gradients
    of that minibatch are discarded.

    Gradients are routed per parameter group (``ContextualActorCriticPolicy.parameter_groups()``), as in
    recurrent SAC, all computed from the same parameters before the optimizer step:

    - actor branch: ``L_actor = L_policy + ent_coef * L_entropy``
    - value branch: ``L_critic = vf_coef * L_value``
    - encoder: ``L_critic + encoder_actor_loss_weight * L_actor``

    With ``encoder_actor_loss_weight=1`` every gradient equals that of PPO's summed loss. Without reward
    normalization the value loss tends to dominate the encoder gradient; a larger weight counteracts it
    (compare the logged ``train/encoder_grad_norm_value_loss`` and ``train/encoder_grad_norm_actor_loss``).
    ``max_grad_norm`` clips the actor and value branches together, as vanilla PPO clips its networks, and the
    encoder is clipped on its own with ``encoder_max_grad_norm``. One Adam optimizer (``policy.optimizer``)
    holds every parameter.

    ``encoder_objective="supervised"`` trains the encoder as an estimator of the true context instead
    (BatteryPlane: the charge ``b_t``, labels collected with ``context_info_keys=("b0", "b")``). The PPO
    epochs then update the actor and critic only: they read ``z_t`` computed without gradient. After them,
    ``supervised_gradient_steps`` encoder steps train on complete episodes of a
    ``SupervisedContextualEpisodeBuffer`` (filled during collection, kept across updates): each step samples
    ``episode_batch_size`` training episodes, runs the encoder from its initial state at every episode's first
    step over ``[o_t, a_{t-1}]`` (the labels are never an input), and minimizes the mean squared error of
    ``z_t`` against ``b_t`` over the valid (non-padding) steps. One batch of validation episodes, which never
    train, measures the fit. ``context_dim`` must equal the label size (1 for BatteryPlane; a sigmoid context
    head matches ``b`` in [0, 1]). The labels are used for training only: acting and evaluation need no
    privileged input.

    With the policy's ``context_head="gaussian"`` (supervised only), the encoder outputs the mean ``mu_t`` and
    variance ``sigma2_t`` of the context, the actor and critic read both, and the encoder steps minimize the
    Gaussian negative log-likelihood ``0.5 * (log sigma2_t + (b_t - mu_t)^2 / sigma2_t)`` (without the constant
    ``0.5 * log(2 pi)``) instead of the squared error, over the same valid steps. The variance thus learns how
    far the mean is from ``b_t`` where the history leaves the context uncertain. The likelihood divides each
    step's error by ``sigma2_t``, so the mean learns little where the variance is large. ``nll_beta`` > 0
    (beta-NLL, Seitzer et al., 2022, https://arxiv.org/abs/2203.09168) multiplies each step's negative
    log-likelihood by ``sigma2_t ** nll_beta``, computed without gradient: 0 is the plain likelihood, 1 makes the
    mean's gradient proportional to that of the squared error, 0.5 is the paper's recommendation. Besides the
    squared error and R^2 of the mean, the logs then hold the likelihood (``context_nll``, without the beta
    weights), the mean predicted variance (``context_variance``) and ``context_calibration``, the mean of
    ``(b_t - mu_t)^2 / sigma2_t``: 1 when the variance is calibrated, above 1 when it is too small (overconfident).

    With the policy's ``auxiliary_dim`` and ``auxiliary_info_keys`` (supervised only), the encoder also has an
    auxiliary head on its embedding, trained in the same encoder steps on the same episodes with the targets read
    from the env infos (BatteryPlane: the true speed ``vx_t / v_ref``, which the observation holds with noise).
    The encoder steps minimize ``L_context + auxiliary_loss_weight * L_auxiliary``, ``L_auxiliary`` the masked mean
    squared error of the auxiliary head. The actor and critic never read the auxiliary head. Logged:
    ``auxiliary_mse`` and ``auxiliary_r2`` (and ``auxiliary_val_*``), ``context_loss`` (``L_context``) and
    ``encoder_loss`` (the weighted sum).

    Paper: https://arxiv.org/abs/1707.06347

    :param policy: The policy model to use (``MlpPolicy``, i.e. ``ContextualActorCriticPolicy``)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
    :param n_minibatches: Number of minibatches per epoch. Each holds ``n_envs / n_minibatches`` complete streams
        of ``n_steps`` transitions, so it must divide ``n_envs`` (this replaces PPO's ``batch_size``).
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping of the actor and value branches
    :param encoder_actor_loss_weight: Weight of the actor loss in the encoder objective
        ``L_critic + encoder_actor_loss_weight * L_actor``; 1 gives PPO's summed loss
    :param encoder_max_grad_norm: Clip the encoder gradient to this norm, no clipping if None
    :param encoder_objective: What trains the encoder: ``"rl"`` (the routed PPO losses, see above) or
        ``"supervised"`` (only the masked mean squared error of ``z_t`` against the true context, on complete
        episodes; the actor and critic read ``z_t`` without gradient). ``"supervised"`` needs ``context_info_keys``.
        With the policy's ``context_head="gaussian"``, the supervised loss is the Gaussian negative log-likelihood
        of the true context, and the RL objective is not allowed.
    :param nll_beta: Gaussian head only: beta of the beta-NLL, in [0, 1]; 0 trains with the plain negative
        log-likelihood
    :param auxiliary_loss_weight: Weight ``lambda`` in [0, 1] of the auxiliary head's loss in the encoder steps,
        ``L_context + lambda * L_auxiliary``; used with an auxiliary head only
    :param episode_buffer_capacity: Supervised only: maximum number of stored timesteps of completed episodes;
        the oldest episodes are evicted whole
    :param episode_batch_size: Supervised only: number of complete episodes per encoder step and validation batch
    :param supervised_gradient_steps: Supervised only: encoder steps after the PPO epochs of every update
    :param validation_fraction: Supervised only: probability that a completed episode is kept for validation
        instead of training
    :param episode_buffer_seed: Supervised only: seed of the episode buffer's split and sampling; None uses ``seed``
    :param bptt_len: Maximum number of steps connected by backpropagation through the encoder memory;
        None backpropagates through the whole stream (episode starts still cut it)
    :param use_sde: Must be False: gSDE is not supported
    :param sde_sample_freq: Unused, kept for the ``OnPolicyAlgorithm`` interface
    :param rollout_buffer_class: Rollout buffer class to use, ``ContextualRolloutBuffer`` or a subclass.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param reset_envs_each_rollout: Start every rollout with new episodes in all envs, see
        ``ContextualOnPolicyAlgorithm`` (with ``n_steps`` at least the episode length, each such episode lies
        complete in its stream)
    :param context_info_keys: ``(reset_key, step_key)`` of the true context in the env ``info`` dicts, stored
        in the buffer for diagnostics only (BatteryPlane: ``("b0", "b")``); None stores nothing
    :param auxiliary_info_keys: ``(reset_key, step_key)`` of the auxiliary head's targets in the env ``info``
        dicts (needs the policy's ``auxiliary_dim`` and ``encoder_objective="supervised"``), see
        ``ContextualOnPolicyAlgorithm``; None for no auxiliary head
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation, e.g.
        ``context_dim``, ``xlstm_config``, ``context_net_arch`` and ``net_arch``, see ``ContextualActorCriticPolicy``
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
    }
    policy: ContextualActorCriticPolicy
    rollout_buffer: ContextualRolloutBuffer

    ENCODER_OBJECTIVES = ("rl", "supervised")

    def __init__(
        self,
        policy: str | type[ContextualActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 512,
        n_minibatches: int = 1,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        encoder_actor_loss_weight: float = 1.0,
        encoder_max_grad_norm: float | None = None,
        encoder_objective: str = "rl",
        nll_beta: float = 0.0,
        auxiliary_loss_weight: float = 1.0,
        episode_buffer_capacity: int = 200_000,
        episode_batch_size: int = 16,
        supervised_gradient_steps: int = 8,
        validation_fraction: float = 0.1,
        episode_buffer_seed: int | None = None,
        bptt_len: int | None = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[ContextualRolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        reset_envs_each_rollout: bool = False,
        context_info_keys: Sequence[str] | None = None,
        auxiliary_info_keys: Sequence[str] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            reset_envs_each_rollout=reset_envs_each_rollout,
            context_info_keys=context_info_keys,
            auxiliary_info_keys=auxiliary_info_keys,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
        )

        for name, value in (
            ("n_minibatches", n_minibatches),
            ("n_epochs", n_epochs),
            ("episode_buffer_capacity", episode_buffer_capacity),
            ("episode_batch_size", episode_batch_size),
            ("supervised_gradient_steps", supervised_gradient_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if bptt_len is not None and (isinstance(bptt_len, bool) or not isinstance(bptt_len, int) or bptt_len < 1):
            raise ValueError(f"bptt_len must be None or an integer >= 1, got {bptt_len!r}")
        if not (math.isfinite(encoder_actor_loss_weight) and encoder_actor_loss_weight >= 0):
            raise ValueError(f"encoder_actor_loss_weight must be finite and >= 0, got {encoder_actor_loss_weight!r}")
        if not (isinstance(validation_fraction, (int, float)) and 0.0 <= validation_fraction < 1.0):
            raise ValueError(f"validation_fraction must be in [0, 1), got {validation_fraction!r}")
        if episode_buffer_seed is not None and (isinstance(episode_buffer_seed, bool) or not isinstance(episode_buffer_seed, int)):
            raise ValueError(f"episode_buffer_seed must be None or an integer, got {episode_buffer_seed!r}")
        if isinstance(nll_beta, bool) or not isinstance(nll_beta, (int, float)) or not 0.0 <= nll_beta <= 1.0:
            raise ValueError(f"nll_beta must be in [0, 1], got {nll_beta!r}")
        if not (isinstance(auxiliary_loss_weight, (int, float)) and not isinstance(auxiliary_loss_weight, bool)
                and 0.0 <= auxiliary_loss_weight <= 1.0):
            raise ValueError(f"auxiliary_loss_weight must be in [0, 1], got {auxiliary_loss_weight!r}")
        if encoder_objective not in self.ENCODER_OBJECTIVES:
            raise ValueError(f"encoder_objective must be one of {self.ENCODER_OBJECTIVES}, got {encoder_objective!r}")
        for name, value in (("encoder_max_grad_norm", encoder_max_grad_norm), ("target_kl", target_kl)):
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be None or positive, got {value!r}")

        if self.env is not None:
            # Each minibatch holds complete streams
            if n_minibatches > self.n_envs or self.n_envs % n_minibatches != 0:
                raise ValueError(
                    f"n_minibatches={n_minibatches} must divide n_envs={self.n_envs}: each minibatch holds complete streams"
                )
            # Advantage normalization needs more than one transition per minibatch (GH issue #325)
            minibatch_transitions = self.n_steps * self.n_envs // n_minibatches
            if normalize_advantage and minibatch_transitions < 2:
                raise ValueError(
                    "normalize_advantage needs at least 2 transitions per minibatch, "
                    f"got n_steps * n_envs / n_minibatches = {minibatch_transitions}"
                )

        self.n_minibatches = n_minibatches
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.encoder_actor_loss_weight = encoder_actor_loss_weight
        self.encoder_max_grad_norm = encoder_max_grad_norm
        self.encoder_objective = encoder_objective
        self.nll_beta = float(nll_beta)
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.episode_buffer_capacity = episode_buffer_capacity
        self.episode_batch_size = episode_batch_size
        self.supervised_gradient_steps = supervised_gradient_steps
        self.validation_fraction = validation_fraction
        self.episode_buffer_seed = episode_buffer_seed
        self.bptt_len = bptt_len
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if self.policy.context_head == "gaussian" and self.encoder_objective != "supervised":
            raise ValueError(
                "context_head='gaussian' learns the variance from the likelihood of the true context: "
                "it needs encoder_objective='supervised'"
            )
        auxiliary_dim = self.policy.auxiliary_dim
        if (auxiliary_dim is None) != (self.auxiliary_info_keys is None):
            raise ValueError(
                "An auxiliary head needs both the policy's auxiliary_dim and auxiliary_info_keys (its targets), "
                f"got auxiliary_dim={auxiliary_dim!r} and auxiliary_info_keys={self.auxiliary_info_keys!r}"
            )
        if auxiliary_dim is not None and self.encoder_objective != "supervised":
            raise ValueError("The auxiliary head is trained in the supervised encoder steps: it needs encoder_objective='supervised'")
        if self.encoder_objective == "supervised":
            label_dim = self.rollout_buffer.context_dim
            if label_dim is None:
                raise ValueError("encoder_objective='supervised' needs true-context labels: set context_info_keys, e.g. ('b0', 'b')")
            if label_dim != self.policy.context_dim:
                raise ValueError(
                    f"encoder_objective='supervised' fits z_t to the label: policy context_dim ({self.policy.context_dim}) "
                    f"must equal the label size ({label_dim})"
                )
            # Complete episodes for the encoder, filled by the collector; not saved with the model.
            # Each label row is the true context, followed by the auxiliary targets
            self.episode_buffer = SupervisedContextualEpisodeBuffer(
                self.n_envs,
                self.observation_space,
                self.action_space,
                label_dim=label_dim + (auxiliary_dim or 0),
                capacity=self.episode_buffer_capacity,
                episode_batch_size=self.episode_batch_size,
                validation_fraction=self.validation_fraction,
                device=self.device,
                seed=self.seed if self.episode_buffer_seed is None else self.episode_buffer_seed,
            )
        else:
            self.episode_buffer = None

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        start_time = time.perf_counter()
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        groups = self.policy.parameter_groups()
        entropy_losses, pg_losses, value_losses, clip_fractions, losses = [], [], [], [], []
        encoder_grad_norms, encoder_value_grad_norms, encoder_actor_grad_norms, minibatch_transitions = [], [], [], []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer: every stream once
            for rollout_data in self.rollout_buffer.get(self.n_minibatches):
                result = self._minibatch_gradients(rollout_data, groups, clip_range, clip_range_vf)

                # Logging
                pg_losses.append(result["policy_loss"])
                value_losses.append(result["value_loss"])
                entropy_losses.append(result["entropy_loss"])
                clip_fractions.append(result["clip_fraction"])
                losses.append(result["loss"])
                approx_kl_divs.append(result["approx_kl"])
                minibatch_transitions.append(result["n_transitions"])

                if self.target_kl is not None and result["approx_kl"] > 1.5 * self.target_kl:
                    # The gradients of this minibatch are discarded
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {result['approx_kl']:.2f}")
                    break

                # Encoder: L_critic + encoder_actor_loss_weight * L_actor
                # (supervised: None, the heads read z_t computed without gradient)
                encoder_grads = [
                    self._add_grads(value_grad, actor_grad, self.encoder_actor_loss_weight)
                    for value_grad, actor_grad in zip(result["encoder_value_grads"], result["encoder_actor_grads"], strict=True)
                ]
                encoder_value_grad_norms.append(self._grad_norm(result["encoder_value_grads"]))
                encoder_actor_grad_norms.append(self._grad_norm(result["encoder_actor_grads"]))
                encoder_grad_norms.append(self._grad_norm(encoder_grads))

                # Optimization step
                self.policy.optimizer.zero_grad()
                self._set_grads(groups["actor"], result["actor_grads"])
                self._set_grads(groups["value"], result["value_grads"])
                self._set_grads(groups["encoder"], encoder_grads)
                # Clip grad norm: the heads together, as vanilla PPO, the encoder on its own
                th.nn.utils.clip_grad_norm_(groups["actor"] + groups["value"], self.max_grad_norm)
                if self.encoder_max_grad_norm is not None:
                    th.nn.utils.clip_grad_norm_(groups["encoder"], self.encoder_max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # After the PPO epochs, which replayed the rollout with the encoder of collection (ratios start at 1)
        supervised_stats = self._train_encoder_supervised() if self.encoder_objective == "supervised" else {}

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", losses[-1])
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        if len(encoder_grad_norms) > 0:
            self.logger.record("train/encoder_grad_norm", np.mean(encoder_grad_norms))
            self.logger.record("train/encoder_grad_norm_value_loss", np.mean(encoder_value_grad_norms))
            self.logger.record("train/encoder_grad_norm_actor_loss", np.mean(encoder_actor_grad_norms))
        for key, value in supervised_stats.items():
            self.logger.record(f"train/{key}", value)
        self.logger.record("train/minibatch_transitions", np.mean(minibatch_transitions))
        self.logger.record("train/bptt_len", self.n_steps if self.bptt_len is None else min(self.bptt_len, self.n_steps))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("time/train_seconds", time.perf_counter() - start_time)

    def _minibatch_gradients(
        self,
        rollout_data: ContextualRolloutSamples,
        groups: dict[str, list[th.nn.Parameter]],
        clip_range: float,
        clip_range_vf: float | None,
    ) -> dict[str, Any]:
        """
        Losses, statistics and routed gradients of one minibatch of complete streams, accumulated over its
        backpropagation segments. Nothing is written to ``.grad`` and no parameter changes.

        :return: The minibatch statistics, the gradients of ``L_actor`` for the actor branch and of ``L_critic``
            for the value branch, and the two encoder parts (from ``L_critic`` and from the unweighted ``L_actor``;
            None with ``encoder_objective="supervised"``, where the heads read ``z_t`` computed without gradient)
        """
        supervised = self.encoder_objective == "supervised"
        n_envs, n_steps = rollout_data.observations.shape[:2]
        n_transitions = n_envs * n_steps
        segment_len = n_steps if self.bptt_len is None else min(self.bptt_len, n_steps)

        # Normalize advantage over the whole minibatch.
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        advantages = rollout_data.advantages
        if self.normalize_advantage and n_transitions > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_params, value_params, encoder_params = groups["actor"], groups["value"], groups["encoder"]
        actor_grads: list[th.Tensor | None] = [None] * len(actor_params)
        value_grads: list[th.Tensor | None] = [None] * len(value_params)
        encoder_actor_grads: list[th.Tensor | None] = [None] * len(encoder_params)
        encoder_value_grads: list[th.Tensor | None] = [None] * len(encoder_params)
        sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0, "clipped": 0.0, "kl": 0.0}

        encoder_state = rollout_data.initial_state
        for start in range(0, n_steps, segment_len):
            segment = slice(start, start + segment_len)
            values, log_prob, entropy, final_state, _ = self.policy.evaluate_actions(
                rollout_data.observations[:, segment],
                rollout_data.actions[:, segment],
                encoder_state,
                rollout_data.prev_actions[:, segment],
                rollout_data.episode_starts[:, segment],
                # Supervised: the encoder runs without gradient, the PPO losses must not reach it
                detach_context=supervised,
            )
            # The losses are means over the minibatch: each segment counts by its share of the transitions
            weight = values.numel() / n_transitions

            # ratio between old and new policy, should be one at the first iteration
            log_ratio = log_prob - rollout_data.old_log_prob[:, segment]
            ratio = th.exp(log_ratio)

            # clipped surrogate loss
            segment_advantages = advantages[:, segment]
            policy_loss_1 = segment_advantages * ratio
            policy_loss_2 = segment_advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

            if clip_range_vf is None:
                # No clipping
                values_pred = values
            else:
                # Clip the difference between old and new value
                # NOTE: this depends on the reward scaling
                old_values = rollout_data.old_values[:, segment]
                values_pred = old_values + th.clamp(values - old_values, -clip_range_vf, clip_range_vf)
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns[:, segment], values_pred)

            # Entropy loss favor exploration
            entropy_loss = -th.mean(entropy)

            # Gradients of each loss for the parameters it trains, all before any update:
            # the actor loss doesn't reach the value branch, the encoder gets both losses
            actor_loss = weight * (policy_loss + self.ent_coef * entropy_loss)
            critic_loss = weight * self.vf_coef * value_loss
            critic_grads = th.autograd.grad(critic_loss, value_params + encoder_params, retain_graph=True, allow_unused=True)
            actor_loss_grads = th.autograd.grad(actor_loss, actor_params + encoder_params, allow_unused=True)
            self._accumulate(value_grads, critic_grads[: len(value_params)])
            self._accumulate(encoder_value_grads, critic_grads[len(value_params) :])
            self._accumulate(actor_grads, actor_loss_grads[: len(actor_params)])
            self._accumulate(encoder_actor_grads, actor_loss_grads[len(actor_params) :])

            with th.no_grad():
                sums["policy_loss"] += weight * policy_loss.item()
                sums["value_loss"] += weight * value_loss.item()
                sums["entropy_loss"] += weight * entropy_loss.item()
                sums["clipped"] += (th.abs(ratio - 1) > clip_range).float().sum().item()
                # Approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                sums["kl"] += ((th.exp(log_ratio) - 1) - log_ratio).sum().item()

            # The memory keeps its values into the next segment, without its gradient
            encoder_state = self._detach_state(final_state)

        return {
            "actor_grads": actor_grads,
            "value_grads": value_grads,
            "encoder_actor_grads": encoder_actor_grads,
            "encoder_value_grads": encoder_value_grads,
            "policy_loss": sums["policy_loss"],
            "value_loss": sums["value_loss"],
            "entropy_loss": sums["entropy_loss"],
            "loss": sums["policy_loss"] + self.ent_coef * sums["entropy_loss"] + self.vf_coef * sums["value_loss"],
            "clip_fraction": sums["clipped"] / n_transitions,
            "approx_kl": sums["kl"] / n_transitions,
            "n_transitions": n_transitions,
        }

    def _train_encoder_supervised(self) -> dict[str, float]:
        """
        ``supervised_gradient_steps`` encoder steps on complete training episodes of the episode buffer, then the
        fit on one batch of validation episodes. Every episode starts from the encoder's initial state and its
        states are recomputed with the current weights. Only the encoder changes.

        :return: Statistics to log: the fit on the training batches (before each step) and on the validation
            batch (``context_val_*``, ``auxiliary_val_*``), the encoder gradient norm, and the buffer's episode counts
        """
        buffer = self.episode_buffer
        assert buffer is not None
        encoder_params = self.policy.parameter_groups()["encoder"]
        stats: dict[str, float] = {
            "episode_buffer_steps": buffer.stored_steps,
            "train_episodes": buffer.n_episodes("train"),
            "validation_episodes": buffer.n_episodes("validation"),
        }
        if buffer.n_episodes("train") > 0:
            fits, grad_norms = [], []
            for _ in range(self.supervised_gradient_steps):
                loss, fit = self._episode_fit(buffer.sample("train"))
                grads = th.autograd.grad(loss, encoder_params, allow_unused=True)
                grad_norms.append(self._grad_norm(grads))
                fits.append(fit)
                self.policy.optimizer.zero_grad()
                self._set_grads(encoder_params, grads)
                if self.encoder_max_grad_norm is not None:
                    th.nn.utils.clip_grad_norm_(encoder_params, self.encoder_max_grad_norm)
                self.policy.optimizer.step()
            stats.update({key: float(np.mean([fit[key] for fit in fits])) for key in fits[0]})
            stats.update(encoder_grad_norm_context_loss=float(np.mean(grad_norms)))
        if buffer.n_episodes("validation") > 0:
            with th.no_grad():
                _, fit = self._episode_fit(buffer.sample("validation"))
            # context_mse -> context_val_mse, auxiliary_r2 -> auxiliary_val_r2
            stats.update({key.replace("_", "_val_", 1): value for key, value in fit.items() if not key.endswith("_loss")})
        return stats

    def _episode_fit(self, samples: ContextualEpisodeSamples) -> tuple[th.Tensor, dict[str, float]]:
        """
        Run the encoder over complete episodes from its initial state (the labels are not an input) and
        compare its outputs with the labels on the valid steps: ``z_t`` with the true context, and the auxiliary
        head (if any) with the auxiliary targets, which follow the context in every label row.

        :return: The loss (differentiable), ``L_context + auxiliary_loss_weight * L_auxiliary``: ``L_context`` the
            masked mean squared error of ``z_t``, or with the Gaussian head the masked Gaussian negative
            log-likelihood of the labels (weighted by ``nll_beta``), and ``L_auxiliary`` the masked mean squared
            error of the auxiliary head. And the fit: ``context_mse`` and ``context_r2`` of ``z_t`` (of the mean
            with the Gaussian head); with the Gaussian head ``context_nll`` (unweighted), ``context_variance`` (mean
            predicted variance) and ``context_calibration`` (mean of squared error / variance, 1 when calibrated);
            with an auxiliary head ``auxiliary_mse``, ``auxiliary_r2``, ``context_loss`` and ``encoder_loss``
        """
        encoder = self.policy.encoder
        initial_state = encoder.initial_state(samples.observations.shape[0])
        observations = self.policy.encoder_observations(samples.observations, n_batch_dims=2)
        auxiliaries = None
        if encoder.auxiliary_dim is None:
            latents, _ = encoder.unroll(initial_state, observations, samples.prev_actions, mask=samples.mask)
        else:
            latents, auxiliaries, _ = encoder.unroll_with_auxiliary(initial_state, observations, samples.prev_actions, mask=samples.mask)
        labels = samples.labels[..., : encoder.context_dim]
        estimates, variances = encoder.split_context(latents)
        valid = samples.mask.unsqueeze(-1).to(latents.dtype)
        squared_errors, loss, r2 = self._masked_fit(estimates, labels, valid)
        fit: dict[str, float] = {"context_mse": loss.item(), "context_r2": r2}
        if variances is not None:
            n_values = valid.sum() * labels.shape[-1]
            # Padding steps have a zero variance (the encoder outputs zeros there): replaced before the log
            variances = th.where(valid.bool(), variances, th.ones_like(variances))
            nll = 0.5 * (th.log(variances) + squared_errors / variances)
            # beta-NLL: each step weighted by its variance ** beta, without gradient through the weight
            weights = variances.detach() ** self.nll_beta if self.nll_beta > 0 else th.ones_like(variances)
            loss = (weights * nll * valid).sum() / n_values
            with th.no_grad():
                fit.update(
                    context_nll=((nll * valid).sum() / n_values).item(),
                    context_variance=((variances * valid).sum() / n_values).item(),
                    context_calibration=((squared_errors / variances * valid).sum() / n_values).item(),
                )
        if auxiliaries is not None:
            context_loss = loss
            _, auxiliary_loss, auxiliary_r2 = self._masked_fit(auxiliaries, samples.labels[..., encoder.context_dim :], valid)
            loss = context_loss + self.auxiliary_loss_weight * auxiliary_loss
            fit.update(
                auxiliary_mse=auxiliary_loss.item(),
                auxiliary_r2=auxiliary_r2,
                context_loss=context_loss.item(),
                encoder_loss=loss.item(),
            )
        return loss, fit

    @staticmethod
    def _masked_fit(predictions: th.Tensor, labels: th.Tensor, valid: th.Tensor) -> tuple[th.Tensor, th.Tensor, float]:
        """
        :param predictions: [B, L, D]
        :param labels: [B, L, D]
        :param valid: [B, L, 1], 1 for valid steps and 0 for padding
        :return: The squared errors [B, L, D], their mean over the valid steps (differentiable) and the R^2 of the
            predictions against the labels
        """
        n_values = valid.sum() * labels.shape[-1]
        squared_errors = (predictions - labels) ** 2
        squared_error = (squared_errors * valid).sum()
        with th.no_grad():
            mean_label = (labels * valid).sum() / n_values
            total = (((labels - mean_label) ** 2) * valid).sum().item()
            r2 = 1.0 - squared_error.item() / total if total > 1e-12 else float("nan")
        return squared_errors, squared_error / n_values, r2

    @staticmethod
    def _detach_state(state: Any) -> Any:
        treedef, leaves = _flatten_recurrent_state(state)
        return _unflatten_recurrent_state(treedef, (leaf.detach() for _, leaf in leaves))

    @staticmethod
    def _accumulate(totals: list[th.Tensor | None], grads: Sequence[th.Tensor | None]) -> None:
        for i, grad in enumerate(grads):
            if grad is not None:
                totals[i] = grad if totals[i] is None else totals[i] + grad  # type: ignore[operator]

    @staticmethod
    def _add_grads(critic_grad: th.Tensor | None, actor_grad: th.Tensor | None, actor_weight: float) -> th.Tensor | None:
        if actor_grad is None:
            return critic_grad
        if critic_grad is None:
            return actor_weight * actor_grad
        return critic_grad + actor_weight * actor_grad

    @staticmethod
    def _grad_norm(grads: Sequence[th.Tensor | None]) -> float:
        norms = [th.linalg.vector_norm(grad) for grad in grads if grad is not None]
        return th.linalg.vector_norm(th.stack(norms)).item() if norms else 0.0

    @staticmethod
    def _set_grads(params: list[th.nn.Parameter], grads: Sequence[th.Tensor | None]) -> None:
        for param, grad in zip(params, grads, strict=True):
            param.grad = grad

    def learn(
        self: SelfContextualPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "ContextualPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfContextualPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
