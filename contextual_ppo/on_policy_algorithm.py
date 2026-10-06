from collections.abc import Sequence
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv

from .buffers import ContextualRolloutBuffer
from .episode_buffers import SupervisedContextualEpisodeBuffer
from .policies import ContextualActorCriticPolicy
from .torch_layers import XLSTMRolloutEncoder
from .type_aliases import ContextualPolicyState


class ContextualOnPolicyAlgorithm(OnPolicyAlgorithm):
    """
    The base for on-policy algorithms with a recurrent context encoder (Contextual_PPO).

    Compared to ``OnPolicyAlgorithm``, collection carries a memory per env (the encoder state and the
    previous executed action), reset at every episode end, whether terminated or truncated. At each step
    the policy encodes ``[o_t, a_{t-1}]`` into the context ``z_t`` before it samples ``a_t``. The previous
    action carried to the next step is the action actually sent to the environment (the Gaussian sample
    clipped to the action-space bounds); the rollout buffer also keeps the unclipped sample, whose log
    probability PPO evaluates.

    Each rollout fills a ``ContextualRolloutBuffer`` with ``n_steps`` chronological transitions per env and
    the encoder state before the first one. The buffer gets the real successor of every transition (the
    ``terminal_observation`` at an episode end) and, on truncations, ``gamma * V(final observation)``. That
    value is computed from the memory after ``o_t``, with ``a_t`` as previous action and without a reset,
    because the final observation still belongs to the ending episode. A true termination takes precedence
    over a simultaneous time limit. The value after the last step comes from the same kind of lookahead.
    Lookaheads never change the carried memory, so every observation is encoded exactly once.

    A rollout boundary does not reset the memory, which continues into the next rollout. The envs start new
    episodes whenever the memory is missing: at the first ``learn()``, with ``reset_num_timesteps=True``,
    after ``set_env()`` or loading, and after a callback stopped training in the middle of a rollout (that
    rollout is not trained on). With ``reset_envs_each_rollout``, every rollout also starts with new episodes.

    Besides the ``BasePolicy`` interface, the policy (``ContextualActorCriticPolicy``) must provide:

    - ``initial_state(n_envs) -> ContextualPolicyState``: the memory at episode start
    - ``reset_state(state, episode_start) -> ContextualPolicyState``: reset the rows where an episode starts
    - ``forward(obs, state, deterministic=False) -> (actions, values, log_prob, latents, state_after)``:
      encode ``[obs, state.prev_actions]`` from ``state.encoder_state``, then sample the actions.
      ``values`` [N, 1], ``log_prob`` [N] of ``actions``, ``latents`` the contexts ``z_t`` [N, D_z],
      ``state_after`` the memory after ``obs`` (the collector replaces its ``prev_actions`` by the executed action)
    - ``predict_values(obs, state) -> values``: ``V`` [N, 1] of ``obs`` encoded from the memory ``state``

    None of them may modify the state passed in.

    :param policy: The policy model to use (``ContextualActorCriticPolicy``)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Must be False: gSDE is not supported
    :param sde_sample_freq: Unused, kept for the ``OnPolicyAlgorithm`` interface
    :param rollout_buffer_class: Rollout buffer class to use, ``ContextualRolloutBuffer`` or a subclass.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation, e.g.
        ``estimate_dim`` to store the collected contexts ``z_t``. The algorithm sets ``n_steps``, ``n_envs``,
        the spaces, ``gamma``, ``gae_lambda`` and ``device`` itself.
    :param reset_envs_each_rollout: Start every rollout with new episodes in all envs. Episodes still running
        at the end of a rollout are cut there (bootstrapped from the value of the last observation, as at
        any rollout boundary), then the envs are reset. With ``n_steps`` at least the episode length, every
        episode that starts with a rollout lies complete in its stream: training backpropagates through its
        whole history, from the exact reset state. Episodes that start after an earlier one ended inside the
        rollout are cut at its end. Raises if an episode that started with the rollout is still running at
        its end (``n_steps`` shorter than the episodes). Cut episodes are missing from the Monitor statistics.
    :param context_info_keys: ``(reset_key, step_key)`` of the true context in the env ``info`` dicts, to
        store privileged labels in the buffer for diagnostics (BatteryPlane: ``("b0", "b")``). The label of
        ``o_t`` is the reset info of its episode at an episode start, else the previous step's info; the step
        info of transition ``t`` labels its successor. None stores no labels. Labels never reach the policy.
    :param auxiliary_info_keys: ``(reset_key, step_key)`` of auxiliary targets in the env ``info`` dicts, labelled
        like the true context (BatteryPlane: the true speed ratio ``vx / v_ref``, ``("vx_ratio", "vx_ratio")``
        with the project's label wrapper). The episode buffer stores them after the context labels, for the
        encoder's auxiliary head; they need the episode buffer and never reach the policy. None reads none.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators, also of the buffer's stream permutation
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    policy: ContextualActorCriticPolicy
    rollout_buffer: ContextualRolloutBuffer
    # Complete episodes for supervised context training, filled during collection when set (ContextualPPO with
    # encoder_objective='supervised'); None otherwise
    episode_buffer: SupervisedContextualEpisodeBuffer | None

    # Buffer arguments that follow from the algorithm and the environment
    _ALGORITHM_BUFFER_ARGS = ("n_steps", "n_envs", "observation_space", "action_space", "gamma", "gae_lambda", "device")

    def __init__(
        self,
        policy: str | type[ContextualActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule,
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[ContextualRolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        reset_envs_each_rollout: bool = False,
        context_info_keys: Sequence[str] | None = None,
        auxiliary_info_keys: Sequence[str] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        monitor_wrapper: bool = True,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: tuple[type[spaces.Space], ...] | None = (spaces.Box,),
    ):
        if use_sde:
            raise ValueError(f"{type(self).__name__} does not support gSDE (use_sde=True)")
        # Set before super().__init__(), which may call _setup_model()
        self.reset_envs_each_rollout = bool(reset_envs_each_rollout)
        self.context_info_keys = self._check_context_info_keys(context_info_keys)
        self.auxiliary_info_keys = self._check_context_info_keys(auxiliary_info_keys, "auxiliary_info_keys")
        # Memory of every env before the next step, None until the envs start new episodes
        self._last_policy_state: ContextualPolicyState | None = None
        # True context of the next observation [n_envs, D_c], only with context_info_keys
        self._last_contexts: np.ndarray | None = None
        # Auxiliary targets of the next observation [n_envs, D_x], only with auxiliary_info_keys
        self._last_auxiliary: np.ndarray | None = None
        self.episode_buffer = None

        super().__init__(
            policy=policy,
            env=env,
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
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            monitor_wrapper=monitor_wrapper,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=supported_action_spaces,
        )

    @staticmethod
    def _check_context_info_keys(keys: Sequence[str] | None, name: str = "context_info_keys") -> tuple[str, str] | None:
        if keys is None:
            return None
        # Hydra passes lists
        keys = tuple(keys)
        if len(keys) != 2 or not all(isinstance(key, str) for key in keys):
            raise ValueError(f"{name} must be (reset_key, step_key), two strings, got {keys!r}")
        return keys  # type: ignore[return-value]

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        # Saved models store the keys as JSON, which turns the tuple into a list
        self.context_info_keys = self._check_context_info_keys(self.context_info_keys)
        self.auxiliary_info_keys = self._check_context_info_keys(self.auxiliary_info_keys, "auxiliary_info_keys")

        if self.rollout_buffer_class is None:
            self.rollout_buffer_class = ContextualRolloutBuffer  # type: ignore[assignment]
        if not issubclass(self.rollout_buffer_class, ContextualRolloutBuffer):  # type: ignore[arg-type]
            raise ValueError(f"{type(self).__name__} requires a ContextualRolloutBuffer, not {self.rollout_buffer_class}")
        conflicts = sorted(set(self._ALGORITHM_BUFFER_ARGS) & set(self.rollout_buffer_kwargs))
        if conflicts:
            raise ValueError(f"rollout_buffer_kwargs can't set {conflicts}: the algorithm and the environment set them")

        buffer_kwargs = {
            "recurrent_state_batch_axis": XLSTMRolloutEncoder.STATE_BATCH_AXIS,
            "seed": self.seed,
            **self.rollout_buffer_kwargs,
        }
        if self.context_info_keys is not None:
            buffer_kwargs.setdefault("context_dim", 1)
        elif buffer_kwargs.get("context_dim") is not None:
            raise ValueError("Storing true contexts (rollout_buffer_kwargs['context_dim']) requires context_info_keys")

        self.rollout_buffer = self.rollout_buffer_class(  # type: ignore[misc]
            self.n_steps,
            self.n_envs,
            self.observation_space,
            self.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
            **buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu("ContextualActorCriticPolicy")

    def _excluded_save_params(self) -> list[str]:
        # The envs start new episodes after loading
        return [*super()._excluded_save_params(), "_last_policy_state", "_last_contexts", "_last_auxiliary", "episode_buffer"]

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:
        """
        cf `BaseAlgorithm`.
        """
        if self._vec_normalize_env is not None:
            raise ValueError(f"{type(self).__name__} does not support VecNormalize: the buffer stores observations as given")
        if self._last_policy_state is None:
            # Without memory the current episodes can't be continued, force an env reset
            self._last_obs = None  # type: ignore[assignment]
        # Same condition as in ``BaseAlgorithm._setup_learn()``, which resets the envs
        reset_envs = reset_num_timesteps or self._last_obs is None

        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        if reset_envs:
            assert self.env is not None
            self._start_new_episodes(self.env, reset_envs=False)
        return total_timesteps, callback

    def _start_new_episodes(self, env: VecEnv, reset_envs: bool = True) -> None:
        """
        Initial memory and true contexts for new episodes in every env.

        :param env: The training environment
        :param reset_envs: Reset the envs first; False when they were just reset
        """
        if self.episode_buffer is not None:
            # The unfinished episodes end here without a termination or truncation: they are not completed
            self.episode_buffer.drop_unfinished()
        if reset_envs:
            self._last_obs = env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((env.num_envs,), dtype=bool)
        self._last_policy_state = self.policy.initial_state(env.num_envs)
        # VecEnv wrappers keep their own (empty) reset_infos
        if self.context_info_keys is not None:
            self._last_contexts = self._context_labels(env.unwrapped.reset_infos, self.context_info_keys[0])
        if self.auxiliary_info_keys is not None:
            self._last_auxiliary = self._context_labels(
                env.unwrapped.reset_infos, self.auxiliary_info_keys[0], "auxiliary_info_keys"
            )

    @staticmethod
    def _context_labels(infos: Sequence[dict[str, Any]], key: str, name: str = "context_info_keys") -> np.ndarray:
        """
        True contexts (or auxiliary targets, ``name``) read from ``info`` dicts, [len(infos), D] float32.
        """
        labels = []
        for i, info in enumerate(infos):
            if key not in info:
                raise KeyError(f"{name}: the info of env {i} has no {key!r} entry (it has {sorted(info)})")
            labels.append(np.asarray(info[key], dtype=np.float32).reshape(-1))
        return np.stack(labels)

    def _next_labels(
        self, env: VecEnv, next_labels: np.ndarray, dones: np.ndarray, reset_key: str, name: str = "context_info_keys"
    ) -> np.ndarray:
        """
        Labels of the next step's observations: the step infos' (``next_labels``), except where an episode ended,
        where the next observation starts a new episode and its label is in the reset info.
        """
        labels = next_labels.copy()
        done_indices = np.flatnonzero(dones)
        if len(done_indices) > 0:
            reset_infos = env.unwrapped.reset_infos
            labels[done_indices] = self._context_labels([reset_infos[i] for i in done_indices], reset_key, name)
        return labels

    def _executed_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        The action sent to the environment: inside the action-space bounds, as the encoder's next previous action.
        """
        assert isinstance(self.action_space, spaces.Box)
        if self.policy.squash_output:
            # Unscale the actions to match env bounds if they were previously squashed (scaled in [-1, 1])
            actions = self.policy.unscale_action(actions)
        # The Gaussian is unbounded: clip to avoid out of bound errors
        return np.clip(actions, self.action_space.low, self.action_space.high)

    @staticmethod
    def _real_successors(new_obs: np.ndarray, dones: np.ndarray, infos: Sequence[dict[str, Any]]) -> np.ndarray:
        """
        Successor of each transition: ``new_obs``, except the final observation where an episode ended
        (the VecEnv already returned the first observation of the next episode there).
        """
        next_obs = np.array(new_obs, copy=True)
        for i in np.flatnonzero(dones):
            terminal_obs = infos[i].get("terminal_observation")
            if terminal_obs is None:
                raise RuntimeError(
                    f"Env {i} ended an episode without info['terminal_observation']: use an SB3 VecEnv, "
                    "which stores the final observation before its automatic reset"
                )
            next_obs[i] = terminal_obs
        return next_obs

    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: ContextualRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``ContextualRolloutBuffer``,
        carrying the memory of every env across steps and rollouts.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment, the buffer's ``n_steps``
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        assert self._last_policy_state is not None, "No memory was set, the envs must start new episodes first"
        if n_rollout_steps != rollout_buffer.n_steps:
            raise ValueError(f"n_rollout_steps={n_rollout_steps} must equal the buffer's n_steps={rollout_buffer.n_steps}")
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        if self.reset_envs_each_rollout and not np.all(self._last_episode_starts):
            self._start_new_episodes(env)

        n_steps = 0
        rollout_buffer.reset()
        rollout_buffer.set_initial_state(self._last_policy_state.encoder_state)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            state = self._last_policy_state
            assert state is not None
            obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)  # type: ignore[arg-type]
            with th.no_grad():
                # z_t from [o_t, a_{t-1}], then a_t ~ pi(. | o_t, z_t)
                actions, values, log_probs, latents, state_after = self.policy(obs_tensor, state)
            actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]
            clipped_actions = self._executed_actions(actions)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                # This step is not stored, so the current episodes can't be continued:
                # the envs start new episodes at the next learn()
                self._last_policy_state = None
                self._last_contexts = None
                self._last_auxiliary = None
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Memory after o_t, whose previous action is the action actually sent
            memory = state_after._replace(prev_actions=th.as_tensor(clipped_actions, dtype=th.float32, device=self.device))

            # Effective end flags: VecEnvs set TimeLimit.truncated only when the episode didn't also terminate
            truncated = np.array([bool(done and info.get("TimeLimit.truncated", False)) for done, info in zip(dones, infos)])
            terminated = dones & ~truncated
            next_obs = self._real_successors(new_obs, dones, infos)

            # Handle timeout by bootstrapping with value function, see GitHub issue #633.
            # The final observation is encoded from the memory after o_t (no reset): it ends the same episode.
            timeout_bootstrap = np.zeros(env.num_envs, dtype=np.float32)
            if truncated.any():
                with th.no_grad():
                    final_values = self.policy.predict_values(self.policy.obs_to_tensor(next_obs)[0], memory)
                timeout_bootstrap[truncated] = self.gamma * final_values.cpu().numpy().reshape(-1)[truncated]

            optional: dict[str, Any] = {}
            if self.context_info_keys is not None:
                # The step info holds the true context after the transition (BatteryPlane: b_{t+1})
                next_contexts = self._context_labels(infos, self.context_info_keys[1])
                optional.update(contexts=self._last_contexts, next_contexts=next_contexts)
            if rollout_buffer.estimate_dim is not None:
                optional.update(context_estimates=latents)
            if self.auxiliary_info_keys is not None:
                # Auxiliary targets after the transition (BatteryPlane: vx_{t+1} / v_ref)
                next_auxiliary = self._context_labels(infos, self.auxiliary_info_keys[1], "auxiliary_info_keys")

            rollout_buffer.add(
                observations=self._last_obs,  # type: ignore[arg-type]
                next_observations=next_obs,
                prev_actions=state.prev_actions.cpu().numpy(),
                actions=actions,
                executed_actions=clipped_actions,
                rewards=rewards,
                episode_starts=self._last_episode_starts,  # type: ignore[arg-type]
                terminated=terminated,
                truncated=truncated,
                values=values,
                log_probs=log_probs,
                timeout_bootstrap=timeout_bootstrap,
                **optional,
            )
            if self.episode_buffer is not None:
                # (o_t, label of o_t) and the executed a_t; where the episode ended, also its final observation
                # and that observation's label (the step info), before the next episode starts
                assert self.context_info_keys is not None, "the episode buffer needs context_info_keys"
                labels, final_labels = self._last_contexts, next_contexts
                if self.auxiliary_info_keys is not None:
                    # The auxiliary targets follow the context in every label row
                    labels = np.concatenate([labels, self._last_auxiliary], axis=1)  # type: ignore[list-item]
                    final_labels = np.concatenate([final_labels, next_auxiliary], axis=1)
                self.episode_buffer.add(
                    observations=self._last_obs,  # type: ignore[arg-type]
                    executed_actions=clipped_actions,
                    labels=labels,  # type: ignore[arg-type]
                    dones=dones,
                    final_observations=next_obs,
                    final_labels=final_labels,
                )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            # Envs that finished an episode start the next one from the initial memory
            self._last_policy_state = self.policy.reset_state(memory, dones)
            if self.context_info_keys is not None:
                self._last_contexts = self._next_labels(env, next_contexts, dones, self.context_info_keys[0])
            if self.auxiliary_info_keys is not None:
                self._last_auxiliary = self._next_labels(
                    env, next_auxiliary, dones, self.auxiliary_info_keys[0], "auxiliary_info_keys"
                )

        with th.no_grad():
            # Compute value for the last timestep, from the carried memory (left unchanged)
            values = self.policy.predict_values(self.policy.obs_to_tensor(new_obs)[0], self._last_policy_state)

        rollout_buffer.finalize()
        if self.reset_envs_each_rollout:
            # Every stream started with an episode start: that episode must have ended within the rollout
            unfinished = ~np.any(rollout_buffer.terminated | rollout_buffer.truncated, axis=0)
            if unfinished.any():
                raise ValueError(
                    f"reset_envs_each_rollout requires n_steps ({n_rollout_steps}) to be at least the episode length: "
                    f"the episodes that started with this rollout in envs {np.flatnonzero(unfinished)} are still running"
                )
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
