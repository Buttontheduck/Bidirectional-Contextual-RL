"""Step 2 of ENCODER_PLAN.md: a frozen battery estimate appended to the observation, so plain PPO trains on [o_t, z_t].

ContextVecEnv wraps a VecEnv of BatteryPlane environments in hidden mode. It keeps one memory of a frozen context source
per environment and, at every step, feeds the source the new observation and the action just executed (clipped to the
action space, as the environment executes it). Where an episode starts, the memory is reset and the previous action is
zero, as in Contextual_PPO's collector. The context is appended to every observation, including the final observation
of an episode (info["terminal_observation"]), which the source sees with the episode's last action before its memory
restarts on the next episode's first observation.

Sources (cfg.context.source; the plain-PPO config, ppo/configs/battery_plane_ppo.yaml):
    none          no wrapper: plain PPO
    ideal         IdealBatteryEstimator (step1_ideal_estimator.py): the posterior mean of b_t, and its std with
                  context.include_std; the best estimate any encoder could make, from hidden observations and actions
    encoder       the frozen encoder of a saved Contextual_PPO model (context.encoder_model): z_t as that model's actor
                  and critic read it (Contextual_PPO's context_inputs)
    true_battery  the true b_t from the env infos (privileged): reproduces observed mode exactly, a check of the wrapper
Only true_battery reads the env infos; the others read hidden-mode observations and executed actions, never the battery.
Nothing in a source is trained: the encoder runs without gradient and is not part of the PPO model.

Project glue, like the training scripts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

CONTEXT_SOURCES = ("none", "ideal", "encoder", "true_battery")
PROJECT_ROOT = Path(__file__).resolve().parent
# [y, vx + noise, vy, tau]: hidden mode, no battery column
HIDDEN_OBS_DIM = 4


class ContextSource(ABC):
    """A frozen battery estimate with one memory per environment."""

    #: Width of the context appended to the observation
    dim: int
    #: True if the source reads the env infos (privileged; checks only)
    privileged: bool = False

    @abstractmethod
    def step(self, rows: np.ndarray, observations: np.ndarray, prev_actions: np.ndarray, starts: np.ndarray,
             finals: np.ndarray, infos) -> np.ndarray:
        """
        Advance the memories of the environments in ``rows`` by one observation.

        :param rows: [n] bool, the environments to advance (the others keep their memory)
        :param observations: [n, obs_dim] the new observations
        :param prev_actions: [n, action_dim] the executed actions that led to them (ignored where ``starts``)
        :param starts: [n] bool, the observation is an episode's first: reset the memory before it
        :param finals: [n] bool, the observation is an episode's last
        :param infos: the env infos of these observations (step infos, or reset infos where ``starts``)
        :return: [n, dim] contexts (meaningful in ``rows`` only)
        """


class TrueBatteryContext(ContextSource):
    """The true battery from the env infos: info["b0"] at a reset, info["b"] after a step. A check, never an input."""

    dim = 1
    privileged = True

    def step(self, rows, observations, prev_actions, starts, finals, infos) -> np.ndarray:
        contexts = np.zeros((len(observations), 1), dtype=np.float32)
        for i in np.flatnonzero(rows):
            contexts[i, 0] = infos[i]["b0"] if starts[i] else infos[i]["b"]
        return contexts


class IdealContext(ContextSource):
    """
    The ideal estimator (step1_ideal_estimator.IdealBatteryEstimator), one per environment: the posterior mean of b_t,
    and its std with ``include_std``.
    """

    def __init__(self, n_envs: int, env_kwargs: dict, include_std: bool = False, b_points: int = 101,
                 vx_points: int = 301, vy_tolerance: float = 0.01) -> None:
        from step1_ideal_estimator import IdealBatteryEstimator

        self.estimators = [IdealBatteryEstimator(env_kwargs, b_points=b_points, vx_points=vx_points,
                                                 vy_tolerance=vy_tolerance) for _ in range(n_envs)]
        self.include_std = include_std
        self.dim = 2 if include_std else 1

    def step(self, rows, observations, prev_actions, starts, finals, infos) -> np.ndarray:
        contexts = np.zeros((len(observations), self.dim), dtype=np.float32)
        for i in np.flatnonzero(rows):
            estimator = self.estimators[i]
            if starts[i]:
                mean, std = estimator.start(observations[i])
            else:
                mean, std = estimator.update(prev_actions[i], observations[i], alive=not finals[i])
            contexts[i] = (mean, std)[:self.dim]
        return contexts


class FrozenEncoderContext(ContextSource):
    """
    The encoder of a saved Contextual_PPO model, frozen: z_t from [o_t, a_{t-1}] with the memory carried through the
    episode, as that model's actor and critic read it (``context_inputs``: e.g. [mu, log sigma2] for a Gaussian head).
    All environments step together, with their memories reset where an episode starts, as in Contextual_PPO's collector.
    """

    def __init__(self, n_envs: int, model_path: str | Path) -> None:
        import torch as th

        from contextual_ppo import ContextualPPO
        from contextual_ppo.torch_layers import XLSTMRolloutEncoder

        model = ContextualPPO.load(str(model_path), device="cpu")
        self.policy = model.policy
        self.policy.set_training_mode(False)
        self.encoder = self.policy.encoder
        if not isinstance(self.encoder, XLSTMRolloutEncoder):
            raise ValueError(f"{model_path} has no learned encoder (an oracle or memoryless policy?)")
        if self.encoder.observation_dim != HIDDEN_OBS_DIM:
            raise ValueError(f"{model_path}: the encoder must read hidden-mode observations ({HIDDEN_OBS_DIM} values)")
        self.th = th
        self.n_envs = n_envs
        self.action_dim = self.encoder.action_dim
        self.state = self.encoder.initial_state(n_envs)
        with th.no_grad():
            latents, _ = self.encoder.step(self.encoder.initial_state(1), th.zeros(1, HIDDEN_OBS_DIM),
                                           th.zeros(1, self.action_dim))
            self.dim = int(self.policy.context_inputs(latents).shape[-1])
        # Frozen: nothing may train it
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def step(self, rows, observations, prev_actions, starts, finals, infos) -> np.ndarray:
        th = self.th
        starts_t = th.as_tensor(starts, dtype=th.bool)
        prev = th.as_tensor(np.asarray(prev_actions, dtype=np.float32))
        prev = th.where(starts_t.unsqueeze(-1), th.zeros_like(prev), prev)
        with th.no_grad():
            state = self.encoder.reset_state(self.state, starts_t)
            obs = self.policy.encoder_observations(th.as_tensor(np.asarray(observations, dtype=np.float32)))
            latents, new_state = self.encoder.step(state, obs, prev)
            contexts = self.policy.context_inputs(latents)
            # Only the rows that stepped take the new memory
            self.state = self.encoder._select_state(th.as_tensor(rows, dtype=th.bool), new_state, self.state)
        return contexts.numpy().astype(np.float32)


class ContextVecEnv(VecEnvWrapper):
    """
    Appends a frozen context source's output to the observations of a VecEnv (see the module docstring).

    :param venv: The environments (hidden mode unless the source is privileged)
    :param source: The context source, with one memory per environment of ``venv``
    """

    def __init__(self, venv: VecEnv, source: ContextSource) -> None:
        space = venv.observation_space
        if not (isinstance(space, spaces.Box) and len(space.shape) == 1):
            raise ValueError("ContextVecEnv needs flat Box observations")
        if not source.privileged and space.shape[0] != HIDDEN_OBS_DIM:
            raise ValueError("The context sources read hidden-mode observations, never the battery: "
                             f"expected {HIDDEN_OBS_DIM} values, got {space.shape[0]}")
        width = space.shape[0] + source.dim
        super().__init__(venv, observation_space=spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32))
        self.source = source
        self.action_low, self.action_high = venv.action_space.low, venv.action_space.high
        self._actions: np.ndarray | None = None

    def _join(self, observations: np.ndarray, contexts: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(observations, dtype=np.float32), contexts.astype(np.float32)], axis=-1)

    def reset(self) -> np.ndarray:
        observations = self.venv.reset()
        everyone = np.ones(self.num_envs, dtype=bool)
        contexts = self.source.step(everyone, observations, np.zeros((self.num_envs, len(self.action_low)), np.float32),
                                    everyone, ~everyone, self.venv.reset_infos)
        return self._join(observations, contexts)

    def step_async(self, actions: np.ndarray) -> None:
        # The executed action: the environment clips it to the action space
        self._actions = np.clip(np.asarray(actions, dtype=np.float32).reshape(self.num_envs, -1),
                                self.action_low, self.action_high)
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        dones = np.asarray(dones, dtype=bool)
        everyone, nobody = np.ones(self.num_envs, dtype=bool), np.zeros(self.num_envs, dtype=bool)
        # Ended episodes: their final observation continues their memory, with their last action
        reached = np.array(observations, copy=True)
        for i in np.flatnonzero(dones):
            reached[i] = infos[i]["terminal_observation"]
        contexts = self.source.step(everyone, reached, self._actions, nobody, dones, infos)
        for i in np.flatnonzero(dones):
            infos[i]["terminal_observation"] = self._join(reached[i], contexts[i])
        if dones.any():
            # The next episode's first observation (VecEnv auto-reset): a fresh memory
            first = self.source.step(dones, observations, np.zeros_like(self._actions), dones, nobody,
                                     self.venv.reset_infos)
            contexts[dones] = first[dones]
        return self._join(observations, contexts), rewards, dones, infos


def context_source_from(cfg) -> str:
    """The run's context source; configs without a context section (step 0, step 1, A-F) have none."""
    context = cfg.get("context")
    return "none" if context is None else str(context.get("source", "none"))


def context_protocol(cfg) -> dict | None:
    """The context settings that change what the policy reads (report_step0.py's protocol): None without a context."""
    from omegaconf import OmegaConf

    if context_source_from(cfg) == "none":
        return None
    settings = OmegaConf.to_container(cfg.context, resolve=True)
    source = settings["source"]
    keep = {"ideal": ("source", "include_std", "b_points", "vx_points", "vy_tolerance"),
            "encoder": ("source", "encoder_model"), "true_battery": ("source",)}[source]
    return {key: settings.get(key) for key in keep}


def encoder_model_path(cfg) -> Path:
    """context.encoder_model, relative paths from the project root (like training.init_model)."""
    path = Path(str(cfg.context.encoder_model))
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_context(cfg, env_kwargs: dict) -> None:
    """Cross-field constraints of cfg.context, checked before a run starts. env_kwargs: the run's BatteryPlane arguments."""
    source = context_source_from(cfg)
    if source not in CONTEXT_SOURCES:
        raise ValueError(f"context.source must be one of {CONTEXT_SOURCES}")
    if source == "none":
        return
    if cfg.env.observation_mode != "hidden":
        raise ValueError(f"context.source={source} appends a battery estimate to hidden-mode observations: it needs "
                         "env.observation_mode=hidden")
    if float(cfg.env.get("battery_noise", 0.0)) != 0.0:
        raise ValueError("context.source and env.battery_noise exclude each other (the noisy battery is observed mode)")
    context = cfg.context
    if source == "ideal":
        for key in ("b_points", "vx_points"):
            value = context.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"context.{key} must be an integer >= 2")
        tolerance = context.get("vy_tolerance")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not tolerance > 0:
            raise ValueError("context.vy_tolerance must be positive")
        if not isinstance(context.get("include_std"), bool):
            raise ValueError("context.include_std must be true or false")
        # The estimator assumes the environment's speed noise: the validation flights must use the same
        import battery_plane as bp

        sigma = bp.BatteryPlaneVec(n_envs=1, observation_mode="hidden", seed=0, **env_kwargs).sigma
        noises = cfg.evaluation.reset_grid.get("eval_vx_obs_noise")
        if noises is not None and [float(x) for x in noises] != [sigma]:
            raise ValueError(f"context.source=ideal assumes the speed noise of the environment ({sigma:g} m/s); the "
                             f"validation flights use {list(noises)}")
    if source == "encoder" and (not context.get("encoder_model") or not encoder_model_path(cfg).is_file()):
        raise ValueError("context.source=encoder needs context.encoder_model: a saved Contextual_PPO model (.zip)")


def make_context_source(cfg, n_envs: int, env_kwargs: dict) -> ContextSource | None:
    source = context_source_from(cfg)
    if source == "none":
        return None
    context = cfg.context
    if source == "ideal":
        return IdealContext(n_envs, env_kwargs, include_std=bool(context.include_std), b_points=int(context.b_points),
                            vx_points=int(context.vx_points), vy_tolerance=float(context.vy_tolerance))
    if source == "encoder":
        return FrozenEncoderContext(n_envs, encoder_model_path(cfg))
    return TrueBatteryContext()


def wrap_context(cfg, venv: VecEnv, env_kwargs: dict) -> VecEnv:
    """venv with cfg.context's source appended to its observations; venv itself without a context."""
    source = make_context_source(cfg, venv.num_envs, env_kwargs)
    return venv if source is None else ContextVecEnv(venv, source)
