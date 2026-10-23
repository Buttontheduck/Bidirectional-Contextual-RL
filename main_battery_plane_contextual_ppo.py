"""Contextual_PPO (PPO + xLSTM context encoder) training for BatteryPlane, with the configuration in config/.

    python main_battery_plane_contextual_ppo.py                                     # base configuration = experiment A
    python main_battery_plane_contextual_ppo.py -m +experiment=A_point_baseline training.seed=0,1,2
    python main_battery_plane_contextual_ppo.py -m +experiment=A_point_baseline,B_point_speed training.seed=0,1,2

Everything it needs is in this repository: the environment (battery_plane.py), the learner
(contextual_ppo/, imported from the repository root) and config/cppo.yaml with
its experiments. The environment, evaluation grid, rendering, checkpoints and W&B logging are those of the
BatteryPlane PPO baseline, whose helpers are copied below.

The learner is contextual_ppo.ContextualPPO (design: Contextual_PPO_project_plan.md in that
package). Evaluation carries the recurrent memory: SB3's evaluate_policy passes the state and the episode starts
to predict(). For a custom inference loop, pass the returned state back at every step:

    state, episode_start = None, np.ones(env.num_envs, dtype=bool)
    action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_start = done
"""

from __future__ import annotations

import json
import math
import time
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import TransformReward
import hydra
import stable_baselines3
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from wandb.integration.sb3 import WandbCallback

# The repository root holds battery_plane.py, the single source of truth for the environment, and contextual_ppo/.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import battery_plane as bp  # noqa: E402
import contextual_ppo  # noqa: E402
from battery_plane import iter_eval_resets  # noqa: E402
from contextual_ppo import ContextualPPO  # noqa: E402
from contextual_ppo.policies import CONTEXT_VARIANCE_INPUTS  # noqa: E402
from contextual_ppo.torch_layers import CONTEXT_ACTIVATIONS, CONTEXT_HEADS, ENCODER_ARCHS  # noqa: E402
from flight_logging import (  # noqa: E402
    FlightLoggingEvalCallback, GradNormLogger, RolloutFlightLogger, evaluate_flights, log_final_flights, terminal_outcome,
)
from context_wrapper import context_source_from as frozen_context_source, wrap_context  # noqa: E402
# A plain-PPO memory baseline's input, for its final validation in report_step0.py (none for every CPPO config)
from memory_baselines import memory_wrapper_from  # noqa: E402
from step0_oracle import NoContextPolicy, TrueBatteryPolicy  # noqa: E402
from step1_controls import (  # noqa: E402
    FLIGHT_PROFILES, BatteryNoiseSwitch, battery_noise_at, make_noisy_battery, switch_source_mismatches,
)

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}
# cppo.context_source: the xLSTM encoder, the true battery of step 0, or no context at all (memoryless; step0_oracle.py)
CONTEXT_SOURCES = ("encoder", "oracle", "none")
# Policy of each context source; None: cppo.policy with the xLSTM encoder
CONTEXT_POLICIES = {"encoder": None, "oracle": TrueBatteryPolicy, "none": NoContextPolicy}


def encoder_arch_from(cfg: DictConfig) -> tuple[str, int, int]:
    """The encoder's architecture and the GRU's width and layers; saved configs from before cppo.encoder_arch used the
    xLSTM."""
    gru = cfg.cppo.get("gru") or {}
    return str(cfg.cppo.get("encoder_arch", "xlstm")), int(gru.get("hidden_dim", 128)), int(gru.get("num_layers", 2))


def context_source_from(cfg: DictConfig) -> str:
    """Saved A–F configs predate context_source and use the learned encoder."""
    return cfg.cppo.get("context_source", "encoder")


def reward_scale_from(cfg: DictConfig) -> float:
    """Saved A–F configs predate reward_scale and trained on unscaled rewards."""
    return float(cfg.cppo.get("reward_scale", 1.0))


def battery_noise_from(cfg: DictConfig) -> float:
    """Std of the noise on the observed battery (step 1's noisy oracle); older saved configs have none."""
    return float(cfg.env.get("battery_noise", 0.0))


def initial_model_from(cfg: DictConfig) -> Path | None:
    """
    The saved model a run continues instead of starting from scratch (training.init_model, step 1's late switch), None
    for a fresh run. Older saved configs have no such key.
    """
    value = cfg.training.get("init_model")
    # Late-switch configs say REQUIRED until the launcher names the model (Hydra keeps a default under ???)
    if OmegaConf.is_missing(cfg.training, "init_model") or value == "REQUIRED":
        raise ValueError("training.init_model must name the saved final_model.zip this run continues")
    if value is None:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


class ScheduledEvalResets(gym.Wrapper):
    """Apply a repeatable sequence of seeds and options to evaluation resets."""

    def __init__(self, env: gym.Env, reset_schedule: Sequence[Mapping[str, Any]]):
        super().__init__(env)
        if not reset_schedule:
            raise ValueError("The evaluation reset schedule cannot be empty")
        self.reset_schedule = tuple(
            {"seed": reset_kwargs["seed"], "options": dict(reset_kwargs.get("options", {}))}
            for reset_kwargs in reset_schedule
        )
        self._next_reset = 0

    def rewind(self) -> None:
        """Start the configured reset sequence from its first episode."""
        self._next_reset = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        reset_kwargs = self.reset_schedule[self._next_reset % len(self.reset_schedule)]
        self._next_reset += 1
        merged_options = dict(options or {})
        merged_options.update(reset_kwargs["options"])
        return self.env.reset(seed=reset_kwargs["seed"], options=merged_options)


class RenderingEvalCallback(FlightLoggingEvalCallback):
    """EvalCallback that logs the flights of every evaluation (FlightLoggingEvalCallback) and opens the pygame
    window only while it is evaluating.

    `render_env` must be the *unwrapped* BatteryPlaneEnv behind the eval env: it renders
    itself inside step()/reset() whenever render_mode == "human", so flipping that
    attribute around the evaluation is all that is needed.
    """

    def __init__(
        self,
        *args,
        render_env: gym.Env | None = None,
        render_fps: int | None = None,
        reset_scheduler: ScheduledEvalResets | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.render_env = render_env
        self.render_fps = render_fps
        self.reset_scheduler = reset_scheduler

    def _on_step(self) -> bool:
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        rendering = evaluating and self.render_env is not None
        if evaluating and self.reset_scheduler is not None:
            self.reset_scheduler.rewind()
        if rendering:
            if self.render_fps is not None:
                self.render_env.metadata = dict(self.render_env.metadata, render_fps=self.render_fps)
            self.render_env.render_mode = "human"
        try:
            return super()._on_step()
        finally:
            if rendering:
                self.render_env.render_mode = None
                self.render_env.close()  # tear the window down until the next eval


def env_kwargs_from(cfg: DictConfig) -> dict[str, Any]:
    """Extra BatteryPlaneVec arguments from env.kwargs, plus the legacy crash_penalty key."""
    extra = cfg.env.get("kwargs")
    kwargs: dict[str, Any] = dict(OmegaConf.to_container(extra, resolve=True)) if extra is not None else {}
    if cfg.env.crash_penalty is not None:
        kwargs["crash_penalty"] = cfg.env.crash_penalty
    return kwargs


class TrueSpeedInfo(gym.Wrapper):
    """
    Adds the true speed ratio vx / v_ref of the returned observation to its reset and step infos, under "vx_ratio":
    the target of the velocity head (cppo.velocity_head). The observation holds the same ratio with noise in hidden
    mode (its second column). BatteryPlane's reset info has no speed, so the wrapper reads the env's state.
    """

    KEY = "vx_ratio"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, {**info, self.KEY: self._speed_ratio()}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, {**info, self.KEY: self._speed_ratio()}

    def _speed_ratio(self) -> float:
        # The single-plane core behind BatteryPlaneEnv; after a final step it still holds the final state
        core = self.env.unwrapped.core
        return float(core.vx[0] / core.v_ref)


def make_train_env(cfg: DictConfig) -> VecEnv:
    """n_envs copies of BatteryPlaneEnv, each with its own random stream; with the velocity head, each copy also
    reports the true speed (TrueSpeedInfo, inside its Monitor). With cppo.reward_scale != 1, the learner gets the
    rewards x reward_scale (TransformReward, outside the Monitor, whose episode returns stay in env units)."""
    kwargs = env_kwargs_from(cfg)
    observation_mode = cfg.env.observation_mode
    base_seed = int(cfg.training.seed)
    speed_labels = bool(cfg.cppo.velocity_head.enabled)
    reward_scale = reward_scale_from(cfg)
    battery_noise = battery_noise_from(cfg)

    def factory(index: int):
        def _init() -> gym.Env:
            env = bp.BatteryPlaneEnv(observation_mode=observation_mode, **kwargs)
            if battery_noise > 0:
                env = make_noisy_battery(env, cfg, battery_noise)
            if speed_labels:
                env = TrueSpeedInfo(env)
            env = Monitor(env)
            if reward_scale != 1.0:
                env = TransformReward(env, lambda reward: reward_scale * reward)
            # Only until learning starts: SB3 then re-seeds the VecEnv (env.seed(training.seed)), so copy i
            # starts training from reset seed training.seed + i. validate_config keeps both from the held-out seeds.
            seed = base_seed * 100 + index
            env.action_space.seed(seed)
            env.reset(seed=seed)
            return env

        return _init

    n_envs = int(cfg.training.n_envs)
    factories = [factory(index) for index in range(n_envs)]
    if str(cfg.training.vec_env).lower() == "subproc" and n_envs > 1:
        return SubprocVecEnv(factories, start_method="spawn")
    return DummyVecEnv(factories)


def make_eval_env(cfg: DictConfig, reset_schedule: Sequence[Mapping[str, Any]], battery_noise: float | None = None):
    """One environment that replays the fixed evaluation grid, as in the SAC script. battery_noise: the noise std on the
    visible battery, env.battery_noise by default (the final validation of an early-noise run passes 0)."""
    raw_env = bp.BatteryPlaneEnv(observation_mode=cfg.env.observation_mode, **env_kwargs_from(cfg))
    noise = battery_noise_from(cfg) if battery_noise is None else battery_noise
    # The noisy oracle's noise sits inside the scheduler, which reseeds it with every validation flight's seed
    observed = make_noisy_battery(raw_env, cfg, noise) if noise > 0 else raw_env
    # A plain-PPO memory baseline (memory_baselines.py, for report_step0.py) reads its window or LSTM input, as in training
    memory = memory_wrapper_from(cfg)
    if memory is not None:
        observed = memory(observed)
    scheduler = ScheduledEvalResets(observed, reset_schedule)
    env = Monitor(scheduler)
    env.action_space.seed(int(cfg.training.seed) + 1)
    env.reset()
    scheduler.rewind()   # the line above consumed the first entry; every evaluation rewinds anyway
    # A plain-PPO run with a frozen context (context_wrapper.py, validated here by report_step0.py) reads it too
    return wrap_context(cfg, DummyVecEnv([lambda: env]), env_kwargs_from(cfg)), scheduler, raw_env


def validate_eval_reset_grid(reset_grid: DictConfig) -> None:
    """Same constraints as BatteryPlane's argparse helper (copied from the SAC script)."""
    if not reset_grid.eval_seeds:
        raise ValueError("evaluation.reset_grid.eval_seeds cannot be empty")
    for seed in reset_grid.eval_seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Evaluation seeds must be non-negative integers")

    batteries = reset_grid.eval_initial_battery
    if batteries is not None:
        if not batteries:
            raise ValueError("evaluation.reset_grid.eval_initial_battery cannot be empty")
        for battery in batteries:
            if not math.isfinite(battery) or not 0.0 < battery <= 1.0:
                raise ValueError("Initial battery levels must be in (0, 1]")

    noises = reset_grid.eval_vx_obs_noise
    if noises is not None:
        if not noises:
            raise ValueError("evaluation.reset_grid.eval_vx_obs_noise cannot be empty")
        for noise in noises:
            if not math.isfinite(noise) or noise < 0.0:
                raise ValueError("Evaluation speed noise must be finite and non-negative")

    y0s = reset_grid.get("eval_y0")
    if y0s is not None:
        if not y0s:
            raise ValueError("evaluation.reset_grid.eval_y0 cannot be empty (use null for random)")
        for y0 in y0s:
            if not math.isfinite(y0) or not 0.0 < y0 <= 200.0:
                raise ValueError("Evaluation start altitudes must be in (0, y_max=200]")
    vx0s = reset_grid.get("eval_vx0")
    if vx0s is not None:
        if not vx0s:
            raise ValueError("evaluation.reset_grid.eval_vx0 cannot be empty (use null for random)")
        for vx0 in vx0s:
            if not math.isfinite(vx0):
                raise ValueError("Evaluation start speeds must be finite")


def _positive_int(cfg: DictConfig, key: str) -> int:
    value = OmegaConf.select(cfg, key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


def validate_config(cfg: DictConfig) -> None:
    """Reject invalid settings before starting logging or worker processes."""
    validate_eval_reset_grid(cfg.evaluation.reset_grid)
    validate_eval_reset_grid(cfg.final_test.reset_grid)
    plots = cfg.evaluation.get("plots")
    if plots is not None:
        _positive_int(cfg, "evaluation.plots.frequency")
        _positive_int(cfg, "evaluation.frequency")
        if int(plots.frequency) % int(cfg.evaluation.frequency):
            raise ValueError("evaluation.plots.frequency must be a multiple of evaluation.frequency")
        grid = cfg.evaluation.reset_grid.eval_initial_battery
        if grid is not None and not set(plots.initial_battery) <= set(grid):
            raise ValueError("evaluation.plots.initial_battery must be starting charges of the validation grid")
    validation_seeds = set(cfg.evaluation.reset_grid.eval_seeds)
    test_seeds = set(cfg.final_test.reset_grid.eval_seeds)
    seed, n_envs = int(cfg.training.seed), int(cfg.training.n_envs)
    # Env copy i is first reset with seed * 100 + i (make_train_env), then SB3 re-seeds it with seed + i when
    # learning starts; that stream drives training. Neither may reach a validation or final-test seed.
    training_seeds = {seed * 100 + i for i in range(n_envs)} | {seed + i for i in range(n_envs)}
    if validation_seeds & test_seeds or (validation_seeds | test_seeds) & training_seeds:
        raise ValueError("Training, validation and final-test seeds must be disjoint")
    if cfg.experiment_id in ("E", "F") and cfg.selected_gaussian.nll_beta not in (0.0, 0.5):
        raise ValueError("Select C or D using validation first: selected_gaussian.nll_beta=0 or 0.5")
    for key in (
        "training.num_threads", "training.num_interop_threads", "training.n_envs", "training.total_timesteps", "evaluation.frequency", "checkpoint.frequency",
        "cppo.n_steps", "cppo.n_epochs", "cppo.n_minibatches", "cppo.context_dim",
        "cppo.supervised.capacity", "cppo.supervised.episode_batch_size", "cppo.supervised.gradient_steps",
    ):
        _positive_int(cfg, key)
    if str(cfg.training.vec_env).lower() not in ("dummy", "subproc"):
        raise ValueError("training.vec_env must be 'dummy' or 'subproc'")
    cppo = cfg.cppo
    if cppo.policy != "MlpPolicy":
        raise ValueError("cppo.policy must be MlpPolicy (ContextualActorCriticPolicy)")
    if str(cppo.activation_fn).lower() not in ACTIVATIONS:
        raise ValueError(f"cppo.activation_fn must be one of {tuple(ACTIVATIONS)}")
    n_envs, n_minibatches = int(cfg.training.n_envs), int(cppo.n_minibatches)
    if n_minibatches > n_envs or n_envs % n_minibatches:
        raise ValueError(
            f"cppo.n_minibatches ({n_minibatches}) must divide training.n_envs ({n_envs}): "
            "each minibatch holds complete environment streams"
        )
    if cppo.normalize_advantage and int(cppo.n_steps) * n_envs // n_minibatches < 2:
        raise ValueError("Advantage normalization requires at least 2 transitions per minibatch")
    if cppo.bptt_len is not None:
        _positive_int(cfg, "cppo.bptt_len")
    weight = cppo.encoder_actor_loss_weight
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("cppo.encoder_actor_loss_weight must be a finite number >= 0")
    scale = cppo.get("reward_scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("cppo.reward_scale must be a finite number > 0")
    noise = cfg.env.get("battery_noise", 0.0)
    if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise < 0:
        raise ValueError("env.battery_noise must be a finite number >= 0")
    if noise > 0 and cfg.env.observation_mode != "observed":
        raise ValueError("env.battery_noise adds noise to the visible battery: it needs env.observation_mode=observed")
    profile = cfg.env.get("battery_noise_flight")
    if profile not in FLIGHT_PROFILES:
        raise ValueError(f"env.battery_noise_flight must be one of {FLIGHT_PROFILES}")
    if profile is not None:
        step = cfg.env.get("battery_noise_flight_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1 or noise <= 0:
            raise ValueError("env.battery_noise_flight needs env.battery_noise > 0 and a positive env.battery_noise_flight_step")
    until = cfg.env.get("battery_noise_until")
    if until is not None:
        if isinstance(until, bool) or not isinstance(until, int) or not 0 < until < int(cfg.training.total_timesteps):
            raise ValueError("env.battery_noise_until must be null or a step inside the training budget")
        if noise <= 0 or str(cfg.training.vec_env).lower() != "dummy":
            raise ValueError("env.battery_noise_until switches battery noise off: it needs env.battery_noise > 0 and "
                             "training.vec_env=dummy")
    if cppo.context_activation is not None and cppo.context_activation not in CONTEXT_ACTIVATIONS:
        raise ValueError(f"cppo.context_activation must be null or one of {sorted(CONTEXT_ACTIVATIONS)}")
    if cppo.context_head not in CONTEXT_HEADS:
        raise ValueError(f"cppo.context_head must be one of {CONTEXT_HEADS}")
    if cppo.context_variance_input not in CONTEXT_VARIANCE_INPUTS:
        raise ValueError(f"cppo.context_variance_input must be one of {CONTEXT_VARIANCE_INPUTS}")
    keys = cppo.context_info_keys
    if keys is not None and (len(keys) != 2 or not all(isinstance(key, str) for key in keys)):
        raise ValueError("cppo.context_info_keys must be null or [reset_key, step_key], e.g. [b0, b]")
    fraction = cppo.supervised.validation_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 <= fraction < 1.0:
        raise ValueError("cppo.supervised.validation_fraction must be in [0, 1)")
    beta = cppo.supervised.nll_beta
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not 0.0 <= beta <= 1.0:
        raise ValueError("cppo.supervised.nll_beta must be in [0, 1]")
    seed = cppo.supervised.seed
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("cppo.supervised.seed must be null or an integer")
    if cppo.encoder_objective not in ContextualPPO.ENCODER_OBJECTIVES:
        raise ValueError(f"cppo.encoder_objective must be one of {ContextualPPO.ENCODER_OBJECTIVES}")
    context_source = context_source_from(cfg)
    initial_model_from(cfg)   # a late switch must name the model it continues
    if frozen_context_source(cfg) != "none":
        raise ValueError("context.source (context_wrapper.py) is for plain PPO; Contextual_PPO has its own encoder "
                         "(cppo.context_source)")
    if context_source not in CONTEXT_SOURCES:
        raise ValueError(f"cppo.context_source must be one of {CONTEXT_SOURCES}")
    if context_source == "oracle" and cfg.env.observation_mode != "observed":
        raise ValueError(
            "cppo.context_source=oracle requires env.observation_mode=observed: "
            "Step 0 keeps noisy speed; full mode supplies exact speed"
        )
    if context_source == "none" and cfg.env.observation_mode != "hidden":
        raise ValueError(
            "cppo.context_source=none requires env.observation_mode=hidden: the memoryless control sees neither the "
            "battery nor exact speed, like experiment A"
        )
    if context_source != "encoder" and (cppo.encoder_objective != "rl" or cppo.velocity_head.enabled
                                        or cppo.context_head != "point" or int(cppo.context_dim) != 1):
        raise ValueError(
            f"cppo.context_source={context_source} has no encoder to train: it needs cppo.encoder_objective=rl, "
            "cppo.velocity_head.enabled=false, cppo.context_head=point and cppo.context_dim=1"
        )
    if cppo.encoder_objective == "supervised" and (keys is None or int(cppo.context_dim) != 1):
        raise ValueError(
            "cppo.encoder_objective=supervised fits z_t to the charge b_t: it needs cppo.context_info_keys=[b0,b] "
            "and cppo.context_dim=1"
        )
    velocity = cppo.velocity_head
    if not isinstance(velocity.enabled, bool):
        raise ValueError("cppo.velocity_head.enabled must be true or false")
    weight = velocity.weight
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0.0 <= weight <= 1.0:
        raise ValueError("cppo.velocity_head.weight must be in [0, 1]")
    if velocity.enabled and cppo.encoder_objective != "supervised":
        raise ValueError(
            "cppo.velocity_head is trained in the supervised encoder steps: it needs cppo.encoder_objective=supervised"
        )
    if cppo.context_head == "gaussian" and cppo.encoder_objective != "supervised":
        raise ValueError(
            "cppo.context_head=gaussian learns the variance from the likelihood of b_t: it needs "
            "cppo.encoder_objective=supervised"
        )
    arch = cppo.get("encoder_arch", "xlstm")
    if arch not in ENCODER_ARCHS:
        raise ValueError(f"cppo.encoder_arch must be one of {ENCODER_ARCHS}")
    if arch == "gru":
        for key in ("hidden_dim", "num_layers"):
            value = cppo.gru.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"cppo.gru.{key} must be an integer >= 1")
        if context_source != "encoder":
            raise ValueError("cppo.encoder_arch=gru is a learned encoder: it needs cppo.context_source=encoder")
    total = int(cfg.training.total_timesteps)
    warmup = cppo.get("policy_warmup_steps", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < total:
        raise ValueError("cppo.policy_warmup_steps must be an integer in [0, training.total_timesteps)")
    if warmup > 0 and cppo.encoder_objective != "supervised":
        raise ValueError("cppo.policy_warmup_steps trains only the encoder at first: it needs cppo.encoder_objective=supervised")
    warmup_steps = cppo.supervised.get("warmup_gradient_steps")
    if warmup_steps is not None and (isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1):
        raise ValueError("cppo.supervised.warmup_gradient_steps must be null or an integer >= 1")
    curiosity = cppo.get("curiosity") or {}
    weight = curiosity.get("weight", 0.0)
    until = curiosity.get("until")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError("cppo.curiosity.weight must be a number >= 0")
    if until is not None and (isinstance(until, bool) or not isinstance(until, int) or not 0 < until <= total):
        raise ValueError("cppo.curiosity.until must be null or a step inside the training budget")
    if weight > 0 and (cppo.context_head != "gaussian" or cppo.encoder_objective != "supervised" or context_source != "encoder"):
        raise ValueError("cppo.curiosity rewards the drop in the Gaussian head's entropy: it needs cppo.context_head=gaussian, "
                         "a learned encoder and cppo.encoder_objective=supervised")


def check_episode_alignment(cfg: DictConfig, episode_length: int) -> None:
    """Episode-aligned rollouts need every episode that starts with a rollout to end within it."""
    if cfg.cppo.reset_envs_each_rollout and int(cfg.cppo.n_steps) < episode_length:
        raise ValueError(
            f"cppo.reset_envs_each_rollout needs cppo.n_steps >= the episode length ({episode_length}), "
            f"got {cfg.cppo.n_steps}: set cppo.n_steps={episode_length}"
        )


def build_model(cfg: DictConfig, env: VecEnv, tensorboard_log: str | None = None) -> ContextualPPO:
    """ContextualPPO with the PPO and encoder settings of cfg.cppo."""
    cppo = cfg.cppo
    net_arch = list(cppo.net_arch)
    keys = cppo.context_info_keys
    velocity = cppo.velocity_head
    return ContextualPPO(
        policy=CONTEXT_POLICIES[context_source_from(cfg)] or cppo.policy,
        env=env,
        learning_rate=cppo.learning_rate,
        n_steps=int(cppo.n_steps),
        n_minibatches=int(cppo.n_minibatches),
        n_epochs=int(cppo.n_epochs),
        gamma=cppo.gamma,
        gae_lambda=cppo.gae_lambda,
        clip_range=cppo.clip_range,
        clip_range_vf=cppo.clip_range_vf,
        normalize_advantage=bool(cppo.normalize_advantage),
        ent_coef=cppo.ent_coef,
        vf_coef=cppo.vf_coef,
        max_grad_norm=cppo.max_grad_norm,
        encoder_actor_loss_weight=float(cppo.encoder_actor_loss_weight),
        encoder_max_grad_norm=cppo.encoder_max_grad_norm,
        encoder_objective=str(cppo.encoder_objective),
        nll_beta=float(cppo.supervised.nll_beta),
        # Velocity head: its targets come from TrueSpeedInfo (make_train_env), the label of o_t is vx_t / v_ref
        auxiliary_info_keys=[TrueSpeedInfo.KEY, TrueSpeedInfo.KEY] if velocity.enabled else None,
        auxiliary_loss_weight=float(velocity.weight),
        # Episode buffer of the supervised encoder (unused with encoder_objective: rl)
        episode_buffer_capacity=int(cppo.supervised.capacity),
        episode_batch_size=int(cppo.supervised.episode_batch_size),
        supervised_gradient_steps=int(cppo.supervised.gradient_steps),
        validation_fraction=float(cppo.supervised.validation_fraction),
        episode_buffer_seed=None if cppo.supervised.seed is None else int(cppo.supervised.seed),
        # Joint-training options; saved configs from before them have neither
        policy_warmup_steps=int(cppo.get("policy_warmup_steps", 0)),
        supervised_warmup_gradient_steps=cppo.supervised.get("warmup_gradient_steps"),
        curiosity_weight=float((cppo.get("curiosity") or {}).get("weight", 0.0)),
        curiosity_until=(cppo.get("curiosity") or {}).get("until"),
        bptt_len=None if cppo.bptt_len is None else int(cppo.bptt_len),
        reset_envs_each_rollout=bool(cppo.reset_envs_each_rollout),
        context_info_keys=None if keys is None else list(OmegaConf.to_container(keys, resolve=True)),
        target_kl=cppo.target_kl,
        tensorboard_log=tensorboard_log,
        policy_kwargs={
            "net_arch": {"pi": net_arch, "vf": net_arch},
            "activation_fn": ACTIVATIONS[str(cppo.activation_fn).lower()],
            "ortho_init": bool(cppo.ortho_init),
            "log_std_init": float(cppo.log_std_init),
            "context_dim": int(cppo.context_dim),
            "context_net_arch": list(cppo.context_net_arch),
            "context_activation": cppo.context_activation,
            "context_head": str(cppo.context_head),
            "context_variance_input": str(cppo.context_variance_input),
            "auxiliary_dim": 1 if velocity.enabled else None,
            "auxiliary_net_arch": list(velocity.net_arch) if velocity.enabled else None,
            "xlstm_config": OmegaConf.to_container(cppo.xlstm, resolve=True),
            **dict(zip(("encoder_arch", "gru_hidden_dim", "gru_num_layers"), encoder_arch_from(cfg))),
        },
        seed=int(cfg.training.seed),
        device=cfg.training.device,
        verbose=int(cppo.verbose),
    )


def load_initial_model(cfg: DictConfig, env: VecEnv, path: Path, tensorboard_log: str | None = None) -> ContextualPPO:
    """
    Step 1's late switch: the saved model of a finished run (its weights and optimizer state), continued on this run's
    environments, e.g. with noise on the battery. The saved run (its config.yaml next to the model) must have used this
    run's settings, apart from the battery noise, step budget, threads and naming; steps and evaluations then count
    from the switch.
    """
    source = path.parent / "config.yaml"
    if not path.is_file() or not source.is_file():
        raise FileNotFoundError(f"training.init_model needs a saved model and its run's config.yaml: {path}")
    differences = switch_source_mismatches(cfg, OmegaConf.load(source), "cppo")
    if differences:
        raise ValueError(f"training.init_model {path} was trained with other settings: {', '.join(differences)}")
    model = ContextualPPO.load(path, env=env, device=cfg.training.device)
    if model.seed != int(cfg.training.seed):
        raise ValueError(f"training.init_model {path} was trained with seed {model.seed}, not {cfg.training.seed}")
    model.tensorboard_log = tensorboard_log
    return model


def configure_cpu_threads(cfg: DictConfig) -> None:
    """Each launcher child has its own PyTorch pools; BLAS limits are set before importing it."""
    torch.set_num_threads(int(cfg.training.num_threads))
    interop = int(cfg.training.num_interop_threads)
    if torch.get_num_interop_threads() != interop:
        torch.set_num_interop_threads(interop)


def plot_batteries_from(cfg: DictConfig) -> list[float]:
    """Starting charges of the episode plots; configs saved before evaluation.plots have none."""
    plots = cfg.evaluation.get("plots")
    return [] if plots is None else list(plots.initial_battery)


def run_schedule(model, cfg: DictConfig, reset_grid: DictConfig, plot_batteries=()):
    """Evaluate the scheduled episodes: the result (JSON-ready) and the recorder of their flights."""
    schedule = list(iter_eval_resets(reset_grid))
    # The battery the policy sees at its step: after an early-noise run's switch, the true battery
    env, scheduler, raw_env = make_eval_env(cfg, schedule, battery_noise_at(cfg, model.num_timesteps))
    try:
        scheduler.rewind()
        # SB3's callback exposes the ending episode's info, even after VecEnv has reset the environment.
        # These privileged diagnostics are used only for reports, never as model inputs.
        rewards, lengths, recorder = evaluate_flights(
            model, env, len(schedule), bool(cfg.evaluation.deterministic), raw_env.core, plot_batteries,
        )
    finally:
        env.close()
    if not np.isfinite(rewards).all():
        raise ValueError("Non-finite evaluation return")
    outcomes = [flight["outcome"] for flight in recorder.flights]
    if len(outcomes) != len(schedule):
        raise ValueError("Evaluation did not record exactly one outcome per scheduled episode")
    return {
        "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
        "episode_rewards": [float(x) for x in rewards], "episode_lengths": [int(x) for x in lengths],
        "episode_outcomes": outcomes,
        "completion_rate": outcomes.count("completed") / len(outcomes),
        "ground_crash_rate": outcomes.count("ground_crash") / len(outcomes),
        "battery_empty_rate": outcomes.count("battery_empty") / len(outcomes),
        "truncated_rate": outcomes.count("truncated") / len(outcomes),
        "outcome_rule": "Exclusive categories; battery_empty takes precedence over simultaneous ground contact.",
        # Return, outcomes and flying behaviour overall and per starting charge (flight_logging.summarize)
        "flight_metrics": recorder.metrics(),
        "reset_schedule": schedule, "checkpoint": "final_model.zip", "timesteps": model.num_timesteps,
    }, recorder


def evaluate_schedule(model, cfg: DictConfig, reset_grid: DictConfig) -> dict:
    return run_schedule(model, cfg, reset_grid)[0]


@hydra.main(version_base=None, config_path="config", config_name="cppo")
def main(cfg: DictConfig) -> None:
    validate_config(cfg)
    configure_cpu_threads(cfg)
    started = time.monotonic()
    schedule = list(iter_eval_resets(cfg.evaluation.reset_grid))
    prefix = cfg.wandb.run_name_prefix
    run_name = f"{prefix}{cfg.wandb.run_name or f'{cfg.env.observation_mode}_s{cfg.training.seed}'}"
    group = f"{prefix}{cfg.wandb.group or cfg.env.observation_mode}"
    tags = list(OmegaConf.to_container(cfg.wandb.tags, resolve=True)) if cfg.wandb.get("tags") else None
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        group=group,
        tags=tags,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        save_code=cfg.wandb.save_code,
        dir=str(PROJECT_ROOT),   # one wandb/ folder at the repository root, whatever cwd you start from
    )
    train_env = eval_env = None
    exit_code = 1
    try:
        # paths.output_dir (runs_cppo/) is resolved from the repository root, whatever directory you start from.
        output_dir = Path(cfg.paths.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        run_dir = Path(cfg.paths.run_dir) if cfg.paths.run_dir is not None else output_dir / run.id
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_dir, best_model_dir, eval_log_dir = (run_dir / name for name in ("checkpoints", "best_model", "eval"))
        for directory in (checkpoint_dir, best_model_dir, eval_log_dir):
            directory.mkdir()
        # Store the resolved configuration even if W&B is disabled.
        OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
        (run_dir / "eval_reset_schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")

        train_env = make_train_env(cfg)
        eval_env, scheduler, raw_eval_env = make_eval_env(cfg, schedule)
        core = raw_eval_env.core
        check_episode_alignment(cfg, core.T)
        initial_model = initial_model_from(cfg)
        if initial_model is None:
            model = build_model(cfg, train_env, str(run_dir / "tensorboard"))
        else:
            model = load_initial_model(cfg, train_env, initial_model, str(run_dir / "tensorboard"))

        # Say which physics and which learner code are running, so a wrong or ignored override is visible.
        n_envs, n_steps = int(cfg.training.n_envs), int(cfg.cppo.n_steps)
        groups = model.policy.parameter_groups()
        versions = {pkg: version(pkg) for pkg in ("stable-baselines3", "xlstm", "torch", "gymnasium", "hydra-core")}
        metadata = {
            "algorithm": "ContextualPPO",
            "experiment_id": cfg.experiment_id,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "effective_env": core._kw,
            "versions": versions,
            "environment_source": str(bp.__file__),
            "sb3_source": str(stable_baselines3.__file__),
            "contextual_ppo_source": str(Path(contextual_ppo.__file__).parent),
            "policy_parameters": sum(p.numel() for p in model.policy.parameters()),
            "parameters_by_group": {name: sum(p.numel() for p in params) for name, params in groups.items()},
            "minibatch_transitions": n_steps * n_envs // int(cfg.cppo.n_minibatches),
            "init_model": None if initial_model is None else str(initial_model),
            "run_name": run_name,
            "group": group,
            "tags": tags,
        }
        run.config.update(metadata, allow_val_change=True)
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"BatteryPlane from {bp.__file__}")
        print(f"ContextualPPO from {metadata['contextual_ppo_source']} | "
              + " ".join(f"{pkg} {v}" for pkg, v in versions.items()))
        print(
            f"  mode={core.observation_mode} sigma={core.sigma:g} E={core.E:g} b0_range={core.b0_range} "
            f"dense_w={core.dense_w:g} crash_penalty={core.C:g} T={core.T}"
        )
        print(
            f"  {n_envs} envs x {n_steps} steps = {n_envs * n_steps} transitions per update, "
            f"{cfg.cppo.n_minibatches} minibatches of {metadata['minibatch_transitions']} x {cfg.cppo.n_epochs} epochs"
            f" | gamma={cfg.cppo.gamma} reward_scale={reward_scale_from(cfg)} battery_noise={battery_noise_from(cfg)}"
            f" until={cfg.env.get('battery_noise_until')} flight={cfg.env.get('battery_noise_flight')}"
            f" | bptt_len={cfg.cppo.bptt_len} reset_envs_each_rollout={cfg.cppo.reset_envs_each_rollout}"
            f" | context_source={context_source_from(cfg)}"
            f" encoder_objective={cfg.cppo.encoder_objective} encoder_actor_loss_weight={cfg.cppo.encoder_actor_loss_weight}"
            f" context_head={cfg.cppo.context_head}"
            + (f" (variance input {cfg.cppo.context_variance_input}, nll_beta {cfg.cppo.supervised.nll_beta})"
               if cfg.cppo.context_head == "gaussian" else "")
        )
        if model.episode_buffer is not None:
            supervised = cfg.cppo.supervised
            print(
                f"  supervised encoder: episode buffer of {supervised.capacity} steps, {supervised.gradient_steps} steps "
                f"of {supervised.episode_batch_size} episodes per update, validation {supervised.validation_fraction:.0%}"
            )
            if cfg.cppo.velocity_head.enabled:
                print(f"  velocity head: L_context + {cfg.cppo.velocity_head.weight} x L_velocity (vx / v_ref)")
        print(f"  parameters={metadata['parameters_by_group']} | eval episodes={len(schedule)}\n  artifacts: {run_dir}")
        if initial_model is not None:
            print(f"  continuing the saved model {initial_model} (late switch), steps counted from the switch")

        # Early-noise runs: the battery noise goes off at env.battery_noise_until, first, before that step's evaluation
        switch = [] if cfg.env.get("battery_noise_until") is None else [BatteryNoiseSwitch(cfg.env.battery_noise_until, eval_env)]
        callbacks = CallbackList([
            *switch,
            # SB3 evaluate_policy passes both state and episode_start to predict(): every evaluation episode
            # starts with fresh memory, and the callback rewinds the evaluation grid.
            RenderingEvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(eval_log_dir),
                # SB3 counts callback calls; one call covers n_envs environment steps.
                eval_freq=max(int(cfg.evaluation.frequency) // n_envs, 1),
                n_eval_episodes=len(schedule),
                deterministic=cfg.evaluation.deterministic,
                render=False,  # the env draws itself in step(); don't double-render
                render_env=raw_eval_env if cfg.evaluation.render else None,
                render_fps=cfg.evaluation.render_fps,
                reset_scheduler=scheduler,
                flight_core=core,
                plot_every=int(cfg.evaluation.plots.frequency),
                plot_batteries=plot_batteries_from(cfg),
            ),
            # rollout/* flight metrics of the training envs and train/grad_norm* of every update
            RolloutFlightLogger(core),
            GradNormLogger(),
            CheckpointCallback(
                save_freq=max(int(cfg.checkpoint.frequency) // n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix=cfg.checkpoint.name_prefix,
            ),
            WandbCallback(
                gradient_save_freq=cfg.wandb.gradient_save_frequency,
                verbose=cfg.wandb.callback_verbose,
            ),
        ])
        model.learn(
            total_timesteps=int(cfg.training.total_timesteps),
            callback=callbacks,
            tb_log_name=cfg.wandb.tensorboard_log_name,
            # Steps and evaluations count from 0, also when continuing a saved model (late switch)
            reset_num_timesteps=True,
        )
        model.save(run_dir / "final_model")
        # A fixed final-checkpoint validation score selects C versus D across all training seeds.
        # The separate final_test grid is never evaluated during training or selection.
        validation, recorder = run_schedule(model, cfg, cfg.evaluation.reset_grid, plot_batteries_from(cfg))
        (run_dir / "final_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        run.summary["validation/final_mean_reward"] = validation["mean_reward"]
        for metric in ("completion_rate", "ground_crash_rate", "battery_empty_rate", "truncated_rate"):
            run.summary[f"validation/final_{metric}"] = validation[metric]
        log_final_flights(run, recorder, model.num_timesteps)
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["actual_timesteps"] = model.num_timesteps
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        run.summary["actual_timesteps"] = model.num_timesteps
        print(f"Saved final model to {run_dir / 'final_model.zip'} ({model.num_timesteps} steps)")
        exit_code = 0
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
