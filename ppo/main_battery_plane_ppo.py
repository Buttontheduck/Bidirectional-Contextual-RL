"""Plain PPO training for BatteryPlane: the p01-p04 baselines and runs 0a/0b of step 0 (ENCODER_PLAN.md).

Copied from the Env_BatteryPlane project so that step 0 runs from this repository. It imports battery_plane.py
and flight_logging.py from the repository root (one directory up), logs the same flight metrics as Contextual_PPO,
and writes to runs_ppo/<wandb run id>/, including its resolved config.yaml (report_step0.py reads it).

    python ppo/main_battery_plane_ppo.py                                   # base config
    python ppo/main_battery_plane_ppo.py +experiment=p02_hidden_baseline   # one experiment
    python ppo/main_battery_plane_ppo.py -m +experiment=S0a_observed_bs256 training.seed=0,1,2   # step 0, run 0a

Differences from the SAC script, all of them forced by PPO being on-policy:
  * training.n_envs parallel copies of the environment (SAC used one). A rollout is
    n_steps x n_envs transitions, and every count below is in ENVIRONMENT steps, so
    evaluation.frequency / checkpoint.frequency mean the same thing in both scripts.
  * PPO needs several times more environment steps than SAC for the same policy, but
    each step is far cheaper (no gradient step per step), so the default budget is
    larger while the wall-clock stays shorter.
  * The evaluation env is a single environment with the fixed reset schedule, exactly
    as in the SAC script, so eval numbers are directly comparable between algorithms.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

# The repository root (one level up) holds battery_plane.py, the single source of truth for the environment.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import battery_plane as bp  # noqa: E402
from battery_plane import iter_eval_resets  # noqa: E402
# Flight diagnostics shared with Contextual_PPO, so both log the same validation/*, rollout/*, train/grad_norm*
# and final_validation/* metrics
from flight_logging import (  # noqa: E402
    FlightLoggingEvalCallback, GradNormLogger, RolloutFlightLogger, evaluate_flights, log_final_flights,
)

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}


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


def make_train_env(cfg: DictConfig, env_wrapper: Callable[[gym.Env], gym.Env] | None = None):
    """n_envs copies of BatteryPlaneEnv, each with its own random stream; env_wrapper (if any) wraps each copy
    inside its Monitor (Contextual_PPO's velocity head adds the true speed to the infos this way)."""
    kwargs = env_kwargs_from(cfg)
    observation_mode = cfg.env.observation_mode
    base_seed = int(cfg.training.seed)

    def factory(index: int):
        def _init() -> gym.Env:
            env = bp.BatteryPlaneEnv(observation_mode=observation_mode, **kwargs)
            if env_wrapper is not None:
                env = env_wrapper(env)
            env = Monitor(env)
            # Only until learning starts: SB3 then re-seeds the VecEnv (env.seed(training.seed)), so copy i
            # starts training from reset seed training.seed + i. validate_ppo keeps both from the evaluation seeds.
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


def make_eval_env(cfg: DictConfig, reset_schedule: Sequence[Mapping[str, Any]]):
    """One environment that replays the fixed evaluation grid, as in the SAC script."""
    raw_env = bp.BatteryPlaneEnv(observation_mode=cfg.env.observation_mode, **env_kwargs_from(cfg))
    scheduler = ScheduledEvalResets(raw_env, reset_schedule)
    env = Monitor(scheduler)
    env.action_space.seed(int(cfg.training.seed) + 1)
    env.reset()
    scheduler.rewind()   # the line above consumed the first entry; every evaluation rewinds anyway
    return DummyVecEnv([lambda: env]), scheduler, raw_env


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


def validate_ppo(cfg: DictConfig) -> None:
    """Catch the PPO-specific mistakes before a run starts."""
    n_envs, n_steps, batch_size = int(cfg.training.n_envs), int(cfg.ppo.n_steps), int(cfg.ppo.batch_size)
    if n_envs < 1:
        raise ValueError("training.n_envs must be >= 1")
    if str(cfg.training.vec_env).lower() not in ("dummy", "subproc"):
        raise ValueError("training.vec_env must be 'dummy' or 'subproc'")
    rollout = n_steps * n_envs
    if rollout < batch_size:
        raise ValueError(
            f"ppo.batch_size ({batch_size}) is larger than one rollout ({n_steps} x {n_envs} = {rollout})"
        )
    if rollout % batch_size:
        raise ValueError(
            f"one rollout ({rollout} transitions) must divide evenly into ppo.batch_size ({batch_size}); "
            "PPO would silently drop the remainder"
        )
    if str(cfg.ppo.activation_fn).lower() not in ACTIVATIONS:
        raise ValueError(f"ppo.activation_fn must be one of {tuple(ACTIVATIONS)}")
    # Env copy i is first reset with seed * 100 + i (make_train_env), then SB3 re-seeds it with seed + i when
    # learning starts; that stream drives training. Neither may reach an evaluation seed.
    seed = int(cfg.training.seed)
    training_seeds = {seed * 100 + i for i in range(n_envs)} | {seed + i for i in range(n_envs)}
    if set(cfg.evaluation.reset_grid.eval_seeds) & training_seeds:
        raise ValueError("Training and evaluation seeds must be disjoint")


@hydra.main(version_base=None, config_path="configs", config_name="battery_plane_ppo")
def main(cfg: DictConfig) -> None:
    validate_eval_reset_grid(cfg.evaluation.reset_grid)
    validate_ppo(cfg)
    eval_reset_schedule = list(iter_eval_resets(cfg.evaluation.reset_grid))
    n_envs = int(cfg.training.n_envs)

    tags = list(OmegaConf.to_container(cfg.wandb.tags, resolve=True)) if cfg.wandb.get("tags") else None
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        group=cfg.wandb.group,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        save_code=cfg.wandb.save_code,
        dir=str(PROJECT_ROOT),   # one wandb/ folder at the project root, whatever cwd you start from
    )

    # runs_ppo/ lives next to the SAC runs/ folder, whatever directory you start from.
    output_dir = Path(cfg.paths.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    run_dir = output_dir / run.id
    checkpoint_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    eval_log_dir = run_dir / "eval"
    for directory in (checkpoint_dir, best_model_dir, eval_log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # The resolved configuration of this run, even if W&B is disabled (report_step0.py reads it)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)

    train_env = make_train_env(cfg)
    eval_env, eval_reset_scheduler, raw_eval_env = make_eval_env(cfg, eval_reset_schedule)

    # Say which physics is running, so a wrong or ignored override is visible in the log.
    core = raw_eval_env.core
    print(f"BatteryPlane from {bp.__file__}")
    print(
        f"  PPO | mode={core.observation_mode} sigma={core.sigma:g} E={core.E:g} b0_range={core.b0_range} "
        f"dense_w={core.dense_w:g} crash_penalty={core.C:g}"
    )
    print(
        f"  {n_envs} train envs x {cfg.ppo.n_steps} steps = {n_envs * int(cfg.ppo.n_steps)} transitions per update"
        f" | {int(cfg.training.total_timesteps) // (n_envs * int(cfg.ppo.n_steps))} updates"
        f" | eval episodes: {len(eval_reset_schedule)}"
    )
    run.config.update({"effective_env": core._kw, "algorithm": "PPO"}, allow_val_change=True)

    net_arch = list(cfg.ppo.net_arch)
    policy_kwargs = {
        "net_arch": {"pi": net_arch, "vf": net_arch},
        "activation_fn": ACTIVATIONS[str(cfg.ppo.activation_fn).lower()],
        "ortho_init": bool(cfg.ppo.ortho_init),
        "log_std_init": float(cfg.ppo.log_std_init),
    }

    model = PPO(
        policy=cfg.ppo.policy,
        env=train_env,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=int(cfg.ppo.n_steps),
        batch_size=int(cfg.ppo.batch_size),
        n_epochs=int(cfg.ppo.n_epochs),
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_range=cfg.ppo.clip_range,
        clip_range_vf=cfg.ppo.clip_range_vf,
        normalize_advantage=bool(cfg.ppo.normalize_advantage),
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        target_kl=cfg.ppo.target_kl,
        tensorboard_log=str(run_dir / "tensorboard"),
        policy_kwargs=policy_kwargs,
        seed=int(cfg.training.seed),
        device=cfg.training.device,
        verbose=int(cfg.ppo.verbose),
    )

    # SB3 counts callback frequencies in CALLS, and one call covers n_envs environment
    # steps. Dividing keeps evaluation.frequency / checkpoint.frequency in env steps,
    # so the numbers mean exactly what they mean in the SAC config.
    eval_freq = max(int(cfg.evaluation.frequency) // n_envs, 1)
    save_freq = max(int(cfg.checkpoint.frequency) // n_envs, 1)
    render_target = raw_eval_env if cfg.evaluation.render else None

    callbacks = CallbackList(
        [
            RenderingEvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(eval_log_dir),
                eval_freq=eval_freq,
                n_eval_episodes=len(eval_reset_schedule),
                deterministic=cfg.evaluation.deterministic,
                render=False,  # the env draws itself in step(); don't double-render
                render_env=render_target,
                render_fps=cfg.evaluation.render_fps,
                reset_scheduler=eval_reset_scheduler,
                flight_core=core,
                plot_every=int(cfg.evaluation.plots.frequency),
                plot_batteries=list(cfg.evaluation.plots.initial_battery),
            ),
            # rollout/* flight metrics of the training envs and train/grad_norm* of every update
            RolloutFlightLogger(core),
            GradNormLogger(),
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(checkpoint_dir),
                name_prefix=cfg.checkpoint.name_prefix,
            ),
            WandbCallback(
                gradient_save_freq=cfg.wandb.gradient_save_frequency,
                verbose=cfg.wandb.callback_verbose,
            ),
        ]
    )

    exit_code = 1
    try:
        model.learn(
            total_timesteps=int(cfg.training.total_timesteps),
            callback=callbacks,
            tb_log_name=cfg.wandb.tensorboard_log_name,
        )
        final_model_path = run_dir / "final_model"
        model.save(final_model_path)
        print(f"Saved final model to {final_model_path}.zip")
        # Final-checkpoint validation on the same grid: final_validation/* and its episode plot, as in Contextual_PPO
        final_env, final_scheduler, final_raw_env = make_eval_env(cfg, eval_reset_schedule)
        try:
            final_scheduler.rewind()
            _, _, recorder = evaluate_flights(
                model, final_env, len(eval_reset_schedule), bool(cfg.evaluation.deterministic), final_raw_env.core,
                list(cfg.evaluation.plots.initial_battery),
            )
        finally:
            final_env.close()
        log_final_flights(run, recorder, model.num_timesteps)
        exit_code = 0
    finally:
        train_env.close()
        eval_env.close()
        # A failed run shows as failed in W&B, not finished
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
