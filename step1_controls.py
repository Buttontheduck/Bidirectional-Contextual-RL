"""Step 1 of ENCODER_PLAN.md: controls that change only the battery value the policy sees.

NoisyBattery is the noisy oracle (experiment 1 of step 1 flies step 0 policies with it, experiment 2 continues
training them with it: switch_source_mismatches checks such a continuation). BatteryNoiseSwitch turns the noise off
during training (early-noise runs: noise first, then the true battery, like an encoder that becomes accurate).

NoisyBattery: in observed mode the observation ends with the true charge b_t; the wrapper replaces it
with b_t + noise, Gaussian noise of standard deviation `sigma` drawn fresh at every step (reset and step observations,
including the final observation of an episode). The actor and critic of either learner then read
[y, vx + speed noise, vy, tau, b_t + noise] and never the true b_t: plain PPO reads the observation, and CPPO's oracle
path (step0_oracle.TrueBatteryEncoder) reads its last column. The env infos keep the true b for logging and labels.

The noise has its own random stream, so the environment's stream (start states, speed noise) is untouched: with the same
seed, a run sees the flights of the noiseless run until the actions differ. A seeded reset restarts the noise stream from
that seed, so every validation flight (seeded by the schedule) gets the same noise in every evaluation, and training is
reproducible from the reset seeds SB3 sets when learning starts.

Project glue, like the training scripts; both import it from the repository root.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Column of the true charge b_t in observed-mode observations (as in step0_oracle.BATTERY_COLUMN)
BATTERY_COLUMN = -1
# Keeps the noise stream apart from the environment's stream for the same seed
NOISE_STREAM = 0x5EEDB477


class NoisyBattery(gym.ObservationWrapper):
    """
    The observed battery plus Gaussian noise: ``b_t + sigma * N(0, 1)``, fresh at every step and not clipped.

    :param env: A BatteryPlaneEnv in observed mode (the wrapper must see the seeded resets, so it goes inside any
        reset scheduler)
    :param sigma: Standard deviation of the noise, in units of a full battery (> 0; BatteryNoiseSwitch may set it to 0
        later, which gives the exact battery)
    """

    def __init__(self, env: gym.Env, sigma: float) -> None:
        super().__init__(env)
        if not (np.isfinite(sigma) and sigma > 0):
            raise ValueError(f"sigma must be finite and positive, got {sigma!r}")
        if getattr(env.unwrapped, "observation_mode", None) != "observed":
            raise ValueError("NoisyBattery needs an observed-mode BatteryPlaneEnv, whose observation ends with the battery")
        self.sigma = float(sigma)
        self.rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng([seed, NOISE_STREAM])
        return super().reset(seed=seed, options=options)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        noisy = np.array(observation, dtype=np.float32, copy=True)
        noisy[BATTERY_COLUMN] += np.float32(self.sigma * self.rng.standard_normal())
        return noisy


def switch_source_mismatches(cfg, source_cfg, learner: str) -> list[str]:
    """
    Late switch (step 1, experiment 2): the parts of a run's config (cfg) that differ from the step 0 run whose saved
    model it continues (source_cfg), other than the battery noise, the step budget, the model to start from, threads,
    how often it evaluates and naming. Empty when the continuation is valid. The source must have trained with the
    true battery.

    :param learner: The learner's config section, "cppo" or "ppo"
    """
    from omegaconf import OmegaConf

    def plain(node):
        return OmegaConf.to_container(node, resolve=True)

    differences = []
    if float(source_cfg.env.get("battery_noise", 0.0)) != 0.0:
        differences.append("the source trained with battery noise")
    env, source_env = plain(cfg.env), plain(source_cfg.env)
    for settings in (env, source_env):
        settings.pop("battery_noise", None)
    if env != source_env:
        differences.append("env")
    if learner not in source_cfg or plain(cfg[learner]) != plain(source_cfg[learner]):
        differences.append(learner)
    training, source_training = plain(cfg.training), plain(source_cfg.training)
    for settings in (training, source_training):
        for key in ("total_timesteps", "init_model", "num_threads", "num_interop_threads"):
            settings.pop(key, None)
    if training != source_training:
        differences.append("training")
    # The same validation flights (how often they are flown does not change the training)
    for key in ("reset_grid", "deterministic"):
        if plain(cfg.evaluation)[key] != plain(source_cfg.evaluation)[key]:
            differences.append(f"evaluation.{key}")
    return differences


def battery_noise_at(cfg, step: int) -> float:
    """
    Noise std on the visible battery at a training step: env.battery_noise, switched off from step
    env.battery_noise_until on when that is set (step 1's early-noise runs: noise first, then the true battery).
    """
    until = cfg.env.get("battery_noise_until")
    if until is not None and step >= until:
        return 0.0
    return float(cfg.env.get("battery_noise", 0.0))


def noisy_batteries(vec_env) -> list[NoisyBattery]:
    """The NoisyBattery wrappers inside the environments of a DummyVecEnv (one per environment that has one)."""
    found = []
    for env in getattr(vec_env, "envs", ()):
        while isinstance(env, gym.Wrapper):
            if isinstance(env, NoisyBattery):
                found.append(env)
                break
            env = env.env
    return found


class BatteryNoiseSwitch(BaseCallback):
    """
    Step 1's early-noise runs: the policy sees the battery plus noise until `switch_step` (training steps), then the
    true battery, in every training environment and in the validation environment. The switch happens at the first
    step at or after `switch_step`, before any evaluation of that step (the callback goes first in the list). Logs
    train/battery_noise with every rollout: the noise std averaged over the rollout's steps (a rollout that switches
    partway shows a value between the two).

    :param switch_step: Training step from which the battery is exact
    :param eval_env: The validation VecEnv (a DummyVecEnv with a NoisyBattery)
    """

    def __init__(self, switch_step: int, eval_env) -> None:
        super().__init__()
        self.switch_step = int(switch_step)
        self.eval_env = eval_env
        self.switched = False
        self.wrappers: list[NoisyBattery] = []
        self._noise_sum, self._steps = 0.0, 0

    def _on_training_start(self) -> None:
        self.wrappers = noisy_batteries(self.training_env) + noisy_batteries(self.eval_env)
        if len(self.wrappers) != self.training_env.num_envs + 1:
            raise RuntimeError("BatteryNoiseSwitch needs a NoisyBattery in every training environment and the validation "
                               "environment, inside DummyVecEnvs")

    def _on_step(self) -> bool:
        # The noise of the observations this step returned, before a switch at this step
        self._noise_sum += self.wrappers[0].sigma
        self._steps += 1
        if not self.switched and self.num_timesteps >= self.switch_step:
            for wrapper in self.wrappers:
                wrapper.sigma = 0.0
            self.switched = True
        return True

    def _on_rollout_end(self) -> None:
        if self._steps:
            self.logger.record("train/battery_noise", self._noise_sum / self._steps)
        self._noise_sum, self._steps = 0.0, 0
