"""Step 1 of ENCODER_PLAN.md: controls that change only the battery value the policy sees.

NoisyBattery is the noisy oracle. In observed mode the observation ends with the true charge b_t; the wrapper replaces it
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

# Column of the true charge b_t in observed-mode observations (as in step0_oracle.BATTERY_COLUMN)
BATTERY_COLUMN = -1
# Keeps the noise stream apart from the environment's stream for the same seed
NOISE_STREAM = 0x5EEDB477


class NoisyBattery(gym.ObservationWrapper):
    """
    The observed battery plus Gaussian noise: ``b_t + sigma * N(0, 1)``, fresh at every step and not clipped.

    :param env: A BatteryPlaneEnv in observed mode (the wrapper must see the seeded resets, so it goes inside any
        reset scheduler)
    :param sigma: Standard deviation of the noise, in units of a full battery (> 0)
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
