"""EASY_ENV_WARMUP.md: how well can the battery be known after a forced full-throttle start?

Simulates flights whose first steps are flown at full throttle by an autopilot (the proposed forced take-off of the
easy environment) from random starting states drawn as in training (battery U(0.2, 0.7), speed U(0, 30) m/s, altitude
U(30, 120) m, speed noise 5 m/s), and runs the ideal estimator (step1_ideal_estimator.IdealBatteryEstimator) on each:
the best estimate any encoder could make after k forced steps. Also reports the battery the forced start uses.
Simulation seeds only (default 5000-5099), away from the training (0-14), validation (1000-1002) and test (2000-2002)
seeds. Nothing is trained.

    python warmup_information.py                       # 100 flights, level and climbing, about 2 minutes
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

CHECK_STEPS = (0, 20, 40, 60, 100, 140, 200)


def forced_start(seed: int, lift_command: float, steps: int) -> tuple[list, list, list]:
    """True battery, ideal estimate and its uncertainty after each of `steps` full-throttle steps from a random start."""
    import battery_plane as bp
    from step1_ideal_estimator import IdealBatteryEstimator

    env = bp.BatteryPlaneEnv(observation_mode="hidden")
    obs, info = env.reset(seed=seed)
    observations, actions, batteries = [obs], [], [info["b0"]]
    for _ in range(steps):
        action = np.array([1.0, lift_command], np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        actions.append(action)
        if terminated or truncated:
            break
        observations.append(obs)
        batteries.append(info["b"])
    out = IdealBatteryEstimator().estimate(observations, actions)
    return batteries, out["estimates"], out["stds"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flights", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=20)
    args = parser.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.flights))
    if any(s <= 14 or 1000 <= s <= 1002 or 2000 <= s <= 2002 for s in seeds):
        parser.error("Use simulation seeds away from the training, validation and test seeds")
    for lift_command, label in ((0.0, "full throttle, level"), (0.2, "full throttle, climbing")):
        with ProcessPoolExecutor(args.jobs) as pool:
            flights = list(pool.map(forced_start, seeds, [lift_command] * len(seeds), [max(CHECK_STEPS)] * len(seeds)))
        print(f"== {label}, {len(flights)} random starts")
        for k in CHECK_STEPS:
            alive = [f for f in flights if len(f[0]) > k]
            errors = np.array([f[1][k] - f[0][k] for f in alive])
            stds = np.array([f[2][k] for f in alive])
            used = np.array([f[0][0] - f[0][k] for f in alive])
            print(f"  after {k:3d} steps ({k * 0.05:4.1f} s): best possible error {np.sqrt(np.mean(errors ** 2)):.3f}, "
                  f"its own uncertainty {np.sqrt(np.mean(stds ** 2)):.3f}, battery used {used.mean():.3f} "
                  f"({used.min():.3f}-{used.max():.3f}), flights alive {len(alive)}", flush=True)


if __name__ == "__main__":
    main()
