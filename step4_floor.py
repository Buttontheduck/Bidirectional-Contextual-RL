"""Step 4 of ENCODER_PLAN.md: the floor, the best battery estimate possible on the pilot flights the encoders are scored on.

The ideal estimator (step1_ideal_estimator.py) reads what the encoder reads, hidden-mode observations and executed
actions, and knows the physics exactly: it replays a grid of starting batteries and speeds through BatteryPlaneVec.step
with the flight's actions and averages the candidates that match the observed speeds and climb rates. No estimator
with the same inputs and prior can beat it on average, so its error is the floor for the encoders of train_encoder.py.

It runs over the validation and test flights of a pilot mix (the split of train_encoder.py, by flight seed; the
training flights are used only for the clock-only estimate) and reports its error in train_encoder.py's format:
overall, at 1, 5 and 20 s into the flight, and per style, next to the clock-only estimate. It never reads the battery,
the starting battery or the speed except as the error's reference. --splits train validation test adds the training
flights: the teacher targets of train_encoder.py --target teacher (the clock-only error on them is in-sample).

Outputs (--output, e.g. runs_encoder/step4/floor_v2_main):
    floor.json      errors per split, the estimator's calibration (mean squared error / its own variance, 1 when its
                    uncertainty matches its errors), its largest climb-rate mismatch (small when the grid is fine
                    enough) and its settings
    estimates.npz   the estimate and std at every step of every flight, with seeds and flight offsets: per-flight
                    comparisons with the encoders, and teacher targets (ENCODER_PLAN.md, section 3)

    nohup nice -n 10 python step4_floor.py --output runs_encoder/step4/floor_v2_main --jobs 16 > floor.log 2>&1 &   # ~4 min
    nohup python step4_floor.py --splits train validation test --output runs_encoder/step4/teacher_v2_main --jobs 30 > teacher.log 2>&1 &

No randomness: the estimator is deterministic (its replay planes have a fixed seed and no speed noise), and the script
opens no environment seeds. It imports neither torch nor W&B.
"""
from __future__ import annotations

import os

# One thread per process: the estimator is elementwise numpy, many processes beat many threads
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

import train_encoder as T

SPLITS = ("train", "validation", "test")
DEFAULT_SPLITS = ("validation", "test")


def estimate_flights(flights: list[tuple[int, np.ndarray, np.ndarray]], grid: dict, env_kwargs: dict) -> list[dict]:
    """
    The ideal estimate along each flight: o_0 starts it, then every step adds the executed action a_{t-1} (the record's
    previous action) and o_t; the final observation does not rule out an empty battery (the plane may have just run out).

    :param flights: (seed, observations [K+1, 4], prev_actions [K+1, 2]) per flight, as pilots.py records them
    """
    from step1_ideal_estimator import IdealBatteryEstimator

    estimator = IdealBatteryEstimator(env_kwargs, **grid)
    out = []
    for seed, observations, prev_actions in flights:
        try:
            mean, std = estimator.start(observations[0])
            means, stds, mismatch = [mean], [std], 0.0
            for t in range(1, len(observations)):
                mean, std = estimator.update(prev_actions[t], observations[t], alive=t < len(observations) - 1)
                means.append(mean)
                stds.append(std)
                mismatch = max(mismatch, estimator.vy_mismatch)
        except RuntimeError as error:
            raise RuntimeError(f"Flight seed {seed}: {error}") from error
        out.append({"seed": seed, "means": np.array(means, np.float32), "stds": np.array(stds, np.float32),
                    "vy_mismatch": mismatch})
    return out


def run_estimator(flights: T.Flights, rows: np.ndarray, grid: dict, env_kwargs: dict, jobs: int, chunk: int) -> dict[int, dict]:
    """The ideal estimate of every flight in rows (by index), in parallel."""
    tasks = [[(int(flights.seeds[i]), flights.observations[i], flights.prev_actions[i]) for i in rows[start:start + chunk]]
             for start in range(0, len(rows), chunk)]
    results = {}
    # Spawned workers: they import only numpy and the environment, and never inherit the parent's data
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for done, part in enumerate(pool.map(estimate_flights, tasks, [grid] * len(tasks), [env_kwargs] * len(tasks)), 1):
            for flight in part:
                results[flight["seed"]] = flight
            if done % max(1, len(tasks) // 10) == 0:
                print(f"{len(results)} / {len(rows)} flights", flush=True)
    return results


def summarize(flights: T.Flights, rows: np.ndarray, estimates: dict[int, dict], steps_at: dict[str, int]) -> dict:
    """The ideal estimator's errors in train_encoder.py's format, its calibration and its largest climb-rate mismatch."""
    tally = T.ErrorTally(steps_at)
    z2, mismatch = [], 0.0
    for i in rows:
        flight = estimates[int(flights.seeds[i])]
        errors = flight["means"].astype(np.float64) - flights.batteries[i]
        tally.add(errors, flights.styles[i])
        z2.append((errors / np.maximum(flight["stds"], 1e-6)) ** 2)
        mismatch = max(mismatch, flight["vy_mismatch"])
    z2 = np.concatenate(z2)
    # The mean is sensitive to steps where the posterior has collapsed onto a few grid points; the median is not
    return {**tally.summary(), "calibration": float(np.mean(z2)), "calibration_median": float(np.median(z2)),
            "max_vy_mismatch": float(mismatch)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=T.DEFAULT_DATA, help="pilots.py output folder")
    parser.add_argument("--mix", default="main", help="A mix of pilots.MIXES")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=20, help="Flights per worker task")
    # Step 2's grid. The full one (201 x 601) gave the same estimates within 0.003 (mean 0.0002) on 160 flights,
    # 12 times slower: its arrays spill out of the CPU caches, so it runs at memory speed
    parser.add_argument("--b-points", type=int, default=101)
    parser.add_argument("--vx-points", type=int, default=301)
    parser.add_argument("--prune-below", type=float, default=30.0, help="Log weight below the best candidate to drop")
    parser.add_argument("--max-flights", type=int, default=None, help="Only some flights of each split (checks)")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(DEFAULT_SPLITS),
                        help="Flights to estimate; add train for teacher targets")
    args = parser.parse_args(argv)
    splits = [s for s in SPLITS if s in args.splits]
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"{args.output} is not empty: choose another --output")

    import battery_plane as bp

    started = time.time()
    flights, manifest = T.load(args.data, args.mix, args.max_flights)
    # The physics the flights were recorded with (train_encoder.py checks that the encoders' match)
    env_kwargs = dict(manifest["env_kwargs"])
    dt = bp.BatteryPlaneVec(n_envs=1, observation_mode="hidden", seed=0, **env_kwargs).dt
    steps_at = {f"{s}s": int(round(s / dt)) for s in T.SECONDS_AT}
    grid = {"b_points": args.b_points, "vx_points": args.vx_points, "prune_below": args.prune_below}
    clock = T.clock_estimate(flights, flights.indices("train"))

    rows = {split: flights.indices(split) for split in splits}
    estimates = run_estimator(flights, np.concatenate([rows[s] for s in splits]), grid, env_kwargs, args.jobs, args.chunk)
    result = {"data": str(args.data), "mix": args.mix, "splits": splits, "grid": grid, "env_kwargs": env_kwargs, "steps_at": steps_at,
              "split_rule": f"train_encoder.split_of (flight seed % {T.SPLIT_MODULUS})", "max_flights": args.max_flights,
              "data_manifest_sha256": T.sha256(args.data / "manifest.json"),
              "code_sha256": {name: T.sha256(T.PROJECT_ROOT / name)
                              for name in ("step4_floor.py", "step1_ideal_estimator.py", "train_encoder.py", "battery_plane.py")},
              "git": T.git_state(), "created": datetime.now().isoformat(timespec="seconds")}
    for split in splits:
        result[split] = summarize(flights, rows[split], estimates, steps_at)
        result[f"clock_{split}"] = T.evaluate_clock(clock, flights, rows[split], steps_at)
    result["minutes"] = (time.time() - started) / 60

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "floor.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    seeds = [int(flights.seeds[i]) for split in splits for i in rows[split]]
    lengths = [len(estimates[s]["means"]) for s in seeds]
    np.savez_compressed(args.output / "estimates.npz", seeds=np.array(seeds), flight_offsets=np.concatenate([[0], np.cumsum(lengths)]),
                        splits=np.array([T.split_of(s) for s in seeds]),
                        means=np.concatenate([estimates[s]["means"] for s in seeds]),
                        stds=np.concatenate([estimates[s]["stds"] for s in seeds]))
    for split in splits:
        r, c = result[split], result[f"clock_{split}"]
        print(f"{split}: floor rmse {r['rmse']:.4f} (at 1/5/20 s: " + " / ".join(f"{v:.4f}" for v in r["rmse_at"].values())
              + f"), clock only {c['rmse']:.4f}, calibration {r['calibration']:.2f}", flush=True)
    print(f"Done in {result['minutes']:.1f} min: {args.output}", flush=True)


if __name__ == "__main__":
    main()
