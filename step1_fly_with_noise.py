"""Step 1 of ENCODER_PLAN.md, experiment 1: fly with a noisy battery, no more learning.

Finished step 0 policies (trained with the true battery) fly the validation flights while they see the battery plus
Gaussian noise (step1_controls.NoisyBattery: the same noise as runs 1a-1c, reseeded with every validation flight).
Nothing is trained. It measures what a noisy battery costs at decision time alone, the most a policy that switches
from the true battery to a noisy estimate could keep. Final checkpoints, deterministic, validation flights only
(never the final test).

    python step1_fly_with_noise.py --runs runs_step0/<launch>/runs.json [...] --reward-scale 0.1 \
        --noises 0 0.03 0.06 0.12 --output runs_step0/fly_with_noise.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

STEP0_CONDITIONS = ("0a", "0b", "0c")
BATTERIES = (20, 30, 40, 50, 60, 70)


def fly_run(condition: str, run_dir: str, noises: list[float]) -> dict:
    """Validation results of one finished step 0 run at every noise level."""
    import torch
    from omegaconf import OmegaConf
    from stable_baselines3 import PPO

    import main_battery_plane_contextual_ppo as training
    import report_step0

    torch.set_num_threads(1)
    directory = Path(run_dir)
    cfg = OmegaConf.load(directory / "config.yaml")
    # A step 0 run: trained with the true battery, under step 0's protocol
    report_step0.evaluation_protocol(cfg, condition)
    model = (training.ContextualPPO if "cppo" in cfg else PPO).load(directory / "final_model.zip", device="cpu")
    if model.num_timesteps != report_step0.FINAL_TIMESTEPS:
        raise ValueError(f"Not a finished step 0 run: {directory}")
    flights = {}
    for noise in noises:
        flown = OmegaConf.merge(cfg, {"env": {"battery_noise": float(noise)}})
        result = training.evaluate_schedule(model, flown, flown.evaluation.reset_grid)
        metrics = result["flight_metrics"]
        throttle = [metrics[f"by_b0/b0_{b}pct/throttle_mean"] for b in BATTERIES]
        flights[f"{noise:g}"] = {
            "mean_reward": result["mean_reward"],
            "completion_rate": result["completion_rate"],
            "ground_crash_rate": result["ground_crash_rate"],
            "battery_empty_rate": result["battery_empty_rate"],
            "time_in_band": metrics["time_in_band"],
            "throttle_range": max(throttle) - min(throttle),
            "return_by_b0": [metrics[f"by_b0/b0_{b}pct/return_mean"] for b in BATTERIES],
        }
    return {"condition": condition, "seed": int(cfg.training.seed), "run_dir": str(directory),
            "reward_scale": float(learner_section(cfg).get("reward_scale", 1.0)), "flights": flights}


def learner_section(cfg):
    return cfg.cppo if "cppo" in cfg else cfg.ppo


def selected_runs(runs_files: list[Path], reward_scale: float) -> list[tuple[str, str]]:
    """Finished 0a/0b/0c runs of the given reward scale, from launcher runs.json files."""
    from omegaconf import OmegaConf

    runs, seen = [], set()
    for path in runs_files:
        for row in json.loads(path.read_text()):
            if row["condition"] not in STEP0_CONDITIONS or row["status"] != "done":
                continue
            cfg = OmegaConf.load(Path(row["run_dir"]) / "config.yaml")
            if float(learner_section(cfg).get("reward_scale", 1.0)) != reward_scale:
                continue
            identity = (row["condition"], int(cfg.training.seed))
            if identity in seen:
                raise ValueError(f"Duplicate condition/seed {identity} in {path}")
            seen.add(identity)
            runs.append((row["condition"], row["run_dir"]))
    return sorted(runs)


def summarize(results: list[dict], noises: list[float]) -> list[str]:
    lines = []
    for condition in STEP0_CONDITIONS:
        rows = [r for r in results if r["condition"] == condition]
        if not rows:
            continue
        parts = []
        for noise in noises:
            values = [r["flights"][f"{noise:g}"] for r in rows]
            parts.append(f"noise {noise:g}: {np.mean([v['mean_reward'] for v in values]):4.0f} "
                         f"({np.mean([v['completion_rate'] for v in values]):.0%} completed)")
        lines.append(f"{condition} ({len(rows)} seeds) | " + " | ".join(parts))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="runs.json files of step 0 launches")
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.03, 0.06, 0.12])
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.noises) < 0:
        parser.error("Noise levels must be >= 0")
    runs = selected_runs(args.runs, args.reward_scale)
    if not runs:
        parser.error("No finished step 0 runs with that reward scale")
    print(f"{len(runs)} step 0 policies x noise {[f'{n:g}' for n in args.noises]}, no training", flush=True)
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(fly_run, *zip(*[(condition, run_dir, args.noises) for condition, run_dir in runs])))
    results.sort(key=lambda r: (r["condition"], r["seed"]))
    args.output.write_text(json.dumps({"noises": args.noises, "reward_scale": args.reward_scale, "runs": results},
                                      indent=2, allow_nan=False) + "\n")
    for line in summarize(results, args.noises):
        print(line, flush=True)


if __name__ == "__main__":
    main()
