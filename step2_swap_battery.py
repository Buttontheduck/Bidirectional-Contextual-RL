"""Step 2 of ENCODER_PLAN.md: fly the ideal-context policies with the true battery in place of the ideal estimate.

Plain PPO trained on the ideal estimator's estimate (runs 2a_ideal, 2b_ideal) scored above plain PPO trained on the true
battery (0a, 0b). This asks why: the same final models fly the validation flights (seeds 1000-1002, never the final
test) once with the estimate they were trained on and once with the true battery in its place (context.source
=true_battery, the same width). If the true battery scores at least as well, the estimate carried no extra information
and the estimate-trained policies are simply better pilots; if it scores worse, the policies rely on something in the
estimate beyond the battery. Nothing is trained.

    python step2_swap_battery.py --runs runs_step0/<launch>/runs.json --output runs_step0/<launch>/swap_true_battery.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("2a_ideal", "2b_ideal")
SOURCES = ("ideal", "true_battery")


def fly(run_dir: str, source: str) -> dict:
    """Final validation of one run's final model with its own context or with the true battery."""
    import torch
    from omegaconf import OmegaConf
    from stable_baselines3 import PPO

    import main_battery_plane_contextual_ppo as training

    torch.set_num_threads(1)
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    if cfg.context.source != "ideal" or cfg.context.include_std:
        raise ValueError(f"Not an estimate-only ideal-context run: {run_dir}")
    model = PPO.load(Path(run_dir) / "final_model.zip", device="cpu")
    if source == "true_battery":
        cfg = OmegaConf.merge(cfg, {"context": {"source": "true_battery"}})
    result = training.evaluate_schedule(model, cfg, cfg.evaluation.reset_grid)
    return {"run_dir": run_dir, "seed": int(cfg.training.seed), "source": source, "mean_reward": result["mean_reward"],
            "completion_rate": result["completion_rate"], "battery_empty_rate": result["battery_empty_rate"],
            "ground_crash_rate": result["ground_crash_rate"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="runs.json of the step 2 launch")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in json.loads(args.runs.read_text()) if r["condition"] in CONDITIONS and r["status"] == "done"]
    if not rows:
        parser.error(f"No finished {CONDITIONS} runs in {args.runs}")
    conditions = {r["run_dir"]: r["condition"] for r in rows}
    jobs = [(r["run_dir"], source) for r in rows for source in SOURCES]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(fly, *zip(*jobs)))
    for result in results:
        result["condition"] = conditions[result["run_dir"]]
    summary = {}
    for condition in CONDITIONS:
        for source in SOURCES:
            group = sorted((r for r in results if r["condition"] == condition and r["source"] == source), key=lambda r: r["seed"])
            if group:
                summary[f"{condition}/{source}"] = {
                    "final_mean_reward": float(np.mean([r["mean_reward"] for r in group])),
                    "seed_rewards": [r["mean_reward"] for r in group],
                    "completion_rate": float(np.mean([r["completion_rate"] for r in group]))}
                print(f"{condition} flown with {source:12s}: final return {summary[f'{condition}/{source}']['final_mean_reward']:6.1f}, "
                      f"completed {summary[f'{condition}/{source}']['completion_rate']:.0%}", flush=True)
    args.output.write_text(json.dumps({"summary": summary, "runs": results}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
