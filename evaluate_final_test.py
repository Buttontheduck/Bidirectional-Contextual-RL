"""Evaluate frozen final checkpoints on the reserved final-test flights, once, after all training and selection.
Only this script reads the final-test grid (seeds 2000-2002, config/cppo.yaml final_test.reset_grid).

    python evaluate_final_test.py --grid runs_cppo/grid
        The A-F grid of run_grid.py (Contextual PPO): final_test.json per run and final_test_results.json.

    python evaluate_final_test.py --reports <report_step0.py reports> --conditions 0a_blind 0b_blind 4a_encoder 4b_encoder \\
        --output runs_step0/final_test.json
        Step 0-4 runs from their report_step0.py reports: plain PPO with or without a frozen context (a frozen encoder
        reloads its encoder file) and the Contextual PPO controls, each with its own saved config. Every run is first
        scored again on the validation flights, which must reproduce the report's final validation, so it is exactly
        the model that was reported. Then it is scored on the final-test flights: its validation grid with only the
        seeds replaced by the reserved ones (for a Contextual PPO config that is its final_test grid). Each run keeps
        its result in its directory (final_test.json), which is reused, never overwritten. The output summarizes the
        final test per condition and frozen encoder, next to the report's late score and final validation.
"""
import os

# One BLAS thread per process (as run_step0.py gives every run): the ideal estimator's numpy work would otherwise start a
# thread per core in every --reports worker. An explicit setting (e.g. the --grid command in CLAUDE.md) is kept.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import subprocess

import numpy as np
from omegaconf import OmegaConf

from main_battery_plane_contextual_ppo import (
    ContextualPPO, configure_cpu_threads, evaluate_schedule, validate_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def final_test_seeds() -> list[int]:
    """The reserved seeds, from their one definition (config/cppo.yaml)."""
    return [int(seed) for seed in OmegaConf.load(PROJECT_ROOT / "config" / "cppo.yaml").final_test.reset_grid.eval_seeds]


def final_test_grid(cfg):
    """A run's validation grid with only the seeds replaced by the reserved final-test seeds. Refused if they overlap the
    run's validation seeds or the start seeds of its training environments (seed + i)."""
    grid = OmegaConf.to_container(cfg.evaluation.reset_grid, resolve=True)
    seeds = final_test_seeds()
    seed, n_envs = int(cfg.training.seed), int(cfg.training.n_envs)
    if set(seeds) & (set(grid["eval_seeds"]) | set(range(seed, seed + n_envs))):
        raise ValueError(f"Final-test seeds {seeds} overlap the validation or training seeds of seed {seed}")
    test = OmegaConf.create({**grid, "eval_seeds": seeds})
    if "final_test" in cfg and OmegaConf.to_container(cfg.final_test.reset_grid, resolve=True) != OmegaConf.to_container(test):
        raise ValueError("A Contextual PPO config whose final_test grid is not its validation grid with the reserved seeds")
    return test


def score_run(run: dict) -> dict:
    """One reported run: its validation reproduced, then its final test (reused if already in its directory)."""
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    directory = Path(run["run_directory"])
    report = directory / "final_test.json"
    cfg = OmegaConf.load(run["config"])
    if report.exists():
        test = json.loads(report.read_text())
        reused = True
    else:
        algorithm = ContextualPPO if "cppo" in cfg else PPO
        model = algorithm.load(directory / "final_model.zip", device="cpu")
        saved = run["final_validation"]
        validation = evaluate_schedule(model, cfg, cfg.evaluation.reset_grid)
        if (validation["episode_lengths"] != saved["episode_lengths"]
                or not np.allclose(validation["episode_rewards"], saved["episode_rewards"], rtol=0, atol=1e-4)):
            raise RuntimeError(f"{directory}: the validation flights don't reproduce the report's final validation")
        test = evaluate_schedule(model, cfg, final_test_grid(cfg))
        test["split"] = "final_test"
        test["validation_reproduced"] = True
        with report.open("x") as stream:
            stream.write(json.dumps(test, indent=2, allow_nan=False) + "\n")
        reused = False
    context = cfg.get("context")
    encoder = str(context.encoder_tag) if context is not None and context.get("source") == "encoder" else None
    return {"condition": run["condition"], "seed": run["seed"], "encoder": encoder, "run_directory": str(directory),
            "late_mean_reward": run["late_mean_reward"], "final_validation_mean_reward": run["final_validation"]["mean_reward"],
            "final_test": test, "reused": reused}


def summarize(rows: list[dict]) -> dict:
    """The final test per condition and frozen encoder, next to the report's late score and final validation."""
    groups = {}
    for row in rows:
        groups.setdefault(row["condition"] + (f" ({row['encoder']})" if row["encoder"] else ""), []).append(row)
    summary = {}
    for name, group in sorted(groups.items()):
        group = sorted(group, key=lambda r: r["seed"])
        test = np.array([r["final_test"]["mean_reward"] for r in group])
        rate = lambda key: float(np.mean([r["final_test"][key] for r in group]))
        summary[name] = {
            "seeds": [r["seed"] for r in group], "final_test_seed_scores": test.tolist(),
            "final_test_mean_reward": float(test.mean()), "final_test_std_across_seeds": float(test.std(ddof=1)) if len(test) > 1 else None,
            "final_test_median": float(np.median(test)), "final_test_min": float(test.min()),
            "final_test_completion_rate": rate("completion_rate"), "final_test_battery_empty_rate": rate("battery_empty_rate"),
            "final_test_ground_crash_rate": rate("ground_crash_rate"),
            "late_mean_reward": float(np.mean([r["late_mean_reward"] for r in group])),
            "final_validation_mean_reward": float(np.mean([r["final_validation_mean_reward"] for r in group])),
        }
    return summary


def evaluate_reports(reports: list[Path], conditions: list[str] | None, output: Path, jobs: int) -> dict:
    runs = [run for path in reports for run in json.loads(path.read_text())["runs"]
            if conditions is None or run["condition"] in conditions]
    if not runs:
        raise ValueError("No runs of the given conditions in the reports")
    keys = [(run["run_directory"]) for run in runs]
    if len(set(keys)) != len(keys):
        raise ValueError("A run appears twice in the reports")
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(score_run, runs))
    git = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    result = {"reports": [str(p) for p in reports], "conditions": conditions, "final_test_seeds": final_test_seeds(),
              "rule": "Each run's validation grid with only the seeds replaced; validation re-scored first and reproduced",
              "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
              "created": datetime.now().isoformat(timespec="seconds"), "summary": summarize(rows), "runs": rows}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def evaluate_grid(grid: Path) -> None:
    grid = grid.resolve()
    results = json.loads((grid / "grid_results.json").read_text())
    summary = []
    for run in results["runs"]:
        directory = grid / run["run_dir"]
        report = directory / "final_test.json"
        if report.exists():
            result = json.loads(report.read_text())
            print(f"REUSE final test: {run['experiment']} seed {run['seed']}")
        else:
            cfg = OmegaConf.load(directory / "config.yaml")
            validate_config(cfg)
            configure_cpu_threads(cfg)
            model = ContextualPPO.load(directory / "final_model.zip", device="cpu")
            result = evaluate_schedule(model, cfg, cfg.final_test.reset_grid)
            result["split"] = "final_test"
            with report.open("x") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        summary.append({"experiment": run["experiment"], "seed": run["seed"], **result})
        print(f"{run['experiment']} seed {run['seed']}: {result['mean_reward']:.3f}")
    (grid / "final_test_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--grid", type=Path, help="Completed run_grid.py output directory")
    mode.add_argument("--reports", type=Path, nargs="+", help="report_step0.py reports of finished step 0-4 runs")
    parser.add_argument("--conditions", nargs="+", default=None, help="--reports: only these conditions (default: all)")
    parser.add_argument("--output", type=Path, default=None, help="--reports: the summary JSON (must not exist)")
    parser.add_argument("--jobs", type=int, default=16, help="--reports: runs scored in parallel, 1 thread each")
    args = parser.parse_args(argv)
    if args.grid is not None:
        if args.conditions or args.output:
            parser.error("--conditions and --output are for --reports")
        evaluate_grid(args.grid)
        return
    if args.output is None or args.output.exists():
        parser.error("--reports needs a new --output")
    result = evaluate_reports(args.reports, args.conditions, args.output, args.jobs)
    for name, s in result["summary"].items():
        print(f"{name}: final test {s['final_test_mean_reward']:.1f} ± {s['final_test_std_across_seeds'] or 0:.1f} "
              f"(median {s['final_test_median']:.1f}, completed {s['final_test_completion_rate']:.0%}) | validation: "
              f"late {s['late_mean_reward']:.1f}, final {s['final_validation_mean_reward']:.1f}", flush=True)
    print(f"Written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
