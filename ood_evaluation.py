"""Out-of-distribution evaluation of trained BatteryPlane agents (PAPER_ASSESSMENT.md, Tier 2 item 10). Nothing trains.

Every run of the given report_step0.py reports flies, frozen, under conditions it never trained on:

    nominal          the training conditions (batteries 0.2-0.7, speed noise 5), on new seeds: the reference
    battery_outside  starting batteries outside the training range 0.2-0.7: 0.1, 0.15, 0.75, 0.8, 0.9
    noise_0          speed reading without noise (training: std 5 m/s)
    noise_10         speed reading with noise std 10 m/s
    capacity_85      a battery that holds 85 hover-seconds instead of 100: it drains about 18% faster
    capacity_120     120 hover-seconds: it drains about 17% slower
    motor_weaker     the cruise motor's thrust cap 1 m/s^2 lower at every charge (h_min 10.5, h_max 17)
    motor_stronger   1 m/s^2 higher at every charge (h_min 12.5, h_max 19)
    battery_sweep    starting batteries 0.10-0.19 and 0.71-1.00, with 0.20 and 0.70 as in-range anchors, on 5 seeds
                     (not in the default list: --settings battery_sweep)

Each setting flies seeds 3000-3002 x its batteries (18 flights; 15 for battery_outside; battery_sweep: seeds 3000-3004 x
17 batteries, 85 flights), never the validation seeds
(1000-1002) or the reserved final-test seeds (2000-2002). Only the plane changes: a frozen battery estimate (the ideal
estimator, a frozen encoder) keeps the run's training physics, prior and noise model, as a fixed estimator would. Before
anything else, every run flies its own validation grid again and must reproduce its report's final validation exactly,
so the models and environments are the reported ones.

    python ood_evaluation.py --reports <report_step0 reports> --output runs_step0/ood/<name> --jobs 32
    python ood_evaluation.py --reports ... --conditions 0a_blind A J1 --settings nominal capacity_85 --output ...

One JSON per run in --output (reused when present, so a stopped evaluation continues) and summary.json: per agent
(condition and frozen encoder) and setting, the mean return over seeds, its spread, the outcome rates, and the change
from nominal.
"""
import os

# One BLAS thread per process (each run is scored in its own process), as in evaluate_final_test.py
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

PROJECT_ROOT = Path(__file__).resolve().parent
OOD_SEEDS = [3000, 3001, 3002]
SWEEP_SEEDS = [3000, 3001, 3002, 3003, 3004]
SWEEP_BATTERIES = [0.10, 0.12, 0.15, 0.17, 0.18, 0.19, 0.20, 0.70, 0.71, 0.72, 0.73, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
NOMINAL_BATTERIES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# Setting -> (starting batteries, speed-noise std, BatteryPlaneVec arguments that change the plane)
SETTINGS = {
    "nominal": (NOMINAL_BATTERIES, 5.0, {}),
    "battery_outside": ([0.1, 0.15, 0.75, 0.8, 0.9], 5.0, {}),
    "noise_0": (NOMINAL_BATTERIES, 0.0, {}),
    "noise_10": (NOMINAL_BATTERIES, 10.0, {}),
    "capacity_85": (NOMINAL_BATTERIES, 5.0, {"E_hover_s": 85.0}),
    "capacity_120": (NOMINAL_BATTERIES, 5.0, {"E_hover_s": 120.0}),
    "motor_weaker": (NOMINAL_BATTERIES, 5.0, {"h_min": 10.5, "h_max": 17.0}),
    "motor_stronger": (NOMINAL_BATTERIES, 5.0, {"h_min": 12.5, "h_max": 19.0}),
    "battery_sweep": (SWEEP_BATTERIES, 5.0, {}),
}
# Seeds per setting (default OOD_SEEDS); the default --settings leave out the sweep
SETTING_SEEDS = {"battery_sweep": SWEEP_SEEDS}
DEFAULT_SETTINGS = [name for name in SETTINGS if name != "battery_sweep"]


def setting_grid(setting: str) -> dict:
    batteries, noise, _ = SETTINGS[setting]
    return {"eval_seeds": SETTING_SEEDS.get(setting, OOD_SEEDS), "eval_initial_battery": batteries, "eval_vx_obs_noise": [noise],
            "eval_y0": None, "eval_vx0": None}


def check_seeds(cfg) -> None:
    """The out-of-distribution seeds never meet the run's validation, final-test or training start seeds."""
    from evaluate_final_test import final_test_seeds

    seed, n_envs = int(cfg.training.seed), int(cfg.training.n_envs)
    taken = (set(cfg.evaluation.reset_grid.eval_seeds) | set(final_test_seeds())
             | {seed + i for i in range(n_envs)} | {seed * 100 + i for i in range(n_envs)})
    used = set(OOD_SEEDS) | {s for seeds in SETTING_SEEDS.values() for s in seeds}
    if used & taken:
        raise ValueError(f"Out-of-distribution seeds {sorted(used)} overlap the validation, final-test or training seeds of seed {seed}")


def make_env(cfg, schedule: list, physics: dict):
    """
    One evaluation environment, built as main_battery_plane_contextual_ppo.make_eval_env builds it, except that the plane
    flies with `physics` on top of the run's arguments while a frozen context (context_wrapper.py) keeps the run's own.
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    import main_battery_plane_contextual_ppo as training
    from context_wrapper import wrap_context
    from memory_baselines import memory_wrapper_from

    if training.battery_noise_from(cfg) > 0:
        raise ValueError("Runs with noise on the visible battery are not evaluated out of distribution")
    trained = training.env_kwargs_from(cfg)
    raw = training.bp.BatteryPlaneEnv(observation_mode=cfg.env.observation_mode, **{**trained, **physics})
    plane = raw
    memory = memory_wrapper_from(cfg)
    if memory is not None:
        plane = memory(plane)
    scheduler = training.ScheduledEvalResets(plane, schedule)
    env = Monitor(scheduler)
    env.action_space.seed(int(cfg.training.seed) + 1)
    env.reset()
    scheduler.rewind()
    return wrap_context(cfg, DummyVecEnv([lambda: env]), trained), scheduler, raw


def fly(model, cfg, grid: dict, physics: dict) -> dict:
    """The scheduled flights of one grid: returns, lengths and the flight metrics (flight_logging.summarize)."""
    from battery_plane import iter_eval_resets
    from flight_logging import evaluate_flights

    schedule = list(iter_eval_resets(OmegaConf.create(grid)))
    env, scheduler, raw = make_env(cfg, schedule, physics)
    try:
        scheduler.rewind()
        rewards, lengths, recorder = evaluate_flights(model, env, len(schedule), bool(cfg.evaluation.deterministic), raw.core)
    finally:
        env.close()
    outcomes = [flight["outcome"] for flight in recorder.flights]
    return {"mean_reward": float(np.mean(rewards)), "episode_rewards": [float(r) for r in rewards],
            "episode_lengths": [int(n) for n in lengths], "episode_outcomes": outcomes,
            "completion_rate": outcomes.count("completed") / len(outcomes),
            "battery_empty_rate": outcomes.count("battery_empty") / len(outcomes),
            "ground_crash_rate": outcomes.count("ground_crash") / len(outcomes),
            "flight_metrics": recorder.metrics()}


def run_name(run: dict) -> str:
    cfg = OmegaConf.load(run["config"])
    context = cfg.get("context")
    encoder = str(context.encoder_tag) if context is not None and context.get("source") == "encoder" else None
    return "_".join(part for part in (run["condition"], encoder, f"s{run['seed']}") if part)


def score_run(job: tuple) -> dict:
    """One reported run: its validation reproduced, then every setting (reused if already in the output folder)."""
    import torch
    from stable_baselines3 import PPO

    from main_battery_plane_contextual_ppo import ContextualPPO
    from memory_baselines import MaskedRecurrentPPO, memory_kind_from

    run, output, settings = job
    torch.set_num_threads(1)
    path = Path(output) / f"{run_name(run)}.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    if saved is not None and all(setting in saved["settings"] for setting in settings):
        return saved
    cfg = OmegaConf.load(run["config"])
    check_seeds(cfg)
    directory = Path(run["run_directory"])
    algorithm = ContextualPPO if "cppo" in cfg else MaskedRecurrentPPO if memory_kind_from(cfg) == "recurrent" else PPO
    model = algorithm.load(directory / "final_model.zip", device="cpu")
    validation = fly(model, cfg, OmegaConf.to_container(cfg.evaluation.reset_grid, resolve=True), {})
    reported = run["final_validation"]
    if (validation["episode_lengths"] != reported["episode_lengths"]
            or not np.allclose(validation["episode_rewards"], reported["episode_rewards"], rtol=0, atol=1e-4)):
        raise RuntimeError(f"{directory}: the validation flights don't reproduce the report's final validation")
    result = saved or {"condition": run["condition"], "seed": run["seed"], "name": run_name(run), "run_directory": str(directory),
                       "late_mean_reward": run["late_mean_reward"], "validation_reproduced": True, "settings": {}}
    for setting in settings:
        if setting not in result["settings"]:
            result["settings"][setting] = {"grid": setting_grid(setting), "physics": SETTINGS[setting][2],
                                           **fly(model, cfg, setting_grid(setting), SETTINGS[setting][2])}
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def agent(row: dict) -> str:
    """The agent of a run: its condition, with the frozen encoder if any (the name without the seed)."""
    return row["name"].rsplit("_s", 1)[0]


def summarize(rows: list[dict], settings: list[str]) -> dict:
    summary = {}
    for name in sorted({agent(row) for row in rows}):
        group = sorted((row for row in rows if agent(row) == name), key=lambda row: row["seed"])
        summary[name] = {"seeds": [row["seed"] for row in group]}
        nominal = np.array([row["settings"]["nominal"]["mean_reward"] for row in group]) if "nominal" in settings else None
        for setting in settings:
            scores = np.array([row["settings"][setting]["mean_reward"] for row in group])
            rate = lambda key: float(np.mean([row["settings"][setting][key] for row in group]))
            entry = {"mean_reward": float(scores.mean()), "std_across_seeds": float(scores.std(ddof=1)) if len(scores) > 1 else None,
                     "seed_scores": scores.tolist(), "completion_rate": rate("completion_rate"),
                     "battery_empty_rate": rate("battery_empty_rate"), "ground_crash_rate": rate("ground_crash_rate")}
            if nominal is not None and setting != "nominal":
                entry["change_from_nominal"] = float((scores - nominal).mean())
            summary[name][setting] = entry
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", type=Path, nargs="+", required=True, help="report_step0.py reports of finished runs")
    parser.add_argument("--conditions", nargs="+", default=None, help="only these conditions (default: all)")
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=DEFAULT_SETTINGS)
    parser.add_argument("--output", type=Path, required=True, help="folder for one JSON per run and summary.json")
    parser.add_argument("--jobs", type=int, default=16, help="runs scored in parallel, 1 thread each")
    args = parser.parse_args(argv)
    runs = [run for path in args.reports for run in json.loads(path.read_text())["runs"]
            if args.conditions is None or run["condition"] in args.conditions]
    if not runs:
        parser.error("No runs of the given conditions in the reports")
    if len({run["run_directory"] for run in runs}) != len(runs):
        parser.error("A run appears twice in the reports")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(run, str(args.output), args.settings) for run in runs]
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(score_run, jobs))
    git = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    summary = summarize(rows, args.settings)
    result = {"reports": [str(p) for p in args.reports], "conditions": args.conditions,
              "settings": {name: {"seeds": SETTING_SEEDS.get(name, OOD_SEEDS), "batteries": SETTINGS[name][0],
                                  "speed_noise": SETTINGS[name][1], "physics": SETTINGS[name][2]}
                           for name in args.settings},
              "rule": "The plane changes, a frozen estimate keeps the training physics; each run's validation reproduced first",
              "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
              "created": datetime.now().isoformat(timespec="seconds"), "summary": summary}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    width = max(len(name) for name in summary)
    print(f"{'agent':{width}s} " + " ".join(f"{setting:>15s}" for setting in args.settings))
    for name, entry in summary.items():
        print(f"{name:{width}s} " + " ".join(f"{entry[setting]['mean_reward']:15.0f}" for setting in args.settings))
    print(f"Written to {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
