"""Step 4 of ENCODER_PLAN.md, the closed-loop check: a frozen encoder's battery estimate on the flights of the PPO policies
that use it, next to the ideal estimate on the same flights.

The encoders were measured on pilot flights (open loop: a pilot flies, the encoder watches). Inside plain PPO the loop
is closed: the encoder's estimate goes into the policy, the policy's actions shape the flight, and the encoder reads that
flight. This flies every finished frozen-encoder run (4a_encoder / 4b_encoder, from their report_step0.py reports) over
its validation flights (seeds 1000-1002, starting batteries 0.2-0.7; never the final test), exactly as its final
validation did, and records at every step:
    the encoder's estimate   what the policy read (StandaloneEncoderContext, stepped as the context wrapper steps it)
    the true battery         from the env infos, only to score the estimates
    the ideal estimate       IdealBatteryEstimator on the same hidden observations and executed actions (step 2's grid)
The flights must reproduce the report's final validation (returns and lengths), so they are the flights that were scored.

Per run: the encoder's and the ideal estimator's error overall, by flight time and by starting battery, the encoder's
bias (estimate - battery: positive = it thinks there is more battery than there is), and the RMS of encoder - ideal.
Per encoder and batch size: the same pooled, and how the runs' errors relate to their late scores.

    python step4_closed_loop.py --reports runs_step0/frozen_gru_a/step0_report_encoder_E1_gru_point_main_s1.json \\
        runs_step0/frozen_gru_b/step0_report_encoder_E1_gru_point_main_s0.json ... --output runs_step0/closed_loop_frozen_gru.json
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

ENCODER_CONDITIONS = ("4a_encoder", "4b_encoder")
# Flight-time windows in steps (dt = 0.05 s): 0-1 s, 1-5 s, 5-10 s, 10-20 s, 20-35 s
WINDOWS = {"0-1s": (0, 20), "1-5s": (20, 100), "5-10s": (100, 200), "10-20s": (200, 400), "20-35s": (400, 701)}
IDEAL_GRID = {"b_points": 101, "vx_points": 301, "prune_below": 30.0}


def fly_run(run: dict) -> dict:
    """Every validation flight of one frozen-encoder PPO run: estimates, ideal estimates and batteries at every step."""
    import torch as th
    from omegaconf import OmegaConf
    from stable_baselines3 import PPO

    import battery_plane as bp
    from context_wrapper import StandaloneEncoderContext, encoder_model_path
    from flight_logging import terminal_outcome
    import main_battery_plane_contextual_ppo as training
    from step1_ideal_estimator import IdealBatteryEstimator

    th.set_num_threads(1)
    run_dir = Path(run["run_directory"])
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if cfg.context.source != "encoder" or encoder_model_path(cfg).suffix != ".pt":
        raise ValueError(f"{run_dir} is not a frozen standalone-encoder run")
    model = PPO.load(run_dir / "final_model.zip", device="cpu")
    env_kwargs = training.env_kwargs_from(cfg)
    source = StandaloneEncoderContext(1, encoder_model_path(cfg))
    low, high = model.action_space.low, model.action_space.high
    flights = []
    for reset in training.iter_eval_resets(cfg.evaluation.reset_grid):
        env = bp.BatteryPlaneEnv(observation_mode="hidden", **env_kwargs)
        obs, info = env.reset(seed=reset["seed"], options=dict(reset["options"]))
        prev, battery, total = np.zeros(2, np.float32), float(info["b0"]), 0.0
        observations, actions, estimates, batteries = [], [], [], []
        while True:
            # As the context wrapper: the estimate from [o_t, a_{t-1}], memory reset at the flight's first observation
            estimate = source.step(np.array([True]), obs[None], prev[None], np.array([not observations]), np.array([False]), [info])
            action, _ = model.predict(np.concatenate([obs, estimate[0]])[None].astype(np.float32), deterministic=True)
            executed = np.clip(np.asarray(action[0], np.float32), low, high)
            observations.append(obs.copy())
            actions.append(executed)
            estimates.append(float(estimate[0, 0]))
            batteries.append(battery)
            obs, reward, terminated, truncated, info = env.step(executed)
            total += float(reward)
            battery, prev = float(info["b"]), executed
            if terminated or truncated:
                break
        ideal = IdealBatteryEstimator(env_kwargs, **IDEAL_GRID)
        means = [ideal.start(observations[0])[0]]
        means += [ideal.update(actions[t - 1], observations[t])[0] for t in range(1, len(observations))]
        flights.append({"seed": reset["seed"], "b0": reset["options"]["b0"], "return": total, "length": len(estimates),
                        "outcome": terminal_outcome(info), "estimates": estimates, "ideal": [float(m) for m in means],
                        "batteries": batteries})
    # These must be the flights the report scored
    saved = run["final_validation"]
    if ([f["length"] for f in flights] != saved["episode_lengths"]
            or not np.allclose([f["return"] for f in flights], saved["episode_rewards"], atol=1e-4)):
        raise RuntimeError(f"These flights differ from the report's final validation of {run_dir}")
    return {"run_directory": str(run_dir), "condition": run["condition"], "seed": run["seed"],
            "encoder": str(cfg.context.encoder_tag), "batch_size": int(cfg.ppo.batch_size),
            "late_mean_reward": run["late_mean_reward"], "final_mean_reward": saved["mean_reward"],
            "completion_rate": saved["completion_rate"], "flights": flights}


def errors(flights: list[dict]) -> dict:
    """Encoder and ideal-estimator errors over the given flights: RMS overall, by window and by starting battery, the
    encoder's bias, and the RMS of encoder - ideal."""
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
    e, i, t, b0 = [], [], [], []
    for f in flights:
        battery = np.array(f["batteries"])
        e.append(np.array(f["estimates"]) - battery)
        i.append(np.array(f["ideal"]) - battery)
        t.append(np.arange(len(battery)))
        b0.append(np.full(len(battery), f["b0"]))
    e, i, t, b0 = (np.concatenate(x) for x in (e, i, t, b0))
    window = {name: (t >= lo) & (t < hi) for name, (lo, hi) in WINDOWS.items()}
    return {
        "encoder_rmse": rms(e), "ideal_rmse": rms(i), "encoder_minus_ideal": rms(e - i), "encoder_bias": float(e.mean()),
        "by_window": {name: {"encoder": rms(e[k]), "ideal": rms(i[k]), "encoder_minus_ideal": rms((e - i)[k]),
                             "encoder_bias": float(e[k].mean()) if k.any() else None} for name, k in window.items()},
        "by_b0": {f"{v:g}": {"encoder": rms(e[b0 == v]), "ideal": rms(i[b0 == v]), "encoder_bias": float(e[b0 == v].mean())}
                  for v in sorted(set(b0.tolist()))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", type=Path, nargs="+", required=True, help="report_step0.py reports of frozen-encoder runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists")
    runs = [run for path in args.reports for run in json.loads(path.read_text())["runs"] if run["condition"] in ENCODER_CONDITIONS]
    if not runs:
        parser.error("No 4a_encoder / 4b_encoder runs in the reports")
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(fly_run, runs))
    for result in results:
        result["errors"] = errors(result["flights"])
    groups = {}
    for result in results:
        groups.setdefault(f"{result['encoder']}, batch {result['batch_size']}", []).append(result)
    summary = {}
    for name, group in sorted(groups.items()):
        pooled = errors([f for r in group for f in r["flights"]])
        late = np.array([r["late_mean_reward"] for r in group])
        rmse = np.array([r["errors"]["encoder_rmse"] for r in group])
        summary[name] = {"runs": len(group), "late_mean_reward": float(late.mean()), **pooled,
                         "run_encoder_rmse": rmse.tolist(), "run_late_scores": late.tolist(),
                         "correlation_rmse_late": float(np.corrcoef(rmse, late)[0, 1]) if len(group) > 2 else None}
        print(f"{name}: late {late.mean():6.1f} | encoder error {pooled['encoder_rmse']:.4f} (bias {pooled['encoder_bias']:+.4f}), "
              f"ideal {pooled['ideal_rmse']:.4f}, encoder - ideal {pooled['encoder_minus_ideal']:.4f} | by window "
              + " ".join(f"{k} {v['encoder']:.3f}/{v['ideal']:.3f}" for k, v in pooled["by_window"].items()), flush=True)
    late_all = np.array([r["late_mean_reward"] for r in results])
    rmse_all = np.array([r["errors"]["encoder_rmse"] for r in results])
    output = {"reports": [str(p) for p in args.reports], "ideal_grid": IDEAL_GRID, "windows": WINDOWS, "summary": summary,
              "correlation_rmse_late_all_runs": float(np.corrcoef(rmse_all, late_all)[0, 1]),
              "runs": [{k: v for k, v in r.items() if k != "flights"} for r in results],
              "flights": {r["run_directory"]: r["flights"] for r in results}}
    args.output.write_text(json.dumps(output, allow_nan=False) + "\n")
    print(f"Correlation of a run's encoder error with its late score, all {len(results)} runs: "
          f"{output['correlation_rmse_late_all_runs']:+.2f}. Written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
