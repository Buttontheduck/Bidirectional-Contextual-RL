"""GIFs of a trained plain-PPO agent on the final-test flights: one run of a report_step0.py report, flown on the reserved
final-test seeds (2000-2002) at the given starting batteries, into <output>/b02 ... b08/<name>_seed<seed>.gif.

The flights are the final test's: the run's validation grid with the reserved seeds (evaluate_final_test.final_test_grid,
the same speed-reading noise, starting heights and speeds drawn by seed), the starting battery set by the reset
options, the deterministic policy and the actions clipped to the action space, as the evaluation runs it. A frozen
context (context.source ideal or encoder) is stepped as step4_closed_loop.py steps it. Where the final test covered a
starting battery (0.2-0.7), the flight's return must equal the run's recorded final-test return for that flight
(final_test.json), so the GIF shows exactly the flight that was scored; other batteries (e.g. 0.8, outside the
training range 0.2-0.7) are flown the same way without that check. The run is picked by --seed, or by default the
training seed with the best validation late score (never by its test score). GIF format: pilot_gifs.save_gif.

    python policy_gifs.py --report runs_step0/20261014-020124/report_blind_and_A.json --condition 0a_blind \\
        --name memoryless_ppo --label "memoryless PPO" --output videos/PPO
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BATTERIES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def battery_folder(b0: float) -> str:
    """The battery's digits without the point: 0.2 -> b02, 0.15 -> b015, 1.0 -> b1."""
    return "b" + f"{b0:g}".replace(".", "")


def fly_frames(run: dict, b0: float, seed: int, every: int, scale: float) -> tuple[list[np.ndarray], dict]:
    """The run's deterministic policy on one final-test flight with starting battery b0: GIF frames and a summary."""
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from stable_baselines3 import PPO

    import battery_plane as bp
    from context_wrapper import make_context_source
    from evaluate_final_test import final_test_grid
    from flight_logging import terminal_outcome
    import main_battery_plane_contextual_ppo as training

    torch.set_num_threads(1)
    cfg = OmegaConf.load(run["config"])
    if float(cfg.env.get("battery_noise", 0.0) or 0.0) > 0:
        raise ValueError("Runs with a noisy battery are not supported")
    grid = OmegaConf.to_container(final_test_grid(cfg))
    if seed not in grid["eval_seeds"]:
        raise ValueError(f"Seed {seed} is not a final-test seed {grid['eval_seeds']}")
    (reset,) = [r for r in training.iter_eval_resets({**grid, "eval_initial_battery": [b0], "eval_seeds": [seed]})]
    model = PPO.load(Path(run["run_directory"]) / "final_model.zip", device="cpu")
    env_kwargs = training.env_kwargs_from(cfg)
    source = make_context_source(cfg, 1, env_kwargs)
    env = bp.BatteryPlaneEnv(observation_mode=cfg.env.observation_mode, render_mode="rgb_array", **env_kwargs)
    obs, info = env.reset(seed=reset["seed"], options=dict(reset["options"]))
    low, high = model.action_space.low, model.action_space.high
    size = (int(round(env.W * scale)), int(round(env.H * scale)))
    prev, frames, total, t = np.zeros(2, np.float32), [], 0.0, 0
    while True:
        frame = env.render()
        if t % every == 0:
            frames.append(np.asarray(Image.fromarray(frame).resize(size, Image.LANCZOS)))
        policy_obs = obs
        if source is not None:
            # As the context wrapper: the estimate from [o_t, a_{t-1}], memory reset at the flight's first observation
            context = source.step(np.array([True]), obs[None], prev[None], np.array([t == 0]), np.array([False]), [info])
            policy_obs = np.concatenate([obs, context[0]])
        action, _ = model.predict(policy_obs[None].astype(np.float32), deterministic=True)
        executed = np.clip(np.asarray(action[0], np.float32), low, high)
        obs, reward, terminated, truncated, info = env.step(executed)
        total, prev, t = total + float(reward), executed, t + 1
        if terminated or truncated:
            frames.append(np.asarray(Image.fromarray(env.render()).resize(size, Image.LANCZOS)))
            break
    env.close()
    # The final test's recorded return for this flight, where it covered this starting battery
    recorded = None
    report = Path(run["run_directory"]) / "final_test.json"
    if report.exists():
        test = json.loads(report.read_text())
        matches = [ret for r, ret in zip(test["reset_schedule"], test["episode_rewards"]) if r == reset]
        if matches:
            recorded = matches[0]
            if not np.isclose(total, recorded, rtol=0, atol=1e-3):
                raise RuntimeError(f"Seed {seed}, b0 {b0}: return {total:.4f}, the final test recorded {recorded:.4f}")
    return frames, {"b0": b0, "seed": seed, "steps": t, "seconds": t * env.core.dt, "outcome": terminal_outcome(info),
                    "return": total, "final_test_return": recorded, "observation_mode": str(cfg.env.observation_mode),
                    "context": str(cfg.context.source) if "context" in cfg else "none"}


def make_gif(job: tuple) -> dict:
    run, label, name, b0, seed, every, scale, output = job
    from pilot_gifs import OUTCOME_TEXT, save_gif

    frames, summary = fly_frames(run, b0, seed, every, scale)
    seen = ("sees the battery" if summary["observation_mode"] != "hidden" else
            {"none": "battery hidden", "ideal": "battery hidden, reads the ideal estimate",
             "encoder": "battery hidden, reads its encoder's estimate"}.get(summary["context"], "battery hidden"))
    text = (f"{label}  |  test seed {seed}  |  start battery {b0:.0%}  |  {seen}  |  "
            f"{OUTCOME_TEXT.get(summary['outcome'], summary['outcome'])} at {summary['seconds']:.1f} s, return {summary['return']:.0f}")
    path = Path(output) / battery_folder(b0) / f"{name}_seed{seed}.gif"
    return {**summary, "file": str(path), "frames": len(frames), "megabytes": save_gif(frames, text, path, scale)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True, help="A report_step0.py report")
    parser.add_argument("--condition", required=True, help="e.g. 0a_blind")
    parser.add_argument("--seed", type=int, default=None, help="Training seed (default: the best validation late score)")
    parser.add_argument("--name", required=True, help="File name prefix, e.g. memoryless_ppo")
    parser.add_argument("--label", required=True, help="Caption name, e.g. 'memoryless PPO'")
    parser.add_argument("--batteries", type=float, nargs="+", default=list(DEFAULT_BATTERIES))
    parser.add_argument("--test-seeds", type=int, nargs="+", default=[2000, 2001, 2002])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "videos" / "PPO")
    parser.add_argument("--every", type=int, default=2, help="Keep every n-th step (20 fps playback: n x real time)")
    parser.add_argument("--scale", type=float, default=2 / 3)
    parser.add_argument("--jobs", type=int, default=21)
    args = parser.parse_args()
    runs = [r for r in json.loads(args.report.read_text())["runs"] if r["condition"] == args.condition]
    if not runs:
        parser.error(f"No {args.condition} runs in {args.report}")
    run = (max(runs, key=lambda r: r["late_mean_reward"]) if args.seed is None
           else next((r for r in runs if r["seed"] == args.seed), None))
    if run is None:
        parser.error(f"No {args.condition} run with training seed {args.seed}")
    if any(not 0 < b <= 1 for b in args.batteries):
        parser.error("--batteries must lie in (0, 1]")
    label = f"{args.label} (training seed {run['seed']})"
    print(f"{args.condition} training seed {run['seed']}: validation late score {run['late_mean_reward']:.1f}, "
          f"{run['run_directory']}", flush=True)
    jobs = [(run, label, args.name, b0, seed, args.every, args.scale, str(args.output))
            for b0 in args.batteries for seed in args.test_seeds]
    with ProcessPoolExecutor(min(args.jobs, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(make_gif, jobs))
    for r in results:
        check = "matches the final test" if r["final_test_return"] is not None else "no final-test record (outside its grid)"
        print(f"b0 {r['b0']:g} seed {r['seed']}: {r['outcome']:13s} at {r['seconds']:5.1f} s, return {r['return']:7.1f} "
              f"({check}) | {r['megabytes']:.1f} MB -> {r['file']}", flush=True)
    print(f"{len(results)} GIFs, {sum(r['megabytes'] for r in results):.0f} MB in total", flush=True)


if __name__ == "__main__":
    main()
