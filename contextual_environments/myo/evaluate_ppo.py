"""A saved PPO model on the validation episodes (reset seeds 1000-1029) of any elbow-task setting, e.g. the no-fatigue
policy on the fatigue task. Same options and numbers as train_ppo.py's validation.

    /root/venvs/carlos-myo/bin/python contextual_environments/myo/evaluate_ppo.py \
        --models contextual_environments/myo/runs/PPO_ElbowPose1D6MRandom_rs0.1_s*/final_model.zip \
        --env-id myoFatiElbowPose1D6MRandom-v0 --fatigue-reset random --output <file>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_ppo import VALIDATION_SEEDS, evaluate


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--env-id", default="myoElbowPose1D6MRandom-v0")
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--fatigue-reset", choices=["fresh", "random"], default="fresh")
    parser.add_argument("--observe-fatigue", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    from stable_baselines3 import PPO

    env_options = {"env_id": args.env_id, "episode_steps": args.episode_steps, "fatigue_reset": args.fatigue_reset,
                   "observe_fatigue": args.observe_fatigue}
    results = []
    for path in args.models:
        result = evaluate(PPO.load(path, device="cpu"), env_options, VALIDATION_SEEDS)
        results.append({"model": str(path), **result})
        print(f"{path}: return {result['mean_return']:.1f}, solved at end {result['solved_at_end']:.0%}, "
              f"steps solved {result['solved_share_of_steps']:.1%}", flush=True)
    keys = [k for k, v in results[0].items() if isinstance(v, float)]
    summary = {k: {"mean": float(np.mean([r[k] for r in results])), "std": float(np.std([r[k] for r in results]))}
               for k in keys}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"env_options": env_options, "summary": summary, "runs": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
