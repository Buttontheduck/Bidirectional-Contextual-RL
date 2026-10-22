"""Which validation episodes of the random-fatigue elbow task can be solved at all?

For every validation reset seed (1000-1029) with random starting fatigue, the tired arm is driven as hard as it goes
each way for one 2 s episode: all flexors at the strongest command and the extensors at rest, then the reverse. The
angles it ends at bound what any policy can hold, so an episode whose target lies more than the task's threshold
(0.175 rad = 10 deg) outside them cannot end on target. Next to that, each given policy group's per-episode outcome from
the final validation in its runs (and from evaluate_ppo.py outputs).

    /root/venvs/carlos-myo/bin/python contextual_environments/myo/reachability.py \
        --runs contextual_environments/myo/runs/PPO_FatiElbowPose1D6MRandom_randomMF_rs0.1_s*_2026* ... \
        --references contextual_environments/myo/runs/reference/nofat_policy_on_fati_random_T100.json \
        --output contextual_environments/myo/runs/reference/reachability.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_ppo import VALIDATION_SEEDS, find_fatigue, make_env

EXTENSORS, FLEXORS = [0, 1, 2], [3, 4, 5]                 # TRIlong, TRIlat, TRImed | BIClong, BICshort, BRA


def drive(env, seed: int, push: list[int], steps: int = 100) -> float:
    """The elbow angle after the muscles in push get the strongest command and the others rest."""
    env.reset(seed=seed)
    action = -np.ones(6)
    action[push] = 1.0
    for _ in range(steps):
        env.step(action)
    return float(env.unwrapped.data.qpos[0])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[], help="train_ppo.py run directories")
    parser.add_argument("--references", nargs="*", default=[], help="evaluate_ppo.py outputs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    env = make_env("myoFatiElbowPose1D6MRandom-v0", fatigue_reset="random")
    fatigue = find_fatigue(env).muscle_fatigue
    model = env.unwrapped.model
    peak = np.asarray(model.actuator_gainprm[:, 2], dtype=float)            # peak isometric force of each muscle
    threshold = float(env.unwrapped.pose_thd)
    episodes = []
    for seed in VALIDATION_SEEDS:
        env.reset(seed=seed)
        target, start, mf = float(env.unwrapped.target_jnt_value[0]), float(env.unwrapped.data.qpos[0]), fatigue.MF.copy()
        highest, lowest = drive(env, seed, FLEXORS), drive(env, seed, EXTENSORS)
        episodes.append({
            "seed": seed, "target": target, "start": start, "start_mf": mf.tolist(),
            "flexor_strength_left": float(peak[FLEXORS] @ (1 - mf[FLEXORS]) / peak[FLEXORS].sum()),
            "extensor_strength_left": float(peak[EXTENSORS] @ (1 - mf[EXTENSORS]) / peak[EXTENSORS].sum()),
            "highest_angle": highest, "lowest_angle": lowest,
            "reachable": bool(lowest - threshold <= target <= highest + threshold)})
    env.close()

    groups = {}
    for run in args.runs:
        config = json.loads((Path(run) / "config.json").read_text())
        final = json.loads((Path(run) / "validation.json").read_text())[-1]
        groups.setdefault(config["group"], []).append(final["episodes"])
    for path in args.references:
        reference = json.loads(Path(path).read_text())
        groups[Path(path).stem] = [r["episodes"] for r in reference["runs"]]

    reachable = np.array([e["reachable"] for e in episodes])
    summary = {"reachable_episodes": int(reachable.sum()), "episodes": len(episodes), "groups": {}}
    print(f"Reachable: {reachable.sum()} of {len(episodes)} validation episodes")
    for name, runs in groups.items():
        solved = np.array([[e["solved_at_end"] for e in run] for run in runs], dtype=float)   # runs x episodes
        assert [e["seed"] for e in runs[0]] == VALIDATION_SEEDS
        row = {"solved_at_end": float(solved.mean()),
               "solved_at_end_if_reachable": float(solved[:, reachable].mean()),
               "solved_at_end_if_unreachable": float(solved[:, ~reachable].mean()) if (~reachable).any() else None,
               "per_episode": solved.mean(axis=0).tolist()}
        summary["groups"][name] = row
        print(f"{name}: ends on target {row['solved_at_end']:.1%}; reachable episodes {row['solved_at_end_if_reachable']:.1%}, "
              f"unreachable {row['solved_at_end_if_unreachable']}")
    Path(args.output).write_text(json.dumps({"threshold": threshold, "summary": summary, "episodes": episodes}, indent=2) + "\n")
    print("Unreachable episodes:")
    for e in episodes:
        if not e["reachable"]:
            print(f"  seed {e['seed']}: target {np.degrees(e['target']):.0f} deg, reachable {np.degrees(e['lowest_angle']):.0f}-"
                  f"{np.degrees(e['highest_angle']):.0f} deg, strength left flexors {e['flexor_strength_left']:.2f}, "
                  f"extensors {e['extensor_strength_left']:.2f}")


if __name__ == "__main__":
    main()
