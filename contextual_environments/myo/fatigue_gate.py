"""The value-of-context gate for a fatigue task: does a memoryless PPO policy that sees the fatigue beat one that does
not, and do the seeing policies' actions depend on the fatigue?

The shown runs are an upper bound for any learner that has to infer the fatigue (Contextual PPO included), so the gap
shown - hidden is the most that perfect inference could win.

Every given run's final model (train_ppo.py run directories: the hidden and the shown runs of one task) flies
--episodes episodes from reset seeds --first-seed onwards, with deterministic actions. Each shown run flies them three
times:
- with its true fatigue;
- with another episode's fatigue: episode k is shown the fatigue the same policy had in episode k + 1 at the same time
  step (a real fatigue trajectory of that policy, from a different history), its own arm and targets unchanged;
- with fresh muscles shown (MF = 0 throughout).

Statistics, with the training seed as the unit:
- the gap shown - hidden in mean return: bootstrap 95% CI (20,000 resamples), Welch's t-test (CI and p), Mann-Whitney U;
- the losses true - decoy of the shown runs, paired by run: mean, bootstrap CI, paired t-test.
The gate passes when the shown runs beat the hidden ones by at least --threshold (default 10%) of the hidden return and
the bootstrap CI of the gap lies above 0.

    /root/venvs/carlos-myo/bin/python contextual_environments/myo/fatigue_gate.py \
        --hidden <hidden run dirs> --shown <shown run dirs> --output contextual_environments/myo/runs/gate/<name>.json
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from train_ppo import FINAL_TEST_SEEDS, VALIDATION_SEEDS, find_fatigue, make_env

OPTION_KEYS = ("env_id", "episode_steps", "fatigue_reset", "observe_fatigue", "target_every", "target_range", "load_kg")


def env_options_of(run_dir: str) -> dict:
    config = json.loads((Path(run_dir) / "config.json").read_text())
    return {key: config.get(key) for key in OPTION_KEYS} | {"load_kg": config.get("load_kg") or 0.0}


def make_decoy_env(env_options: dict):
    """The task with a hidden observation, plus a fatigue vector taken from `env.trajectory` (set before each reset):
    entry 0 at the reset, entry t after step t."""
    import gymnasium

    inner = make_env(**{**env_options, "observe_fatigue": False})
    n = find_fatigue(inner).muscle_fatigue.MF.shape[0]

    class DecoyFatigue(gymnasium.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            low, high = env.observation_space.low, env.observation_space.high
            self.observation_space = gymnasium.spaces.Box(np.concatenate([low, np.zeros(n)]),
                                                          np.concatenate([high, np.ones(n)]),
                                                          dtype=env.observation_space.dtype)
            self.trajectory, self.t = np.zeros((1, n)), 0

        def reset(self, **kwargs):
            self.t = 0
            return super().reset(**kwargs)

        def step(self, action):
            self.t += 1
            return super().step(action)

        def observation(self, observation):
            shown = self.trajectory[min(self.t, len(self.trajectory) - 1)]
            return np.concatenate([observation, shown]).astype(self.observation_space.dtype)

    return DecoyFatigue(inner)


def fly(model, env, seeds: list[int], trajectories=None, record: bool = False) -> dict:
    """Deterministic episodes; per episode the return, the share of steps on target and (if record) the true fatigue."""
    fatigue = find_fatigue(env).muscle_fatigue
    returns, on_target, recorded = [], [], []
    for k, seed in enumerate(seeds):
        if trajectories is not None:
            env.trajectory = trajectories[k]
        obs, _ = env.reset(seed=seed)
        mf, total, solved, done = [fatigue.MF.copy()], 0.0, [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            solved.append(bool(np.asarray(info["solved"]).item()))
            mf.append(fatigue.MF.copy())
            done = terminated or truncated
        returns.append(total)
        on_target.append(float(np.mean(solved)))
        if record:
            recorded.append(np.array(mf))
    env.close()
    return {"returns": returns, "on_target": on_target, "trajectories": recorded}


def model_path(run_dir: str, which: str) -> tuple[Path, int | None]:
    """The final model, or (which = "best") the checkpoint with the highest validation return (train_ppo.py
    --save-checkpoints; the untrained first validation excluded)."""
    if which == "final":
        return Path(run_dir) / "final_model.zip", None
    history = [h for h in json.loads((Path(run_dir) / "validation.json").read_text())
               if not h.get("final") and h["timesteps"] > 8]
    best = max(history, key=lambda h: h["mean_return"])["timesteps"]
    return Path(run_dir) / "checkpoints" / f"model_{best}.zip", best


def evaluate_run(job: tuple[str, str, list[int], str]) -> dict:
    run_dir, role, seeds, which = job
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    options = env_options_of(run_dir)
    if options["observe_fatigue"] != (role == "shown"):
        raise ValueError(f"{run_dir} is not a {role} run")
    path, selected = model_path(run_dir, which)
    model = PPO.load(path, device="cpu")
    true = fly(model, make_env(**options), seeds, record=role == "shown")
    result = {"run": run_dir, "role": role, "options": options, "model": str(path), "selected_timesteps": selected,
              "true": {k: true[k] for k in ("returns", "on_target")}}
    if role == "shown":
        others = [true["trajectories"][(k + 1) % len(seeds)] for k in range(len(seeds))]
        fresh = [np.zeros((1, others[0].shape[1]))] * len(seeds)
        for name, trajectories in (("other_episode", others), ("fresh", fresh)):
            decoy = fly(model, make_decoy_env(options), seeds, trajectories=trajectories)
            result[name] = {k: decoy[k] for k in ("returns", "on_target")}
    return result


def bootstrap_gap(a, b, rng, n=20_000):
    """95% bootstrap CI of mean(a) - mean(b), resampling each group's runs."""
    a, b = np.asarray(a), np.asarray(b)
    gaps = rng.choice(a, (n, len(a))).mean(axis=1) - rng.choice(b, (n, len(b))).mean(axis=1)
    return np.percentile(gaps, [2.5, 97.5]).tolist()


def main(argv=None) -> None:
    from scipy import stats

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", nargs="+", required=True)
    parser.add_argument("--shown", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--hidden-model", choices=["final", "best"], default="final",
                        help="best: each hidden run's checkpoint with the highest validation return (seeds 1000-1029); "
                             "then evaluate on episodes outside them, e.g. --first-seed 1030")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    if set(seeds) & set(FINAL_TEST_SEEDS):
        parser.error("The evaluation seeds overlap the reserved final-test seeds")
    if args.hidden_model == "best" and set(seeds) & set(VALIDATION_SEEDS):
        parser.error("Checkpoints selected on the validation seeds must be evaluated on other episodes (--first-seed 1030)")
    task = {k: v for k, v in env_options_of(args.hidden[0]).items() if k != "observe_fatigue"}
    for run in args.hidden + args.shown:
        if {k: v for k, v in env_options_of(run).items() if k != "observe_fatigue"} != task:
            parser.error(f"{run} is a different task")

    jobs = [(run, "hidden", seeds, args.hidden_model) for run in args.hidden] + [(run, "shown", seeds, "final")
                                                                                 for run in args.shown]
    with Pool(min(args.jobs, len(jobs))) as pool:
        results = pool.map(evaluate_run, jobs)

    rng = np.random.default_rng(0)
    mean = lambda r, key="true": float(np.mean(r[key]["returns"]))
    hidden = [mean(r) for r in results if r["role"] == "hidden"]
    shown = [mean(r) for r in results if r["role"] == "shown"]
    base = float(np.mean(hidden))
    gap = float(np.mean(shown)) - base
    welch = stats.ttest_ind(shown, hidden, equal_var=False)
    welch_ci = welch.confidence_interval(0.95)
    summary = {
        "task": task, "episodes": args.episodes, "seeds": [seeds[0], seeds[-1]], "hidden_model": args.hidden_model,
        "hidden_selected_timesteps": [r["selected_timesteps"] for r in results if r["role"] == "hidden"],
        "hidden": {"per_run": hidden, "mean": base, "std": float(np.std(hidden)),
                   "on_target": float(np.mean([np.mean(r["true"]["on_target"]) for r in results if r["role"] == "hidden"]))},
        "shown": {"per_run": shown, "mean": float(np.mean(shown)), "std": float(np.std(shown)),
                  "on_target": float(np.mean([np.mean(r["true"]["on_target"]) for r in results if r["role"] == "shown"]))},
        "gap": {"shown_minus_hidden": gap, "relative": gap / abs(base),
                "bootstrap_ci": bootstrap_gap(shown, hidden, rng),
                "welch_ci": [float(welch_ci.low), float(welch_ci.high)], "welch_p": float(welch.pvalue),
                "mann_whitney_p": float(stats.mannwhitneyu(shown, hidden, alternative="two-sided").pvalue)},
        "decoys": {},
    }
    for name in ("other_episode", "fresh"):
        losses = [mean(r) - mean(r, name) for r in results if r["role"] == "shown"]
        boot = rng.choice(losses, (20_000, len(losses))).mean(axis=1)
        summary["decoys"][name] = {
            "loss_per_run": losses, "loss": float(np.mean(losses)), "relative": float(np.mean(losses)) / abs(float(np.mean(shown))),
            "bootstrap_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
            "paired_t_p": float(stats.ttest_rel([mean(r) for r in results if r["role"] == "shown"],
                                                [mean(r, name) for r in results if r["role"] == "shown"]).pvalue)}
    summary["gate_passed"] = bool(gap >= args.threshold * abs(base) and summary["gap"]["bootstrap_ci"][0] > 0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "runs": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
