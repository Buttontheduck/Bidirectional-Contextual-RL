"""Gym4ReaL's MicroGridEnv agents and rule baselines, evaluated as the project's notebook does
(examples/microgrid/benchmarks.ipynb), in the unchanged environment.

Protocol, as in the notebook:
- the test configuration (world_test.yaml: 2019 data) with parameter_generator's defaults;
- the 28 test households (profiles 370-397), one full test year each;
- rule baselines on gym.make("gym4real/microgrid-v0"): random (action_space.sample()), only market (a = 0),
  battery first (a = 1), 50-50 (a = 0.5);
- PPO models through make_vec_env, set_options and model.predict(obs): sampled actions (Stable-Baselines3's default),
  here over --eval-seeds evaluation seeds, plus a deterministic run;
- the notebook's metric: per household, total trading reward + total degradation reward, both raw (pure_rewards, EUR).

Models: the project's pretrained PPO_trained and PPO_quick, and (--runs) final models of train_ppo.py runs, evaluated
with the observation they were trained on (health shown or hidden). With runs of both kinds, the gate: health shown
minus hidden, the training seed as the unit (bootstrap CI and Welch's t-test), and paired by household.

Two additions for reproducibility: the random baseline's action space and the PPO sampling are seeded.

--random-battery-init --starts K: Gym4ReaL's random starting battery states (health 0.6-1, charge 0.2-1, temperature
0-50 C), K per household, drawn from numpy's global generator seeded per (household, start) right before the reset,
so every policy meets the same starting battery. Reported besides the score: the score per hour (a battery starting
near 0.6 can reach its end of life and end the episode early) and the gate on batteries starting below 0.8 health.

    /root/venvs/carlos-gym4real/bin/python contextual_environments/microgrid/evaluate_checkpoint.py --output <file>.json
    ... --runs <train_ppo run dirs> --no-pretrained --output <file>.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GYM4REAL = os.environ.get("GYM4REAL_DIR", "/root/venvs/src/gym4ReaL")
TEST_PROFILES = [str(p) for p in range(370, 398)]
RULES = {"random": None, "only_market": 0.0, "battery_first": 1.0, "50-50": 0.5}
PRETRAINED = ("PPO_trained", "PPO_quick")


def _setup():
    warnings.filterwarnings("ignore")
    os.chdir(GYM4REAL)
    import torch

    torch.set_num_threads(1)


TEST_YEAR_STEPS = 8759


def _test_params(show_soh: bool = False, random_init: bool = False) -> dict:
    from gym4real.envs.microgrid.utils import parameter_generator

    params = parameter_generator(world_options="gym4real/envs/microgrid/world_test.yaml",
                                 random_battery_init=True if random_init else None)
    params["soh"] = show_soh
    return params


def _seed_start(profile: str, start: int, random_init: bool) -> None:
    """With random starting battery states, the env draws them from numpy's global generator at reset: seed it per
    (household, start) right before the reset, so every policy meets the same starting battery."""
    if random_init:
        np.random.seed(100_000 + 100 * int(profile) + start)


def episode_summary(pure: list, rewards: list, soh: float, start_soh: float) -> dict:
    pure = np.array(pure)
    score = float(pure[:, 0].sum() + pure[:, 1].sum())
    return {"trading": float(pure[:, 0].sum()), "degradation": float(pure[:, 1].sum()),
            "clipping": float(pure[:, 2].sum()), "score": score, "score_per_hour": score / len(pure),
            "return": float(np.sum(rewards)), "steps": len(pure), "ended_early": len(pure) < TEST_YEAR_STEPS,
            "start_soh": float(start_soh), "soh_end": float(soh)}


def run_rule(job: tuple[str, str, int, bool]) -> dict:
    name, profile, start, random_init = job
    _setup()
    import gymnasium as gym
    import gym4real  # noqa: F401  (registers the env)

    env = gym.make(id="gym4real/microgrid-v0", **{"settings": _test_params(random_init=random_init)})
    env.action_space.seed(int(profile) + 1000 * start)
    _seed_start(profile, start, random_init)
    obs, info = env.reset(options={"eval_profile": profile})
    start_soh = env.unwrapped._battery.soh_series[0]
    pure, rewards, done = [], [], False
    while not done:
        action = env.action_space.sample() if RULES[name] is None else np.array([RULES[name]])
        with contextlib.redirect_stdout(io.StringIO()):
            obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        pure.append(list(info["pure_rewards"].values()))
        rewards.append(reward)
    return {"policy": name, "profile": profile, "start": start, "run_seed": None, "eval_seed": None,
            **episode_summary(pure, rewards, info["battery"]["soh"], start_soh)}


def run_ppo(job: tuple[str, str, bool, int | None, str, int, bool, int | None]) -> dict:
    label, path, show_soh, run_seed, profile, start, random_init, eval_seed = job
    _setup()
    import gym4real  # noqa: F401
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    vec_env = make_vec_env("gym4real/microgrid-v0", n_envs=1,
                           env_kwargs={"settings": _test_params(show_soh, random_init)})
    model = PPO.load(path, device="cpu")
    assert model.observation_space.shape == vec_env.observation_space.shape, (path, show_soh)
    if eval_seed is not None:
        torch.manual_seed(eval_seed)
    vec_env.set_options({"eval_profile": profile})
    _seed_start(profile, start, random_init)
    obs = vec_env.reset()
    start_soh = vec_env.envs[0].unwrapped._battery.soh_series[0]
    pure, rewards, done = [], [], False
    while not done:
        action, _ = model.predict(obs, deterministic=eval_seed is None)
        with contextlib.redirect_stdout(io.StringIO()):
            obs, r, dones, info = vec_env.step(action)
        done = dones[0]
        pure.append(list(info[0]["pure_rewards"].values()))
        rewards.append(float(r[0]))
    return {"policy": f"{label}_{'deterministic' if eval_seed is None else 'sampled'}", "profile": profile,
            "start": start, "run_seed": run_seed, "eval_seed": eval_seed, "show_soh": show_soh,
            **episode_summary(pure, rewards, info[0]["battery"]["soh"], start_soh)}


def per_profile_means(rows: list[dict], key: str = "score") -> np.ndarray:
    return np.array([np.mean([r[key] for r in rows if r["profile"] == prof]) for prof in TEST_PROFILES])


def summarize(results: list[dict]) -> dict:
    policies = sorted({r["policy"] for r in results})
    summary = {}
    for p in policies:
        rows = [r for r in results if r["policy"] == p]
        scores = per_profile_means(rows)
        summary[p] = {"score_mean": float(scores.mean()), "score_std": float(scores.std()),
                      "ci95": float(1.96 * scores.std() / np.sqrt(len(scores))),
                      **{f"{k}_mean": float(per_profile_means(rows, k).mean())
                         for k in ("trading", "degradation", "return", "soh_end", "score_per_hour", "ended_early")}}
    rules = [p for p in RULES if p in summary]
    best_rule = max(rules, key=lambda p: summary[p]["score_mean"]) if rules else None
    paired = {}
    if best_rule:
        base = per_profile_means([r for r in results if r["policy"] == best_rule])
        for p in policies:
            if p in RULES:
                continue
            diff = per_profile_means([r for r in results if r["policy"] == p]) - base
            boot = np.random.default_rng(0).choice(diff, (20_000, len(diff))).mean(axis=1)
            paired[p] = {"versus": best_rule, "mean_difference": float(diff.mean()),
                         "bootstrap_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
                         "share_of_households_better": float((diff > 0).mean())}
    return {"summary": summary, "best_rule": best_rule, "paired_vs_best_rule": paired}


LOW_START_SOH = 0.8


def _compare(shown: list[dict], hidden: list[dict], key: str) -> dict:
    """Shown minus hidden in the mean of `key`, the training seed as the unit; and paired by scenario."""
    from scipy import stats

    seed_means = lambda rows: np.array([np.mean([r[key] for r in rows if r["run_seed"] == s])
                                        for s in sorted({r["run_seed"] for r in rows})])
    s, h = seed_means(shown), seed_means(hidden)
    rng = np.random.default_rng(0)
    boot = rng.choice(s, (20_000, len(s))).mean(axis=1) - rng.choice(h, (20_000, len(h))).mean(axis=1)
    welch = stats.ttest_ind(s, h, equal_var=False)
    ci = welch.confidence_interval(0.95)
    scenario_means = lambda rows: {k: np.mean([r[key] for r in rows if (r["profile"], r["start"]) == k])
                                   for k in {(r["profile"], r["start"]) for r in rows}}
    ms, mh = scenario_means(shown), scenario_means(hidden)
    diff = np.array([ms[k] - mh[k] for k in sorted(set(ms) & set(mh))])
    return {"shown_per_seed": s.tolist(), "hidden_per_seed": h.tolist(),
            "shown_mean": float(s.mean()), "hidden_mean": float(h.mean()),
            "shown_minus_hidden": float(s.mean() - h.mean()),
            "bootstrap_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
            "welch_ci": [float(ci.low), float(ci.high)], "welch_p": float(welch.pvalue),
            "scenarios": int(len(diff)), "share_of_scenarios_shown_better": float((diff > 0).mean())}


def gate(results: list[dict], group: str) -> dict:
    """Health shown minus hidden for the train_ppo.py runs of one group, per action mode: the score (primary with
    deterministic actions), the score per hour, and, with random starting states, the scenarios whose battery starts
    below LOW_START_SOH health."""
    out = {}
    for mode in ("deterministic", "sampled"):
        shown = [r for r in results if r["policy"] == f"{group}_seeSoH_{mode}"]
        hidden = [r for r in results if r["policy"] == f"{group}_hidden_{mode}"]
        if not shown or not hidden:
            continue
        out[mode] = _compare(shown, hidden, "score")
        out[f"{mode}_per_hour"] = _compare(shown, hidden, "score_per_hour")
        low = lambda rows: [r for r in rows if r["start_soh"] < LOW_START_SOH]
        if low(shown) and len(low(shown)) < len(shown):
            out[f"{mode}_low_start_soh"] = _compare(low(shown), low(hidden), "score")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=[], help="train_ppo.py run directories")
    parser.add_argument("--no-pretrained", action="store_true", help="skip PPO_trained and PPO_quick")
    parser.add_argument("--no-rules", action="store_true")
    parser.add_argument("--eval-seeds", type=int, default=5, help="seeds for PPO's sampled actions")
    parser.add_argument("--random-battery-init", action="store_true",
                        help="evaluate with Gym4ReaL's random starting battery states, seeded per household and start")
    parser.add_argument("--starts", type=int, default=1, help="starts per household (with --random-battery-init)")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = str(Path(args.output).resolve())
    eval_seeds = [None] + list(range(args.eval_seeds))
    models = [] if args.no_pretrained else [
        (name, f"{GYM4REAL}/examples/microgrid/trained_models/{name}", False, None) for name in PRETRAINED]
    groups = set()
    for run in args.runs:
        config = json.loads((Path(run) / "config.json").read_text())
        label = config["group"]
        groups.add(label.replace("_seeSoH", "").replace("_hidden", ""))
        models.append((label, str((Path(run) / "final_model.zip").resolve()), bool(config["show_soh"]), int(config["seed"])))
    if args.starts > 1 and not args.random_battery_init:
        parser.error("--starts needs --random-battery-init (without it every start is the same)")
    scenarios = [(prof, start) for prof in TEST_PROFILES for start in range(args.starts)]
    rule_jobs = [] if args.no_rules else [(name, prof, start, args.random_battery_init)
                                          for name in RULES for prof, start in scenarios]
    ppo_jobs = [(label, path, show, seed, prof, start, args.random_battery_init, ev) for label, path, show, seed in models
                for prof, start in scenarios for ev in eval_seeds]
    with Pool(args.jobs) as pool:
        results = (pool.map(run_rule, rule_jobs) if rule_jobs else []) + (pool.map(run_ppo, ppo_jobs) if ppo_jobs else [])
    report = {"protocol": "examples/microgrid/benchmarks.ipynb, test configuration, profiles 370-397"
                          + (f", random starting battery states, {args.starts} per household" if args.random_battery_init else ""),
              "results": results, **summarize(results), "gate": {g: gate(results, g) for g in sorted(groups)}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    for p, s in sorted(report["summary"].items(), key=lambda kv: -kv[1]["score_mean"]):
        print(f"{p:<38} score {s['score_mean']:8.1f} +- {s['ci95']:5.1f} (std {s['score_std']:6.1f}); trading {s['trading_mean']:7.1f}, "
              f"degradation {s['degradation_mean']:7.1f}, return {s['return_mean']:7.1f}, SoH end {s['soh_end_mean']:.4f}, "
              f"per hour {s['score_per_hour_mean']:.4f}, ended early {s['ended_early_mean']:.0%}")
    print("best rule:", report["best_rule"])
    for p, d in report["paired_vs_best_rule"].items():
        print(f"{p} - {d['versus']}: {d['mean_difference']:+.1f} EUR per household, 95% CI [{d['bootstrap_ci'][0]:+.1f}, "
              f"{d['bootstrap_ci'][1]:+.1f}], better in {d['share_of_households_better']:.0%} of households")
    for g, modes in report["gate"].items():
        for mode, d in modes.items():
            print(f"gate {g} ({mode}): shown {d['shown_mean']:.4f} vs hidden {d['hidden_mean']:.4f}, shown - hidden "
                  f"{d['shown_minus_hidden']:+.4f}, bootstrap CI [{d['bootstrap_ci'][0]:+.4f}, {d['bootstrap_ci'][1]:+.4f}], "
                  f"Welch p {d['welch_p']:.3f}, shown better in {d['share_of_scenarios_shown_better']:.0%} of "
                  f"{d['scenarios']} scenarios")


if __name__ == "__main__":
    main()
