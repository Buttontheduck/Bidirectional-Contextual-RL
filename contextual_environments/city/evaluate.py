"""Policies on the 80 validation scenarios of city_env.py: cost, the episode boundary, how often the battery sits empty
or full, and how predictable the charge is from what a hidden-setting agent sees.

Per policy:
- cost: the money paid for grid electricity per 48-hour episode, and the energy-adjusted cost: cost + (starting
  energy - ending energy) x P_REF, so that spending the free starting charge, or emptying the battery just before the
  end, does not count as a saving (P_REF = the tariff's mean price, $0.273 per kWh); starting and ending stored energy;
- saturation: the share of hours that end with the battery empty (charge <= 2%) or full (>= 95%), the median hours to
  the first empty-or-full, and the share of episodes that reach it within 6 hours;
- predictability: a gradient-boosted regressor learns the charge from the hidden-setting observation on 300 training
  scenarios flown by the same policy, and is scored (R^2) on the validation flights; a second one from the clock alone
  (hour, weekday, month, time left). A high R^2 means a memoryless agent effectively knows the charge.

Rules (city_env.RULES) are a preliminary diagnostic of what the charge can be worth, not an upper bound.

    /root/venvs/carlos-citylearn/bin/python contextual_environments/city/evaluate.py --rules --output <file>.json
    /root/venvs/carlos-citylearn/bin/python contextual_environments/city/evaluate.py --runs <train_ppo run dirs> --output <file>.json
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from city_env import EPISODE_STEPS, RULES, TRAIN_STARTS, VALIDATION_SCENARIOS, CityBattery, fly, rule_action

P_REF = 0.2731
CLOCK = ("month_cos", "month_sin", "day_type_cos", "day_type_sin", "hour_cos", "hour_sin", "time_left")


def training_scenarios(n: int = 300, seed: int = 7) -> list[tuple[int, float]]:
    rng = np.random.default_rng(seed)
    return [(int(rng.choice(TRAIN_STARTS)), float(rng.uniform(0.0, 1.0))) for _ in range(n)]


def summarize(episodes: list[dict]) -> dict:
    cost = np.array([e["cost"] for e in episodes])
    start = np.array([e["start_energy"] for e in episodes])
    end = np.array([e["end_energy"] for e in episodes])
    charges = np.array([e["charges"][1:] for e in episodes])           # after each hour
    saturated = (charges <= 0.02) | (charges >= 0.95)
    first = np.array([np.argmax(s) + 1 if s.any() else EPISODE_STEPS + 1 for s in saturated])
    return {"cost": float(cost.mean()), "adjusted_cost": float((cost + (start - end) * P_REF).mean()),
            "start_energy": float(start.mean()), "end_energy": float(end.mean()),
            "share_empty": float((charges <= 0.02).mean()), "share_full": float((charges >= 0.95).mean()),
            "median_hours_to_saturation": float(np.median(first)), "share_saturated_within_6h": float((first <= 6).mean()),
            "per_episode_cost": cost.tolist(), "per_episode_adjusted_cost": (cost + (start - end) * P_REF).tolist()}


def predictability(fit: list[dict], test: list[dict], names: list[str]) -> dict:
    """R^2 of the charge from the hidden-setting observation, and from the clock alone (the stale terminal
    observation left out)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score

    hidden = [i for i, n in enumerate(names) if n != "electrical_storage_soc"]
    clock = [names.index(n) for n in CLOCK]

    def rows(episodes, columns):
        x = np.concatenate([e["observations"][:-1, columns] for e in episodes])
        y = np.concatenate([np.array(e["charges"][:-1]) for e in episodes])
        return x, y

    result = {}
    for label, columns in (("r2_hidden_observation", hidden), ("r2_clock_only", clock)):
        x_fit, y_fit = rows(fit, columns)
        x_test, y_test = rows(test, columns)
        model = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(x_fit, y_fit)
        result[label] = float(r2_score(y_test, model.predict(x_test)))
    return result


def evaluate_rule(name: str) -> dict:
    env = CityBattery(observe_charge=True)                  # rules read the raw state; the observation is only recorded
    act = lambda env, obs: rule_action(name, env)
    test = fly(env, act, VALIDATION_SCENARIOS, record_observations=True)
    fit = fly(env, act, training_scenarios(), record_observations=True)
    return {"policy": name, **summarize(test), **predictability(fit, test, env.observation_names)}


def evaluate_run(run_dir: str) -> dict:
    import torch
    from stable_baselines3 import PPO, SAC

    torch.set_num_threads(1)
    config = json.loads((Path(run_dir) / "config.json").read_text())
    model = (SAC if config.get("algo") == "sac" else PPO).load(Path(run_dir) / "final_model.zip", device="cpu")
    env = CityBattery(observe_charge=config["observe_charge"])
    act = lambda env, obs: model.predict(obs, deterministic=True)[0]
    test = fly(env, act, VALIDATION_SCENARIOS, record_observations=True)
    fit = fly(env, act, training_scenarios(), record_observations=True)
    return {"policy": config["group"], "run": run_dir, "seed": config["seed"], "observe_charge": config["observe_charge"],
            "algo": config.get("algo", "ppo"), **summarize(test), **predictability(fit, test, env.observation_names)}


def versus_rule(runs: list[dict], rule: dict, rng=None) -> dict:
    """Learned controllers (final models, one per seed) against a hand-written rule on the same scenarios. A group
    beats the rule when the bootstrap 95% CI of its mean energy-adjusted cost lies entirely below the rule's."""
    rng = np.random.default_rng(0) if rng is None else rng
    result = {"rule": rule["policy"], "seeds": [r["seed"] for r in runs]}
    for key in ("adjusted_cost", "cost"):
        values = np.array([r[key] for r in runs])
        ci = np.percentile(rng.choice(values, (20_000, len(values))).mean(axis=1), [2.5, 97.5])
        per_scenario = np.mean([r[f"per_episode_{key}"] for r in runs], axis=0) - np.array(rule[f"per_episode_{key}"])
        result[key] = {"learned_mean": float(values.mean()), "learned_std": float(values.std()), "per_seed": values.tolist(),
                       "bootstrap_ci_of_learned_mean": ci.tolist(), "rule": rule[key],
                       "learned_minus_rule": float(values.mean() - rule[key]), "beats_rule": bool(ci[1] < rule[key]),
                       "share_of_scenarios_learned_cheaper": float((per_scenario < 0).mean())}
    result["verdict"] = "beats the rule" if result["adjusted_cost"]["beats_rule"] else "does not beat the rule"
    return result


def gate(runs: list[dict], rng=None) -> dict:
    """The cost of hiding the charge: hidden minus observed, per 48-hour episode, with the training seed as the unit."""
    from scipy import stats

    rng = np.random.default_rng(0) if rng is None else rng
    seen = [r for r in runs if r["observe_charge"]]
    hidden = [r for r in runs if not r["observe_charge"]]
    result = {"seeds": {"observed": [r["seed"] for r in seen], "hidden": [r["seed"] for r in hidden]}}
    for key in ("cost", "adjusted_cost"):
        s, h = np.array([r[key] for r in seen]), np.array([r[key] for r in hidden])
        boot = rng.choice(h, (20_000, len(h))).mean(axis=1) - rng.choice(s, (20_000, len(s))).mean(axis=1)
        welch = stats.ttest_ind(h, s, equal_var=False)
        ci = welch.confidence_interval(0.95)
        per_scenario = (np.mean([r[f"per_episode_{key}"] for r in hidden], axis=0)
                        - np.mean([r[f"per_episode_{key}"] for r in seen], axis=0))
        result[key] = {"observed": {"mean": float(s.mean()), "std": float(s.std()), "per_seed": s.tolist()},
                       "hidden": {"mean": float(h.mean()), "std": float(h.std()), "per_seed": h.tolist()},
                       "hidden_minus_observed": float(h.mean() - s.mean()),
                       "relative_to_observed": float((h.mean() - s.mean()) / s.mean()),
                       "bootstrap_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
                       "welch_ci": [float(ci.low), float(ci.high)], "welch_p": float(welch.pvalue),
                       "share_of_scenarios_hidden_pays_more": float((per_scenario > 0).mean())}
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules", action="store_true")
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--versus-rule", default="self_consumption", help="with --rules: the rule each run group is compared with")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with Pool(args.jobs) as pool:
        rules = pool.map(evaluate_rule, RULES) if args.rules else []
        runs = pool.map(evaluate_run, args.runs) if args.runs else []
    results = rules + runs
    output = {"policies": results}
    if {r["observe_charge"] for r in runs} == {True, False}:
        output["gate"] = gate(runs)
    if rules and runs:
        rule = next(r for r in rules if r["policy"] == args.versus_rule)
        groups = sorted({r["policy"] for r in runs})
        output["versus_rule"] = {g: versus_rule([r for r in runs if r["policy"] == g], rule) for g in groups}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    keys = ("cost", "adjusted_cost", "start_energy", "end_energy", "share_empty", "share_full",
            "median_hours_to_saturation", "share_saturated_within_6h", "r2_hidden_observation", "r2_clock_only")
    for r in results:
        print(f"{r['policy']}{'' if 'seed' not in r else ' s' + str(r['seed'])}: " + ", ".join(f"{k} {r[k]:.3f}" for k in keys))
    for key in ("gate", "versus_rule"):
        if key in output:
            print(json.dumps(output[key], indent=2))


if __name__ == "__main__":
    main()
