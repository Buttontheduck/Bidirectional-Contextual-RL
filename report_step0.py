"""Report Step 0 validation curves and frozen-final-checkpoint outcomes.

Pass each run's actual resolved configuration explicitly. Both training scripts save it
in the run directory as config.yaml (plain PPO: runs_ppo/<run id>/, CPPO: runs_cppo/<run id>/).
This script never evaluates the reserved final test. Condition A is experiment A (learned encoder, hidden mode)
rerun with step 0's settings (config/rerun/), and 1a, 1b, 1c are step 1's noisy oracle (0a, 0b, 0c seeing the battery
plus noise), and 1a_switch, 1b_switch, 1c_switch are step 1's late switch (the finished 0a, 0b, 0c models trained 1M
more steps with noise; scored on 0.5M-1M after the switch); each is scored the same way, in a report of its own setting
(the protocol includes the battery noise and the schedule). 1a_early, 1b_early, 1c_early see the noise only until
env.battery_noise_until (500k), then the true battery; scored like step 0, the final validation with the true battery.
1a_sharp ... 1c_linear see the noise early in every flight and the exact battery from flight step 350 (in-flight noise).
2a_ideal, 2b_ideal (and _std) are step 2's ideal context: plain PPO in hidden mode with the ideal estimator's battery
estimate (and its std) appended to the observation; each variant in a report of its own.
J0-J5 are the joint-training rungs (config/joint/): Contextual_PPO with a GRU encoder (cppo.encoder_arch=gru) trained
together with the policy at the A rerun's settings, each in a report of its own.
4a_encoder, 4b_encoder are step 4's frozen encoders: plain PPO in hidden mode (0a / 0b's learner) with a learned
encoder's battery estimate appended to the observation (context.encoder_model); each encoder in a report of its own.
0a_blind, 0b_blind, 0c_blind are the memoryless controls: 0a, 0b, 0c in hidden mode, without the battery and without
memory (0c_blind: CPPO with no context, cppo.context_source=none). They share A's protocol, so one report can hold A and
the three of them.

Example (repeat --run for every condition and seed):
    python report_step0.py --run 0a runs_ppo/<id>/config.yaml runs_ppo/<id> \
        --run 0c runs_cppo/<id>/config.yaml runs_cppo/<id> --output step0_report.json

Plain-PPO runs made with the old Env_BatteryPlane script have no config.yaml: export one with the same
overrides used for that individual run, including its seed and minibatch size:
    python ppo/main_battery_plane_ppo.py +experiment=p04_observed ppo.batch_size=256 \
        training.total_timesteps=2000000 training.seed=0 --cfg job --resolve > /path/to/resolved_s0.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
import torch

from context_wrapper import context_protocol
import main_battery_plane_contextual_ppo as training
from contextual_ppo.torch_layers import GRURolloutEncoder, XLSTMRolloutEncoder


CONDITIONS = ("0a", "0b", "0c", "A", "1a", "1b", "1c", "1a_switch", "1b_switch", "1c_switch", "1a_early", "1b_early", "1c_early",
              "1a_sharp", "1b_sharp", "1c_sharp", "1a_linear", "1b_linear", "1c_linear", "0a_blind", "0b_blind", "0c_blind",
              "2a_ideal", "2b_ideal", "2a_ideal_std", "2b_ideal_std", "4a_encoder", "4b_encoder", "J0", "J1", "J2", "J3", "J4", "J5")
# Joint training with a GRU encoder: hidden mode, Contextual_PPO's learned encoder, like A but with the GRU
JOINT = ("J0", "J1", "J2", "J3", "J4", "J5")
# Memoryless controls: the step 0 run each condition repeats in hidden mode, without the battery and without memory
BLIND = {"0a_blind": "0a", "0b_blind": "0b", "0c_blind": "0c"}
# Step 2's ideal context: the step 0 learner each condition uses in hidden mode, with the ideal estimator's estimate
# appended (context_wrapper.py), and whether its std is appended too
IDEAL_CONTEXT = {"2a_ideal": ("0a", False), "2b_ideal": ("0b", False), "2a_ideal_std": ("0a", True), "2b_ideal_std": ("0b", True)}
# Step 4's frozen encoders: the step 0 learner each condition uses in hidden mode, with a learned encoder's estimate appended
ENCODER_CONTEXT = {"4a_encoder": "0a", "4b_encoder": "0b"}
# Step 1's noisy oracle: the step 0 run each condition repeats with noise on the battery
NOISY_REPEATS = {"1a": "0a", "1b": "0b", "1c": "0c"}
# Step 1's late switch: the step 0 run whose final model each condition keeps training
SWITCH_SOURCES = {"1a_switch": "0a", "1b_switch": "0b", "1c_switch": "0c"}
# Step 1's early noise: the step 0 run each condition repeats with noise on the battery until a step, then the true one
EARLY_NOISE = {"1a_early": "0a", "1b_early": "0b", "1c_early": "0c"}
# Step 1's in-flight noise: the step 0 run each condition repeats with noise early in every flight (profile: sharp, linear)
INFLIGHT_NOISE = {f"1{x}_{profile}": (f"0{x}", profile) for x in "abc" for profile in ("sharp", "linear")}
EVAL_FREQUENCY = 50_000
# Requested steps, scored window and the steps of a finished run (complete rollouts of 8 x 512): runs from scratch
# (step 0, A, noisy oracle) and the late switch, whose steps count from the switch
SCHEDULES = {
    "step0": {"total": 2_000_000, "late": (1_500_000, 2_000_000), "final": 2_002_944},
    "switch": {"total": 1_000_000, "late": (500_000, 1_000_000), "final": 1_003_520},
}
LATE_START, LATE_END = SCHEDULES["step0"]["late"]
EXPECTED_LATE_STEPS = np.arange(LATE_START, LATE_END + 1, EVAL_FREQUENCY)
FINAL_TIMESTEPS = SCHEDULES["step0"]["final"]
SHARED_PPO_SETTINGS = (
    "learning_rate", "n_steps", "n_epochs", "gamma", "gae_lambda", "clip_range", "clip_range_vf",
    "normalize_advantage", "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "net_arch",
    "activation_fn", "ortho_init", "log_std_init", "reward_scale",
)
VALIDATION_GRID = {
    "eval_seeds": [1000, 1001, 1002],
    "eval_initial_battery": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "eval_vx_obs_noise": [5.0],
    "eval_y0": None,
    "eval_vx0": None,
}


def run_kind(condition: str) -> str:
    """The step 0 run (0a, 0b, 0c) a step 1 condition repeats with some battery noise; A and step 0 runs themselves."""
    if condition in INFLIGHT_NOISE:
        return INFLIGHT_NOISE[condition][0]
    if condition in IDEAL_CONTEXT:
        return IDEAL_CONTEXT[condition][0]
    if condition in ENCODER_CONTEXT:
        return ENCODER_CONTEXT[condition]
    if condition in JOINT:
        return "J"
    return (NOISY_REPEATS.get(condition) or SWITCH_SOURCES.get(condition) or EARLY_NOISE.get(condition)
            or BLIND.get(condition) or condition)


def evaluation_protocol(cfg, condition: str) -> dict:
    """Reject mislabeled controls or a different Step 0 validation protocol."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    # A noisy oracle repeats a step 0 run with noise on the visible battery
    kind = run_kind(condition)
    noise = float(cfg.env.get("battery_noise", 0.0))
    until = cfg.env.get("battery_noise_until")
    flight = cfg.env.get("battery_noise_flight")
    noisy_conditions = {*NOISY_REPEATS, *EARLY_NOISE, *INFLIGHT_NOISE}
    if condition in noisy_conditions and noise <= 0:
        raise ValueError(f"{condition} requires env.battery_noise > 0")
    if condition not in noisy_conditions and condition not in SWITCH_SOURCES and noise > 0:
        raise ValueError(f"{condition} requires no battery noise")
    expected_flight = INFLIGHT_NOISE[condition][1] if condition in INFLIGHT_NOISE else None
    if flight != expected_flight:
        raise ValueError(f"{condition} requires env.battery_noise_flight={expected_flight}, got {flight}")
    # A frozen context appended to the observation (context_wrapper.py): the ideal estimate in step 2, none elsewhere
    context = context_protocol(cfg)
    if condition in IDEAL_CONTEXT:
        include_std = IDEAL_CONTEXT[condition][1]
        if context is None or context["source"] != "ideal" or context["include_std"] != include_std:
            raise ValueError(f"{condition} requires context.source=ideal with context.include_std={include_std}")
    elif condition in ENCODER_CONTEXT:
        if context is None or context["source"] != "encoder" or not context.get("encoder_model"):
            raise ValueError(f"{condition} requires context.source=encoder with context.encoder_model")
    elif context is not None:
        raise ValueError(f"{condition} requires no frozen context (context.source=none)")
    if (until is not None) != (condition in EARLY_NOISE):
        raise ValueError(f"{condition} " + ("requires env.battery_noise_until (noise first, then the true battery)"
                                            if condition in EARLY_NOISE else "must not switch the battery noise off"))
    # A late switch continues a saved step 0 model (noise 0 is its control); every other run starts from scratch
    schedule = "switch" if condition in SWITCH_SOURCES else "step0"
    continues = cfg.training.get("init_model") not in (None, "REQUIRED")
    if continues != (schedule == "switch"):
        raise ValueError(f"{condition} " + ("must continue a saved step 0 model (training.init_model)" if schedule == "switch"
                                            else "must start from scratch (no training.init_model)"))
    # A, the memoryless controls and the ideal context see no battery
    mode = ("hidden" if kind in ("A", "J") or condition in BLIND or condition in IDEAL_CONTEXT or condition in ENCODER_CONTEXT
            else "observed")
    if cfg.env.observation_mode != mode:
        raise ValueError(f"Step 0 requires env.observation_mode={mode} for {condition}")
    if not cfg.evaluation.deterministic or int(cfg.evaluation.frequency) != EVAL_FREQUENCY:
        raise ValueError("Step 0 requires deterministic evaluation every 50,000 transitions")
    grid = OmegaConf.to_container(cfg.evaluation.reset_grid, resolve=True)
    if {key: grid.get(key) for key in VALIDATION_GRID} != VALIDATION_GRID:
        raise ValueError("Step 0 requires the fixed validation grid (1000–1002, battery 0.2–0.7, noise 5)")
    if int(cfg.training.total_timesteps) != SCHEDULES[schedule]["total"]:
        raise ValueError(f"{condition} requires {SCHEDULES[schedule]['total']:,} requested training transitions")
    if int(cfg.training.n_envs) != 8:
        raise ValueError("Step 0 requires eight training environments")
    if kind == "0c":
        source = "none" if condition in BLIND else "oracle"
        if "cppo" not in cfg or cfg.cppo.get("context_source") != source:
            raise ValueError(f"{condition} requires CPPO with context_source={source}")
        training.validate_config(cfg)
        if int(cfg.cppo.n_minibatches) != 4:
            raise ValueError(f"{condition} requires four whole-stream minibatches")
    elif kind == "A":
        c = cfg.get("cppo")
        if (c is None or c.get("context_source", "encoder") != "encoder" or c.get("encoder_arch", "xlstm") != "xlstm"
                or c.encoder_objective != "supervised" or c.context_head != "point" or c.velocity_head.enabled
                or int(c.n_minibatches) != 4):
            raise ValueError("A requires CPPO with experiment A's learned point encoder (the xLSTM) and four whole-stream minibatches")
        training.validate_config(cfg)
    elif kind == "J":
        c = cfg.get("cppo")
        if (c is None or c.get("context_source", "encoder") != "encoder" or c.get("encoder_arch") != "gru"
                or c.encoder_objective != "supervised" or int(c.n_minibatches) != 4):
            raise ValueError(f"{condition} requires CPPO with a supervised GRU encoder (cppo.encoder_arch=gru) and four "
                             "whole-stream minibatches")
        training.validate_config(cfg)
    elif "cppo" in cfg or "ppo" not in cfg:
        raise ValueError(f"{condition} requires plain PPO")
    elif int(cfg.ppo.batch_size) != (256 if kind == "0a" else 1024):
        raise ValueError(f"Unexpected plain-PPO batch size for {condition}")
    learner = cfg.cppo if kind in ("0c", "A", "J") else cfg.ppo
    if int(learner.n_steps) != 512 or int(learner.n_epochs) != 10:
        raise ValueError("Step 0 requires 512 steps per environment and ten training epochs")
    env = training.bp.BatteryPlaneEnv(observation_mode=mode, **training.env_kwargs_from(cfg))
    try:
        # Normalize tuples/lists so equivalent environment settings compare equally.
        effective_env = json.loads(json.dumps(env.core._kw))
    finally:
        env.close()
    if effective_env["vx_obs_noise"] != 5.0:
        raise ValueError("Step 0 requires training speed noise 5")
    settings = OmegaConf.to_container(learner, resolve=True)
    # Configs from before reward_scale trained on unscaled rewards
    settings.setdefault("reward_scale", 1.0)
    return {"environment": effective_env, "context": context, "battery_noise": noise, "battery_noise_until": until,
            "battery_noise_flight": flight,
            "battery_noise_flight_step": cfg.env.get("battery_noise_flight_step") if flight else None, "schedule": schedule,
            "reset_grid": VALIDATION_GRID,
            "deterministic": True, "frequency": EVAL_FREQUENCY,
            "shared_ppo_settings": {key: settings[key] for key in SHARED_PPO_SETTINGS}}


def summarize_curve(path: Path, n_episodes: int = 18, schedule: str = "step0") -> dict:
    """Average the 11 scheduled late validation means of the schedule's window, including both endpoints."""
    late_start, late_end = SCHEDULES[schedule]["late"]
    expected = np.arange(late_start, late_end + 1, EVAL_FREQUENCY)
    with np.load(path, allow_pickle=False) as saved:
        timesteps = np.asarray(saved["timesteps"])
        rewards = np.asarray(saved["results"])
        lengths = np.asarray(saved["ep_lengths"])
    if timesteps.ndim != 1 or len(timesteps) == 0:
        raise ValueError(f"Invalid evaluation timesteps: {path}")
    if not np.isfinite(timesteps).all() or not np.equal(timesteps, timesteps.astype(np.int64)).all():
        raise ValueError(f"Evaluation timesteps must be finite integers: {path}")
    if np.any(np.diff(timesteps) <= 0):
        raise ValueError(f"Evaluation timesteps must be unique and increasing: {path}")
    if rewards.shape != (len(timesteps), n_episodes) or lengths.shape != rewards.shape:
        raise ValueError(f"Each evaluation must contain {n_episodes} episode rewards and lengths: {path}")
    if not np.isfinite(rewards).all() or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError(f"Non-finite returns or invalid episode lengths: {path}")
    late = (timesteps >= late_start) & (timesteps <= late_end)
    if not np.array_equal(timesteps[late], expected):
        raise ValueError(f"Need every 50k evaluation from {late_start:,} through {late_end:,} ({len(expected)} points): {path}")
    means = rewards.mean(axis=1)
    return {
        "late_mean_reward": float(means[late].mean()),
        "late_evaluation_timesteps": timesteps[late].astype(int).tolist(),
        "late_evaluation_mean_rewards": means[late].tolist(),
        "curve": [
            {"timesteps": int(step), "mean_reward": float(mean),
             "std_reward": float(np.std(episode_rewards)),
             "episode_rewards": episode_rewards.tolist(), "episode_lengths": episode_lengths.astype(int).tolist()}
            for step, mean, episode_rewards, episode_lengths in zip(timesteps, means, rewards, lengths)
        ],
    }


def summarize_conditions(runs: list[dict]) -> dict:
    """Give each training seed equal weight, with no pooling of evaluations."""
    identities = [(run["condition"], run["seed"]) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate condition/training-seed pair")
    summary = {}
    for condition in sorted({run["condition"] for run in runs}):
        rows = sorted((run for run in runs if run["condition"] == condition), key=lambda run: run["seed"])
        scores = [run["late_mean_reward"] for run in rows]
        summary[condition] = {
            "seeds": [run["seed"] for run in rows], "n_seeds": len(rows),
            "late_seed_scores": scores, "late_mean_reward": float(np.mean(scores)),
            "late_std_across_seeds": float(np.std(scores)),
        }
        if all("final_validation" in run for run in rows):
            for metric in ("mean_reward", "completion_rate", "ground_crash_rate", "battery_empty_rate"):
                values = [run["final_validation"][metric] for run in rows]
                summary[condition][f"final_{metric}"] = float(np.mean(values))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=3, action="append", required=True,
                        metavar=("CONDITION", "CONFIG_YAML", "RUN_DIRECTORY"))
    parser.add_argument("--output", type=Path, required=True, help="Combined JSON report; run artifacts stay unchanged")
    args = parser.parse_args()
    prepared, identities, protocol = [], set(), None
    for condition, config_path, directory in args.run:
        config_path, directory = Path(config_path).resolve(), Path(directory).resolve()
        cfg = OmegaConf.load(config_path)
        candidate = evaluation_protocol(cfg, condition)
        if protocol is not None and candidate != protocol:
            raise ValueError("Run configurations have different evaluation/environment protocols")
        protocol = candidate
        identity = (condition, int(cfg.training.seed))
        if identity in identities:
            raise ValueError(f"Duplicate condition/training-seed pair: {identity}")
        identities.add(identity)
        checkpoint = directory / "final_model.zip"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        curve = summarize_curve(directory / "eval" / "evaluations.npz", schedule=candidate["schedule"])
        prepared.append((condition, config_path, directory, cfg, curve))

    torch.set_num_threads(1)
    runs = []
    for condition, config_path, directory, cfg, curve in prepared:
        algorithm = training.ContextualPPO if "cppo" in cfg else PPO
        model = algorithm.load(directory / "final_model.zip", device="cpu")
        if model.num_timesteps != SCHEDULES[protocol["schedule"]]["final"] or model.seed != int(cfg.training.seed):
            raise ValueError(f"Checkpoint training duration or seed does not match the supplied config: {directory}")
        if (model.n_envs, model.n_steps, model.n_epochs) != (8, 512, 10):
            raise ValueError(f"Checkpoint rollout/epoch settings do not match Step 0: {directory}")
        kind = run_kind(condition)
        if kind == "0c":
            policy = training.NoContextPolicy if condition in BLIND else training.TrueBatteryPolicy
            if model.n_minibatches != 4 or not isinstance(model.policy, policy):
                raise ValueError(f"{condition} checkpoint must use {policy.__name__} and four minibatches: {directory}")
        elif kind == "A":
            if model.n_minibatches != 4 or not isinstance(model.policy.encoder, XLSTMRolloutEncoder):
                raise ValueError(f"A checkpoint must use the learned xLSTM encoder and four minibatches: {directory}")
        elif kind == "J":
            if model.n_minibatches != 4 or not isinstance(model.policy.encoder, GRURolloutEncoder):
                raise ValueError(f"{condition} checkpoint must use the learned GRU encoder and four minibatches: {directory}")
        elif model.batch_size != int(cfg.ppo.batch_size):
            raise ValueError(f"Checkpoint batch size does not match the supplied config: {directory}")
        validation = training.evaluate_schedule(model, cfg, cfg.evaluation.reset_grid)
        validation["split"] = "validation"
        runs.append({"condition": condition, "seed": int(cfg.training.seed),
                     "algorithm": algorithm.__name__, "config": str(config_path), "run_directory": str(directory),
                     **curve, "final_validation": validation})
        print(f"{condition} seed {cfg.training.seed}: late={curve['late_mean_reward']:.3f}, "
              f"final={validation['mean_reward']:.3f}")
    late_start, late_end = SCHEDULES[protocol["schedule"]]["late"]
    report = {
        "scoring": {"start": late_start, "end": late_end, "endpoints_inclusive": True,
                    "evaluations_per_seed": len(range(late_start, late_end + 1, EVAL_FREQUENCY)),
                    "criterion": "Mean of scheduled validation episode-return means; equal weight per training seed",
                    "spread": "Population standard deviation across supplied training seeds",
                    "outcomes": "Final-checkpoint validation only; battery-empty takes precedence over simultaneous ground contact"},
        "protocol": protocol, "runs": runs, "conditions": summarize_conditions(runs),
        "complete_nine_run_screen": identities == {(condition, seed) for condition in ("0a", "0b", "0c") for seed in (0, 1, 2)},
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
