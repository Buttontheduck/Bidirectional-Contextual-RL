"""Report Step 0 validation curves and frozen-final-checkpoint outcomes.

Pass each run's actual resolved configuration explicitly. Both training scripts save it
in the run directory as config.yaml (plain PPO: runs_ppo/<run id>/, CPPO: runs_cppo/<run id>/).
This script never evaluates the reserved final test.

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

import main_battery_plane_contextual_ppo as training


LATE_START = 1_500_000
LATE_END = 2_000_000
EVAL_FREQUENCY = 50_000
EXPECTED_LATE_STEPS = np.arange(LATE_START, LATE_END + 1, EVAL_FREQUENCY)
FINAL_TIMESTEPS = 2_002_944  # ceil(2M / (8 * 512)) complete rollouts
SHARED_PPO_SETTINGS = (
    "learning_rate", "n_steps", "n_epochs", "gamma", "gae_lambda", "clip_range", "clip_range_vf",
    "normalize_advantage", "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "net_arch",
    "activation_fn", "ortho_init", "log_std_init",
)
VALIDATION_GRID = {
    "eval_seeds": [1000, 1001, 1002],
    "eval_initial_battery": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "eval_vx_obs_noise": [5.0],
    "eval_y0": None,
    "eval_vx0": None,
}


def evaluation_protocol(cfg, condition: str) -> dict:
    """Reject mislabeled controls or a different Step 0 validation protocol."""
    if condition not in ("0a", "0b", "0c"):
        raise ValueError("condition must be 0a, 0b or 0c")
    if cfg.env.observation_mode != "observed":
        raise ValueError("Step 0 requires env.observation_mode=observed")
    if not cfg.evaluation.deterministic or int(cfg.evaluation.frequency) != EVAL_FREQUENCY:
        raise ValueError("Step 0 requires deterministic evaluation every 50,000 transitions")
    grid = OmegaConf.to_container(cfg.evaluation.reset_grid, resolve=True)
    if {key: grid.get(key) for key in VALIDATION_GRID} != VALIDATION_GRID:
        raise ValueError("Step 0 requires the fixed validation grid (1000–1002, battery 0.2–0.7, noise 5)")
    if int(cfg.training.total_timesteps) != LATE_END:
        raise ValueError("Step 0 requires 2,000,000 requested training transitions")
    if int(cfg.training.n_envs) != 8:
        raise ValueError("Step 0 requires eight training environments")
    if condition == "0c":
        if "cppo" not in cfg or cfg.cppo.get("context_source") != "oracle":
            raise ValueError("0c requires CPPO with context_source=oracle")
        training.validate_config(cfg)
        if int(cfg.cppo.n_minibatches) != 4:
            raise ValueError("0c requires four whole-stream minibatches")
    elif "cppo" in cfg or "ppo" not in cfg:
        raise ValueError(f"{condition} requires plain PPO")
    elif int(cfg.ppo.batch_size) != (256 if condition == "0a" else 1024):
        raise ValueError(f"Unexpected plain-PPO batch size for {condition}")
    learner = cfg.cppo if condition == "0c" else cfg.ppo
    if int(learner.n_steps) != 512 or int(learner.n_epochs) != 10:
        raise ValueError("Step 0 requires 512 steps per environment and ten training epochs")
    env = training.bp.BatteryPlaneEnv(observation_mode="observed", **training.env_kwargs_from(cfg))
    try:
        # Normalize tuples/lists so equivalent environment settings compare equally.
        effective_env = json.loads(json.dumps(env.core._kw))
    finally:
        env.close()
    if effective_env["vx_obs_noise"] != 5.0:
        raise ValueError("Step 0 requires training speed noise 5")
    settings = OmegaConf.to_container(learner, resolve=True)
    return {"environment": effective_env, "reset_grid": VALIDATION_GRID,
            "deterministic": True, "frequency": EVAL_FREQUENCY,
            "shared_ppo_settings": {key: settings[key] for key in SHARED_PPO_SETTINGS}}


def summarize_curve(path: Path, n_episodes: int = 18) -> dict:
    """Average the 11 scheduled late validation means, including both endpoints."""
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
    late = (timesteps >= LATE_START) & (timesteps <= LATE_END)
    if not np.array_equal(timesteps[late], EXPECTED_LATE_STEPS):
        raise ValueError(f"Need every 50k evaluation from 1.5M through 2M (11 points): {path}")
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
        curve = summarize_curve(directory / "eval" / "evaluations.npz")
        prepared.append((condition, config_path, directory, cfg, curve))

    torch.set_num_threads(1)
    runs = []
    for condition, config_path, directory, cfg, curve in prepared:
        algorithm = training.ContextualPPO if "cppo" in cfg else PPO
        model = algorithm.load(directory / "final_model.zip", device="cpu")
        if model.num_timesteps != FINAL_TIMESTEPS or model.seed != int(cfg.training.seed):
            raise ValueError(f"Checkpoint training duration or seed does not match the supplied config: {directory}")
        if (model.n_envs, model.n_steps, model.n_epochs) != (8, 512, 10):
            raise ValueError(f"Checkpoint rollout/epoch settings do not match Step 0: {directory}")
        if condition == "0c":
            if model.n_minibatches != 4 or not isinstance(model.policy, training.TrueBatteryPolicy):
                raise ValueError(f"0c checkpoint must use the oracle policy and four minibatches: {directory}")
        elif model.batch_size != int(cfg.ppo.batch_size):
            raise ValueError(f"Checkpoint batch size does not match the supplied config: {directory}")
        validation = training.evaluate_schedule(model, cfg, cfg.evaluation.reset_grid)
        validation["split"] = "validation"
        runs.append({"condition": condition, "seed": int(cfg.training.seed),
                     "algorithm": algorithm.__name__, "config": str(config_path), "run_directory": str(directory),
                     **curve, "final_validation": validation})
        print(f"{condition} seed {cfg.training.seed}: late={curve['late_mean_reward']:.3f}, "
              f"final={validation['mean_reward']:.3f}")
    report = {
        "scoring": {"start": LATE_START, "end": LATE_END, "endpoints_inclusive": True,
                    "evaluations_per_seed": len(EXPECTED_LATE_STEPS),
                    "criterion": "Mean of scheduled validation episode-return means; equal weight per training seed",
                    "spread": "Population standard deviation across supplied training seeds",
                    "outcomes": "Final-checkpoint validation only; battery-empty takes precedence over simultaneous ground contact"},
        "protocol": protocol, "runs": runs, "conditions": summarize_conditions(runs),
        "complete_nine_run_screen": identities == {(condition, seed) for condition in ("0a", "0b", "0c") for seed in (0, 1, 2)},
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
