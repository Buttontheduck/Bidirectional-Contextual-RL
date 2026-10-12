"""Checks of Step 0 reporting without consuming any real run or final-test episode."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

import report_step0 as reporting


def report_config(condition="0a"):
    from checks.step0_checks import config

    cfg = config("+env.kwargs.T=4", "cppo.net_arch=[16]", "cppo.verbose=0")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if condition != "0c":
        cfg.ppo = {key: OmegaConf.to_container(cfg.cppo, resolve=True)[key]
                   for key in reporting.SHARED_PPO_SETTINGS}
        cfg.ppo.batch_size = 256 if condition == "0a" else 1024
        del cfg["cppo"]
    return cfg


class Step0ReportingChecks(unittest.TestCase):
    def save_curve(self, folder, timesteps, rewards=None):
        if rewards is None:
            rewards = np.repeat(np.arange(len(timesteps), dtype=float)[:, None], 18, axis=1)
        path = Path(folder) / "evaluations.npz"
        np.savez(path, timesteps=timesteps, results=rewards, ep_lengths=np.full_like(rewards, 700))
        return path

    def test_late_score_includes_both_boundaries_and_excludes_outside(self):
        timesteps = np.r_[1_450_000, reporting.EXPECTED_LATE_STEPS, 2_050_000]
        rewards = np.repeat(np.r_[9999, np.arange(11), 9999][:, None], 18, axis=1)
        rewards = rewards + np.arange(18)[None, :]
        with tempfile.TemporaryDirectory() as folder:
            result = reporting.summarize_curve(self.save_curve(folder, timesteps, rewards))
        self.assertEqual(result["late_mean_reward"], 13.5)
        self.assertEqual(result["late_evaluation_timesteps"], reporting.EXPECTED_LATE_STEPS.tolist())
        self.assertEqual(len(result["curve"]), 13)

    def test_missing_or_duplicate_late_points_are_rejected(self):
        for steps in (reporting.EXPECTED_LATE_STEPS[1:], reporting.EXPECTED_LATE_STEPS[:-1],
                      np.delete(reporting.EXPECTED_LATE_STEPS, 4),
                      np.r_[reporting.EXPECTED_LATE_STEPS, 2_000_000]):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    reporting.summarize_curve(self.save_curve(folder, steps))

    def test_invalid_episode_arrays_are_rejected(self):
        for rewards in (np.full((11, 18), np.nan), np.ones((11, 17))):
            with self.subTest(shape=rewards.shape), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    reporting.summarize_curve(self.save_curve(folder, reporting.EXPECTED_LATE_STEPS, rewards))

    def test_each_training_seed_has_equal_weight_and_duplicates_fail(self):
        runs = [{"condition": "0b", "seed": seed, "late_mean_reward": score}
                for seed, score in ((2, 30.0), (0, 0.0), (1, 0.0))]
        result = reporting.summarize_conditions(runs)["0b"]
        self.assertEqual(result["seeds"], [0, 1, 2])
        self.assertEqual(result["late_seed_scores"], [0, 0, 30])
        self.assertEqual(result["late_mean_reward"], 10.0)
        self.assertAlmostEqual(result["late_std_across_seeds"], np.sqrt(200))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reporting.summarize_conditions([*runs, runs[0]])

    def test_protocol_requires_noisy_observed_validation_and_correct_condition(self):
        cfg = report_config()
        protocol = reporting.evaluation_protocol(cfg, "0a")
        self.assertEqual(protocol["environment"]["vx_obs_noise"], 5)
        for field, value in (("env.observation_mode", "full"), ("evaluation.deterministic", False),
                             ("evaluation.reset_grid.eval_seeds", [2000, 2001, 2002]),
                             ("env.kwargs.vx_obs_noise", 0), ("ppo.batch_size", 1024),
                             ("training.n_envs", 32), ("ppo.n_steps", 128), ("ppo.n_epochs", 5)):
            changed = OmegaConf.create(OmegaConf.to_container(cfg))
            OmegaConf.update(changed, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                reporting.evaluation_protocol(changed, "0a")
        with self.assertRaisesRegex(ValueError, "0c requires CPPO"):
            reporting.evaluation_protocol(cfg, "0c")
        changed = OmegaConf.create(OmegaConf.to_container(cfg))
        changed.env.kwargs.T = 100
        self.assertNotEqual(protocol, reporting.evaluation_protocol(changed, "0a"))

    def test_cli_reloads_both_learners_and_only_evaluates_validation(self):
        cfg = report_config("0c")
        env = reporting.training.make_train_env(cfg)
        reporting.torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                arguments = ["report_step0.py"]
                for condition in ("0a", "0c"):
                    directory = folder / condition
                    (directory / "eval").mkdir(parents=True)
                    run_cfg = report_config(condition)
                    if condition == "0a":
                        model = reporting.PPO("MlpPolicy", env, seed=0, n_steps=512, batch_size=256,
                                              policy_kwargs={"net_arch": [16]}, device="cpu")
                    else:
                        model = reporting.training.build_model(run_cfg, env)
                    # Synthetic completion metadata, for testing artifact loading only.
                    model.num_timesteps = reporting.FINAL_TIMESTEPS
                    model.save(directory / "final_model")
                    OmegaConf.save(run_cfg, directory / "config.yaml", resolve=True)
                    self.save_curve(directory / "eval", reporting.EXPECTED_LATE_STEPS)
                    arguments.extend(["--run", condition, str(directory / "config.yaml"), str(directory)])
                output = folder / "report.json"
                arguments.extend(["--output", str(output)])
                with patch("sys.argv", arguments), patch.object(
                    reporting.training, "evaluate_schedule", wraps=reporting.training.evaluate_schedule
                ) as evaluate:
                    reporting.main()
                self.assertEqual(evaluate.call_count, 2)
                for call in evaluate.call_args_list:
                    self.assertEqual(list(call.args[2].eval_seeds), [1000, 1001, 1002])
                report = json.loads(output.read_text())
                self.assertEqual(set(report["conditions"]), {"0a", "0c"})
                self.assertFalse(report["complete_nine_run_screen"])
                for run in report["runs"]:
                    self.assertEqual(run["late_mean_reward"], 5.0)
                    self.assertEqual(len(run["final_validation"]["episode_outcomes"]), 18)
                    self.assertEqual(run["final_validation"]["split"], "validation")
                    self.assertFalse((Path(run["run_directory"]) / "final_validation.json").exists())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
