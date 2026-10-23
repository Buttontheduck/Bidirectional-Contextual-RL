"""Checks of ood_evaluation.py, the out-of-distribution evaluation: python -m unittest checks.ood_checks -v"""
from pathlib import Path
import json
import tempfile
import unittest

from omegaconf import OmegaConf
import torch

from checks.step0_checks import plain_ppo_config
from checks.step2_checks import PLAIN_SMALL
import main_battery_plane_contextual_ppo as training
import ood_evaluation as O


class OodChecks(unittest.TestCase):
    def test_settings_and_seeds(self):
        self.assertEqual(O.OOD_SEEDS, [3000, 3001, 3002])
        self.assertEqual(set(O.SETTINGS), {"nominal", "battery_outside", "noise_0", "noise_10", "capacity_85", "capacity_120",
                                           "motor_weaker", "motor_stronger", "battery_sweep"})
        self.assertNotIn("battery_sweep", O.DEFAULT_SETTINGS)
        for setting in O.SETTINGS:
            grid = O.setting_grid(setting)
            self.assertEqual(grid["eval_seeds"], [3000, 3001, 3002, 3003, 3004] if setting == "battery_sweep" else [3000, 3001, 3002])
        self.assertEqual([b for b in O.SWEEP_BATTERIES if 0.2 < b < 0.7], [])
        self.assertFalse(set(O.SETTINGS["battery_outside"][0]) & set(O.NOMINAL_BATTERIES))
        O.check_seeds(plain_ppo_config("+experiment=S0a_blind_bs256", "training.seed=7"))
        for overrides in (["training.seed=30"], ["training.seed=2999"], ["evaluation.reset_grid.eval_seeds=[3001]"],
                          ["evaluation.reset_grid.eval_seeds=[3004]"]):
            with self.assertRaises(ValueError, msg=overrides):
                O.check_seeds(plain_ppo_config("+experiment=S0a_blind_bs256", *overrides))

    def test_the_plane_changes_the_estimate_keeps_the_training_physics(self):
        cfg = plain_ppo_config("+experiment=S2a_ideal_bs256", "context.b_points=11", "context.vx_points=31")
        schedule = list(training.iter_eval_resets(OmegaConf.create(O.setting_grid("capacity_85"))))
        env, _, raw = O.make_env(cfg, schedule, O.SETTINGS["capacity_85"][2])
        self.assertEqual(raw.core.E, 85.0)
        estimator = env.source.estimators[0]
        self.assertEqual(estimator.env_kwargs, training.env_kwargs_from(cfg))
        self.assertNotIn("E_hover_s", estimator.env_kwargs)
        env.close()
        env, _, raw = O.make_env(cfg, schedule, O.SETTINGS["motor_stronger"][2])
        self.assertEqual((raw.core.h_min, raw.core.h_max), (12.5, 19.0))
        env.close()

    def test_a_run_reproduces_its_validation_then_flies_every_setting(self):
        from stable_baselines3 import PPO

        from checks.step0_checks import plain_ppo_script

        torch.set_num_threads(1)
        script = plain_ppo_script()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cfg = plain_ppo_config("+experiment=S0a_blind_bs256", *PLAIN_SMALL, "evaluation.reset_grid.eval_initial_battery=[0.3]")
            model = PPO("MlpPolicy", script.make_train_env(cfg), n_steps=8, batch_size=8, device="cpu",
                        policy_kwargs={"net_arch": [16]}, seed=0)
            model.learn(total_timesteps=32)
            run_dir = folder / "run"
            run_dir.mkdir()
            model.save(run_dir / "final_model.zip")
            OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
            validation = training.evaluate_schedule(model, cfg, cfg.evaluation.reset_grid)
            run = {"condition": "0a_blind", "seed": 0, "config": str(run_dir / "config.yaml"), "run_directory": str(run_dir),
                   "late_mean_reward": 1.0, "final_validation": validation}
            settings = ["nominal", "battery_outside", "capacity_85"]
            (folder / "out").mkdir()
            row = O.score_run((run, str(folder / "out"), settings))
            self.assertTrue(row["validation_reproduced"])
            self.assertEqual(set(row["settings"]), set(settings))
            self.assertEqual(len(row["settings"]["battery_outside"]["episode_rewards"]), 15)
            self.assertEqual(len(row["settings"]["nominal"]["episode_rewards"]), 18)
            flights = sorted({b0 for b0 in row["settings"]["battery_outside"]["grid"]["eval_initial_battery"]})
            self.assertEqual(flights, [0.1, 0.15, 0.75, 0.8, 0.9])
            # Reused the second time
            saved = (folder / "out" / "0a_blind_s0.json").read_text()
            self.assertEqual(O.score_run((run, str(folder / "out"), settings)), json.loads(saved))
            # A run that doesn't reproduce its validation is refused
            tampered = {**run, "final_validation": {**validation, "episode_rewards": [r + 1 for r in validation["episode_rewards"]]}}
            (folder / "out2").mkdir()
            with self.assertRaisesRegex(RuntimeError, "don't reproduce"):
                O.score_run((tampered, str(folder / "out2"), settings))


if __name__ == "__main__":
    unittest.main()
