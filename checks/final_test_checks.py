"""Checks of evaluate_final_test.py --reports, the final test of step 0-4 runs:
python -m unittest checks.final_test_checks -v
"""
from pathlib import Path
import json
import shutil
import tempfile
import unittest

from omegaconf import OmegaConf
import torch

from checks.step0_checks import config, plain_ppo_config
from checks.step2_checks import PLAIN_SMALL
import evaluate_final_test as E
import main_battery_plane_contextual_ppo as training
import recurrent_encoders as R


class FinalTestChecks(unittest.TestCase):
    def test_final_test_grid_is_the_validation_grid_with_the_reserved_seeds(self):
        self.assertEqual(E.final_test_seeds(), [2000, 2001, 2002])
        for cfg in (plain_ppo_config("+experiment=S4a_encoder_bs256", "context.encoder_model=/x/encoder.pt", "context.encoder_tag=x"),
                    plain_ppo_config("+experiment=S0b_blind_bs1024"), config()):
            validation = OmegaConf.to_container(cfg.evaluation.reset_grid, resolve=True)
            test = OmegaConf.to_container(E.final_test_grid(cfg))
            self.assertEqual(test["eval_seeds"], [2000, 2001, 2002])
            self.assertEqual({k: v for k, v in test.items() if k != "eval_seeds"},
                             {k: v for k, v in validation.items() if k != "eval_seeds"})
        # A Contextual PPO config: exactly its own final_test grid
        cppo = config()
        self.assertEqual(OmegaConf.to_container(E.final_test_grid(cppo)), OmegaConf.to_container(cppo.final_test.reset_grid, resolve=True))
        # Overlap with the validation or the training seeds is refused
        for overrides in (["evaluation.reset_grid.eval_seeds=[1000,2001]"], ["training.seed=1995"]):
            with self.assertRaises(ValueError, msg=overrides):
                E.final_test_grid(plain_ppo_config("+experiment=S0a_blind_bs256", *overrides))

    def test_a_reported_run_is_reproduced_then_tested_once(self):
        from stable_baselines3 import PPO

        from checks.step0_checks import plain_ppo_script

        torch.set_num_threads(1)
        script = plain_ppo_script()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            torch.manual_seed(0)
            R.save_encoder(R.build_encoder("gru", hidden=16, head_arch=[8]).eval(), folder / "encoder.pt")
            cfg = plain_ppo_config("+experiment=S4a_encoder_bs256", *PLAIN_SMALL, f"context.encoder_model={folder / 'encoder.pt'}",
                                   "context.encoder_tag=small", "evaluation.reset_grid.eval_initial_battery=[0.3]")
            model = PPO("MlpPolicy", script.make_train_env(cfg), n_steps=8, batch_size=8, device="cpu",
                        policy_kwargs={"net_arch": [16]}, seed=0)
            model.learn(total_timesteps=32)
            run_dir = folder / "run"
            run_dir.mkdir()
            model.save(run_dir / "final_model.zip")
            OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
            validation = training.evaluate_schedule(model, cfg, cfg.evaluation.reset_grid)
            run = {"condition": "4a_encoder", "seed": 0, "config": str(run_dir / "config.yaml"), "run_directory": str(run_dir),
                   "late_mean_reward": 1.0, "final_validation": validation}
            report = folder / "report.json"
            report.write_text(json.dumps({"runs": [run]}))

            row = E.score_run(run)
            test = row["final_test"]
            self.assertFalse(row["reused"])
            self.assertTrue(test["validation_reproduced"])
            self.assertEqual(test["split"], "final_test")
            self.assertEqual(row["encoder"], "small")
            self.assertEqual(sorted({r["seed"] for r in test["reset_schedule"]}), [2000, 2001, 2002])
            self.assertEqual(len(test["episode_rewards"]), 3)
            self.assertNotEqual(test["episode_rewards"], validation["episode_rewards"])
            # The second time it is reused, never evaluated again or overwritten
            saved = (run_dir / "final_test.json").read_text()
            again = E.score_run(run)
            self.assertTrue(again["reused"])
            self.assertEqual((run_dir / "final_test.json").read_text(), saved)
            # A report whose final validation the model doesn't reproduce is refused
            tampered_dir = folder / "tampered"
            shutil.copytree(run_dir, tampered_dir)
            (tampered_dir / "final_test.json").unlink()
            tampered = {**run, "run_directory": str(tampered_dir),
                        "final_validation": {**validation, "episode_rewards": [r + 1.0 for r in validation["episode_rewards"]]}}
            with self.assertRaisesRegex(RuntimeError, "don't reproduce"):
                E.score_run(tampered)
            self.assertFalse((tampered_dir / "final_test.json").exists())
            # The summary over the reports, and an existing output is refused
            output = folder / "final_test.json"
            E.main(["--reports", str(report), "--conditions", "4a_encoder", "--output", str(output), "--jobs", "1"])
            summary = json.loads(output.read_text())["summary"]["4a_encoder (small)"]
            self.assertEqual(summary["final_test_seed_scores"], [test["mean_reward"]])
            with self.assertRaises(SystemExit):
                E.main(["--reports", str(report), "--output", str(output)])
            with self.assertRaises(SystemExit):
                E.main(["--grid", str(folder), "--output", str(folder / "other.json")])


if __name__ == "__main__":
    unittest.main()
