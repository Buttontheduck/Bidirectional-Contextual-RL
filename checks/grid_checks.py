"""CPU checks: python -m unittest checks.grid_checks -v"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

import main_battery_plane_contextual_ppo as training
import run_grid


def config(letter, beta=None, extra=()):
    overrides = [f"+experiment={run_grid.EXPERIMENTS[letter]}", *extra]
    if beta is not None:
        overrides.append(f"selected_gaussian.nll_beta={beta}")
    with initialize_config_dir(version_base=None, config_dir=str(run_grid.ROOT / "config")):
        return compose(config_name="cppo", overrides=overrides)


class GridChecks(unittest.TestCase):
    def test_grid_changes_only_planned_learner_settings(self):
        expected = {
            "A": ("point", "log_variance", 0.0, False),
            "B": ("point", "log_variance", 0.0, True),
            "C": ("gaussian", "log_variance", 0.0, False),
            "D": ("gaussian", "log_variance", 0.5, False),
            "E": ("gaussian", "none", 0.5, False),
            "F": ("gaussian", "log_variance", 0.5, True),
        }
        reference = config("A")
        for letter, values in expected.items():
            with self.subTest(letter=letter):
                cfg = config(letter, beta=0.5)
                training.validate_config(cfg)
                self.assertEqual((cfg.cppo.context_head, cfg.cppo.context_variance_input,
                                  cfg.cppo.supervised.nll_beta, cfg.cppo.velocity_head.enabled), values)
                self.assertEqual(cfg.cppo.velocity_head.weight, 0.1)
                for section in ("training", "env", "evaluation", "final_test"):
                    self.assertEqual(cfg[section], reference[section])
                learner = OmegaConf.to_container(cfg.cppo, resolve=True)
                baseline = OmegaConf.to_container(reference.cppo, resolve=True)
                for item in (learner, baseline):
                    del item["context_head"], item["context_variance_input"]
                    del item["supervised"]["nll_beta"], item["velocity_head"]["enabled"]
                self.assertEqual(learner, baseline)

    def test_followups_require_explicit_selected_beta(self):
        for letter in "EF":
            with self.assertRaisesRegex(ValueError, "Select C or D"):
                training.validate_config(config(letter))
            for beta in (0.0, 0.5):
                cfg = config(letter, beta)
                training.validate_config(cfg)
                self.assertEqual(cfg.cppo.supervised.nll_beta, beta)
            with self.assertRaisesRegex(ValueError, "Select C or D"):
                training.validate_config(config(letter, 0.2))

    def test_held_out_seeds_cannot_overlap(self):
        for override in ("final_test.reset_grid.eval_seeds=[1000]",
                         "evaluation.reset_grid.eval_seeds=[0]",
                         "final_test.reset_grid.eval_seeds=[0]"):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "disjoint"):
                training.validate_config(config("A", extra=[override]))

    def test_selection_uses_all_seed_validation_returns(self):
        rows = [
            {"experiment": letter, "seed": seed, "validation": {"mean_reward": value},
             "test_return": 100000 if letter == "C" else -100000}
            for letter, values in (("C", [100, -50, -50]), ("D", [1, 2, 3]))
            for seed, value in enumerate(values)
        ]
        selected = run_grid.choose_gaussian(rows, [0, 1, 2])
        self.assertEqual(selected["selected_experiment"], "D")
        self.assertEqual(selected["nll_beta"], 0.5)
        self.assertEqual(selected["mean_validation_return"], {"C": 0, "D": 2})
        with self.assertRaisesRegex(ValueError, "every seed"):
            run_grid.choose_gaussian(rows[:-1], [0, 1, 2])
        duplicate = copy.deepcopy(rows)
        duplicate[-1]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "every seed"):
            run_grid.choose_gaussian(duplicate, [0, 1, 2])
        for row in rows:
            row["validation"]["mean_reward"] = 7
        self.assertEqual(run_grid.choose_gaussian(rows, [0, 1, 2])["selected_experiment"], "C")

    def test_mean_only_reaches_both_branches_but_variance_still_learns(self):
        torch.set_num_threads(1)
        cfg = config("E", beta=0.5, extra=[*run_grid.SMOKE_OVERRIDES, "training.total_timesteps=32"])
        env = training.make_train_env(cfg)
        try:
            model = training.build_model(cfg, env)
            obs = torch.as_tensor(env.reset())
            low_variance = torch.tensor([[0.4, 0.01], [0.6, 0.01]])
            high_variance = torch.tensor([[0.4, 2.0], [0.6, 5.0]])
            for left, right in zip(model.policy._branch_inputs(obs, low_variance),
                                   model.policy._branch_inputs(obs, high_variance)):
                torch.testing.assert_close(left, right)
                self.assertEqual(left.shape[-1], obs.shape[-1] + 1)
            projection = [m for m in model.policy.encoder.context_projection if isinstance(m, torch.nn.Linear)][-1]
            self.assertEqual(projection.out_features, 2)
            before = projection.weight[1].detach().clone()
            model.learn(total_timesteps=32)
            self.assertFalse(torch.equal(before, projection.weight[1].detach()))
            self.assertTrue(torch.isfinite(torch.tensor(model.logger.name_to_value["train/context_nll"])))
        finally:
            env.close()

    def test_resume_checks_artifacts_and_selected_beta(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            run = output / "E_s0" / "attempt"
            run.mkdir(parents=True)
            (run / "final_model.zip").write_bytes(b"checkpoint")
            validation = {"mean_reward": 1.0}
            run_grid.save_json(run / "final_validation.json", validation)
            result = {"experiment": "E", "seed": 0, "selected_beta": 0.5,
                      "run_dir": "E_s0/attempt", "validation": validation}
            run_grid.save_json(run.parent / "result.json", result)
            self.assertEqual(run_grid.completed(output, "E", 0, 0.5), result)
            with self.assertRaisesRegex(ValueError, "identity"):
                run_grid.completed(output, "E", 0, 0.0)
            (run / "final_model.zip").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                run_grid.completed(output, "E", 0, 0.5)

    def test_concurrent_launcher_limits_pools_and_uses_independent_processes(self):
        args = run_grid.parse_args(["--jobs", "4", "--threads", "4"])
        self.assertEqual(args.seeds, [0, 1, 2])
        for letter in "ABCDEF":
            for seed in args.seeds:
                command = run_grid.command_for(args, letter, seed, Path("/tmp/example"), 0.5)
                self.assertNotIn("-m", command)
                self.assertIn("training.num_threads=4", command)
                self.assertIn(f"training.seed={seed}", command)
        with patch.dict("os.environ", {"OMP_NUM_THREADS": "32", "OPENBLAS_NUM_THREADS": "32"}):
            env = run_grid.child_environment(4)
            self.assertEqual(env["OMP_NUM_THREADS"], "4")
            self.assertEqual(env["OPENBLAS_NUM_THREADS"], "4")


if __name__ == "__main__":
    unittest.main()
