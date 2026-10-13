"""CPU checks of step 1 (ENCODER_PLAN.md), noisy oracle: python -m unittest checks.step1_checks -v"""
import contextlib
import io
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

import main_battery_plane_contextual_ppo as training
from checks.step0_checks import SMALL, config, plain_ppo_config, plain_ppo_script
from step1_controls import NoisyBattery

PLAIN_SMALL = ["training.n_envs=2", "ppo.n_steps=8", "ppo.batch_size=8", "+env.kwargs.T=4"]


def noisy_cppo(*overrides):
    return config("+step1=cppo_noisy_oracle", *overrides, step0=False)


class Step1Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_noisy_oracle_configs_change_only_the_battery_noise(self):
        import report_step0

        script = plain_ppo_script()
        pairs = [("1c", noisy_cppo(), config(), "cppo", "S1c_noisy_oracle"),
                 ("1a", plain_ppo_config("+experiment=S1a_noisy_bs256"), plain_ppo_config("+experiment=S0a_observed_bs256"), "ppo",
                  "S1a_noisy_bs256"),
                 ("1b", plain_ppo_config("+experiment=S1b_noisy_bs1024"), plain_ppo_config("+experiment=S0b_observed_bs1024"), "ppo",
                  "S1b_noisy_bs1024")]
        protocols = {}
        for condition, cfg, step0, learner, name in pairs:
            with self.subTest(condition=condition):
                (training.validate_config if learner == "cppo" else script.validate_ppo)(cfg)
                for section in (learner, "training", "evaluation"):
                    self.assertEqual(cfg[section], step0[section])
                env, env0 = OmegaConf.to_container(cfg.env), OmegaConf.to_container(step0.env)
                self.assertEqual((env.pop("battery_noise"), env0.pop("battery_noise")), (0.06, 0.0))
                self.assertEqual(env, env0)
                self.assertEqual((cfg.wandb.project, cfg.wandb.run_name, cfg.wandb.group),
                                 ("cppo", f"{name}_noise0.06_gamma0.99_rs0.1_s0", f"{name}_noise0.06_gamma0.99_rs0.1"))
                protocols[condition] = report_step0.evaluation_protocol(cfg, condition)
        # The three noisy runs share one protocol, which differs from step 0's only in the noise
        self.assertEqual(protocols["1a"], protocols["1c"])
        self.assertEqual(protocols["1b"], protocols["1c"])
        step0_protocol = report_step0.evaluation_protocol(config(), "0c")
        self.assertEqual({key for key in step0_protocol if step0_protocol[key] != protocols["1c"][key]}, {"battery_noise"})
        with self.assertRaisesRegex(ValueError, "battery_noise > 0"):
            report_step0.evaluation_protocol(noisy_cppo("env.battery_noise=0"), "1c")
        with self.assertRaisesRegex(ValueError, "no battery noise"):
            report_step0.evaluation_protocol(config("env.battery_noise=0.06"), "0c")
        # Noise only on a visible battery, and never negative
        for check, cfg in ((training.validate_config, config("env.battery_noise=0.05", step0=False)),
                           (script.validate_ppo, plain_ppo_config("env.battery_noise=0.05"))):
            with self.assertRaisesRegex(ValueError, "observation_mode=observed"):
                check(cfg)
        with self.assertRaisesRegex(ValueError, "battery_noise must be"):
            training.validate_config(noisy_cppo("env.battery_noise=-0.1"))

    def test_wrapper_adds_noise_of_the_right_size_and_leaves_the_flights_alone(self):
        sigma = 0.06
        plain = training.bp.BatteryPlaneEnv(observation_mode="observed")
        noisy = NoisyBattery(training.bp.BatteryPlaneEnv(observation_mode="observed"), sigma)
        rng = np.random.default_rng(0)
        errors = []
        clean_obs, _ = plain.reset(seed=7)
        noisy_obs, info = noisy.reset(seed=7)
        for step in range(3000):
            # Same flights: the noise has its own stream, so speed noise and start states are unchanged
            np.testing.assert_array_equal(noisy_obs[:-1], clean_obs[:-1])
            errors.append(noisy_obs[-1] - clean_obs[-1])
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            clean_obs, _, done, _, clean_info = plain.step(action)
            noisy_obs, _, noisy_done, _, info = noisy.step(action)
            self.assertEqual(done, noisy_done)
            self.assertAlmostEqual(float(clean_obs[-1]), info["b"], places=6)
            if done:
                clean_obs, _ = plain.reset()
                noisy_obs, _ = noisy.reset()
        errors = np.array(errors)
        self.assertLess(abs(errors.mean()), 4 * sigma / np.sqrt(len(errors)))
        self.assertAlmostEqual(errors.std(), sigma, delta=0.1 * sigma)
        # A seeded reset restarts the same noise; another seed draws other noise
        def batteries_seen(seed):
            seen = [noisy.reset(seed=seed)[0][-1]]
            return seen + [noisy.step(np.zeros(2, np.float32))[0][-1] for _ in range(20)]

        np.testing.assert_array_equal(batteries_seen(11), batteries_seen(11))
        self.assertFalse(np.array_equal(batteries_seen(11), batteries_seen(12)))
        with self.assertRaisesRegex(ValueError, "observed-mode"):
            NoisyBattery(training.bp.BatteryPlaneEnv(observation_mode="hidden"), sigma)
        with self.assertRaisesRegex(ValueError, "positive"):
            NoisyBattery(training.bp.BatteryPlaneEnv(observation_mode="observed"), 0.0)

    def test_both_scripts_add_the_noise_to_training_and_validation(self):
        script = plain_ppo_script()
        cases = [("cppo", training, noisy_cppo(*SMALL), config(*SMALL)),
                 ("ppo", script, plain_ppo_config("+experiment=S1a_noisy_bs256", *PLAIN_SMALL),
                  plain_ppo_config("+experiment=S0a_observed_bs256", *PLAIN_SMALL))]
        for learner, module, noisy_cfg, clean_cfg in cases:
            with self.subTest(learner=learner):
                for cfg, wrapped in ((noisy_cfg, True), (clean_cfg, False)):
                    env = module.make_train_env(cfg)
                    try:
                        self.assertEqual(env.env_is_wrapped(NoisyBattery), [wrapped] * 2)
                    finally:
                        env.close()
                schedule = list(training.iter_eval_resets(noisy_cfg.evaluation.reset_grid))
                flights = {}
                for name, cfg in (("noisy", noisy_cfg), ("clean", clean_cfg)):
                    eval_env, scheduler, raw_env = module.make_eval_env(cfg, schedule)
                    try:
                        runs = []
                        for _ in range(2):
                            # Two evaluations of the first validation flight
                            scheduler.rewind()
                            observations = [eval_env.reset()[0]]
                            for _ in range(5):
                                observations.append(eval_env.step(np.zeros((1, 2), np.float32))[0][0])
                            runs.append(np.array(observations))
                        flights[name] = runs
                    finally:
                        eval_env.close()
                np.testing.assert_array_equal(flights["noisy"][0], flights["noisy"][1])
                np.testing.assert_array_equal(flights["noisy"][0][:, :-1], flights["clean"][0][:, :-1])
                self.assertFalse(np.array_equal(flights["noisy"][0][:, -1], flights["clean"][0][:, -1]))

    def test_cppo_noisy_oracle_never_reads_the_true_battery(self):
        cfg = noisy_cppo(*SMALL, "env.battery_noise=0.12")
        env = training.make_train_env(cfg)
        try:
            model = training.build_model(cfg, env)
            model.learn(total_timesteps=32)
            buffer = model.rollout_buffer
            seen, true = buffer.observations[..., -1], buffer.contexts[..., 0]
            self.assertFalse(np.any(seen == true))
            self.assertLess(np.abs(seen - true).max(), 6 * 0.12)
            # The oracle encoder passes on what the policy sees: the noisy battery
            obs = torch.as_tensor(np.array(buffer.observations[0]))
            contexts, _ = model.policy.encode(obs, model.policy.initial_state(obs.shape[0]))
            torch.testing.assert_close(contexts, obs[:, -1:])
        finally:
            env.close()

    def test_fly_with_noise_evaluates_finished_policies_without_training(self):
        import json
        from pathlib import Path
        import tempfile

        import report_step0
        import step1_fly_with_noise as fly
        from checks.step0_reporting_checks import report_config

        cfg = report_config("0c")
        env = training.make_train_env(cfg)
        try:
            model = training.build_model(cfg, env)
            # Synthetic completion metadata, for testing artifact loading only
            model.num_timesteps = report_step0.FINAL_TIMESTEPS
            with tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                model.save(folder / "final_model")
                OmegaConf.save(cfg, folder / "config.yaml", resolve=True)
                (folder / "runs.json").write_text(json.dumps([{"condition": "0c", "status": "done", "run_dir": str(folder)}]))
                self.assertEqual(fly.selected_runs([folder / "runs.json"], 0.1), [("0c", str(folder))])
                self.assertEqual(fly.selected_runs([folder / "runs.json"], 0.01), [])
                result = fly.fly_run("0c", str(folder), [0.0, 0.06])
                plain = training.evaluate_schedule(training.ContextualPPO.load(folder / "final_model.zip", device="cpu"), cfg,
                                                   cfg.evaluation.reset_grid)
                saved = training.ContextualPPO.load(folder / "final_model.zip", device="cpu")
            self.assertEqual(set(result["flights"]), {"0", "0.06"})
            self.assertEqual((result["condition"], result["seed"], result["reward_scale"]), ("0c", 0, 0.1))
            # Noise 0 is the ordinary validation; nothing was trained
            self.assertEqual(result["flights"]["0"]["mean_reward"], plain["mean_reward"])
            for name, param in model.policy.state_dict().items():
                torch.testing.assert_close(param, saved.policy.state_dict()[name])
        finally:
            env.close()

    def test_late_switch_configs_continue_the_noisy_oracle_from_a_saved_model(self):
        import report_step0
        from step1_controls import switch_source_mismatches

        script = plain_ppo_script()
        cases = [("1c_switch", config("+step1=cppo_late_switch", step0=False), noisy_cppo(), "cppo", training.validate_config,
                  "S1c_late_switch"),
                 ("1a_switch", plain_ppo_config("+experiment=S1a_late_switch_bs256"), plain_ppo_config("+experiment=S1a_noisy_bs256"),
                  "ppo", script.validate_ppo, "S1a_late_switch_bs256")]
        for condition, cfg, noisy, learner, validate, name in cases:
            with self.subTest(condition=condition):
                # The noisy oracle's settings, with 1M more steps from a saved model that must be named
                self.assertEqual((cfg.training.total_timesteps, cfg.training.init_model), (1_000_000, "REQUIRED"))
                for section in (learner, "env", "evaluation"):
                    self.assertEqual(cfg[section], noisy[section])
                self.assertEqual(cfg.wandb.run_name, f"{name}_noise0.06_gamma0.99_rs0.1_s0")
                with self.assertRaisesRegex(ValueError, "init_model must name"):
                    validate(cfg)
                named = OmegaConf.merge(cfg, {"training": {"init_model": "/runs/0x/final_model.zip"}})
                validate(named)
                self.assertEqual(report_step0.evaluation_protocol(named, condition)["schedule"], "switch")
                with self.assertRaisesRegex(ValueError, "must continue a saved step 0 model"):
                    report_step0.evaluation_protocol(OmegaConf.merge(cfg, {"training": {"init_model": None}}), condition)
        with self.assertRaisesRegex(ValueError, "must start from scratch"):
            report_step0.evaluation_protocol(OmegaConf.merge(noisy_cppo(), {"training": {"init_model": "/m.zip"}}), "1c")
        # A source differs only where the switch may differ; anything else is reported
        source = config()
        switch = OmegaConf.merge(config("+step1=cppo_late_switch", step0=False), {"training": {"init_model": "/m.zip"}})
        self.assertEqual(switch_source_mismatches(switch, source, "cppo"), [])
        self.assertEqual(switch_source_mismatches(switch, config("cppo.gamma=0.995"), "cppo"), ["cppo"])
        self.assertEqual(switch_source_mismatches(switch, config("training.seed=1"), "cppo"), ["training"])
        self.assertEqual(switch_source_mismatches(switch, config("env.battery_noise=0.03"), "cppo"),
                         ["the source trained with battery noise"])
        # How often it evaluates does not matter, the validation flights do
        def evaluated(*overrides):
            return OmegaConf.merge(config("+step1=cppo_late_switch", *overrides, step0=False), {"training": {"init_model": "/m.zip"}})

        self.assertEqual(switch_source_mismatches(evaluated("evaluation.frequency=4096", "evaluation.plots.frequency=8192"),
                                                  source, "cppo"), [])
        self.assertEqual(switch_source_mismatches(evaluated("evaluation.reset_grid.eval_seeds=[1000,1001]"), source, "cppo"),
                         ["evaluation.reset_grid"])
        self.assertEqual(switch_source_mismatches(evaluated("evaluation.deterministic=false"), source, "cppo"),
                         ["evaluation.deterministic"])

    def test_both_scripts_continue_a_saved_model_with_noise(self):
        from pathlib import Path
        import tempfile
        from stable_baselines3 import PPO

        script = plain_ppo_script()
        with tempfile.TemporaryDirectory() as folder:
            cases = [("cppo", training, config(*SMALL), lambda path: config("+step1=cppo_late_switch", *SMALL,
                      f'training.init_model="{path}"', step0=False)),
                     ("ppo", script, plain_ppo_config("+experiment=S0a_observed_bs256", *PLAIN_SMALL),
                      lambda path: plain_ppo_config("+experiment=S1a_late_switch_bs256", *PLAIN_SMALL, f'training.init_model="{path}"'))]
            for learner, module, source_cfg, switch_cfg in cases:
                with self.subTest(learner=learner):
                    run = Path(folder) / learner
                    run.mkdir()
                    env = module.make_train_env(source_cfg)
                    try:
                        if learner == "cppo":
                            model = training.build_model(source_cfg, env)
                        else:
                            model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, seed=0, device="cpu")
                        model.learn(total_timesteps=32)
                        model.save(run / "final_model")
                        saved = {k: v.clone() for k, v in model.policy.state_dict().items()}
                    finally:
                        env.close()
                    OmegaConf.save(source_cfg, run / "config.yaml", resolve=True)
                    cfg = switch_cfg(run / "final_model.zip")
                    (training.validate_config if learner == "cppo" else script.validate_ppo)(cfg)
                    path = module.initial_model_from(cfg)
                    self.assertEqual(path, run / "final_model.zip")
                    env = module.make_train_env(cfg)
                    try:
                        self.assertTrue(all(env.env_is_wrapped(NoisyBattery)))
                        continued = module.load_initial_model(cfg, env, path)
                        # The saved weights and optimizer state, continued from the switch
                        for name, value in continued.policy.state_dict().items():
                            torch.testing.assert_close(value, saved[name])
                        self.assertTrue(continued.policy.optimizer.state_dict()["state"])
                        continued.learn(total_timesteps=16, reset_num_timesteps=True)
                        self.assertEqual(continued.num_timesteps, 16)
                        self.assertTrue(any(not torch.equal(v, saved[k]) for k, v in continued.policy.state_dict().items()))
                    finally:
                        env.close()
                    # A model trained with other settings is refused
                    mismatched = OmegaConf.merge(cfg, {learner: {"gamma": 0.9}})
                    with self.assertRaisesRegex(ValueError, "trained with other settings"):
                        module.load_initial_model(mismatched, None, path)

    def test_launcher_finds_the_step0_models_to_continue(self):
        import json
        from pathlib import Path
        import tempfile

        import run_step0

        with tempfile.TemporaryDirectory() as folder:
            rows = []
            for condition, seed, scale in (("0c", 0, 0.1), ("0c", 1, 0.1), ("0c", 0, 0.01), ("0a", 0, 0.1)):
                run = Path(folder) / f"{condition}_{seed}_{scale}"
                run.mkdir()
                cfg = config(f"training.seed={seed}", f"cppo.reward_scale={scale}") if condition == "0c" else \
                    plain_ppo_config("+experiment=S0a_observed_bs256", f"training.seed={seed}", f"ppo.reward_scale={scale}")
                OmegaConf.save(cfg, run / "config.yaml", resolve=True)
                rows.append({"condition": condition, "status": "done", "run_dir": str(run)})
            runs_file = Path(folder) / "runs.json"
            runs_file.write_text(json.dumps(rows))
            sources = run_step0.switch_sources([runs_file], 0.1)
            self.assertEqual(sorted(sources), [("0a", 0), ("0c", 0), ("0c", 1)])
            self.assertEqual(sources[("0c", 1)], Path(folder) / "0c_1_0.1" / "final_model.zip")
            args = run_step0.parse_args(["--conditions", "1c_switch", "--seeds", "0", "1", "--battery-noises", "0", "0.03",
                                         "--from-runs", str(runs_file), "--output", "/tmp/switch_launch"])
            command = run_step0.command_for("1c_switch", 1, 1, args.output, battery_noise=0.03,
                                            init_model=run_step0.Launcher(args).init_model("1c_switch", 1))
            cfg = config(*[item for item in command[3:] if not item.startswith("hydra.")], "+step1=cppo_late_switch", step0=False)
            self.assertEqual((cfg.training.init_model, cfg.env.battery_noise, cfg.training.seed),
                             (str(Path(folder) / "0c_1_0.1" / "final_model.zip"), 0.03, 1))
            for argv in (["--conditions", "1c_switch", "--seeds", "2", "--from-runs", str(runs_file)],   # no seed 2 source
                         ["--conditions", "1c_switch"]):                                                 # no --from-runs
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    run_step0.parse_args(argv)

    def test_launcher_runs_the_noise_screen(self):
        import run_step0

        args = run_step0.parse_args(["--conditions", "1a", "1b", "1c", "--battery-noises", "0.03", "0.06", "0.12",
                                     "--output", "/tmp/step1_launch"])
        self.assertEqual(args.battery_noises, [0.03, 0.06, 0.12])
        command = run_step0.command_for("1c", 2, 2, args.output, battery_noise=0.12)
        cfg = noisy_cppo(*[item for item in command[3:] if not item.startswith("hydra.")])
        self.assertEqual((cfg.env.battery_noise, cfg.training.seed, cfg.wandb.run_name),
                         (0.12, 2, "S1c_noisy_oracle_noise0.12_gamma0.99_rs0.1_s2"))
        self.assertEqual(run_step0.launch_name("1b", 0, battery_noise=0.03), "1b_noise0.03_s0")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_step0.parse_args(["--conditions", "0c", "--battery-noises", "0.06"])
        # One report command per setting, whichever of reward scale and battery noise the launch set
        def done(condition, seed, scale, noise):
            return {"condition": condition, "seed": seed, "reward_scale": scale, "battery_noise": noise, "run_dir": f"/r/{condition}{seed}"}

        for scales, noises, expected in (([None], [0.03, 0.12], ["(battery noise 0.03)", "(battery noise 0.12)"]),
                                         ([0.1], [None], ["(reward scale 0.1)"]), ([None], [None], [":"]),
                                         ([0.1], [0.06], ["(reward scale 0.1, battery noise 0.06)"])):
            runs = [done(c, s, scale, noise) for scale in scales for noise in noises for c in ("1a", "1c") for s in (0, 1)]
            lines = run_step0.report_commands(runs, scales, noises, args.output)
            self.assertEqual(len(lines), len(expected))
            for line, label in zip(lines, expected):
                self.assertIn(label, line)
                self.assertEqual(line.count("--run "), 4)


if __name__ == "__main__":
    unittest.main()
