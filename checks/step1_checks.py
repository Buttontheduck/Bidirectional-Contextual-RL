"""CPU checks of step 1 (ENCODER_PLAN.md), noisy oracle: python -m unittest checks.step1_checks -v"""
import contextlib
import io
import unittest
from pathlib import Path

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

    def test_early_noise_configs_switch_to_the_true_battery_at_500k(self):
        import report_step0
        from step1_controls import battery_noise_at

        script = plain_ppo_script()
        cases = [("1c_early", config("+step1=cppo_early_noise", step0=False), noisy_cppo(), "cppo", training.validate_config,
                  "S1c_early_noise0.06_until500000"),
                 ("1a_early", plain_ppo_config("+experiment=S1a_early_noise_bs256"), plain_ppo_config("+experiment=S1a_noisy_bs256"),
                  "ppo", script.validate_ppo, "S1a_early_noise_bs256_noise0.06_until500000")]
        for condition, cfg, noisy, learner, validate, name in cases:
            with self.subTest(condition=condition):
                validate(cfg)
                # The noisy oracle from scratch, with the noise switched off at 500k
                for section in (learner, "training", "evaluation"):
                    self.assertEqual(cfg[section], noisy[section])
                env, noisy_env = OmegaConf.to_container(cfg.env), OmegaConf.to_container(noisy.env)
                self.assertEqual((env.pop("battery_noise_until"), noisy_env.pop("battery_noise_until")), (500_000, None))
                self.assertEqual(env, noisy_env)
                self.assertEqual(cfg.wandb.run_name, f"{name}_gamma0.99_rs0.1_s0")
                self.assertEqual([battery_noise_at(cfg, step) for step in (0, 499_999, 500_000, 2_000_000)], [0.06, 0.06, 0.0, 0.0])
                protocol = report_step0.evaluation_protocol(cfg, condition)
                noisy_protocol = report_step0.evaluation_protocol(noisy, condition.replace("_early", ""))
                self.assertEqual({key for key in protocol if protocol[key] != noisy_protocol[key]}, {"battery_noise_until"})
                self.assertEqual(protocol["schedule"], "step0")
        with self.assertRaisesRegex(ValueError, "requires env.battery_noise_until"):
            report_step0.evaluation_protocol(noisy_cppo(), "1c_early")
        with self.assertRaisesRegex(ValueError, "must not switch the battery noise off"):
            report_step0.evaluation_protocol(noisy_cppo("env.battery_noise_until=500000"), "1c")

    def test_noise_switch_turns_the_noise_off_in_training_and_validation(self):
        from stable_baselines3 import PPO
        from step1_controls import BatteryNoiseSwitch, noisy_batteries

        script = plain_ppo_script()
        cases = [("cppo", training, config("+step1=cppo_early_noise", *SMALL, step0=False)),
                 ("ppo", script, plain_ppo_config("+experiment=S1a_early_noise_bs256", *PLAIN_SMALL))]
        for learner, module, cfg in cases:
            with self.subTest(learner=learner):
                env = module.make_train_env(cfg)
                eval_env, _, _ = module.make_eval_env(cfg, list(training.iter_eval_resets(cfg.evaluation.reset_grid)))
                try:
                    if learner == "cppo":
                        model = training.build_model(cfg, env)
                    else:
                        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, seed=0, device="cpu")
                    self.assertEqual([w.sigma for w in noisy_batteries(env) + noisy_batteries(eval_env)], [0.06] * 3)
                    # Two rollouts of 2 envs x 8 steps; the switch falls inside the first
                    model.learn(total_timesteps=32, callback=[BatteryNoiseSwitch(8, eval_env)])
                    self.assertEqual([w.sigma for w in noisy_batteries(env) + noisy_batteries(eval_env)], [0.0] * 3)
                    if learner == "cppo":
                        # The second rollout, all after the switch: the policy saw exactly the true battery
                        buffer = model.rollout_buffer
                        np.testing.assert_array_equal(buffer.observations[..., -1], buffer.contexts[..., 0])
                finally:
                    env.close()
                    eval_env.close()

    def test_final_validation_sees_the_battery_of_the_last_step(self):
        from unittest.mock import patch

        class FixedPolicy:
            def __init__(self, steps):
                self.num_timesteps = steps

            def predict(self, obs, **kwargs):
                return np.tile([0.5, 0.0], (len(obs), 1)), None

        cfg = config("+step1=cppo_early_noise", *SMALL, "evaluation.reset_grid.eval_seeds=[1000]",
                     "evaluation.reset_grid.eval_initial_battery=[0.5]", step0=False)
        for steps, noise in ((100_000, 0.06), (2_002_944, 0.0)):
            with self.subTest(steps=steps), patch.object(training, "make_eval_env", wraps=training.make_eval_env) as made:
                training.evaluate_schedule(FixedPolicy(steps), cfg, cfg.evaluation.reset_grid)
                self.assertEqual(made.call_args.args[2], noise)

    def test_inflight_profiles_remove_the_noise_by_flight_step_350(self):
        def flight_errors(profile):
            env = NoisyBattery(training.bp.BatteryPlaneEnv(observation_mode="observed"), 0.12, profile, 350)
            obs, info = env.reset(seed=3)
            errors = [obs[-1] - info["b0"]]
            # A gentle pilot that completes the 700-step flight
            for _ in range(699):
                obs, _, done, _, info = env.step(np.array([0.3, 0.2], np.float32))
                errors.append(obs[-1] - info["b"])
                if done:
                    break
            return np.array(errors)

        constant, sharp, linear = (flight_errors(profile) for profile in (None, "sharp", "linear"))
        self.assertEqual(len(constant), 700)
        # Sharp: the same noise as constant noise until step 350, then the exact battery
        np.testing.assert_allclose(sharp[:350], constant[:350], atol=1e-6)
        self.assertEqual(np.abs(sharp[350:]).max(), 0.0)
        # Linear: the same draws, scaled by 1 - k / 350, then the exact battery
        steps = np.arange(350)
        np.testing.assert_allclose(linear[:350], constant[:350] * (1 - steps / 350), atol=1e-6)
        self.assertEqual(np.abs(linear[350:]).max(), 0.0)
        self.assertGreater(np.abs(constant[350:]).max(), 0.1)
        for profile, step in (("ramp", 350), ("sharp", 0)):
            with self.assertRaisesRegex(ValueError, "profile must be one of"):
                NoisyBattery(training.bp.BatteryPlaneEnv(observation_mode="observed"), 0.12, profile, step)

    def test_inflight_configs_scripts_and_report_use_the_profile(self):
        import report_step0
        import run_step0
        from step1_controls import noisy_batteries

        script = plain_ppo_script()
        for profile in ("sharp", "linear"):
            cases = [(f"1c_{profile}", config("+step1=cppo_inflight_noise", f"env.battery_noise_flight={profile}", step0=False),
                      noisy_cppo(), "cppo", training, f"S1c_inflight_{profile}350"),
                     (f"1a_{profile}", plain_ppo_config("+experiment=S1a_inflight_noise_bs256", f"env.battery_noise_flight={profile}"),
                      plain_ppo_config("+experiment=S1a_noisy_bs256"), "ppo", script, f"S1a_inflight_bs256_{profile}350")]
            for condition, cfg, noisy, learner, module, name in cases:
                with self.subTest(condition=condition):
                    (training.validate_config if learner == "cppo" else script.validate_ppo)(cfg)
                    # The noisy oracle with a within-flight profile
                    for section in (learner, "training", "evaluation"):
                        self.assertEqual(cfg[section], noisy[section])
                    env, noisy_env = OmegaConf.to_container(cfg.env), OmegaConf.to_container(noisy.env)
                    self.assertEqual((env.pop("battery_noise_flight"), noisy_env.pop("battery_noise_flight")), (profile, None))
                    self.assertEqual(env, noisy_env)
                    self.assertEqual(cfg.wandb.run_name, f"{name}_noise0.06_gamma0.99_rs0.1_s0")
                    protocol = report_step0.evaluation_protocol(cfg, condition)
                    noisy_protocol = report_step0.evaluation_protocol(noisy, condition.split("_")[0])
                    self.assertEqual({key for key in protocol if protocol[key] != noisy_protocol[key]},
                                     {"battery_noise_flight", "battery_noise_flight_step"})
                    # Training and validation both fly the profile
                    small = OmegaConf.merge(cfg, OmegaConf.create({"training": {"n_envs": 2}}))
                    train_env = module.make_train_env(small)
                    eval_env, _, _ = module.make_eval_env(small, list(training.iter_eval_resets(small.evaluation.reset_grid)))
                    try:
                        self.assertEqual({(w.profile, w.profile_step) for w in noisy_batteries(train_env) + noisy_batteries(eval_env)},
                                         {(profile, 350)})
                    finally:
                        train_env.close()
                        eval_env.close()
        with self.assertRaisesRegex(ValueError, "requires env.battery_noise_flight=None"):
            report_step0.evaluation_protocol(noisy_cppo("env.battery_noise_flight=sharp"), "1c")
        with self.assertRaisesRegex(ValueError, "requires env.battery_noise_flight=linear"):
            report_step0.evaluation_protocol(config("+step1=cppo_inflight_noise", step0=False), "1c_linear")
        for check, cfg in ((training.validate_config, config("env.battery_noise_flight=sharp")),
                           (training.validate_config, noisy_cppo("env.battery_noise_flight=ramp")),
                           (script.validate_ppo, plain_ppo_config("+experiment=S1a_inflight_noise_bs256", "env.battery_noise_flight_step=0"))):
            with self.assertRaisesRegex(ValueError, "battery_noise_flight"):
                check(cfg)
        # The launcher's linear runs switch the profile, the sharp runs keep the config's
        linear = run_step0.command_for("1b_linear", 0, 1, Path("/tmp/launch"), battery_noise=0.12)
        sharp = run_step0.command_for("1b_sharp", 0, 1, Path("/tmp/launch"), battery_noise=0.12)
        self.assertIn("env.battery_noise_flight=linear", linear)
        self.assertFalse(any(item.startswith("env.battery_noise_flight=") for item in sharp))
        self.assertEqual(plain_ppo_config(*[i for i in linear[2:] if not i.startswith("hydra.")]).env.battery_noise_flight, "linear")

    def test_encoder_error_reads_the_pilots_flight_without_the_battery(self):
        import battery_plane as bp
        import step1_encoder_error as measure

        def model(cfg):
            return training.build_model(cfg, training.make_train_env(cfg))

        a = model(config(*SMALL, step0=False)).policy
        pilot = model(config(*SMALL)).policy
        env_kwargs, reset = {"T": 30}, {"seed": 1000, "options": {"b0": 0.3, "vx_obs_noise": 5.0}}
        for flyer in (None, pilot):
            flight = measure.fly(a, flyer, env_kwargs, reset)
            # The same flight, flown by the policy alone through predict(), as in a validation
            flies = flyer or a
            env = bp.BatteryPlaneEnv(observation_mode="observed" if flyer else "hidden", **env_kwargs)
            obs, info = env.reset(seed=1000, options=dict(reset["options"]))
            state, observations, actions, batteries, total = None, [], [], [info["b0"]], 0.0
            while True:
                action, state = flies.predict(obs[None], state, np.array([state is None]), deterministic=True)
                observations.append(obs[:4] if flyer else obs)
                actions.append(action[0])
                obs, reward, done, _, info = env.step(action[0])
                total += reward
                if done:
                    break
                batteries.append(info["b"])
            self.assertEqual((flight["length"], flight["batteries"]), (len(actions), batteries))
            self.assertAlmostEqual(flight["return"], total, places=6)
            # The encoder read the hidden observations and the executed actions of that flight, the pilot's included
            previous = np.concatenate([np.zeros((1, 2), np.float32), np.array(actions[:-1])])
            with torch.no_grad():
                contexts, _ = a.encoder.unroll(a.encoder.initial_state(1), torch.as_tensor(np.array(observations)[None]),
                                               torch.as_tensor(previous[None]))
            np.testing.assert_allclose(flight["estimates"], contexts[0, :, 0].numpy(), atol=1e-5)
            # The recorded flight (for the ideal estimator) is that flight's hidden observations and executed actions
            recorded = measure.fly(a, flyer, env_kwargs, reset, record=True)
            np.testing.assert_array_equal(np.array(recorded["observations"], np.float32), np.array(observations))
            np.testing.assert_array_equal(np.array(recorded["actions"], np.float32), np.array(actions))
            self.assertEqual(recorded["batteries"], flight["batteries"])

    def test_ideal_estimator_replays_the_physics_and_learns_only_at_the_throttle_cap(self):
        import battery_plane as bp
        from step1_ideal_estimator import IdealBatteryEstimator

        def flight(b0, policy, steps=700):
            env = bp.BatteryPlaneEnv(observation_mode="hidden")
            obs, info = env.reset(seed=1001, options={"b0": b0, "vx_obs_noise": 5.0})
            vx0, observations, actions, batteries = float(env.core.vx[0]), [obs], [], [info["b0"]]
            rng = np.random.default_rng(0)
            for k in range(steps):
                action = policy(rng)
                obs, _, terminated, truncated, info = env.step(action)
                actions.append(action)
                if terminated or truncated:
                    break
                observations.append(obs)
                batteries.append(info["b"])
            return observations, actions, np.array(batteries), vx0

        # With the true starting state as its only candidate, the replay is the environment's flight
        observations, actions, batteries, vx0 = flight(0.37, lambda rng: rng.uniform(-1, 1, 2).astype(np.float32), 400)
        out = IdealBatteryEstimator(b_values=[0.37], vx_values=[vx0]).estimate(observations, actions)
        np.testing.assert_allclose(out["estimates"], batteries, atol=1e-9)
        self.assertLess(max(out["vy_mismatch"]), 1e-5)
        self.assertGreater(len(batteries), 100)
        # Full throttle reveals the battery; flying below the cap reveals nothing
        estimator = IdealBatteryEstimator(b_points=101, vx_points=151)
        for b0 in (0.3, 0.6):
            observations, actions, batteries, _ = flight(b0, lambda rng: np.array([1.0, 0.2], np.float32))
            out = estimator.estimate(observations, actions)
            errors, stds = np.abs(np.array(out["estimates"]) - batteries), np.array(out["stds"])
            # Take-off: the prior's mean (0.45) and spread (0.5 / sqrt(12))
            self.assertAlmostEqual(out["estimates"][0], 0.45, places=3)
            self.assertAlmostEqual(stds[0], 0.5 / np.sqrt(12), delta=0.002)
            self.assertLess(errors[100], 0.05)
            self.assertLess(stds[100], 0.06)
            observations, actions, batteries, _ = flight(b0, lambda rng: np.array([0.3, 0.0], np.float32))
            stds = np.array(estimator.estimate(observations, actions)["stds"])
            self.assertEqual(len(batteries), 700)
            self.assertGreater(stds.min(), 0.14)
        # It reads hidden-mode observations only
        env = bp.BatteryPlaneEnv(observation_mode="observed")
        obs, _ = env.reset(seed=1000)
        with self.assertRaisesRegex(ValueError, "never the battery"):
            estimator.estimate([obs], [])

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
        early = run_step0.parse_args(["--conditions", "1a_early", "1b_early", "1c_early", "--battery-noises", "0.03", "0.12",
                                      "--output", "/tmp/step1_launch"])
        self.assertEqual(early.battery_noises, [0.03, 0.12])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            run_step0.parse_args(["--conditions", "1c_early", "--battery-noises", "0", "0.06"])
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
        # In-flight runs: report_step0.py refuses to mix profiles, so one report per noise level and profile
        runs = [done(f"1{x}_{profile}", s, None, noise) for noise in (0.06, 0.12) for profile in ("linear", "sharp")
                for x in "ac" for s in (0, 1)]
        lines = run_step0.report_commands(runs, [None], [0.06, 0.12], args.output)
        self.assertEqual([line.split(":")[0] for line in lines],
                         [f"Report (battery noise {noise}, {profile} in-flight noise)" for noise in ("0.06", "0.12")
                          for profile in ("sharp", "linear")])
        for line, (noise, profile) in zip(lines, [(n, p) for n in ("0.06", "0.12") for p in ("sharp", "linear")]):
            self.assertEqual(line.count("--run "), 4)
            self.assertEqual(line.count(f"_{profile} "), 4)
            self.assertTrue(line.endswith(f"step0_report_noise{noise}_{profile}.json"))


if __name__ == "__main__":
    unittest.main()
