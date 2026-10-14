"""CPU checks of step 2 (ENCODER_PLAN.md), the frozen-context wrapper: python -m unittest checks.step2_checks -v"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

import battery_plane as bp
from checks.step0_checks import SMALL, config, plain_ppo_config, plain_ppo_script
from context_wrapper import ContextVecEnv, FrozenEncoderContext, IdealContext, TrueBatteryContext
import main_battery_plane_contextual_ppo as training

PLAIN_SMALL = ["training.n_envs=2", "ppo.n_steps=8", "ppo.batch_size=8", "+env.kwargs.T=6"]


def vec_env(mode: str, n_envs: int = 2, seed: int = 7, **kwargs) -> DummyVecEnv:
    envs = DummyVecEnv([lambda: Monitor(bp.BatteryPlaneEnv(observation_mode=mode, **kwargs)) for _ in range(n_envs)])
    envs.seed(seed)
    return envs


def episodes(env, steps: int, seed: int = 0):
    """Run random actions; per env, the episodes as (observations, executed actions, contexts) with the final one."""
    rng = np.random.default_rng(seed)
    obs = env.reset()
    n = env.num_envs
    current = [{"obs": [obs[i]], "actions": []} for i in range(n)]
    done_episodes = []
    for _ in range(steps):
        actions = rng.uniform(-1.3, 1.3, size=(n, 2)).astype(np.float32)   # some outside the bounds: clipped
        obs, _, dones, infos = env.step(actions)
        for i in range(n):
            current[i]["actions"].append(np.clip(actions[i], -1, 1))
            if dones[i]:
                current[i]["obs"].append(infos[i]["terminal_observation"])
                done_episodes.append(current[i])
                current[i] = {"obs": [obs[i]], "actions": []}
            else:
                current[i]["obs"].append(obs[i])
    return done_episodes


class Step2Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_true_battery_context_reproduces_observed_mode(self):
        wrapped, observed = ContextVecEnv(vec_env("hidden", T=20), TrueBatteryContext()), vec_env("observed", T=20)
        self.assertEqual(wrapped.observation_space.shape, observed.observation_space.shape)
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(wrapped.reset(), observed.reset())
        ended = 0
        for _ in range(70):
            actions = rng.uniform(-1, 1, size=(2, 2)).astype(np.float32)
            (a_obs, a_rew, a_done, a_info), (b_obs, b_rew, b_done, b_info) = wrapped.step(actions), observed.step(actions)
            np.testing.assert_array_equal(a_obs, b_obs)
            np.testing.assert_array_equal(a_rew, b_rew)
            np.testing.assert_array_equal(a_done, b_done)
            for i in np.flatnonzero(a_done):
                np.testing.assert_array_equal(a_info[i]["terminal_observation"], b_info[i]["terminal_observation"])
                ended += 1
        # Episode ends and automatic resets were exercised in both envs
        self.assertGreaterEqual(ended, 4)

    def test_ideal_context_matches_the_offline_estimator(self):
        from step1_ideal_estimator import IdealBatteryEstimator

        grid = {"b_points": 41, "vx_points": 61, "vy_tolerance": 0.01, "prune_below": 30.0}
        env = ContextVecEnv(vec_env("hidden", T=40), IdealContext(2, {"T": 40}, include_std=True, **grid))
        self.assertEqual(env.observation_space.shape, (6,))
        finished = episodes(env, 130)
        self.assertGreaterEqual(len(finished), 4)
        offline = IdealBatteryEstimator({"T": 40}, **grid)
        for episode in finished:
            observations = np.array(episode["obs"])
            # The offline estimator on the hidden part of the same flight, the last observation as final
            expected = [offline.start(observations[0, :4])]
            for t, action in enumerate(episode["actions"], start=1):
                expected.append(offline.update(action, observations[t, :4], alive=t < len(observations) - 1))
            np.testing.assert_allclose(observations[:, 4:], np.array(expected, np.float32), atol=1e-6)

    def test_frozen_encoder_context_matches_the_encoder_and_stays_frozen(self):
        cfg = config(*SMALL, step0=False)
        model = training.build_model(cfg, training.make_train_env(cfg))
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder) / "a")
            source = FrozenEncoderContext(2, Path(folder) / "a.zip")
        self.assertEqual(source.dim, 1)
        self.assertTrue(all(not p.requires_grad for p in source.encoder.parameters()))
        before = {name: p.clone() for name, p in source.encoder.state_dict().items()}
        env = ContextVecEnv(vec_env("hidden", T=12), source)
        finished = episodes(env, 40)
        self.assertGreaterEqual(len(finished), 4)
        encoder, policy = source.encoder, source.policy
        for episode in finished:
            observations = torch.as_tensor(np.array(episode["obs"]))
            # o_t's previous action is a_{t-1}, zero at the start
            previous = torch.as_tensor(np.concatenate([np.zeros((1, 2), np.float32), np.array(episode["actions"])]))
            # Each observation's context: the encoder run over the episode from its initial state, as Contextual_PPO reads it
            with torch.no_grad():
                latents, _ = encoder.unroll(encoder.initial_state(1), policy.encoder_observations(observations[None, :, :4], 2),
                                            previous[None])
            expected = policy.context_inputs(latents[0]).numpy()
            np.testing.assert_allclose(observations[:, 4:].numpy(), expected, atol=1e-5)
        # Plain PPO trains on top; the encoder does not change
        ppo = PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu", policy_kwargs={"net_arch": [16]})
        ppo.learn(total_timesteps=32)
        for name, value in source.encoder.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_ideal_estimator_never_rules_out_every_candidate_while_the_plane_flies(self):
        from step1_ideal_estimator import IdealBatteryEstimator

        # Only a candidate with less battery than the plane: it runs empty first; the constraint is skipped, not fatal
        env = bp.BatteryPlaneEnv(observation_mode="hidden")
        obs, info = env.reset(seed=1001, options={"b0": 0.5})
        estimator = IdealBatteryEstimator(b_values=[0.2], vx_values=[float(env.core.vx[0])])
        estimator.start(obs)
        for _ in range(700):
            action = np.array([1.0, 0.2], np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            mean, _ = estimator.update(action, obs)
        self.assertGreater(estimator.constraint_skips, 0)
        self.assertEqual(mean, 0.0)

    def test_pruning_keeps_the_ideal_estimates(self):
        from step1_ideal_estimator import IdealBatteryEstimator

        # A probe-then-hold flight: pruned and exact estimates agree, and few candidates are left to replay
        env = bp.BatteryPlaneEnv(observation_mode="hidden")
        obs, _ = env.reset(seed=6001)
        observations, actions = [obs], []
        for k in range(500):
            action = np.array([1.0, 0.1] if (k // 60) % 3 == 0 else [0.45, 0.0], np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            if terminated or truncated:
                break
            observations.append(obs)
        grid = {"b_points": 51, "vx_points": 121, "vy_tolerance": 0.01}
        exact = IdealBatteryEstimator(**grid).estimate(observations, actions)
        estimator = IdealBatteryEstimator(prune_below=30.0, **grid)
        pruned = estimator.estimate(observations, actions)
        np.testing.assert_allclose(pruned["estimates"], exact["estimates"], atol=1e-9)
        np.testing.assert_allclose(pruned["stds"], exact["stds"], atol=1e-9)
        self.assertLess(estimator.sim.n, 51 * 121 // 2)
        # A new flight starts from the whole grid again
        estimator.start(observations[0])
        self.assertEqual(estimator.sim.n, 51 * 121)

    def test_plain_ppo_trains_with_a_frozen_context_and_is_validated_with_it(self):
        import report_step0

        script = plain_ppo_script()
        cfg = plain_ppo_config("+experiment=S0a_blind_bs256", *PLAIN_SMALL, "context.source=ideal", "context.include_std=true",
                               "context.b_points=11", "context.vx_points=21", "evaluation.reset_grid.eval_initial_battery=[0.3]")
        script.validate_ppo(cfg)
        env = script.make_train_env(cfg)
        self.assertIsInstance(env, ContextVecEnv)
        self.assertEqual(env.observation_space.shape, (6,))
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu", policy_kwargs={"net_arch": [16]})
        model.learn(total_timesteps=32)
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder) / "model")
            loaded = PPO.load(Path(folder) / "model.zip", device="cpu")
        # Validation through either script's evaluation env reads the same context
        eval_env, _, _ = script.make_eval_env(cfg, list(training.iter_eval_resets(cfg.evaluation.reset_grid)))
        self.assertIsInstance(eval_env, ContextVecEnv)
        result = training.evaluate_schedule(loaded, cfg, cfg.evaluation.reset_grid)
        self.assertEqual(len(result["episode_rewards"]), 3)
        # The default is plain PPO, unwrapped
        self.assertNotIsInstance(script.make_train_env(plain_ppo_config("+experiment=S0a_blind_bs256", *PLAIN_SMALL)), ContextVecEnv)
        # The step 0 and step 1 conditions have no context
        with self.assertRaisesRegex(ValueError, "requires no frozen context"):
            report_step0.evaluation_protocol(cfg, "0a_blind")

    def test_ideal_context_runs_differ_from_the_memoryless_runs_only_in_the_context(self):
        from omegaconf import OmegaConf

        import report_step0
        import run_step0

        script = plain_ppo_script()
        cases = [("2a_ideal", "S2a_ideal_bs256", "S0a_blind_bs256", False), ("2b_ideal", "S2b_ideal_bs1024", "S0b_blind_bs1024", False),
                 ("2a_ideal_std", "S2a_ideal_std_bs256", "S0a_blind_bs256", True),
                 ("2b_ideal_std", "S2b_ideal_std_bs1024", "S0b_blind_bs1024", True)]
        for condition, name, blind_name, include_std in cases:
            with self.subTest(condition=condition):
                cfg, blind = plain_ppo_config(f"+experiment={name}"), plain_ppo_config(f"+experiment={blind_name}")
                script.validate_ppo(cfg)
                for section in ("env", "ppo", "training", "evaluation"):
                    self.assertEqual(cfg[section], blind[section])
                self.assertEqual((cfg.context.source, cfg.context.include_std, blind.context.source), ("ideal", include_std, "none"))
                self.assertEqual((cfg.wandb.project, cfg.wandb.run_name), ("cppo", f"{name}_gamma0.99_rs0.1_s0"))
                # Scored like step 0: the protocol of the memoryless run plus the context
                protocol, blind_protocol = (report_step0.evaluation_protocol(cfg, condition),
                                            report_step0.evaluation_protocol(blind, blind_name.replace("S0", "0").split("_bs")[0]))
                self.assertEqual({key for key in protocol if protocol[key] != blind_protocol[key]}, {"context"})
                self.assertEqual(protocol["context"], {"source": "ideal", "include_std": include_std, "b_points": 101,
                                                       "vx_points": 301, "vy_tolerance": 0.01, "prune_below": 30.0})
                # The launcher builds the same run
                command = run_step0.command_for(condition, 5, 1, Path("/tmp/ideal_launch"))
                launched = plain_ppo_config(*[item for item in command[2:] if not item.startswith("hydra.")])
                self.assertEqual(OmegaConf.to_container(launched.context), OmegaConf.to_container(cfg.context))
                self.assertEqual((launched.training.seed, launched.wandb.run_name), (5, f"{name}_gamma0.99_rs0.1_s5"))
        # Mislabeled runs are rejected
        with self.assertRaisesRegex(ValueError, "include_std=False"):
            report_step0.evaluation_protocol(plain_ppo_config("+experiment=S2a_ideal_std_bs256"), "2a_ideal")
        with self.assertRaisesRegex(ValueError, "context.source=ideal"):
            report_step0.evaluation_protocol(plain_ppo_config("+experiment=S0a_blind_bs256"), "2a_ideal")
        with self.assertRaisesRegex(ValueError, "batch size"):
            report_step0.evaluation_protocol(plain_ppo_config("+experiment=S2b_ideal_bs1024"), "2a_ideal")
        # One report per variant: the estimate alone, and with its std
        runs = [{"condition": c, "seed": s, "reward_scale": None, "battery_noise": None, "run_dir": f"/r/{c}{s}"}
                for c in ("2a_ideal", "2b_ideal", "2a_ideal_std", "2b_ideal_std") for s in (0, 1)]
        lines = run_step0.report_commands(runs, [None], [None], Path("/out"))
        self.assertEqual([line.split(":")[0] for line in lines],
                         ["Report (ideal context)", "Report (ideal context with its std)"])
        self.assertTrue(lines[0].endswith("step0_report_ideal.json") and lines[1].endswith("step0_report_ideal_std.json"))
        self.assertEqual([line.count("_std ") for line in lines], [0, 4])

    def test_context_settings_are_validated(self):
        script = plain_ppo_script()
        for overrides, message in (
                (["+experiment=S0a_observed_bs256", "context.source=ideal"], "env.observation_mode=hidden"),
                (["+experiment=S0a_blind_bs256", "context.source=unknown"], "context.source must be"),
                (["+experiment=S0a_blind_bs256", "context.source=encoder"], "context.encoder_model"),
                (["+experiment=S0a_blind_bs256", "context.source=ideal", "context.b_points=1"], "context.b_points"),
                (["+experiment=S0a_blind_bs256", "context.source=ideal", "context.prune_below=0"], "context.prune_below"),
                (["+experiment=S1a_noisy_bs256", "env.observation_mode=hidden", "context.source=ideal"], "battery_noise")):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                script.validate_ppo(plain_ppo_config(*overrides))
        # The ideal estimator assumes the environment's speed noise in validation too
        with self.assertRaisesRegex(ValueError, "speed noise"):
            script.validate_ppo(plain_ppo_config("+experiment=S0a_blind_bs256", "context.source=ideal",
                                                 "evaluation.reset_grid.eval_vx_obs_noise=[3.0]"))
        # Contextual_PPO has its own encoder
        with self.assertRaisesRegex(ValueError, "context.source"):
            training.validate_config(config("+context.source=ideal", step0=False))
        # The sources that read no infos refuse observations with the battery
        with self.assertRaisesRegex(ValueError, "never the battery"):
            ContextVecEnv(vec_env("observed"), IdealContext(2, {}, b_points=3, vx_points=3))


if __name__ == "__main__":
    unittest.main()
