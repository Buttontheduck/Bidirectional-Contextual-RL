"""CPU checks of step 0 (ENCODER_PLAN.md), run 0c: python -m unittest checks.step0_checks -v"""
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
import torch

import main_battery_plane_contextual_ppo as training
from step0_oracle import TrueBatteryEncoder, TrueBatteryPolicy

# Small networks and 4-step episodes, so that every rollout holds episode starts and terminal transitions
SMALL = ["training.n_envs=2", "cppo.n_steps=8", "cppo.n_minibatches=2", "cppo.n_epochs=1", "cppo.net_arch=[16]",
         "+env.kwargs.T=4", "cppo.verbose=0"]


def config(*overrides, step0=True):
    with initialize_config_dir(version_base=None, config_dir=str(training.PROJECT_ROOT / "config")):
        return compose(config_name="cppo", overrides=[*(["+step0=cppo_oracle"] if step0 else []), *overrides])


def plain_ppo(cfg, env):
    """SB3 PPO with the policy settings of plain PPO (Env_BatteryPlane/ppo) for the same cppo section."""
    net_arch = list(cfg.cppo.net_arch)
    return PPO("MlpPolicy", env, n_steps=int(cfg.cppo.n_steps), batch_size=int(cfg.cppo.n_steps), device="cpu",
               policy_kwargs={"net_arch": {"pi": net_arch, "vf": net_arch},
                              "activation_fn": training.ACTIVATIONS[str(cfg.cppo.activation_fn).lower()],
                              "ortho_init": bool(cfg.cppo.ortho_init), "log_std_init": float(cfg.cppo.log_std_init)})


class Step0Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_oracle_changes_only_the_battery_input(self):
        cfg, base = config(), config(step0=False)
        training.validate_config(cfg)
        self.assertEqual((cfg.env.observation_mode, cfg.training.total_timesteps), ("observed", 2_000_000))
        learner = OmegaConf.to_container(cfg.cppo, resolve=True)
        baseline = OmegaConf.to_container(base.cppo, resolve=True)
        self.assertEqual((learner.pop("context_source"), learner.pop("encoder_objective")), ("oracle", "rl"))
        del baseline["context_source"], baseline["encoder_objective"]
        self.assertEqual(learner, baseline)
        for section in ("evaluation", "final_test"):
            self.assertEqual(cfg[section], base[section])

    def test_oracle_rejects_settings_that_need_an_encoder(self):
        for override in ("env.observation_mode=hidden", "cppo.encoder_objective=supervised",
                         "cppo.velocity_head.enabled=true", "cppo.context_head=gaussian", "cppo.context_dim=2"):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "context_source=oracle"):
                training.validate_config(config(override))

    def test_oracle_requires_noisy_observed_mode(self):
        training.validate_config(config("env.observation_mode=observed"))
        for mode in ("full", "hidden", "clean"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "requires env.observation_mode=observed"):
                training.validate_config(config(f"env.observation_mode={mode}"))

    def test_explicit_invalid_context_source_is_not_a_legacy_default(self):
        for value in ("unknown", "null"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cppo.context_source must be"):
                training.validate_config(config(f"cppo.context_source={value}"))

    def test_evaluation_reports_outcomes_before_vecenv_auto_reset(self):
        class FixedPolicy:
            num_timesteps = 32

            def predict(self, obs, **kwargs):
                return np.tile([1.0, -1.0], (len(obs), 1)), None

        cases = [
            (0.7, 100.0, "completed"),
            (0.7, 0.001, "ground_crash"),
            (0.000001, 100.0, "battery_empty"),
            # Both failures on the same transition use the environment UI's battery-first priority.
            (0.000001, 0.001, "battery_empty"),
        ]
        for battery, altitude, expected in cases:
            with self.subTest(expected=expected, battery=battery, altitude=altitude):
                cfg = config(*SMALL, "evaluation.reset_grid.eval_seeds=[1000,1001]",
                             f"evaluation.reset_grid.eval_initial_battery=[{battery}]",
                             f"evaluation.reset_grid.eval_y0=[{altitude}]", "evaluation.reset_grid.eval_vx0=[15]")
                result = training.evaluate_schedule(FixedPolicy(), cfg, cfg.evaluation.reset_grid)
                self.assertEqual(result["episode_outcomes"], [expected, expected])
                rates = {"completed": result["completion_rate"], "ground_crash": result["ground_crash_rate"],
                         "battery_empty": result["battery_empty_rate"], "truncated": result["truncated_rate"]}
                self.assertEqual(rates[expected], 1.0)
                self.assertEqual(sum(rates.values()), 1.0)
                self.assertEqual(len(result["episode_rewards"]), 2)

    def test_branches_read_exactly_plain_ppo_observed_inputs(self):
        cfg = config(*SMALL)
        env = training.make_train_env(cfg)
        try:
            policy = training.build_model(cfg, env).policy
            plain = plain_ppo(cfg, env).policy
            self.assertIsInstance(policy, TrueBatteryPolicy)
            self.assertIsInstance(policy.encoder, TrueBatteryEncoder)
            self.assertEqual(policy.parameter_groups()["encoder"], [])
            # Same networks: the plain PPO policy loads the oracle's weights under the same names
            plain.load_state_dict(policy.state_dict())

            obs = torch.as_tensor(env.reset())
            self.assertEqual(obs.shape[-1], 5)
            state = policy.initial_state(obs.shape[0])
            contexts, _ = policy.encode(obs, state)
            torch.testing.assert_close(contexts, obs[:, -1:])
            for branch in policy._branch_inputs(obs, contexts):
                torch.testing.assert_close(branch, plain.extract_features(obs))

            actions, values, _, _, _ = policy(obs, state, deterministic=True)
            torch.testing.assert_close(values, plain.predict_values(obs))
            torch.testing.assert_close(actions, plain.get_distribution(obs).get_actions(deterministic=True))
            # No history: another previous action changes nothing
            other = state._replace(prev_actions=torch.ones_like(state.prev_actions))
            other_actions, other_values, _, _, _ = policy(obs, other, deterministic=True)
            torch.testing.assert_close((other_actions, other_values), (actions, values))
        finally:
            env.close()

    def test_rollouts_carry_the_true_battery_and_only_the_heads_learn(self):
        cfg = config(*SMALL)
        env = training.make_train_env(cfg)
        try:
            model = training.build_model(cfg, env)
            self.assertIsNone(model.episode_buffer)
            before = {name: param.detach().clone() for name, param in model.policy.named_parameters()}
            model.learn(total_timesteps=32)

            # The last rollout: resets and terminal transitions included, the observation's last column is b_t
            buffer = model.rollout_buffer
            self.assertTrue(buffer.episode_starts[1:].any() and buffer.terminated.any())
            np.testing.assert_array_equal(buffer.observations[..., -1], buffer.contexts[..., 0])
            np.testing.assert_array_equal(buffer.next_observations[..., -1], buffer.next_contexts[..., 0])

            groups = model.policy.parameter_groups()
            heads = {id(param) for param in groups["actor"] + groups["value"]}
            self.assertEqual(heads, {id(param) for param in model.policy.parameters()})
            self.assertTrue(any(not torch.equal(param, before[name]) for name, param in model.policy.named_parameters()))

            with tempfile.TemporaryDirectory() as folder:
                model.save(Path(folder) / "model")
                loaded = training.ContextualPPO.load(Path(folder) / "model.zip", device="cpu")
            self.assertIsInstance(loaded.policy, TrueBatteryPolicy)
            obs = env.reset()
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
