"""CPU checks of the learner's preludes (EASY_ENV_WARMUP.md, E3): python -m unittest checks.prelude_checks -v

On the easy version of BatteryPlane (env.kwargs.warmup_steps > 0) a learned encoder reads the autopilot's take-off
before every episode: Contextual_PPO's prelude_info_keys. These checks cover the policy's start_episodes, collection,
the episode buffer, PPO's cached contexts, validation (PreludePredictor), saving and the training script's settings.
"""
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import battery_plane as bp
from checks.step0_checks import config
from contextual_ppo import ContextualPPO, PreludePredictor
from contextual_ppo.type_aliases import ContextualPolicyState
import main_battery_plane_contextual_ppo as training

# Short flights with a short take-off and transition: several episodes end inside every 32-step stream
EASY = {"T": 30, "warmup_steps": 6, "transition_steps": 4}
SMALL_EASY = ["training.n_envs=2", "cppo.n_steps=32", "cppo.n_minibatches=2", "cppo.n_epochs=1", "cppo.net_arch=[16]",
              "cppo.verbose=0", "cppo.gru.hidden_dim=16", "cppo.gru.num_layers=1", "cppo.supervised.episode_batch_size=4",
              "cppo.supervised.gradient_steps=2", *(f"+env.kwargs.{key}={value}" for key, value in EASY.items())]
KEYS, LABEL = ("warmup_observations", "warmup_actions"), "warmup_batteries"


class ResetRecorder(gym.Wrapper):
    """Keeps every reset's observation and info: what the environment gave, to compare the learner with."""

    def __init__(self, env):
        super().__init__(env)
        self.resets = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.resets.append((obs.copy(), info))
        return obs, info


def easy_env(n_envs=2):
    """n_envs hidden-mode copies of the short easy version, each recording its resets."""
    recorders = [ResetRecorder(bp.BatteryPlaneEnv(observation_mode="hidden", **EASY)) for _ in range(n_envs)]
    return DummyVecEnv([lambda r=r: Monitor(r) for r in recorders]), recorders


def easy_model(env=None, *overrides):
    cfg = config("+joint=J0_gru_a_budget", *SMALL_EASY, *overrides, step0=False)
    training.validate_config(cfg)
    return training.build_model(cfg, env if env is not None else training.make_train_env(cfg))


def prelude_of(info):
    return info[KEYS[0]], info[KEYS[1]], info[LABEL]


class PreludeChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_start_episodes_is_reading_the_prelude_first(self):
        policy = easy_model().policy
        torch.manual_seed(0)
        length = 7
        observations, actions = torch.randn(2, length, 4), torch.rand(2, length, 2) * 2 - 1
        first = torch.randn(2, 4)
        with torch.no_grad():
            state = policy.start_episodes(policy.initial_state(2), np.ones(2, bool), (observations.numpy(), actions.numpy()))
            context, _ = policy.encode(first, state)
            # The same as one sequence: the prelude, then the first observation, each with the action before it
            sequence = torch.cat([observations, first[:, None]], dim=1)
            prev = torch.cat([torch.zeros(2, 1, 2), actions], dim=1)
            whole, _ = policy.encoder.unroll(policy.encoder.initial_state(2), sequence, prev)
        torch.testing.assert_close(context, whole[:, -1], rtol=0, atol=1e-5)
        torch.testing.assert_close(state.prev_actions, actions[:, -1])
        # Rows that don't start keep their memory
        with torch.no_grad():
            running = ContextualPolicyState({"gru_state": torch.randn(1, 2, 16)}, torch.rand(2, 2))
            mixed = policy.start_episodes(running, np.array([True, False]), (observations.numpy(), actions.numpy()))
        torch.testing.assert_close(mixed.encoder_state["gru_state"][:, 1], running.encoder_state["gru_state"][:, 1])
        torch.testing.assert_close(mixed.prev_actions[1], running.prev_actions[1])
        torch.testing.assert_close(mixed.encoder_state["gru_state"][:, 0], state.encoder_state["gru_state"][:, 0])

    def test_collection_and_ppo_use_every_prelude(self):
        env, recorders = easy_env()
        model = easy_model(env)
        self.assertEqual(model.prelude_info_keys, KEYS)
        policy = model.policy
        _, callback = model._setup_learn(model.n_steps * model.n_envs * 3)
        for rollout in range(3):          # the memory carries over rollouts; episodes start inside streams
            model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
            (streams,) = list(model.rollout_buffer.get(1, shuffle=False))
            self.assertTrue(bool(streams.episode_starts[:, 1:].any()), "no episode started inside a stream")
            # The cached contexts are the encoder's, with every episode started after its own prelude (from the env)
            for row, e in enumerate(streams.env_indices.tolist()):
                resets = {r[0].tobytes(): r[1] for r in recorders[e].resets}
                state = ContextualPolicyState({k: v[:, row:row + 1] for k, v in streams.initial_state.items()},
                                              streams.prev_actions[row:row + 1, 0])
                with torch.no_grad():
                    for t in range(model.n_steps):
                        if streams.episode_starts[row, t]:
                            observations, actions, _ = prelude_of(resets[streams.observations[row, t].numpy().tobytes()])
                            state = policy.start_episodes(state, np.ones(1, bool), (observations[None], actions[None]))
                            # The episode's first previous action is the prelude's last action
                            torch.testing.assert_close(streams.prev_actions[row, t], torch.as_tensor(actions[-1]))
                        context, encoder_state = policy.encode(streams.observations[row:row + 1, t], state)
                        torch.testing.assert_close(context[0], streams.context_estimates[row, t], rtol=0, atol=1e-5)
                        state = ContextualPolicyState(encoder_state, streams.executed_actions[row:row + 1, t])
            # PPO on the cached contexts reproduces collection: the ratios start at 1
            with torch.no_grad():
                values, log_prob, _ = policy.evaluate_actions_from_contexts(
                    streams.observations, streams.actions, streams.context_estimates)
            torch.testing.assert_close(values, streams.old_values, rtol=0, atol=1e-5)
            torch.testing.assert_close(log_prob, streams.old_log_prob, rtol=0, atol=1e-5)
        # The episode buffer's episodes begin with their preludes and the true battery as their labels
        episodes = model.episode_buffer.episodes("train") + model.episode_buffer.episodes("validation")
        self.assertGreater(len(episodes), 4)
        by_prelude = {info[KEYS[0]].tobytes(): (obs, info) for recorder in recorders for obs, info in recorder.resets}
        for episode in episodes:
            length = EASY["warmup_steps"] + EASY["transition_steps"]
            hand_over, info = by_prelude[episode.observations[:length].tobytes()]
            observations, actions, batteries = prelude_of(info)
            np.testing.assert_array_equal(episode.observations[:length], observations)
            np.testing.assert_array_equal(episode.observations[length], hand_over)
            np.testing.assert_array_equal(episode.prev_actions[0], 0.0)
            np.testing.assert_array_equal(episode.prev_actions[1:length + 1], actions)
            np.testing.assert_allclose(episode.labels[:length, 0], batteries, rtol=0, atol=1e-7)
            self.assertAlmostEqual(float(episode.labels[length, 0]), info["b0"], places=6)
            self.assertEqual(episode.observations.shape[1], 4)          # hidden mode: never the battery as input
        # PPO's own minibatch pass uses them and never replays the encoder (which would start the episodes from blank
        # memories; the KL estimate can't tell, its float32 rounds such small differences to 0)
        groups, seen = policy.parameter_groups(), []
        original = policy.evaluate_actions_from_contexts
        policy.evaluate_actions_from_contexts = lambda obs, actions, latents: seen.append(latents) or original(obs, actions, latents)
        policy.evaluate_actions = lambda *args, **kwargs: self.fail("PPO replayed the encoder despite the preludes")
        try:
            for minibatch in model.rollout_buffer.get(model.n_minibatches, shuffle=False):
                n_seen = len(seen)
                model._minibatch_gradients(minibatch, groups, 0.2, None)
                torch.testing.assert_close(torch.cat(seen[n_seen:], dim=1), minibatch.context_estimates, rtol=0, atol=0)
        finally:
            del policy.evaluate_actions_from_contexts, policy.evaluate_actions
        # An update trains the actor, the critic and (after the PPO epochs) the encoder
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        model.train()
        changed = {k for k, v in policy.state_dict().items() if not torch.equal(before[k], v)}
        self.assertTrue(any(k.startswith("encoder.") for k in changed) and any(k.startswith("action_net") for k in changed))

    def test_validation_reads_the_prelude(self):
        model = easy_model()
        env, _ = easy_env(1)
        env.seed(1000)
        rewards, lengths = evaluate_policy(PreludePredictor(model, env), env, n_eval_episodes=3, deterministic=True,
                                           return_episode_rewards=True)
        # By hand: each episode starts after its prelude, from the same seeds
        manual, _ = easy_env(1)
        manual.seed(1000)
        obs = manual.reset()
        for episode in range(3):
            observations, actions, _ = prelude_of(manual.reset_infos[0])
            state = model.policy.start_episodes(model.policy.initial_state(1), np.ones(1, bool), (observations[None], actions[None]))
            total, steps, done = 0.0, 0, False
            while not done:
                action, state = model.predict(obs, state=state, episode_start=None, deterministic=True)
                obs, reward, dones, _ = manual.step(action)
                total, steps, done = total + float(reward[0]), steps + 1, bool(dones[0])
            self.assertEqual(steps, lengths[episode])
            self.assertAlmostEqual(total, rewards[episode], places=4)
        # The predictor stands in for the model
        self.assertIs(PreludePredictor(model, env).policy, model.policy)

    def test_training_script_validates_after_the_preludes(self):
        cfg = config("+joint=J0_gru_a_budget", *SMALL_EASY, "evaluation.reset_grid.eval_seeds=[1000]",
                     "evaluation.reset_grid.eval_initial_battery=[0.3,0.6]", step0=False)
        model = training.build_model(cfg, training.make_train_env(cfg))
        # The final validation and the final test (run_schedule): every flight read after its prelude
        starts, original = [], model.policy.start_episodes
        model.policy.start_episodes = lambda *args, **kwargs: starts.append(args[1]) or original(*args, **kwargs)
        try:
            result, _ = training.run_schedule(model, cfg, cfg.evaluation.reset_grid)
        finally:
            del model.policy.start_episodes
        self.assertEqual(len(result["episode_rewards"]), 2)
        self.assertEqual(sum(int(np.sum(s)) for s in starts), 2)
        # The validation during training (the evaluation callback) predicts through PreludePredictor
        env, scheduler, raw_env = training.make_eval_env(cfg, list(bp.iter_eval_resets(cfg.evaluation.reset_grid)))
        callback = training.RenderingEvalCallback(env, eval_freq=1, n_eval_episodes=2, flight_core=raw_env.core,
                                                  reset_scheduler=scheduler)
        callback.init_callback(model)
        self.assertIsInstance(callback.model, PreludePredictor)
        self.assertIs(callback.model.model, model)

    def test_saved_model_keeps_its_preludes(self):
        model = easy_model()
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder) / "model")
            loaded = ContextualPPO.load(Path(folder) / "model.zip", device="cpu")
        self.assertEqual((loaded.prelude_info_keys, loaded.prelude_label_key), (KEYS, LABEL))
        self.assertTrue(loaded.rollout_buffer.preludes)
        self.assertEqual(loaded.rollout_buffer.estimate_dim, loaded.policy.latent_dim)

    def test_settings(self):
        # Only a learned encoder on the easy version reads preludes
        self.assertEqual(training.prelude_keys_from(config("+joint=J0_gru_a_budget", *SMALL_EASY, step0=False)), (list(KEYS), LABEL))
        self.assertEqual(training.prelude_keys_from(config("+joint=J0_gru_a_budget", step0=False)), (None, None))
        easy = [f"+env.kwargs.{key}={value}" for key, value in EASY.items()]
        for step0 in ("cppo_oracle", "cppo_blind"):
            cfg = config(f"+step0={step0}", *easy, step0=False)
            self.assertEqual(training.prelude_keys_from(cfg), (None, None))
        hard = easy_model(None, "~env.kwargs.warmup_steps", "~env.kwargs.transition_steps")
        self.assertIsNone(hard.prelude_info_keys)
        self.assertFalse(hard.rollout_buffer.preludes)
        # Refusals: the PPO epochs use the cached contexts, which needs the supervised encoder; no auxiliary labels
        for overrides in (["cppo.encoder_objective=rl"], ["cppo.velocity_head.enabled=true"]):
            with self.assertRaises(ValueError, msg=overrides):
                training.validate_config(config("+joint=J0_gru_a_budget", *SMALL_EASY, *overrides, step0=False))
        env, _ = easy_env()
        common = dict(n_steps=32, context_info_keys=("b0", "b"), policy_kwargs={"encoder_arch": "gru", "gru_hidden_dim": 8,
                                                                                   "gru_num_layers": 1, "context_dim": 1})
        for kwargs in ({"encoder_objective": "rl", "prelude_info_keys": KEYS, "prelude_label_key": LABEL},
                       {"encoder_objective": "supervised", "prelude_info_keys": KEYS},
                       {"encoder_objective": "supervised", "prelude_label_key": LABEL}):
            with self.assertRaises(ValueError, msg=kwargs):
                ContextualPPO("MlpPolicy", env, **common, **kwargs)


if __name__ == "__main__":
    unittest.main()
