"""Checks of train_ppo.py's elbow-task options: no fatigue in the hidden observation, the fatigue-seeing control shows
exactly the wrapper's fatigue, reset seeds fix the random starting fatigue, training and validation draws stay apart,
longer episodes last as long as asked, validation is reproducible; repeated targets switch on time, within range, in
the observation and the reward at once, and follow the reset seed; the hand load pulls the arm down; fatigue_gate.py's
decoy shows exactly the given fatigue.

    /root/venvs/carlos-myo/bin/python contextual_environments/myo/elbow_checks.py -v
"""
from __future__ import annotations

import unittest

import numpy as np

from train_ppo import VALIDATION_SEEDS, evaluate, find_fatigue, make_env, train_env_factory

FATI = "myoFatiElbowPose1D6MRandom-v0"


class ElbowChecks(unittest.TestCase):
    def test_hidden_observation_has_no_fatigue(self):
        plain, tired = make_env("myoElbowPose1D6MRandom-v0"), make_env(FATI, fatigue_reset="random")
        self.assertEqual(plain.observation_space.shape, (9,))
        self.assertEqual(tired.observation_space.shape, (9,))
        self.assertEqual(list(find_fatigue(tired).unwrapped.obs_keys), ["qpos", "qvel", "pose_err", "act"])

    def test_observed_fatigue_is_the_wrappers(self):
        hidden, seen = make_env(FATI, fatigue_reset="random"), make_env(FATI, fatigue_reset="random", observe_fatigue=True)
        self.assertEqual(seen.observation_space.shape, (15,))
        fatigue = find_fatigue(seen).muscle_fatigue
        obs_h, _ = hidden.reset(seed=1005)
        obs_s, _ = seen.reset(seed=1005)
        rng = np.random.default_rng(0)
        for _ in range(200):
            np.testing.assert_allclose(obs_s[:9], obs_h)
            np.testing.assert_allclose(obs_s[9:], fatigue.MF)
            self.assertGreater(obs_s[9:].max(), 0.0)
            action = rng.uniform(-1, 1, 6)
            obs_h, *_ = hidden.step(action)
            obs_s, *_ = seen.step(action)

    def test_reset_seed_fixes_random_fatigue(self):
        env = make_env(FATI, fatigue_reset="random")
        fatigue = find_fatigue(env).muscle_fatigue
        env.reset(seed=1000)
        first = fatigue.MF.copy()
        env.reset(seed=1001)
        other = fatigue.MF.copy()
        env.reset(seed=1000)
        np.testing.assert_array_equal(fatigue.MF, first)
        self.assertFalse(np.allclose(first, other))
        fresh = make_env(FATI)
        fresh.reset(seed=1000)
        np.testing.assert_array_equal(find_fatigue(fresh).muscle_fatigue.MF, 0.0)

    def test_training_draws_differ_from_validation(self):
        from stable_baselines3.common.vec_env import DummyVecEnv

        options = {"env_id": FATI, "episode_steps": None, "fatigue_reset": "random", "observe_fatigue": False}
        vec = DummyVecEnv([train_env_factory(options, 0.1) for _ in range(8)])
        vec.seed(2)                                       # what SB3 does when learning starts with seed 2
        vec.reset()
        starts = [find_fatigue(e).muscle_fatigue.MF.copy() for e in vec.envs]
        probe = make_env(FATI, fatigue_reset="random")
        for i, start in enumerate(starts):
            probe.reset(seed=2 + i)
            np.testing.assert_array_equal(start, find_fatigue(probe).muscle_fatigue.MF)
        for seed in VALIDATION_SEEDS:
            probe.reset(seed=seed)
            self.assertFalse(any(np.allclose(s, find_fatigue(probe).muscle_fatigue.MF) for s in starts))

    def test_longer_episodes(self):
        env = make_env(FATI, episode_steps=3000)
        env.reset(seed=0)
        steps, done = 0, False
        while not done:
            _, _, terminated, truncated, _ = env.step(np.zeros(6))
            steps += 1
            done = terminated or truncated
        self.assertEqual(steps, 3000)
        self.assertTrue(truncated and not terminated)

    def test_validation_is_reproducible(self):
        class Zero:
            def predict(self, obs, deterministic=True):
                return np.zeros(6), None

        options = {"env_id": FATI, "episode_steps": 600, "fatigue_reset": "random", "observe_fatigue": False}
        first, second = evaluate(Zero(), options, VALIDATION_SEEDS[:5]), evaluate(Zero(), options, VALIDATION_SEEDS[:5])
        self.assertEqual(first, second)
        self.assertEqual(len(first["episodes"][0]["solved_by_window"]), 2)

    def repeated(self, **kwargs):
        return make_env(FATI, episode_steps=350, target_every=100, target_range=(20, 110), load_kg=5, **kwargs)

    def targets_of(self, seed, steps=350):
        env, rng = self.repeated(), np.random.default_rng(0)
        obs, _ = env.reset(seed=seed)
        task = env.unwrapped
        targets = [float(task.target_jnt_value[0])]
        for t in range(1, steps + 1):
            obs, _, terminated, truncated, info = env.step(rng.uniform(-1, 1, 6))
            if terminated or truncated:
                break
            targets.append(float(task.target_jnt_value[0]))
            np.testing.assert_allclose(obs, task.get_obs(), atol=1e-6)      # the returned observation is the current one
            self.assertAlmostEqual(float(obs[2]), targets[-1] - float(task.data.qpos[0]), places=5)
            if t % 100 == 1:                                                # the reward scores the new target at once
                self.assertAlmostEqual(float(info["pose"]), -abs(targets[-2] - float(task.data.qpos[0])), places=5)
        return np.array(targets)

    def test_repeated_targets_switch_on_time_within_range(self):
        targets = self.targets_of(1000)
        self.assertEqual(len(targets), 350)                                 # steps 0..349; the episode ends at 350
        changes = np.flatnonzero(np.diff(targets)) + 1
        np.testing.assert_array_equal(changes, [100, 200, 300])
        self.assertTrue(np.all((targets >= np.radians(20)) & (targets <= np.radians(110))))

    def test_repeated_targets_follow_the_reset_seed(self):
        np.testing.assert_array_equal(self.targets_of(1000, 250), self.targets_of(1000, 250))
        self.assertFalse(np.allclose(self.targets_of(1000, 250), self.targets_of(1001, 250)))

    def test_hand_load_pulls_the_arm_down(self):
        held = []
        for load in (0.0, 5.0):
            env = make_env(FATI, episode_steps=400, load_kg=load)
            env.reset(seed=0)
            action = -np.ones(6)
            action[3:] = 0.5 + np.log(0.4 / 0.6) / 5                          # flexors at excitation 0.4
            for _ in range(300):
                env.step(action)
            held.append(np.degrees(env.unwrapped.data.qpos[0]))
            self.assertEqual(bool(np.any(env.unwrapped.data.qfrc_applied)), load > 0)
        self.assertGreater(held[0] - held[1], 10.0)

    def test_gate_decoy_shows_the_given_fatigue(self):
        from fatigue_gate import make_decoy_env

        options = {"env_id": FATI, "episode_steps": 150, "fatigue_reset": "random", "observe_fatigue": False,
                   "target_every": 100, "target_range": (0, 110), "load_kg": 5.0}
        decoy, hidden = make_decoy_env(options), make_env(**options)
        trajectory = np.random.default_rng(1).uniform(0, 1, (151, 6))
        decoy.trajectory = trajectory
        obs_d, _ = decoy.reset(seed=1003)
        obs_h, _ = hidden.reset(seed=1003)
        rng = np.random.default_rng(2)
        for t in range(150):
            np.testing.assert_allclose(obs_d[:9], obs_h)
            np.testing.assert_allclose(obs_d[9:], trajectory[t], atol=1e-6)
            action = rng.uniform(-1, 1, 6)
            obs_d, *_ = decoy.step(action)
            obs_h, *_ = hidden.step(action)


if __name__ == "__main__":
    unittest.main()
