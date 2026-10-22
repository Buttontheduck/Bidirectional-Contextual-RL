"""Checks of city_env.py: the hidden setting differs from the observed one only by the charge; scenarios fix the start
and the starting charge in both settings; 48-step episodes with the time left; the reward is CityLearn's cost; the
energy accounting; training, validation and final-test hours never mix.

    /root/venvs/carlos-citylearn/bin/python contextual_environments/city/city_checks.py -v
"""
from __future__ import annotations

import unittest

import numpy as np

from city_env import (EPISODE_STEPS, TEST_SCENARIOS, TRAIN_STARTS, VALIDATION_SCENARIOS, CityBattery, fly, rule_action,
                      split_of_hour)


class CityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seen, cls.hidden = CityBattery(observe_charge=True), CityBattery(observe_charge=False)

    def test_hidden_differs_only_by_the_charge(self):
        self.assertIn("electrical_storage_soc", self.seen.observation_names)
        self.assertNotIn("electrical_storage_soc", self.hidden.observation_names)
        self.assertEqual(self.seen.observation_names[-1], "time_left")
        self.assertEqual(len(self.seen.observation_names), len(self.hidden.observation_names) + 1)
        soc = self.seen.observation_names.index("electrical_storage_soc")
        rng = np.random.default_rng(0)
        obs_s, info_s = self.seen.reset(options={"scenario": VALIDATION_SCENARIOS[3]})
        obs_h, info_h = self.hidden.reset(options={"scenario": VALIDATION_SCENARIOS[3]})
        for _ in range(EPISODE_STEPS):
            np.testing.assert_allclose(np.delete(obs_s, soc), obs_h)
            self.assertAlmostEqual(float(obs_s[soc]), info_s["charge"], places=5)   # normalized charge = charge
            self.assertEqual(info_s["charge"], info_h["charge"])
            action = rng.uniform(-1, 1, 1)
            obs_s, r_s, *_, info_s = self.seen.step(action)
            obs_h, r_h, *_, info_h = self.hidden.step(action)
            self.assertEqual(r_s, r_h)

    def test_scenario_fixes_start_and_charge(self):
        start, charge = VALIDATION_SCENARIOS[7]
        _, info = self.hidden.reset(options={"scenario": (start, charge)})
        self.assertEqual(info["start"], start)
        self.assertAlmostEqual(info["charge"], charge, places=6)
        self.assertEqual(self.hidden.raw()["hour"], self.hidden.building.energy_simulation.hour[0])
        draws = []
        for seed in (0, 0, 1):
            _, info = self.seen.reset(seed=seed)
            draws.append((info["start"], round(info["charge"], 6)))
        self.assertEqual(draws[0], draws[1])
        self.assertNotEqual(draws[0], draws[2])

    def test_episode_is_48_steps_with_time_left(self):
        obs, _ = self.seen.reset(options={"scenario": VALIDATION_SCENARIOS[0]})
        self.assertEqual(obs[-1], 1.0)
        for t in range(1, EPISODE_STEPS + 1):
            obs, _, terminated, truncated, _ = self.seen.step(np.zeros(1))
            self.assertAlmostEqual(float(obs[-1]), (EPISODE_STEPS - t) / EPISODE_STEPS, places=6)
            self.assertEqual(terminated, t == EPISODE_STEPS)
            self.assertFalse(truncated)

    def test_reward_is_the_cost_of_the_hour(self):
        env, rng = self.seen, np.random.default_rng(1)
        env.reset(options={"scenario": VALIDATION_SCENARIOS[5]})
        total = 0.0
        for t in range(EPISODE_STEPS):
            raw = env.raw()
            self.assertAlmostEqual(raw["electricity_pricing"], env.year_price[env.start + t], places=6)
            _, reward, terminated, _, info = env.step(rng.uniform(-1, 1, 1))
            net = float(env.building.net_electricity_consumption[t])               # the hour just executed
            self.assertAlmostEqual(reward, -max(net * env.year_price[env.start + t], 0.0), places=4)
            if not terminated:                                                     # the observation shows the same hour
                self.assertAlmostEqual(env.raw()["net_electricity_consumption"], net, places=4)
                self.assertAlmostEqual(env.raw()["electrical_storage_soc"], info["charge"], places=6)
            total += reward
        kpi = float(np.clip(env.building.net_electricity_consumption_cost[:EPISODE_STEPS], 0, None).sum())
        self.assertAlmostEqual(-total, kpi, places=4)

    def test_energy_accounting(self):
        episode = fly(self.hidden, lambda env, obs: rule_action("grid_before_peak", env), VALIDATION_SCENARIOS[:1])[0]
        self.assertAlmostEqual(episode["start_energy"], VALIDATION_SCENARIOS[0][1] * self.hidden.capacity, places=4)
        self.assertEqual(len(episode["charges"]), EPISODE_STEPS + 1)
        self.assertAlmostEqual(episode["end_energy"], episode["charges"][-1] * self.hidden.capacity, places=6)
        self.assertAlmostEqual(episode["charges"][-1],
                               float(self.hidden.battery.soc[EPISODE_STEPS - 1]), places=5)   # charge after the last hour

    def test_time_splits_never_mix(self):
        for start in TRAIN_STARTS[:: 97]:
            self.assertTrue(all(split_of_hour(h) == "train" for h in range(start, start + EPISODE_STEPS + 1)))
        for scenarios, split in ((VALIDATION_SCENARIOS, "validation"), (TEST_SCENARIOS, "test")):
            for start, charge in scenarios:
                self.assertTrue(all(split_of_hour(h) == split for h in range(start, start + EPISODE_STEPS + 1)))
                self.assertTrue(0.0 <= charge <= 1.0)
        self.assertFalse({s for s, _ in VALIDATION_SCENARIOS} & {s for s, _ in TEST_SCENARIOS})
        self.assertGreater(len(TRAIN_STARTS), 6000)


if __name__ == "__main__":
    unittest.main()
