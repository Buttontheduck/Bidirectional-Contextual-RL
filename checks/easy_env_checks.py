"""Checks of the easy version of BatteryPlane (EASY_ENV_WARMUP.md): a forced take-off of warmup_steps steps before the
agent takes control, optionally followed by transition_steps steps flying back to the mission's starting speed.
python -m unittest checks.easy_env_checks -v
"""
import unittest

import numpy as np

import battery_plane as bp

WARMUP, TRANSITION = 100, 80


def episode(env, seed, options=None, steps=60, rng_seed=0):
    """Observations, rewards and info dicts of a short episode with random actions."""
    rng = np.random.default_rng(rng_seed)
    obs, info = env.reset(seed=seed, options=options)
    out = [(obs.copy(), None, dict(info))]
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        out.append((obs.copy(), r, dict(info)))
        if term or trunc:
            break
    return out


class EasyEnvChecks(unittest.TestCase):
    def test_no_takeoff_is_todays_environment(self):
        for mode in ("hidden", "observed"):
            for seed, options in ((0, None), (5, {"b0": 0.3, "vx_obs_noise": 5.0}), (1000, {"b0": 0.7, "y0": 80.0, "vx0": 20.0})):
                a = episode(bp.BatteryPlaneEnv(observation_mode=mode), seed, options)
                b = episode(bp.BatteryPlaneEnv(observation_mode=mode, warmup_steps=0), seed, options)
                self.assertEqual(len(a), len(b))
                for (oa, ra, ia), (ob, rb, ib) in zip(a, b):
                    np.testing.assert_array_equal(oa, ob)
                    self.assertEqual(ra, rb)
                    self.assertEqual(ia, ib)
                self.assertEqual(set(a[0][2]), {"b0"})                  # no take-off keys at reset
        # The design report of the default environment is unchanged by the new option
        self.assertEqual(bp.BatteryPlaneVec(auto_reset=False).describe(),
                         bp.BatteryPlaneVec(auto_reset=False, warmup_steps=0).describe())

    def test_handover_battery_is_exact_and_drawn_as_today(self):
        hard, easy = bp.BatteryPlaneEnv(), bp.BatteryPlaneEnv(warmup_steps=WARMUP)
        used = []
        for seed in range(0, 3000, 37):
            _, ih = hard.reset(seed=seed)
            obs, ie = easy.reset(seed=seed)
            # The hand-over battery is the battery today's environment starts with for the same seed: U(0.2, 0.7)
            self.assertEqual(ie["b0"], ih["b0"])
            self.assertEqual(float(easy.core.b[0]), ie["b0"])
            self.assertTrue(0.2 <= ie["b0"] <= 0.7)
            used.append(ie["b_start"] - ie["b0"])
            self.assertAlmostEqual(ie["warmup_batteries"][0], ie["b_start"], places=12)
            self.assertTrue(np.all(np.diff(ie["warmup_batteries"]) < 0))      # the take-off drains the battery
        self.assertTrue(0.07 < min(used) and max(used) < 0.16, (min(used), max(used)))
        # Given hand-over batteries (the validation grid) are met exactly
        for b0 in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            _, info = easy.reset(seed=1000, options={"b0": b0, "vx_obs_noise": 5.0})
            self.assertEqual(info["b0"], b0)
            self.assertEqual(float(easy.core.b[0]), b0)

    def test_takeoff_record_reveals_nothing_but_what_the_plane_shows(self):
        env = bp.BatteryPlaneEnv(observation_mode="hidden", warmup_steps=WARMUP)
        _, low = env.reset(seed=3, options={"b0": 0.2})
        _, high = env.reset(seed=3, options={"b0": 0.7})
        for info in (low, high):
            self.assertEqual(info["warmup_observations"].shape, (WARMUP, 4))     # hidden mode: no battery column
            self.assertEqual(info["warmup_actions"].shape, (WARMUP, 2))
            np.testing.assert_array_equal(info["warmup_actions"], np.tile([1.0, 0.0], (WARMUP, 1)))   # never depends on b
            np.testing.assert_array_equal(info["warmup_observations"][:, 3], 1.0)  # the mission clock has not started
        self.assertEqual(set(low), {"b0", "warmup_observations", "warmup_actions", "warmup_batteries", "b_start"})
        # The take-off is the full-throttle situation: a fuller battery accelerates harder (the battery shows in the speed)
        speed_gain = lambda info: info["warmup_observations"][-1, 1] - info["warmup_observations"][0, 1]
        self.assertGreater(speed_gain(high), speed_gain(low))
        # The agent's first observation is the hand-over state, with the full mission ahead
        obs, info = env.reset(seed=11, options={"b0": 0.5})
        self.assertEqual(obs[3], 1.0)
        self.assertEqual(int(env.core.t[0]), 0)
        observed = bp.BatteryPlaneEnv(observation_mode="observed", warmup_steps=WARMUP)
        obs, info = observed.reset(seed=11, options={"b0": 0.5})
        self.assertEqual(info["warmup_observations"].shape, (WARMUP, 5))
        self.assertAlmostEqual(float(obs[4]), 0.5, places=6)

    def test_no_flight_ends_during_takeoff_and_the_mission_has_T_steps(self):
        env = bp.BatteryPlaneEnv(warmup_steps=WARMUP)
        for seed in range(400):
            env.reset(seed=10_000 + seed)                       # would raise if a flight ended during the take-off
        for b0 in (0.2, 0.7):
            env.reset(seed=11, options={"b0": b0, "y0": 30.0, "vx0": 0.0})   # the lowest, slowest start
        env.reset(seed=11, options={"b0": 0.7})
        steps, done = 0, False
        while not done:
            _, _, term, trunc, info = env.step(np.array([0.3, 0.0], np.float32))
            steps, done = steps + 1, term or trunc
        self.assertEqual(steps, env.core.T)
        self.assertTrue(info["mission_complete"])
        self.assertEqual(info["b0"], 0.7)

    def test_reproducible_and_seed_dependent(self):
        a, b = bp.BatteryPlaneEnv(warmup_steps=WARMUP), bp.BatteryPlaneEnv(warmup_steps=WARMUP)
        oa, ia = a.reset(seed=42)
        ob, ib = b.reset(seed=42)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ia["warmup_observations"], ib["warmup_observations"])
        oc, ic = b.reset(seed=43)
        self.assertFalse(np.array_equal(ia["warmup_observations"], ic["warmup_observations"]))

    def test_refusals(self):
        env = bp.BatteryPlaneEnv(warmup_steps=WARMUP)
        with self.assertRaises(ValueError):                     # a full battery cannot hand over 0.95
            env.reset(seed=0, options={"b0": 0.95})
        for kwargs in ({"warmup_steps": -1}, {"warmup_steps": 700}, {"warmup_steps": 1.5}, {"warmup_steps": True},
                       {"warmup_steps": 10, "cap_on": "b0"}, {"warmup_action": (2.0, 0.0)}, {"warmup_action": (1.0,)}):
            with self.assertRaises(ValueError, msg=kwargs):
                bp.BatteryPlaneVec(**kwargs)
        # The vectorized auto-reset cannot fly take-offs
        vec = bp.BatteryPlaneVec(n_envs=1, warmup_steps=WARMUP, auto_reset=True, seed=0)
        vec.reset(b0=0.01)
        with self.assertRaises(RuntimeError):
            for _ in range(200):
                vec.step(np.array([[1.0, 1.0]]))
        # A core built from run settings that include the take-off still works for the scripts that only read it
        core = bp.BatteryPlaneVec(n_envs=1, observation_mode="hidden", seed=0, warmup_steps=WARMUP)
        self.assertEqual((core.dt, core.sigma), (0.05, 5.0))


class TransitionChecks(unittest.TestCase):
    """warmup_steps=100, transition_steps=80: after the take-off the autopilot flies back to the reset's speed."""

    def easy(self, mode="hidden"):
        return bp.BatteryPlaneEnv(observation_mode=mode, warmup_steps=WARMUP, transition_steps=TRANSITION)

    def test_handover_state_is_todays_start_state(self):
        hard, easy = bp.BatteryPlaneEnv(), self.easy()
        for seed in range(0, 3000, 61):
            oh, ih = hard.reset(seed=seed)
            oe, ie = easy.reset(seed=seed)
            self.assertEqual(ie["b0"], ih["b0"])                            # the battery: exact
            self.assertEqual(float(easy.core.b[0]), ie["b0"])
            self.assertEqual(float(easy.core.y[0]), float(hard.core.y[0]))   # the altitude: the take-off flies level
            self.assertLess(abs(float(easy.core.vx[0]) - float(hard.core.vx[0])), 3.0)   # the speed: within 3 m/s
            self.assertLess(abs(float(easy.core.vy[0])), 1e-9)
            self.assertEqual((oe[3], int(easy.core.t[0])), (1.0, 0))
            self.assertEqual(ie["warmup_observations"].shape, (WARMUP + TRANSITION, 4))
            self.assertEqual(ie["warmup_actions"].shape, (WARMUP + TRANSITION, 2))
            self.assertTrue(np.all(ie["warmup_observations"][:, 0] == oe[0]))   # level all the way: no jump
            self.assertTrue(np.all(np.diff(ie["warmup_batteries"]) < 0))
            self.assertAlmostEqual(ie["warmup_batteries"][0], ie["b_start"], places=12)
        # Given hand-over states (the validation grid) are met: the battery exactly, the speed closely
        for b0 in (0.2, 0.45, 0.7):
            for vx0 in (0.0, 15.0, 30.0):
                _, info = easy.reset(seed=1000, options={"b0": b0, "y0": 30.0, "vx0": vx0, "vx_obs_noise": 5.0})
                self.assertEqual(info["b0"], b0)
                self.assertEqual(float(easy.core.b[0]), b0)
                self.assertLess(abs(float(easy.core.vx[0]) - vx0), 3.0)

    def test_handover_state_reveals_nothing_about_the_battery(self):
        easy = self.easy()
        # The same seed (same noise, same start speed, same target) with a low and a high battery: the take-off ends
        # far apart, the hand-over is almost the same
        for seed in (3, 17, 99):
            _, low = easy.reset(seed=seed, options={"b0": 0.2})
            v_low = float(easy.core.vx[0])
            _, high = easy.reset(seed=seed, options={"b0": 0.7})
            v_high = float(easy.core.vx[0])
            end_of_takeoff = lambda info: info["warmup_observations"][WARMUP, 1] * 60.0
            self.assertGreater(end_of_takeoff(high) - end_of_takeoff(low), 3.0)
            self.assertLess(abs(v_high - v_low), 1.0)
        # Across seeds the hand-over speed does not predict the battery
        b0, speed, first = [], [], []
        for seed in range(500):
            obs, info = easy.reset(seed=20_000 + seed)
            b0.append(info["b0"]); speed.append(float(easy.core.vx[0])); first.append(float(obs[1]))
        for x in (speed, first):
            self.assertLess(abs(np.corrcoef(x, b0)[0, 1]), 0.15)

    def test_autopilot_reads_only_the_observations(self):
        hard = bp.BatteryPlaneEnv()
        for mode in ("hidden", "observed", "full"):
            easy = self.easy(mode)
            for seed in (0, 7, 1234):
                hard.reset(seed=seed)
                _, info = easy.reset(seed=seed)
                # Replayed on the recorded observations with the reset's speed as target, it gives the recorded actions
                pilot = bp.PreludeAutopilot(easy.core, np.array([float(hard.core.vx[0])]))
                replay = np.concatenate([pilot.act(obs[None, :]) for obs in info["warmup_observations"]]).astype(np.float32)
                np.testing.assert_array_equal(replay, info["warmup_actions"])
                # Take-off at full throttle; transition level, never asking for more than the lowest cap
                np.testing.assert_array_equal(info["warmup_actions"][:WARMUP], np.tile([1.0, 0.0], (WARMUP, 1)))
                back = info["warmup_actions"][WARMUP:]
                np.testing.assert_array_equal(back[:, 1], 0.0)
                self.assertTrue(np.all(np.abs(back[:, 0]) * easy.core.a_max <= easy.core.h_min + 1e-5))
                self.assertLess(back[:, 0].min(), -0.5)                     # it brakes

    def test_no_flight_ends_and_the_mission_has_T_steps(self):
        env = self.easy()
        for seed in range(200):
            env.reset(seed=30_000 + seed)                       # would raise if a flight ended during the prelude
        for y0 in (30.0, 120.0):                                # slowest and fastest take-offs, every target speed
            for vx0 in (0.0, 30.0):
                for b0 in (0.2, 0.7):
                    env.reset(seed=11, options={"b0": b0, "y0": y0, "vx0": vx0})
        env.reset(seed=11, options={"b0": 0.7})
        steps, done = 0, False
        while not done:
            _, _, term, trunc, info = env.step(np.array([0.3, 0.0], np.float32))
            steps, done = steps + 1, term or trunc
        self.assertEqual(steps, env.core.T)
        self.assertTrue(info["mission_complete"])

    def test_reproducible_and_seed_dependent(self):
        a, b = self.easy(), self.easy()
        oa, ia = a.reset(seed=42)
        ob, ib = b.reset(seed=42)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ia["warmup_observations"], ib["warmup_observations"])
        np.testing.assert_array_equal(ia["warmup_actions"], ib["warmup_actions"])
        _, ic = b.reset(seed=43)
        self.assertFalse(np.array_equal(ia["warmup_observations"], ic["warmup_observations"]))

    def test_refusals(self):
        for kwargs in ({"transition_steps": 10}, {"warmup_steps": WARMUP, "transition_steps": -1},
                       {"warmup_steps": WARMUP, "transition_steps": 700}, {"warmup_steps": WARMUP, "transition_steps": True}):
            with self.assertRaises(ValueError, msg=kwargs):
                bp.BatteryPlaneVec(**kwargs)
        with self.assertRaises(ValueError):                     # the solve needs the target speeds
            bp.BatteryPlaneVec(warmup_steps=WARMUP, transition_steps=TRANSITION).start_battery_for([0.5], [100.0], [15.0])
        with self.assertRaises(ValueError):
            self.easy().reset(seed=0, options={"b0": 0.95})


if __name__ == "__main__":
    unittest.main()
