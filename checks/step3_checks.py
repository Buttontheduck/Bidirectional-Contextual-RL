"""CPU checks of step 3 (ENCODER_PLAN.md), the pilots: python -m unittest checks.step3_checks -v"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

import battery_plane as bp
from checks.step0_checks import plain_ppo_config
import pilots as P

ROOT = Path(__file__).resolve().parent.parent


def flight_rng(style: str, index: int) -> np.random.Generator:
    seed = P.flight_seed(style, index)
    return np.random.default_rng([seed, P.STYLES.index(style), 17])


def saved_policy(folder: Path, experiment: str) -> str:
    """An untrained plain-PPO model with its config, as a run directory the policy pilots can load."""
    cfg = plain_ppo_config(f"+experiment={experiment}")
    env = DummyVecEnv([lambda: bp.BatteryPlaneEnv(observation_mode=cfg.env.observation_mode)])
    model = PPO("MlpPolicy", env, device="cpu", policy_kwargs={"net_arch": [16]}, seed=0)
    run = folder / experiment
    run.mkdir()
    OmegaConf.save(cfg, run / "config.yaml", resolve=True)
    model.save(run / "final_model")
    return str(run)


def actions_with_battery(pilot: P.Pilot, b0: float, rng_seed: int, seed: int = 777_777, steps: int = 200) -> np.ndarray:
    env = bp.BatteryPlaneEnv(observation_mode="observed")
    obs, _ = env.reset(seed=seed, options={"b0": b0})
    view = (lambda o: o) if pilot.observation == "observed" else (lambda o: o[:4])
    pilot.reset(np.random.default_rng(rng_seed), view(obs), seed)
    actions = []
    for t in range(steps):
        action = pilot.act(view(obs), t)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return np.array(actions)


class Step3Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.physics = P.PlanePhysics()

    def test_scripted_pilots_read_only_the_hidden_observation(self):
        # Replaying a recorded flight's hidden observations, without any plane, gives the same actions: a pilot has no
        # other channel (plane state, true speed, battery)
        for style in (*P.SCRIPTED_STYLES, "switching"):
            with self.subTest(style=style):
                pilot = P.make_pilot(style, self.physics, {})
                self.assertEqual((pilot.observation, pilot.leaks), ("hidden", False))
                flight = P.fly(pilot, P.flight_seed(style, 0), flight_rng(style, 0))
                replay = P.make_pilot(style, self.physics, {})
                replay.reset(flight_rng(style, 0), flight["observations"][0], flight["seed"])
                actions = [replay.act(flight["observations"][t], t) for t in range(flight["length"])]
                np.testing.assert_array_equal(np.array(actions, np.float32), flight["prev_actions"][1:])

    def test_below_cap_pilots_never_reach_the_cap_and_probes_do(self):
        for index in range(6):
            flight = P.fly(P.make_pilot("steady_below", self.physics, {}), P.flight_seed("steady_below", index),
                           flight_rng("steady_below", index))
            self.assertFalse(flight["saturated"].any())
            self.assertLessEqual(np.abs(flight["prev_actions"][:, 0]).max(), P.BELOW_CAP_THROTTLE)
        for index in range(6):
            pilot = P.make_pilot("probe_hold", self.physics, {})
            flight = P.fly(pilot, P.flight_seed("probe_hold", index), flight_rng("probe_hold", index))
            # The action before observation t was taken at step t - 1
            probing = np.array([pilot.probing(t - 1) for t in range(1, flight["length"] + 1)])
            self.assertFalse(flight["saturated"][1:][~probing].any())
            self.assertTrue(flight["saturated"][1:][probing].any())

    def test_flights_are_recorded_as_the_encoder_reads_them(self):
        style = "random_hold"
        flight = P.fly(P.make_pilot(style, self.physics, {}), P.flight_seed(style, 3), flight_rng(style, 3))
        n = flight["length"]
        self.assertEqual(flight["observations"].shape, (n + 1, 4))
        self.assertEqual(flight["prev_actions"].shape, (n + 1, 2))
        np.testing.assert_array_equal(flight["prev_actions"][0], [0, 0])
        self.assertEqual(flight["batteries"][0], np.float32(flight["b0"]))
        # The same actions in a hidden-mode plane give the recorded observations and batteries
        env = bp.BatteryPlaneEnv(observation_mode="hidden")
        obs, _ = env.reset(seed=flight["seed"])
        hidden, batteries = [obs], [env.core.b0[0]]
        for action in flight["prev_actions"][1:]:
            obs, _, _, _, info = env.step(action)
            hidden.append(obs)
            batteries.append(info["b"])
        np.testing.assert_array_equal(np.array(hidden), flight["observations"])
        np.testing.assert_allclose(np.array(batteries, np.float32), flight["batteries"])
        # Reproducible, and never on a validation or final-test seed
        again = P.fly(P.make_pilot(style, self.physics, {}), P.flight_seed(style, 3), flight_rng(style, 3))
        np.testing.assert_array_equal(again["prev_actions"], flight["prev_actions"])
        with self.assertRaisesRegex(ValueError, "validation or final-test"):
            P.fly(P.make_pilot(style, self.physics, {}), 1001, flight_rng(style, 0))

    def test_replayed_good_pilots_do_not_depend_on_the_recorded_battery(self):
        with tempfile.TemporaryDirectory() as folder:
            good = saved_policy(Path(folder), "S0a_observed_bs256")
            blind = saved_policy(Path(folder), "S0a_blind_bs256")
            driver = P.PolicyDriver(good)
            self.assertTrue(driver.battery_aware)
            self.assertFalse(P.PolicyDriver(blind).battery_aware)
            for rng_seed in range(4):   # deterministic and stochastic flights
                replay = P.make_pilot("replay_good", self.physics, {"good": [good]})
                low, high = actions_with_battery(replay, 0.25, rng_seed), actions_with_battery(replay, 0.65, rng_seed)
                n = min(len(low), len(high))
                np.testing.assert_array_equal(low[:n], high[:n])
                # Reproducible, stochastic flights included
                np.testing.assert_array_equal(actions_with_battery(replay, 0.25, rng_seed), low)
            # Flying the recorded plane itself, the same pilot's actions follow its battery: flagged as leaking
            aware = P.make_pilot("aware_good", self.physics, {"good": [good]})
            self.assertTrue(aware.leaks and aware.observation == "observed")
            self.assertFalse(np.array_equal(actions_with_battery(aware, 0.25, 0), actions_with_battery(aware, 0.65, 0)))
            # The memoryless pilots must be battery-blind
            with self.assertRaisesRegex(ValueError, "battery-blind"):
                P.make_pilot("blind_ppo", self.physics, {"blind": [good]})
            flight = P.fly(P.make_pilot("blind_ppo", self.physics, {"blind": [blind]}), P.flight_seed("blind_ppo", 0),
                           flight_rng("blind_ppo", 0))
            self.assertGreater(flight["length"], 0)

    def test_seeds_never_overlap(self):
        # Recorded flights: one seed per flight, above every training reset seed (at most seed * 100 + 7 for the seeds
        # 0-7 of the runs) and never a validation or final-test seed
        recorded = [P.flight_seed(style, i) for style in P.STYLES for i in range(P.DEFAULT_FLIGHTS[style])]
        self.assertEqual(len(set(recorded)), len(recorded))
        self.assertGreater(min(recorded), 7 * 100 + 7)
        self.assertFalse(set(recorded) & P.HELD_OUT_SEEDS)
        # Shadow planes: one seed per flight and shadow flight, above every recorded flight seed
        shadows = [P.shadow_seed(seed, k) for seed in recorded[::50] for k in range(P.SEEDS_PER_FLIGHT)]
        self.assertEqual(len(set(shadows)), len(shadows))
        self.assertGreater(min(shadows), max(recorded))
        # Action-sampling seeds: one per flight and policy start
        sampling = [P.sampling_seed(seed, k) for seed in recorded[::50] for k in range(P.SEEDS_PER_FLIGHT)]
        self.assertEqual(len(set(sampling)), len(sampling))
        # Switching segments: one seed per flight and segment, and the seeds derived from them in ranges of their own
        segments = [P.segment_seed(seed, j) for seed in recorded[::50] for j in range(P.SEEDS_PER_FLIGHT)]
        self.assertEqual(len(set(segments)), len(segments))
        self.assertGreater(min(segments), max(shadows))
        derived = ([P.shadow_seed(seed, k) for seed in segments[::97] for k in range(P.SEEDS_PER_FLIGHT)]
                   + [P.sampling_seed(seed, k) for seed in segments[::97] for k in range(P.SEEDS_PER_FLIGHT)])
        self.assertGreater(min(derived), max(segments))
        self.assertFalse(set(derived) & (set(shadows) | set(sampling) | set(recorded)))
        with self.assertRaises(ValueError):
            P.shadow_seed(recorded[0], P.SEEDS_PER_FLIGHT)

    def test_stochastic_pilots_get_their_own_noise_in_every_flight(self):
        with tempfile.TemporaryDirectory() as folder:
            good = saved_policy(Path(folder), "S0a_observed_bs256")
            blind = saved_policy(Path(folder), "S0a_blind_bs256")
            # Models load first: loading re-seeds torch with the model's training seed, once per model and worker
            pilots = {style: P.make_pilot(style, self.physics, policies) for style, policies in
                      (("blind_ppo", {"blind": [blind]}), ("replay_good", {"good": [good]}), ("aware_good", {"good": [good]}),
                       ("switching", {"blind": [blind], "good": [good]}))}
            seeds_used = []
            original = torch.manual_seed

            def record(seed):
                seeds_used.append(seed)
                return original(seed)

            torch.manual_seed = record
            try:
                for style, pilot in pilots.items():
                    for index in range(12):
                        P.fly(pilot, P.flight_seed(style, index), flight_rng(style, index))
            finally:
                torch.manual_seed = original
            # Every stochastic start, in every flight and style, had its own seed
            self.assertGreater(len(seeds_used), 10)
            self.assertEqual(len(set(seeds_used)), len(seeds_used))
            # A stochastic replay: two flights fly different noise; the same flight twice the same
            pilot = P.make_pilot("replay_good", self.physics, {"good": [good]})
            flights = {}
            for index in range(40):
                rng = flight_rng("replay_good", index)
                flight = P.fly(pilot, P.flight_seed("replay_good", index), rng)
                if not pilot.deterministic:
                    flights[index] = flight
                if len(flights) == 2:
                    break
            (i, a), (j, b) = flights.items()
            n = min(a["length"], b["length"])
            self.assertFalse(np.array_equal(a["prev_actions"][1:n], b["prev_actions"][1:n]))
            again = P.fly(pilot, P.flight_seed("replay_good", i), flight_rng("replay_good", i))
            np.testing.assert_array_equal(again["prev_actions"], a["prev_actions"])

    def test_switching_never_ties_gentle_flying_to_a_low_battery(self):
        pilot = P.make_pilot("switching", self.physics, {})
        self.assertFalse(pilot.leaks)
        # The plan is drawn at take-off: the same flight with a low or a high battery switches at the same times to the
        # same styles
        actions_with_battery(pilot, 0.25, 3, steps=700)
        low = list(pilot.plan)
        actions_with_battery(pilot, 0.65, 3, steps=700)
        self.assertEqual(pilot.plan, low)
        self.assertGreater(len(low), 2)
        # A segment's style doesn't depend on the time: gentle styles are as common late in a flight as early
        early, late = [], []
        for index in range(400):
            pilot.reset(flight_rng("switching", index), np.zeros(4, np.float32), P.flight_seed("switching", index))
            for start, style in pilot.plan:
                (early if start < 350 else late).append(style in P.GENTLE_STYLES)
        self.assertLess(abs(np.mean(early) - np.mean(late)), 0.05)
        # Flown: the battery when a gentle segment starts is the battery when a hard one starts (no "low -> gentle")
        gentle, hard = [], []
        for index in range(150):
            flight = P.fly(pilot, P.flight_seed("switching", index), flight_rng("switching", index))
            for start, style in pilot.plan:
                if 0 < start <= flight["length"]:
                    battery = flight["batteries"][start]
                    if style in P.GENTLE_STYLES:
                        gentle.append(battery)
                    elif style in P.HARD_STYLES:
                        hard.append(battery)
        self.assertGreater(min(len(gentle), len(hard)), 100)
        self.assertLess(abs(np.mean(gentle) - np.mean(hard)), 0.05)

    def test_leaking_pilots_stay_out_of_the_main_mixes(self):
        for mix in ("narrow", "main"):
            self.assertFalse(set(P.MIXES[mix]) & P.LEAKING_STYLES)
        self.assertFalse(set(P.SWITCH_STYLES) & P.LEAKING_STYLES)
        self.assertEqual(P.LEAKING_STYLES, {style for style in P.STYLES if style == "aware_good"})
        self.assertTrue(set(P.MIXES["main_plus_aware"]) >= P.LEAKING_STYLES)
        with tempfile.TemporaryDirectory() as folder:
            good = saved_policy(Path(folder), "S0a_observed_bs256")
            # A policy trained with other physics can't fly the flights
            with self.assertRaisesRegex(ValueError, "env.kwargs"):
                P.make_pilot("replay_good", self.physics, {"good": [good]}, env_kwargs={"E_hover_s": 80.0})

    def test_clock_r2_measures_how_much_a_clock_explains(self):
        steps = np.arange(100)
        same = [0.6 - 0.004 * steps for _ in range(5)]
        self.assertAlmostEqual(P.clock_r2(same), 1.0)
        rng = np.random.default_rng(0)
        random = [rng.uniform(0, 1, 100) for _ in range(50)]
        self.assertLess(abs(P.clock_r2(random)), 0.05)

    def test_dataset_generation_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "data"
            styles = ["full_throttle", "steady_below", "probe_hold"]
            result = subprocess.run([sys.executable, str(ROOT / "pilots.py"), "--output", str(out), "--styles", *styles,
                                     "--flights", "3", "--jobs", "2", "--batch", "2"], capture_output=True, text=True,
                                    cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(set(manifest["styles"]), set(styles))
            self.assertEqual(list(manifest["mixes"]), ["narrow"])
            self.assertEqual(manifest["styles"]["full_throttle"]["cap_fraction"], 1.0)
            self.assertEqual(manifest["styles"]["steady_below"]["cap_fraction"], 0.0)
            data = P.load_flights(out, styles)
            self.assertEqual(len(data["batteries"]), 9)
            with np.load(out / "probe_hold.npz") as arrays:
                offsets, lengths = arrays["flight_offsets"], arrays["lengths"]
                np.testing.assert_array_equal(np.diff(offsets), lengths + 1)
                self.assertTrue((arrays["prev_actions"][offsets[:-1]] == 0).all())
                self.assertFalse(set(arrays["seeds"].tolist()) & P.HELD_OUT_SEEDS)
            # The same command again gives the same flights
            again = Path(folder) / "again"
            subprocess.run([sys.executable, str(ROOT / "pilots.py"), "--output", str(again), "--styles", "probe_hold",
                            "--flights", "3", "--jobs", "1"], check=True, capture_output=True, cwd=ROOT)
            with np.load(out / "probe_hold.npz") as a, np.load(again / "probe_hold.npz") as b:
                np.testing.assert_array_equal(a["prev_actions"], b["prev_actions"])


if __name__ == "__main__":
    unittest.main()
