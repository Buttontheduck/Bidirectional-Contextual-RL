"""CPU checks of the flight logging (flight_logging.py): python -m unittest checks.logging_checks -v"""
from types import SimpleNamespace
import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import KVWriter, Logger
import torch

import flight_logging
import main_battery_plane_contextual_ppo as training
from checks.step0_checks import SMALL, config

CORE = SimpleNamespace(v_min=52.0, v_goal=62.0, dt=0.05)


def step(vx, y, b, saturated=False, done=False, complete=False, b0=0.2):
    return {"b0": b0, "b": b, "vx": vx, "y": y, "saturated": saturated, "mission_complete": complete}, done


class Capture(KVWriter):
    """Every value the SB3 logger dumps, by key."""

    def __init__(self):
        self.values = {}

    def write(self, key_values, key_excluded, step=0):
        for key, value in key_values.items():
            self.values.setdefault(key, []).append(value)


class LoggingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_flight_metrics_follow_the_env_definitions(self):
        recorder = flight_logging.FlightRecorder(CORE, plot_batteries=[0.2])
        flights = [
            # b0 = 0.2: completed; 3 of 4 steps at or above v_min, none at 90% of the main reward (61.49 m/s)
            [(step(50, 10, 0.18, True), (1, 0)), (step(53, 20, 0.15, True), (1, 0)),
             (step(60, 30, 0.12), (0.5, -1)), (step(61, 40, 0.1, done=True, complete=True), (0.5, -1))],
            # b0 = 0.7: the battery empties while touching the ground; battery first
            [(step(62, 5, 0.3, b0=0.7), (1, 1)), (step(63, -1, 0.0, done=True, b0=0.7), (1, 1))],
        ]
        for flight in flights:
            for (info, done), action in flight:
                recorder.record({"n_envs": 1, "i": 0, "info": info, "reward": 1.0, "done": done,
                                 "actions": np.array([action], dtype=float)})
        first, second = recorder.flights
        self.assertEqual((first["outcome"], second["outcome"]), ("completed", "battery_empty"))
        self.assertEqual((first["time_in_band"], first["time_at_goal"], first["saturation_rate"]), (0.75, 0.0, 0.5))
        self.assertAlmostEqual(first["distance_m"], 224 * 0.05)
        self.assertEqual((first["altitude_mean"], first["altitude_min"], first["throttle_mean"], first["lift_mean"]),
                         (25.0, 10.0, 0.75, -0.5))
        self.assertAlmostEqual(first["battery_used"], 0.1)
        self.assertAlmostEqual(first["mean_abs_action"], 0.625)
        self.assertAlmostEqual(first["mean_action_change"], 0.25)
        self.assertEqual(second["time_at_goal"], 1.0)

        metrics = recorder.metrics()
        self.assertEqual((metrics["completion_rate"], metrics["battery_depleted_rate"], metrics["ground_crash_rate"]),
                         (0.5, 0.5, 0.0))
        self.assertAlmostEqual(metrics["final_battery_completed_mean"], 0.1)
        self.assertEqual(metrics["by_b0/b0_20pct/completion_rate"], 1.0)
        self.assertEqual(metrics["by_b0/b0_70pct/battery_depleted_rate"], 1.0)
        self.assertAlmostEqual(metrics["by_b0/throttle_mean_range"], 0.25)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(set(recorder.traces), {0.2})
        image = recorder.plot("check")
        self.assertEqual((image.ndim, image.shape[-1], image.dtype), (3, 3, np.uint8))

    def test_both_learners_log_the_same_flight_metrics(self):
        logged = {}
        for learner in ("cppo", "ppo"):
            cfg = config(*SMALL, "evaluation.reset_grid.eval_seeds=[1000]",
                         "evaluation.reset_grid.eval_initial_battery=[0.3,0.6]")
            env = training.make_train_env(cfg)
            schedule = list(training.iter_eval_resets(cfg.evaluation.reset_grid))
            eval_env, scheduler, raw_eval_env = training.make_eval_env(cfg, schedule)
            original = torch.nn.utils.clip_grad_norm_
            try:
                if learner == "cppo":
                    model = training.build_model(cfg, env)
                else:
                    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, seed=0, device="cpu")
                capture = Capture()
                model.set_logger(Logger(folder=None, output_formats=[capture]))
                callbacks = [
                    training.RenderingEvalCallback(eval_env, eval_freq=8, n_eval_episodes=len(schedule),
                                                   reset_scheduler=scheduler, flight_core=raw_eval_env.core,
                                                   plot_every=32, plot_batteries=[0.3, 0.6], verbose=0),
                    flight_logging.RolloutFlightLogger(raw_eval_env.core),
                    flight_logging.GradNormLogger(),
                ]
                model.learn(total_timesteps=64, callback=callbacks)
            finally:
                env.close()
                eval_env.close()
            # The wrapped clipping function is restored once training ends
            self.assertIs(torch.nn.utils.clip_grad_norm_, original)
            values = capture.values
            # 2 envs x 8 steps per rollout: 2 minibatches x 1 epoch for both learners
            self.assertEqual(set(values["train/optimizer_steps"]), {2})
            self.assertEqual(len(values["validation/completion_rate"]), len(values["eval/mean_reward"]))
            self.assertEqual(len(values["episode_plots/validation"]), 2)
            self.assertIn("rollout/saturation_rate", values)
            logged[learner] = {key for key in values if key.split("/")[0] in ("validation", "rollout", "episode_plots")}
            logged[learner] |= {key for key in values if key.startswith("train/grad")}
        self.assertEqual(logged["cppo"], logged["ppo"])
        self.assertIn("validation/by_b0/b0_30pct/throttle_mean", logged["cppo"])

    def test_plot_settings_are_validated(self):
        training.validate_config(config())
        for override in ("evaluation.plots.frequency=75000", "evaluation.plots.initial_battery=[0.25]"):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "evaluation.plots"):
                training.validate_config(config(override))


if __name__ == "__main__":
    unittest.main()
