"""CPU checks of step 4 (ENCODER_PLAN.md), offline encoder training: python -m unittest checks.step4_checks -v

The checks that train read the pilot flights of dataset v2 (runs_encoder/pilot_flights/v2, gitignored) and are skipped
without it; the others use a few flights generated on the spot.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from context_wrapper import FrozenEncoderContext
import pilots as P
import train_encoder as T

ROOT = Path(__file__).resolve().parents[1]
V2 = T.DEFAULT_DATA
HAS_V2 = (V2 / "manifest.json").is_file()
STYLES = ["probe_hold", "steady_below", "full_throttle"]


def train(output_root: Path, *extra) -> Path:
    """A tiny training run (8 flights per split, 1 epoch) on dataset v2; its run directory."""
    T.main(["--design", "E3", "--seed", "0", "--output-root", str(output_root), "--max-flights", "8", "--max-epochs", "1",
            "--batch-size", "4", "--threads", "1", "--wandb-mode", "disabled", "--log-every", "1000", *extra])
    return output_root / "E3_point_b0_main_s0"


class Step4Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.folder = tempfile.TemporaryDirectory()
        cls.data = Path(cls.folder.name) / "data"
        subprocess.run([sys.executable, str(ROOT / "pilots.py"), "--output", str(cls.data), "--styles", *STYLES,
                        "--flights", "20", "--jobs", "2"], check=True, capture_output=True, cwd=ROOT)
        cls.flights = T.Flights(P.load_flights(cls.data, STYLES))
        cls.cfg = T.encoder_config()

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_flights_are_split_by_their_seed_alone(self):
        self.assertEqual([T.split_of(s) for s in (100_007, 100_008, 100_009, 1_100_018, 2_100_029)],
                         ["train", "validation", "test", "validation", "test"])
        splits = self.flights.splits
        for split in ("train", "validation", "test"):
            rows = self.flights.indices(split)
            self.assertTrue(all(T.split_of(self.flights.seeds[i]) == split for i in rows))
            # Every style is on every side (20 flights each: 16 / 2 / 2)
            self.assertEqual({self.flights.styles[i] for i in rows}, set(STYLES))
        self.assertEqual(len(splits), len(self.flights.indices("train")) + len(self.flights.indices("validation"))
                         + len(self.flights.indices("test")))
        # A repeated flight would be on both sides of the split: refused
        data = P.load_flights(self.data, STYLES)
        data = {key: values + values[:1] for key, values in data.items()}
        with self.assertRaises(ValueError):
            T.Flights(data)

    def test_the_encoder_reads_only_observations_and_previous_actions(self):
        rows = self.flights.indices("train")[:3]
        batch = self.flights.batch(rows)
        self.assertEqual(batch["observations"].shape[-1], 4)
        self.assertEqual(batch["prev_actions"].shape[-1], 2)
        for j, i in enumerate(rows):
            n = self.flights.lengths[i]
            np.testing.assert_array_equal(batch["observations"][j, :n].numpy(), self.flights.observations[i])
            np.testing.assert_array_equal(batch["prev_actions"][j, :n].numpy(), self.flights.prev_actions[i])
            np.testing.assert_array_equal(batch["labels"][j, :n].numpy(), self.flights.batteries[i])
            self.assertTrue((batch["prev_actions"][j, 0] == 0).all())
        # The hidden-mode observation: no battery column
        encoder = T.make_encoder(self.cfg, b0_head=True)
        self.assertEqual((encoder.observation_dim, encoder.action_dim), (4, 2))
        # b0, the E3 label, is the starting battery the plane was reset with
        with np.load(self.data / "probe_hold.npz") as arrays:
            starts = arrays["batteries"][arrays["flight_offsets"][:-1]]
            np.testing.assert_allclose(starts, arrays["privileged_b0"], atol=1e-6)
        probe = [i for i, s in enumerate(self.flights.styles) if s == "probe_hold"]
        np.testing.assert_allclose(self.flights.batch(probe)["b0"].numpy(), starts, atol=1e-6)

    def test_padding_never_changes_the_estimates_or_the_loss(self):
        torch.manual_seed(0)
        encoder = T.make_encoder(self.cfg, b0_head=True)
        short, long = (int(i) for i in np.argsort(self.flights.lengths)[[0, -1]])
        self.assertLess(self.flights.lengths[short], self.flights.lengths[long])
        with torch.no_grad():
            alone_b, alone_b0 = T.predict(encoder, self.flights.batch([short]), True)
            batch = self.flights.batch([short, long])
            together_b, together_b0 = T.predict(encoder, batch, True)
        n = self.flights.lengths[short]
        torch.testing.assert_close(together_b[0, :n], alone_b[0], rtol=0, atol=1e-6)
        torch.testing.assert_close(together_b0[0, :n], alone_b0[0], rtol=0, atol=1e-6)
        # The padded steps count nowhere: the loss is the mean over the real steps
        errors = np.concatenate([together_b[j, :self.flights.lengths[i]].numpy() - self.flights.batteries[i]
                                 for j, i in enumerate([short, long])])
        loss = T.masked_mse(together_b, batch["labels"], batch["mask"])
        self.assertAlmostEqual(float(loss), float(np.mean(errors.astype(np.float64) ** 2)), places=5)

    def test_batches_cover_every_training_flight_once(self):
        rows = self.flights.indices("train")
        lengths = self.flights.lengths
        first = T.length_batches(rows, lengths, 5, np.random.default_rng([0, 4]))
        self.assertEqual(sorted(np.concatenate(first).tolist()), sorted(rows.tolist()))
        self.assertTrue(all(len(b) <= 5 for b in first))
        again = T.length_batches(rows, lengths, 5, np.random.default_rng([0, 4]))
        other = T.length_batches(rows, lengths, 5, np.random.default_rng([1, 4]))
        self.assertTrue(all((a == b).all() for a, b in zip(first, again)))
        self.assertFalse(all(len(a) == len(b) and (a == b).all() for a, b in zip(first, other)))

    def test_clock_estimate_and_error_tally(self):
        data = {"observations": [np.zeros((n, 4)) for n in (3, 2)], "prev_actions": [np.zeros((n, 2)) for n in (3, 2)],
                "batteries": [np.array([0.6, 0.5, 0.4]), np.array([0.4, 0.3])], "styles": ["a", "b"], "seeds": [0, 1]}
        flights = T.Flights(data)
        clock = T.clock_estimate(flights, np.array([0, 1]))
        np.testing.assert_allclose(clock, [0.5, 0.4, 0.4])
        result = T.evaluate_clock(clock, flights, np.array([0, 1]), {"t1": 1, "t2": 2})
        squared = [0.01, 0.01, 0.0, 0.01, 0.01]
        self.assertAlmostEqual(result["mse"], np.mean(squared))
        self.assertAlmostEqual(result["rmse_at"]["t1"], 0.1)          # both flights at step 1: errors 0.1 and 0.1
        self.assertAlmostEqual(result["rmse_at"]["t2"], 0.0)          # only the first flight reaches step 2
        self.assertEqual(result["flights_at"], {"t1": 2, "t2": 1})
        self.assertAlmostEqual(result["rmse_by_style"]["b"], 0.1)

    def test_saved_model_holds_the_trained_encoder_for_the_wrapper(self):
        torch.manual_seed(3)
        encoder = T.make_encoder(self.cfg, b0_head=True)
        with tempfile.TemporaryDirectory() as folder:
            T.save_policy_model(self.cfg, encoder, Path(folder) / "model.zip")
            source = FrozenEncoderContext(1, Path(folder) / "model.zip")
        # The same weights, without the b0 head
        self.assertIsNone(source.encoder.auxiliary_projection)
        trained = {k: v for k, v in encoder.state_dict().items() if not k.startswith("auxiliary_projection.")}
        saved = source.encoder.state_dict()
        self.assertEqual(set(saved), set(trained))
        for key, value in trained.items():
            torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
        # Stepped by the wrapper, one observation at a time, it gives what training saw over the whole flight
        i = int(self.flights.indices("test")[0])
        with torch.no_grad():
            expected, _ = T.predict(encoder.eval(), self.flights.batch([i]), True)
        observations, previous = self.flights.observations[i], self.flights.prev_actions[i]
        stepped = [source.step(np.array([True]), observations[t:t + 1], previous[t:t + 1], np.array([t == 0]),
                               np.array([False]), [{}])[0, 0] for t in range(len(observations))]
        np.testing.assert_allclose(stepped, expected[0].numpy(), atol=1e-5)

    @unittest.skipUnless(HAS_V2, "needs dataset v2 (runs_encoder/pilot_flights/v2)")
    def test_training_end_to_end_is_reproducible_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            first = train(Path(folder) / "a")
            again = train(Path(folder) / "b")
            for name in ("config.json", "metrics.jsonl", "best_encoder.pt", "last_encoder.pt", "encoder_model.zip", "result.json"):
                self.assertTrue((first / name).is_file(), name)
            a = torch.load(first / "best_encoder.pt", weights_only=False)["state_dict"]
            b = torch.load(again / "best_encoder.pt", weights_only=False)["state_dict"]
            self.assertTrue(any(k.startswith("auxiliary_projection.") for k in a))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            settings = json.loads((first / "config.json").read_text())
            result = json.loads((first / "result.json").read_text())
            self.assertEqual(settings["leaking_styles"], [])
            self.assertEqual(set(settings["split"]["flights"]), {"train", "validation", "test"})
            self.assertEqual(settings["steps_at"], {"1s": 20, "5s": 100, "20s": 400})
            self.assertTrue(all(T.split_of(s) == "test" for s in result["test_seeds"]))
            self.assertFalse(set(result["test_seeds"]) & P.HELD_OUT_SEEDS)
            self.assertIn("b0_rmse", result["test"])
            # 8 test flights here (every flight in real runs): styles of the main mix only, several of them
            styles = set(result["test"]["rmse_by_style"])
            self.assertTrue(styles <= set(P.MIXES["main"]) and len(styles) > 1, styles)
            # The run directory is never overwritten
            with self.assertRaises(SystemExit):
                train(Path(folder) / "a")
            # Flights recorded with other physics settings are refused
            other = Path(folder) / "other_physics"
            other.mkdir()
            for style in P.MIXES["main"]:
                (other / f"{style}.npz").symlink_to(V2 / f"{style}.npz")
            manifest = json.loads((V2 / "manifest.json").read_text())
            manifest["env_kwargs"] = {"T": 600}
            (other / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(SystemExit):
                train(Path(folder) / "c", "--data", str(other))


if __name__ == "__main__":
    unittest.main()
