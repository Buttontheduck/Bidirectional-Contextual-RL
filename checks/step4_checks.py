"""CPU checks of step 4 (ENCODER_PLAN.md), offline encoder training, its floor, the probes and the standalone encoders:
python -m unittest checks.step4_checks -v

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

from context_wrapper import ContextVecEnv, FrozenEncoderContext, StandaloneEncoderContext, make_context_source
import pilots as P
import recurrent_encoders as R
import step4_floor as F
import step4_probes as PR
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

    def test_floor_reads_hidden_inputs_and_agrees_with_the_physics(self):
        grid = {"b_points": 41, "vx_points": 121, "prune_below": 30.0}
        rows = range(len(self.flights.seeds))
        flights = [(self.flights.seeds[i], self.flights.observations[i], self.flights.prev_actions[i]) for i in rows]
        # Every flight is explained, including those that end with an empty battery or on the ground
        estimates = F.estimate_flights(flights, grid, {})
        for i, flight in zip(rows, estimates):
            n = self.flights.lengths[i]
            self.assertEqual((len(flight["means"]), len(flight["stds"])), (n, n))
            errors = np.abs(flight["means"] - self.flights.batteries[i])
            if self.flights.styles[i] == "full_throttle":
                # At the cap all flight: the battery is known within a few percent by 20 s (or the flight's end)
                self.assertLess(errors[min(400, n - 1)], 0.05)
            if self.flights.styles[i] == "steady_below":
                # Never at the cap: the battery stays uncertain
                self.assertGreater(flight["stds"][-1], 0.08)
        # Nothing carries over between flights in a worker: a flight alone or after others gets the same estimates,
        # so the results don't depend on --jobs or --chunk
        alone = F.estimate_flights(flights[-1:], grid, {})[0]
        np.testing.assert_array_equal(alone["means"], estimates[-1]["means"])
        np.testing.assert_array_equal(alone["stds"], estimates[-1]["stds"])
        # An observation with a battery column is refused
        seed, observations, prev_actions = flights[0]
        with_battery = np.concatenate([observations, np.full((len(observations), 1), 0.5, np.float32)], axis=1)
        with self.assertRaises(ValueError):
            F.estimate_flights([(seed, with_battery, prev_actions)], grid, {})

    def test_probe_labels_come_from_the_flights_as_defined(self):
        data = P.load_flights(self.data, STYLES)
        dt, E = 0.05, 100.0
        count, find, battery = (PR.task_labels(task, data, dt, E) for task in ("count", "find", "battery"))
        for i in range(len(data["seeds"])):
            prev = data["prev_actions"][i].astype(np.float64)
            # count: zero at the start, then a_{t-1}^2 * dt / E added at every step (inputs only)
            self.assertEqual(count[i][0], 0.0)
            # (float32 labels: rounding of about 1e-7, against increments of about 1e-4)
            np.testing.assert_allclose(np.diff(count[i]), dt / E * (prev[1:] ** 2).sum(axis=1), atol=1e-6)
            self.assertTrue((np.diff(count[i]) <= 2 * dt / E + 1e-6).all())
            # find: the starting battery at every step; battery: the true battery
            np.testing.assert_array_equal(find[i], np.full(len(find[i]), data["batteries"][i][0], np.float32))
            np.testing.assert_array_equal(battery[i], data["batteries"][i])
        # count never reads the battery: other batteries, the same labels
        changed = {**data, "batteries": [b * 0.5 for b in data["batteries"]]}
        for a, b in zip(count, PR.task_labels("count", changed, dt, E)):
            np.testing.assert_array_equal(a, b)

    def test_no_encoder_looks_ahead(self):
        # Changing a flight after step t0 never changes the estimates up to t0 (e.g. how long the flight lasts, which on
        # full-throttle flights gives the starting battery away, can't reach an earlier estimate)
        i, t0 = int(np.argmax(self.flights.lengths)), 30
        batch = self.flights.batch([i])
        changed = {key: value.clone() for key, value in batch.items()}
        changed["observations"][:, t0 + 1:] += torch.randn_like(changed["observations"][:, t0 + 1:])
        changed["prev_actions"][:, t0 + 1:] = -changed["prev_actions"][:, t0 + 1:]
        changed["mask"][:, t0 + 40:] = False                                  # and a shorter flight
        for arch in PR.ARCHS:
            torch.manual_seed(0)
            model = PR.make_model(arch, self.cfg, 32, 2).eval()
            with torch.no_grad():
                before, after = PR.predict(model, arch, batch), PR.predict(model, arch, changed)
            torch.testing.assert_close(after[0, :t0 + 1], before[0, :t0 + 1], rtol=0, atol=1e-6, msg=arch)
            self.assertFalse(torch.allclose(after[0, t0 + 1:t0 + 40], before[0, t0 + 1:t0 + 40]), arch)

    def test_gru_and_lstm_read_the_same_inputs_and_padding_changes_nothing(self):
        short, long = (int(i) for i in np.argsort(self.flights.lengths)[[0, -1]])
        for arch in ("gru", "lstm"):
            torch.manual_seed(0)
            model = PR.make_model(arch, self.cfg, 32, 2).eval()
            self.assertEqual(model.input_projection.in_features, 6)       # o_t (4) and a_{t-1} (2): no battery
            with torch.no_grad():
                alone = PR.predict(model, arch, self.flights.batch([short]))
                together = PR.predict(model, arch, self.flights.batch([short, long]))
            n = self.flights.lengths[short]
            torch.testing.assert_close(together[0, :n], alone[0], rtol=0, atol=1e-6)
            self.assertTrue(((together > 0) & (together < 1)).all())

    @unittest.skipUnless(HAS_V2, "needs dataset v2 (runs_encoder/pilot_flights/v2)")
    def test_probes_end_to_end_are_reproducible_and_refuse_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            def probe(root, *extra):
                PR.main(["--arch", "gru", "--task", "find", "--seed", "1", "--output-root", str(root), "--max-flights", "8",
                         "--max-epochs", "1", "--batch-size", "4", "--threads", "1", "--hidden", "16", "--wandb-mode",
                         "disabled", "--log-every", "1000", *extra])
                return root / "probe_gru_find_narrow_n5_max1_s1"           # 1 epoch, not the default 20
            first, again = probe(Path(folder) / "a", "--train-flights", "5"), probe(Path(folder) / "b", "--train-flights", "5")
            a = torch.load(first / "best.pt", weights_only=False)["state_dict"]
            b = torch.load(again / "best.pt", weights_only=False)["state_dict"]
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            settings = json.loads((first / "config.json").read_text())
            self.assertEqual(settings["split"]["flights"], {"train": 5, "validation": 8, "test": 8})
            self.assertEqual(settings["styles"], ["full_throttle"])
            result = json.loads((first / "result.json").read_text())
            self.assertEqual(set(result["test"]["rmse_by_style"]), {"full_throttle"})
            with self.assertRaises(SystemExit):
                probe(Path(folder) / "a", "--train-flights", "5")

    @unittest.skipUnless(HAS_V2, "needs dataset v2 (runs_encoder/pilot_flights/v2)")
    def test_floor_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "floor"
            args = ["--output", str(out), "--max-flights", "6", "--jobs", "2", "--chunk", "3", "--b-points", "41",
                    "--vx-points", "121"]
            # In a fresh process: it never loads torch, W&B or Hydra, so it can't touch the training runs' W&B
            code = (f"import sys, step4_floor as F; F.main({args!r}); "
                    "print('loaded:', sorted(m for m in ('torch', 'wandb', 'hydra') if m in sys.modules))")
            run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr[-2000:])
            self.assertIn("loaded: []", run.stdout)
            result = json.loads((out / "floor.json").read_text())
            self.assertGreater(result["validation"]["calibration_median"], 0)
            for split in ("validation", "test"):
                self.assertLess(result[split]["rmse"], result[f"clock_{split}"]["rmse"] + 0.05)
                self.assertLess(result[split]["max_vy_mismatch"], 0.05)
            with np.load(out / "estimates.npz") as arrays:
                seeds, offsets, splits = arrays["seeds"], arrays["flight_offsets"], arrays["splits"]
                self.assertEqual(len(seeds), 12)
                self.assertEqual(offsets[-1], len(arrays["means"]))
                self.assertTrue(all(T.split_of(s) == split != "train" for s, split in zip(seeds, splits)))
                self.assertFalse(set(seeds.tolist()) & P.HELD_OUT_SEEDS)
            with self.assertRaises(SystemExit):
                F.main(args)

    @staticmethod
    def small_encoder(arch, drop_tau, aux=True, seed=0):
        torch.manual_seed(seed)
        extra = {"conv_channels": 8, "window": 16} if arch == "convgru" else {}
        return R.build_encoder(arch, drop_tau=drop_tau, hidden=32, layers=2, head_arch=[16], aux_dim=1 if aux else None, **extra).eval()

    def test_standalone_encoders_give_the_same_estimates_over_a_flight_and_step_by_step(self):
        rows = [int(i) for i in np.argsort(self.flights.lengths)[[0, len(self.flights.lengths) // 2, -1]]]
        batch = self.flights.batch(rows)
        obs, prev, mask = batch["observations"], batch["prev_actions"], batch["mask"]
        for arch in R.ARCHITECTURES:
            for drop_tau in (False, True):
                encoder = self.small_encoder(arch, drop_tau)
                with torch.no_grad():
                    whole, whole_aux = encoder(obs, prev)
                    state, stepped, stepped_aux = encoder.initial_state(len(rows)), [], []
                    for t in range(obs.shape[1]):
                        estimate, aux, state = encoder.step(state, obs[:, t], prev[:, t])
                        stepped.append(estimate)
                        stepped_aux.append(aux)
                stepped, stepped_aux = torch.stack(stepped, 1), torch.stack(stepped_aux, 1)
                torch.testing.assert_close(stepped[mask], whole[mask], rtol=0, atol=1e-5, msg=f"{arch} drop_tau={drop_tau}")
                torch.testing.assert_close(stepped_aux[mask], whole_aux[mask], rtol=0, atol=1e-5)
                self.assertTrue(((whole > 0) & (whole < 1)).all())

    def test_standalone_encoders_never_look_ahead_and_ignore_tau_when_told(self):
        i, t0 = int(np.argmax(self.flights.lengths)), 30
        batch = self.flights.batch([i])
        obs, prev = batch["observations"], batch["prev_actions"]
        later = obs.clone()
        later[:, t0 + 1:] += torch.randn_like(later[:, t0 + 1:])
        other_tau = obs.clone()
        other_tau[..., R.TAU_INDEX] = torch.rand_like(other_tau[..., R.TAU_INDEX])
        for arch in R.ARCHITECTURES:
            for drop_tau in (False, True):
                encoder = self.small_encoder(arch, drop_tau)
                with torch.no_grad():
                    base, changed, tau_changed = (encoder(o, prev)[0] for o in (obs, later, other_tau))
                # The filter window and the GRU see only the past: estimates up to t0 don't move
                torch.testing.assert_close(changed[0, :t0 + 1], base[0, :t0 + 1], rtol=0, atol=1e-6, msg=arch)
                self.assertFalse(torch.allclose(changed[0, t0 + 1:t0 + 40], base[0, t0 + 1:t0 + 40]))
                if drop_tau:
                    torch.testing.assert_close(tau_changed, base, rtol=0, atol=0, msg=f"{arch} reads tau")
                else:
                    self.assertFalse(torch.allclose(tau_changed, base), arch)
        self.assertEqual(R.build_encoder("convgru", drop_tau=True).conv.in_channels, 6)    # y, v_x, v_y, a_x, a_y, valid

    def test_standalone_encoder_in_the_context_wrapper(self):
        from checks.step2_checks import episodes, vec_env
        from stable_baselines3 import PPO

        encoder = self.small_encoder("convgru", True, aux=True, seed=3)
        with tempfile.TemporaryDirectory() as folder:
            R.save_encoder(encoder, Path(folder) / "encoder.pt")
            source = StandaloneEncoderContext(2, Path(folder) / "encoder.pt")
            # context.source=encoder with a .pt model picks it
            from omegaconf import OmegaConf
            cfg = OmegaConf.create({"context": {"source": "encoder", "encoder_model": str(Path(folder) / "encoder.pt")}})
            self.assertIsInstance(make_context_source(cfg, 2, {}), StandaloneEncoderContext)
        self.assertEqual(source.dim, 1)
        self.assertTrue(all(not p.requires_grad for p in source.encoder.parameters()))
        before = {name: value.clone() for name, value in source.encoder.state_dict().items()}
        env = ContextVecEnv(vec_env("hidden", T=12), source)
        finished = episodes(env, 40)
        self.assertGreaterEqual(len(finished), 4)
        for episode in finished:
            observations = torch.as_tensor(np.array(episode["obs"]), dtype=torch.float32)
            previous = torch.as_tensor(np.concatenate([np.zeros((1, 2), np.float32), np.array(episode["actions"])]))
            # The wrapper's estimate at every step, final observation included, is the whole-flight estimate
            with torch.no_grad():
                expected = source.encoder(observations[None, :, :4], previous[None])[0][0].numpy()
            np.testing.assert_allclose(observations[:, 4].numpy(), expected, atol=1e-5)
        PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu", policy_kwargs={"net_arch": [16]}).learn(total_timesteps=32)
        for name, value in source.encoder.state_dict().items():
            torch.testing.assert_close(value, before[name])

    @unittest.skipUnless(HAS_V2, "needs dataset v2 (runs_encoder/pilot_flights/v2)")
    def test_probe_gru_exports_exactly_and_new_architectures_train_resume_and_finalize(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            # A probe GRU of the battery task, exported to encoder.pt, gives exactly the probe's estimates
            PR.main(["--arch", "gru", "--task", "battery", "--seed", "0", "--output-root", str(folder / "probe"), "--max-flights", "8",
                     "--max-epochs", "1", "--batch-size", "4", "--threads", "1", "--hidden", "16", "--wandb-mode", "disabled",
                     "--log-every", "1000"])
            probe = folder / "probe" / "probe_gru_battery_main_max1_s0"
            R.export_probe(probe, folder / "probe.pt")
            original = PR.make_model("gru", self.cfg, 16, 2)
            original.load_state_dict(torch.load(probe / "best.pt", weights_only=False)["state_dict"])
            batch = self.flights.batch(self.flights.indices("test")[:3])
            with torch.no_grad():
                torch.testing.assert_close(R.load_encoder(folder / "probe.pt")(batch["observations"], batch["prev_actions"])[0],
                                           original.eval()(batch["observations"], batch["prev_actions"]), rtol=0, atol=0)
            # train_encoder.py --arch convgru --drop-tau (E3): its name, its encoder.pt (the best checkpoint)
            options = ["--design", "E3", "--arch", "convgru", "--drop-tau", "--hidden", "16", "--head-arch", "8", "--seed", "0",
                       "--max-flights", "8", "--batch-size", "4", "--threads", "1", "--wandb-mode", "disabled", "--log-every", "1000"]
            T.main([*options, "--output-root", str(folder / "whole"), "--max-epochs", "2", "--patience", "5"])
            name = "E3_convgru_notau_h16l2_head8_point_b0_main_s0"
            whole = folder / "whole" / name
            settings = json.loads((whole / "config.json").read_text())
            self.assertEqual((settings["arch"], settings["drop_tau"], settings["conv_window"]), ("convgru", True, 16))
            self.assertTrue((whole / "encoder.pt").is_file() and not (whole / "encoder_model.zip").exists())
            saved = R.load_encoder(whole / "encoder.pt")
            best = torch.load(whole / "best_encoder.pt", weights_only=False)["state_dict"]
            for key, value in saved.state_dict().items():
                torch.testing.assert_close(value, best[key], rtol=0, atol=0)
            # Resume continues it exactly (with its architecture), and finalize gives its natural end
            T.main([*options, "--output-root", str(folder / "first"), "--max-epochs", "1"])
            T.main(["--resume", str(folder / "first" / name), "--max-epochs", "2", "--patience", "5", "--threads", "1",
                    "--output-root", str(folder / "resumed"), "--wandb-mode", "disabled", "--log-every", "1000"])
            resumed = folder / "resumed" / name.replace("_main_s0", "_main_resumed_s0")
            a, b = (torch.load(run / "last_encoder.pt", weights_only=False)["state_dict"] for run in (whole, resumed))
            for key in a:
                torch.testing.assert_close(b[key], a[key], rtol=0, atol=0)
            stopped = folder / "stopped"
            shutil.copytree(whole, stopped)
            (stopped / "result.json").unlink()
            (stopped / "encoder.pt").unlink()
            T.main(["--finalize", str(stopped), "--wandb-mode", "disabled", "--threads", "1"])
            ra, rb = (json.loads((run / "result.json").read_text()) for run in (whole, stopped))
            self.assertEqual(ra["test"], rb["test"])
            self.assertTrue((stopped / "encoder.pt").is_file())
            # The xLSTM refuses the standalone encoders' options
            with self.assertRaises(SystemExit):
                T.main(["--design", "E1", "--seed", "0", "--drop-tau", "--output-root", str(folder / "refused"), "--wandb-mode", "disabled"])

    @unittest.skipUnless(HAS_V2, "needs dataset v2 (runs_encoder/pilot_flights/v2)")
    def test_finalize_gives_what_a_natural_end_gives(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            same = ("best_epoch", "epochs", "steps", "validation", "test", "clock_validation", "clock_test")
            # train_encoder.py: a finished run, and a copy of it stopped by hand (no result.json, no model) and finalized
            natural = train(folder / "natural", "--max-epochs", "2", "--patience", "5")
            stopped = folder / "stopped"
            shutil.copytree(natural, stopped)
            (stopped / "result.json").unlink()
            (stopped / "encoder_model.zip").unlink()
            T.main(["--finalize", str(stopped), "--wandb-mode", "disabled", "--threads", "1"])
            a, b = (json.loads((run / "result.json").read_text()) for run in (natural, stopped))
            self.assertEqual({k: a[k] for k in (*same, "test_seeds")}, {k: b[k] for k in (*same, "test_seeds")})
            self.assertTrue(b["stopped_by_hand"])
            saved = [FrozenEncoderContext(1, run / "encoder_model.zip").encoder.state_dict() for run in (natural, stopped)]
            for key in saved[0]:
                torch.testing.assert_close(saved[1][key], saved[0][key], rtol=0, atol=0)
            # Refused: a run that has finished, and checkpoints that don't match metrics.jsonl
            mismatched = folder / "mismatched"
            shutil.copytree(natural, mismatched)
            (mismatched / "result.json").unlink()
            rows = (mismatched / "metrics.jsonl").read_text().splitlines()
            (mismatched / "metrics.jsonl").write_text(rows[0] + "\n")
            for run in (natural, mismatched):
                with self.assertRaises(SystemExit, msg=run):
                    T.main(["--finalize", str(run), "--wandb-mode", "disabled"])
            # step4_probes.py, the same way
            PR.main(["--arch", "gru", "--task", "find", "--seed", "1", "--output-root", str(folder / "probe"), "--max-flights", "8",
                     "--max-epochs", "2", "--patience", "5", "--batch-size", "4", "--threads", "1", "--hidden", "16",
                     "--wandb-mode", "disabled", "--log-every", "1000", "--train-flights", "5"])
            natural = folder / "probe" / "probe_gru_find_narrow_n5_p5_max2_s1"
            stopped = folder / "probe_stopped"
            shutil.copytree(natural, stopped)
            (stopped / "result.json").unlink()
            PR.main(["--finalize", str(stopped), "--wandb-mode", "disabled", "--threads", "1"])
            a, b = (json.loads((run / "result.json").read_text()) for run in (natural, stopped))
            self.assertEqual({k: a[k] for k in same}, {k: b[k] for k in same})
            self.assertTrue(b["stopped_by_hand"])

    @unittest.skipUnless(HAS_V2, "needs dataset v2 (runs_encoder/pilot_flights/v2)")
    def test_resume_continues_exactly_like_an_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            whole = train(folder / "whole", "--max-epochs", "2", "--patience", "5")
            first = train(folder / "first")                                   # 1 epoch, finished
            options = ["--output-root", str(folder / "resumed"), "--threads", "1", "--wandb-mode", "disabled", "--log-every", "1000"]
            T.main(["--resume", str(first), "--max-epochs", "2", "--patience", "5", *options])
            resumed = folder / "resumed" / "E3_point_b0_main_resumed_s0"
            # The same weights, optimizer state and step count as the run that never stopped
            a, b = (torch.load(run / "last_encoder.pt", weights_only=False) for run in (whole, resumed))
            self.assertEqual((a["epoch"], a["step"]), (b["epoch"], b["step"]))
            for key in a["state_dict"]:
                torch.testing.assert_close(b["state_dict"][key], a["state_dict"][key], rtol=0, atol=0)
            torch.testing.assert_close(b["optimizer"]["state"], a["optimizer"]["state"], rtol=0, atol=0)
            rows_a, rows_b = ([json.loads(line) for line in open(run / "metrics.jsonl")] for run in (whole, resumed))
            self.assertEqual([r["validation"] for r in rows_a], [r["validation"] for r in rows_b])
            settings = json.loads((resumed / "config.json").read_text())
            self.assertEqual(settings["resumed_from"]["name"], "E3_point_b0_main_s0")
            old_style = folder / "old_style"
            shutil.copytree(first, old_style)
            config = json.loads((old_style / "config.json").read_text())
            for key in T.ARCH_SETTINGS:
                config.pop(key)
            config["code_sha256"].pop("recurrent_encoders.py")
            (old_style / "config.json").write_text(json.dumps(config))
            T.main(["--resume", str(old_style), "--max-epochs", "2", "--patience", "5", "--output-root", str(folder / "old_resumed"),
                    "--threads", "1", "--wandb-mode", "disabled", "--log-every", "1000"])
            self.assertTrue((folder / "old_resumed" / "E3_point_b0_main_resumed_s0" / "encoder_model.zip").is_file())
            self.assertEqual((settings["group"], settings["patience"], settings["max_flights"]), ("E3_point_b0_main_resumed", 5, 8))
            self.assertTrue((resumed / "result.json").is_file() and (resumed / "encoder_model.zip").is_file())
            # Refused: an unfinished run, other training settings, no epochs left, and a patience it already ran out of
            unfinished = folder / "unfinished"
            shutil.copytree(first, unfinished)
            (unfinished / "result.json").unlink()
            stalled = folder / "stalled"                      # its second epoch made worse than its first
            shutil.copytree(whole, stalled)
            rows_a[1]["validation"]["mse"] = rows_a[0]["validation"]["mse"] + 1.0
            (stalled / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_a))
            for argv in (["--resume", str(unfinished), "--max-epochs", "2"], ["--resume", str(first), "--max-epochs", "2", "--batch-size", "8"],
                         ["--resume", str(first), "--max-epochs", "1"], ["--resume", str(stalled), "--max-epochs", "4", "--patience", "1"],
                         ["--resume", str(first), "--design", "E1", "--max-epochs", "2"]):
                with self.assertRaises(SystemExit, msg=argv):
                    T.main([*argv, "--output-root", str(folder / "refused"), "--wandb-mode", "disabled"])

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
