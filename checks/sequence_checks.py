"""CPU checks of what the end-to-end encoder trains on: python -m unittest checks.sequence_checks -v

cppo.supervised.sequence: episode (whole episodes from their start, full BPTT; the default), chunk (plain chunking:
windows of chunk_len steps from a blank memory) or stored_state (R2D2: windows from the memory saved at collection,
burn_in steps without loss, then chunk_len trained steps). Conditions J1_chunk and J1_r2d2 (config/joint/).
"""
from pathlib import Path
import unittest

from gymnasium import spaces
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from checks.joint_checks import flat
from checks.prelude_checks import SMALL_EASY
from checks.step0_checks import config
from contextual_ppo.episode_buffers import SupervisedContextualEpisodeBuffer, encoder_state_from_rows, encoder_state_rows
from contextual_ppo.torch_layers import GRURolloutEncoder, XLSTMRolloutEncoder
import main_battery_plane_contextual_ppo as training

# Short flights with several episodes per rollout; tiny windows so that some start mid-episode
WINDOWS = ["cppo.supervised.chunk_len=4", "cppo.supervised.burn_in=3", "cppo.supervised.chunk_batch_size=16"]
HARD = ["training.n_envs=2", "cppo.n_steps=32", "cppo.n_minibatches=2", "cppo.n_epochs=1", "cppo.net_arch=[16]", "cppo.verbose=0",
        "cppo.gru.hidden_dim=16", "cppo.gru.num_layers=2", "cppo.supervised.episode_batch_size=4", "cppo.supervised.gradient_steps=2",
        "+env.kwargs.T=30"]


def model_for(rung, *overrides, easy=False):
    cfg = config(f"+joint={rung}", *(SMALL_EASY if easy else HARD), *WINDOWS, *overrides, step0=False)
    training.validate_config(cfg)
    return training.build_model(cfg, training.make_train_env(cfg))


def collect(model, rollouts=4):
    _, callback = model._setup_learn(model.n_steps * model.n_envs * rollouts)
    for _ in range(rollouts):
        model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
    return model.episode_buffer.episodes("train") + model.episode_buffer.episodes("validation")


def states_by_replay(model, episode):
    """The encoder memory before every row of a stored episode, by reading it from its start."""
    encoder, policy = model.policy.encoder, model.policy
    state, rows = encoder.initial_state(1), []
    with torch.no_grad():
        for t in range(len(episode.observations)):
            rows.append(encoder_state_rows(encoder, state)[0])
            obs = torch.as_tensor(episode.observations[t : t + 1])
            _, state = encoder.step(state, policy.encoder_observations(obs), torch.as_tensor(episode.prev_actions[t : t + 1]))
    return np.stack(rows)


class SequenceChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_state_rows_round_trip(self):
        torch.manual_seed(0)
        gru = GRURolloutEncoder(4, 2, 1, 16, 2)
        xlstm = XLSTMRolloutEncoder(4, 2, 1, xlstm_config=OmegaConf.to_container(config(step0=False).cppo.xlstm, resolve=True))
        for encoder in (gru, xlstm):
            state = encoder.initial_state(3)
            with torch.no_grad():                        # a non-trivial memory: a few steps of random inputs
                for _ in range(5):
                    _, state = encoder.step(state, torch.randn(3, 4), torch.randn(3, 2))
            rows = encoder_state_rows(encoder, state)
            back = encoder_state_from_rows(encoder, rows)
            from contextual_ppo.buffers import _flatten_recurrent_state
            for (path, a), (_, b) in zip(_flatten_recurrent_state(state)[1], _flatten_recurrent_state(back)[1], strict=True):
                torch.testing.assert_close(torch.as_tensor(b), torch.as_tensor(a), rtol=0, atol=0, msg=str(path))
            self.assertEqual(rows.shape[0], 3)
        self.assertEqual(encoder_state_rows(gru, gru.initial_state(1)).shape, (1, 2 * 16))

    def test_windows_are_slices_of_stored_episodes(self):
        obs_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        buffer = SupervisedContextualEpisodeBuffer(1, obs_space, act_space, label_dim=1, validation_fraction=0.0, device="cpu",
                                                   seed=0, state_dim=3)
        rng = np.random.default_rng(0)
        stored = []
        for length in (3, 9, 25):                       # one episode shorter than a window
            obs, acts = rng.normal(size=(length, 4)).astype(np.float32), rng.uniform(-1, 1, (length, 2)).astype(np.float32)
            labels, states = rng.uniform(size=(length, 1)).astype(np.float32), rng.normal(size=(length + 1, 3)).astype(np.float32)
            for t in range(length):
                done = np.array([t == length - 1])
                buffer.add(observations=obs[t:t + 1], executed_actions=acts[t:t + 1], labels=labels[t:t + 1], dones=done,
                           final_observations=obs[t:t + 1], final_labels=labels[t:t + 1], encoder_states=states[t:t + 1],
                           final_encoder_states=states[t + 1:t + 2])
        stored = buffer.episodes("train")
        for stored_states, burn_in in ((False, 0), (True, 3)):
            samples = buffer.sample_chunks("train", 200, chunk_len=4, burn_in=burn_in, stored_states=stored_states)
            self.assertEqual(tuple(samples.observations.shape), (200, burn_in + 4, 4))
            self.assertEqual(samples.burn_in, burn_in)
            for row in range(200):
                n = int(samples.mask[row].sum())
                window = samples.observations[row, :n].numpy()
                matches = [(e, s) for e in stored for s in range(len(e.observations) - n + 1)
                           if np.array_equal(e.observations[s:s + n], window)]
                self.assertEqual(len(matches), 1)
                episode, start = matches[0]
                np.testing.assert_array_equal(samples.prev_actions[row, :n].numpy(), episode.prev_actions[start:start + n])
                np.testing.assert_array_equal(samples.labels[row, :n].numpy(), episode.labels[start:start + n])
                self.assertTrue(n == burn_in + 4 or start == 0)   # shorter only for an episode shorter than a window
                if stored_states:
                    np.testing.assert_array_equal(samples.initial_states[row].numpy(), episode.states[start])
                    self.assertGreater(len(episode.observations), burn_in)
                else:
                    self.assertIsNone(samples.initial_states)
        with self.assertRaises(ValueError):
            SupervisedContextualEpisodeBuffer(1, obs_space, act_space, device="cpu").sample_chunks("train", 4, 4, stored_states=True)

    def test_stored_memories_are_the_collectors(self):
        for easy in (False, True):                      # the hard version, and the easy one with its prelude
            with self.subTest(easy=easy):
                model = model_for("J1_gru_budget_r2d2", easy=easy)
                episodes = collect(model)
                self.assertGreater(len(episodes), 2)
                for episode in episodes:
                    self.assertEqual(episode.states.shape, (len(episode.observations), model.episode_buffer.state_dim))
                    np.testing.assert_allclose(episode.states, states_by_replay(model, episode), rtol=0, atol=1e-5)
                # Other modes store no memories
                self.assertIsNone(model_for("J1_gru_budget_chunk", easy=easy).episode_buffer.state_dim)

    def test_stored_state_window_reproduces_the_whole_episode(self):
        model = model_for("J1_gru_budget_r2d2", easy=True)
        collect(model)
        encoder, policy = model.policy.encoder, model.policy
        samples = model.episode_buffer.sample_chunks("train", 64, chunk_len=4, burn_in=3, stored_states=True)
        with torch.no_grad():
            state = encoder_state_from_rows(encoder, samples.initial_states)
            obs = policy.encoder_observations(samples.observations, n_batch_dims=2)
            _, state = encoder.unroll(state, obs[:, :3], samples.prev_actions[:, :3], mask=samples.mask[:, :3])
            windowed, _ = encoder.unroll(state, obs[:, 3:], samples.prev_actions[:, 3:], mask=samples.mask[:, 3:])
            # The same steps read from each episode's start
            for row in range(64):
                episode = next(e for e in model.episode_buffer.episodes("train")
                               if any(np.array_equal(e.observations[s:s + 7], samples.observations[row].numpy())
                                      for s in range(len(e.observations) - 6)))
                start = next(s for s in range(len(episode.observations) - 6)
                             if np.array_equal(episode.observations[s:s + 7], samples.observations[row].numpy()))
                whole, _ = encoder.unroll(encoder.initial_state(1), policy.encoder_observations(torch.as_tensor(episode.observations[None]), 2),
                                          torch.as_tensor(episode.prev_actions[None]))
                torch.testing.assert_close(windowed[row], whole[0, start + 3:start + 7], rtol=0, atol=1e-5)
            # The training step itself starts the windows from their stored memories: the same loss
            loss, _ = model._chunk_fit(samples)
            expected, _ = model._context_fit(windowed, samples.labels[:, 3:, :1], samples.mask[:, 3:])
        torch.testing.assert_close(loss, expected, rtol=0, atol=1e-6)
        # A plain chunk starts blank: the same window from a blank memory differs from the stored-state result
        blank = model_for("J1_gru_budget_chunk", easy=True)
        blank.policy.load_state_dict(model.policy.state_dict())
        with torch.no_grad():
            fresh, _ = blank.policy.encoder.unroll(blank.policy.encoder.initial_state(64), obs[:, 3:], samples.prev_actions[:, 3:],
                                                   mask=samples.mask[:, 3:])
        self.assertGreater(float((fresh - windowed).abs().max()), 1e-3)

    def test_every_mode_trains_the_encoder(self):
        for rung, sequence in (("J1_gru_budget", "episode"), ("J1_gru_budget_chunk", "chunk"), ("J1_gru_budget_r2d2", "stored_state")):
            with self.subTest(sequence=sequence):
                # Half the episodes held out, so the validation fit always has some
                model = model_for(rung, "cppo.supervised.validation_fraction=0.5", easy=True)
                self.assertEqual(model.supervised_sequence, sequence)
                collect(model, rollouts=3)
                self.assertGreater(model.episode_buffer.n_episodes("validation"), 0)
                calls = {"sample": 0, "chunks": []}
                buffer = model.episode_buffer
                sample, sample_chunks = buffer.sample, buffer.sample_chunks
                buffer.sample = lambda *a, **k: calls.__setitem__("sample", calls["sample"] + 1) or sample(*a, **k)
                buffer.sample_chunks = lambda *a, **k: calls["chunks"].append(k) or sample_chunks(*a, **k)
                before = {k: v.clone() for k, v in model.policy.encoder.state_dict().items()}
                stats = model._train_encoder_supervised()
                self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.policy.encoder.state_dict().items()))
                self.assertIn("context_val_mse", stats)                      # validation: always whole episodes
                if sequence == "episode":
                    self.assertEqual(calls["chunks"], [])
                else:
                    self.assertEqual(len(calls["chunks"]), model.supervised_gradient_steps)
                    self.assertEqual(calls["chunks"][0], {"burn_in": 3 if sequence == "stored_state" else 0,
                                                          "stored_states": sequence == "stored_state"})

    def test_settings(self):
        j1 = flat(config("+joint=J1_gru_budget", step0=False))
        for rung, sequence in (("J1_gru_budget_chunk", "chunk"), ("J1_gru_budget_r2d2", "stored_state")):
            variant = flat(config(f"+joint={rung}", step0=False))
            changed = {k for k in set(j1) | set(variant) if j1.get(k) != variant.get(k)}
            self.assertEqual({k for k in changed if not k.startswith("wandb.")} - {"experiment_id"}, {"cppo.supervised.sequence"})
            self.assertEqual(variant["cppo.supervised.sequence"], sequence)
        names = {rung: config(f"+joint={rung}", step0=False).wandb.run_name for rung in ("J1_gru_budget", "J1_gru_budget_chunk", "J1_gru_budget_r2d2")}
        self.assertEqual(len(set(names.values())), 3)
        self.assertIn("chunk16", names["J1_gru_budget_chunk"])
        self.assertIn("r2d2_burn16_chunk16", names["J1_gru_budget_r2d2"])
        # The default is whole episodes, so every earlier config keeps its behaviour
        self.assertEqual(config(step0=False).cppo.supervised.sequence, "episode")
        for overrides in (["cppo.supervised.sequence=windows"], ["cppo.supervised.chunk_len=0"], ["cppo.supervised.burn_in=-1"],
                          ["cppo.supervised.chunk_batch_size=0"], ["cppo.supervised.sequence=chunk", "cppo.velocity_head.enabled=true"],
                          ["cppo.supervised.sequence=chunk", "cppo.encoder_objective=rl"]):
            with self.assertRaises(ValueError, msg=overrides):
                training.validate_config(config("+joint=J1_gru_budget", *overrides, step0=False))
        # Launcher and report know the two conditions, one report each
        import report_step0
        import run_step0
        for condition, rung in (("J1_chunk", "J1_gru_budget_chunk"), ("J1_r2d2", "J1_gru_budget_r2d2")):
            command = run_step0.command_for(condition, 1, 1, Path("/tmp/sequence_launch"), warmup_steps=100)
            self.assertIn(f"+joint={rung}", command)
            self.assertIn("+env.kwargs.warmup_steps=100", command)
            cfg = config(*[item for item in command[2:] if not item.startswith("hydra.")], step0=False)
            report_step0.evaluation_protocol(cfg, condition)
        runs = [{"condition": c, "seed": s, "reward_scale": None, "battery_noise": None, "encoder_tag": None, "run_dir": f"/r/{c}{s}"}
                for c in ("J1_chunk", "J1_r2d2") for s in (0, 1)]
        lines = run_step0.report_commands(runs, [None], [None], Path("/out"))
        self.assertEqual(sorted(line.split(":")[0] for line in lines), ["Report (joint training J1_chunk)", "Report (joint training J1_r2d2)"])


if __name__ == "__main__":
    unittest.main()
