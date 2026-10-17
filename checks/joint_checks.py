"""CPU checks of joint training with a GRU encoder (ENCODER_PLAN.md, config/joint/): python -m unittest checks.joint_checks -v

Contextual_PPO's encoder can be a GRU (cppo.encoder_arch=gru, contextual_ppo.torch_layers.GRURolloutEncoder) instead
of the xLSTM. These checks cover the encoder, the learner with it, the rungs J0-J3 and the launcher and report.
"""
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import torch
from torch import nn

from checks.step0_checks import SMALL, config
from contextual_ppo import ContextualPPO
from contextual_ppo.torch_layers import GRURolloutEncoder, XLSTMRolloutEncoder
import main_battery_plane_contextual_ppo as training

RUNGS = {"J0": "J0_gru_a_budget", "J1": "J1_gru_budget", "J2": "J2_gru_budget_entropy", "J3": "J3_gru_budget_entropy_gaussian",
         "J4": "J4_gru_budget_entropy_gaussian_curiosity", "J5": "J5_gru_budget_warmup"}


def flat(cfg) -> dict:
    """Every leaf of a config as {"a.b.c": value}."""
    out = {}

    def walk(node, prefix):
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")
            else:
                out[f"{prefix}{key}"] = value
    walk(OmegaConf.to_container(cfg, resolve=True), "")
    return out


class JointChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_gru_encoder_paths_agree(self):
        torch.manual_seed(0)
        for head in ("point", "gaussian"):
            encoder = GRURolloutEncoder(4, 2, 1, 32, 2, [16], nn.Tanh, "sigmoid", head, auxiliary_dim=1, auxiliary_net_arch=[8]).eval()
            batch, length = 3, 25
            obs, prev = torch.randn(batch, length, 4), torch.randn(batch, length, 2)
            start = encoder.initial_state(batch)
            with torch.no_grad():
                whole, whole_state = encoder.unroll(start, obs, prev)
                state, stepped = start, []
                for i in range(length):
                    context, state = encoder.step(state, obs[:, i], prev[:, i])
                    stepped.append(context)
                stepped = torch.stack(stepped, 1)
                torch.testing.assert_close(whole, stepped, rtol=0, atol=1e-5)
                torch.testing.assert_close(whole_state["gru_state"], state["gru_state"], rtol=0, atol=1e-5)
                # Complete episodes padded at the end: the real steps as alone, zeros elsewhere, each row's own final state
                lengths = torch.tensor([25, 10, 17])
                mask = torch.arange(length)[None] < lengths[:, None]
                masked, masked_state = encoder.unroll(start, obs, prev, mask=mask)
                for row, n in enumerate(lengths.tolist()):
                    alone, alone_state = encoder.unroll(encoder.initial_state(1), obs[row:row + 1, :n], prev[row:row + 1, :n])
                    torch.testing.assert_close(masked[row, :n], alone[0], rtol=0, atol=1e-5)
                    torch.testing.assert_close(masked_state["gru_state"][:, row], alone_state["gru_state"][:, 0], rtol=0, atol=1e-5)
                self.assertEqual(float(masked[~mask].abs().max()), 0.0)
                # A mask that isn't a prefix steps, with the same meaning
                holes = mask.clone()
                holes[0, 3] = False
                stepped_mask, _ = encoder.unroll(start, obs, prev, mask=holes)
                self.assertEqual(float(stepped_mask[0, 3].abs().max()), 0.0)
                # An episode start inside a stream: a fresh start from there, the same before
                starts = torch.zeros(batch, length, dtype=torch.bool)
                starts[0, 12] = True
                streamed, _ = encoder.unroll(start, obs, prev, episode_starts=starts)
                fresh, _ = encoder.unroll(encoder.initial_state(1), obs[0:1, 12:], prev[0:1, 12:])
                torch.testing.assert_close(streamed[0, 12:], fresh[0], rtol=0, atol=1e-5)
                torch.testing.assert_close(streamed[0, :12], stepped[0, :12], rtol=0, atol=1e-5)
                # The auxiliary head runs on the same embeddings
                contexts, auxiliary, _ = encoder.unroll_with_auxiliary(start, obs, prev, mask=mask)
                torch.testing.assert_close(contexts, masked, rtol=0, atol=0)
                self.assertEqual(float(auxiliary[~mask].abs().max()), 0.0)
                self.assertEqual(contexts.shape[-1], 2 if head == "gaussian" else 1)
                self.assertEqual(encoder.split_context(contexts)[1] is None, head == "point")
            # Resetting some rows zeroes their memory only
            state = {"gru_state": torch.randn(2, 3, 32)}
            reset = encoder.reset_state(state, torch.tensor([True, False, True]))
            self.assertEqual(float(reset["gru_state"][:, [0, 2]].abs().max()), 0.0)
            torch.testing.assert_close(reset["gru_state"][:, 1], state["gru_state"][:, 1])

    def test_ppo_reencoding_reproduces_collection_with_the_gru(self):
        for rung in ("J1", "J3"):
            with self.subTest(rung=rung):
                # SMALL's 4-step episodes end inside the 8-step streams
                cfg = config(f"+joint={RUNGS[rung]}", *SMALL, step0=False)
                model = training.build_model(cfg, training.make_train_env(cfg))
                self.assertIsInstance(model.policy.encoder, GRURolloutEncoder)
                _, callback = model._setup_learn(model.n_steps * model.n_envs * 3)
                for _ in range(3):        # several rollouts: the memory carries over, episodes end inside streams
                    model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
                    (streams,) = list(model.rollout_buffer.get(1, shuffle=False))
                    with torch.no_grad():
                        values, log_prob, _, _, latents = model.policy.evaluate_actions(
                            streams.observations, streams.actions, streams.initial_state, streams.prev_actions,
                            streams.episode_starts, detach_context=True)
                    self.assertTrue(bool(streams.episode_starts[:, 1:].any()), "no episode ended inside a stream")
                    torch.testing.assert_close(values, streams.old_values, rtol=0, atol=1e-5)
                    torch.testing.assert_close(log_prob, streams.old_log_prob, rtol=0, atol=1e-5)
                    if streams.context_estimates is not None:
                        torch.testing.assert_close(latents, streams.context_estimates, rtol=0, atol=1e-5)

    def test_gru_model_trains_saves_and_loads(self):
        cfg = config("+joint=J3_gru_budget_entropy_gaussian", *SMALL, step0=False)
        model = training.build_model(cfg, training.make_train_env(cfg))
        before = {k: v.clone() for k, v in model.policy.encoder.state_dict().items()}
        model.learn(total_timesteps=64)
        # The supervised steps train the encoder
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.policy.encoder.state_dict().items()))
        with tempfile.TemporaryDirectory() as folder:
            model.save(Path(folder) / "model")
            loaded = ContextualPPO.load(Path(folder) / "model.zip", device="cpu")
        self.assertIsInstance(loaded.policy.encoder, GRURolloutEncoder)
        self.assertEqual((loaded.policy.encoder_arch, loaded.policy.gru_hidden_dim, loaded.policy.gru_num_layers), ("gru", 128, 2))
        for (key, a), b in zip(model.policy.state_dict().items(), loaded.policy.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=key)
        # Experiment A and the saved configs from before encoder_arch keep the xLSTM
        plain = config(*SMALL, step0=False)
        self.assertIsInstance(training.build_model(plain, training.make_train_env(plain)).policy.encoder, XLSTMRolloutEncoder)
        old = OmegaConf.create(OmegaConf.to_container(plain))
        del old.cppo["encoder_arch"], old.cppo["gru"]
        self.assertEqual(training.encoder_arch_from(old), ("xlstm", 128, 2))

    def test_rungs_change_one_thing_each(self):
        rerun, rungs = flat(config("+rerun=A_point_baseline", step0=False)), {r: flat(config(f"+joint={n}", step0=False)) for r, n in RUNGS.items()}
        naming = {"experiment_id", "wandb.run_name", "wandb.group", "wandb.tags"}

        def changed(a, b):
            return {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - naming

        self.assertEqual(changed(rerun, rungs["J0"]), {"cppo.encoder_arch"})
        self.assertEqual(changed(rungs["J0"], rungs["J1"]),
                         {"cppo.supervised.gradient_steps", "cppo.supervised.episode_batch_size", "cppo.supervised.capacity"})
        self.assertEqual(changed(rungs["J1"], rungs["J2"]), {"cppo.ent_coef"})
        self.assertEqual(changed(rungs["J2"], rungs["J3"]), {"cppo.context_head", "cppo.supervised.nll_beta"})
        self.assertEqual(changed(rungs["J3"], rungs["J4"]), {"cppo.curiosity.weight", "cppo.curiosity.until"})
        self.assertEqual(changed(rungs["J1"], rungs["J5"]), {"cppo.policy_warmup_steps", "cppo.supervised.warmup_gradient_steps"})
        names = {r: config(f"+joint={n}", step0=False).wandb for r, n in RUNGS.items()}
        self.assertEqual(len({w.run_name for w in names.values()}), len(RUNGS))
        self.assertEqual(len({w.group for w in names.values()}), len(RUNGS))
        # Curiosity needs the Gaussian head; the warm-up must end inside the run
        with self.assertRaisesRegex(ValueError, "curiosity"):
            training.validate_config(config(f"+joint={RUNGS['J2']}", "cppo.curiosity.weight=1.0", step0=False))
        with self.assertRaisesRegex(ValueError, "policy_warmup_steps"):
            training.validate_config(config(f"+joint={RUNGS['J5']}", "cppo.policy_warmup_steps=2000000", step0=False))
        # Every finished episode kept from J1 on: more than the run's steps
        self.assertGreater(rungs["J1"]["cppo.supervised.capacity"], rungs["J1"]["training.total_timesteps"])
        for name in RUNGS.values():
            training.validate_config(config(f"+joint={name}", step0=False))
        # A GRU needs a learned encoder
        with self.assertRaisesRegex(ValueError, "encoder_arch=gru"):
            training.validate_config(config("cppo.encoder_arch=gru"))      # step 0's oracle config
        with self.assertRaisesRegex(ValueError, "encoder_arch must be"):
            training.validate_config(config("cppo.encoder_arch=lstm", step0=False))

    def test_curiosity_bonus_is_the_entropy_drop_in_the_training_reward_only(self):
        rollouts = {}
        for weight in (0.0, 2.0):
            cfg = config(f"+joint={RUNGS['J4']}", *SMALL, f"cppo.curiosity.weight={weight}", "cppo.curiosity.until=null", step0=False)
            model = training.build_model(cfg, training.make_train_env(cfg))
            _, callback = model._setup_learn(model.n_steps * model.n_envs)
            model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)
            rollouts[weight] = (model, model.rollout_buffer)
        (model, plain), (_, curious) = rollouts[0.0], rollouts[2.0]
        # The same seed, so the same flights: the bonus never changes what is collected, nor the stored env rewards
        self.assertTrue((plain.observations == curious.observations).all() and (plain.actions == curious.actions).all())
        self.assertTrue((plain.rewards == curious.rewards).all())
        self.assertEqual(float(abs(plain.intrinsic_rewards).max()), 0.0)
        # The bonus: 2 x the drop in the Gaussian head's entropy from o_t to o_{t+1}, the latents re-encoded from the streams
        (streams,) = list(plain.get(1, shuffle=False))
        with torch.no_grad():
            *_, latents = model.policy.evaluate_actions(streams.observations, streams.actions, streams.initial_state,
                                                        streams.prev_actions, streams.episode_starts, detach_context=True)
        variance = latents[..., 1].T                                                # [T, E]: sigma2 of o_t
        drop = 0.5 * (torch.log(variance[:-1]) - torch.log(variance[1:])).numpy()  # o_t -> o_{t+1}
        within = ~(plain.terminated[:-1] | plain.truncated[:-1])                  # o_{t+1} is the next step's input
        self.assertTrue(within.any())
        self.assertLess(abs(curious.intrinsic_rewards[:-1][within] - 2.0 * drop[within]).max(), 1e-4)
        self.assertGreater(float(abs(curious.intrinsic_rewards).max()), 0.0)
        # ... and it reaches the advantages
        self.assertFalse((plain.advantages == curious.advantages).all())
        # The weight falls linearly to 0 at curiosity_until
        model.curiosity_weight, model.curiosity_until = 1.0, 1000
        for steps, expected in ((0, 1.0), (250, 0.75), (1000, 0.0), (5000, 0.0)):
            model.num_timesteps = steps
            self.assertAlmostEqual(model._curiosity_weight(), expected)

    def test_warmup_trains_only_the_encoder_then_ppo_starts(self):
        cfg = config(f"+joint={RUNGS['J5']}", *SMALL, "cppo.policy_warmup_steps=32", "cppo.supervised.warmup_gradient_steps=3",
                     step0=False)
        model = training.build_model(cfg, training.make_train_env(cfg))
        groups = model.policy.parameter_groups()
        snapshot = lambda name: [p.detach().clone() for p in groups[name]]
        actor, value, encoder = snapshot("actor"), snapshot("value"), snapshot("encoder")
        model.learn(total_timesteps=32)                       # two rollouts of 16 steps: both inside the warm-up
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actor, groups["actor"])))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(value, groups["value"])))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(encoder, groups["encoder"])))
        model.learn(total_timesteps=32, reset_num_timesteps=False)    # past the warm-up: PPO trains the actor
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(actor, groups["actor"])))

    def test_launcher_and_report(self):
        import report_step0
        import run_step0

        for rung, name in RUNGS.items():
            command = run_step0.command_for(rung, 3, 1, Path("/tmp/joint_launch"))
            self.assertIn(f"+joint={name}", command)
            cfg = config(*[item for item in command[2:] if not item.startswith("hydra.")], step0=False)
            self.assertEqual((cfg.training.seed, cfg.cppo.encoder_arch), (3, "gru"))
            report_step0.evaluation_protocol(cfg, rung)          # accepted, as a hidden-mode run with the GRU
            self.assertEqual(cfg.env.observation_mode, "hidden")
        # A is the xLSTM, J the GRU: each refuses the other's config
        with self.assertRaisesRegex(ValueError, "GRU encoder"):
            report_step0.evaluation_protocol(config("+rerun=A_point_baseline", step0=False), "J1")
        with self.assertRaisesRegex(ValueError, "A requires"):
            report_step0.evaluation_protocol(config("+joint=J1_gru_budget", step0=False), "A")
        # One report per rung
        runs = [{"condition": c, "seed": s, "reward_scale": None, "battery_noise": None, "encoder_tag": None, "run_dir": f"/r/{c}{s}"}
                for c in RUNGS for s in (0, 1)]
        lines = run_step0.report_commands(runs, [None], [None], Path("/out"))
        self.assertEqual([line.split(":")[0] for line in lines], [f"Report (joint training {r})" for r in sorted(RUNGS)])
        self.assertTrue(all(line.count("--run ") == 2 for line in lines))


if __name__ == "__main__":
    unittest.main()
