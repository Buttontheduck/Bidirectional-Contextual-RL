"""CPU checks of the memory baselines (PAPER_ASSESSMENT.md, Tier 1 item 1; memory_baselines.py):
python -m unittest checks.memory_baseline_checks -v"""
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from omegaconf import OmegaConf
import torch

import battery_plane as bp
from checks.step0_checks import plain_ppo_config, plain_ppo_script
from flight_logging import Flight, evaluate_flights
from memory_baselines import (
    AUTOPILOT_KEY, FusedLstmPolicy, HistoryWindow, MaskedRecurrentPPO, RecurrentInput, memory_kind_from, memory_wrapper_from,
)
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

MEMORY_CONFIGS = {"M1a_lstm_bs256": ("recurrent", 256, None), "M1b_lstm_bs1024": ("recurrent", 1024, None),
                  "M2a_win16_bs256": ("window", 256, 16), "M2b_win16_bs1024": ("window", 1024, 16),
                  "M2a_win64_bs256": ("window", 256, 64), "M2b_win64_bs1024": ("window", 1024, 64)}
SMALL = ["training.n_envs=2", "ppo.n_steps=16", "ppo.batch_size=16", "ppo.n_epochs=1", "ppo.net_arch=[8]",
         "memory.lstm_hidden_size=8", "+env.kwargs.T=12", "ppo.verbose=0"]
# Keys the inputs must never depend on (the privileged infos)
PRIVILEGED = ("b", "b0", "cap", "saturated", "warmup_batteries", "b_start")


class FakeTakeoffEnv(gym.Env):
    """A stand-in for the easy version on branches without it: a take-off of `prelude` steps in the reset info,
    then `mission` steps whose observation counts the steps and echoes the last action."""

    def __init__(self, prelude: int = 3, mission: int = 5):
        self.core = SimpleNamespace(prelude_steps=prelude)
        self.prelude, self.mission = prelude, mission
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.inner_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.start = 0, float(self.np_random.integers(100))
        take_off = self.start + np.arange(self.prelude * 4, dtype=np.float32).reshape(self.prelude, 4) / 10
        info = {"warmup_observations": take_off, "warmup_actions": -take_off[:, :2] / 100,
                "warmup_batteries": np.full(self.prelude, 0.5), "b0": 0.5, "b": 0.5}
        return np.array([self.start, -1.0, 0.0, 1.0], dtype=np.float32), info

    def step(self, action):
        self.inner_steps += 1
        self.t += 1
        observation = np.array([self.start, self.t, *np.asarray(action)[:2]], dtype=np.float32)
        info = {"vx": 30.0, "y": 100.0, "saturated": False, "b0": 0.5, "b": 0.4, "mission_complete": self.t == self.mission}
        return observation, 1.0, False, self.t == self.mission, info


def random_episode(env, rng, steps, low=-1.4, high=1.4):
    """Observations of the wrapped env and of its inner env, executed actions and infos over `steps` random steps."""
    outer = []
    observation, info = env.reset(seed=int(rng.integers(1000)))
    outer.append((observation, info))
    actions = []
    for _ in range(steps):
        action = rng.uniform(low, high, size=2).astype(np.float32)
        observation, _, terminated, truncated, info = env.step(action)
        actions.append(action)
        outer.append((observation, info))
        if terminated or truncated:
            break
    return outer, actions


class MemoryBaselineChecks(unittest.TestCase):
    def test_configs_compose_and_validate(self):
        script = plain_ppo_script()
        for name, (kind, batch, window) in MEMORY_CONFIGS.items():
            for seed in (0, 7):
                cfg = plain_ppo_config(f"+experiment={name}", f"training.seed={seed}")
                script.validate_ppo(cfg)
                self.assertEqual(memory_kind_from(cfg), kind)
                self.assertEqual(cfg.env.observation_mode, "hidden")
                self.assertEqual((int(cfg.ppo.batch_size), int(cfg.ppo.n_steps), int(cfg.ppo.n_epochs)), (batch, 512, 10))
                self.assertEqual((cfg.ppo.gamma, cfg.ppo.reward_scale), (0.99, 0.1))
                self.assertEqual((int(cfg.training.total_timesteps), int(cfg.evaluation.frequency)), (2_000_000, 50_000))
                self.assertEqual(list(cfg.evaluation.reset_grid.eval_seeds), [1000, 1001, 1002])
                self.assertEqual(cfg.ppo.policy, "MlpLstmPolicy" if kind == "recurrent" else "MlpPolicy")
                self.assertEqual(cfg.context.source, "none")
                self.assertTrue(cfg.memory.fused_lstm)
                if window is not None:
                    self.assertEqual(int(cfg.memory.window), window)
                self.assertIn(f"_s{seed}", cfg.wandb.run_name)
        # Old configs (no memory section) are plain PPO
        old = plain_ppo_config("+experiment=S0a_blind_bs256")
        OmegaConf.set_struct(old, False)
        del old["memory"]
        self.assertEqual(memory_kind_from(old), "none")
        self.assertIsNone(memory_wrapper_from(old))

    def test_bad_memory_options_are_rejected(self):
        script = plain_ppo_script()
        for overrides in (["+experiment=M1a_lstm_bs256", "ppo.policy=MlpPolicy"],
                          ["+experiment=S0a_blind_bs256", "ppo.policy=MlpLstmPolicy"],
                          ["+experiment=M2a_win16_bs256", "memory.window=0"],
                          ["+experiment=M2a_win16_bs256", "memory.kind=lstm"],
                          ["+experiment=M2a_win16_bs256", "context.source=ideal"],
                          ["+experiment=M1a_lstm_bs256", "memory.lstm_hidden_size=0"],
                          ["+experiment=M1a_lstm_bs256", "memory.fused_lstm=1"]):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                script.validate_ppo(plain_ppo_config(*overrides))

    def test_window_holds_the_last_steps(self):
        rng = np.random.default_rng(0)
        for window in (1, 5, 16):
            env = HistoryWindow(bp.BatteryPlaneEnv(observation_mode="hidden", T=12), window)
            self.assertEqual(env.observation_space.shape, (window * 7,))
            for _ in range(3):   # several episodes: the window is cleared at every reset
                outer, actions = random_episode(env, rng, 40)
                # Reference: rebuild the frames from the inner observations (columns 4:7 of the newest frame are
                # [a_prev, valid]; the newest frame's first 4 numbers are the inner observation)
                frames = []
                for t, (observation, _) in enumerate(outer):
                    inner = observation.reshape(window, 7)[-1, :4]
                    previous = np.zeros(2, np.float32) if t == 0 else np.clip(actions[t - 1], -1, 1)
                    frames.append(np.concatenate([inner, previous, [1.0]]))
                    expected = np.zeros((window, 7), np.float32)
                    recent = frames[-window:]
                    expected[window - len(recent):] = recent
                    np.testing.assert_array_equal(observation.reshape(window, 7), expected)

    def test_inputs_never_read_the_privileged_infos(self):
        """The same observations and actions with other privileged infos give the same inputs."""

        class OtherInfos(FakeTakeoffEnv):
            def reset(self, *, seed=None, options=None):
                observation, info = super().reset(seed=seed, options=options)
                return observation, {**info, **{key: np.full_like(np.asarray(info.get(key, 0.0)), 7.0) for key in PRIVILEGED}}

            def step(self, action):
                observation, reward, terminated, truncated, info = super().step(action)
                return observation, reward, terminated, truncated, {**info, **{key: 7.0 for key in PRIVILEGED}}

        for wrap in (RecurrentInput, lambda env: HistoryWindow(env, 4)):
            outputs = []
            for inner in (FakeTakeoffEnv(), OtherInfos()):
                env, rng = wrap(inner), np.random.default_rng(2)
                outputs.append([observation for observation, _ in random_episode(env, rng, 12)[0]])
            np.testing.assert_array_equal(np.array(outputs[0]), np.array(outputs[1]))
        # Hidden mode: 4 numbers per observation, no battery column
        self.assertEqual(bp.BatteryPlaneEnv(observation_mode="hidden").observation_space.shape, (4,))

    def test_recurrent_input_without_takeoff(self):
        rng = np.random.default_rng(1)
        env = RecurrentInput(bp.BatteryPlaneEnv(observation_mode="hidden", T=12))
        self.assertIsNone(env.autopilot_index)
        self.assertEqual(env.observation_space.shape, (6,))
        outer, actions = random_episode(env, rng, 20)
        np.testing.assert_array_equal(outer[0][0][4:], [0.0, 0.0])
        for t in range(1, len(outer)):
            np.testing.assert_array_equal(outer[t][0][4:], np.clip(actions[t - 1], -1, 1))

    def test_takeoff_replay(self):
        inner = FakeTakeoffEnv(prelude=3, mission=5)
        env = RecurrentInput(inner)
        self.assertEqual(env.autopilot_index, 6)
        observation, info = env.reset(seed=0)
        take_off, autopilot = info["warmup_observations"], info["warmup_actions"]
        np.testing.assert_array_equal(observation, [*take_off[0], 0.0, 0.0, 1.0])
        rewards = []
        for k in range(1, 3):
            observation, reward, terminated, truncated, step_info = env.step(np.array([1.0, 1.0]))
            np.testing.assert_array_equal(observation, [*take_off[k], *autopilot[k - 1], 1.0])
            self.assertEqual((reward, terminated, truncated, step_info), (0.0, False, False, {AUTOPILOT_KEY: True}))
            rewards.append(reward)
        # The hand-over: the env's own reset observation, the autopilot's last action, autopilot input 0
        observation, reward, _, _, step_info = env.step(np.array([1.0, 1.0]))
        np.testing.assert_array_equal(observation, [inner.start, -1.0, 0.0, 1.0, *autopilot[-1], 0.0])
        self.assertEqual((reward, step_info), (0.0, {AUTOPILOT_KEY: True}))
        self.assertEqual(inner.inner_steps, 0, "the agent's actions must not reach the plane during the take-off")
        # The mission: actions reach the plane, clipped
        length = 3
        while True:
            observation, reward, terminated, truncated, step_info = env.step(np.array([2.0, -0.5]))
            length += 1
            np.testing.assert_array_equal(observation[4:], [1.0, -0.5, 0.0])
            self.assertEqual(reward, 1.0)
            self.assertNotIn(AUTOPILOT_KEY, step_info)
            if terminated or truncated:
                break
        self.assertEqual((inner.inner_steps, length), (5, 8))
        # A new episode replays its own take-off
        observation, info = env.reset(seed=1)
        np.testing.assert_array_equal(observation[:4], info["warmup_observations"][0])

    def test_window_starts_with_the_takeoff(self):
        inner = FakeTakeoffEnv(prelude=3, mission=5)
        env = HistoryWindow(inner, 6)
        observation, info = env.reset(seed=0)
        frames = observation.reshape(6, 7)
        take_off, autopilot = info["warmup_observations"], info["warmup_actions"]
        np.testing.assert_array_equal(frames[:2], 0.0)   # padding
        np.testing.assert_array_equal(frames[2], [*take_off[0], 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(frames[3], [*take_off[1], *autopilot[0], 1.0])
        np.testing.assert_array_equal(frames[4], [*take_off[2], *autopilot[1], 1.0])
        np.testing.assert_array_equal(frames[5], [inner.start, -1.0, 0.0, 1.0, *autopilot[2], 1.0])
        # A window of 2 keeps only the end of the take-off
        observation, _ = HistoryWindow(FakeTakeoffEnv(3, 5), 2).reset(seed=0)
        np.testing.assert_array_equal(observation.reshape(2, 7)[0, :4], take_off[2])

    def test_masked_steps_leave_the_losses(self):
        torch.manual_seed(0)
        envs = DummyVecEnv([lambda: RecurrentInput(FakeTakeoffEnv(prelude=3, mission=5)) for _ in range(2)])
        model = MaskedRecurrentPPO("MlpLstmPolicy", envs, n_steps=16, batch_size=8, n_epochs=1, seed=0, device="cpu",
                                   policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]}, autopilot_index=6)
        model.learn(32)
        buffer = model.rollout_buffer
        np.random.seed(0)
        masked = list(buffer.get(8))
        buffer.autopilot_index = None
        np.random.seed(0)
        plain = list(buffer.get(8))
        buffer.autopilot_index = 6
        replayed = 0
        for a, b in zip(masked, plain):
            flag = b.observations[:, 6]
            np.testing.assert_array_equal(a.mask.numpy(), (b.mask * (1 - flag)).numpy())
            replayed += int((flag * b.mask).sum())
            self.assertGreater(float(a.mask.sum()), 0)
        self.assertGreater(replayed, 0, "the rollout must contain replayed take-off steps")
        # sb3-contrib applies the mask to the policy, value and entropy losses and to the advantage normalization
        source = inspect.getsource(RecurrentPPO.train)
        for term in ("advantages[mask]", "policy_loss_2)[mask]", ") ** 2)[mask]", "entropy[mask]"):
            self.assertIn(term, source)
        # Saved and loaded with its mask
        with tempfile.TemporaryDirectory() as directory:
            model.save(Path(directory) / "model")
            loaded = MaskedRecurrentPPO.load(Path(directory) / "model", env=envs, device="cpu")
        self.assertEqual(loaded.autopilot_index, 6)
        self.assertEqual(loaded.rollout_buffer.autopilot_index, 6)

    def test_fused_lstm_is_sb3_contribs_lstm(self):
        """One nn.LSTM call per sequence = sb3-contrib's step-by-step loop, for its sequences (resets only at the start)."""
        torch.manual_seed(0)
        lstm = torch.nn.LSTM(6, 32)
        for n_seq, length, starts_at in ((3, 40, "first"), (5, 1, "first"), (4, 12, "middle")):
            features = torch.randn(n_seq * length, 6, requires_grad=True)
            states = (torch.randn(1, n_seq, 32), torch.randn(1, n_seq, 32))
            starts = torch.zeros(n_seq, length)
            starts[::2, 0] = 1.0
            features.data.view(n_seq, length, 6)[-1, length // 2:] = 0.0   # padding at the end of the last sequence
            if starts_at == "middle":
                starts[1, length // 2] = 1.0   # not a pattern of sb3-contrib's buffer: the loop runs
            outputs = [f(features, states, starts.flatten(), lstm)[0] for f in
                       (RecurrentActorCriticPolicy._process_sequence, FusedLstmPolicy._process_sequence)]
            np.testing.assert_array_equal(outputs[0].detach().numpy(), outputs[1].detach().numpy())
            grads = [torch.autograd.grad(out.square().sum(), [lstm.weight_hh_l0, features]) for out in outputs]
            for a, b in zip(*grads):
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-6)

    def test_fused_update_matches_sb3_contrib(self):
        """The same seed with either policy: the same first rollout and, after one update, the same weights up to rounding."""
        torch.set_num_threads(1)
        models = []
        for policy in ("MlpLstmPolicy", FusedLstmPolicy):
            envs = DummyVecEnv([lambda: RecurrentInput(bp.BatteryPlaneEnv(observation_mode="hidden", T=40)) for _ in range(2)])
            model = MaskedRecurrentPPO(policy, envs, n_steps=64, batch_size=32, n_epochs=2, seed=3, device="cpu",
                                       policy_kwargs={"lstm_hidden_size": 16, "net_arch": [16]})
            model.learn(128)
            models.append(model)
        stock, fused = models
        self.assertIsInstance(fused.policy, FusedLstmPolicy)
        self.assertIs(type(stock.policy), RecurrentActorCriticPolicy)
        np.testing.assert_array_equal(stock.rollout_buffer.observations, fused.rollout_buffer.observations)
        np.testing.assert_array_equal(stock.rollout_buffer.actions, fused.rollout_buffer.actions)
        for (name, a), (_, b) in zip(stock.policy.state_dict().items(), fused.policy.state_dict().items()):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=0, atol=1e-5, err_msg=name)

    def test_flight_logging_skips_the_replayed_takeoff(self):
        flight = Flight(SimpleNamespace(v_min=40.0, v_goal=60.0, dt=0.05))
        flight.add({AUTOPILOT_KEY: True}, 0.0, [1.0, 0.0])
        self.assertEqual(flight.steps, 0)
        flight.add({"vx": 50.0, "y": 100.0, "saturated": True, "b0": 0.5, "b": 0.49}, 1.0, [1.0, 0.0])
        self.assertEqual(flight.steps, 1)

    def test_recurrent_run_reproduces_its_validation(self):
        script = plain_ppo_script()
        cfg = plain_ppo_config("+experiment=M1a_lstm_bs256", *SMALL, "evaluation.reset_grid.eval_seeds=[1000]",
                               "evaluation.reset_grid.eval_initial_battery=[0.3,0.6]")
        script.validate_ppo(cfg)
        torch.set_num_threads(1)
        schedule = list(bp.iter_eval_resets(cfg.evaluation.reset_grid))
        train_env = script.make_train_env(cfg)
        model = MaskedRecurrentPPO("MlpLstmPolicy", train_env, n_steps=16, batch_size=16, n_epochs=1, seed=0, device="cpu",
                                   policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]}, autopilot_index=None)
        model.learn(64)
        self.assertEqual(train_env.observation_space.shape, (6,))

        def validate(m):
            env, scheduler, raw = script.make_eval_env(cfg, schedule)
            scheduler.rewind()
            rewards, lengths, recorder = evaluate_flights(m, env, len(schedule), True, raw.core)
            env.close()
            return rewards, lengths, recorder.flights

        first = validate(model)
        self.assertEqual(first, validate(model))
        with tempfile.TemporaryDirectory() as directory:
            model.save(Path(directory) / "final_model")
            self.assertEqual(first, validate(MaskedRecurrentPPO.load(Path(directory) / "final_model", device="cpu")))
        self.assertEqual(len(first[2]), 2)

    def test_report_accepts_the_memory_baselines(self):
        import report_step0
        for condition, name in (("M1a_lstm", "M1a_lstm_bs256"), ("M1b_lstm", "M1b_lstm_bs1024"), ("M2a_win16", "M2a_win16_bs256"),
                                ("M2b_win16", "M2b_win16_bs1024"), ("M2a_win64", "M2a_win64_bs256"), ("M2b_win64", "M2b_win64_bs1024")):
            protocol = report_step0.evaluation_protocol(plain_ppo_config(f"+experiment={name}"), condition)
            self.assertEqual((protocol["schedule"], protocol["frequency"], protocol["prelude_steps"]), ("step0", 50_000, 0))
            self.assertEqual(protocol["memory"]["kind"], MEMORY_CONFIGS[name][0])
            if MEMORY_CONFIGS[name][0] == "recurrent":
                self.assertTrue(protocol["memory"]["fused_lstm"])
        for condition, name in (("M1a_lstm", "M2a_win16_bs256"), ("M2a_win64", "M2a_win16_bs256"), ("0a_blind", "M2a_win16_bs256"),
                                ("M2a_win16", "S0a_blind_bs256"), ("M1b_lstm", "M1a_lstm_bs256")):
            with self.subTest(condition=condition, config=name), self.assertRaises(ValueError):
                report_step0.evaluation_protocol(plain_ppo_config(f"+experiment={name}"), condition)
        # The replay schedule of RecurrentPPO on the easy version (take-off of 100 steps), as run_step0.py launches it
        self.assertEqual(report_step0.schedule_from("replay", 100),
                         {"total": 2_285_760, "late": (1_714_320, 2_285_760), "final": 2_289_664, "frequency": 57_144})
        self.assertEqual(report_step0.schedule_from("step0")["frequency"], 50_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.npz"
            steps = np.arange(1, 41) * 57_144
            np.savez(path, timesteps=steps, results=np.tile(np.arange(40.0)[:, None], (1, 18)), ep_lengths=np.full((40, 18), 800))
            curve = report_step0.summarize_curve(path, schedule="replay", prelude_steps=100)
            self.assertEqual(curve["late_evaluation_timesteps"], steps[29:].tolist())
            self.assertAlmostEqual(curve["late_mean_reward"], float(np.mean(np.arange(29.0, 40.0))))
            with self.assertRaises(ValueError):
                report_step0.summarize_curve(path, schedule="step0")

    def test_window_run_trains(self):
        script = plain_ppo_script()
        cfg = plain_ppo_config("+experiment=M2b_win16_bs1024", *SMALL)
        script.validate_ppo(cfg)
        env = script.make_train_env(cfg)
        self.assertEqual(env.observation_space.shape, (16 * 7,))
        env.close()


def supports_takeoff() -> bool:
    """The easy version's take-off (branch easy-env); main's battery_plane.py has no warmup_steps."""
    try:
        bp.BatteryPlaneEnv(observation_mode="hidden", warmup_steps=100)
    except TypeError:
        return False
    return True


@unittest.skipUnless(supports_takeoff(), "this battery_plane.py has no take-off (warmup_steps)")
class TakeoffChecks(unittest.TestCase):
    """The memory baselines on the easy version's take-off-only variant (warmup_steps=100, EASY_ENV_WARMUP.md)."""

    def test_replay_then_the_same_mission(self):
        rng = np.random.default_rng(4)
        replayed = RecurrentInput(bp.BatteryPlaneEnv(observation_mode="hidden", warmup_steps=100))
        plain = bp.BatteryPlaneEnv(observation_mode="hidden", warmup_steps=100)
        self.assertEqual((replayed.prelude_steps, replayed.autopilot_index), (100, 6))
        for seed, b0 in ((11, 0.25), (12, 0.65)):
            options = {"b0": b0}
            observation, info = replayed.reset(seed=seed, options=options)
            reference, reference_info = plain.reset(seed=seed, options=options)
            take_off, autopilot = info["warmup_observations"], info["warmup_actions"]
            np.testing.assert_array_equal(take_off, reference_info["warmup_observations"])
            self.assertEqual(take_off.shape, (100, 4))
            np.testing.assert_array_equal(observation, [*take_off[0], 0.0, 0.0, 1.0])
            for k in range(1, 101):
                observation, reward, terminated, truncated, step_info = replayed.step(rng.uniform(-1, 1, 2))
                self.assertEqual((reward, terminated, truncated, step_info), (0.0, False, False, {AUTOPILOT_KEY: True}))
                expected = [*take_off[k], *autopilot[k - 1], 1.0] if k < 100 else [*reference, *autopilot[-1], 0.0]
                np.testing.assert_allclose(observation, expected, rtol=0, atol=1e-6)
            # The mission: identical to the plane's own episode for the same actions
            steps = 0
            while True:
                action = rng.uniform(-1.3, 1.3, 2).astype(np.float32)
                observation, reward, terminated, truncated, step_info = replayed.step(action)
                reference, reference_reward, reference_terminated, reference_truncated, _ = plain.step(np.clip(action, -1, 1))
                steps += 1
                np.testing.assert_array_equal(observation[:4], reference)
                np.testing.assert_array_equal(observation[4:], [*np.clip(action, -1, 1), 0.0])
                self.assertEqual((reward, terminated, truncated), (reference_reward, reference_terminated, reference_truncated))
                if terminated or truncated:
                    break
            self.assertGreater(steps, 10)

    def test_report_schedules_on_the_takeoff(self):
        import report_step0
        from run_step0 import replay_schedule
        protocol = report_step0.evaluation_protocol(
            plain_ppo_config("+experiment=M1a_lstm_bs256", "+env.kwargs.warmup_steps=100", *replay_schedule(100)), "M1a_lstm")
        self.assertEqual((protocol["schedule"], protocol["frequency"], protocol["prelude_steps"]), ("replay", 57_144, 100))
        expected = report_step0.schedule_from("replay", 100)
        self.assertEqual(replay_schedule(100), [f"training.total_timesteps={expected['total']}",
                                                f"evaluation.frequency={expected['frequency']}",
                                                f"evaluation.plots.frequency={2 * expected['frequency']}",
                                                f"checkpoint.frequency={10 * expected['frequency']}"])
        with self.assertRaises(ValueError):   # without the scaled budget
            report_step0.evaluation_protocol(plain_ppo_config("+experiment=M1a_lstm_bs256", "+env.kwargs.warmup_steps=100"), "M1a_lstm")
        # The windows read the take-off at reset and count only mission steps: step 0's schedule
        protocol = report_step0.evaluation_protocol(
            plain_ppo_config("+experiment=M2b_win64_bs1024", "+env.kwargs.warmup_steps=100"), "M2b_win64")
        self.assertEqual((protocol["schedule"], protocol["prelude_steps"]), ("step0", 100))

    def test_window_starts_with_the_real_takeoff(self):
        env = HistoryWindow(bp.BatteryPlaneEnv(observation_mode="hidden", warmup_steps=100), 16)
        observation, info = env.reset(seed=5, options={"b0": 0.4})
        frames = observation.reshape(16, 7)
        take_off, autopilot = info["warmup_observations"], info["warmup_actions"]
        for row, k in enumerate(range(85, 100)):
            np.testing.assert_allclose(frames[row], [*take_off[k], *autopilot[k - 1], 1.0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(frames[15, 4:], [*autopilot[-1], 1.0], rtol=0, atol=1e-6)

    def test_validation_counts_mission_steps_only(self):
        script = plain_ppo_script()
        cfg = plain_ppo_config("+experiment=M1a_lstm_bs256", *SMALL[:-2], "+env.kwargs.warmup_steps=100", "ppo.verbose=0",
                               "evaluation.reset_grid.eval_seeds=[1000]", "evaluation.reset_grid.eval_initial_battery=[0.3,0.6]")
        script.validate_ppo(cfg)
        torch.set_num_threads(1)
        train_env = script.make_train_env(cfg)
        self.assertEqual(train_env.observation_space.shape, (7,))
        model = MaskedRecurrentPPO("MlpLstmPolicy", train_env, n_steps=128, batch_size=128, n_epochs=1, seed=0, device="cpu",
                                   policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]}, autopilot_index=6)
        model.learn(256)
        buffer = model.rollout_buffer
        np.random.seed(0)
        for samples in buffer.get(128):
            flag = samples.observations[:, 6]
            self.assertEqual(float((samples.mask * flag).sum()), 0.0)
        schedule = list(bp.iter_eval_resets(cfg.evaluation.reset_grid))
        env, scheduler, raw = script.make_eval_env(cfg, schedule)
        scheduler.rewind()
        rewards, lengths, recorder = evaluate_flights(model, env, len(schedule), True, raw.core)
        env.close()
        for reward, length, flight in zip(rewards, lengths, recorder.flights):
            self.assertEqual(flight["length"], length - 100)   # Monitor counts the replayed take-off, the flight doesn't
            self.assertAlmostEqual(flight["return"], reward, places=4)
        self.assertEqual([flight["b0"] for flight in recorder.flights], [0.3, 0.6])


if __name__ == "__main__":
    unittest.main()
