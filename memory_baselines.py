"""Memory baselines for BatteryPlane (PAPER_ASSESSMENT.md, Tier 1 item 1): standard end-to-end memory learners, set
against Contextual PPO's supervised encoder at step 0's settings.

- HistoryWindow: plain PPO reads the last `window` steps of [o_t, a_{t-1}, valid] as one flat input (no memory beyond
  the window). On the easy version the window starts filled with the take-off.
- RecurrentInput: sb3-contrib's RecurrentPPO, whose LSTM reads only the observation, gets [o_t, a_{t-1}], the history
  Contextual PPO's encoder reads. On the easy version the take-off is replayed as the first steps of every episode,
  with an autopilot flag; MaskedRecurrentPPO leaves those steps out of every PPO loss, so the LSTM reads the take-off
  (with gradients through it) but is never trained to act there.
- FusedLstmPolicy: sb3-contrib's MlpLstmPolicy with the LSTM run over each training sequence in one call instead of one
  step at a time in Python (the same computation, 10-30 times faster on this CPU; memory.fused_lstm, on by default).

Both read only what the plane showed: the hidden-mode observation, the executed (clipped) actions and, on the easy
version, the reset info's warmup_observations and warmup_actions. The privileged keys (b, b0, cap, saturated,
warmup_batteries, b_start) are never read. Without a take-off (warmup_steps = 0, the hard version, and every
battery_plane.py without the option) both wrappers just start from the first observation.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
import torch as th

MEMORY_KINDS = ("none", "window", "recurrent")
# Reset-info keys of the easy version's take-off that may be inputs (battery_plane.py, warmup_steps > 0)
PRELUDE_KEYS = ("warmup_observations", "warmup_actions")
# Info flag of a replayed take-off step: not part of the mission (flight_logging.py skips it)
AUTOPILOT_KEY = "autopilot"


def prelude_steps(env: gym.Env) -> int:
    """Steps of the take-off (and transition) the autopilot flies before the mission; 0 on the hard version."""
    return int(getattr(env.unwrapped.core, "prelude_steps", 0))


def prelude_from(info: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """The take-off's observations (before each autopilot action) and executed actions from a reset info, or None."""
    if PRELUDE_KEYS[0] not in info:
        return None
    observations, actions = (np.asarray(info[key], dtype=np.float32) for key in PRELUDE_KEYS)
    if observations.ndim != 2 or actions.shape != (len(observations), 2) or len(observations) == 0:
        raise ValueError(f"Unexpected take-off shapes: {observations.shape}, {actions.shape}")
    return observations, actions


class _ExecutedAction(gym.Wrapper):
    """Shared parts: the executed action is the agent's action clipped to the action space (SB3 clips before stepping
    already; clipping again keeps the recorded action exactly the executed one)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("The memory baselines need a Box action space")
        self.low = env.action_space.low.astype(np.float32)
        self.high = env.action_space.high.astype(np.float32)
        self.zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    def executed(self, action) -> np.ndarray:
        return np.clip(np.asarray(action, dtype=np.float32).reshape(self.low.shape), self.low, self.high)

    def bounds(self, *extra: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Bounds of one frame [o, a_prev, extra...]."""
        observation = self.env.observation_space
        low = np.concatenate([observation.low, self.low, [lo for lo, _ in extra]]).astype(np.float32)
        high = np.concatenate([observation.high, self.high, [hi for _, hi in extra]]).astype(np.float32)
        return low, high


class HistoryWindow(_ExecutedAction):
    """
    The last `window` steps of [o_t, a_{t-1}, valid] as one flat observation, oldest first. Steps before the first
    one are zeros with valid = 0. a_{-1} = 0; on the easy version the window starts filled with the take-off
    ([warmup_observations[k], warmup_actions[k-1]], valid 1), and the first mission step's previous action is the
    autopilot's last one, as Contextual PPO's encoder reads it.
    """

    def __init__(self, env: gym.Env, window: int):
        super().__init__(env)
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
            raise ValueError("window must be a positive integer")
        self.window = int(window)
        low, high = self.bounds((0.0, 1.0))
        self.frame_size = len(low)
        self.observation_space = spaces.Box(np.tile(low, self.window), np.tile(high, self.window), dtype=np.float32)
        self._frames = np.zeros((self.window, self.frame_size), dtype=np.float32)

    def _push(self, observation, previous_action) -> None:
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = np.concatenate([np.asarray(observation, dtype=np.float32), previous_action, [1.0]])

    def _observation(self) -> np.ndarray:
        return self._frames.reshape(-1).copy()

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._frames[:] = 0.0
        previous = self.zero_action
        prelude = prelude_from(info)
        if prelude is not None:
            observations, actions = prelude
            for k in range(len(observations)):
                self._push(observations[k], actions[k - 1] if k else self.zero_action)
            previous = actions[-1]
        self._push(observation, previous)
        return self._observation(), info

    def step(self, action):
        executed = self.executed(action)
        observation, reward, terminated, truncated, info = self.env.step(executed)
        self._push(observation, executed)
        return self._observation(), reward, terminated, truncated, info


class RecurrentInput(_ExecutedAction):
    """
    [o_t, a_{t-1}] for an LSTM that reads only the observation (sb3-contrib's RecurrentPPO), a_{-1} = 0.

    Easy version (a take-off before the mission): reset returns the take-off's first observation, and the next
    prelude steps replay the take-off: the autopilot's observations and executed actions, whatever the agent does,
    reward 0, never done, info {"autopilot": True}. The step after the last one returns the hand-over observation
    (the env's own reset observation) with the autopilot's last action, and the mission runs as usual from there.
    A last input, `autopilot`, is 1 on the replayed observations (the agent's action there has no effect) and 0
    from the hand-over on; MaskedRecurrentPPO masks the steps taken from autopilot observations.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.prelude_steps = prelude_steps(env)
        extra = ((0.0, 1.0),) if self.prelude_steps else ()
        low, high = self.bounds(*extra)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # Index of the autopilot input (MaskedRecurrentPPO), None without a take-off
        self.autopilot_index = len(low) - 1 if self.prelude_steps else None
        self._replay: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._k = 0

    def _frame(self, observation, previous_action, autopilot: float | None = None) -> np.ndarray:
        parts = [np.asarray(observation, dtype=np.float32), previous_action]
        if self.prelude_steps:
            parts.append(np.array([0.0 if autopilot is None else autopilot], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if not self.prelude_steps:
            return self._frame(observation, self.zero_action), info
        prelude = prelude_from(info)
        if prelude is None or len(prelude[0]) != self.prelude_steps:
            raise ValueError("The take-off is missing from the reset info")
        observations, actions = prelude
        self._replay, self._k = (observations, actions, np.asarray(observation, dtype=np.float32)), 0
        return self._frame(observations[0], self.zero_action, 1.0), info

    def step(self, action):
        if self._replay is not None:
            observations, actions, handover = self._replay
            self._k += 1
            if self._k < len(observations):
                observation = self._frame(observations[self._k], actions[self._k - 1], 1.0)
            else:
                observation = self._frame(handover, actions[-1], 0.0)
                self._replay = None
            return observation, 0.0, False, False, {AUTOPILOT_KEY: True}
        executed = self.executed(action)
        observation, reward, terminated, truncated, info = self.env.step(executed)
        return self._frame(observation, executed, 0.0), reward, terminated, truncated, info


class AutopilotMaskedRolloutBuffer(RecurrentRolloutBuffer):
    """
    RecurrentPPO's rollout buffer, with every step taken from an autopilot observation (input `autopilot_index` = 1)
    masked out of the PPO losses: sb3-contrib's loss mask (its padding) times 1 - autopilot. The LSTM still reads those
    steps in each training sequence; only their policy, value and entropy terms (and their share of the advantage
    normalization) are dropped. None: the plain sb3-contrib buffer.
    """

    def __init__(self, *args, autopilot_index: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.autopilot_index = autopilot_index

    def _get_samples(self, batch_inds, env_change, env=None):
        samples = super()._get_samples(batch_inds, env_change, env)
        if self.autopilot_index is None:
            return samples
        # get() has flattened the buffer to (n_steps * n_envs, ...); pad like the other fields (padding: 0)
        controlled = 1.0 - self.observations[batch_inds][:, self.autopilot_index]
        return samples._replace(mask=samples.mask * self.pad_and_flatten(controlled))


class FusedLstmPolicy(RecurrentActorCriticPolicy):
    """
    sb3-contrib's MlpLstmPolicy, with the LSTM run over each sequence in one call.

    sb3-contrib walks through a sequence one step at a time in Python whenever the sequence holds an episode start, to
    reset the memory at that step; in training nearly every minibatch does. Its rollout buffer starts a new sequence at
    every episode start (and pads at the end, with episode start 0), so a reset can only fall on a sequence's first
    step: resetting the initial memory there and running the whole sequence in one nn.LSTM call is the same computation.
    The outputs are identical and the gradients equal up to floating-point rounding (checks/memory_baseline_checks.py).
    Any other pattern of episode starts falls back to sb3-contrib's loop.
    """

    @staticmethod
    def _process_sequence(features, lstm_states, episode_starts, lstm):
        n_seq = lstm_states[0].shape[1]
        starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        if not th.all(starts[1:] == 0.0):
            return RecurrentActorCriticPolicy._process_sequence(features, lstm_states, episode_starts, lstm)
        sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        keep = (1.0 - starts[0]).view(1, n_seq, 1)
        output, lstm_states = lstm(sequence, (keep * lstm_states[0], keep * lstm_states[1]))
        return th.flatten(output.transpose(0, 1), start_dim=0, end_dim=1), lstm_states


class MaskedRecurrentPPO(RecurrentPPO):
    """sb3-contrib's RecurrentPPO with AutopilotMaskedRolloutBuffer; identical to it when autopilot_index is None."""

    def __init__(self, *args, autopilot_index: int | None = None, **kwargs):
        self.autopilot_index = autopilot_index
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        buffer = self.rollout_buffer
        self.rollout_buffer = AutopilotMaskedRolloutBuffer(
            buffer.buffer_size, self.observation_space, self.action_space, buffer.hidden_state_shape, self.device,
            gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=self.n_envs, autopilot_index=self.autopilot_index,
        )


def memory_kind_from(cfg) -> str:
    """cfg.memory.kind of a plain-PPO config; none for configs from before the memory baselines."""
    memory = cfg.get("memory")
    kind = "none" if memory is None else str(memory.get("kind", "none"))
    if kind not in MEMORY_KINDS:
        raise ValueError(f"memory.kind must be one of {MEMORY_KINDS}, got {kind!r}")
    return kind


def memory_wrapper_from(cfg):
    """The per-environment wrapper of cfg.memory (None for kind none), applied inside each Monitor."""
    kind = memory_kind_from(cfg)
    if kind == "window":
        window = int(cfg.memory.window)
        return lambda env: HistoryWindow(env, window)
    if kind == "recurrent":
        return RecurrentInput
    return None
