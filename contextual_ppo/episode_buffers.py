"""
Episode buffer for supervised training of the context encoder, next to ``ContextualRolloutBuffer``,
which keeps serving PPO's rollouts.
"""

from collections import deque
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.utils import get_device

from .buffers import ContextualRolloutBuffer, _check_finite_real, _check_int, _flatten_recurrent_state, _unflatten_recurrent_state
from .type_aliases import ContextualChunkSamples, ContextualEpisodeSamples


class _Episode(NamedTuple):
    episode_id: int
    split: str
    observations: np.ndarray  # [L, D_o]
    prev_actions: np.ndarray  # [L, D_a]
    labels: np.ndarray  # [L, D_c]
    states: np.ndarray | None = None  # [L, D_s] the encoder memory before each observation (stored-state windows only)


def encoder_state_rows(encoder: Any, state: Any) -> np.ndarray:
    """
    A batched encoder state as one row per batch element [B, D_s] float32: every leaf with its batch axis
    (``encoder._batch_axis``) moved first, flattened and concatenated in the state's fixed leaf order.
    """
    parts = []
    for path, leaf in _flatten_recurrent_state(state)[1]:
        leaf = leaf.detach().cpu().numpy() if isinstance(leaf, th.Tensor) else np.asarray(leaf)
        moved = np.moveaxis(leaf, encoder._batch_axis(path), 0)
        parts.append(moved.reshape(moved.shape[0], -1).astype(np.float32))
    return np.concatenate(parts, axis=1)


def encoder_state_from_rows(encoder: Any, rows: np.ndarray | th.Tensor, device: th.device | str = "cpu") -> Any:
    """The inverse of ``encoder_state_rows``: a batched encoder state, with the leaves' shapes of ``encoder.initial_state``."""
    rows = th.as_tensor(rows, dtype=th.float32, device=device)
    treedef, leaves = _flatten_recurrent_state(encoder.initial_state(1))
    parts, start = [], 0
    for path, leaf in leaves:
        axis = encoder._batch_axis(path)
        shape = list(leaf.shape)
        shape.pop(axis)
        size = int(np.prod(shape)) if shape else 1
        part = rows[:, start : start + size].reshape(rows.shape[0], *shape)
        parts.append(th.movedim(part, 0, axis).to(dtype=leaf.dtype).contiguous())
        start += size
    if start != rows.shape[1]:
        raise ValueError(f"State rows of width {rows.shape[1]} don't match this encoder's state ({start} values)")
    return _unflatten_recurrent_state(treedef, iter(parts))


class SupervisedContextualEpisodeBuffer:
    """
    Complete episodes of ``(o_t, a_{t-1}, label_t)`` for supervised training of the context encoder: the
    encoder reads ``[o_t, a_{t-1}]`` and learns to output ``label_t``, the true context of the same observation
    (BatteryPlane: the charge ``b_t``). The labels are targets only, never an encoder input.

    **Collection.** One unfinished episode per environment, which continues across rollout boundaries.
    ``add()`` appends step ``t`` of every environment: the observation ``o_t``, the previous executed action
    and the label of ``o_t``. The previous action is derived from the executed actions passed to ``add()``:
    zero at the first step of an episode, otherwise the action actually sent to the environment (after
    clipping) at the step before. When an episode ends (``dones``: termination or truncation), ``add()`` also
    appends the final observation (the VecEnv's ``terminal_observation``, never the automatic reset
    observation), with the executed action that led to it and its label, and closes the episode. The
    environment's next ``add()`` starts a new episode, so an automatic reset never mixes two episodes.
    ``drop_unfinished()`` discards the unfinished episodes, e.g. when the environments are reset without
    ending them. ``start_episodes()`` begins new episodes with a prelude (steps the agent did not act in, read by
    the encoder before the first observation), whose last action is then the first observation's previous action.

    **Storage.** Completed episodes only, oldest first, as numpy copies. Each is assigned, when it completes,
    to the ``"train"`` or the ``"validation"`` split (validation with probability ``validation_fraction``,
    seeded), so the two splits never share an episode. ``capacity`` bounds the stored timesteps: completing
    an episode evicts the oldest completed episodes, whole, until it fits. There are no PPO quantities
    (values, log probabilities, advantages, returns) and no encoder states: the learner starts the encoder
    from its initial state at each episode's first step and recomputes the states with the current weights.

    **Sampling.** ``sample()`` draws ``episode_batch_size`` distinct episodes of one split at random
    (seeded); ``iterate()`` visits a whole split once. Episode selection is shuffled, the timesteps inside an
    episode never are. Batches are ``ContextualEpisodeSamples``: ``[B, L, ...]`` tensors left-aligned and
    zero-padded to the longest sampled episode, with the episode lengths and a valid-step mask.

    :param n_envs: Number of parallel environments
    :param observation_space: Observation space, flat float32 ``Box``
    :param action_space: Action space, flat continuous ``Box``
    :param label_dim: Size of the label of each observation (BatteryPlane: 1, the charge; 2 with the speed
        target of the auxiliary head after it)
    :param capacity: Maximum number of stored timesteps over all completed episodes
    :param episode_batch_size: Number of episodes per ``sample()`` / ``iterate()`` batch
    :param validation_fraction: Probability that a completed episode goes to the validation split, in [0, 1)
    :param device: PyTorch device of the returned samples
    :param seed: Seed of the split assignment and of the sampling
    :param state_dim: Width of the encoder memory stored with every step (``encoder_state_rows``), for windows that start
        from the memory saved at collection (``sample_chunks(..., stored_states=True)``); None stores none
    """

    SPLITS = ("train", "validation")

    def __init__(
        self,
        n_envs: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        label_dim: int = 1,
        capacity: int = 200_000,
        episode_batch_size: int = 16,
        validation_fraction: float = 0.1,
        device: th.device | str = "auto",
        seed: int | None = None,
        state_dim: int | None = None,
    ):
        self.n_envs = _check_int("n_envs", n_envs, minimum=1)
        self.state_dim = None if state_dim is None else _check_int("state_dim", state_dim, minimum=1)
        if not (isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 1 and observation_space.dtype == np.float32):
            raise NotImplementedError(f"SupervisedContextualEpisodeBuffer supports flat float32 Box observation spaces, got {observation_space}")
        if not (isinstance(action_space, spaces.Box) and len(action_space.shape) == 1 and np.issubdtype(action_space.dtype, np.floating)):
            raise NotImplementedError(f"SupervisedContextualEpisodeBuffer supports flat continuous Box action spaces, got {action_space}")
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = int(observation_space.shape[0])
        self.action_dim = int(action_space.shape[0])
        self.label_dim = _check_int("label_dim", label_dim, minimum=1)
        self.capacity = _check_int("capacity", capacity, minimum=1)
        self.episode_batch_size = _check_int("episode_batch_size", episode_batch_size, minimum=1)
        self.validation_fraction = _check_finite_real("validation_fraction", validation_fraction)
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must be in [0, 1), got {validation_fraction}")
        self.device = get_device(device)
        self.seed = None if seed is None else _check_int("seed", seed)
        # Independent streams: sampling never changes which split a later episode goes to
        split_seed, sample_seed = np.random.SeedSequence(self.seed).spawn(2)
        self._split_rng = np.random.default_rng(split_seed)
        self._sample_rng = np.random.default_rng(sample_seed)

        # Completed episodes, oldest first
        self._episodes: deque[_Episode] = deque()
        self.stored_steps = 0
        self.completed_episodes = 0
        self.evicted_episodes = 0
        self._new_unfinished()

    def _new_unfinished(self) -> None:
        # Rows of the unfinished episode of every env, and the action it executed last (the next step's previous action)
        self._unfinished: list[dict[str, list[np.ndarray]]] = [self._empty_episode() for _ in range(self.n_envs)]
        self._last_actions = np.zeros((self.n_envs, self.action_dim), dtype=np.float32)

    def _empty_episode(self) -> dict[str, list[np.ndarray]]:
        return {"observations": [], "prev_actions": [], "labels": [], "states": []}

    def drop_unfinished(self) -> None:
        """
        Discard the unfinished episode of every environment, e.g. when the environments are reset without ending
        their episodes. Completed episodes are kept.
        """
        self._new_unfinished()

    def start_episodes(
        self,
        starts: np.ndarray,
        observations: np.ndarray,
        actions: np.ndarray,
        labels: np.ndarray,
        states: np.ndarray | None = None,
    ) -> None:
        """
        Begin the new episodes of the environments in ``starts`` with their preludes: steps the agent did not act in
        (e.g. an autopilot's take-off) that the encoder reads before the episode's first observation. The prelude's
        rows ``(o_k, a_{k-1}, label_k)`` come first, with ``a_{-1} = 0``, and its last action becomes the previous
        action of the episode's first observation, as during collection. Their unfinished episodes must be empty.

        :param starts: [N] bool, the environments whose episodes start
        :param observations: [N, K, D_o] the preludes' observations (rows outside ``starts`` are ignored)
        :param actions: [N, K, D_a] the preludes' executed actions
        :param labels: [N, K, D_c] the true context of each prelude observation (training targets only)
        :param states: [N, K, D_s] the encoder memory before each prelude observation; required with ``state_dim``
        """
        starts = np.asarray(starts, dtype=bool).reshape(-1)
        length = observations.shape[1] if np.ndim(observations) == 3 else 0
        expected = ((self.n_envs, length, self.obs_dim), (self.n_envs, length, self.action_dim), (self.n_envs, length, self.label_dim))
        given = (np.shape(observations), np.shape(actions), np.shape(labels))
        if starts.shape != (self.n_envs,) or length < 1 or given != expected:
            raise ValueError(f"start_episodes(): expected {self.n_envs} starts and preludes of shapes {expected}, got {given}")
        if (self.state_dim is None) != (states is None) or (states is not None and np.shape(states) != (self.n_envs, length, self.state_dim)):
            raise ValueError(f"start_episodes(): the prelude's encoder memories must be [{self.n_envs}, {length}, {self.state_dim}] "
                             f"exactly when state_dim is set, got {None if states is None else np.shape(states)}")
        for env_idx in np.flatnonzero(starts):
            episode = self._unfinished[env_idx]
            if episode["observations"]:
                raise RuntimeError(f"start_episodes(): env {env_idx} has an unfinished episode")
            prelude_actions = np.asarray(actions[env_idx], dtype=np.float32)
            outside = np.any((prelude_actions < self.action_space.low) | (prelude_actions > self.action_space.high), axis=-1)
            if outside.any():
                raise ValueError(f"start_episodes(): the prelude actions of env {env_idx} lie outside the action space bounds")
            prev_actions = np.concatenate([np.zeros((1, self.action_dim), np.float32), prelude_actions[:-1]])
            episode["observations"].extend(np.asarray(observations[env_idx], dtype=np.float32))
            episode["prev_actions"].extend(prev_actions)
            episode["labels"].extend(np.asarray(labels[env_idx], dtype=np.float32))
            if states is not None:
                episode["states"].extend(np.asarray(states[env_idx], dtype=np.float32))
            self._last_actions[env_idx] = prelude_actions[-1]

    def add(
        self,
        *,
        observations: np.ndarray | th.Tensor,
        executed_actions: np.ndarray | th.Tensor,
        labels: np.ndarray | th.Tensor,
        dones: np.ndarray,
        final_observations: np.ndarray | th.Tensor,
        final_labels: np.ndarray | th.Tensor,
        encoder_states: np.ndarray | None = None,
        final_encoder_states: np.ndarray | None = None,
    ) -> int:
        """
        Append step ``t`` of every environment, then close the episodes that ended with it.

        :param observations: ``o_t`` [N, D_o], exactly as given to the encoder during collection
        :param executed_actions: ``a_t`` [N, D_a], the action sent to the environment (after clipping):
            the previous action of the next step, and of the final observation where the episode ends
        :param labels: true context of ``o_t`` [N, D_c] (BatteryPlane: ``b_t``), mandatory
        :param dones: [N] bool, the episode ended with ``a_t`` (termination or truncation)
        :param final_observations: [N, D_o], the final observation (``terminal_observation``) where ``dones``;
            other rows are ignored
        :param final_labels: [N, D_c], the true context of the final observation where ``dones``; other rows are ignored
        :param encoder_states: [N, D_s] the encoder memory before ``o_t`` (``encoder_state_rows``); required with ``state_dim``
        :param final_encoder_states: [N, D_s] the memory before the final observation (after ``o_t``); with ``state_dim``
        :return: Number of episodes completed by this call
        """
        N = self.n_envs
        check = ContextualRolloutBuffer._check_array
        try:
            observations = check("observations", observations, (N, self.obs_dim), float)
            executed_actions = check("executed_actions", executed_actions, (N, self.action_dim), float)
            labels = check("labels", labels, (N, self.label_dim), float)
            dones = check("dones", dones, (N,), bool)
            final_observations = check("final_observations", final_observations, (N, self.obs_dim), float)
            final_labels = check("final_labels", final_labels, (N, self.label_dim), float)
            if (self.state_dim is None) != (encoder_states is None) or (encoder_states is None) != (final_encoder_states is None):
                raise ValueError("encoder_states and final_encoder_states are required exactly when state_dim is set")
            if self.state_dim is not None:
                encoder_states = check("encoder_states", encoder_states, (N, self.state_dim), float)
                final_encoder_states = check("final_encoder_states", final_encoder_states, (N, self.state_dim), float)
        except (TypeError, ValueError) as error:
            raise type(error)(f"add(): {error}") from error
        outside = np.any((executed_actions < self.action_space.low) | (executed_actions > self.action_space.high), axis=-1)
        if outside.any():
            raise ValueError(
                f"add(): executed_actions must lie inside the action space bounds, not for envs {np.flatnonzero(outside)}: "
                "pass the clipped action sent to the environment"
            )

        completed = 0
        for env_idx in range(N):
            episode = self._unfinished[env_idx]
            # Zero at the first step of an episode, the action sent at the step before otherwise
            prev_action = self._last_actions[env_idx] if episode["observations"] else np.zeros(self.action_dim, dtype=np.float32)
            episode["observations"].append(observations[env_idx])
            episode["prev_actions"].append(prev_action.copy())
            episode["labels"].append(labels[env_idx])
            if encoder_states is not None:
                episode["states"].append(encoder_states[env_idx])
            self._last_actions[env_idx] = executed_actions[env_idx]
            if dones[env_idx]:
                # The final observation belongs to this episode; the next add() starts the next one
                episode["observations"].append(final_observations[env_idx])
                episode["prev_actions"].append(executed_actions[env_idx].copy())
                episode["labels"].append(final_labels[env_idx])
                if final_encoder_states is not None:
                    episode["states"].append(final_encoder_states[env_idx])
                self._store(episode)
                self._unfinished[env_idx] = self._empty_episode()
                self._last_actions[env_idx] = 0.0
                completed += 1
        return completed

    def _store(self, rows: dict[str, list[np.ndarray]]) -> None:
        length = len(rows["observations"])
        if length > self.capacity:
            raise ValueError(f"An episode of {length} steps does not fit in capacity={self.capacity} timesteps")
        split = "validation" if self._split_rng.random() < self.validation_fraction else "train"
        episode = _Episode(
            episode_id=self.completed_episodes,
            split=split,
            observations=np.stack(rows["observations"]).astype(np.float32),
            prev_actions=np.stack(rows["prev_actions"]).astype(np.float32),
            labels=np.stack(rows["labels"]).astype(np.float32),
            states=np.stack(rows["states"]).astype(np.float32) if rows["states"] else None,
        )
        # Evict the oldest completed episodes, whole, until the new one fits
        while self.stored_steps + length > self.capacity:
            evicted = self._episodes.popleft()
            self.stored_steps -= len(evicted.observations)
            self.evicted_episodes += 1
        self._episodes.append(episode)
        self.stored_steps += length
        self.completed_episodes += 1

    def _check_split(self, split: str) -> None:
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")

    def episodes(self, split: str) -> list[_Episode]:
        """The stored episodes of a split, oldest first."""
        self._check_split(split)
        return [episode for episode in self._episodes if episode.split == split]

    def n_episodes(self, split: str | None = None) -> int:
        """Number of stored completed episodes, in one split or in both (None)."""
        return len(self._episodes) if split is None else len(self.episodes(split))

    def unfinished_steps(self) -> list[int]:
        """Number of steps of each environment's unfinished episode."""
        return [len(episode["observations"]) for episode in self._unfinished]

    def sample(self, split: str = "train", batch_size: int | None = None) -> ContextualEpisodeSamples:
        """
        ``batch_size`` (default ``episode_batch_size``) distinct complete episodes of a split, chosen at random,
        all of them when the split has fewer.

        :param split: ``"train"`` or ``"validation"``
        :param batch_size: Number of episodes
        """
        episodes = self.episodes(split)
        if not episodes:
            raise RuntimeError(f"No completed {split} episode is stored yet")
        batch_size = self.episode_batch_size if batch_size is None else _check_int("batch_size", batch_size, minimum=1)
        chosen = self._sample_rng.choice(len(episodes), size=min(batch_size, len(episodes)), replace=False)
        return self._to_samples([episodes[index] for index in chosen])

    def sample_chunks(
        self, split: str, n_chunks: int, chunk_len: int, burn_in: int = 0, stored_states: bool = False
    ) -> ContextualChunkSamples:
        """
        ``n_chunks`` windows of ``burn_in + chunk_len`` consecutive steps of a split's episodes, drawn so that every
        possible window start is equally likely (episodes shorter than a window give one, from their start). With
        ``stored_states``, each window comes with the encoder memory saved at collection before its first step (it needs
        ``state_dim``) and only episodes longer than ``burn_in`` are used, so every window has a trained step.

        :param split: ``"train"`` or ``"validation"``
        :param n_chunks: Number of windows (drawn with replacement)
        :param chunk_len: Steps that the loss is computed on
        :param burn_in: Leading steps that only rebuild the memory
        :param stored_states: Start every window from its stored memory (else from a blank memory)
        """
        _check_int("n_chunks", n_chunks, minimum=1)
        _check_int("chunk_len", chunk_len, minimum=1)
        _check_int("burn_in", burn_in, minimum=0)
        if stored_states and self.state_dim is None:
            raise ValueError("Windows from stored memories need a buffer built with state_dim")
        episodes = [e for e in self.episodes(split) if not stored_states or len(e.observations) > burn_in]
        if not episodes:
            raise RuntimeError(f"No completed {split} episode long enough for these windows is stored yet")
        window = burn_in + chunk_len
        lengths = np.array([len(e.observations) for e in episodes], dtype=np.int64)
        starts = np.maximum(lengths - window + 1, 1)
        chosen = self._sample_rng.choice(len(episodes), size=n_chunks, replace=True, p=starts / starts.sum())
        offsets = np.minimum((self._sample_rng.random(n_chunks) * starts[chosen]).astype(np.int64), starts[chosen] - 1)
        observations = np.zeros((n_chunks, window, self.obs_dim), dtype=np.float32)
        prev_actions = np.zeros((n_chunks, window, self.action_dim), dtype=np.float32)
        labels = np.zeros((n_chunks, window, self.label_dim), dtype=np.float32)
        mask = np.zeros((n_chunks, window), dtype=bool)
        states = np.zeros((n_chunks, self.state_dim), dtype=np.float32) if stored_states else None
        for row, (index, start) in enumerate(zip(chosen, offsets)):
            episode = episodes[index]
            n = min(window, len(episode.observations) - start)
            observations[row, :n] = episode.observations[start : start + n]
            prev_actions[row, :n] = episode.prev_actions[start : start + n]
            labels[row, :n] = episode.labels[start : start + n]
            mask[row, :n] = True
            if states is not None:
                states[row] = episode.states[start]
        tensor = lambda array: th.as_tensor(array, device=self.device)
        return ContextualChunkSamples(
            observations=tensor(observations), prev_actions=tensor(prev_actions), labels=tensor(labels), mask=tensor(mask),
            initial_states=None if states is None else tensor(states), burn_in=burn_in,
        )

    def iterate(
        self, split: str = "validation", batch_size: int | None = None, shuffle: bool = False
    ) -> Generator[ContextualEpisodeSamples, None, None]:
        """
        Every stored episode of a split once, in batches of ``batch_size`` (default ``episode_batch_size``).

        :param split: ``"train"`` or ``"validation"``
        :param batch_size: Number of episodes per batch
        :param shuffle: Random episode order (seeded); False keeps the completion order
        """
        episodes = self.episodes(split)
        batch_size = self.episode_batch_size if batch_size is None else _check_int("batch_size", batch_size, minimum=1)
        order = self._sample_rng.permutation(len(episodes)) if shuffle else np.arange(len(episodes))
        # The split is fixed now, not on the first next()
        return (
            self._to_samples([episodes[index] for index in order[start : start + batch_size]])
            for start in range(0, len(episodes), batch_size)
        )

    def _to_samples(self, episodes: list[_Episode]) -> ContextualEpisodeSamples:
        lengths = np.array([len(episode.observations) for episode in episodes], dtype=np.int64)
        batch, longest = len(episodes), int(lengths.max())

        def padded(field: str, dim: int) -> th.Tensor:
            array = np.zeros((batch, longest, dim), dtype=np.float32)
            for row, episode in enumerate(episodes):
                array[row, : lengths[row]] = getattr(episode, field)
            return th.as_tensor(array, device=self.device)

        mask = np.arange(longest)[None, :] < lengths[:, None]
        return ContextualEpisodeSamples(
            observations=padded("observations", self.obs_dim),
            prev_actions=padded("prev_actions", self.action_dim),
            labels=padded("labels", self.label_dim),
            mask=th.as_tensor(mask, device=self.device),
            lengths=th.as_tensor(lengths, device=self.device),
            episode_ids=th.as_tensor(np.array([episode.episode_id for episode in episodes], dtype=np.int64), device=self.device),
        )

    def state(self) -> dict[str, Any]:
        """Counts for logging."""
        return {
            "stored_steps": self.stored_steps,
            "train_episodes": self.n_episodes("train"),
            "validation_episodes": self.n_episodes("validation"),
            "completed_episodes": self.completed_episodes,
            "evicted_episodes": self.evicted_episodes,
        }
