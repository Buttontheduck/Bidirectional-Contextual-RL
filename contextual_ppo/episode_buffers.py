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

from .buffers import ContextualRolloutBuffer, _check_finite_real, _check_int
from .type_aliases import ContextualEpisodeSamples


class _Episode(NamedTuple):
    episode_id: int
    split: str
    observations: np.ndarray  # [L, D_o]
    prev_actions: np.ndarray  # [L, D_a]
    labels: np.ndarray  # [L, D_c]


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
    ending them.

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
    ):
        self.n_envs = _check_int("n_envs", n_envs, minimum=1)
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
        self._unfinished: list[dict[str, list[np.ndarray]]] = [
            {"observations": [], "prev_actions": [], "labels": []} for _ in range(self.n_envs)
        ]
        self._last_actions = np.zeros((self.n_envs, self.action_dim), dtype=np.float32)

    def drop_unfinished(self) -> None:
        """
        Discard the unfinished episode of every environment, e.g. when the environments are reset without ending
        their episodes. Completed episodes are kept.
        """
        self._new_unfinished()

    def add(
        self,
        *,
        observations: np.ndarray | th.Tensor,
        executed_actions: np.ndarray | th.Tensor,
        labels: np.ndarray | th.Tensor,
        dones: np.ndarray,
        final_observations: np.ndarray | th.Tensor,
        final_labels: np.ndarray | th.Tensor,
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
            self._last_actions[env_idx] = executed_actions[env_idx]
            if dones[env_idx]:
                # The final observation belongs to this episode; the next add() starts the next one
                episode["observations"].append(final_observations[env_idx])
                episode["prev_actions"].append(executed_actions[env_idx].copy())
                episode["labels"].append(final_labels[env_idx])
                self._store(episode)
                self._unfinished[env_idx] = {"observations": [], "prev_actions": [], "labels": []}
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
