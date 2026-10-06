import numbers
from collections.abc import Generator, Iterator, Mapping
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.utils import get_device

from .type_aliases import ContextualRolloutSamples


# Same helpers as in recurrent_sac/buffers.py. Copied rather than imported: importing that module
# runs recurrent_sac/__init__.py, which imports the SAC policies and xlstm.
def _flatten_recurrent_state(state: Any, path: tuple = ()) -> tuple[tuple, list[tuple[tuple, th.Tensor | np.ndarray]]]:
    """
    Split a nested recurrent state (dicts, lists, tuples, None, tensor/array leaves)
    into a structure description and its ``(path, leaf)`` pairs, in a fixed order.
    """
    if isinstance(state, (th.Tensor, np.ndarray)):
        return ("leaf",), [(path, state)]
    if state is None:
        return ("none",), []
    if isinstance(state, dict):
        keys = tuple(state.keys())
        children = [_flatten_recurrent_state(state[key], (*path, key)) for key in keys]
        return ("dict", keys, tuple(child[0] for child in children)), [leaf for child in children for leaf in child[1]]
    if isinstance(state, (list, tuple)):
        children = [_flatten_recurrent_state(value, (*path, i)) for i, value in enumerate(state)]
        return ("seq", type(state), tuple(child[0] for child in children)), [leaf for child in children for leaf in child[1]]
    raise TypeError(f"Unsupported recurrent-state element at {path}: {type(state)}")


def _unflatten_recurrent_state(treedef: tuple, leaves: Iterator[th.Tensor]) -> Any:
    """
    Inverse of ``_flatten_recurrent_state``.
    """
    kind = treedef[0]
    if kind == "leaf":
        return next(leaves)
    if kind == "none":
        return None
    if kind == "dict":
        return {key: _unflatten_recurrent_state(child, leaves) for key, child in zip(treedef[1], treedef[2], strict=True)}
    container_type, children = treedef[1], treedef[2]
    values = [_unflatten_recurrent_state(child, leaves) for child in children]
    # Namedtuples take positional fields, lists and tuples take an iterable
    return container_type(*values) if hasattr(container_type, "_fields") else container_type(values)


def _check_int(name: str, value: Any, minimum: int | None = None) -> int:
    # bool is an Integral, but True is never a meaningful size, axis or horizon
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return float(value)


class ContextualRolloutBuffer:
    """
    Rollout buffer for Contextual_PPO: fixed-length, chronological streams from ``N = n_envs`` parallel
    environments, sampled as complete streams, with one complete encoder state per environment
    at the start of the rollout.

    **Storage.** Arrays are time-major ``[T, N, ...]`` with ``T = n_steps``, detached CPU numpy copies,
    float32 for values and bool for flags. Each environment contributes exactly ``T`` real transitions:
    a stream may hold several episodes or part of a longer one. Unfinished episodes are trained on
    immediately, there is no replay, eviction or burn-in. See ``add()`` for the stored fields.

    **Timing.** For transition ``t``, ``S_t`` is the complete encoder state before it consumes
    ``[observations[t], prev_actions[t]]`` and ``z_t, S_{t+1} = encoder.step(S_t, observations[t], prev_actions[t])``.
    ``actions[t]`` is the action whose log probability PPO evaluates, ``executed_actions[t]`` the action
    sent to the environment (the policy's action transformation applied, so inside the action-space bounds),
    ``contexts[t]`` the true context (BatteryPlane: current battery ``b_t``) when ``observations[t]`` is observed,
    ``next_contexts[t]`` after the transition. Contexts are privileged diagnostics (Contextual_PPO's probe
    target): never an input of the encoder or the actor. Within a stream, with ``dones = terminated | truncated``,
    ``finalize()`` checks for ``t < T - 1``:

    - ``episode_starts[t + 1] == dones[t]``, and ``prev_actions`` is zero wherever ``episode_starts`` is set
    - if not ``dones[t]``: ``next_observations[t] == observations[t + 1]``, ``prev_actions[t + 1] == executed_actions[t]``
      and ``next_contexts[t] == contexts[t + 1]``

    At an episode end, ``next_observations[t]`` is the real final observation, never the automatic reset
    observation. A rollout boundary inside an episode does not reset memory: the collector carries
    observation, previous action, context and encoder state into the next rollout.

    **Recurrent state.** ``set_initial_state()`` stores ``S_0`` of every environment, the state the
    collector actually used, before the first observation of the rollout is processed. It may be any nesting
    of dicts, lists, tuples (including namedtuples) and None with tensor/array leaves; every leaf is stored
    whole (sLSTM: ``h, c, n, m`` and the convolution cache). Each leaf's batch axis comes from
    ``recurrent_state_batch_axis``: one int for every leaf, or a mapping from a key in the leaf's path to its
    axis, where the innermost configured key wins. The default ``{"slstm_state": 1, "mlstm_state": 0,
    "conv_state": 0}`` matches xlstm's ``step()`` states and must be checked against the encoder in use.
    The first snapshot fixes the structure, batch axes, dtypes and per-environment shapes; later rollouts
    must match it. Encoders that create a state entry lazily (e.g. a convolution cache from None) must
    return their complete initial state before the snapshot.

    **Targets.** ``compute_returns_and_advantage()`` runs GAE once on
    ``effective_rewards = rewards + intrinsic_rewards + timeout_bootstrap``. ``rewards`` stays the unmodified
    environment reward. ``timeout_bootstrap = gamma * V(final observation)`` on truncations, so a truncation's
    final value enters exactly once and its trace still stops at the reset; a true termination is not
    bootstrapped; a rollout boundary inside an episode bootstraps from ``last_values``. Afterwards the
    transition fields, values, log probabilities, advantages and returns are read-only until ``reset()``.

    **Lifecycle.** ``reset()``, ``set_initial_state()``, ``T`` calls to ``add()``, ``finalize()``,
    optionally ``add_intrinsic_rewards()``, ``compute_returns_and_advantage()``, then ``get()`` once per PPO
    epoch and/or ``get_all()``. Out-of-order calls raise.

    **Learner protocol** (the buffer never runs the encoder or updates parameters). For each sample, unroll
    the encoder chronologically from ``sample.initial_state``; before step ``t``, reset only the rows where
    ``sample.episode_starts[:, t]`` is set (prescribed initial state, which also cuts gradients to the previous
    episode). Recompute ``z_t`` with the current parameters (``context_estimates`` are cached collection
    outputs, never a replacement), then flatten ``[E, T]`` for the PPO losses. ``episode_starts`` is not a
    padding mask. Before the first parameter update, replay should reproduce ``old_log_prob`` and
    ``old_values`` within numerical tolerance. Clone state leaves the encoder mutates in place.

    This is not a drop-in replacement for SB3's ``RolloutBuffer`` / ``RecurrentRolloutBuffer``: the
    constructor takes ``n_steps`` instead of ``buffer_size``, and ``get()`` takes ``n_minibatches``
    instead of ``batch_size``. Observations are stored as given and never normalized: ``VecNormalize``
    is not supported and must be rejected by the algorithm, which has access to the environment.

    :param n_steps: Number of steps ``T`` per environment and rollout
    :param n_envs: Number of parallel environments ``N``
    :param observation_space: Observation space, flat float32 ``Box``
    :param action_space: Action space, flat continuous ``Box``
    :param gamma: Discount factor, shared with the algorithm
    :param gae_lambda: Factor for trade-off of bias vs variance for GAE, shared with the algorithm
    :param device: PyTorch device of the returned samples
    :param context_dim: Size of the stored true context; None stores neither ``contexts`` nor ``next_contexts``
    :param estimate_dim: Size of the stored collection latents ``z_t``; None does not store them
    :param recurrent_state_batch_axis: Batch axis of the recurrent-state leaves: one int for all leaves,
        or a mapping from a key in a leaf's path to its axis (the innermost matching key wins).
        None selects ``DEFAULT_STATE_BATCH_AXIS``.
    :param seed: Seed of the generator that permutes environments in ``get()``
    """

    # Batch axis of the state leaves returned by xlstm's sLSTMLayer / mLSTMLayer / xLSTMBlockStack .step()
    DEFAULT_STATE_BATCH_AXIS: dict[str, int] = {"slstm_state": 1, "mlstm_state": 0, "conv_state": 0}

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: th.device | str = "auto",
        context_dim: int | None = None,
        estimate_dim: int | None = None,
        recurrent_state_batch_axis: int | Mapping[str, int] | None = None,
        seed: int | None = None,
    ):
        self.n_steps = _check_int("n_steps", n_steps, minimum=1)
        self.n_envs = _check_int("n_envs", n_envs, minimum=1)

        if not (
            isinstance(observation_space, spaces.Box)
            and len(observation_space.shape) == 1
            and observation_space.dtype == np.float32
        ):
            raise NotImplementedError(
                f"ContextualRolloutBuffer supports flat float32 Box observation spaces, got {observation_space}"
            )
        if not (
            isinstance(action_space, spaces.Box)
            and len(action_space.shape) == 1
            and np.issubdtype(action_space.dtype, np.floating)
        ):
            raise NotImplementedError(f"ContextualRolloutBuffer supports flat continuous Box action spaces, got {action_space}")
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = int(observation_space.shape[0])
        self.action_dim = int(action_space.shape[0])

        self.gamma = _check_finite_real("gamma", gamma)
        self.gae_lambda = _check_finite_real("gae_lambda", gae_lambda)
        for name, value in (("gamma", self.gamma), ("gae_lambda", self.gae_lambda)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.device = get_device(device)
        self.context_dim = None if context_dim is None else _check_int("context_dim", context_dim, minimum=1)
        self.estimate_dim = None if estimate_dim is None else _check_int("estimate_dim", estimate_dim, minimum=1)

        # Axis 0 is valid, so only None selects the default
        if recurrent_state_batch_axis is None:
            self.recurrent_state_batch_axis: int | dict[str, int] = dict(self.DEFAULT_STATE_BATCH_AXIS)
        elif isinstance(recurrent_state_batch_axis, Mapping):
            self.recurrent_state_batch_axis = {}
            for key, axis in recurrent_state_batch_axis.items():
                if not isinstance(key, str):
                    raise TypeError(f"recurrent_state_batch_axis keys must be strings, got {key!r}")
                self.recurrent_state_batch_axis[key] = _check_int(f"recurrent_state_batch_axis[{key!r}]", axis)
        else:
            self.recurrent_state_batch_axis = _check_int("recurrent_state_batch_axis", recurrent_state_batch_axis)

        self.seed = None if seed is None else _check_int("seed", seed)
        self._rng = np.random.default_rng(self.seed)

        T, N = self.n_steps, self.n_envs
        self.observations = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.prev_actions = np.zeros((T, N, self.action_dim), dtype=np.float32)
        self.actions = np.zeros((T, N, self.action_dim), dtype=np.float32)
        self.executed_actions = np.zeros((T, N, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.timeout_bootstrap = np.zeros((T, N), dtype=np.float32)
        self.intrinsic_rewards = np.zeros((T, N), dtype=np.float32)
        self.episode_starts = np.zeros((T, N), dtype=bool)
        self.terminated = np.zeros((T, N), dtype=bool)
        self.truncated = np.zeros((T, N), dtype=bool)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)
        self.episode_ids = np.zeros((T, N), dtype=np.int64)
        self.contexts = None if self.context_dim is None else np.zeros((T, N, self.context_dim), dtype=np.float32)
        self.next_contexts = None if self.context_dim is None else np.zeros((T, N, self.context_dim), dtype=np.float32)
        self.context_estimates = None if self.estimate_dim is None else np.zeros((T, N, self.estimate_dim), dtype=np.float32)

        # Schema fixed by the first snapshot: structure and (path, batch axis, dtype, per-env shape) per leaf
        self._state_treedef: tuple | None = None
        self._state_leaf_specs: list[tuple[tuple, int, th.dtype | np.dtype, tuple[int, ...]]] | None = None
        # One array per leaf, [N, *per_env_shape]
        self._state_leaves: list[np.ndarray] = []

        self.reset()

    @property
    def full(self) -> bool:
        """
        Whether exactly ``n_steps`` rows are stored.
        """
        return self.pos == self.n_steps

    def reset(self) -> None:
        """
        Clear the rollout data and lifecycle flags, keeping the allocated arrays and the recurrent-state schema.
        Does not reset the environments or the encoder.
        """
        for array in self._arrays():
            array.flags.writeable = True
            array.fill(0)
        self._state_leaves = []
        self.pos = 0
        self.initial_state_set = False
        self.finalized = False
        self.intrinsic_rewards_set = False
        self.targets_computed = False

    def set_initial_state(self, state: Any) -> None:
        """
        Snapshot the encoder state of every environment before the first observation of the rollout
        is processed. Call it once per rollout, before the first ``add()``.

        :param state: Complete recurrent state batched over ``n_envs`` (see class docstring).
            None is accepted for an encoder without state.
        """
        if self.initial_state_set:
            raise RuntimeError("The initial state of this rollout was already set, call reset() before the next rollout")

        treedef, leaves = _flatten_recurrent_state(state)
        if self._state_treedef is not None and treedef != self._state_treedef:
            raise ValueError(
                "The structure of the recurrent state differs from the first snapshot. Encoders that create "
                "a state entry lazily (e.g. a convolution cache from None) must return their complete initial "
                f"state before the snapshot.\nExpected: {self._state_treedef}\nGot: {treedef}"
            )

        specs, arrays = [], []
        for leaf_idx, (path, leaf) in enumerate(leaves):
            if isinstance(leaf, th.Tensor):
                dtype = leaf.dtype
                leaf = leaf.detach()
                # numpy has no bfloat16; float32 holds every bfloat16 value exactly and the dtype is restored on read
                leaf = leaf.float() if dtype == th.bfloat16 else leaf
                # Copy: for CPU tensors .numpy() shares memory, and xlstm updates conv_state in place
                array = leaf.cpu().numpy().copy()
            else:
                dtype = leaf.dtype
                array = np.array(leaf, copy=True)
            axis = self._state_batch_axis(path)
            if not -array.ndim <= axis < array.ndim or array.shape[axis] != self.n_envs:
                raise ValueError(
                    f"Recurrent-state leaf {path} with shape {array.shape} has no batch axis {axis} "
                    f"of size n_envs={self.n_envs}, check recurrent_state_batch_axis"
                )
            axis %= array.ndim
            array = np.ascontiguousarray(np.moveaxis(array, axis, 0))
            array.flags.writeable = False
            spec = (path, axis, dtype, array.shape[1:])
            if self._state_leaf_specs is not None and spec != self._state_leaf_specs[leaf_idx]:
                expected = self._state_leaf_specs[leaf_idx]
                raise ValueError(
                    f"Recurrent-state leaf {path} changed since the first snapshot: expected batch axis {expected[1]}, "
                    f"dtype {expected[2]}, per-env shape {expected[3]}; got {axis}, {dtype}, {array.shape[1:]}"
                )
            specs.append(spec)
            arrays.append(array)

        if self._state_leaf_specs is None:
            self._state_treedef = treedef
            self._state_leaf_specs = specs
        self._state_leaves = arrays
        self.initial_state_set = True

    def add(
        self,
        *,
        observations: np.ndarray,
        next_observations: np.ndarray,
        prev_actions: np.ndarray,
        actions: np.ndarray,
        executed_actions: np.ndarray,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        values: np.ndarray | th.Tensor,
        log_probs: np.ndarray | th.Tensor,
        timeout_bootstrap: np.ndarray | None = None,
        intrinsic_rewards: np.ndarray | None = None,
        contexts: np.ndarray | None = None,
        next_contexts: np.ndarray | None = None,
        context_estimates: np.ndarray | th.Tensor | None = None,
    ) -> None:
        """
        Append transition ``t = pos`` of every environment. Arguments are keyword-only, numpy arrays
        or tensors with ``n_envs`` rows; they are validated and copied, nothing is reshaped except
        ``values`` of shape [N, 1] (SB3 value head output) to [N].

        :param observations: ``o_t`` [N, D_o], exactly as supplied to the policy
        :param next_observations: real successor [N, D_o]; at an episode end the final observation
            (``infos[i]["terminal_observation"]``), not the reset observation
        :param prev_actions: previous executed action [N, D_a], zero at episode starts
        :param actions: action evaluated by ``log_probs`` [N, D_a] (for an unsquashed Gaussian, the unclipped sample)
        :param executed_actions: action sent to the environment [N, D_a], inside the action-space bounds
            (for an unsquashed Gaussian, the clipped sample); the encoder's next ``prev_actions``
        :param rewards: unmodified environment reward [N]
        :param episode_starts: [N] bool, ``observations`` is the first observation of an episode
        :param terminated: [N] bool, effective true termination
        :param truncated: [N] bool, effective truncation. A row can't have both flags: a true termination
            takes precedence over a simultaneous time limit.
        :param values: ``V`` at collection [N] or [N, 1]
        :param log_probs: summed log probability of ``actions`` at collection [N]
        :param timeout_bootstrap: ``gamma * V(final observation)`` on truncated rows, zero elsewhere [N].
            Defaults to zero, allowed only when no row is truncated.
        :param intrinsic_rewards: already weighted reward bonus [N]; zero when omitted.
            Excludes a later ``add_intrinsic_rewards()``.
        :param contexts: true context ``b_t`` [N, D_c]; required if and only if ``context_dim`` is set
        :param next_contexts: true successor context [N, D_c], also at episode ends; same requirement
        :param context_estimates: collected latent ``z_t`` [N, D_z]; required if and only if ``estimate_dim`` is set
        """
        if not self.initial_state_set:
            raise RuntimeError("Call set_initial_state() with the encoder state at rollout start before the first add()")
        if self.full:
            raise RuntimeError(f"The rollout already holds n_steps={self.n_steps} rows, call finalize() or reset()")

        try:
            row = self._check_row(
                observations=observations,
                next_observations=next_observations,
                prev_actions=prev_actions,
                actions=actions,
                executed_actions=executed_actions,
                rewards=rewards,
                episode_starts=episode_starts,
                terminated=terminated,
                truncated=truncated,
                values=values,
                log_probs=log_probs,
                timeout_bootstrap=timeout_bootstrap,
                intrinsic_rewards=intrinsic_rewards,
                contexts=contexts,
                next_contexts=next_contexts,
                context_estimates=context_estimates,
            )
        except (TypeError, ValueError) as error:
            raise type(error)(f"add() at step {self.pos}: {error}") from error

        # Only write once every field is valid, so a rejected call leaves no partial row
        for name, array in row.items():
            getattr(self, name)[self.pos] = array
        if intrinsic_rewards is not None:
            self.intrinsic_rewards_set = True
        self.pos += 1

    def finalize(self) -> None:
        """
        Check the complete rollout (``n_steps`` rows, finite values, alignment of consecutive rows,
        see class docstring), derive ``episode_ids`` and make the transition fields read-only.
        """
        if self.finalized:
            raise RuntimeError("The rollout was already finalized, call reset() before the next rollout")
        if not self.full:
            raise RuntimeError(f"finalize() needs exactly n_steps={self.n_steps} rows, got {self.pos}")

        for name in self._float_fields():
            array = getattr(self, name)
            self._raise_where(~np.isfinite(array).reshape(self.n_steps, self.n_envs, -1).all(axis=-1), f"{name} is not finite")

        dones = self.terminated | self.truncated
        continuing = ~dones[:-1]
        self._raise_where(self.terminated & self.truncated, "terminated and truncated are both set")
        self._raise_where(self.episode_starts[1:] != dones[:-1], "episode_starts[t + 1] must equal terminated[t] | truncated[t]")
        self._raise_where(self.episode_starts & np.any(self.prev_actions != 0, axis=-1), "prev_actions must be zero at episode starts")
        self._raise_where(
            continuing & np.any(self.next_observations[:-1] != self.observations[1:], axis=-1),
            "next_observations[t] must equal observations[t + 1] within an episode",
        )
        self._raise_where(
            continuing & np.any(self.prev_actions[1:] != self.executed_actions[:-1], axis=-1),
            "prev_actions[t + 1] must equal executed_actions[t] within an episode",
        )
        if self.contexts is not None:
            self._raise_where(
                continuing & np.any(self.next_contexts[:-1] != self.contexts[1:], axis=-1),  # type: ignore[index]
                "next_contexts[t] must equal contexts[t + 1] within an episode",
            )

        # Episode-piece index, 0 at the first row of every stream; only meaningful per env and rollout
        cumulative_starts = np.cumsum(self.episode_starts, axis=0)
        self.episode_ids[:] = cumulative_starts - cumulative_starts[0]

        for name in (*self._float_fields(), *self._flag_fields(), "episode_ids"):
            getattr(self, name).flags.writeable = False
        self.finalized = True

    def add_intrinsic_rewards(self, intrinsic: np.ndarray | th.Tensor, coef: float) -> None:
        """
        Optional extension hook: set ``intrinsic_rewards = coef * intrinsic`` once,
        after ``finalize()`` and before ``compute_returns_and_advantage()``.

        :param intrinsic: [T, N] unweighted bonus
        :param coef: Finite weight
        """
        if not self.finalized:
            raise RuntimeError("add_intrinsic_rewards() needs a finalized rollout, call finalize() first")
        if self.targets_computed:
            raise RuntimeError("Intrinsic rewards must be added before compute_returns_and_advantage()")
        if self.intrinsic_rewards_set:
            raise RuntimeError("Intrinsic rewards were already assigned for this rollout, through add() or add_intrinsic_rewards()")
        coef = _check_finite_real("coef", coef)
        weighted = coef * self._check_array("intrinsic", intrinsic, (self.n_steps, self.n_envs), float)
        if not np.isfinite(weighted).all():
            raise ValueError("coef * intrinsic is not finite")
        self.intrinsic_rewards[:] = weighted
        self.intrinsic_rewards_set = True

    def compute_returns_and_advantage(self, last_values: np.ndarray | th.Tensor, dones: np.ndarray) -> None:
        """
        Compute GAE(lambda) advantages and lambda-returns once, from the collection values:

        ``delta[t] = effective_rewards[t] + gamma * (1 - dones[t]) * next_value - values[t]``
        ``advantage[t] = delta[t] + gamma * gae_lambda * (1 - dones[t]) * advantage[t + 1]``

        where ``next_value`` is ``values[t + 1]``, or ``last_values`` after the last row,
        and ``advantage[T] = 0``.

        :param last_values: ``V`` of the live observation after the last row [N] or [N, 1]
            (excluded by the done mask where that observation started a new episode)
        :param dones: ``terminated | truncated`` of the last row [N], as returned by the last ``env.step()``
        """
        if not self.finalized:
            raise RuntimeError("Call finalize() before compute_returns_and_advantage()")
        if self.targets_computed:
            raise RuntimeError("Returns and advantages were already computed; they stay fixed during the PPO epochs")
        last_values = self._check_array("last_values", last_values, (self.n_envs,), float, allow_column=True)
        dones = self._check_array("dones", dones, (self.n_envs,), bool)
        stored_dones = self.terminated | self.truncated
        if not np.array_equal(dones, stored_dones[-1]):
            raise ValueError(f"dones {dones} differ from terminated | truncated of the last stored row {stored_dones[-1]}")

        # The timeout correction is already in the reward: no second truncation bootstrap below
        effective_rewards = self.rewards + self.intrinsic_rewards + self.timeout_bootstrap
        next_alive = 1.0 - stored_dones.astype(np.float32)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_values = last_values if t == self.n_steps - 1 else self.values[t + 1]
            delta = effective_rewards[t] + self.gamma * next_alive[t] * next_values - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_alive[t] * last_gae
            self.advantages[t] = last_gae
        self.returns[:] = self.advantages + self.values
        if not (np.isfinite(self.advantages).all() and np.isfinite(self.returns).all()):
            raise ValueError("Advantages or returns are not finite")

        for array in (self.intrinsic_rewards, self.advantages, self.returns):
            array.flags.writeable = False
        self.targets_computed = True

    def same_episode_ahead(self, k: int) -> np.ndarray:
        """
        Validity mask of ``k``-step-ahead targets: ``[t, env]`` is True if and only if ``t + k < T`` and
        ``observations[t + k]`` belongs to the same episode piece as ``observations[t]``. Not a padding mask:
        one-step targets from ``next_observations`` exist for every row.

        :param k: Horizon, integer >= 1; ``k >= T`` gives an all-False mask
        :return: [T, N] bool, time-major like the storage
        """
        if not self.finalized:
            raise RuntimeError("same_episode_ahead() needs a finalized rollout, call finalize() first")
        k = _check_int("k", k, minimum=1)
        valid = np.zeros((self.n_steps, self.n_envs), dtype=bool)
        if k < self.n_steps:
            valid[:-k] = self.episode_ids[k:] == self.episode_ids[:-k]
        return valid

    def get(self, n_minibatches: int, shuffle: bool = True) -> Generator[ContextualRolloutSamples, None, None]:
        """
        One PPO epoch of complete-stream minibatches: a permutation of the environments split into
        ``n_minibatches`` groups of ``E = n_envs / n_minibatches`` streams. Every environment appears once;
        timesteps are never shuffled.

        :param n_minibatches: Number of minibatches, ``1 <= n_minibatches <= n_envs`` and dividing ``n_envs``
        :param shuffle: Permute environments with the buffer's generator; False keeps the natural order
        """
        self._require_targets()
        n_minibatches = _check_int("n_minibatches", n_minibatches, minimum=1)
        if n_minibatches > self.n_envs or self.n_envs % n_minibatches != 0:
            raise ValueError(
                f"n_minibatches must divide n_envs={self.n_envs} (each minibatch holds complete streams), got {n_minibatches}"
            )
        env_order = self._rng.permutation(self.n_envs) if shuffle else np.arange(self.n_envs)
        # Validation and the permutation happen now, not on the first next()
        return (self._get_samples(env_indices) for env_indices in env_order.reshape(n_minibatches, -1))

    def get_all(self) -> ContextualRolloutSamples:
        """
        All streams in natural environment order.
        """
        self._require_targets()
        return self._get_samples(np.arange(self.n_envs))

    def _get_samples(self, env_indices: np.ndarray) -> ContextualRolloutSamples:
        def streams(array: np.ndarray | None) -> th.Tensor | None:
            # [T, N, ...] -> [E, T, ...]. Fancy indexing copies, so samples never alias the storage.
            if array is None:
                return None
            return th.as_tensor(np.ascontiguousarray(array.swapaxes(0, 1)[env_indices]), device=self.device)

        return ContextualRolloutSamples(
            env_indices=th.as_tensor(np.array(env_indices, dtype=np.int64), device=self.device),
            initial_state=self._get_initial_state(env_indices),
            observations=streams(self.observations),  # type: ignore[arg-type]
            next_observations=streams(self.next_observations),  # type: ignore[arg-type]
            prev_actions=streams(self.prev_actions),  # type: ignore[arg-type]
            actions=streams(self.actions),  # type: ignore[arg-type]
            executed_actions=streams(self.executed_actions),  # type: ignore[arg-type]
            episode_starts=streams(self.episode_starts),  # type: ignore[arg-type]
            terminated=streams(self.terminated),  # type: ignore[arg-type]
            truncated=streams(self.truncated),  # type: ignore[arg-type]
            episode_ids=streams(self.episode_ids),  # type: ignore[arg-type]
            old_values=streams(self.values),  # type: ignore[arg-type]
            old_log_prob=streams(self.log_probs),  # type: ignore[arg-type]
            advantages=streams(self.advantages),  # type: ignore[arg-type]
            returns=streams(self.returns),  # type: ignore[arg-type]
            contexts=streams(self.contexts),
            next_contexts=streams(self.next_contexts),
            context_estimates=streams(self.context_estimates),
        )

    def _get_initial_state(self, env_indices: np.ndarray) -> Any:
        """
        Stored rollout-start state of the given environments, with the snapshot's structure, batch axes and dtypes.
        """
        assert self._state_leaf_specs is not None and self._state_treedef is not None
        leaves = []
        for storage, (_, axis, dtype, _) in zip(self._state_leaves, self._state_leaf_specs, strict=True):
            # Fancy indexing copies, so the returned tensors never alias the storage or each other
            leaf = th.as_tensor(np.ascontiguousarray(np.moveaxis(storage[env_indices], 0, axis)), device=self.device)
            leaves.append(leaf.to(dtype) if isinstance(dtype, th.dtype) else leaf)
        return _unflatten_recurrent_state(self._state_treedef, iter(leaves))

    def _check_row(
        self,
        *,
        timeout_bootstrap: Any,
        intrinsic_rewards: Any,
        contexts: Any,
        next_contexts: Any,
        context_estimates: Any,
        **fields: Any,
    ) -> dict[str, np.ndarray]:
        N = self.n_envs
        row = {
            name: self._check_array(name, fields[name], (N, dim), float)
            for name, dim in (
                ("observations", self.obs_dim),
                ("next_observations", self.obs_dim),
                ("prev_actions", self.action_dim),
                ("actions", self.action_dim),
                ("executed_actions", self.action_dim),
            )
        }
        row["rewards"] = self._check_array("rewards", fields["rewards"], (N,), float)
        row["values"] = self._check_array("values", fields["values"], (N,), float, allow_column=True)
        row["log_probs"] = self._check_array("log_probs", fields["log_probs"], (N,), float)
        for name in self._flag_fields():
            row[name] = self._check_array(name, fields[name], (N,), bool)

        # Catches a collector that stores (and feeds back to the encoder) the raw sample instead of the executed action
        executed = row["executed_actions"]
        outside = np.any((executed < self.action_space.low) | (executed > self.action_space.high), axis=-1)
        if outside.any():
            raise ValueError(
                f"executed_actions must lie inside the action space bounds, not for envs {np.flatnonzero(outside)}: "
                "store the clipped action sent to the environment"
            )

        both = row["terminated"] & row["truncated"]
        if both.any():
            raise ValueError(
                f"terminated and truncated are both set for envs {np.flatnonzero(both)}: pass effective flags "
                "(a true termination takes precedence over a simultaneous time limit)"
            )
        if timeout_bootstrap is None:
            if row["truncated"].any():
                raise ValueError("timeout_bootstrap (gamma * V(final observation)) is required when a row is truncated")
            row["timeout_bootstrap"] = np.zeros(N, dtype=np.float32)
        else:
            row["timeout_bootstrap"] = self._check_array("timeout_bootstrap", timeout_bootstrap, (N,), float)
            if np.any(row["timeout_bootstrap"][~row["truncated"]] != 0):
                raise ValueError("timeout_bootstrap must be zero on rows that are not truncated")
        if intrinsic_rewards is not None:
            row["intrinsic_rewards"] = self._check_array("intrinsic_rewards", intrinsic_rewards, (N,), float)

        for name, value, dim, switch in (
            ("contexts", contexts, self.context_dim, "context_dim"),
            ("next_contexts", next_contexts, self.context_dim, "context_dim"),
            ("context_estimates", context_estimates, self.estimate_dim, "estimate_dim"),
        ):
            if dim is None and value is not None:
                raise ValueError(f"{name} was given but the buffer was built with {switch}=None")
            if dim is not None:
                if value is None:
                    raise ValueError(f"{name} is required because {switch}={dim}")
                row[name] = self._check_array(name, value, (N, dim), float)
        return row

    @staticmethod
    def _check_array(
        name: str,
        value: Any,
        shape: tuple[int, ...],
        kind: type[bool] | type[float],
        allow_column: bool = False,
    ) -> np.ndarray:
        """
        Validated independent copy of ``value``: float32 and finite for ``kind=float``, bool for ``kind=bool``.
        Only with ``allow_column``, an SB3 value column ``[*shape, 1]`` becomes ``shape``.
        """
        if isinstance(value, th.Tensor):
            value = value.detach()
            value = (value.float() if value.dtype == th.bfloat16 else value).cpu().numpy()
        array = np.asarray(value)
        if allow_column and array.shape == (*shape, 1):
            array = array[..., 0]
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")

        if kind is bool:
            if array.dtype != np.bool_ and not (np.issubdtype(array.dtype, np.number) and np.isin(array, (0, 1)).all()):
                raise TypeError(f"{name} must hold booleans, got dtype {array.dtype}")
            return array.astype(bool)
        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
            raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
        array = array.astype(np.float32)
        if not np.isfinite(array).all():
            index = tuple(int(i) for i in np.argwhere(~np.isfinite(array))[0])
            raise ValueError(f"{name} contains non-finite values (first at index {index}: {array[index]})")
        return array

    def _raise_where(self, mask: np.ndarray, message: str) -> None:
        if mask.any():
            t, env_idx = np.argwhere(mask)[0]
            raise ValueError(f"{message} (first at step t={t}, env {env_idx})")

    def _require_targets(self) -> None:
        if not self.targets_computed:
            raise RuntimeError("Sampling needs a finalized rollout with returns and advantages, call compute_returns_and_advantage()")

    def _state_batch_axis(self, path: tuple) -> int:
        if isinstance(self.recurrent_state_batch_axis, int):
            return self.recurrent_state_batch_axis
        # The innermost key of the path with a configured axis wins
        for key in reversed(path):
            if isinstance(key, str) and key in self.recurrent_state_batch_axis:
                return self.recurrent_state_batch_axis[key]
        raise ValueError(f"No batch axis configured for recurrent-state leaf {path}, set recurrent_state_batch_axis")

    def _float_fields(self) -> list[str]:
        names = [
            "observations",
            "next_observations",
            "prev_actions",
            "actions",
            "executed_actions",
            "rewards",
            "timeout_bootstrap",
            "values",
            "log_probs",
        ]
        return names + [name for name in ("contexts", "next_contexts", "context_estimates") if getattr(self, name) is not None]

    @staticmethod
    def _flag_fields() -> tuple[str, ...]:
        return ("episode_starts", "terminated", "truncated")

    def _arrays(self) -> list[np.ndarray]:
        names = (*self._float_fields(), *self._flag_fields(), "intrinsic_rewards", "advantages", "returns", "episode_ids")
        return [getattr(self, name) for name in names]
