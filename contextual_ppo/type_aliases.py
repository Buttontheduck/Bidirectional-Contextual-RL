"""Common aliases for type hints"""

from typing import Any, NamedTuple

import torch as th


class ContextualRolloutSamples(NamedTuple):
    """
    Complete rollout streams of ``E`` environments, as returned by ``ContextualRolloutBuffer``.
    Transition fields are contiguous ``[E, T, ...]`` tensors on the buffer's device, in chronological
    order. Every entry is a real transition, so there is no padding mask.

    :param env_indices: [E] int64, environment that produced each stream
    :param initial_state: Encoder state before ``observations[:, 0]``, with the structure, dtypes and
        batch axes passed to ``set_initial_state()`` and batch size ``E``. An independent copy: mutating
        it changes neither the stored snapshot nor any other sample.
    :param observations: ``o_t`` [E, T, D_o], exactly as supplied to the policy
    :param next_observations: real successor ``o_{t+1}`` [E, T, D_o], the final observation at an episode end
    :param prev_actions: previous executed action [E, T, D_a], zero at episode starts.
        The encoder input at ``t`` is ``[observations[:, t], prev_actions[:, t]]``.
    :param actions: action evaluated by ``old_log_prob`` [E, T, D_a]
    :param executed_actions: action sent to the environment [E, T, D_a]
    :param episode_starts: [E, T] bool, reset the encoder state of that row before ``observations[:, t]``
    :param terminated: [E, T] bool, ``executed_actions[:, t]`` ended the episode in a true terminal state
    :param truncated: [E, T] bool, the episode was truncated after ``executed_actions[:, t]``
    :param episode_ids: [E, T] int64, episode-piece index within each stream, 0 at ``t = 0``
    :param old_values: [E, T] values from collection
    :param old_log_prob: [E, T] log probabilities of ``actions`` from collection, summed over action dimensions
    :param advantages: [E, T] GAE advantages (not normalized)
    :param returns: [E, T] ``advantages + old_values``
    :param contexts: [E, T, D_c] true context ``b_t``, None when contexts are not stored.
        Privileged: a diagnostic target (Contextual_PPO probes ``z_t -> b_t``), never an encoder or actor input
    :param next_contexts: [E, T, D_c] true successor context ``b_{t+1}``, None when contexts are not stored.
        Privileged, like ``contexts``
    :param context_estimates: [E, T, D_z] latents ``z_t`` cached at collection (diagnostics only),
        None when they are not stored
    """

    env_indices: th.Tensor
    initial_state: Any
    observations: th.Tensor
    next_observations: th.Tensor
    prev_actions: th.Tensor
    actions: th.Tensor
    executed_actions: th.Tensor
    episode_starts: th.Tensor
    terminated: th.Tensor
    truncated: th.Tensor
    episode_ids: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    contexts: th.Tensor | None
    next_contexts: th.Tensor | None
    context_estimates: th.Tensor | None


class ContextualEpisodeSamples(NamedTuple):
    """
    Complete episodes for supervised context training, as returned by ``SupervisedContextualEpisodeBuffer``.
    Episodes are left-aligned in ``[B, L, ...]`` tensors, ``L`` the longest sampled episode, in time order;
    shorter episodes are zero-padded after their end. Every episode starts at its first observation, so the
    encoder starts each row from its initial state.

    :param observations: ``o_t`` [B, L, D_o], the last valid step of an episode being its final observation
    :param prev_actions: previous executed action [B, L, D_a], zero at the first step of every episode.
        The encoder input at ``t`` is ``[observations[:, t], prev_actions[:, t]]``: never the labels.
    :param labels: true context of ``o_t`` [B, L, D_c] (BatteryPlane: the charge ``b_t``), the training target only;
        with an auxiliary head, followed by the auxiliary targets of ``o_t`` ([B, L, D_c + D_x])
    :param mask: [B, L] bool, True for valid steps, False for padding
    :param lengths: [B] int64, number of valid steps per episode
    :param episode_ids: [B] int64, the buffer's id of each episode (completion order)
    """

    observations: th.Tensor
    prev_actions: th.Tensor
    labels: th.Tensor
    mask: th.Tensor
    lengths: th.Tensor
    episode_ids: th.Tensor


class ContextualPolicyState(NamedTuple):
    """
    Memory that the Contextual_PPO collector and ``predict()`` carry between steps, batched over envs.

    :param encoder_state: Recurrent state of the context encoder before the next observation,
        i.e. ``S_t`` of the next step (see ``XLSTMContextEncoder``)
    :param prev_actions: [n_envs, action_dim] previous executed action (the action sent to the environment,
        inside the action-space bounds), zero at episode start
    """

    encoder_state: Any
    prev_actions: th.Tensor
