from collections.abc import Mapping
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from omegaconf import OmegaConf
from torch import nn

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim, preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from .torch_layers import CONTEXT_HEADS, XLSTMRolloutEncoder
from .type_aliases import ContextualPolicyState

# How the actor and critic read the Gaussian head's variance: as it is, or its logarithm
CONTEXT_VARIANCE_INPUTS = ("variance", "log_variance")


class ContextualActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-critic policy for Contextual_PPO, with a recurrent context encoder shared by the actor and the critic.

    The encoder (``XLSTMRolloutEncoder``, recurrent SAC's encoder) reads ``[o_t, a_{t-1}]``, where ``a_{t-1}``
    is the previous executed action, and gives the context ``z_t``. The actor ``pi(a_t | o_t, z_t)`` and the
    critic ``V(o_t, z_t)`` are vanilla PPO's separate MLP branches (``mlp_extractor``) on the observation
    features concatenated with ``z_t``, followed by ``action_net`` (unsquashed diagonal Gaussian) and
    ``value_net``. ``z_t`` is never detached: how the policy and value losses train the encoder is decided by
    the algorithm (``ContextualPPO`` routes the gradients with ``parameter_groups()``).

    With ``context_head="gaussian"``, the encoder estimates the context as a Gaussian: ``z_t = [mu_t, sigma2_t]``,
    its mean and variance, each ``context_dim`` wide. The actor and critic read ``[o_t, mu_t, log sigma2_t]``
    (``context_variance_input="log_variance"``, the default) or ``[o_t, mu_t, sigma2_t]`` (``"variance"``), see
    ``context_inputs``; the variance itself is often tiny next to the observations, its logarithm is not.
    ``latent_dim`` is the width of ``z_t``: ``context_dim``, or ``2 * context_dim`` with the Gaussian head.

    ``auxiliary_dim`` gives the encoder an auxiliary head for supervised targets (BatteryPlane: the true speed),
    see ``XLSTMRolloutEncoder``. The actor and critic never read it: it only shapes the encoder's embedding.

    The policy keeps no memory: the ``ContextualPolicyState`` (encoder state and previous executed action) is
    passed in and returned explicitly, and no method modifies the state passed in. An executed action is the
    Gaussian sample clipped to the action-space bounds, in environment units, and that is what the memory
    carries as previous action.

    Initialization: the encoder is its own attribute, created before ``ActorCriticPolicy._build()``. It is
    therefore registered before that method creates the optimizer, and the orthogonal initialization
    (``ortho_init``), which covers only the features extractor, ``mlp_extractor``, ``action_net`` and
    ``value_net``, leaves it untouched: the xLSTM blocks keep the xlstm initialization and the projections
    PyTorch's, as in recurrent SAC. The heads are initialized exactly as in vanilla PPO.

    Interfaces:

    - collection (``ContextualOnPolicyAlgorithm``): ``initial_state``, ``reset_state``, ``forward``, ``predict_values``
    - PPO update: ``evaluate_actions`` over complete streams (or consecutive segments), ``parameter_groups``
    - inference: ``predict(observation, state, episode_start, deterministic)``, also through ``evaluate_policy``

    :param observation_space: Observation space, flat (not ``Dict``)
    :param action_space: Action space, continuous ``Box``
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function, also of the hidden layers of the context projection
    :param ortho_init: Whether to use or not orthogonal initialization (of the heads, never of the encoder)
    :param use_sde: Must be False: gSDE is not supported
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Unused without gSDE, kept for the ``ActorCriticPolicy`` interface
    :param use_expln: Unused without gSDE, kept for the ``ActorCriticPolicy`` interface
    :param squash_output: Must be False: actions are clipped, not squashed
    :param features_extractor_class: Features extractor of the observation input of the actor and critic.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param context_dim: Dimension of the context ``z_t`` (with the Gaussian head: of its mean and of its variance)
    :param xlstm_config: Config of the encoder's xLSTM block stack, see ``create_xlstm``.
        An OmegaConf node is stored as a plain dict.
    :param context_net_arch: Hidden layers of the encoder's context projection, linear by default
    :param context_activation: Output activation of the context projection (``"sigmoid"``: ``z_t`` in (0, 1)),
        None for a linear output, see ``XLSTMRolloutEncoder``. With the Gaussian head, the activation of the mean.
    :param context_head: ``"point"``: the encoder outputs the context ``z_t``; ``"gaussian"``: its mean and
        variance ``[mu_t, sigma2_t]``, which the actor and critic both read
    :param context_variance_input: With the Gaussian head, what the actor and critic read of the variance:
        ``"log_variance"`` (``log sigma2_t``) or ``"variance"`` (``sigma2_t``); unused with the point head
    :param auxiliary_dim: Size of the encoder's auxiliary head, None for no auxiliary head
    :param auxiliary_net_arch: Hidden layers of the auxiliary head, linear by default
    """

    encoder: XLSTMRolloutEncoder

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        context_dim: int = 8,
        xlstm_config: Mapping[str, Any] | None = None,
        context_net_arch: list[int] | None = None,
        context_activation: str | None = None,
        context_head: str = "point",
        context_variance_input: str = "log_variance",
        auxiliary_dim: int | None = None,
        auxiliary_net_arch: list[int] | None = None,
    ):
        if isinstance(observation_space, spaces.Dict):
            raise NotImplementedError("ContextualActorCriticPolicy doesn't support Dict observation spaces")
        if not isinstance(action_space, spaces.Box):
            raise NotImplementedError(f"ContextualActorCriticPolicy supports continuous Box action spaces, got {action_space}")
        if use_sde or squash_output:
            raise ValueError("ContextualActorCriticPolicy doesn't support gSDE (use_sde=True) or squashed actions")
        if context_head not in CONTEXT_HEADS:
            raise ValueError(f"context_head must be one of {CONTEXT_HEADS}, got {context_head!r}")
        if context_variance_input not in CONTEXT_VARIANCE_INPUTS:
            raise ValueError(f"context_variance_input must be one of {CONTEXT_VARIANCE_INPUTS}, got {context_variance_input!r}")

        # Read by _build(), which ActorCriticPolicy.__init__() calls
        self.context_dim = context_dim
        # A plain dict is saved and loaded with the policy
        self.xlstm_config = OmegaConf.to_container(xlstm_config, resolve=True) if OmegaConf.is_config(xlstm_config) else xlstm_config
        self.context_net_arch = context_net_arch
        self.context_activation = context_activation
        self.context_head = context_head
        self.context_variance_input = context_variance_input
        self.auxiliary_dim = auxiliary_dim
        self.auxiliary_net_arch = auxiliary_net_arch
        # Width of z_t, which the actor and critic read
        self.latent_dim = 2 * context_dim if context_head == "gaussian" else context_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                context_dim=self.context_dim,
                xlstm_config=self.xlstm_config,
                context_net_arch=self.context_net_arch,
                context_activation=self.context_activation,
                context_head=self.context_head,
                context_variance_input=self.context_variance_input,
                auxiliary_dim=self.auxiliary_dim,
                auxiliary_net_arch=self.auxiliary_net_arch,
            )
        )
        return data

    def _build(self, lr_schedule: Schedule) -> None:
        # Before ActorCriticPolicy._build(), which applies the orthogonal initialization to its own modules
        # and then creates the optimizer over all registered parameters
        self.encoder = self.make_encoder()
        super()._build(lr_schedule)
        self._check_parameter_groups()

    def _build_mlp_extractor(self) -> None:
        # Both branches receive [observation features, z_t]
        self.mlp_extractor = MlpExtractor(
            self.features_dim + self.latent_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def make_encoder(self) -> XLSTMRolloutEncoder:
        return XLSTMRolloutEncoder(
            observation_dim=get_flattened_obs_dim(self.observation_space),
            action_dim=get_action_dim(self.action_space),
            context_dim=self.context_dim,
            xlstm_config=self.xlstm_config,
            projection_net_arch=self.context_net_arch,
            activation_fn=self.activation_fn,
            context_activation=self.context_activation,
            context_head=self.context_head,
            auxiliary_dim=self.auxiliary_dim,
            auxiliary_net_arch=self.auxiliary_net_arch,
        ).to(self.device)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """
        The parameters by the loss that trains them (``ContextualPPO``'s gradient routing). Every parameter
        is in exactly one group:

        - ``encoder``: the context encoder (input projection, xLSTM blocks, context projection, auxiliary head) and the shared
          observation features extractor, if it has parameters: everything both branches depend on
        - ``actor``: the actor's features extractor when not shared, ``mlp_extractor.policy_net``, ``action_net``, ``log_std``
        - ``value``: the critic's features extractor when not shared, ``mlp_extractor.value_net``, ``value_net``
        """
        encoder = list(self.encoder.parameters())
        actor = [*self.mlp_extractor.policy_net.parameters(), *self.action_net.parameters(), self.log_std]
        value = [*self.mlp_extractor.value_net.parameters(), *self.value_net.parameters()]
        if self.share_features_extractor:
            encoder += list(self.features_extractor.parameters())
        else:
            actor = [*self.pi_features_extractor.parameters(), *actor]
            value = [*self.vf_features_extractor.parameters(), *value]
        return {"encoder": encoder, "actor": actor, "value": value}

    def _check_parameter_groups(self) -> None:
        grouped = [id(param) for params in self.parameter_groups().values() for param in params]
        if len(grouped) != len(set(grouped)) or set(grouped) != {id(param) for param in self.parameters()}:
            raise ValueError("parameter_groups() must hold every policy parameter exactly once")

    def initial_state(self, n_envs: int) -> ContextualPolicyState:
        """
        Memory at episode start: the encoder's initial state and a zero previous action.

        :param n_envs: Number of environments
        """
        prev_actions = th.zeros(n_envs, get_action_dim(self.action_space), device=self.device)
        return ContextualPolicyState(self.encoder.initial_state(n_envs), prev_actions)

    def reset_state(self, state: ContextualPolicyState, episode_start: th.Tensor | np.ndarray) -> ContextualPolicyState:
        """
        Reset the memory of the environments that start a new episode.

        :param state: Current memory
        :param episode_start: [n_envs] bool, True where a new episode starts
        """
        episode_start = th.as_tensor(episode_start, dtype=th.bool, device=self.device).reshape(-1)
        prev_actions = th.where(episode_start.unsqueeze(-1), th.zeros_like(state.prev_actions), state.prev_actions)
        return ContextualPolicyState(self.encoder.reset_state(state.encoder_state, episode_start), prev_actions)

    def encoder_observations(self, observations: th.Tensor, n_batch_dims: int = 1) -> th.Tensor:
        """
        Observation input of the encoder, preprocessed like the actor and critic inputs
        (float conversion, image scaling, one-hot encoding) and flattened.
        Collection and training must both use it, so the encoder always sees the same inputs.

        :param observations: [*batch_shape, *obs_shape], with ``n_batch_dims`` batch dimensions
        :param n_batch_dims: Number of batch dimensions, e.g. 2 for [envs, time] streams
        :return: [*batch_shape, n_features]
        """
        batch_shape = observations.shape[:n_batch_dims]
        flat_obs = observations.reshape(-1, *observations.shape[n_batch_dims:])
        preprocessed = preprocess_obs(flat_obs, self.observation_space, normalize_images=self.normalize_images)
        # Explicit width: streams can be empty
        return preprocessed.reshape(*batch_shape, self.encoder.observation_dim)  # type: ignore[union-attr]

    def obs_to_tensor(self, observation: np.ndarray | dict[str, np.ndarray]) -> tuple[PyTorchObs, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        float64 observations become float32: the networks are float32 and MPS doesn't support float64.

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        if isinstance(observation, np.ndarray) and observation.dtype == np.float64:
            observation = observation.astype(np.float32)
        return super().obs_to_tensor(observation)

    def executed_actions(self, actions: th.Tensor) -> th.Tensor:
        """
        The action sent to the environment: the Gaussian sample clipped to the action-space bounds.

        :param actions: [N, action_dim] sampled actions
        """
        assert isinstance(self.action_space, spaces.Box)
        low = th.as_tensor(self.action_space.low, dtype=actions.dtype, device=actions.device)
        high = th.as_tensor(self.action_space.high, dtype=actions.dtype, device=actions.device)
        return th.clamp(actions, low, high)

    def encode(self, obs: th.Tensor, state: ContextualPolicyState) -> tuple[th.Tensor, Any]:
        """
        One encoder step: ``z_t`` from ``[o_t, state.prev_actions]``, starting from ``state.encoder_state``.

        :param obs: ``o_t`` [N, *obs_shape]
        :param state: Memory before ``o_t``
        :return: The context ``z_t`` [N, latent_dim] and the encoder state after ``o_t``
        """
        return self.encoder.step(state.encoder_state, self.encoder_observations(obs), state.prev_actions)

    def context_inputs(self, latents: th.Tensor) -> th.Tensor:
        """
        What the actor and critic read of ``z_t``: ``z_t`` itself, except with the Gaussian head and
        ``context_variance_input="log_variance"``, where the variance is replaced by its logarithm.

        :param latents: ``z_t`` [..., latent_dim]
        :return: [..., latent_dim]
        """
        if self.context_head == "gaussian" and self.context_variance_input == "log_variance":
            mean, variance = self.encoder.split_context(latents)
            return th.cat([mean, th.log(variance)], dim=-1)  # type: ignore[arg-type]
        return latents

    def _branch_inputs(self, obs: th.Tensor, latents: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Inputs of the actor and critic branches: the observation features, concatenated with ``z_t``
        (``context_inputs``: with the log of the Gaussian head's variance by default).
        """
        features = self.extract_features(obs)
        pi_features, vf_features = (features, features) if self.share_features_extractor else features
        contexts = self.context_inputs(latents)
        return th.cat([pi_features, contexts], dim=-1), th.cat([vf_features, contexts], dim=-1)  # type: ignore[list-item]

    def forward(  # type: ignore[override]
        self, obs: th.Tensor, state: ContextualPolicyState, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, ContextualPolicyState]:
        """
        Encode ``[o_t, a_{t-1}]``, then sample ``a_t`` from ``pi(. | o_t, z_t)`` and evaluate ``V(o_t, z_t)``.

        :param obs: ``o_t`` [N, *obs_shape]
        :param state: Memory before ``o_t``
        :param deterministic: Whether to sample or use deterministic actions
        :return: The sampled actions (unclipped, as evaluated by ``log_prob``), the values [N, 1],
            the log probabilities [N], the contexts ``z_t`` [N, latent_dim] and the memory after ``o_t``,
            whose previous action is the executed (clipped) action
        """
        latents, encoder_state = self.encode(obs, state)
        pi_input, vf_input = self._branch_inputs(obs, latents)
        values = self.value_net(self.mlp_extractor.forward_critic(vf_input))
        distribution = self._get_action_dist_from_latent(self.mlp_extractor.forward_actor(pi_input))
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob, latents, ContextualPolicyState(encoder_state, self.executed_actions(actions))

    def get_distribution(  # type: ignore[override]
        self, obs: PyTorchObs, state: ContextualPolicyState
    ) -> tuple[Distribution, Any]:
        """
        Get the current policy distribution at ``o_t``, given the memory before it.

        :param obs: ``o_t`` [N, *obs_shape]
        :param state: Memory before ``o_t``
        :return: the action distribution and the encoder state after ``o_t``
            (the next memory also needs the executed action, see ``forward``)
        """
        latents, encoder_state = self.encode(obs, state)  # type: ignore[arg-type]
        pi_input, _ = self._branch_inputs(obs, latents)  # type: ignore[arg-type]
        return self._get_action_dist_from_latent(self.mlp_extractor.forward_actor(pi_input)), encoder_state

    def predict_values(self, obs: PyTorchObs, state: ContextualPolicyState) -> th.Tensor:  # type: ignore[override]
        """
        Get the estimated values of ``obs`` encoded from the memory ``state``, e.g. a lookahead to the next
        observation from the memory after ``o_t``, which already carries the executed ``a_t``.

        :param obs: Observation [N, *obs_shape]
        :param state: Memory before ``obs`` (left unchanged)
        :return: the estimated values [N, 1].
        """
        latents, _ = self.encode(obs, state)  # type: ignore[arg-type]
        _, vf_input = self._branch_inputs(obs, latents)  # type: ignore[arg-type]
        return self.value_net(self.mlp_extractor.forward_critic(vf_input))

    def evaluate_actions(  # type: ignore[override]
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        initial_state: Any,
        prev_actions: th.Tensor,
        episode_starts: th.Tensor,
        detach_context: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, Any, th.Tensor]:
        """
        Evaluate recorded actions over complete streams (or consecutive segments of them), re-encoding the
        history with the current parameters, as the PPO update needs it. The rows reset their memory where
        ``episode_starts`` is set, which also cuts the gradient to the earlier episode.

        :param obs: [E, T, *obs_shape]
        :param actions: [E, T, action_dim] recorded samples (unclipped)
        :param initial_state: Encoder state before ``obs[:, 0]``: ``ContextualRolloutSamples.initial_state``,
            or the (detached) state after the previous segment for truncated backpropagation
        :param prev_actions: [E, T, action_dim] previous executed actions, zero at episode starts
        :param episode_starts: [E, T] bool, reset the memory before ``obs[:, t]``
        :param detach_context: Run the encoder without gradient, so the actor's and critic's losses never
            reach it (e.g. an encoder trained only by a supervised loss on complete episodes)
        :return: estimated values, log likelihood of taking those actions and entropy of the action
            distribution, each [E, T], the encoder state after the last step, and the contexts ``z_t``
            [E, T, latent_dim] (without gradient when ``detach_context``)
        """
        if isinstance(initial_state, ContextualPolicyState):
            # The previous actions come from prev_actions, the stored stream
            raise TypeError("initial_state is the encoder state (e.g. ContextualPolicyState.encoder_state), not the whole memory")
        n_envs, n_steps = obs.shape[:2]
        with th.set_grad_enabled(th.is_grad_enabled() and not detach_context):
            latents, final_state = self.encoder.unroll(
                initial_state, self.encoder_observations(obs, n_batch_dims=2), prev_actions, episode_starts=episode_starts
            )
        flat_obs = obs.reshape(n_envs * n_steps, *obs.shape[2:])
        pi_input, vf_input = self._branch_inputs(flat_obs, latents.reshape(n_envs * n_steps, self.latent_dim))
        distribution = self._get_action_dist_from_latent(self.mlp_extractor.forward_actor(pi_input))
        log_prob = distribution.log_prob(actions.reshape(n_envs * n_steps, -1))
        values = self.value_net(self.mlp_extractor.forward_critic(vf_input))
        entropy = distribution.entropy()
        assert entropy is not None
        return (
            values.reshape(n_envs, n_steps),
            log_prob.reshape(n_envs, n_steps),
            entropy.reshape(n_envs, n_steps),
            final_state,
            latents,
        )

    def _predict(  # type: ignore[override]
        self, observation: PyTorchObs, state: ContextualPolicyState, deterministic: bool = False
    ) -> tuple[th.Tensor, ContextualPolicyState]:
        """
        Encode ``[o_t, a_{t-1}]``, then select ``a_t`` from ``o_t`` and ``z_t``.

        :return: The executed action (clipped to the action-space bounds) and the memory for the next step,
            whose previous action is that action
        """
        _, _, _, _, state = self.forward(observation, state, deterministic=deterministic)  # type: ignore[arg-type]
        return state.prev_actions, state

    def predict(  # type: ignore[override]
        self,
        observation: np.ndarray,
        state: ContextualPolicyState | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, ContextualPolicyState]:
        """
        Get the policy action from an observation, carrying the recurrent memory between calls.
        SB3's ``evaluate_policy`` already passes the state and the episode starts.

        The returned action is the executed action, clipped to the action-space bounds, and the returned memory
        carries it as previous action. A caller that sends a different action must update ``state.prev_actions``.
        Deterministic evaluation also needs the memory: pass the state back at every call.

        :param observation: the input observation
        :param state: The memory returned by the previous call, None to start new episodes
        :param episode_start: [n_envs] bool, True where a new episode starts
            (the memory of these environments is reset before the step)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the memory for the next call
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        n_envs = obs_tensor.shape[0]  # type: ignore[union-attr]

        with th.no_grad():
            if state is None:
                state = self.initial_state(n_envs)
            elif episode_start is not None:
                state = self.reset_state(state, episode_start)
            actions, state = self._predict(obs_tensor, state, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[assignment]

        return actions, state  # type: ignore[return-value]


MlpPolicy = ContextualActorCriticPolicy
