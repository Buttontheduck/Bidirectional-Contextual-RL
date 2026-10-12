"""Step 0 of ENCODER_PLAN.md: Contextual_PPO with the true battery in place of the learned context.

Selected by cppo.context_source=oracle (config/step0/cppo_oracle.yaml). The environment runs in observed mode,
whose observation is the four hidden-mode values followed by the true charge b_t (BatteryPlaneVec._obs). The xLSTM
encoder is replaced by TrueBatteryEncoder, which has no parameters and no memory, and the actor and critic read the
observation alone: the five inputs of plain PPO in observed mode. The rest is unchanged CPPO: the collector, the
whole-stream minibatches and the PPO losses.

Project glue, like the training script: the learner (contextual_ppo/) stays environment-agnostic.
"""

import numpy as np
import torch as th
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from torch import nn

from contextual_ppo.policies import ContextualActorCriticPolicy

# Column of the true charge b_t in observed- and full-mode observations
BATTERY_COLUMN = -1


class TrueBatteryEncoder(nn.Module):
    """
    Stand-in for ``XLSTMRolloutEncoder`` with the interface the policy and the learner use: ``z_t = b_t``, read
    from the observation. It has no parameters, so no update can change it, and no state (None, which the rollout
    buffer accepts for an encoder without state).
    """

    context_head = "point"
    auxiliary_dim = None

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.context_dim = self.output_dim = 1

    def initial_state(self, batch_size: int) -> None:
        return None

    def reset_state(self, state: None, reset_mask: th.Tensor | np.ndarray) -> None:
        return None

    def step(self, state: None, observations: th.Tensor, prev_actions: th.Tensor) -> tuple[th.Tensor, None]:
        return observations[..., BATTERY_COLUMN].unsqueeze(-1), None

    def unroll(
        self,
        state: None,
        observations: th.Tensor,
        prev_actions: th.Tensor,
        mask: th.Tensor | None = None,
        episode_starts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, None]:
        contexts = observations[..., BATTERY_COLUMN].unsqueeze(-1)
        if mask is not None:
            # Padding gets a zero context, as with the xLSTM encoder
            contexts = contexts * mask.unsqueeze(-1).to(contexts.dtype)
        return contexts, None

    def split_context(self, contexts: th.Tensor) -> tuple[th.Tensor, None]:
        return contexts, None


class TrueBatteryPolicy(ContextualActorCriticPolicy):
    """
    ``ContextualActorCriticPolicy`` with ``TrueBatteryEncoder``, whose actor and critic read the observation only.
    ``z_t = b_t`` is still computed and returned (rollout diagnostics), but never appended: the observed-mode
    observation already ends with ``b_t``, so the branches get ``[y, vx, vy, tau, b]``, the input of plain PPO.
    The xLSTM and context-head arguments are accepted and ignored.
    """

    def make_encoder(self) -> TrueBatteryEncoder:  # type: ignore[override]
        return TrueBatteryEncoder(get_flattened_obs_dim(self.observation_space), get_action_dim(self.action_space)).to(self.device)

    def _build_mlp_extractor(self) -> None:
        # Nothing of z_t reaches the branches
        self.context_input_dim = 0
        super()._build_mlp_extractor()

    def context_inputs(self, latents: th.Tensor) -> th.Tensor:
        return latents[..., :0]
