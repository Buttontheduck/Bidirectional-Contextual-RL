# Contextual_PPO: implementation plan for PPO with an xLSTM context encoder

**Audience:** coding agent implementing the algorithm, and a human reviewing the design.  
**Current milestone:** recurrent PPO with an inferred context supplied to both actor and value critic; the encoder is trained jointly through PPO losses.  
**Project name:** `Contextual_PPO`. Suggested Python class: `ContextualPPO`.  
**Status:** implementation specification. Implemented so far: the encoder adapter (`torch_layers.py`, tested by `test_torch_layers.py`), the rollout buffer (`buffers.py`, tested by `test_buffer.py`), the recurrent collector (`on_policy_algorithm.py`, tested by `test_on_policy_algorithm.py`) the policy (`policies.py`, tested by `test_policies.py`) and the PPO learner (`contextual_ppo.py`, tested by `test_contextual_ppo.py`, which includes a small memory task the agent must learn). The project integration (Stage 6) is in `Env_BatteryPlane/ppo/`: `main_battery_plane_contextual_ppo.py`, `configs/battery_plane_contextual_ppo.yaml`, `configs/experiment/c01_hidden_contextual.yaml`, tested by `tests/test_contextual_ppo_entry.py`. Nothing has been validated experimentally on BatteryPlane yet.

## 1. Objective and authoritative design

Extend Stable-Baselines3's vanilla PPO so that it can use interaction history in a partially observed environment such as BatteryPlane. A shared xLSTM encoder processes the current observation and the previous executed action, maintains memory across the episode, and produces a compact inferred context. Both the actor and the critic receive this context together with the current observation.

The intended architecture is:

> **PPO + shared xLSTM context encoder + recurrent rollout buffer + joint policy/value training.**

This is **JCPL-inspired joint context and policy learning**, adapted to a recurrent, on-policy algorithm. It is not a reproduction of JCPL's original architecture or training procedure. The learned context is a task-relevant representation; it is not required to equal the true battery or to be a calibrated probability distribution.

This document defines the current implementation milestone. In particular, it supersedes the earlier buffer draft's privileged-critic example: **the critic receives inferred context `z_t`, not true battery `b_t`. Both policy and value losses train the encoder.**

| Requirement | Decision |
|---|---|
| RL algorithm | PPO with its clipped policy objective and GAE |
| Encoder | Existing xLSTM implementation; start with one sLSTM block |
| Encoder inputs | Current observation and previous executed action |
| Actor input | Current observation plus inferred context |
| Critic input | Current observation plus the same inferred context |
| Encoder supervision | Value loss plus `encoder_actor_loss_weight` × (policy loss + entropy term), routed per parameter group as in RecurrentSAC (§8) |
| Memory reset | At episode starts, including after truncation |
| Training data | Current on-policy rollout, consumed for several epochs |
| Package layout | `stable_baselines3/contextual_ppo/` in the installed SB3 fork, mirroring how `recurrent_sac/` extends `sac/` (§4) |
| Environment | The project's local `battery_plane.py` through `bp.BatteryPlaneEnv`; autogenic context (§1.1) |
| Configuration | Explicit, serializable constructor arguments; the project's Hydra config composes the PPO baseline's (§10) |
| Initial environment support | Flat float32 Box observations and continuous Box actions |

Rewards enter the PPO return/advantage calculation, but are not encoder inputs. True context from environment information may be logged for diagnostics; neither network requires it to act, estimate values, or train this baseline. The diagnostic target is the current charge `b_t`, not the starting charge `b0` (§12).

### 1.1 Environment

Always use the project's local `battery_plane.py`, the one next to `main_battery_plane.py`, `ppo/` and `configs/`. Import it as `import battery_plane as bp`, build each copy with `bp.BatteryPlaneEnv(observation_mode=..., **env_kwargs)`, and wrap the copies in `Monitor` and an SB3 `DummyVecEnv`, as `ppo/main_battery_plane_ppo.py` already does. Never call `gym.make("BatteryPlane-v0")` or the other registered BatteryPlane ids. In the `carlos` env they resolve to an older copy installed inside gymnasium (`E_hover_s=80`, `b0_range=(0.2, 1.0)`, backward-flight lift bug). Do not wrap the vectorized `BatteryPlaneVec` core directly either: SB3's VecEnv provides the auto-reset and `terminal_observation` conventions the collector relies on (§7).

The context is **autogenic**. The cruise-motor cap follows the current charge (`cap_on="b"`, the environment default), and only the agent's own thrust drains the charge. The environment's allogenic control `cap_on="b0"` (cap frozen at the starting charge) is not used in this project.

With the default `terminal_at_T=True`, reaching the horizon is a termination, so BatteryPlane never truncates. The truncation path in §7–§8 is still required for general VecEnvs and is tested with synthetic fixtures.

The `contextual_ppo` package stays environment-agnostic and never imports `battery_plane`. Only training scripts, evaluation scripts and BatteryPlane-specific checks do.

## 2. Architecture and timestep alignment

Let `S_t` be the complete encoder state before processing the observation at time `t`. Let `u_{t-1}` be the previous action sent to the environment. Define:

\[
x_t=[\operatorname{preprocess}(o_t),u_{t-1}],\qquad
(e_t,S_{t+1})=\operatorname{xLSTM}_{\phi}(W_{\phi}x_t,S_t),\qquad
z_t=P_{\phi}(e_t).
\]

The input projection, recurrent network, and context projection all belong to encoder parameters `phi`. Start with an xLSTM embedding width of 64 and a context dimension of 8; both remain configurable.

The context projection `P_phi` is an MLP head (`context_net_arch`) with an optional output activation (`context_activation`, in `XLSTMRolloutEncoder`). The project config uses `64 -> 64 -> 1 -> sigmoid`: `z_t` is one number in (0, 1), and the actor and critic read `[o_t, z_t]`. It is trained by the PPO losses only, so nothing fixes what `z_t` means. On a cue-memory test task it learns to separate the cues, with the direction differing between seeds. `context_dim=8`, `context_net_arch=[]`, `context_activation=null` restores the 8-dim linear context.

The project's xLSTM stack has two blocks: an mLSTM block (matrix memory, causal convolution of 4 steps, 4 heads) and then the sLSTM block (`num_blocks: 2`, `slstm_at: [1]`). Its state adds the mLSTM's `c, n, m` and the convolution cache (batch axis 0, as `STATE_BATCH_AXIS` already declares), all zero at an episode start. The buffer snapshots, resets and replays them like the sLSTM state. The tests check that the gradient is cut at resets and that the PPO ratio is exactly 1 after collection. `context_length` is required by xlstm's mLSTM config but only sizes the full-sequence causal mask; the encoder always steps, so it does not limit the memory.

\[
a_t\sim\pi_{\theta}(\cdot\mid o_t,z_t),\qquad
V_t=V_{\psi}(o_t,z_t).
\]

Use separate actor and value MLP branches after concatenating observation features with the shared `z_t`. PPO's critic estimates a scalar state/history value. Do not reuse the SAC action-conditioned twin-Q heads or target networks.

The context for selecting `a_t` can depend on `o_0,...,o_t` and executed actions before `t`. It must not depend on `a_t`, `r_t`, or `o_{t+1}`. At an episode start, reset the encoder and set the previous action to a zero sentinel before consuming the first observation.

For the supplied BatteryPlane environment, observations already contain scaled coordinates and velocities. Reuse the existing preprocessing behavior for float vector observations; do not divide them again by physical reference values. In particular, the environment already scales both velocities using `v_ref`.

## 3. What to reuse from the existing recurrent SAC code

The provided source files have been inspected. Reuse the encoder and state-handling ideas, while building the PPO-specific policy, collector, buffer, and learner.

| Existing component | Reuse in Contextual_PPO |
|---|---|
| `create_xlstm`, `DEFAULT_XLSTM_CONFIG` | Build the same configurable official xLSTM blocks |
| `XLSTMContextEncoder` | Input projection, recurrent step, context projection, state initialization and copying |
| Nested state flatten/unflatten helpers | Preserve arbitrary complete xLSTM state trees |
| `RecurrentPolicyState` pattern | Carry encoder state and previous executed action during inference |
| Encoder observation preprocessing | Keep collection and training inputs consistent |
| SAC episodic replay | Replace with an on-policy recurrent rollout buffer |
| SAC actor and twin-Q critics | Replace with PPO distribution and value branches |
| SAC target encoder/critics and temperature optimizer | Not part of this PPO design |

The existing encoder's `unroll(..., mask=...)` treats `mask` as padding validity. PPO also needs `episode_starts`, meaning reset memory before an input. Add a separate argument or a PPO adapter; never reinterpret the existing padding argument and break SAC behavior. Implemented as `XLSTMRolloutEncoder.unroll(state, observations, prev_actions, episode_starts=...)` in `contextual_ppo/torch_layers.py`: a subclass with no new parameters, so the state dict and state layout equal RecurrentSAC's.

The complete recurrent state includes sLSTM's `h,c,n,m` and any convolution cache. Preserve any mLSTM state leaves when that configuration is introduced. State sizes and batch axes must come from the actual backend layout; do not assume an ordinary LSTM `(h,c)` pair is sufficient.

Import the encoder narrowly from `stable_baselines3.recurrent_sac.torch_layers`; do not copy it, so both algorithms always run the same encoder code. `contextual_ppo/torch_layers.py` re-exports `create_xlstm`, `DEFAULT_XLSTM_CONFIG` and `XLSTMContextEncoder`, and the rest of the package imports them from there, as `recurrent_sac` imports from its own `torch_layers.py`. `recurrent_sac` never imports `contextual_ppo`, so there is no import cycle. Do not modify `recurrent_sac/`: existing recurrent SAC behavior must remain intact. The buffer keeps its own copy of the state flatten/unflatten helpers.

## 4. Package layout and SB3 integration (mirrors `recurrent_sac`)

Contextual_PPO extends SB3's PPO the same way `recurrent_sac/` extends `sac/`. The package lives directly in the SB3 fork installed in the `carlos` env, at `stable_baselines3/contextual_ppo/`, next to `recurrent_sac/`. Keep this setup; there is no separate repository or editable install.

| `recurrent_sac/` (derived from `sac/`) | `contextual_ppo/` (derived from `ppo/`) | Responsibility |
|---|---|---|
| `__init__.py` | `__init__.py` | Export `ContextualPPO`, `ContextualActorCriticPolicy`, `MlpPolicy`, `ContextualRolloutBuffer` and the types below |
| `type_aliases.py` (`EpisodicReplayBufferSamples`, `RecurrentPolicyState`) | `type_aliases.py` | **Done**: `ContextualRolloutSamples`; `ContextualPolicyState` (encoder state and previous executed action, same fields as `RecurrentPolicyState`) |
| `torch_layers.py` (`XLSTMContextEncoder`, `create_xlstm`) | `torch_layers.py` | **Done**: imports the encoder from `recurrent_sac`; `XLSTMRolloutEncoder` adds the reset-aware unroll with `episode_starts` (§3) |
| `buffers.py` (`EpisodicReplayBuffer`) | `buffers.py` (`ContextualRolloutBuffer`) | **Done**: chronological storage, rollout-start state snapshots, GAE, complete-stream minibatches (§6) |
| `policies.py` (`RecurrentSACPolicy`, `MlpPolicy`) | `policies.py` (`ContextualActorCriticPolicy`, `MlpPolicy`) | **Done**: encoder, actor and value branches, recurrent `predict()`, sequence `evaluate_actions()`, `parameter_groups()` (§5) |
| `off_policy_algorithm.py` (`RecurrentOffPolicyAlgorithm(OffPolicyAlgorithm)`) | `on_policy_algorithm.py` (`ContextualOnPolicyAlgorithm(OnPolicyAlgorithm)`) | **Done**: recurrent collection with memory per env, executed actions, terminal observations, lookahead values and optional `b_t` labels (§7); memory resets in `_setup_learn`; `_excluded_save_params` |
| `recurrent_sac.py` (`RecurrentSAC(RecurrentOffPolicyAlgorithm)`) | `contextual_ppo.py` (`ContextualPPO(ContextualOnPolicyAlgorithm)`) | **Done**: PPO hyperparameters and `train()` adapted from `ppo/ppo.py`: recurrent replay, routed joint loss, clipping, KL stopping, logging (§8–§9) |

`RecurrentSAC` does not subclass `SAC`; it subclasses its own off-policy base and adapts SAC's constructor and `train()`. Do the same here. `ContextualPPO` subclasses `ContextualOnPolicyAlgorithm` and adapts PPO's constructor, `_setup_model()` and `train()` from `ppo/ppo.py`, rather than inheriting from `PPO`. `ContextualActorCriticPolicy` subclasses `ActorCriticPolicy` to reuse its distributions and heads (§5). Keep PPO's objective and public behavior; do not redesign the optimization algorithm as part of adding recurrence.

Carry over these RecurrentSAC conventions unchanged:

- **Policy kwargs:** `context_dim`, `xlstm_config` and `context_net_arch`, with the same meaning. The policy converts an OmegaConf `xlstm_config` into a plain dict and returns all three from `_get_constructor_parameters()`, so save/load restores the encoder.
- **Algorithm arguments:** `encoder_actor_loss_weight` and `encoder_max_grad_norm`, with the same meaning (§8).
- **Memory lifecycle:** environments start new episodes whenever the memory is missing. That covers the first `learn()`, `reset_num_timesteps=True`, `set_env()` or loading, and a callback that stopped collection mid-rollout (§7). The live memory is never saved; exclude it in `_excluded_save_params()`.
- **`predict()`:** returns `(action, ContextualPolicyState)`. The caller passes the state back and sets `episode_start` where episodes begin.
- **Logging keys:** `train/encoder_grad_norm`, `train/encoder_grad_norm_value_loss` and `train/encoder_grad_norm_actor_loss`, the counterparts of RecurrentSAC's `..._critic_loss` / `..._actor_loss` keys.

The integration must cover setup, collection, action evaluation, training, prediction, and serialization. Swapping only the feature extractor or rollout-buffer class is insufficient.

Retain compatible learning-rate/clipping schedules, callbacks, logging, environment wrapping, seeding, `learn()`, `predict()`, `save()`, and `load()`. Reject unsupported configurations explicitly in the first version: Dict/image observations, discrete actions, gSDE, and VecNormalize. The last is a scoped initial limitation, not a claim that recurrent PPO cannot support normalization wrappers.

Expose `n_minibatches` as the authoritative minibatch setting for the stream-based design below. Derive the number of transitions per minibatch from the actual environment count and `n_steps`; do not introduce an independent, potentially conflicting `batch_size` setting. Adapt PPO's setup and validation deliberately, including the environment-less loading path. PPO's `batch_size > 1` assertion does not apply, and `OnPolicyAlgorithm._setup_model()` calls `rollout_buffer_class(n_steps, observation_space, action_space, ...)`. That does not match `ContextualRolloutBuffer(n_steps, n_envs, observation_space, action_space, ...)`, which samples with `get(n_minibatches)`, so override `_setup_model()` as RecurrentSAC does. Pass the algorithm seed to the buffer so the stream permutation is reproducible.

## 5. Policy interface and action conventions

Provide these operations, with explicit state arguments rather than hidden mutable policy memory:

- `initial_state(n_envs) -> ContextualPolicyState` and `reset_state(state, episode_start) -> ContextualPolicyState`: memory at episode start, and a reset of the rows where an episode starts (as in RecurrentSAC).
- `forward(obs, state, deterministic=False) -> (actions, values, log_prob, latents, state_after)`: encode `[o_t, state.prev_actions]` from `state.encoder_state`, then sample `a_t`. It returns `values` `[N, 1]`, `log_prob` `[N]` of `actions`, the contexts `z_t` as `latents` `[N, D_z]` (for diagnostics such as the `b_t` probe, §12), and `state_after`, the memory after `o_t`. The collector replaces `state_after.prev_actions` with the executed action.
- `evaluate_actions(obs, actions, initial_state, prev_actions, episode_starts) -> (values, log_prob, entropy, final_encoder_state)`: sequence evaluation of recorded actions over `[E, T]` streams, re-encoding the history with the current parameters through `encoder.unroll(..., episode_starts=...)`. The outputs are `[E, T]`. `initial_state` is the encoder state (`ContextualRolloutSamples.initial_state`, or the detached state after the previous TBPTT segment), not the whole `ContextualPolicyState`.
- `parameter_groups() -> {"encoder", "actor", "value"}`: the parameter partition for the gradient routing of §8. A shared observation features extractor with parameters counts as encoder-side; separate extractors join their branch. The policy checks at construction that the groups hold every parameter exactly once.
- `predict_values(obs, state) -> values` `[N, 1]`: value-only evaluation of `obs` encoded from the memory `state`.

None of these may modify the state passed in. The collector (`ContextualOnPolicyAlgorithm`) relies on exactly this interface. `test_on_policy_algorithm.py` exercises it with a stub policy built on the real encoder, and `test_policies.py` with `ContextualActorCriticPolicy` itself, including collection, `evaluate_policy` and save/load. In `forward()`, `state_after.prev_actions` is already the executed (clipped) action.
- `predict(observation, state=None, episode_start=None, deterministic=False)` returning environment actions and the state needed at the next call.

Start with vanilla PPO's unsquashed diagonal Gaussian policy for continuous actions. Store the original Gaussian sample and its log probability. Clip the sample to the Box bounds before calling the environment. Feed that **executed, clipped action** back as the next encoder input.

Use this same action convention in replayed training and inference. Do not copy SAC's normalized `[-1,1]` storage convention blindly when environment bounds differ. BatteryPlane happens to have `[-1,1]` action bounds. A future action-scaling layer must be explicit and consistent everywhere.

`predict()` must put the clipped action it returns into the carried previous-action field. If an external caller changes that action before execution, it must also update the state accordingly. Deterministic evaluation still needs to carry memory; resetting it at every call would evaluate a different agent.

Use deterministic feature computation during collection and replay: disable dropout and avoid changing normalization statistics in the policy/encoder path.

### Construction and orthogonal initialization

`ActorCriticPolicy._build()` has two traps for a policy with an extra encoder:

1. With `ortho_init=True` (the PPO baseline's setting), it re-initializes every `nn.Linear` inside the modules listed in its `module_gains`: `features_extractor` (or the pi/vf extractors), `mlp_extractor`, `action_net` and `value_net`. It uses orthogonal weights and zero biases. An encoder placed in any of these would lose the xlstm library's initialization, including its input and context projections.
2. It creates `self.optimizer` over `self.parameters()` as its last step. An encoder registered after `_build()` returns would never be updated.

Therefore:

- Keep the encoder as its own attribute, `self.encoder` (an `XLSTMContextEncoder`). Never pass it as `features_extractor_class`, and never put it inside `mlp_extractor`. The observation feature extractor stays `FlattenExtractor`, which has no parameters.
- Override `_build()`: create `self.encoder` first, then call the parent `_build()`. The encoder is then registered before the optimizer is created, and the orthogonal initialization leaves it untouched. The xLSTM blocks keep xlstm's initialization and the projections keep PyTorch defaults, as in RecurrentSAC.
- Override `_build_mlp_extractor()` so `MlpExtractor` receives `features_dim + context_dim` inputs, because both branches consume `[o_t, z_t]`. Orthogonal initialization of the MLP branches, `action_net` (gain 0.01) and `value_net` (gain 1) stays exactly as in vanilla PPO, so the heads match the PPO baseline.
- Set `context_dim`, `xlstm_config` and `context_net_arch` before calling `ActorCriticPolicy.__init__()`, because the parent constructor calls `_build()`.

Required checks: two policies built from the same torch seed, with `ortho_init=True` and `ortho_init=False`, have identical encoder parameters. Every parameter belongs to exactly one group (encoder, actor branch, value branch) and appears in the optimizer exactly once.

## 6. Rollout buffer design

Let `T = n_steps` and `N = n_envs`. Store transition arrays in `(T,N,...)` order. Return minibatches in `(E,T,...)` order for `E` selected environments.

The initial implementation samples **whole environment streams**. Every minibatch starts at that rollout's first timestep for its selected environments, so the buffer needs only one complete initial encoder-state snapshot per environment. It does not need per-transition states or sequence padding.

A stream can contain several episodes. Keep episode-start/end flags and derived episode IDs so those episodes remain distinct. Reset both memory and gradient flow at episode starts; never blend history across a reset.

| Stored field | Shape | Meaning |
|---|---|---|
| `observations`, `next_observations` | `(T,N,D_o)` each | Policy observation and real successor; use the final observation at an episode end |
| `prev_actions` | `(T,N,D_a)` | Previous executed action; zero at an episode start |
| `actions` | `(T,N,D_a)` | Gaussian sample used for PPO log-probability evaluation; may lie outside the bounds |
| `executed_actions` | `(T,N,D_a)` | Clipped action sent to the environment; `add()` rejects values outside the action-space bounds |
| `rewards` | `(T,N)` | Original environment rewards |
| `timeout_bootstrap` | `(T,N)` | Discounted final-state value for truncations; zero otherwise |
| `intrinsic_rewards` | `(T,N)` | Optional weighted bonus hook; stays zero in this project (§12) |
| `episode_starts` | `(T,N)` bool | Reset before processing this observation |
| `terminated`, `truncated` | `(T,N)` bool each | Effective end flags of this transition |
| `values`, `log_probs` | `(T,N)` each | Frozen collection outputs |
| `advantages`, `returns` | `(T,N)` each | Frozen PPO training targets |
| `episode_ids` | `(T,N)` int64 | Derived episode-piece IDs within each environment stream |
| `initial_encoder_state` | Nested state, batched over `N` | Detached snapshot before the first rollout input |
| `contexts`, `next_contexts` | `(T,N,1)` each, optional (`context_dim=1`) | Privileged `b_t` when `o_t` is observed and `b_{t+1}` after the transition; probe targets only, never a network input (§7, §12) |
| `context_estimates` | `(T,N,D_z)`, optional (`estimate_dim`) | Collection-time `z_t`, diagnostics only |

This buffer is implemented as `ContextualRolloutBuffer` in `buffers.py`, with `ContextualRolloutSamples` in `type_aliases.py`. `test_buffer.py` checks the timing alignment, episodes crossing rollout boundaries, state snapshot independence, GAE (by hand and against SB3's `RolloutBuffer`), truncation bootstraps, stream coverage, `N=1`, lifecycle ordering, and Hydra instantiation.

Store independent CPU copies; restore sampled tensors on the selected device. Preserve state structure, dtype, and batch axes. Never normalize recurrent state leaves. Optional cached collection latents can help diagnostics, but must not replace differentiable encoder replay during training.

Required lifecycle: `reset()` -> `set_initial_state()` -> `T` calls to `add()` -> `finalize()` -> GAE (`compute_returns_and_advantage()`) -> repeated `get(n_minibatches=...)` calls. The optional `add_intrinsic_rewards()` hook between `finalize()` and GAE is unused here. Reject invalid ordering, malformed shapes, non-finite populated fields, and sampling of incomplete data.

Each epoch permutes environment indices and visits each stream once. Require `N % n_minibatches == 0` and `1 <= n_minibatches <= N`. With one environment, only one whole-stream minibatch is possible. Time order within a stream never changes. These restrictions are a deliberate simplicity trade-off.

PPO can learn from incomplete episodes inside a completed rollout. Do not wait for an episode to finish before making its rollout transitions available, and do not maintain an SAC-style replay history across updates.

## 7. Collection and memory lifecycle

Use an SB3 VecEnv interface. Its auto-reset conventions must be resolved by the collector, not guessed by the buffer. For BatteryPlane, that VecEnv is a `DummyVecEnv` of `Monitor(bp.BatteryPlaneEnv(...))` built from the local `battery_plane.py` (§1.1).

Carry the current observation, episode-start flags, previous executed actions, and complete encoder state across steps and rollout boundaries. Keep parameters fixed while collecting a rollout.

For each step:

1. Reset state and previous-action rows whose observations start a new episode.
2. Under `no_grad()`, process `[o_t,u_{t-1}]` and obtain `z_t`, the new encoder state, sampled action, old value, and old log probability.
3. Clip the action, step the environment, and preserve the raw reward.
4. If done, use `info['terminal_observation']` as the transition's true successor. The live returned observation belongs to the next episode. Missing required final observations must cause a clear error.
5. Resolve termination/truncation. Under SB3's convention, a true termination takes precedence over a simultaneous time limit.
6. Calculate any truncation bootstrap on the ending episode's final observation, then store the transition.
7. Carry the live observation, executed action, state, and reset flags into the next step.

**Diagnostic true context (optional, off by default).** An algorithm argument `context_info_keys: tuple[str, str] | None` names the reset-info key and the step-info key of the true context; for BatteryPlane it is `("b0", "b")`. When it is set, the buffer is built with `context_dim=1`, and the collector stores:

- `contexts[t]`: the charge when `o_t` is observed. For a row that starts an episode, this is `vec_env.reset_infos[i]["b0"]`. Otherwise it is the previous step's `infos[i]["b"]`, carried across rollout boundaries like the observation.
- `next_contexts[t]`: this step's `infos[i]["b"]`, i.e. `b_{t+1}`. At an episode end, `infos[i]` still belongs to the ending episode.

`infos[i]["b"]` returned by step `t` is `b_{t+1}`, not `b_t`; the probe needs `b_t`. These values never reach the encoder, actor or critic. They exist only for the probe in §12.

For a value bootstrap at `o_{t+1}`, start from the encoder state after processing `o_t`, and use executed `a_t` as the previous action. At a truncation, do not reset that temporary state: the final observation belongs to the ending episode.

Use an independent state copy for every value-only lookahead and discard its returned state. The next live observation must not be consumed twice. Although the current encoder clones its state input, keep this caller contract explicit and verify it for any future backend; `no_grad()` alone does not prevent in-place mutation.

Initialize fresh environment/memory state after model loading, environment replacement, or a fresh `learn(reset_num_timesteps=True)`. Continue memory when a subsequent `learn(reset_num_timesteps=False)` continues the same live environments. If a callback interrupts collection before the buffer is complete, skip its update and reset environment and memory together before restarting collection.

**Episode-aligned rollouts (`reset_envs_each_rollout`, off by default).** By default a rollout boundary does not reset anything: episodes continue into the next rollout, from the stored memory, but gradients stop at the boundary. With `reset_envs_each_rollout=True`, the collector instead resets all environments at the start of every rollout. Episodes still running at the end of a rollout are cut there and bootstrapped from the value of the last observation, as at any rollout boundary. With `n_steps` at least the episode length (BatteryPlane: 700), every episode that starts with a rollout lies complete in its stream. Its whole history is then replayed with gradients (with `bptt_len: null`), and the stored initial state is the exact reset state, so there is no stale snapshot. The costs:

- Episodes that start after a crash inside a rollout are cut at its end.
- Cut episodes are missing from the Monitor episode statistics.

The collector raises if an episode that started with the rollout is still running at its end, i.e. if `n_steps` is shorter than the episodes. Without this option, `n_steps=700` alone does not keep episodes together: the first crash shifts that environment's episodes against the rollout boundaries, permanently.

## 8. Returns, losses, and gradient ownership

Keep original rewards separate from timeout corrections. Set `timeout_bootstrap[t] = gamma * V(final_obs, final_z)` for truncations and zero otherwise. Let `d_t = terminated_t OR truncated_t` and `r'_t = reward_t + timeout_bootstrap_t`.

Use the ordinary reverse GAE recursion with zero continuation advantage after the rollout:

\[
\delta_t=r'_t+\gamma(1-d_t)V_{t+1}^{old}-V_t^{old},\qquad
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1},\qquad
\hat R_t=\hat A_t+V_t^{old}.
\]

At a continuing rollout boundary, use a value-only lookahead for `V_{t+1}^{old}`. End-of-episode masks stop GAE from entering the auto-reset episode. A truncation already bootstraps through its correction; do not add that value twice. True terminations do not bootstrap.

Compute all old values, old log probabilities, advantages, and returns before any update. Keep them unchanged throughout the PPO epochs.

Recompute current contexts sequentially with current parameters. For the recorded sampled actions, define:

\[
\rho_t=\exp\left[\log\pi_{\theta}(a_t\mid o_t,z_t)-\log\pi^{old}_t\right],
\]
\[
L_{policy}=-\mathbb E\left[\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right],
\]
\[
L_{value}=\mathbb E[(V_{\psi}(o_t,z_t)-\hat R_t)^2],\qquad
L_{entropy}=-\mathbb E[\mathcal H(\pi_{\theta}(\cdot\mid o_t,z_t))],
\]
\[
L=L_{policy}+c_vL_{value}+c_eL_{entropy}.
\]

Retain the selected SB3 version's optional value clipping and advantage normalization. Normalize advantages consistently over the intended minibatch and guard the single-element case. Preserve approximate-KL early stopping, clip-fraction reporting, and gradient clipping.

**Do not detach `z_t` before either MLP.** The encoder receives gradients from both the policy side (policy loss, plus entropy when enabled) and the value loss, weighted as described below. Advantages, returns, recorded log probabilities, and initial state snapshots remain detached.

### Gradient routing and `encoder_actor_loss_weight` (as in RecurrentSAC)

If `L` is simply backpropagated, the value loss dominates the shared encoder. Rewards are not normalized (VecNormalize is unsupported, and the PPO baseline does not use it). BatteryPlane's discounted returns are of order 100, and a crash costs −300. So the gradient that `c_v L_value` sends into `z_t` is far larger than the gradient of the clipped policy loss on normalized advantages, and the encoder would effectively learn to predict returns.

Route the gradients per parameter group as `RecurrentSAC.train()` does, computing everything from the same parameters before any optimizer step. Split the loss into an actor side and a critic side:

\[
L_{actor}=L_{policy}+c_eL_{entropy},\qquad L_{critic}=c_vL_{value}.
\]

| Parameter group | Gradient |
|---|---|
| Actor branch (policy MLP branch, `action_net`, `log_std`) | \(\nabla L_{actor}\) |
| Value branch (value MLP branch, `value_net`) | \(\nabla L_{critic}\) |
| Encoder (input projection, xLSTM blocks, context projection) | \(\nabla\,[L_{critic}+w_a L_{actor}]\), with \(w_a\) = `encoder_actor_loss_weight` |

`encoder_actor_loss_weight` has the same name and meaning as in RecurrentSAC: the critic-side loss enters the encoder with weight one and the actor-side loss with weight `w_a`. With `w_a = 1`, every gradient equals that of PPO's summed loss `L`. Raise `w_a` to counteract the value-loss dominance; `w_a = 0` trains the encoder from the value loss only. Choose `w_a` from the logged per-loss encoder gradient norms (below), so that the weighted actor part is not negligible next to the value part.

Implement it with two `th.autograd.grad` calls, as RecurrentSAC does: the critic loss over value-branch and encoder parameters with `retain_graph=True`, then the actor loss over actor-branch and encoder parameters. Combine the two encoder parts with `w_a` (RecurrentSAC's `_add_grads`), and write the results into `.grad` (`_set_grads`). Then step a single Adam optimizer, vanilla PPO's `policy.optimizer`, which holds every parameter exactly once. Adam keeps per-parameter state, so one optimizer behaves like RecurrentSAC's separate optimizers.

**Gradient clipping.** `max_grad_norm` clips the actor and value branches together, exactly as vanilla PPO clips its networks. The encoder is excluded from that norm and clipped on its own with `encoder_max_grad_norm` (`None` = no clipping, as in RecurrentSAC). Otherwise a large encoder gradient, for example with a large `w_a`, would shrink the heads' updates through the shared global norm.

**Logging.** Log, as RecurrentSAC does, `train/encoder_grad_norm` (combined, before clipping), `train/encoder_grad_norm_value_loss` (from `L_critic`) and `train/encoder_grad_norm_actor_loss` (from `L_actor`, before weighting by `w_a`).

KL early stopping is unchanged: when it triggers, the minibatch's gradients are discarded and no step is taken.

### Supervised encoder objective (`encoder_objective="supervised"`, implemented)

The project's goal is that the battery is inferable from the history and that agents acting on the inferred battery come close to battery-visible agents. Observed mode (p04) is the fair upper bound, since hidden mode keeps the noisy speed reading. JCPL-style RL-only training is not required. With `encoder_objective="supervised"`, the encoder becomes an estimator of the charge, `z_t = b̂_t` (one sigmoid number, `context_dim=1`), and the labels are used for training only:

- **PPO epochs:** they replay the rollout with the encoder frozen (`evaluate_actions(..., detach_context=True)` runs it without gradient), so they update the actor and critic only.
- **Encoder steps:** after the PPO epochs, `supervised_gradient_steps` steps train the encoder on complete episodes from a separate `SupervisedContextualEpisodeBuffer` (`episode_buffers.py`). `ContextualRolloutBuffer` stays PPO's buffer.

The episode buffer:

- keeps one unfinished episode per environment across rollout boundaries;
- stores `(o_t, a_{t-1}, b_t)` with the label of the same observation, and derives the previous action itself from the executed (clipped) actions, so it is zero at an episode start;
- closes an episode only at termination or truncation, after appending the final observation and its label, so an automatic reset never mixes two episodes;
- drops unfinished episodes when the environments are reset without ending them;
- assigns every completed episode to the training or validation split (seeded, `validation_fraction`) and evicts whole, oldest episodes beyond `capacity` timesteps;
- stores no PPO quantities and no encoder states. Each encoder step samples `episode_batch_size` complete training episodes, runs the encoder from its initial state at every episode's first step (labels are never an input), and minimizes the masked mean squared error against `b_t`. One validation batch measures the fit.

The project config sets capacity, batch size, gradient steps, validation fraction and seed under `cppo.supervised`.

Logged: `train/context_mse` and `train/context_r2` (training batches), `train/context_val_mse` and `train/context_val_r2` (validation episodes), `train/encoder_grad_norm_context_loss`, and the buffer's episode counts. Project experiment: `c02_hidden_supervised_battery`. On the cue-memory test task the supervised encoder reached validation R² > 0.95 and the agent scored perfectly, like an agent that sees the cue. Measured on BatteryPlane (MacBook CPU): one encoder step on 16 complete episodes takes about 2.2 s, and one PPO epoch with the encoder frozen about 1.1 s (7.0 s with encoder gradients). That is about 30 s per update with the defaults.

### Gaussian context head (`context_head="gaussian"`, implemented)

A policy option for the supervised objective. The encoder estimates the context as a diagonal Gaussian instead of a point: the head's last linear layer has `2 * context_dim` outputs, and the parameter-free `GaussianContextOutput` turns them into the mean `mu_t` (through `context_activation`, so a sigmoid mean for BatteryPlane) and the variance `sigma2_t = softplus(.) + 1e-6`. The encoder outputs `z_t = [mu_t, sigma2_t]` (`latent_dim = 2 * context_dim`). The actor and critic read `[o_t, mu_t, log sigma2_t]` by default (policy `context_variance_input="log_variance"`; `"variance"` feeds `sigma2_t` itself, typically 1e-4 to 0.02 next to observations of order 1; `"none"` feeds the mean only, with the variance still trained). The encoder steps minimize the masked Gaussian negative log-likelihood `0.5 * (log sigma2_t + (b_t - mu_t)^2 / sigma2_t)` instead of the squared error. The plain likelihood divides each error by `sigma2_t`, so the mean learns little where the estimate is uncertain; `nll_beta` (beta-NLL, Seitzer et al. 2022) multiplies each step's term by `sigma2_t ** nll_beta` without gradient through the weight (0 = plain NLL, the algorithm's default; the project config uses 0.5). Everything else is the supervised objective above. The RL objective is rejected with this head, because nothing would give the variance its meaning.

Logged besides the squared error and R² of the mean: `context_nll` (unweighted, whatever `nll_beta` is), `context_variance` (mean predicted variance) and `context_calibration` (mean of `(b_t - mu_t)^2 / sigma2_t`, 1 when calibrated, above 1 when overconfident), each also on the validation batch (`context_val_*`). Project experiment: `c03_hidden_gaussian_battery` (`cppo.context_head` in the project config). Policies saved before this option load with the point head.

### Auxiliary head (`auxiliary_dim` + `auxiliary_info_keys`, implemented)

A second supervised head for the supervised objective. `XLSTMRolloutEncoder(auxiliary_dim=...)` adds `auxiliary_projection` (hidden layers `auxiliary_net_arch`, linear output) on the same xLSTM embedding as the context head. `step` and `unroll` never run it, so `z_t`, the actor and the critic are unchanged; only `unroll_with_auxiliary` (built on `embed_step`, the base `step` without the context head) computes it during the encoder steps. Its targets come from the env infos like the true context (`auxiliary_info_keys=(reset_key, step_key)`), and each episode-buffer label row is `[context, auxiliary targets]`. The encoder steps minimize `L_context + auxiliary_loss_weight * L_auxiliary` (`lambda` in [0, 1]), `L_auxiliary` the masked MSE on the same complete episodes. Logged: `auxiliary_mse`, `auxiliary_r2` (and `auxiliary_val_*`), `context_loss` and `encoder_loss`.

BatteryPlane: the head predicts the true speed ratio `vx_t / v_ref`, which hidden mode observes with noise (`(5 / 60)^2 = 0.0069` error variance). The project's `TrueSpeedInfo` wrapper adds it to the reset and step infos under `vx_ratio` (the env's reset info has no speed; `battery_plane.py` is unchanged). Project config: `cppo.velocity_head` (`enabled`, `weight`, `net_arch`); experiment `c04_hidden_gaussian_velocity_battery` (c03 + the head, weight 1.0).

Beware copying SB3-Contrib's shared-LSTM branch: that implementation detaches actor recurrent features before its critic branch. That gradient rule does not satisfy this project's requirement that the value loss train the shared encoder.

## 9. Recurrent replay, TBPTT, and consistency

For each minibatch, restore its saved rollout-start state and process observations in time order. Reset at each episode start: `encoder.unroll(sample.initial_state, observations, sample.prev_actions, episode_starts=sample.episode_starts)`. Both MLP branches consume the same freshly computed `z_t`.

Keep two lengths distinct:

- `n_steps`: collected transitions per environment per PPO rollout.
- `bptt_len`: maximum number of steps connected through a training graph; `None` means the full rollout stream, subject to episode resets.

For bounded-memory TBPTT, process consecutive segments, detach carried state at segment boundaries, and accumulate appropriately weighted gradients for each routed parameter group (§8). Retain its numerical memory; do not zero it. Keep parameters fixed until all segments in that minibatch are processed, then clip and step once. Weight short segments by their transition count, and compute advantage normalization and KL statistics consistently for the complete minibatch. If KL stopping rejects the update, discard accumulated gradients.

Do not concatenate all segment graphs and assume detaching alone bounds activation memory. Backward/free each segment before continuing when memory reduction is the purpose.

Inference memory can extend across an entire episode, while training gradients stop at rollout/TBPTT boundaries. This does not guarantee that the encoder learns every long-range dependency.

The saved state is the exact collection snapshot, but after parameter updates it is an approximation to a full history re-encoding. Use that fixed snapshot during replay; do not silently refresh it from previous rollouts. **Before the first parameter update, replay must reproduce the recorded action log probabilities and values, giving PPO ratios approximately equal to one.**

For the supplied xLSTM backend, different batching of newly reset and continuing rows can change internal stabilizer scaling. Test equivalence of the resulting context, policy distribution, and value within tolerance; do not require unrelated internal rescalings to be identical.

## 10. Configuration

**Project configuration (Stage 6, done).** In `Env_BatteryPlane/ppo/`, `configs/battery_plane_contextual_ppo.yaml` composes `battery_plane_ppo.yaml`, exactly as `configs/rsac_battery_plane.yaml` composes the SAC config. Environment, budget, `n_envs`, evaluation grid, checkpoints and paths are the PPO baseline's, and its `cppo:` section takes every shared PPO setting from `${ppo.*}`. So p01–p04 run unchanged with `main_battery_plane_contextual_ppo.py`, and `cppo.*` overrides change this learner only. The experiment `configs/experiment/c01_hidden_contextual.yaml` is the main run: the hidden task of p02, with its defining learner settings written out. `run_experiments_ppo.py` sends every experiment with a `cppo:` section to the contextual script, as `run_experiments.py` sends `policies:` configs to `builtin_policies.py`. The section names and values differ from the bare block below (`cppo.xlstm` instead of `policy_kwargs.xlstm_config`, `n_epochs` inherited as 10, 256×256 heads); the entry point's `build_model()` maps them to the constructor. Measured on the MacBook CPU with the project's encoder (mLSTM block, then sLSTM block, width 64): one 4096-step rollout 0.7 s, one update epoch 7.0 s, about 9.6 h for 2M steps with `n_epochs: 10` (3.9 h with 4). The sLSTM-only encoder took 3.1 s per epoch, about 4.3 h.

The block below records the bare constructor arguments, e.g. for `instantiate(cfg.algorithm, env=env)` outside the project.

Keep Hydra in the training entry point. Algorithm, policy, encoder, and buffer constructors accept normal Python arguments. Convert nested OmegaConf containers to ordinary dictionaries before passing strict xLSTM configuration to the existing builder.

The following is a starting configuration, not a tuned result or speed guarantee. The training entry point builds the VecEnv from the local `battery_plane.py` (§1.1; no gym id is used), then calls `instantiate(cfg.algorithm, env=env)`.

```yaml
seed: 0
device: cpu

algorithm:
  _target_: stable_baselines3.contextual_ppo.ContextualPPO
  _convert_: all
  policy: MlpPolicy
  seed: ${seed}
  device: ${device}
  learning_rate: 0.0003
  n_steps: 512
  n_minibatches: 4
  n_epochs: 4
  # null: backpropagate through the whole stream (episode resets still cut it)
  bptt_len: null
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  clip_range_vf: null
  normalize_advantage: true
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: 0.03
  use_sde: false
  # Encoder objective (§8): L_critic + encoder_actor_loss_weight * L_actor; 1.0 = PPO's summed loss
  encoder_actor_loss_weight: 1.0
  # Encoder clipped separately from max_grad_norm (§8); null = no clipping, as in RecurrentSAC
  encoder_max_grad_norm: 0.5
  # Diagnostic b_t storage for the probe (§7); BatteryPlane: [b0, b]. null = not stored
  context_info_keys: null
  # Episode-aligned alternative (§7): n_steps: 700 with reset_envs_each_rollout: true
  reset_envs_each_rollout: false
  policy_kwargs:
    context_dim: 8
    context_net_arch: []
    net_arch:
      pi: [64, 64]
      vf: [64, 64]
    xlstm_config:
      embedding_dim: 64
      num_blocks: 1
      slstm_at: all
      slstm_block:
        slstm:
          backend: vanilla
          num_heads: 4
          conv1d_kernel_size: 0
        feedforward: null
```

Use `MlpPolicy` as an alias for this package's contextual policy, not SB3's ordinary feedforward policy. The algorithm supplies buffer spaces, device, discount settings, and actual `env.num_envs` at runtime. Validate that configured and actual environment counts agree. Avoid duplicated values in `rollout_buffer_kwargs` that conflict with algorithm-owned settings.

With 16 environments, each minibatch has 4 streams and 2,048 transitions. The sequential encoder steps per training run, n_epochs × n_minibatches × total steps / n_envs, do not depend on `n_steps`, so a longer stream costs memory per minibatch but no extra update time. `n_envs` does not imply parallel CPU execution when using DummyVecEnv. Begin on CPU with the configured vanilla sLSTM backend for the Mac setup; expose device/backend choices and measure performance before changing them.

Save the resolved configuration and actual SB3/xLSTM dependency versions. Preserve schedules and policy kwargs in checkpoints. Make paths robust to Hydra changing the current working directory. Do not require Hydra to use the algorithm directly from Python.

## 11. Implementation order and acceptance criteria

| Stage | Deliverable | Required verification |
|---|---|---|
| 1. Interfaces and encoder adapter | **Done:** `type_aliases.py`, `torch_layers.py`, `test_torch_layers.py`. Package, state/sample types, reset-aware unroll | Existing recurrent SAC imports remain valid; step/unroll outputs agree on a controlled sequence |
| 2. Rollout buffer | **Done:** `buffers.py`, `type_aliases.py`, `test_buffer.py` | Alignment, snapshot independence, episode boundaries, timeout targets, exact minibatch coverage, executed actions inside the bounds |
| 3. Contextual policy | **Done:** `policies.py`, `test_policies.py`. Actor/value branches using the shared latent; recurrent prediction | Policy loss and value loss each independently produce encoder gradients on nondegenerate fixtures; all parameters belong to the optimizer once; encoder parameters are identical with `ortho_init=True` and `False` (§5) |
| 4. Recurrent collector | **Done:** `on_policy_algorithm.py`, `test_on_policy_algorithm.py`. Correct action history, resets, terminal observations, lookahead values, optional `b_t` labels, episode-aligned rollouts | Clipped actions feed memory; final observations are preserved; lookahead does not mutate live state; `contexts[t]` is `b_t`, not `b_{t+1}` (§7) |
| 5. PPO learner | **Done:** `contextual_ppo.py`, `test_contextual_ppo.py`. Routed joint loss, recurrent replay, TBPTT, clipping and KL stopping | Ratios are approximately one before updates; old targets stay fixed; gradients stop only at intended boundaries; with `encoder_actor_loss_weight=1` the routed gradients equal those of the summed loss; the actor loss reaches no value-branch parameter and vice versa; the encoder is clipped separately from `max_grad_norm` |
| 6. Integration | **Done:** `Env_BatteryPlane/ppo/` `main_battery_plane_contextual_ppo.py`, `configs/battery_plane_contextual_ppo.yaml`, `configs/experiment/c01_hidden_contextual.yaml`, `tests/test_contextual_ppo_entry.py`. Hydra entry point, callbacks, save/load, inference | Configuration overrides work; a short smoke run has finite losses; loaded predictions match using an identical supplied history/state |

Use small deterministic fixtures for correctness checks. Include an episode crossing a rollout boundary, several episodes within a stream, `N=1`, and a forced time-limit truncation. Test full and shortened TBPTT, with correct weighting of a final short segment. Ordinary model loading starts new episodes; restoring an exact live simulation state is outside this checkpoint contract.

For standard evaluation helpers and callbacks, verify that `predict()` receives and returns memory and honors episode-start flags. A helper that discards recurrent state must be adapted before using its results.

Log the usual PPO diagnostics plus the three encoder gradient norms of §8, effective minibatch transition count, rollout/optimization duration, and the configured gradient length. Explained variance is not always defined when return variance is zero; this alone is not proof of a fatal error.

Deliver implementation, configuration, focused tests, and a short README explaining recurrent prediction. Do not start long training sweeps or change BatteryPlane dynamics as part of implementing this plan.

## 12. Later experiments and research claims

The first working agent is RL-only. Later comparisons can add prediction-only and hybrid encoder objectives. When implementing the first dynamics learner, retain the agreed one-step and action-conditioned ten-step targets (`k=1` and `k=10`), with future executed actions confined to the auxiliary decoder. These auxiliary losses do not belong in the current RL-only baseline.

Keep the intrinsic bonus disabled. If studied later, compare predictions of the same successor battery before and after its observation and describe the score as prediction-error reduction; it is not automatically information gain.

Useful subsequent evaluations include feedforward PPO versus Contextual_PPO under hidden observations, an observed-context PPO reference, and multiple training seeds. Report both return versus environment interactions and return versus wall-clock time. Better battery-probe accuracy alone does not prove better control or zero-shot generalization.

**Context probe (target `b_t`).** Measure what `z_t` encodes by regressing the current charge `b_t` (the battery when `o_t` is observed) from `z_t`. Do not use the starting charge `b0` as the target: the context is autogenic and drifts within every episode.

- **Data:** episodes of a frozen policy, with `z_t` from the one-step forward pass (§5) and `b_t` read from `info` with the alignment of §7. During training, the optional `context_info_keys` labels can serve the same purpose.
- **Split:** by episode, not by transition, because neighbouring steps are nearly identical.
- **Report:** MSE and R² for a linear probe and a small nonlinear one (JCPL used a random forest with 5-fold cross-validation).

The probe is a diagnostic only: it never trains the encoder.

## 13. Source grounding

This plan is based on the supplied recurrent SAC modules (`buffers.py`, `off_policy_algorithm.py`, `policies.py`, `recurrent_sac.py`, `torch_layers.py`, and `type_aliases.py`) and the current project requirements. The supplied JCPL paper motivates joint context/control learning; the recurrent PPO adaptation here is a project design choice.

Primary implementation references inspected on 2026-10-02:

- [SB3 PPO source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html): PPO training and configuration reference.
- [SB3 OnPolicyAlgorithm source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/on_policy_algorithm.html): collection and environment integration reference.
- [SB3-Contrib recurrent policy source](https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/common/recurrent/policies.html): recurrent interface reference, including the shared-feature detachment behavior to avoid for this design.

Implement against the project's installed/pinned versions and check their actual signatures. This specification does not establish performance or substitute for the targeted integration checks above.
