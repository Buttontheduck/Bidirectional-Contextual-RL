# Autogenic context in RL: the BatteryPlane benchmark and Contextual PPO

Most work on contextual reinforcement learning studies contexts that the environment imposes on the agent, such as gravity, mass or friction. The agent can't change them, and they stay fixed during an episode. This project studies the other kind: a context that the agent's own behaviour creates and changes. Our example is the plane's battery charge. The battery limits what the plane can do, and every action the plane takes drains it.

The project has two sides:

1. **BatteryPlane** (`battery_plane.py`), an environment built so that the battery matters. An agent that can't see the battery and has no memory should do poorly, and the same agent should do much better when it sees the battery.
2. **Contextual PPO** (`contextual_ppo/`), an agent with memory. A recurrent encoder infers the battery from the history of observations and actions, and the actor and critic both use that estimate. If this agent recovers most of the gap between "can't see the battery" and "sees the battery", the environment is a valid test of autogenic context, and the agent is a working baseline for it.

**Project page:** `docs/`, five pages (overview, environment, Contextual PPO, results, ablations) for GitHub Pages, served from this branch's `/docs` folder. **Trained models:** the release [`models-v1`](https://github.com/Buttontheduck/Bidirectional-Contextual-RL/releases/tag/models-v1), with the two frozen GRU encoders, 48 PPO policies, their final-test scores and a script to fly them. **Branches:** `main` holds the hard version, the project page and the figures; this branch, `easy-env`, also holds the easy version (a forced take-off before the agent takes control) with its code and experiments, summarized in section 6 and recorded in `EASY_ENV_WARMUP.md`.

## 1. Allogenic and autogenic context

The terms come from Biedenkapp, *Contextual Intelligence: The Next Leap for Reinforcement Learning* (2026; `papers/main_paper.pdf`).

**Contextual MDPs.** A contextual MDP is a family of MDPs that share states and actions. A context variable `c` changes the transition dynamics, the rewards and the starting states. A policy can be *context-oblivious*, `π(s)`, which is the domain-randomization setting where the agent must do well on average. Or it can be *context-aware*, `π(s, c)`, where it either reads the context directly or infers it from its history.

The paper argues that the field has treated context as one static input. It proposes splitting it in two:

| | Allogenic context | Autogenic context |
|---|---|---|
| Where it comes from | Imposed by the environment, independent of the agent's actions (exogenous) | Arises from the agent's own behaviour or internal state (endogenous) |
| Can the agent change it? | No. It can only observe or infer it | Yes, and it can even control it deliberately |
| Reach | Global: how the whole environment behaves | Local: the agent's own situation |
| How it changes over time | Roughly constant within an episode; rare, sudden, large shifts | Changes often, smoothly and continuously, but slower than the raw observations |
| Examples | Gravity, limb length, payload, motor torque, sensor noise, map layout, terrain | **Battery level**, actuator wear, limb failure, fatigue; goals or difficulty the agent chooses itself |

The paper uses a robot's battery as its main autogenic example: "as the charge depletes, the robot preferentially selects low-energy maneuvers, causing the battery level to drift gradually rather than jump abruptly". It names three open problems:

1. Learning with both kinds of context at once.
2. Learning from signals that change at different speeds.
3. Bringing in abstract contexts such as roles, resource budgets and rules.

This project isolates a single autogenic context, so it sits under the first two problems and under the paper's "resource awareness" example.

**Why autogenic context is harder than allogenic context:**

- **It changes during the episode**, so an estimate made early goes out of date. The agent has to track the context, not identify it once.
- **The agent causes the change.** Every action moves the context, so the value of an action depends on what it does to the context later.
- **Observing it can cost something.** In BatteryPlane the battery only shows in the plane's behaviour while the throttle is at its maximum, which spends charge. "Observing the context moves it."
- **Behaviour and estimate depend on each other.** What the agent can learn about its context depends on how it behaves, and how it should behave depends on its context. Section 6 shows this happening in practice.

## 2. How allogenic context has been handled (related work, not used here)

The three papers in `papers/allogenic context papers/` infer an allogenic context that the agent doesn't observe. We don't use their methods, but they set the standard evaluation design that we follow:

- **Hidden context:** the agent trains across contexts without seeing them. This is the lower bound.
- **Explicit context:** the true context is appended to the observation. This is usually the upper bound.
- **Inferred context:** the method being tested.

| | Ndir, Biedenkapp & Awad 2024: behaviour-specific context (jcpl) | Grooten et al. 2026: SPARC | Röder et al. 2025: DALI |
|---|---|---|---|
| Contexts | CARL: CartPole time step, Pendulum length, MountainCar power, Ant torso mass | Gran Turismo 7: about 500 car models (20% outliers held out), engine power and mass; MuJoCo with wind | CARL DeepMind Control: gravity, string length, actuator strength |
| Learner | SAC | QR-SAC, distributed rollouts | DreamerV3 (model-based) |
| Encoder input | A set of past transitions `(s, a, s')` from the same context, averaged into one embedding | A fixed-length history of recent `(o, a)` pairs | A K-step window of `(o, a)` (transformer) |
| What trains the encoder | The actor and critic losses, backpropagated into the encoder, so the context fits the current behaviour. The baseline trains it to predict the next state | Supervised regression to the latent that an *expert* policy computes from the true context (as in RMA). Expert and adapter train together in one phase, and the adapter collects the data | Self-supervised next-observation prediction, plus alignment with the world model's latent state |
| Uses the true context in training | No | Yes (expert and critic) | No |
| Main finding | Matches the explicit-context agent in training and beats next-state prediction in 3 of 4 environments. Giving the true context directly can even hurt (Ant) | The most completed laps on unseen cars overall, and the fastest on 2 of 3 tracks. The oracle with the true context generalizes worse on 2 of 3 tracks | Up to 96% better than context-unaware DreamerV3 when extrapolating, sometimes better than the ground-truth context |

**What carries over to this project:**

- The hidden / explicit / inferred comparison. Here it is plain PPO without the battery, plain PPO with the battery, and Contextual PPO.
- Training-only privileged labels, as in SPARC and RMA. Contextual PPO's encoder regresses to the true battery `b_t`, read from the training environments' info dicts and never from an input.
- The choice between training the encoder together with the policy (Ndir et al., SPARC) or separately and then freezing it (RMA phase 2, the next-state baseline). The A–F grid does the first, and `ENCODER_PLAN.md` tests the second.

**What doesn't carry over:**

- **All three assume the context is fixed within an episode.** Averaging a set of transitions (Ndir et al.) or reading a short window (DALI) identifies a constant. DALI's theory relies on the observations decorrelating over time, so that short windows pin the context down. A battery that changes every step has to be tracked, by counting the drain from the actions and correcting the estimate whenever the plane reveals its battery.
- **They assume any reasonably exploratory behaviour reveals the context.** DALI states this explicitly. In BatteryPlane the battery is hidden unless the throttle is at its maximum, and that spends the battery, so identifying the context has a price paid in the context itself.

## 3. Goal and how validity is shown

**Goal.** Build an autogenic-context environment, and show it is valid by showing that an agent with memory can infer the context and use it.

The claims, and the agents that test them, all scored on the same validation flights:

| Claim | Agent | Sees the battery | Memory | Expected |
|---|---|---|---|---|
| The context matters | Plain PPO, hidden mode | No | No | Low |
| The context helps when known | Plain PPO, observed mode | Yes (true value) | No | High: the upper reference |
| The memory agent's training works | Contextual PPO with the true battery in place of the encoder (`step0_oracle.py`) | Yes, through the context path | Not needed | About as high as observed PPO |
| The context can be inferred and used | Contextual PPO with its learned encoder, hidden mode | No: infers it from `[o_t, a_{t-1}]` | Yes (xLSTM) | Close to observed, well above hidden |

The environment's own design checks (section 4) add a scripted version of the first two rows. A hand-written policy that knows the battery scores 445, while the best hand-written policy that can't see it scores 283. So 36% of the return depends on the context.

## 4. Side 1: the BatteryPlane environment (`battery_plane.py`)

A 2D point-mass plane with a lift motor, a cruise motor, a wing and a battery. The module docstring is the full specification; the essentials:

- **State:** altitude `y`, speeds `v_x` and `v_y`, battery `b` in [0, 1], and time left `tau`.
- **Action:** `(a_x, a_y)` in [−1, 1]². `a_x` drives the cruise motor and `a_y` climbs or descends around level flight.
- **Battery:** drains with the thrust actually delivered, `b -= (ax² + ay²)/g² · dt/E`; hovering costs 1/E per second, with E = 100. Nothing else changes it. The battery sets the cruise motor's thrust limit, `cap = 11.5 + 6.5·b`, which follows the current charge.
- **Reward:** a term that grows from 0 to 1 as the speed rises through the reward band, 52 to 62 m/s, plus an equally weighted dense term for speed toward 62 m/s. A failure (ground contact or empty battery) costs 300.
- **Episode:** 700 steps of 0.05 s, so 35 s. It ends early on a failure.
- **Starting battery:** drawn from U[0.2, 0.7]. The range is chosen so that flying flat out survives only from 0.7 up, so almost every flight must ration its energy.

**Why it is autogenic.** The battery moves only because of the agent's own thrust, and it then limits that thrust: a two-way coupling between agent and context. It changes smoothly within every flight, which matches the paper's description.

**Observation modes**, a 2 × 2 of "is the battery visible?" × "is the speed reading noisy?":

| Mode | Battery | Speed reading | Role |
|---|---|---|---|
| `hidden` | Hidden | Noisy (std 5 m/s) | **The test:** infer the battery from how the plane responds |
| `clean` | Hidden | Exact | Separates inference from filtering noise |
| `observed` | Visible | Noisy | The context-aware reference, `π(s, c)` |
| `full` | Visible | Exact | Plain MDP |

The observation is `[y/y_ref, (v_x + noise)/v_ref, v_y/v_ref, tau]`, plus `b` when it is visible.

**How a hidden-mode agent can learn the battery.** Only through the speed, and only while the cruise throttle is at its cap: below the cap the dynamics don't depend on `b` at all.
- Each unit of battery is worth about 13.5 m/s of steady speed.
- One noisy speed reading pins `b` only to ±0.37. About 14 steps at the cap pin it to ±0.1.
- Between those moments the agent must carry its estimate forward from its own actions: the drain follows from the delivered thrust.

**Design rules**, checked by simulation when you run `python battery_plane.py`:
- Efficient cruising never earns reward.
- The reward band is out of reach at ground level.
- Full reward needs a healthy battery.
- Always sprinting kills the plane for the low part of the starting range.
- A full battery can hold the band for a substantial part of the mission.

The same report measures the value of context with hand-written policies, averaged over starting batteries:

| Policy | Mean return |
|---|---:|
| Oracle (knows `b`) | 445 |
| Best single blind rule (hold 56 m/s until the speed sags, then cruise) | 283 |
| Always sprint | 129 |
| Cruise only | 132 |

**Evaluation protocol:**
- **Validation:** seeds 1000–1002 × starting battery {0.2, 0.3, …, 0.7} × speed noise 5, so 18 fixed flights.
- **Final test:** seeds 2000–2002, reserved for `evaluate_final_test.py` and never used for tuning or selection.
- **Training:** reset seeds start at the training seed (0–9 across seeds 0–2), disjoint from both.
- **Privileged info:** the info keys `b`, `b0`, `cap` and `saturated` may be training labels or diagnostics, never policy or encoder inputs.

## 5. Side 2: Contextual PPO (`contextual_ppo/`)

Stable-Baselines3 PPO with a recurrent context encoder. It doesn't depend on BatteryPlane (it never imports it) and mirrors SB3's layout. The numbers below are the defaults in `config/cppo.yaml`, which is experiment A.

**Acting: one step `t` in one training environment.** The memory, the xLSTM state and the last executed action, carries over from step to step and is reset to zeros at every episode start.

```
  o_t = [y, v_x + noise, v_y, tau]      a_{t−1} = last executed action      h_{t−1} = xLSTM memory
  BatteryPlane, hidden mode (no b)      (clipped to [−1, 1])                (from step t−1)
            │                                     │                                   │
            └──────────────────┬──────────────────┘                                   │
                               ▼                                                      ▼
  ┌─ ENCODER ────────────────────────────────────────────────────────────────────────────────────┐
  │ input layer   Linear([o_t, a_{t−1}]: 4 + 2 → 128)                                            │
  │ xLSTM stack   mLSTM block → sLSTM block, width 128  ──────────────► h_t (new memory)         │
  │ embedding e_t (128 numbers)                                                                  │
  │   ├──► context head  128 → 128 (tanh) ─┬─ point:    → 1 → sigmoid:  z_t ≈ b_t                │
  │   │                                    └─ gaussian: → 2 → μ_t (sigmoid), σ²_t (softplus);    │
  │   │                                                 z_t = [μ_t, log σ²_t]                    │
  │   └──► speed head (optional)  128 → 128 (tanh) → 1:  ≈ v_x / v_ref   (training loss only)    │
  └──────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                 │ z_t
                      ┌──────────────────────────┴──────────────────────────┐
                      ▼                                                     ▼
  ┌─ ACTOR  π(a | o_t, z_t) ───────────────┐          ┌─ CRITIC  V(o_t, z_t) ────────────────────┐
  │ input   [o_t, z_t]                     │          │ input   [o_t, z_t]   (no action)         │
  │ MLP     256 → 256 (tanh)               │          │ MLP     256 → 256 (tanh)                 │
  │ output  mean μ_a (2 numbers),          │          │ output  V_t, one number: the             │
  │         learned log σ_a (2 numbers,    │          │         expected discounted return       │
  │         the same in every state)       │          │         from here under the current      │
  │ sample  a_t ~ N(μ_a, σ_a²)             │          │         policy                           │
  │         (evaluation: a_t = μ_a)        │          │                                          │
  └───────────────────┬────────────────────┘          └─────────────────────┬────────────────────┘
                      │ a_t (unclipped: log π(a_t) uses it)                 │ V_t
                      ▼                                                     ▼
      clip to [−1, 1] = executed a_t                                 rollout buffer
                      │
                      ├──► env.step(executed a_t) ──► o_{t+1}, r_t, done, info
                      └──► memory for step t+1: (h_t, executed a_t)

  info["b0"] at a reset and info["b"] after each step hold the true battery b_t:
  labels for the episode buffer only, never an input to the encoder, actor or critic
```

**Training: one update after every 512 steps × 8 environments = 4,096 transitions** (`encoder_objective: supervised`, used by every experiment so far).

```
  ROLLOUT BUFFER (for PPO)                              EPISODE BUFFER (for the encoder)
  8 complete streams of 512 steps:                      complete episodes, up to 200k steps:
  o_t, a_{t−1}, a_t, log π_old(a_t), V_t,               (o_t, a_{t−1}) with labels b_t
  r_t × reward_scale, episode starts,                   (and v_x / v_ref for the speed head);
  each stream's encoder memory at its start             10% of the episodes kept for validation
                │                                                           │
                ▼                                                           │
  ① GAE (γ 0.99, λ 0.95): advantages Â_t, returns R_t = Â_t + V_t           │
                │                                                           │
                ▼                                                           │
  ② PPO epochs: 10 epochs × 4 minibatches of 2 whole streams                │
     • re-run the encoder over each stream from its stored memory,          │
       without gradient (the encoder is frozen here) → z_t                  │
     • actor loss    −min(ρ_t Â_t, clip(ρ_t, 0.8, 1.2) Â_t),                │
                     ρ_t = π(a_t | o_t, z_t) / π_old(a_t | o_t, z_t);       │
                     Â_t normalized per minibatch, entropy weight 0         │
     • critic loss   0.5 · (V(o_t, z_t) − R_t)²                             │
     • Adam (lr 3e−4) on the actor and critic only, gradient                │
       norm of the two clipped together at 0.5                              │
                │                                                           │
                │                                                           ▼
                └────────────────────────────────►  ③ ENCODER STEPS: 8 per update
                                                       • each: 16 complete training episodes, the
                                                         encoder run from its initial state over
                                                         [o_t, a_{t−1}]
                                                       • point:    MSE(z_t, b_t)
                                                         gaussian: β-NLL(μ_t, σ²_t; b_t)
                                                         + 0.1 × speed MSE if the speed head is on
                                                       • Adam on the encoder only; one batch of
                                                         validation episodes measures the fit
```

**Why the critic has no action input.** PPO's critic is a state-value function, `V(o_t, z_t)`. It predicts the discounted return from the current situation, averaged over the actions the current policy might take, and its job is to be the baseline in the advantage `Â_t`, which GAE computes from the rewards and the values. The action enters PPO's learning through the actor loss instead: the probability ratio `ρ_t` of the action actually taken, weighted by `Â_t`. A critic that reads the action, `Q(s, a)`, belongs to off-policy methods such as SAC, which Ndir et al. and SPARC use. Past actions still reach the critic indirectly: `a_{t−1}` is an encoder input, so `z_t` summarizes the action history. That matters here, because the battery drains with the actions.

**Options and controls:**

- **Context head and inputs:** `cppo.context_head` is `point` or `gaussian`. With `gaussian`, `cppo.context_variance_input` sets what the actor and critic read: `[μ_t, log σ²_t]` (default), `[μ_t, σ²_t]`, or the mean only. `cppo.velocity_head.enabled` adds the speed head; the actor and critic never read it.
- **Encoder objective:** `supervised` (above) trains the encoder only on the true battery. The alternative, `rl`, sends the PPO losses into the encoder instead: critic loss plus `encoder_actor_loss_weight` × actor loss.
- **Backpropagation through time:** `cppo.bptt_len` cuts the PPO re-encoding into segments; the default is the whole 512-step stream.
- **Controls:** `cppo.context_source=oracle` replaces the encoder with the true battery, `z_t = b_t`, read from an observed-mode observation (`step0_oracle.py`). `cppo.context_source=none` removes it altogether (`z_t = 0`, hidden mode): Contextual PPO without memory, the matched memoryless control for experiment A. `step1_controls.py` adds noise of a chosen size or schedule to that battery. These separate "can the policy use a battery estimate?" from "can the encoder produce one?".
- **Experiment grid A–F** (`GRID_PLAN.md`): point vs. Gaussian estimate, beta-NLL, whether the variance is an input, and the speed head. Three seeds each.
- **A frozen encoder trained offline** (`ENCODER_PLAN.md`, step 4): `train_encoder.py` trains an encoder on the pilot flights of `pilots.py`. It can be experiment A's xLSTM or a GRU / ConvGRU from `recurrent_encoders.py`. `context_wrapper.py` then appends its frozen battery estimate to plain PPO's observation (`run_step0.py --conditions 4a_encoder 4b_encoder --encoder-models ...`).
- **The ideal estimator as a teacher:** `train_encoder.py --target teacher` trains the GRU on the ideal estimator's estimate instead of the true battery. The targets come from `step4_floor.py --splits train validation test`. With `--context-head gaussian` the GRU learns the ideal estimator's uncertainty as well.
- **Joint training with a GRU encoder** (`config/joint/`, J0–J5): `cppo.encoder_arch=gru`, with options for more encoder training per update, a warm-up before PPO starts (`policy_warmup_steps`), and a curiosity bonus for flights that reveal the battery (`cppo.curiosity`).

## 6. Where things stand (2026-10-22)

Details, tables and W&B groups are in `ENCODER_PLAN.md`.
- **Validation:** the mean validation return over the last quarter of training (1.5M–2M steps).
- **Final test:** the final checkpoint on the reserved final-test flights (seeds 2000–2002), used once, after all training and selection (`evaluate_final_test.py --reports`).
- **Flights completed:** on the final test.

| Agent | Battery | Memory | Validation | Final test | Flights completed |
|---|---|---|---:|---:|---:|
| Plain PPO, hidden mode (batch 256 / 1024) | No | No | 23 / 77 | 49 / 119 | 25% / 17% |
| Contextual PPO with no context | No | No | 115 | 155 | 27% |
| Plain PPO, observed mode (batch 256 / 1024) | Yes | No | 246 / 258 | 283 / 280 | 90% / 92% |
| Contextual PPO with the true battery | Yes | Oracle | 282 | 286 | 89% |
| Contextual PPO with its learned encoder (experiment A rerun, 3 seeds) | No, inferred | xLSTM | 22 | 42 | 17% |
| Plain PPO with the ideal estimate (batch 256 / 1024) | No, estimated perfectly | Frozen ideal estimator | 283 / 323 | 316 / 348 | 97% / 97% |
| Contextual PPO with a GRU encoder trained jointly (best rung, J1, 5 seeds) | No, inferred | GRU | 114 | 150 | 30% |
| Plain PPO with a frozen GRU trained offline on the true battery (batch 256 / 1024) | No, inferred | Frozen GRU | 209 / 164 | 238 / 184 | 92% / 76% |
| **Plain PPO with a frozen GRU trained offline on the ideal estimate** (batch 256 / 1024) | No, inferred | Frozen GRU (teacher T1) | **291 / 262** | **330 / 319** | 86% / 76% |

All rows: discount 0.99, reward scale 0.1, 2M steps, 18 flights per evaluation. Seeds 0–7, except the A rerun (0–2) and joint training (0–4). Every run on the final test first reproduced its validation result exactly.

**On the final test:**
- **The validation ranking and gaps hold.** Every condition scores 20–51 points higher, because these flights are a little easier, but about equally.
- **The teacher-trained GRU is on a level with the ideal estimate:** 330 against 316 at batch 256, and 319 against 348 at batch 1024. Neither difference is significant at 8 seeds.
- **It scores above the true battery** (+47 / +38, not significant).
- **It clearly beats the GRU trained on the true battery at batch 256:** +92, p < 0.001 (Welch's t-test).
- **Contextual PPO behaves the same way on the final test:**
  - With the true battery it matches plain PPO with the true battery (286 against 283 / 280).
  - Its own encoder (A, 42) stays far below the memoryless control (155).
  - Joint training (J1, 150) stays at the memoryless level.
- **The scale from no memory (0%) to the ideal estimate (100%), batch 256 / 1024:**
  - GRU trained on the true battery: 71% / 53%
  - teacher-trained GRU: 105% / 87%
  - true battery: 88% / 71%
- **The frozen-GRU rows** use the encoder seed with the best validation error.
- **GRU trained on the true battery:**
  - At batch 256, all 8 seeds score 167–261 on validation.
  - At batch 1024, one seed collapsed (−294 on validation, −299 on the final test), so the median, 218, is the fairer summary.
  - Other encoder seeds at batch 1024 score 179 and 120 on validation, and 240 and 152 on the final test.
- **GRU trained on the ideal estimate:** all seeds score 216–333 at batch 256 and 189–316 at batch 1024 on validation, with no collapse; on the final test the worst seed scores 250 / 220.
  - **Its encoder seed hardly matters.** Two more encoder seeds at batch 1024 score 313 and 331 on the final test (validation 279 and 272), against 152–240 for the three encoder seeds of the GRU trained on the true battery.
- **Swap test** (validation flights): the frozen-encoder policies flown with the true battery in place of their encoder's estimate.
  - **The teacher-trained GRU's policies** score as well or better (+18 on average, 23 of 32 runs, p = 0.008). Like the ideal-estimate policies of step 2, they are simply better pilots, and for them the estimate and the true battery are interchangeable.
  - **The policies of the GRU trained on the true battery** mostly score worse (−20 on average, 17 of 32 runs). Their policies rarely fly at the motor's limit, so the battery stays hidden: on their flights even the ideal estimator is pulled toward the average battery (too high for low batteries, too low for high ones), and the GRU only a little more. The policies adapted to that compressed input, so the truth is an unfamiliar input.

**Battery estimates on held-out pilot flights** (error in battery units, 1,900 test flights of 11 pilot styles):

| Estimate | Error | Share of the gap from clock to ideal |
|---|---:|---:|
| Clock only (the mean battery at each flight step) | 0.144 | 0% |
| A's xLSTM trained offline, stopped by impatient early stopping | 0.129 | 26% |
| A's xLSTM trained offline, given enough passes (6 runs) | 0.101–0.107 (mean 0.103) | about 70% |
| GRU trained offline on the true battery (5 seeds) | 0.097 | 80% |
| GRU trained offline on the ideal estimate, T1 (5 seeds) | 0.093 | 86% |
| GRU trained on the ideal estimate and its uncertainty, T2 (5 seeds) | 0.093 | 86% |
| Ideal estimator (exact physics, the same inputs) | 0.085 | 100% |

**What this shows so far:**

- **The environment side holds for memoryless agents.** With every learner, knowing the battery is worth 170–220 points (246 vs 23, 258 vs 77, 282 vs 115), and completion rises from 15–26% to 86–88%.
- **The memory agent's machinery works.** Given the true battery, Contextual PPO matches or beats plain PPO with the battery.
- **A perfect battery estimate is enough.** Plain PPO reading the ideal estimator's estimate (the best estimate possible from the same inputs) scores 283 / 323, above plain PPO with the true battery (246 / 258). Flown with the true battery instead, the same policies score 280 / 310, so they are simply better pilots. The bottleneck is the learned encoder.
- **The encoder learned inside Contextual PPO doesn't deliver.** Experiment A scores 22, below Contextual PPO with no context at all (115, all 8 seeds between 90 and 139). Its planes fly like blind agents, fast until the battery runs out, even though its battery estimate is fairly accurate on average (R² 0.80).
- **Trained offline on varied flights, an encoder does learn the battery.** A GRU gets 80% of the way from a clock-only guess to the ideal estimator, and A's xLSTM gets about as far when given enough training (about 15 times slower).
  - Both sit at the clock for 9–11 passes over the data, then suddenly "click".
  - A filter front-end and dropping the time-left input don't help.
- **A frozen learned encoder works in the loop.** Plain PPO with the frozen GRU scores 209 at batch 256 against 23 without memory, and completes 88% of flights, as with the true battery. That is about 72% of the way to the ideal estimate (283).
- **How a policy flies decides what can be known about its battery.** Measured on PPO's own flights (the closed-loop check):
  - Policies that never fly at the cap hide the battery from everyone, the ideal estimator included.
  - Policies that probe reveal it. There the GRU lags the ideal estimate and overestimates the battery after about 5 s, which caused the one collapsed run.
- **Training the encoder on the ideal estimate instead of the true battery closes most of what was left.**
  - **Offline the gain is small:** 0.097 → 0.093 on the test flights, a third of the remaining gap to the ideal estimator. T1 is ahead of the GRU trained on the true battery at every pass, in every seed, so it isn't just longer training.
  - **Inside PPO it is large:** 291 at batch 256 (above the true battery's 246, on a level with the ideal estimate's 283) and 262 at batch 1024 (on a level with the true battery's 258).
  - **The reason:** on the PPO policies' own flights, the teacher-trained GRU stays within 0.012–0.016 of the ideal estimate, with no overestimation after probing, where the GRU trained on the true battery was 0.03–0.05 off.
  - **Why the target matters:** where the battery can't be known, the true battery is a noisy label. The ideal estimate is the clean version of the best possible guess.
  - **The catch:** the teacher knows the exact physics, the prior and the speed noise, so this encoder isn't purely learned from experience.
  - **Learning the uncertainty as well (T2)** costs no accuracy, and its uncertainty is calibrated (squared error over its variance about 1).
- **Joint training still fails.** Contextual PPO with a GRU encoder trained alongside the policy reaches 114 at best (J1, 16 times more encoder training), about the memoryless level (115). An entropy bonus, a Gaussian head, a curiosity bonus and a warm-up on random flights didn't help. The policy flies narrowly, so its encoder learns a clock again (R² about 0.8 on its own flights).
- **A likely reason A failed, not tested directly:** inside Contextual PPO the encoder got about 1,950 supervised steps of 16 episodes (A–F, 1M steps), or 3,900 (the A rerun, 2M). It trained on its own clock-like flights. That is less training, on narrower data, than the offline encoders needed before their click (about 4,300–5,200 steps of 32 varied flights).

**Why, from the step 1 controls**, which replace the encoder with the true battery plus controlled noise:

- **Noise that never goes away is costly.** The policies here have no memory, so they can't average noise that changes every step. With noise of std 0.03 on the battery throughout training they score 85–108, and 51–64 at 0.12.
- **Early training noise does no lasting damage.** Noise during the first 500k steps, followed by the exact battery, still reaches 270–334. What matters is how accurate the estimate ends up.
- **Timing within each flight is decisive.** Noise that fades to zero by mid-flight is as good as the true battery. Noise that stays until mid-flight loses most low-battery flights, because a plane starting low must decide to save energy in about its first 250 steps.
- **The A encoder learned a clock, not the battery.**
  - On A's own flights its error falls from 0.17 at take-off to 0.02–0.03 by step 175.
  - On flights flown by a policy that sees the battery, the error stays at 0.13–0.17 for the whole flight.
  - A always flies about the same way, so in its data the battery is roughly "starting guess minus a fixed amount per step". The encoder learned that rule instead of reading the battery from how the plane responds.
- **Part of it can't be avoided.** An ideal estimator (exact physics, the same inputs as the encoder) shows that flying gently below the throttle limit hides the battery completely. On a good pilot's flights it stays 0.17 off on 20%-battery flights, while from 40% up it is within 0.01–0.03 by mid-flight, where A's encoder is 0.12–0.17 off. Even at full throttle nothing can know the battery in the first 1–2 seconds.

This is the autogenic difficulty from section 1 showing up in practice. The agent's behaviour decides what its encoder can learn about the battery, and the encoder's estimate decides whether the agent learns to behave differently.

**Done since 2026-10-14:**
- the ideal estimator as a target at every flight step
- offline encoder training on varied pilot flights (`pilots.py`, `train_encoder.py`)
- plain PPO with the frozen encoder, and its closed-loop check
- joint training of Contextual PPO with a GRU encoder (J0–J5)
- the ideal estimator as a teacher (T1, T2), and plain PPO with the frozen teacher-trained GRU
- the final test of every row in the table above
- the teacher-trained GRU's encoder-seed check, and the swap test of all frozen-encoder policies
- GIFs of every pilot style and of the four plain-PPO agents on the final-test flights (`pilot_gifs.py`, `policy_gifs.py`)
- the paper figures and the project page (`figures/`, `docs/`), and the trained models as the release `models-v1`
- the easy version and its experiments E0–E3d (this branch, below)

**Next steps** (`ENCODER_PLAN.md`):

1. Let the policy also read T2's uncertainty, as the ideal estimate's uncertainty was tested in step 2.
2. Return to Contextual PPO with the teacher-trained encoder as a starting point, or with the ideal estimate as the joint encoder's target.

**The easy version** (this branch, 2026-10-20 to 2026-10-22; `EASY_ENV_WARMUP.md` has every setting and result). The code: `battery_plane.py`'s `warmup_steps` and `transition_steps`, the learner's preludes (the encoder reads the take-off before every episode) and window training (`cppo.supervised.sequence`), and `run_step0.py --warmup-steps / --transition-steps`. Before every flight an autopilot flies a 5-second take-off at full throttle, which reveals the battery to an agent with memory, and hands over with the battery still uniform on 0.2–0.7. It removes the hardest autogenic step (deciding to spend battery to learn the battery); the hard version stays the main one.
- **It is a fair test.** Knowing the battery is still worth a lot (Contextual PPO on validation: 254 with the true battery, 158 without memory). Two variants: the take-off alone, where memoryless agents get a faster start and a partial hint but still fail every low-battery flight, and a 4-second transition back to the starting speed, which makes the hand-over identical to the hard version's start.
- **End-to-end Contextual PPO still doesn't beat no memory**, although its GRU encoder reads the take-off first (a new prelude option of the learner) and reaches the best possible error at hand-over, about 0.05. With the transition: J1 95, J0 67 and J5 (an encoder warm-up) 70, against 58 without memory and 256 with the true battery. The policies settle on a no-memory flying style before the encoder becomes useful.
- **The take-off alone comes closest.** J1 scores 169 on validation and 197 on the final test, against 155–158 without memory and 254–275 with the true battery (not a clear gain: final test p = 0.27). Its policies depend on the estimate, and given the true battery they fly as well as the true-battery agent (247 against 254): the encoder's remaining error holds them back.
- **Training that encoder on short windows instead of whole flights makes it worse:** plain chunking 138 and R2D2's stored-state windows 31, with the error at hand-over rising from 0.05 to 0.11–0.16.
- **Next:** a 7-second take-off (best possible error 0.037 instead of 0.050) or a Gaussian head, chosen on validation.

## 7. Repository map

| Path | What it is |
|---|---|
| `battery_plane.py` | The environment (NumPy core `BatteryPlaneVec`, gymnasium wrapper `BatteryPlaneEnv`, evaluation grid helpers, design report) |
| `contextual_ppo/` | The learner: `ContextualPPO`, policy, xLSTM encoder, rollout and episode buffers. `Contextual_PPO_project_plan.md` is its design document |
| `main_battery_plane_contextual_ppo.py`, `config/` | Contextual PPO training (Hydra). `config/cppo.yaml` is experiment A; `config/experiment/` holds A–F; `config/step0/`, `config/step1/` and `config/rerun/` hold the controls |
| `ppo/` | Plain SB3 PPO baselines (hidden and observed mode) |
| `step0_oracle.py`, `step1_controls.py` | The true-battery and noisy-battery controls |
| `context_wrapper.py` | A frozen battery estimate (the ideal estimator, a saved xLSTM, or a GRU / ConvGRU `encoder.pt`) appended to plain PPO's observations |
| `pilots.py` | Pilot flights in many styles for training a battery encoder offline |
| `train_encoder.py`, `recurrent_encoders.py` | Offline encoder training (A's xLSTM, GRU, ConvGRU; the true battery or the ideal estimate as the target; resume and finalize) and the standalone GRU / ConvGRU encoders, with an optional Gaussian head |
| `step1_ideal_estimator.py`, `step4_floor.py` | The ideal estimator (exact physics, the encoder's inputs), the floor it sets on the pilot flights, and the teacher targets |
| `figures/` | The paper figures: `collect_data.py` gathers the numbers from the saved results into `figures/data/`, `make_figures.py` draws them into `figures/out/` (PDF, PNG, SVG); `make_figures_person.py`, `fig1_overview.py`, `fig1_diagram.py` and `env_snapshots.py` draw the project page's figures into `figures/out_person/` |
| `docs/` | The project page for GitHub Pages: five pages with a shared `assets/site.css` and `assets/site.js`, the figures and the flight videos |
| `pilot_gifs.py`, `policy_gifs.py` | GIFs of the pilot styles and of trained plain-PPO agents on the final-test flights, written to `videos/` (gitignored) |
| `step4_probes.py`, `step4_closed_loop.py` | Encoder diagnostics: the counting and finding tasks, GRU / LSTM vs xLSTM, and the encoder on PPO's own flights |
| `flight_logging.py` | Flight diagnostics shared by both learners (validation, rollouts, gradient norms, final validation) |
| `run_grid.py`, `run_step0.py`, `report_step0.py`, `evaluate_final_test.py` | Launchers, reports and the one-time final test |
| `step1_fly_with_noise.py`, `step1_encoder_error.py`, `step2_swap_battery.py` | Step 1 and step 2 analyses |
| `checks/` | `unittest` checks: configs, seeds, logging, controls, the context wrapper, the pilots, encoder training, joint training and the final test (`step0`–`step4_checks`, `joint_checks`, `final_test_checks`), and on this branch the easy version, the learner's preludes and window training (`easy_env_checks`, `prelude_checks`, `sequence_checks`) |
| `GRID_PLAN.md`, `ENCODER_PLAN.md` | The A–F grid; the encoder-versus-policy plan with all results so far |
| `EASY_ENV_WARMUP.md`, `warmup_information.py` | The easy version (forced take-off): plan, settings and every result, and the first measurement of what a take-off reveals |
| `CLAUDE.md` | Working notes: commands, architecture, protocol rules |
| `papers/` | The taxonomy paper and the three allogenic-context papers |

## 8. Quick start

Training runs on CPU in the `carlos` conda environment (Python 3.11).

```bash
conda activate carlos
python battery_plane.py                                    # environment design checks and value of context
python main_battery_plane_contextual_ppo.py training.seed=0 wandb.mode=disabled            # experiment A, one seed
python main_battery_plane_contextual_ppo.py +step0=cppo_oracle training.seed=0 wandb.mode=disabled   # with the true battery
python ppo/main_battery_plane_ppo.py +experiment=S0a_observed_bs256 training.seed=0        # plain PPO, observed mode
python train_encoder.py --design E1 --arch gru --seed 0 --smoke                            # offline encoder training, tiny run
python -m unittest checks.grid_checks checks.step0_checks checks.step1_checks checks.step2_checks checks.step3_checks \
    checks.step4_checks checks.joint_checks checks.final_test_checks checks.logging_checks
```

`CLAUDE.md` lists the launchers that run many seeds in parallel.

## References

- A. Biedenkapp. *Contextual Intelligence: The Next Leap for Reinforcement Learning.* Blue Sky Ideas track, 2026. arXiv:2604.02348.
- T. Camaret Ndir, A. Biedenkapp, N. Awad. *Inferring Behavior-Specific Context Improves Zero-Shot Generalization in Reinforcement Learning.* EWRL 2024. arXiv:2404.09521.
- B. Grooten, P. MacAlpine, K. Subramanian, P. Stone, P. R. Wurman. *Out-of-Distribution Generalization with a SPARC: Racing 100 Unseen Vehicles with a Single Policy.* AAAI 2026. arXiv:2511.09737.
- F. Röder, J. Benad, M. Eppe, P. K. Banerjee. *Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization.* NeurIPS 2025. arXiv:2508.20294.
- A. Kumar, Z. Fu, D. Pathak, J. Malik. *RMA: Rapid Motor Adaptation for Legged Robots.* RSS 2021. (The two-phase scheme SPARC simplifies.)
- A. Hallak, D. Di Castro, S. Mannor. *Contextual Markov Decision Processes.* 2015.
