# Easy alternative environment: a forced take-off before the agent takes control

**Status:** idea with first measurements (2026-10-14). Nothing is implemented in the environment or the learner yet. The current environment stays the main (hard) version.

**Update (2026-10-17): kept as plan B.** In the hard version, a GRU encoder trained offline on varied pilot flights now makes plain PPO score 209 at batch 256: 23 without memory, 246 with the true battery and 283 with the ideal estimate (ENCODER_PLAN.md, step 4).
- **The closed-loop check** found that the best of these policies probe by themselves: they fly at full throttle for a while, so the battery shows.
- **Still, the forced take-off would probably help the encoder.** It is the full-throttle "find" situation, which the GRU handles well (about 0.06–0.07 after 5 s, near the best possible).
- **When to switch to the easy version:** if the remaining gap to the ideal estimate turns out to come from early-flight error.

**Update (2026-10-19): left for later work, with a new purpose.**
- **The hard version doesn't need it any more.** In the hard version, a frozen GRU trained offline on the ideal estimate (teacher T1, ENCODER_PLAN.md) now brings plain PPO to the ideal estimate's level on the final test: 330 / 319 at batch 256 / 1024, against 316 / 348 with the ideal estimate. So the early-flight error isn't what limits the two-stage method.
- **Its new purpose is a difficulty control for end-to-end learning.** End-to-end Contextual PPO still fails in the hard version (experiment A 42, joint training J1 150, memoryless 155 on the final test). If it closes most of the gap once the take-off reveals the battery for free, that shows the obstacle is the information-gathering loop (deciding to spend battery to learn the battery), not the memory agent.
- **Code status:** only `warmup_information.py` (section 3) exists. Neither the environment, the learner, the configs nor the checks have been changed yet.
- **Estimated effort** (this 16-core CPU server):

| Step | Code | Compute |
|---|---|---|
| E0: the environment option, its checks, the design report | about 2–3 h | minutes |
| E1: true-battery agents and memoryless controls | about 30 min | about 20–30 min |
| E2: PPO with the ideal estimate (the context wrapper runs the calculator over the take-off) | about 1–2 h | about 3.5 h |
| E3: end-to-end Contextual PPO with prelude support in `contextual_ppo/` (the delicate part) | about 4–6 h plus checks | 4.5 h (A-style xLSTM) to 10 h (J1-style GRU), all seeds at once |

  - E0 + E1, the quick answer to "is the easy version still a fair task?": about half a day.
  - E0–E3: about two days.

- **On a rented GPU:**
  - **Offline GRU training with true-battery labels** (`train_encoder.py`) would take about 10–30 min per run, against 2–3.5 h here. That training is whole flights in batches, which suits cuDNN.
  - **End-to-end Contextual PPO** would gain maybe 1.5–3× per run. It mostly steps the environment and runs the encoder one step at a time for 8 environments; only the encoder's training steps and PPO's re-encoding speed up. Many seeds side by side on this CPU server are about as efficient.

**In one paragraph.** Every flight starts with a short take-off flown by an autopilot at full throttle, which the agent does not control. Full throttle is the only way the battery shows in the plane's behaviour, so this stretch reveals the battery to anyone watching. The agent's memory, the encoder, watches it and can know the battery before the agent makes its first decision. The starting battery is raised by exactly what the take-off uses, so when the agent takes over, its battery is still uniform on 0.2–0.7, as in the current environment. The measurements below say the take-off must last about **5 seconds** (100 steps); 2 seconds reveals almost nothing.

## 1. Why: what the experiments so far showed

The details are in `ENCODER_PLAN.md`; this is the part that motivates the idea.

- **The battery only shows when the plane flies at full throttle.** Below the throttle limit, the plane's motion doesn't depend on the battery at all. With the ideal estimator (a "perfect calculator" with the exact physics), a scripted flight below the limit gave no battery information for the whole 35 seconds.
- **Even at full throttle, the battery shows slowly.** The speed reading is noisy (±5 m/s), so even the perfect calculator needs about 100 full-throttle steps (5 s) to get within 0.05.
- **The flights where the battery matters most are the ones that hide it.** A low-battery plane must save energy, so it flies gently, so its battery stays hidden. On a good pilot's flights the perfect calculator stays 0.17 off on 20%-battery flights and 0.09 off on 30%-battery flights.
- **The learned encoder (experiment A) fell into this trap.** It reads the battery almost perfectly on its own hard flights: within about 0.01 of the ideal from step 100. But it learned a clock from that one flying style, and it is 0.12–0.17 off when someone flies differently.
- **A ended up worse than having no memory at all:** 22 against 115 for the same learner without context, with all memoryless seeds between 90 and 139.

So in the current environment the agent must *decide* to spend battery to learn its battery. That is the hardest, most autogenic part of the task ("observing the context moves it"), and our agents never get past it. The easy version removes that one obstacle at the start of each flight, so that inference and tracking can be tested on their own.

## 2. The idea

1. **Sample the hand-over battery** `b_h ~ U(0.2, 0.7)`: the battery the agent will have when it takes control.
2. **Draw the starting state** as today: altitude `U(30, 120)` m, speed `U(0, 30)` m/s, climb rate 0.
3. **Compute the starting battery** `b_s` so that the take-off ends exactly at `b_h` (section 4.3).
4. **Fly the take-off:** an autopilot holds full throttle and level flight, `a = (1, 0)`, for `warmup_steps` steps. The observations along the way, with the usual speed noise, are recorded.
5. **Hand over.** The agent's first observation is the state after the take-off, and its battery is exactly `b_h`. The agent then flies the usual 700-step (35-second) mission.

The encoder gets the take-off's observations and actions before the first decision. Agents without memory just start from the hand-over state.

## 3. How long the take-off must be (measured)

`warmup_information.py` simulates 100 flights whose first steps are flown at full throttle, from random starting states drawn as in training. It uses simulation seeds 5000–5099, away from the training, validation and test seeds, and trains nothing. On each flight it runs the ideal estimator (`step1_ideal_estimator.IdealBatteryEstimator`), the best estimate any encoder could make from what it sees.

```bash
python warmup_information.py          # 100 flights, level and climbing, about 2 minutes
```

Full throttle, level flight:

| Forced full throttle for | 0 s | 1 s | 2 s | 3 s | 5 s | 7 s | 10 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best possible battery error after it | 0.148 | 0.136 | 0.116 | 0.083 | **0.050** | 0.037 | 0.030 |
| The estimator's own uncertainty | 0.145 | 0.137 | 0.114 | 0.086 | 0.054 | 0.041 | 0.031 |
| Battery used, average | 0 | 0.024 | 0.045 | 0.065 | **0.103** | 0.141 | 0.194 |
| Battery used, range | 0 | 0.017–0.033 | 0.034–0.060 | 0.050–0.084 | **0.081–0.131** | 0.111–0.176 | 0.154–0.241 |

- **2 seconds is almost useless.** 0.116 is close to knowing nothing (0.148), because the noisy speed reading can't separate batteries in that time.
- **5 seconds is about the minimum that works:** 0.05, for about 0.10 battery. 7 seconds gives 0.037 for about 0.14.
- **Climbing instead of level flight** (lift command 0.2) gives the same information, for slightly more battery.
- **The estimator's own uncertainty matches its errors,** so the numbers can be trusted as the best case.
- **Why it takes this long:** at full throttle the plane accelerates with the motor's battery-dependent limit, `11.5 + 6.5·b` m/s². Between a 20% and a 70% battery that is a difference of 3.25 m/s², and it has to be read off a speed signal with ±5 m/s noise.
  - The error of an acceleration fitted from noisy readings shrinks roughly with the duration to the power 1.5.
  - From the take-off alone that gives about 0.21 after 2 s, worse than knowing nothing (0.145), and about 0.05 after 5 s.
  - Combined with the prior, 2 s gives about 0.12, matching the measured 0.116.

**Recommendation: 100 steps (5 s).** Consider 140 (7 s) if the remaining 0.05 turns out to matter (section 6).

The hand-over state after the take-off (200 simulated starts):

| Take-off | Hand-over speed | Altitude | Correlation of hand-over speed with battery | Correlation of battery used with starting battery |
|---|---|---|---:|---:|
| 2 s | 34.5 m/s (23.9–43.9) | unchanged (30–120 m) | 0.24 | 0.80 |
| 5 s | 44.9 m/s (38.8–51.2) | unchanged (30–120 m) | 0.62 | 0.96 |

## 4. Environment design

### 4.1 Parameters

New arguments of `BatteryPlaneEnv`. They can be set in a run's `env.kwargs`, like the other environment arguments.

| Argument | Default | Meaning |
|---|---|---|
| `warmup_steps` | `0` | Length of the forced take-off. **0 is today's environment, unchanged.** The easy version uses 100 |
| `warmup_action` | `(1.0, 0.0)` | The autopilot's command: full cruise throttle, level flight. It never depends on the battery |

With `warmup_steps=0`, every observation, info dict and random draw must be identical to today's environment for the same seeds. That keeps A–F, step 0, step 1 and their reports valid.

### 4.2 The forced take-off

- The autopilot sends `warmup_action` at every take-off step. The agent's actions are not involved.
- The physics, the battery drain and the speed noise are the normal ones; nothing is free or simplified.
- The take-off is not part of the mission. Its rewards are not given to the agent and not counted in the return, and the mission clock (`tau`) has not started yet.
- No flight can end during the take-off. Starting at 30 m or more and flying level, the plane doesn't reach the ground in 5 s, and the battery stays above 0.2. The environment should still check this and raise an error if it ever happens.

### 4.3 Starting battery per flight

The battery used is **not a fixed amount**. It depends on the battery itself, because a fuller battery gives a stronger motor, which drains faster (the correlation is 0.96). It also depends on the starting speed and altitude, because the lift motor works harder at low speed. For 5 s it ranges from 0.081 to 0.131. A fixed `x` would therefore make the hand-over battery only roughly uniform, off by up to about ±0.025.

Instead, solve for the starting battery of each flight:

1. The take-off is deterministic given the starting battery `b_s`, altitude and speed. The observation noise doesn't affect the physics.
2. The hand-over battery increases with `b_s`. A higher start uses more, but still ends higher, by about 0.9 per unit of `b_s`. So there is exactly one `b_s` for each `b_h`.
3. Find it by bisection on `[b_h, min(1, b_h + 0.3)]`, simulating the take-off for each trial value without observation noise, to a tolerance of `1e-12`. Alternatively, simulate a batch of trial values at once in one vectorized `BatteryPlaneVec` and refine.
4. Reset with `b0 = b_s`, fly the take-off for real (with noise), and check that the battery at hand-over equals `b_h` to within `1e-9`.

With 5 s, `b_s` reaches at most about 0.83; with 10 s, about 0.94. Both are within the battery's range of (0, 1].

### 4.4 What the agent sees and when the mission starts

- **First observation:** the agent's first observation is the hand-over state, and the mission clock starts there: `tau = 1`, then 700 controlled steps, as today.
- **During the take-off:** observations follow the observation mode. Hidden mode has no battery column, observed mode does. During the take-off `tau` stays at 1, because the mission hasn't started.
- **The encoder's input:** the encoder reads the take-off as `[o_k, a_{k−1}]` pairs, with `a_{−1} = 0`, exactly like mission steps. At hand-over its "previous action" is the last autopilot action.
- **Validation:** the existing grid stays the same. Seeds 1000–1002 × battery 0.2–0.7, where `b0` now means the hand-over battery. The final-test seeds 2000–2002 stay untouched.

### 4.5 What `reset()` returns

`reset()` returns the hand-over observation and an info dict:

| Key | Content | May be an input? |
|---|---|---|
| `b0` | Hand-over battery `b_h`, the label of the first mission observation, as today | No (privileged) |
| `warmup_observations` | `[warmup_steps, obs_dim]`: the observation before each autopilot action (the one after the last action is the agent's first observation) | **Yes**, it is what the plane showed |
| `warmup_actions` | `[warmup_steps, 2]`: the autopilot's executed actions | **Yes** |
| `warmup_batteries` | The true battery at each take-off observation | No: training labels and diagnostics only |
| `b_start` | The starting battery `b_s` | No (privileged) |

The battery keys stay privileged under the project's rule: they may be training labels or diagnostics, never policy or encoder inputs.

### 4.6 Everything else

- **The design report** (`python battery_plane.py`) must be rerun for the easy version, with the take-off added to its scripted rollouts. Starting the mission at 39–51 m/s instead of 0–30 changes the early flight. The value-of-context check (a scripted policy that knows the battery against the best blind rule) must still clear its 20% threshold. "Always sprint dies for the bottom of the battery range" must still hold.
- **Implementation:** most simply in `BatteryPlaneEnv.reset()` (one plane), which every training and validation environment uses. The vectorized core's own auto-reset is only used by the design report.

## 5. Changes to the learner

The take-off is only useful if the encoder sees it. Today Contextual PPO's memory starts at the first decision, so the learner needs a **prelude**: steps that are encoded before the first decision but never acted on. `contextual_ppo/` must stay environment-agnostic, so it should take a generic option, for example `prelude_info_keys=("warmup_observations", "warmup_actions")`, and never import `battery_plane`.

| Part | Change |
|---|---|
| Collection (`ContextualOnPolicyAlgorithm`) | At every episode start, the first reset and every automatic reset, read the prelude from `env.unwrapped.reset_infos[i]`. The collector already reads its labels there. Run the encoder over it from the initial state, then continue from that state, with the last prelude action as the previous action |
| PPO re-encoding (`ContextualRolloutBuffer`) | Episodes that start inside a rollout currently reset the memory to the initial state, which would lose the prelude. Store the encoder state after each prelude and restart from it. This is exact while the encoder doesn't change during the PPO epochs, which holds for the supervised objective used by every experiment. With `encoder_objective=rl` the prelude would have to be stored and re-encoded with gradient, so the first version should only allow `supervised` |
| Encoder training (`SupervisedContextualEpisodeBuffer`) | Store the prelude steps as part of each episode, with their battery labels (`warmup_batteries`), so the encoder trains on the identification stretch too. Log the fit at hand-over separately |
| Evaluation | SB3's `evaluate_policy` never shows the policy the reset info. The validation loop (`flight_logging.evaluate_flights`, `ScheduledEvalResets`) needs a CPPO path that encodes the prelude at every validation reset |
| Plain PPO, the oracle, memoryless CPPO | No change: they have no memory, or see the battery directly. They start from the hand-over state |
| Ideal estimator | Run it on the prelude followed by the mission. It already handles any sequence of hidden observations and actions |

## 6. What changes about the benchmark

- **What is still autogenic:** the battery is still changed only by the agent's own thrust, it still limits that thrust, and it still changes every step. After hand-over it still can't be seen directly. The agent must keep track of it from its own actions, and if it wants a fresh reading later, it must pay for it by flying at full throttle again.
- **What becomes easier:** the agent no longer has to decide to spend battery to find out how much it has at the start of a flight. The environment pays for that measurement.
- **Two levels, reported separately:**
  - **Hard (today, `warmup_steps=0`):** infer, track, *and* decide when to spend battery on finding out.
  - **Easy (`warmup_steps=100`):** infer from a guaranteed identification stretch, then track.
- **A natural story for the paper:** an autopilot takes off at full power for 5 seconds, as real aircraft take off at full throttle, then hands over to the agent.

## 7. Risks and open questions

1. **About 0.05 of error remains at hand-over.** In step 1, policies struggled with estimate errors of 0.03–0.06. Those errors jumped at every step, though, while the hand-over error is a steady offset that is probably easier to live with; that is untested. The Gaussian head's variance (experiments C–F) gives the policy that uncertainty. A 7-second take-off cuts the error to 0.037.
2. **Too easy for memoryless agents?** After 5 s the hand-over speed is correlated with the battery at 0.62, so a memoryless agent's first speed reading carries a weak hint (one reading with ±5 m/s noise). Experiment E1 below measures whether the gap between memory and no memory stays large.
3. **A different starting state.** Missions start fast (39–51 m/s), so all scores shift, including those of agents that see the battery. Results from the easy and hard versions can't be compared directly.
4. **Tracking after hand-over can still go wrong.** If the policy flies one way every time, an encoder can still learn a clock for the tracking part. The take-off at least gives every flight a reliable identification point that doesn't depend on the policy.
5. **Cost:** each episode simulates 100 more steps (+14%), and the encoder runs 800 steps per episode instead of 700.

## 8. Checks to write before any training

- **Exact hand-over battery:** for random starts and for every validation reset, the hand-over battery equals the requested or sampled `b_h` to within `1e-9`. Over many samples it is uniform on 0.2–0.7.
- **The default is today's environment:** with `warmup_steps=0`, observations, rewards, info dicts and random draws are identical to the current environment for the same seeds.
- **No leakage through the autopilot:** take-off actions are identical for every battery. The prelude contains hidden-mode observations without the battery column, and the battery appears only under the privileged keys.
- **No flight ends during the take-off,** over many random starts.
- **The prelude reaches the encoder:**
  - Encoding the prelude and then the first observation gives the same context as encoding them in one sequence.
  - PPO re-encoding reproduces the contexts of collection for episodes that start mid-rollout.
  - Validation encodes the prelude.
- **Ideal estimator at hand-over:** on random starts, its error at hand-over is about 0.05, matching the table in section 3.
- **Seeds:** the take-off's observation noise uses the environment's random stream, so runs stay reproducible, and the training, validation and test seeds stay as they are.

## 9. Experiments and decision rules

Everything as in step 0: discount 0.99, reward scale 0.1, 2M steps, 8 environments × 512 steps, validation seeds 1000–1002, scored on 1.5M–2M, and at least 3 seeds (8 where it's cheap). Decide the rules before running.

| # | Experiment | What it answers | Cost |
|---|---|---|---|
| E0 | The environment option, its checks, and the design report for the easy version | Is the easy version still a fair task? (value of context above 20%, design rules pass) | Code, minutes to run |
| E1 | The true-battery agents (0a, 0b, 0c) and the memoryless controls (0a_blind, 0b_blind, 0c_blind) on the easy version | How much knowing the battery is worth there, and whether memoryless agents get much better | About 20 minutes per 24-run launch |
| E2 | **The ideal estimator as the encoder:** PPO with the ideal estimate (and its uncertainty) appended to the observation, on both versions | The best score any agent without the battery can reach in each version: the honest target for an encoder | An online wrapper with a coarser grid; roughly 0.5–2 hours per run |
| E3 | Experiment A (learned encoder) on the easy version, with prelude support | Does a learned encoder deliver once identification is guaranteed? | Learner changes; about 4–5 hours per run, like the A rerun |
| E4 | Optional: take-off length 3 / 5 / 7 s | How much the remaining error at hand-over matters | Repeats E2 or E3 |

**Decision rules:**
- **E1 must still show a big value of knowing the battery** (for example at least 100 points between each true-battery agent and its memoryless control). Otherwise the easy version is too easy and says nothing about memory.
- **E2 on the easy version:**
  - If the ideal-encoder agent lands within the seed spread of the true-battery agents, the battery is usefully inferable there, and E3 has a clear target.
  - If it stays far below, a 5-second take-off isn't enough: try 7 s (E4), or give the policy the uncertainty, before touching the learner.
- **E2 on the hard version** gives the ceiling for agents that must find their battery themselves. Comparing the two versions measures what the take-off is worth.
- **E3 counts as a success** if A clearly beats memoryless CPPO (beyond the seed spread) and comes close to the ideal-encoder agent. If it doesn't, compare its error at hand-over and after it with the ideal estimator (`step1_encoder_error.py` and `step1_ideal_estimator.py` with the prelude) to see whether identification or tracking fails.

## 10. Implementation checklist

| File | Work |
|---|---|
| `battery_plane.py` | `warmup_steps` and `warmup_action` in `BatteryPlaneEnv`; the starting-battery solver; the reset info keys; the spec in the module docstring; the take-off in the design report's scripted rollouts |
| `contextual_ppo/` | Generic prelude support: collector, rollout-buffer re-encoding, episode buffer, evaluation and `predict` |
| `main_battery_plane_contextual_ppo.py`, `ppo/main_battery_plane_ppo.py` | Pass the prelude keys; validation encodes the prelude; refuse combinations that aren't supported yet (such as `encoder_objective=rl` with a take-off) |
| `config/`, `ppo/configs/` | Easy-version configs with W&B names marking the version (for example `_warm100`) |
| `run_step0.py`, `report_step0.py` | Easy-version conditions; the report protocol must include `warmup_steps`, so the two versions are never mixed in one report |
| `step1_ideal_estimator.py`, `step1_encoder_error.py` | Accept and record preludes |
| `checks/` | The checks of section 8 |
| `run_grid.py` | `battery_plane.py` is already in the source digest, so grid outputs made before the change refuse to resume |

**Suggested order:** E0 (environment and checks), then E1, then E2. Only if E2 shows the easy version is usefully inferable, build the learner's prelude support and run E3.
