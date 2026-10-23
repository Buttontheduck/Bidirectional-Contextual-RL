# Easy alternative environment: a forced take-off before the agent takes control

**On branch `main`** this document records the plan, every training setting and every result. The code it describes (the take-off and transition options of `battery_plane.py`, the learner's preludes and its window training, their checks in `checks/`, and the `run_step0.py` options and conditions) lives on branch `easy-env`: check that branch out to rerun these experiments.

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

**Update (2026-10-20): E0 and E1 done; the take-off now ends with a transition back to the starting speed** (section 11).
- **E0:** `warmup_steps=100` in `battery_plane.py`, with checks (`checks/easy_env_checks.py`). The default environment is unchanged; its design report is byte-identical.
- **E1 on the take-off alone failed the rule.** Memoryless agents improved a lot (72 → 154 on average), because the hand-over speed gives the battery away: a fuller battery accelerates harder during the take-off. Two of the three gaps fell below 100 points.
- **The fix: a 4-second transition** (`transition_steps=80`). After the take-off the autopilot flies the plane back, level and smoothly, to the mission's starting speed. The agent takes over at the hard version's starting state for the same seed, and nothing at hand-over depends on the battery.
- **E1 with the transition passes:** gaps of 182, 171 and 198 points (rule: at least 100), and memoryless agents score as in the hard version (69 against 72 on average).
- **E3 (end-to-end Contextual PPO with the GRU encoder, reading the take-off) fails** (section 11): J1 scores 95 and J0 67, against 58 without memory and 256 with the true battery. The take-off solved identification: the online encoder is 0.047–0.049 off at hand-over, at the best possible. But even given the true battery, its policies score only about 157 and fly near full throttle at every battery: the policy settles on a no-memory flying style in the first ~130k steps, before the online encoder becomes useful.
- **J5 (the encoder learns for 200k steps before the policy) fails too:** 70. Its policies use the battery input much more than J1's, but still average only 151 given the true battery. The remaining suspect is the encoder changing while the policy learns; J6 (the encoder stops after the warm-up) would test it.
- **The take-off-only version is a usable test bench after all** (revised 2026-10-21): its memoryless agents' higher returns come mostly from the faster start, plus a partial hint; judged on survival it separates agents with and without the battery as clearly as the transition version (section 11).
- **E3c, end-to-end on the take-off-only version, is the closest end-to-end result so far** (section 11): J1 scores 169 on validation and 197 on the final test, against 155–158 without memory and 254–275 with the true battery (not a clear gain over no memory: final test p = 0.27). Its policies depend on the battery estimate, and given the true battery they match the true-battery agent (247 against 254); the encoder's remaining error (0.05 at hand-over) is what holds them back. Next: a 7-second take-off or a Gaussian head, chosen on validation.
- **E3d, training the end-to-end encoder on short windows instead of whole flights, makes things worse** (section 11): plain chunking (16-step windows from a blank memory) 138 and R2D2 (windows from the saved memory, 16 burn-in + 16 trained steps) 31, against 169 with whole flights. The encoder's error at hand-over doubles or triples (0.11–0.16 against 0.05): short windows don't teach it to carry the take-off's information. Whole-flight training stays the best.
- **E2 skipped for now** (the user's choice, 2026-10-20): E3 went first.

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
| `transition_steps` | `0` | Added 2026-10-20 (section 11): steps after the take-off in which the autopilot flies back to the mission's starting speed. The easy version uses 80 (4 s) |

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
   - **Outcome (2026-10-20): yes, it was too easy.** Memoryless agents scored 154 on average instead of 72. The transition removes the hint (section 11).
3. **A different starting state.** Missions start fast (39–51 m/s), so all scores shift, including those of agents that see the battery. Results from the easy and hard versions can't be compared directly.
   - **Outcome:** with the transition the mission starts from the hard version's starting state, so the scores compare directly.
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

**Done (2026-10-20):** `battery_plane.py` (with the transition), the environment checks of section 8, and `run_step0.py --warmup-steps / --transition-steps`. The configs need no change (the launcher passes `env.kwargs`), and `report_step0.py` already keeps the two versions apart (its protocol lists every environment argument).

## 11. Results

### Training settings of every run in this section

**Environment versions** (`battery_plane.py`, set through `env.kwargs`; everything else at the defaults of the hard version):

| Version | Settings | Prelude before the mission |
|---|---|---|
| Hard (for reference) | `warmup_steps=0` | none |
| Take-off only | `warmup_steps=100` | 5 s at full cruise throttle, level (100 steps) |
| Take-off + transition | `warmup_steps=100`, `transition_steps=80` | 5 s full throttle, then 4 s flying back to the mission's starting speed (180 steps) |

In all versions the hand-over battery is uniform in 20–70% (the reset's b0), the mission is 700 steps of 0.05 s, the speed reading has noise σ = 5 m/s, and the rest of the physics, reward and failure rules are the hard version's (`battery_plane.py` module docstring).

**Plain PPO** (0a, 0b, 0a_blind, 0b_blind; `ppo/main_battery_plane_ppo.py`, step 0's configs):
- Stable-Baselines3 PPO; actor and critic separate, 2 hidden layers × 256, tanh, orthogonal initialization; Gaussian actions with a state-independent learned std (log std 0 at the start), clipped to [−1, 1].
- Adam, learning rate 3 × 10⁻⁴ constant; 8 environments × 512 steps = 4,096 transitions per update; minibatches of 256 (0a, 0a_blind) or 1,024 (0b, 0b_blind) shuffled transitions; 10 epochs.
- γ = 0.99, GAE λ = 0.95, clip range 0.2, no value clipping, advantages normalized; entropy coefficient 0, value-loss coefficient 0.5, gradient clipping 0.5.
- Reward scale 0.1 during training only (all returns in environment units); 2M steps (488 updates).
- Inputs: observed mode (hidden observation plus the true battery) for 0a/0b; hidden mode (4 numbers) for the memoryless controls.

**Contextual PPO** (0c, 0c_blind, J0, J1, J5; `main_battery_plane_contextual_ppo.py`):
- The same PPO settings, except the minibatches: 4 per epoch, each 2 whole environment streams of 512 steps (1,024 transitions in time order). Actor and critic read the observation plus the context z<sub>t</sub>.
- 0c: the true battery as z<sub>t</sub> (observed mode, `TrueBatteryPolicy`). 0c_blind: z<sub>t</sub> = 0 (hidden mode, `NoContextPolicy`).
- J0, J1, J5 (`config/joint/`): a GRU encoder learned online. Input [o<sub>t</sub>, a<sub>t−1</sub>] (6 numbers), linear 6 → 128, GRU of 2 layers × 128, head of 2 hidden layers × 128 (tanh) with a sigmoid output; 232,193 parameters. After the PPO epochs of every update it takes supervised steps towards the true battery b<sub>t</sub> (mean squared error) on complete episodes of the agent's own flights; PPO's losses never reach it. Adam 3 × 10⁻⁴ shared with the actor and critic, no encoder gradient clipping; 10% of the episodes held out for validation.
- Preludes (`contextual_ppo/`, on automatically for a learned encoder when `warmup_steps > 0`): before every episode the encoder reads the prelude (the take-off, and the transition if any) from the reset info, and the episode starts from that memory with the autopilot's last action as previous action; the episode buffer stores each episode with its prelude and the prelude's batteries as labels; the PPO epochs use the contexts cached at collection; validation and the final test feed the prelude through `PreludePredictor`.

| Variant | Encoder steps per update | Episodes per encoder step | Episode buffer | Other |
|---|---:|---:|---:|---|
| J0 (A's encoder budget) | 8 | 16 | 200,000 steps | |
| J1 | 32 | 32 | 2.1 million steps | |
| J5 | 32 | 32 | 2.1 million steps | no PPO update for the first 200,000 steps, with 128 encoder steps per update during that warm-up |
| J1_chunk | 32 | 512 windows of 16 steps | 2.1 million steps | plain chunking (`cppo.supervised.sequence=chunk`): every window starts from a blank memory |
| J1_r2d2 | 32 | 512 windows of 16 + 16 steps | 2.1 million steps | R2D2 (`cppo.supervised.sequence=stored_state`): every window starts from the memory saved at collection; 16 burn-in steps without loss or gradient, then 16 trained |

**Evaluation** (all runs): validation every 50,000 steps on 18 flights (seeds 1000–1002 × starting battery 0.2, 0.3, … 0.7, where the starting battery is the hand-over battery), deterministic actions. Score: the mean validation return over 1.5M–2M steps (11 evaluations). Final test: seeds 2000–2002 on the same grid, only through `evaluate_final_test.py`, which first re-runs each run's validation and refuses if the flights don't reproduce.

**Seeds, launches and run times** (16-core Ryzen 9 9950X, CPU only; `run_step0.py` gives every run its own process; W&B names end in `_warm100` for the take-off-only version and `_warm100_trans80` with the transition):

| Experiment | Conditions × seeds | Launch | Run time per run |
|---|---|---|---|
| E1, take-off only | 0a 0b 0c 0a_blind 0b_blind 0c_blind × 0–7 | `--jobs 32 --threads 1 --warmup-steps 100 --output runs_step0/easy_e1` | 11–21 min |
| E1, with the transition | the same | `--jobs 32 --threads 1 --warmup-steps 100 --transition-steps 80 --output runs_step0/easy_e1_trans80` | 17–29 min |
| E3, with the transition | J0 J1 × 0–4 | `--jobs 10 --threads 3 --warmup-steps 100 --transition-steps 80 --output runs_step0/easy_e3` | J0 68–70 min, J1 227–232 min |
| E3b (J5), with the transition | J5 × 0–4 | `--jobs 5 --threads 3 --warmup-steps 100 --transition-steps 80 --output runs_step0/easy_e3_j5` | 249–251 min |
| E3c, take-off only | J0 J1 × 0–4 | `--jobs 10 --threads 3 --warmup-steps 100 --output runs_step0/easy_e3_takeoff` | J0 53–62 min, J1 206–208 min |
| E3d, take-off only | J1_chunk J1_r2d2 × 0–2 | `--jobs 6 --threads 5 --warmup-steps 100 --output runs_step0/easy_e3_windows` | J1_chunk 58 min, J1_r2d2 69 min |

Each launch line is `nohup python run_step0.py --conditions <conditions> --seeds <seeds> <launch> > run_<name>.log 2>&1 &`.

### E0: the environment option (2026-10-20)

- **Implemented as in section 4.** The starting battery is solved by a vectorized grid search refined 7 times (to about 1e-11); the hand-over battery is checked to within 1e-9, then set exactly.
- **The default is unchanged.** With `warmup_steps=0`: the same observations, rewards, info and random draws, and a byte-identical design report.
- **Design report with the take-off alone:**
  - The take-off uses 0.089–0.150 of the battery (from 100 m, 15 m/s), and hand-over batteries up to 0.84 can be reached from every start.
  - Always-sprint still dies below 0.70.
  - **But the value of knowing the battery fell from 36% to 20%,** right at the threshold. That was a first sign of risk 2.
- **Launch:** `run_step0.py --warmup-steps 100` adds `+env.kwargs.warmup_steps=100`, and `_warm100` to the W&B names and groups.

### E1 on the take-off alone (2026-10-20)

48 runs: 0a, 0b, 0c and their memoryless controls, seeds 0–7, step 0's settings (`runs_step0/easy_e1/`). There are two reports, because the report refuses to mix observation modes. Scores are mean validation returns over 1.5M–2M steps.

| Learner | True battery | No memory | Gap | Rule (≥ 100) |
|---|---:|---:|---:|---|
| Plain PPO, batch 256 (0a) | 288 | 128 | 160 | met |
| Plain PPO, batch 1024 (0b) | 245 | 177 | 68 | **not met** |
| Contextual PPO (0c) | 254 | 158 | 96 | **not met** |

- **Memoryless agents got much better:** 154 on average, against 72 in the hard version.
- **First explanation (revised below):** the hand-over speed gives the battery away. During 5 s at full throttle a fuller battery accelerates harder, so the hand-over speed is correlated with the battery at 0.62, and a memoryless agent reads that from its first observation.

**All six agents in detail** (added 2026-10-21). Score: mean validation return over 1.5M–2M steps, ± the standard deviation across the 8 seeds; final: the final checkpoint's validation.

| Agent | Score | Final |
|---|---:|---:|
| Plain PPO batch 256, true battery (0a) | 288 ± 27 | 273 ± 26 |
| Plain PPO batch 1024, true battery (0b) | 245 ± 35 | 225 ± 51 |
| Contextual PPO, true battery (0c) | 254 ± 49 | 265 ± 69 |
| Plain PPO batch 256, no memory (0a_blind) | 128 ± 51 | 127 ± 39 |
| Plain PPO batch 1024, no memory (0b_blind) | 177 ± 26 | 193 ± 38 |
| Contextual PPO, no memory (0c_blind) | 158 ± 42 | 144 ± 60 |

Failed flights by starting battery (battery empty or ground crash; final validation, 24 flights per battery):

| Agent | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0a, true battery | 100% | 0% | 0% | 0% | 0% | 0% | 17% |
| 0b, true battery | 100% | 12% | 0% | 0% | 0% | 0% | 19% |
| 0c, true battery | 79% | 12% | 0% | 0% | 0% | 0% | 15% |
| 0a_blind, no memory | 100% | 100% | 100% | 88% | 62% | 0% | 75% |
| 0b_blind, no memory | 100% | 100% | 100% | 75% | 46% | 0% | 70% |
| 0c_blind, no memory | 100% | 100% | 100% | 100% | 54% | 0% | 76% |

- The true-battery agents fail at 0.2 by both causes (0a: 62% battery empty, 38% ground; 0c: 25% / 54%); their few failures at 0.3 are ground crashes. The memoryless agents fail only by emptying the battery.
- For comparison, with the transition the memoryless agents fail 79% of the flights at 0.6 and 29% at 0.7.

**Revised explanation (2026-10-21).** The failure rates don't support "memoryless agents read the battery":
- **They still fail every flight from 0.2 to 0.4,** like agents with no battery information. They do better only at the top: every flight at 0.7 completes and fewer fail at 0.6.
- **A good part of their higher return comes from the faster start.** The mission begins at about 40–50 m/s instead of 0–30, so they reach the reward band sooner and spend no battery accelerating; agents that fly near full throttle anyway profit most. The design report shows the same: the scripted blind pilot gains far more from this version (284 → 398) than the one that knows the battery (443 → 500).
- **There is a small real hint** (the hand-over speed correlates 0.62 with the battery), visible at 0.6–0.7.
- So this version gave memoryless agents a faster start plus a partial hint. The transition removes both.

**Revised verdict: the take-off-only version is a usable test bench.** It missed the return rule, but returns mix two things: whether an agent survives (where knowing the battery matters) and how much reward it collects before the flight ends (where the fast start helps everyone, and memoryless agents most). Judged on survival it separates the two kinds of agent as clearly as the transition version: 15–19% of flights failed with the true battery against 70–76% without memory, and 88–100% of the flights at 0.3–0.5 completed against 0–25%. The transition version stays the main one for comparisons with the hard version: the mission starts from the hard version's state, so the scores compare directly, and nothing about the battery is visible at hand-over. That is a preference for clean comparisons, not a flaw of the take-off-only version, which is simpler and has a natural story (take off at full power, hand over at cruising speed). Section "E3c" below runs end-to-end Contextual PPO on it.

### The fix: a transition back to the starting speed (2026-10-20)

After the take-off the autopilot flies `transition_steps` more steps, level, back to the mission's starting speed.

- **The hand-over state is the hard version's starting state.**
  - The reset's battery, altitude and speed (drawn, or given as options) describe the mission's start, as in the hard version. The take-off starts at that altitude from a fresh speed drawn from `U(0, 30)`.
  - For the same seed the agent takes over with the same battery and altitude as in the hard version, and with its speed within 0.55 m/s on average (2.2 m/s at worst, 600 seeds).
- **No sudden jump.** The plane slows down continuously, and its altitude doesn't change during the take-off or the transition (level flight).
- **The autopilot adds nothing about the battery.**
  - It reads only the observations: a filter on the noisy speed reading that predicts each step from its own thrust and the known drag, then corrects by each reading. A check replays it on the recorded observations and gets the recorded actions exactly.
  - It never asks for more than 11.5 m/s², the lowest motor limit any battery has. So the battery never limits it, and the plane responds the same whatever the battery. The battery shows only during the take-off's full throttle.
- **The hand-over battery stays exact.** The speed noise of the whole prelude is drawn before it, so the solve for the starting battery replays the same flight.
- **How long the transition must be.** After the take-off the plane flies at about 40–56 m/s, and braking is limited. A prototype on 6,000 random starts:

| Transition | Speed off target at hand-over (rms / worst) | Battery readable from the hand-over speed? |
|---|---|---|
| 1 s | 15 m/s / 32 m/s | yes (R² 0.20) |
| 2 s | 5 m/s / 17 m/s | slightly (R² 0.01) |
| 3 s | 0.95 m/s / 4.5 m/s | no |
| 4 s | 0.54 m/s / 2.1 m/s | no |

- **Chosen: 80 steps (4 s).** The prelude then lasts 9 s and uses 0.10–0.21 of the battery, and the starting battery reaches at most 0.90. Each reset takes about 75 ms for the solve.
- **Measured in the environment** (600 seeds): the hand-over speed's correlation with the battery is −0.01, against 0.62 with the take-off alone.
- **Design report:**
  - The prelude uses 0.126–0.179 of the battery (from 100 m and 15 m/s, back to 15 m/s), and hand-over batteries up to 0.78 can be reached from every start.
  - Always-sprint dies below 0.70.
  - **The value of knowing the battery is 36%, as in the hard version** (20% with the take-off alone).
  - The full-battery rule can't be checked here, because a full battery can't be handed over. The mission after hand-over is the hard version's, so its report covers that rule.
- **Code:** `transition_steps` and `PreludeAutopilot` in `battery_plane.py`; 6 more checks in `checks/easy_env_checks.py` (12 in all); `run_step0.py --transition-steps` (W&B suffix `_warm100_trans80`).

### E1 with the transition (2026-10-20)

The same 48 runs (`runs_step0/easy_e1_trans80/`; reports `step0_report_easy_true_battery.json` and `step0_report_easy_blind.json`; W&B names ending in `_warm100_trans80`). The gap's interval is a 95% bootstrap interval over seeds; p is from Welch's test.

| Learner | True battery | No memory | Gap (95% CI) | p | Rule (≥ 100) |
|---|---:|---:|---:|---:|---|
| Plain PPO, batch 256 (0a) | 242 | 60 | 182 (144–222) | 2e-6 | met |
| Plain PPO, batch 1024 (0b) | 258 | 88 | 171 (142–199) | 2e-6 | met |
| Contextual PPO (0c) | 256 | 58 | 198 (148–245) | 3e-5 | met |

All three versions side by side (true battery / no memory, same scoring):

| Learner | Hard | Take-off alone | Take-off + transition |
|---|---:|---:|---:|
| 0a / 0a_blind | 246 / 23 | 288 / 128 | 242 / 60 |
| 0b / 0b_blind | 258 / 77 | 245 / 177 | 258 / 88 |
| 0c / 0c_blind | 282 / 115 | 254 / 158 | 256 / 58 |
| Mean | 262 / 72 | 262 / 154 | 252 / 69 |

- **The rule is met with a wide margin.** Every gap's interval lies above 140.
- **Memoryless agents no longer gain anything:** 69 on average, against 72 in the hard version and 154 with the take-off alone.
- **Per learner, the memoryless scores move both ways** against the hard version: 0a_blind +37, 0b_blind +10, 0c_blind −57.
  - For a memoryless agent the two versions are the same task, with the same hand-over state. The take-off only changes the noise stream, so these are two independent sets of training runs.
  - The 0c_blind drop is larger than its hard-version seed spread (16) suggests, so it is worth watching.
  - **It isn't a code change.** Today's code reproduces the hard version's 0c_blind seed 0 exactly, with the same return on every validation flight at 50k and 100k steps, despite the joint-training options added since 2026-10-14.
- **True-battery agents** score about as in the hard version (252 against 262). 0c's lower mean comes mostly from seed 0 (134).
- **Verdict: with the transition, the easy version is a fair task.** The take-off reveals the battery only to agents that watch it, which is what E2 and E3 need.

### E3: end-to-end Contextual PPO reading the take-off (2026-10-20)

**Design.** Contextual PPO with the GRU encoder trained online, exactly the hard version's joint-training rungs J0 (A's encoder budget) and J1 (16 times more encoder training, the best rung there: 114), on the easy version with the transition. 5 seeds each, 10 runs at once with 3 threads, `runs_step0/easy_e3/`, W&B names ending in `_warm100_trans80`. J0 took about 70 minutes per run, J1 about 230.

**Learner change: preludes** (`contextual_ppo/`, off by default; `checks/prelude_checks.py`, 6 checks).
- At every episode start the encoder first reads the take-off and transition from the reset info (`policy.start_episodes`), and the episode continues from that memory, with the autopilot's last action as its previous action.
- The episode buffer stores each episode with its prelude and the true battery as labels, so the encoder trains on the identification stretch of every flight.
- The PPO epochs use the contexts cached at collection instead of replaying the encoder (a replay would restart episodes from blank memories). This is exact with the supervised encoder, whose steps come after the PPO epochs.
- Validation and the final test feed every flight's prelude through `PreludePredictor`.
- The checks compare against what the environment returned at each reset. Two deliberately broken versions (collection ignoring the prelude; PPO replaying the encoder) are both caught.

**Results** (late validation scores, mean over seeds, 95% bootstrap interval):

| Agent (easy version) | Score | Seeds |
|---|---:|---|
| Contextual PPO, true battery (0c, E1) | 256 (211–299) | 8 |
| **J1: online GRU, 16× encoder training** | **95** (42–135) | 2, 155, 116, 124, 81 |
| **J0: online GRU, A's encoder budget** | **67** (−5–114) | 87, −70, 91, 97, 129 |
| Contextual PPO, no memory (0c_blind, E1) | 58 (41–79) | 8 |
| *J1 / J0 in the hard version* | *114 / 45* | |

- **Neither beats no memory** significantly (J1 +37, Welch p = 0.24; J0 +8, p = 0.83), and both are far below the true battery (−161 and −189, p ≤ 0.002).
- **No gain over the hard version** (J1 −18, p = 0.59).
- **76–80% of the flights end with an empty battery:** the policies don't ration.

Failed flights by starting battery (battery empty or ground crash; final validation, 3 flights per battery per seed):

| Agent (easy version) | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| True battery (0c, 8 seeds) | 83% | 4% | 0% | 0% | 0% | 0% | 15% |
| J1 online GRU (5 seeds) | 100% | 100% | 100% | 93% | 60% | 0% | 76% |
| J0 online GRU (5 seeds) | 100% | 100% | 100% | 100% | 87% | 33% | 87% |
| No memory (0c_blind, 8 seeds) | 100% | 100% | 100% | 100% | 79% | 29% | 85% |

- **J1 fails like an agent without memory** up to a starting battery of 0.5, almost always by emptying the battery. The true-battery agent completes every flight from 0.4 up and nearly every one at 0.3. J1 only does better than no memory at 0.6 and 0.7.
- **The true-battery agent's failures are at 0.2 only,** half by an empty battery and half by hitting the ground. J0 also hits the ground at 0.6–0.7 (20%).

**Diagnostics** (`runs_step0/easy_e3/diagnostics_validation.json`, `diagnostics_followup.json`; validation flights only, reproducing each run's saved final validation exactly):

| Run | With its estimate | With the true battery | With a fixed 0.45 | Error at hand-over (200 random flights) |
|---|---:|---:|---:|---:|
| J1 s0 | 12 | 204 | 15 | 0.048 |
| J1 s1 | 174 | 176 | 105 | 0.048 |
| J1 s2 | 108 | 110 | 48 | 0.047 |
| J1 s3 | 107 | 171 | 63 | 0.049 |
| J1 s4 | 123 | 124 | 127 | 0.047 |
| J0 s0–s4 | −62 to 75 | −61 to 82 | −76 to 95 | 0.052–0.068 |

- **The take-off solved identification.** The online encoder's error at hand-over is 0.047–0.049 for J1, the best possible after a 5-second take-off (section 3: 0.050), and its estimate correlates 0.93–0.95 with the hand-over battery. Over the validation flights it stays 0.02–0.06 off. That is better than the offline GRU in the hard version (0.097). No leak: the encoder doesn't beat the best possible.
- **Most J1 policies use the estimate.** A fixed 0.45 instead costs 45–70 points in seeds 1–3; seed 4 ignores it; seed 0 depends on it so heavily that the estimate's small bias (+0.05 at 0.2) wrecks it.
- **But they are poor pilots even with the truth.** Given the true battery, the J1 policies average about 157, against 256 for Contextual PPO trained on the true battery from the start.
- **J1 flies the same at every battery.** Mean throttle by starting battery 0.2 → 0.7 (final validation): J1 0.95, 0.94, 0.94, 0.91, 0.86, 0.89; no memory 1.00, 1.00, 1.00, 0.99, 0.93, 0.88; true battery −0.19, 0.29, 0.38, 0.46, 0.55, 0.64. Despite an accurate estimate, J1's policy doesn't scale its throttle with the battery.
- **Not the size of the estimate's error.** The frozen GRU of the hard version was less accurate (0.097) and plain PPO learned well with it (209; 330 with the teacher-trained GRU). (A first reading here, "the online estimate is never exact", was wrong for that reason.)
- **Why, most likely: the estimate becomes useful too late.** In J1's training (seeds 1 and 3), the encoder's held-out error is 0.13–0.19 for the first ~130k steps (no better than a clock), while the policy learns to fly: its throttle climbs from 0 to 0.3–0.7 and its return from −290 to about 0. The encoder reaches about 0.06 between 200k and 400k steps, but by then the policy flies near full throttle, and it stays at 0.5–0.9 for the rest of training. To learn to slow down at a low battery it would have to try gentler flying, which loses reward at once and pays only at the end of the flight, while its actor and critic have learned to pay little attention to the battery input.
  - A frozen encoder is meaningful from the first step, so the policy learns "low battery, fly gently" while it learns to fly; its errors are also fixed, so the policy can compensate for them (the labels-GRU policies even did worse given the truth).
  - Consistent with step 1: a noisy battery early in training did no lasting harm, because the true battery plus noise is informative from the start; an untrained encoder isn't.
- **What it means for the benchmark:** the take-off removes the identification obstacle, but end-to-end learning still fails, now because the policy forms its flying style before the online encoder becomes useful.

### E3b: J5, the encoder learns first (2026-10-20)

**Design.** J5 of the hard version's joint-training rungs on the easy version: J1, except that for the first 200k steps the policy is not updated (it flies randomly) while the encoder trains, 128 encoder steps per update. Then normal J1 training for the remaining 1.8M steps. Still end-to-end: one run from scratch, the agent's own flights, nothing pretrained or frozen. 5 seeds at once with 3 threads, about 250 minutes per run, `runs_step0/easy_e3_j5/`, W&B `J5_gru_budget_warmup_gamma0.99_rs0.1_warm100_trans80_s<seed>`. The question: if the encoder is accurate before the policy starts learning, does the policy learn to use it?

**The condition held.** During the warm-up the encoder's held-out error fell to 0.074–0.112 by 49k steps and 0.062–0.077 when the policy started learning (averaged over whole flights of the random policy), while the policy's throttle was still about 0. J1 was at 0.13–0.19 at the same point, with its policy already at throttle 0.3–0.7.

**Results** (late validation scores):

| Agent (easy version) | Score | Seeds | Flights completed |
|---|---:|---|---:|
| True battery (0c) | 256 (211–299) | 8 | 85% |
| J1 | 95 (48–135) | 2, 155, 116, 124, 81 | 24% |
| **J5** | **70** (19–116) | −13, 103, 143, 31, 86 | 24% |
| No memory (0c_blind) | 58 (41–80) | 8 | 15% |
| *J5 in the hard version* | *94* | | |

- **J5 fails too.** It is no better than no memory (+12, Welch p = 0.71), than J1 (−25, p = 0.52) or than the hard version's J5 (−24, p = 0.56).

| By starting battery | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| Failed flights, J5 | 100% | 100% | 100% | 80% | 53% | 20% |
| Mean throttle, J5 | 0.49 | 0.70 | 0.79 | 0.78 | 0.77 | 0.80 |
| Mean throttle, J1 | 0.95 | 0.94 | 0.94 | 0.91 | 0.86 | 0.89 |
| Mean throttle, true battery | −0.19 | 0.29 | 0.38 | 0.46 | 0.55 | 0.64 |

**Diagnostics** (validation flights; `runs_step0/easy_e3_j5/diagnostics_*.json`):

| Run | With its estimate | With the true battery | With a fixed 0.45 | Error at hand-over / after 20 s |
|---|---:|---:|---:|---|
| J5 s0 | 25 | 275 | −60 | 0.034 / 0.094 |
| J5 s1 | 163 | 202 | 49 | 0.040 / 0.076 |
| J5 s2 | 144 | 138 | −71 | 0.044 / 0.033 |
| J5 s3 | 3 | 3 | −15 | 0.030 / 0.041 |
| J5 s4 | 95 | 139 | 66 | 0.033 / 0.036 |

- **The policies now depend on the battery input much more than J1's.** A fixed 0.45 costs 30–215 points (J1: 45–70, one seed none), and the throttle drops at low batteries (0.49 at 0.2, against 0.95 for J1). So the early accurate encoder did change what the policy learned.
- **The encoder is as good at hand-over as J1's** (0.048–0.055 on 200 random flights), but in two seeds it drifts later in the flight (0.076–0.094 after 20 s). Seed 0 shows what that costs: 25 with its estimate, 275 with the true battery, the true-battery agent's level.
- **But the policies are still poor on average.** Given the true battery they average 151 (J1: 157), against 256; seed 3 fails either way.

**What it means.**
- **The timing explanation is only part of the story.** With an accurate encoder from the start, the policy learns to use the battery, but the result is still a poor pilot.
- **The main remaining suspect is the encoder changing during training.** It trains after every update on the latest flights, so what a given estimate means keeps shifting while the policy learns to read it. Its errors also shift as the policy's flying changes, which shows as the drift later in the flight. A frozen encoder's estimate means the same thing throughout training, which plain PPO handled well even at 0.10 error.

**Next test (not run): J6, the encoder stops after the warm-up.** J5, except that the encoder stops training once the policy starts (after 200k steps). Still end-to-end in one run from scratch, on the agent's own flights, with no separate data. Its estimate then means the same thing for the rest of training. If J6 comes close to 256, the moving encoder was the obstacle. It needs a small option in `contextual_ppo/` (stop the encoder steps after a given step) and a check, then about 1–1.5 hours for 5 seeds (no encoder training after the warm-up).

### E3c: end-to-end Contextual PPO on the take-off-only version (2026-10-20 to 2026-10-21)

**Design.** E3 again, on the take-off-only version (5-second full-throttle take-off, no transition back): Contextual PPO with the GRU encoder learned online from scratch, J0 and J1, 5 seeds each (0–4), with no warm-up and no pretraining. The encoder reads the 100-step take-off before every episode. Settings as in "Training settings" above; `runs_step0/easy_e3_takeoff/`, W&B names `J0_gru_Abudget_gamma0.99_rs0.1_warm100_s<seed>` and `J1_gru_budget_gamma0.99_rs0.1_warm100_s<seed>`. The bars are this version's E1 runs: Contextual PPO with the true battery (0c) and without memory (0c_blind), 8 seeds each.

**Validation results** (late score, mean ± std across seeds, 95% bootstrap interval of the mean):

| Agent (take-off-only version) | Score | Seeds | Flights completed |
|---|---:|---|---:|
| True battery (0c) | 254 ± 49 (222–285) | 8 | 85% |
| **J1** | **169 ± 41** (137–201) | 139, 219, 152, 207, 128 | 32% |
| **J0** | **83 ± 181** (−79–188) | −234, 125, 218, 168, 138 | 20% |
| No memory (0c_blind) | 158 ± 42 (128–183) | 8 | 24% |

- **J1 is no better than no memory** (+11, Welch p = 0.65) and below the true battery (−85, p = 0.008).
- **J0 also fails:** one seed collapsed to −234; without it J0 averages 162, the no-memory level (J0 − no memory: −75, p = 0.41).

| By starting battery (final validation) | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| Failed flights, true battery | 79% | 12% | 0% | 0% | 0% | 0% |
| Failed flights, J1 | 100% | 100% | 87% | 80% | 20% | 20% |
| Failed flights, J0 | 100% | 100% | 100% | 80% | 80% | 20% |
| Failed flights, no memory | 100% | 100% | 100% | 100% | 54% | 0% |
| Mean throttle, true battery | 0.10 | 0.35 | 0.47 | 0.59 | 0.69 | 0.78 |
| Mean throttle, J1 | 0.81 | 0.81 | 0.74 | 0.76 | 0.78 | 0.83 |
| Mean throttle, J0 | 0.55 | 0.69 | 0.72 | 0.67 | 0.67 | 0.69 |
| Mean throttle, no memory | 1.00 | 1.00 | 1.00 | 0.96 | 0.89 | 0.87 |

**Diagnostics** (validation flights only; `runs_step0/easy_e3_takeoff/diagnostics_validation.json`, `diagnostics_followup.json`; every run reproduces its saved final validation exactly):

| J1 seed | With its estimate | With the true battery | With a fixed 0.45 | Error at hand-over (200 random flights) |
|---|---:|---:|---:|---:|
| 0 | 223 | 383 | 187 | 0.052 |
| 1 | 200 | 206 | 166 | 0.053 |
| 2 | 108 | 112 | −12 | 0.053 |
| 3 | 256 | 336 | 17 | 0.058 |
| 4 | 143 | 197 | 38 | 0.053 |
| Mean | 186 | 247 | 79 | |

- **J1's policies really use the battery input:** with a fixed 0.45 they drop to 79 on average (seeds 2–4 lose 105–240 points).
- **Given the true battery they are as good as the true-battery agent:** 247 on average against 254 (its validation score), and seeds 0 and 3 beat it (383, 336). In the transition version, J1's policies reached only 157 even with the truth.
- **The encoder is at the best possible at hand-over:** 0.052–0.058 off on 200 random flights (correlation 0.93 with the battery; section 3: 0.050 after a 5-second take-off). On the validation flights it is 0.035–0.049 off at hand-over and 0.02–0.06 later in the flight.
- **So the remaining gap is the estimate's error,** not the policy: those errors cost about 60 points on average (186 → 247 with the truth), and up to 160 in seed 0.
- J0's policies use the input less (a fixed 0.45 changes little except in seed 1), and its encoder is a little less accurate at hand-over (0.055–0.066).

**Final test** (2026-10-21; seeds 2000–2002, run once; `runs_step0/easy_e3_takeoff/final_test_takeoff.json`). All 21 runs first reproduced their validation exactly, so the code changes since the E1 runs (the transition and the prelude support) left the take-off-only version unchanged.

```bash
python evaluate_final_test.py --reports runs_step0/easy_e3_takeoff/step0_report_joint_J1.json \
  runs_step0/easy_e1/step0_report_easy_true_battery.json runs_step0/easy_e1/step0_report_easy_blind.json \
  --conditions J1 0c 0c_blind --output runs_step0/easy_e3_takeoff/final_test_takeoff.json --jobs 16
```

| Agent (take-off-only version) | Final test (mean ± std, 95% CI) | Seeds | Flights completed |
|---|---:|---|---:|
| Contextual PPO, true battery (0c) | **275 ± 56** (237–309) | 307, 283, 206, 330, 219, 203, 324, 326 | 85% |
| **J1, encoder trained online** | **197 ± 61** (150–244) | 206, 211, 124, 285, 157 | 33% |
| Contextual PPO, no memory (0c_blind) | **155 ± 61** (114–192) | 84, 223, 138, 179, 177, 214, 176, 51 | 24% |

- **J1 against no memory:** +41 points (95% CI −19 to +104; Welch p = 0.27, Mann-Whitney p = 0.52). It is ahead, but with 5 seeds against 8 the difference isn't clear.
- **J1 against the true battery:** −78 points (95% CI −136 to −16; Welch p = 0.049, Mann-Whitney p = 0.065).
- **The final test agrees with validation:** true battery 254 → 275, J1 169 → 197, no memory 158 → 155.

Failed flights on the final test by starting battery:

| Agent | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| True battery | 79% | 8% | 0% | 0% | 0% | 0% | 15% |
| J1 | 100% | 100% | 80% | 80% | 20% | 20% | 67% |
| No memory | 100% | 100% | 100% | 100% | 54% | 0% | 76% |

- **J1 saves some mid-battery flights that no memory loses** (20% completed at 0.4–0.5, 80% at 0.6 against 46%).
- **It still fails nearly all low-battery flights,** and loses 20% at 0.7, where no memory completes every flight. The true-battery agent completes almost everything from 0.3 up.

**What it means.**
- This is the best end-to-end result of the project so far: on the take-off-only version, end-to-end training produced policies that depend on the battery and that match the true-battery agent when given the truth. That never happened in the hard version or with the transition.
- The remaining obstacle is the accuracy of the online estimate (about 0.05 off at hand-over): not enough for these policies, which rely on it heavily.
- Why this version and not the transition version: not established. Possible factors are the shorter prelude (100 instead of 180 steps), the faster start (the policy reaches the reward band sooner, so it learns its flying style under different pressure), and the partial hint in the hand-over speed.

**Next tests (not run; choose between them on validation only, never on the final test above):**
- **A 7-second take-off** (`--warmup-steps 140`): the best possible error after 7 s is 0.037 instead of 0.050 (section 3). If the estimate's error is the limit, J1 should move towards the true-battery agent.
- **A Gaussian head** (J3's encoder: the estimate plus its variance, both read by the actor and critic): lets the policy fly more cautiously when the estimate is uncertain.
- Each is about 3.5 hours for J1 with 5 seeds (10 runs at once with 3 threads, as here).

### E3d: the end-to-end encoder trained on short windows (2026-10-22)

**Question.** In E3 and E3c the encoder trains on whole flights with full backpropagation through time. Would training it on short windows help? The hope: many more, cheaper encoder steps, so the encoder becomes useful sooner and the policy doesn't settle before it does. Two standard ways to cut flights into windows:
- **Plain chunking:** windows of 16 steps, each from a blank memory, with the battery error over those 16 steps.
- **R2D2** (Kapturowski et al., 2019): store the encoder's memory with every collected step; each window starts from the stored memory, runs 16 burn-in steps without loss or gradient to refresh it, then 16 trained steps.

**Design.** J1 on the take-off-only version (as E3c, whose J1 trains on whole flights), with only the encoder's training samples changed: 32 encoder steps per update, each on 512 windows; 16 trained steps per window (plus 16 burn-in steps for R2D2); the 2.1-million-step episode buffer. The validation error is still measured on whole flights. 3 seeds each (0–2), 6 runs at once with 5 threads. Conditions `J1_chunk` and `J1_r2d2` (`config/joint/J1_gru_budget_chunk.yaml`, `J1_gru_budget_r2d2.yaml`; option `cppo.supervised.sequence`, with `chunk_len`, `burn_in` and `chunk_batch_size`; checks in `checks/sequence_checks.py`). `runs_step0/easy_e3_windows/`; W&B `J1_gru_budget_chunk16_gamma0.99_rs0.1_warm100_s<seed>` and `J1_gru_budget_r2d2_burn16_chunk16_gamma0.99_rs0.1_warm100_s<seed>`. Run times 58 and 69 minutes per run, against 207 for J1 on whole flights.

**Results** (validation, late score, mean ± std across seeds, 95% bootstrap interval):

| Agent (take-off-only version) | Score | Seeds | Flights completed |
|---|---:|---|---:|
| True battery (0c) | 254 ± 49 (222–285) | 8 | 85% |
| J1, whole flights (E3c) | 169 ± 41 (137–201) | 139, 219, 152, 207, 128 | 32% |
| **J1_chunk** | **138 ± 69** (73–210) | 132, 73, 210 | 39% |
| **J1_r2d2** | **31 ± 219** (−210–219) | 84, −210, 219 | 28% |
| No memory (0c_blind) | 158 ± 42 (128–183) | 8 | 24% |

- **Neither beats no memory** (J1_chunk −20, Welch p = 0.68; J1_r2d2 −127, p = 0.42) **or whole flights** (−31, p = 0.53; −138, p = 0.39).
- **R2D2 is unstable:** seed 1 collapsed to −210 (12 of its 18 validation flights hit the ground); seed 2 completes 15 of 18.

| By starting battery (final validation) | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| Failed flights, true battery | 79% | 12% | 0% | 0% | 0% | 0% |
| Failed flights, J1 whole flights | 100% | 100% | 87% | 80% | 20% | 20% |
| Failed flights, J1_chunk | 100% | 100% | 67% | 67% | 33% | 0% |
| Failed flights, J1_r2d2 | 100% | 67% | 67% | 67% | 67% | 67% |
| Failed flights, no memory | 100% | 100% | 100% | 100% | 54% | 0% |
| Mean throttle, true battery | 0.10 | 0.35 | 0.47 | 0.59 | 0.69 | 0.78 |
| Mean throttle, J1 whole flights | 0.81 | 0.81 | 0.74 | 0.76 | 0.78 | 0.83 |
| Mean throttle, J1_chunk | 0.86 | 0.77 | 0.67 | 0.67 | 0.65 | 0.66 |
| Mean throttle, J1_r2d2 | 0.69 | 0.37 | 0.25 | 0.18 | 0.17 | 0.17 |
| Mean throttle, no memory | 1.00 | 1.00 | 1.00 | 0.96 | 0.89 | 0.87 |

R2D2's validation outcomes per seed: seed 0, 12 battery empty and 6 ground crashes; seed 1, 6 battery empty and 12 ground crashes; seed 2, 15 completed and 3 battery empty.

**Diagnostics** (validation flights only; `runs_step0/easy_e3_windows/diagnostics_validation.json`, `diagnostics_followup.json`; every run reproduces its saved final validation exactly):

| Run | With its estimate | With the true battery | With a fixed 0.45 | Error at hand-over (200 random flights) | Validation flights: at hand-over / after 20 s |
|---|---:|---:|---:|---:|---|
| J1_chunk s0 | 133 | 332 | 91 | 0.151 (corr 0.55) | 0.175 / 0.156 |
| J1_chunk s1 | 134 | 288 | 84 | 0.124 (corr 0.65) | 0.143 / 0.131 |
| J1_chunk s2 | 224 | 231 | 169 | 0.155 (corr 0.67) | 0.154 / 0.054 |
| J1_r2d2 s0 | 37 | 43 | 147 | 0.118 (corr 0.85) | 0.118 / 0.048 |
| J1_r2d2 s1 | −208 | −208 | −208 | 0.111 (corr 0.84) | 0.117 / 0.048 |
| J1_r2d2 s2 | 182 | 184 | 221 | 0.114 (corr 0.82) | 0.088 / 0.134 |
| *J1 whole flights (E3c), mean* | *186* | *247* | *79* | *0.052–0.058 (corr 0.93)* | |

- **Plain chunking made the encoder much worse:** 0.12–0.16 off at hand-over against 0.05 with whole flights, and its estimate correlates only 0.55–0.67 with the battery (0.93). From a blank memory every 16 steps it cannot learn to use the 100-step take-off.
- **But its policies learned to use the battery:** given the true battery they average 284 (332, 288, 231), even above the true-battery agent (254); with their own estimate, 164; with a fixed 0.45, 115. The poor estimate is what sinks them.
- **R2D2's encoder is better than plain chunking but still twice as far off as with whole flights** (0.11–0.12, correlation 0.82–0.85). The stored memory helps, but backpropagating only 16 steps doesn't teach the encoder to keep the take-off's information for a whole flight.
- **R2D2's policies don't use the estimate:** the true battery changes nothing (37 → 43, 182 → 184), and a fixed 0.45 scores higher for two seeds (147, 221). Seed 1 collapsed whatever its input, flying with very little throttle (0.17–0.37) and hitting the ground.

**What it means.**
- **Shorter training windows hurt the one thing that limited end-to-end learning here: the estimate's accuracy.** They make encoder training 3–4 times faster, but the encoder can no longer carry the take-off's information across the flight.
- **Whole-flight backpropagation stays the right way to train this encoder.** These flights are at most 800 steps, short enough to backpropagate through in full.
- **The policies can become excellent pilots** (J1_chunk's reach 284 with the true battery): what end-to-end learning still lacks is an accurate estimate. The next tests stay those of E3c: a 7-second take-off (best possible error 0.037 instead of 0.050) or a Gaussian head.

### Memory baselines on the take-off-only version (2026-10-23)

**Question.** E3c's J1 reads the take-off and still doesn't beat no memory. Do standard end-to-end memory learners do better when the take-off shows them the battery? The plan is in `PAPER_ASSESSMENT.md` ("Plan: Tier 1, item 1"); the hard version and the fused LSTM are in `ENCODER_PLAN.md` ("Standard memory baselines").

**Design.** Step 0's settings, take-off only (`warmup_steps=100`), seeds 0–7, minibatches of 256 and 1,024 (`memory_baselines.py`).
- **RecurrentPPO** (M1a_lstm, M1b_lstm; sb3-contrib 2.9.0, an LSTM of 256 for the actor and another for the critic, reading `[o_t, a_{t−1}]`, with the fused LSTM):
  - sb3-contrib has no prelude, so the take-off is replayed as the first 100 steps of every episode: the autopilot's observations and executed actions, the agent's actions ignored, reward 0, plus an extra autopilot input.
  - Those steps are masked out of every PPO loss (`RecurrentInput`, `MaskedRecurrentPPO`). So the LSTM reads what Contextual PPO's encoder reads, with gradients through the take-off.
  - Episodes are then 800 steps, so `run_step0.py` scales the budget by 800/700: 2,285,760 steps (2M mission steps), with validation every 57,144 steps (40 evaluations, the late score over the last 11).
  - The checks show that the mission after the replay is step-for-step the plane's own episode, and that the flight metrics leave the replayed steps out.
- **History windows** (M2a/M2b_win16, M2a/M2b_win64): plain PPO on the last 16 or 64 steps of `[o_t, a_{t−1}, valid]`. The window starts filled with the take-off from the reset info (`warmup_observations`, `warmup_actions`; never `warmup_batteries`).
- **Runs:**
  - launched from a worktree of branch `memory-baselines-easy`;
  - folders `runs_step0/memory_takeoff/` (the windows) and `runs_step0/memory_takeoff_fused/` (RecurrentPPO);
  - W&B groups end in `_warm100` (windows) and `_warm100_fused` (RecurrentPPO);
  - run times: windows 40–120 min, RecurrentPPO 228–230 min.

**Results** (validation late score, mean ± std across 8 seeds):
- outcome rates come from the final validation;
- "vs no memory" is Welch's t-test against this version's E1 run without memory, with the same minibatches;
- throttle spread is the range of the mean throttle over the six starting batteries.

| Agent (take-off-only version) | Score | vs no memory | Completed | Battery empty | Ground crash | Throttle spread |
|---|---:|---:|---:|---:|---:|---:|
| No memory, batch 256 (0a_blind) | 127 ± 48 | — | 25% | 75% | 0% | 0.22 |
| **RecurrentPPO, batch 256** | **23 ± 37** | −104 (p < 0.01) | 8% | 92% | 0% | 0.00 |
| Window 16, batch 256 | 73 ± 39 | −55 (p = 0.03) | 16% | 84% | 0% | 0.07 |
| Window 64, batch 256 | 46 ± 22 | −81 (p < 0.01) | 13% | 87% | 0% | 0.11 |
| No memory, batch 1,024 (0b_blind) | 177 ± 24 | — | 30% | 70% | 0% | 0.21 |
| **RecurrentPPO, batch 1,024** | **44 ± 12** | −133 (p < 0.01) | 17% | 83% | 0% | 0.00 |
| Window 16, batch 1,024 | 70 ± 37 | −107 (p < 0.01) | 9% | 91% | 0% | 0.01 |
| Window 64, batch 1,024 | 56 ± 25 | −122 (p < 0.01) | 19% | 81% | 0% | 0.05 |
| True battery, batch 256 / 1,024 (0a / 0b) | 288 / 245 | — | 83% / 81% | 10% / 12% | 6% / 6% | 1.39 / 1.23 |

| By starting battery (final validation) | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| Failed flights, true battery (batch 256) | 100% | 0% | 0% | 0% | 0% | 0% |
| Failed flights, no memory (batch 256) | 100% | 100% | 100% | 88% | 62% | 0% |
| Failed flights, RecurrentPPO (batch 256) | 100% | 100% | 100% | 100% | 100% | 50% |
| Failed flights, RecurrentPPO (batch 1,024) | 100% | 100% | 100% | 100% | 100% | 0% |
| Failed flights, window 64 (batch 1,024) | 100% | 100% | 100% | 88% | 88% | 12% |
| Mean throttle, true battery (batch 256) | −0.68 | 0.31 | 0.42 | 0.52 | 0.62 | 0.71 |
| Mean throttle, no memory (batch 256) | 1.00 | 1.00 | 0.99 | 0.92 | 0.82 | 0.78 |
| Mean throttle, RecurrentPPO (batch 256) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Mean throttle, window 64 (batch 1,024) | 1.00 | 1.00 | 1.00 | 0.96 | 0.95 | 0.95 |

**What it means.**
- **RecurrentPPO ignores the take-off.** Its LSTM reads the 5-second full-throttle take-off, which reveals the battery (best possible error 0.050 after it, section 3). Yet it flies at full throttle in all 16 runs, at every starting battery, and fails every flight from 0.2 to 0.6. It ends 104–133 points below no memory.
- **The agents without memory do better than RecurrentPPO here,** although they can't remember the take-off: they ease off the throttle at high batteries (0.78–0.82 at 0.6–0.7), where the hand-over speed hints at the battery, and complete every flight from 0.7.
- **The windows hold the take-off at hand-over but lose it after 0.8 or 3.2 s;** all four are worse than no memory (p ≤ 0.03).
- **Next to E3c:** Contextual PPO's supervised encoder (J1, 169 with 5 seeds, against 158 without memory) at least learns the battery (0.05 off at hand-over), and its policies use it. RecurrentPPO and the windows do neither.
- **Same answer as the hard version:** standard end-to-end memory RL doesn't learn to use the autogenic context, even when the take-off reveals it for free.
