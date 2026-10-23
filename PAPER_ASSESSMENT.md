Reread finished: README, GRID_PLAN, ENCODER_PLAN, EASY_ENV_WARMUP, the project plan, CLAUDE.md and the three side-environment notes. My estimate is close to the earlier one. The reread confirmed that the work is careful, but it also showed two gaps I hadn't weighted enough: the related work, and the objection that the only real unknown is the starting charge `b0`.

## Chances (my judgment, assuming a clean write-up; there is no draft yet, only the figures and the project page)

| Venue | Now | With additions | Additions that matter |
|---|---|---|---|
| Workshop (ICLR/ICML 2027 RL workshops, EWRL) | 65–75% | 80–90% | Tier 1 |
| TMLR | 40–50% | 65–75% | Tiers 1–2 |
| RLC 2027 | 15–25% | 35–45% | Tiers 1–2 |
| ICML / NeurIPS / ICLR main track | 5–10% | 15–25% | Tiers 1–3 |
| NeurIPS Datasets & Benchmarks | 5–10% | 25–35% | Tier 3 (more environments) and Tier 1 |

**Strengths:**
- **A named, open problem.** BatteryPlane is a concrete case of Biedenkapp's autogenic context. It breaks both assumptions of the allogenic methods: the context stays fixed, and any behaviour reveals it.
- **The environment passes its own gate for every learner.** Knowing the battery is worth 170–220 points.
- **Unusually careful method for RL.** Seeds are disjoint, the final test was used once and reproduced validation exactly, and there is an exact Bayes floor (the ideal estimator), plus swap tests, closed-loop checks and noisy-oracle controls.
- **A clear negative result with a mechanism.** End-to-end learning stays at the memoryless level on the final test: A scores 42 and J1 150, against 155 without memory. Yet a perfect estimate is enough (316 / 348), and the battery can be learned offline (80–86% of the gap). How the policy flies decides what can be known, so the encoder learns a clock.
- **A positive result.** The frozen teacher-trained GRU scores 330 / 319, level with the ideal estimate.

**Weaknesses that will cost acceptance:**
1. **No standard memory baseline.** There is no RecurrentPPO, no history window and no world model, and Contextual PPO's own `encoder_objective="rl"` path was never run. So "end-to-end fails" currently means only "Contextual PPO's supervised encoder fails". Any reviewer will raise this first.
2. **Related work leaves out the closest work.** The README cites only the allogenic papers. Reviewers will say:
   - the chicken-and-egg loop is known from meta-RL exploration (DREAM, VariBAD);
   - "spend to learn" is dual control / active sensing;
   - recurrent model-free baselines should be there (Ni et al. 2022, POPGym), and so should work on privileged information (asymmetric actor-critic, the imitation gap).
3. **"It's only `b0`."** The drain is deterministic ([battery_plane.py:17](battery_plane.py#L17)), so the starting charge is the only unknown. A sharp reviewer will call it a hidden constant with a reveal gated by actions, plus bookkeeping.
4. **The working pipeline needs exact physics** for the teacher, plus hand-designed pilot styles. So it is a diagnosis, not a method.
5. **One environment, and weak statistics.** There are 3–8 seeds, and several headline gaps are not significant (T1 vs. the true battery; easy-version J1, p = 0.27).

So the paper to write is "benchmark plus diagnosis", not "a method that works".

## Experiments to add, by priority

**Tier 1: needed before anything beyond a workshop (about a week, mostly server time)**
1. **Memory baselines** at step 0's settings, 8 seeds, on the hard version and the take-off-only version:
   - RecurrentPPO: sb3-contrib 2.9.0 is already in `carlos`, so nothing to install;
   - plain PPO on the last 16 or 64 `(o, a)` pairs;
   - Contextual PPO with `encoder_objective="rl"`;
   - one 10M-step run of the best end-to-end learner.

   This decides the headline. If RecurrentPPO also fails, the claim becomes general. If it works, the story changes, and you need to know that before writing.
2. **16 seeds for the headline rows**, reported with the interquartile mean and bootstrap CIs (rliable). This settles whether "the estimate beats the truth" (316 / 348 vs. 283 / 280) is real.
3. **Writing, no compute:**
   - the related work above;
   - a formal definition: context dynamics that depend on the action, observability that depends on the action, the value-of-context gap, and policy-dependent identifiability;
   - the three side-environment gates as a table.

**Tier 2: what moves TMLR and RLC (1–2 weeks)**

4. **Find where end-to-end breaks by handing over one piece at a time.** In my view this is the most informative new experiment.
   - **A passive noisy battery gauge** at every step, so the battery can be averaged without probing; `NoisyBattery` already exists. If end-to-end works here, the obstacle is that information depends on actions; if it still fails, the problem is the recurrent training itself.
   - **The drain so far as an input**, which leaves only `b0` to find.
   - **A hidden wind per episode** (allogenic) alongside the battery. The prediction is that the learner infers the wind but not the battery, which would show the paper's thesis in one figure.
5. **The pipeline without the physics teacher:**
   - **More pilot flights (2×, 4×, 8×) for the GRU trained on the true battery.** Fitting the true battery with mean-squared error converges to the same posterior mean the teacher provides; the teacher only removes label noise.
   - **A hindsight teacher:** a bidirectional GRU that reads whole flights, used as the causal GRU's target.
   - **Retraining the encoder on the PPO policies' own flights**, which fixes the overestimation behind the collapsed run.
   - **Pilots built without domain knowledge.**
6. **Teacher–student distillation** (DAgger) from observed-mode PPO into a memory student. This is the standard robotics recipe for privileged information, and it may hit the imitation gap because the expert never needs to probe.
7. **Finish the ladder the md files already plan:** J6, Contextual PPO started from T1, T2's uncertainty as an input, A's encoder frozen under plain PPO, the leak test, the clock-only control.
8. **Stochastic drain:** noise on the drain that grows with thrust. The agent then creates new uncertainty, and probing pays off in mid-flight too. This answers weakness 3.
9. **Why the estimate beats the truth:** show the true battery only from the point where the ideal estimator would know it, and rerun the in-flight-noise result (284 / 351 / 309 vs. 276 / 275 / 282) at 16 seeds.
10. **Out-of-distribution evaluation of the existing models:** starting charge 0.1–0.2 or 0.7–0.9, on new seeds (not 2000–2002).

**Tier 3: for a main track or Datasets & Benchmarks (weeks)**

11. **Two more purpose-built autogenic environments**, built to the four criteria in the Myo note and each checked with the same gate (2–4 days each):
    - a lander whose burned fuel changes its mass, or a motor whose hidden temperature lowers its torque;
    - a fast discrete one, such as a lamp whose oil runs down and flickers only when low.
12. **DreamerV3**, since DALI uses it. It will be slow on this CPU server.

**Path and caveat:**
- Write a workshop paper now, from the current results plus Tier 1; ICLR 2027 workshop deadlines are usually in early February.
- Add Tier 2 for TMLR (rolling) or RLC 2027 (usually late winter).
- Tier 3 only if you aim for NeurIPS 2027 in May.
- Several items need edits to main code: the window option and items 4, 6, 7, 8 and 9. Edits to `battery_plane.py` also change `Env_BatteryPlane` through the symlink. I'd ask before each.

Nothing is running and nothing was changed. I can save this as a plan file if you want to keep it.

---

## Plan: Tier 1, item 1 — memory baselines (2026-10-23)

**Question.** Contextual PPO with a learned encoder never beat a memoryless agent: A and J0–J5 on the hard version, J0, J1 and J5 on the easy versions. Is that a property of the problem, or of Contextual PPO's supervised encoder? Standard memory learners, trained end to end by the RL losses alone, answer it.

**Status (2026-10-23):** RecurrentPPO and the windows are done on both versions; see [Results](#results-2026-10-23) at the end. Their final test is done too. Not run yet: M3 (Contextual PPO with `encoder_objective=rl`) and the 10M-step stage.

### The learners

| Condition | Learner | Memory | What trains the memory | Minibatches |
|---|---|---|---|---|
| `M1a_lstm` / `M1b_lstm` | RecurrentPPO (sb3-contrib 2.9.0, already in `carlos`) | An LSTM of 256 for the actor and another for the critic (sb3-contrib's defaults), then 2 × 256 tanh | The PPO losses only, backpropagated through time | 256 / 1,024 transitions |
| `M2a_win16` / `M2b_win16`, `M2a_win64` / `M2b_win64` | Plain PPO on a window | The last 16 or 64 steps (0.8 s or 3.2 s) of `[o_t, a_{t−1}]` plus a padding flag, as one flat input | Nothing: the window is fixed | 256 / 1,024 transitions |
| `M3_rl` | Contextual PPO with `encoder_objective=rl` | J1's GRU (linear 6 → 128, 2 GRU layers × 128, head 2 × 128 tanh), with 16 free outputs instead of the battery | The PPO losses: critic loss + actor loss (weight 1, PPO's summed loss); no labels, no supervised steps | 4 per epoch, each 2 whole streams of 512 steps |

"a" and "b" match step 0's 0a and 0b (minibatches of 256 and 1,024); M3 has Contextual PPO's single scheme.

- **Why the previous action is an input:** sb3-contrib's LSTM reads only the observation. But the battery shows in how the plane responds to the agent's commands, and during training the commands are sampled with noise the memory can't know. Contextual PPO's encoder reads `[o_t, a_{t−1}]` too, so every memory learner sees the same history. A wrapper appends the executed (clipped) action, zero at the first step.
- **How RecurrentPPO trains its memory:** each minibatch is a run of consecutive steps of one environment (256) or two whole streams (1,024), cut at episode starts, and each piece starts from the memory saved at collection. So batch 256 backpropagates through at most 256 steps (12.8 s) and batch 1,024 through at most 512, while the battery has to be carried for 700 steps. Both are the standard method; running both shows whether the length matters.
- **Why 16 free outputs for M3:** the RL losses don't ask the output to be the battery, so a one-number bottleneck would be an arbitrary restriction. Ndir et al.'s jcpl also uses an embedding.

### Settings shared by every run (step 0's)

- **PPO:** Adam 3 × 10⁻⁴; 8 environments × 512 steps = 4,096 transitions per update; 10 epochs; γ 0.99; GAE λ 0.95; clip 0.2; entropy 0; value-loss coefficient 0.5; gradient clipping 0.5; advantages normalized. Actor and critic 2 × 256 tanh, orthogonal initialization, log std 0 at the start.
- **Budget and task:** reward scale 0.1 during training only (returns in environment units); 2M mission steps; hidden mode (4 numbers), speed noise σ 5 m/s.
- **Seeds and scoring:**
  - training seeds 0–7;
  - validation every 50,000 steps on the 18 flights of seeds 1000–1002 × starting battery 0.2–0.7, with deterministic actions;
  - score = the mean validation return over 1.5M–2M steps;
  - the final-test seeds 2000–2002 only at the very end, through `evaluate_final_test.py`.
- **Threads:** one per run (`--jobs 32 --threads 1`).

### The two versions

| | Hard | Take-off only |
|---|---|---|
| Branch | `main` | `easy-env`, launched from a git worktree (e.g. `/root/Bidirectional-Contextual-RL-easy`) so `main` stays checked out for the hard runs |
| Environment | `warmup_steps=0` | `warmup_steps=100`: 5 s at full throttle before the agent takes over; W&B names end in `_warm100` |
| Window learners | Empty window (padding) at the first step | Window filled at reset with the take-off's observations and actions from the reset info (`warmup_observations`, `warmup_actions`; `warmup_batteries` is never read) |
| RecurrentPPO | Starts from a zero memory | sb3-contrib has no prelude, so a wrapper replays the take-off as the first 100 steps of each episode (details below) |
| M3 (Contextual PPO, rl) | As it is | **Optional** (details below) |

**RecurrentPPO's take-off replay:**
- The 100 replayed steps carry the autopilot's observations and actions; the agent's actions are ignored and the reward is 0.
- An extra input says "autopilot", and these steps are masked out of every PPO loss (a small subclass of sb3-contrib's rollout buffer).
- So the LSTM reads exactly what Contextual PPO's encoder reads, with gradients through the take-off.
- Episodes are 800 steps long, so the budget is 2.29M environment steps (2M mission steps), with validation every ~57,000.

**Why M3 is optional on the take-off version:**
- Contextual PPO's preludes work only with the supervised encoder, because the PPO epochs reuse the contexts cached at collection.
- With `rl`, the preludes would have to be stored in the rollout buffer and re-encoded with gradient. That is a change to `contextual_ppo/` of about a day.
- RecurrentPPO already covers RL-trained memory on this version, so I would add M3 here only if it shows something on the hard version.

### Reference rows (already run, no new runs)

| Version and learner | No memory | True battery | With memory so far |
|---|---:|---:|---:|
| Hard, batch 256 | 23 | 246 | frozen T1 GRU: 291 |
| Hard, batch 1,024 | 77 | 258 | frozen T1 GRU: 262 |
| Hard, Contextual PPO | 115 | 282 | J1: 114 |
| Take-off only, batch 256 | 128 | 288 | — |
| Take-off only, batch 1,024 | 177 | 245 | — |
| Take-off only, Contextual PPO | 158 | 254 | J1: 169 (5 seeds) |

These are validation late scores, 8 seeds unless noted. M1a and M2a are compared with the batch-256 rows, M1b and M2b with the batch-1,024 rows, and M3 with Contextual PPO's rows.

### Decision rules (fixed before the runs)

- **Uses its memory:** the late validation score is above the matching no-memory control, with Welch's t-test p < 0.05 over 8 seeds each. Reported with the share of the gap closed: (score − no memory) / (true battery − no memory).
- **On the take-off version, also judged on survival:** flights completed at starting batteries 0.3–0.5. Returns there mix the faster start with knowing the battery (E1's revised verdict).

| Result | What it means for the paper |
|---|---|
| Every learner stays at the no-memory level on both versions | The headline generalizes: standard end-to-end memory RL fails on autogenic context, not just Contextual PPO's supervised encoder |
| Above no memory on the take-off version only | End-to-end learning works when the battery is shown for free, and fails when the agent has to pay to see it: the identifiability loop is the obstacle |
| Above no memory on the hard version | The supervised encoder was the problem. The story changes, and that has to be known before writing; the memory probe below then says why |
| A window of 64 works and one of 16 doesn't | The time scale the battery needs |

### Diagnostics after the runs (no training)

- **Memory probe.** Does the memory hold the battery even when the policy ignores it?
  - **Method:** a ridge regression from the memory (the LSTM states, M3's GRU state, the window itself) to the true battery. It is fit on half of the flights, scored on the other half, and shown next to the clock-only and ideal estimates.
  - **On each run's own validation flights** (closed loop).
  - **On the 1,900 pilot test flights of dataset v2** (open loop).
  - **What it tells:** "learned the battery but doesn't use it" apart from "never learned it". The true battery is used only as the probe's label.
- **Throttle and failed flights by starting battery**, from the final validation logs, as in E3c.

### The 10M-step stage

- **Why:** reviewers will ask whether 2M steps is simply too short.
- **Which learner:** per version, the end-to-end learner with the best late validation score among M1a, M1b, M3 and J1 (already run). Chosen on validation only.
- **Runs:** 8 seeds × 10M steps; validation every 200,000 steps (50 evaluations); score = the mean over 7.5M–10M.

### Final test

Once, after the 10M stage, for every new run (`evaluate_final_test.py --reports`). Each run first reproduces its validation exactly. Nothing is chosen on it.

### Code to write

| File | Change | |
|---|---|---|
| `memory_baselines.py` | The previous-action, window and take-off replay wrappers; the masked rollout buffer for RecurrentPPO | new |
| `ppo/main_battery_plane_ppo.py` | `ppo.algorithm: ppo \| recurrent_ppo`, `ppo.lstm_hidden_size`, `memory.window`; the same wrappers on the training, validation and final-validation environments; checks of the new options | edit |
| `ppo/configs/experiment/M1*_lstm_*.yaml`, `M2*_win*.yaml`; `config/memory/M3_rl.yaml` | The seven conditions | new |
| `run_step0.py` | The seven conditions; RecurrentPPO's take-off budget | edit |
| `report_step0.py`, `evaluate_final_test.py` | The new run kinds; load RecurrentPPO and rebuild the wrappers from each run's `config.yaml` | edit |
| `flight_logging.py` | Leave the replayed take-off steps out of the flight metrics (a flag in the info) | edit; also reaches `Env_BatteryPlane` through the symlink, harmless there |
| `memory_probe.py` | The memory probe | new |
| `checks/memory_baseline_checks.py` | The checks below | new |
| `contextual_ppo/` (optional) | Preludes with `encoder_objective=rl` | edit, `easy-env` only |
| `CLAUDE.md` | The new commands | edit |

Written on `main`, then merged into `easy-env` like the other shared files.

### Checks before any training

1. **No leak:** every learner reads only the hidden-mode observation, the executed actions and, on the take-off version, the take-off's observations and actions. Never `b`, `b0`, `cap`, `saturated` or `warmup_batteries`.
2. **Previous action:** the executed (clipped) action; zero at the first step; the autopilot's last action at hand-over.
3. **Window:** holds exactly the last k frames in order (checked against a slow reference), is cleared at reset, and is filled correctly from the take-off.
4. **Take-off replay:**
   - its observations equal the reset info's, and the hand-over observation equals the environment's own;
   - the agent's actions have no effect during it, and its rewards are 0;
   - the mission that follows is identical to the easy environment's own episode for the same seed and actions;
   - masked steps contribute nothing to any loss (gradient test).
5. **Seeds:** the training start seeds of the new learners stay clear of 1000–1002 and 2000–2002.
6. **Reproducible evaluation:** a reloaded model reproduces its final validation exactly, with the LSTM memory carried through each flight (the precondition of `evaluate_final_test.py`).
7. **M3:** the PPO losses change the encoder, no supervised step runs, and no label is read.
8. **Smoke runs** of every condition on both versions (`run_step0.py --smoke`).

### Compute (timed 2026-10-23 on this server, 1 thread per run, 3 updates each)

| Learner | Per update | Per 2M-step run, alone | Under full load (estimate) |
|---|---:|---:|---:|
| RecurrentPPO, batch 256 | 14.2 s | 1.9 h | ~3 h |
| RecurrentPPO, batch 1,024 | 12.8 s | 1.7 h | ~3 h |
| Contextual PPO, rl (GRU) | ~13.5 s | ~1.8 h | ~3 h |
| Plain PPO on a window | as plain PPO | — | 15–25 min (E1: 11–21 min at 32 side by side) |

- **2M-step runs:** 56 on the hard version (16 RecurrentPPO, 32 window, 8 M3) and 48 on the take-off version (+8 if M3 is added). On 32 slots, about 6–7 h for both versions together.
- **10M-step stage:** 16 runs (8 seeds × 2 versions). About 10–15 h each with RecurrentPPO or M3, about 20 h with J1: one overnight batch plus a day.
- **Final test:** about 1 h.

### Order of work

1. Code and checks on both branches, and the smoke runs: about 1.5 days, +1 day for M3 on the take-off version.
2. The 2M-step runs on both versions together: about 7 h.
3. Reports, decision rules and the memory probe: about half a day.
4. The 10M-step stage: about a night and a day.
5. The final test; results into `ENCODER_PLAN.md`, `EASY_ENV_WARMUP.md`, the README table and this file.

About 4–5 working days in all.

### Follow-ups if every learner fails (not in this plan)

- **A small RecurrentPPO sweep,** so that "it wasn't tuned" can't be said (Ni et al. 2022's point): LSTM 64 or 256, learning rate 1e-4 or 3e-4, rollouts of 1,024 steps; 4 seeds each on the hard version, chosen on validation.
- **Other memory families:** recurrent off-policy (SAC with a recurrent critic), transformers, DreamerV3.

### Results (2026-10-23)

**What was run:** M1a/M1b_lstm (RecurrentPPO) and M2a/M2b_win16/win64 (windows), 8 seeds each, on both versions: 96 runs.
- **M3 and the 10M-step stage** were not run.
- **Final test:** done (2026-10-23), see below. `evaluate_final_test.py` now loads RecurrentPPO runs. The take-off version's plain-PPO controls (0a, 0b, 0a_blind, 0b_blind) were final-tested for the first time with them.
- **Details:** `ENCODER_PLAN.md` ("Standard memory baselines", hard version) and `EASY_ENV_WARMUP.md` ("Memory baselines on the take-off-only version").

**Changes from the plan:**
- **The fused LSTM.** sb3-contrib's RecurrentPPO runs its LSTM one step at a time in Python whenever a training sequence holds an episode start. With 32 runs side by side, that made the first RecurrentPPO runs about 15 times slower than timed: after 9–10 hours they were at 0.1–0.57M of 2M steps.
  - They were stopped and rerun with `memory_baselines.FusedLstmPolicy`, which resets the memory at each sequence's first step and runs the sequence in one call.
  - It gives identical outputs and gradients equal up to rounding. A rerun with sb3-contrib's loop reproduces the stopped runs exactly; the fused runs take their own path after 2–4 updates, as any rounding change does.
  - Over all seeds the fused and stopped runs don't differ at the same steps (p = 0.30–1.00).
  - `memory.fused_lstm=false` restores sb3-contrib's loop.
- **Run times** with 32 runs side by side: windows 40–120 min (planned 15–25), RecurrentPPO 3.5–4 h with the fused LSTM.

**Results** (validation late score, mean ± std across 8 seeds; "vs no memory" is Welch's t-test against the no-memory run with the same minibatches; completed and battery empty are from the final validation):

| Version | Agent | Batch 256 | vs no memory | Batch 1,024 | vs no memory | Completed | Battery empty |
|---|---|---:|---:|---:|---:|---:|---:|
| Hard | No memory | 23 ± 41 | — | 77 ± 32 | — | 26% / 15% | 74% / 85% |
| Hard | **RecurrentPPO** | **21 ± 65** | −2 (p = 0.94) | **−7 ± 18** | −84 (p < 0.01) | 8% / 17% | 92% / 83% |
| Hard | Window 16 | 30 ± 43 | +7 (p = 0.77) | 21 ± 24 | −56 (p < 0.01) | 15% / 13% | 81% / 87% |
| Hard | Window 64 | −7 ± 27 | −30 (p = 0.13) | −56 ± 93 | −134 (p = 0.01) | 15% / 10% | 81% / 85% |
| Hard | True battery | 246 ± 52 | | 258 ± 30 | | 88% / 86% | 6% / 5% |
| Take-off only | No memory | 127 ± 48 | — | 177 ± 24 | — | 25% / 30% | 75% / 70% |
| Take-off only | **RecurrentPPO** | **23 ± 37** | −104 (p < 0.01) | **44 ± 12** | −133 (p < 0.01) | 8% / 17% | 92% / 83% |
| Take-off only | Window 16 | 73 ± 39 | −55 (p = 0.03) | 70 ± 37 | −107 (p < 0.01) | 16% / 9% | 84% / 91% |
| Take-off only | Window 64 | 46 ± 22 | −81 (p < 0.01) | 56 ± 25 | −122 (p < 0.01) | 13% / 19% | 87% / 81% |
| Take-off only | True battery | 288 ± 25 | | 245 ± 33 | | 83% / 81% | 10% / 12% |

**Final test** (seeds 2000–2002, flown once; every run first reproduced its validation exactly; mean ± std across 8 seeds; Welch's t-test against no memory):

| Version | Agent | Batch 256 | vs no memory | Batch 1,024 | vs no memory | Completed |
|---|---|---:|---:|---:|---:|---:|
| Hard | No memory | 49 ± 45 | — | 118 ± 65 | — | 25% / 17% |
| Hard | **RecurrentPPO** | **66 ± 81** | +17 (p = 0.61) | **14 ± 1** | −104 (p < 0.01) | 9% / 17% |
| Hard | Window 16 | 61 ± 58 | +12 (p = 0.65) | 38 ± 43 | −80 (p = 0.01) | 17% / 15% |
| Hard | Window 64 | 5 ± 51 | −44 (p = 0.09) | −37 ± 108 | −155 (p < 0.01) | 12% / 11% |
| Hard | True battery | 283 ± 66 | | 280 ± 36 | | 90% / 92% |
| Take-off only | No memory | 139 ± 53 | — | 203 ± 39 | — | 26% / 31% |
| Take-off only | **RecurrentPPO** | **23 ± 31** | −116 (p < 0.01) | **67 ± 45** | −137 (p < 0.01) | 8% / 17% |
| Take-off only | Window 16 | 94 ± 53 | −45 (p = 0.11) | 72 ± 59 | −131 (p < 0.01) | 16% / 10% |
| Take-off only | Window 64 | 64 ± 57 | −76 (p = 0.02) | 78 ± 45 | −125 (p < 0.01) | 13% / 20% |
| Take-off only | True battery | 279 ± 28 | | 242 ± 41 | | 83% / 83% |

**The final test confirms validation.** No memory learner beats no memory, and 8 of the 12 comparisons are significantly worse. RecurrentPPO's +17 on the hard version at batch 256 is not significant, and it completes 9% of flights against 25%.

**By the decision rules: the headline generalizes.**
- **No memory learner beats its no-memory control** on either version. 9 of the 12 comparisons are significantly worse: all at batch 1,024 on the hard version and all on the take-off version. The three at batch 256 on the hard version are level.
- **RecurrentPPO flies at full throttle at every starting battery** (mean throttle 1.00 in 30 of its 32 runs, 0.97–0.98 in the other two), even when its LSTM reads the take-off that reveals the battery. Its memory changes nothing it does.
- **The windows fly the same at every starting battery too.** Their extra inputs make them worse than no memory.
- **Together with Contextual PPO's own results** (A, J0–J5, E3–E3d): every end-to-end memory learner tried stays at or below the no-memory level, while an encoder trained offline and frozen reaches the ideal estimate's level. Weakness 1 above is answered for RecurrentPPO and windows. A reviewer can still ask for tuning: the follow-up sweep is below.

**Next, if wanted:**
- the small RecurrentPPO sweep;
- M3 (Contextual PPO with `encoder_objective=rl`);
- the 10M-step stage.

## Out-of-distribution evaluation (Tier 2, item 10; 2026-10-23)

**Done on the hard version.** 120 frozen agents: the README's table plus the memory baselines. Nothing trains.
- **Settings:**
  - shifted speed noise (0, 10);
  - battery capacity (85, 120 hover-seconds);
  - motor thrust cap (±1 m/s²);
  - a starting-battery sweep 0.10–0.19 and 0.71–1.00.
- **Seeds:** 3000–3004, never the validation or final-test seeds.
- **Fixed estimates keep the training physics.** Every run first reproduced its validation exactly.
- **Details and full tables:** `ENCODER_PLAN.md` ("Out-of-distribution evaluation").

**Main results:**
- **The true battery generalizes:** Contextual PPO and plain PPO with it lose 46–61 points on a battery that drains 18% faster and still complete 83–86% of flights.
- **The end-to-end learners don't adapt.** That is Contextual PPO's A and J1, RecurrentPPO and no memory: a faster-draining battery collapses them to 0–6% of flights completed, and below a starting battery of 0.2 they save no flight.
- **The decoupled method's weakness:** T1 (305 in distribution) falls to 101–130 with a 1 m/s² stronger motor, completing 5–15% of flights. Its encoder learned the training physics. The ideal-estimate agents lose 38–40% of their flights there too, and 116–138 points with a weaker motor.
- **Starting battery is a weak axis, as expected:** no cliff at the range's edges, nearly nobody saved below 0.15, nearly everyone completes above 0.7.

**For the paper:** an honest limitation of the decoupled method (frozen encoders trained under one physics), and a natural next experiment: train the encoder on varied physics (capacity and motor drawn per flight) and repeat this evaluation.
