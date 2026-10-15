# Planned experiment: pretrained frozen encoder vs. policy

**Status:** step 0 and step 1's noisy oracle, fly-with-noise, late-switch, early-noise and in-flight-noise tests done, and the A rerun's encoder error measured over a flight (2026-10-13), the memoryless controls at step 0's settings, the ideal estimator on the validation flights, the context wrapper, and plain PPO with the ideal estimate (2026-10-14; results below), the pilots and their flights (step 3: dataset v2 with style switching, 2026-10-15), and offline training of the encoder on those flights (step 4: E1 and E3 running since 2026-10-15) with its floor measured (results below); diagnostic probes of the encoder, then freezing it under plain PPO, next.

**Question:** in hidden mode, is the bottleneck the context encoder or the policy?

**Idea:** train the encoder offline on BatteryPlane flights from a range of pilots. Freeze it, then train PPO on top of it. Compare against controls that give the policy a perfect, a degraded, or a clock-only battery estimate.

## Background

These are the results from the A–F grid (W&B `duck13/cppo`) and the vanilla PPO baselines (`duck13/battery-plane-ppo`) that motivate this plan. All returns are on the deterministic, final-checkpoint, 18-episode validation grid.

- **The context isn't helping.** Most CPPO variants score at or below PPO with no memory (52.7). Only C (87.6) is above it. Observed-mode PPO, the fair upper bound, scores 212 at its final checkpoint and peaked around 360 near 600k steps.
- **Most runs settled on one bad policy.** 10 of 18 CPPO runs give returns that are bit-identical across battery levels. They fly at full throttle, at low altitude, and empty the battery in 83% of episodes. Their policies ignore the battery.
- **The encoder learns only part of the battery.** On those runs' flights, a clock-only estimate already gets pooled R² 0.45. The encoders reach 0.83–0.87, closing roughly 70–77% of the gap to a perfect estimate. In the runs that throttle back, the closed-loop R² is negative. Pooled R² alone overstated the encoder.
- **The earlier offline encoder test (`enc1`)** found that the xLSTM closes 52–68% of the prior-to-ideal error gap and the LSTM 28–33%. Training on shaky flights only gave 0–37%. One second into a flight, the estimate is no better than the prior.

## Plan

A frozen encoder gives PPO a fixed input that no longer shifts while the policy learns, so you can test the two halves separately. But on its own it can't settle "encoder or policy." If PPO fails on top of a frozen encoder, you still won't know whether the encoder was too weak or PPO can't use any battery estimate. Adding an oracle control fixes that.

### 0. First check CPPO itself: give it the true battery

Before building better encoders, ask: **can CPPO learn when battery estimation is perfect?** Replace the learned context with the true current battery for both the actor and critic. Bypass the encoder and disable its training; do not give the networks an additional learned estimate alongside the true battery.

Observed-mode plain PPO (p04) already provides encouraging evidence: 212 final return and about 360 at its peak. Repeat it alongside the new conditions under the same scoring rule. Those historical scores are reference points, not guaranteed reproduction targets: p04 used unscaled rewards, step 0 scales them (see the critic settings below).

**CPPO and plain PPO train the networks differently.** Both collect 4,096 transitions per rollout (8 environments × 512 steps), and both use each transition 10 times. They split those transitions into minibatches differently:

| | Minibatch | Optimizer steps per rollout | Optimizer steps over 2M transitions |
|---|---|---:|---:|
| Plain PPO (p01–p04) | 256 transitions, shuffled from all 8 environments | 160 | ≈ 78k |
| CPPO (A–F) | 1,024 transitions: two complete 512-step stretches, each from one environment | 40 | ≈ 19.5k |

The learning rate is the same, so CPPO makes a quarter as many weight updates, but not a quarter as many passes over the data. Its minibatches contain two environment streams, potentially spanning episode boundaries; plain PPO shuffles individual transitions across all eight streams.

**Primary runs: three conditions × three seeds = nine fresh runs.** All use 8 environments × 512 steps, 10 epochs, 2M requested environment transitions (rounded to complete rollouts), and training seeds 0, 1, 2.

**Critic settings, the same for all three conditions.** The learners train on rewards × `reward_scale` (`cppo.reward_scale` / `ppo.reward_scale`), because the −300 crash penalty made the critic's error explode. A wrapper outside each training env's `Monitor` applies the scale, so episode returns, evaluation and every score stay in environment units. The discount is 0.99. Both were chosen by the screens in the Step 0 results below. A–F and p01–p04 used unscaled rewards and 0.99.

Seeds, the same for all three conditions:
- Training env `i` starts from reset seed `seed + i`, so 0–9. SB3 re-seeds the envs when learning starts. The same seed in 0a, 0b and 0c starts from the same first flights.
- Validation uses seeds 1000–1002. The final-test seeds 2000–2002 stay unused.
- Both scripts reject training seeds that reach the held-out seeds, and `checks.grid_checks` verifies the real start seeds.
- Neighbouring training seeds share their first flight in 7 of the 8 envs, then diverge. This is negligible, and it is kept for comparability with A–F and p04.

| Run | Learner and battery input | Minibatch | Optimizer steps per rollout | Purpose |
|---|---|---|---:|---|
| 0a | Plain PPO, true battery (`observed` mode) | 256 shuffled transitions | 160 | The original baseline's minibatch, with step 0's critic settings |
| 0b | Plain PPO, true battery (`observed` mode) | 1,024 shuffled transitions | 40 | Match CPPO's batch size and optimizer-step count |
| 0c | CPPO, true battery replacing the encoder | 1,024 transitions (2 streams × 512 steps) | 40 | Test CPPO without estimation error or encoder drift |

- Every actor and critic receives the same five values: the four hidden-mode observations plus the true current battery, in the same order and scale. Speed remains noisy. There is no predicted mean, variance, or other encoder feature in 0c.
- Keep the environment, reward and reward scale, actor/critic architecture, learning rate, discount, GAE settings, and other PPO settings identical. The intended differences are the learner and the minibatch settings shown above.
- Compare 0b with 0c to investigate differences beyond batch size and optimizer-step count, including shuffled transitions versus complete streams. Compare 0a with 0b to investigate the batch-size/update-count setting within plain PPO; these two effects are not separated by this comparison.
- The earlier 32-environment × 128-step condition is deferred. It changes trajectory collection and GAE bootstrapping as well as minibatches, so success would not establish that CPPO "only needed more updates."

**The 0c oracle path (implemented 2026-10-12).** Setting only `env.observation_mode=observed` would leave the encoder active and append its predictions to the visible battery, which is a different experiment. 0c instead uses the Hydra option `cppo.context_source=oracle`, set by `config/step0/cppo_oracle.yaml`:
- `step0_oracle.py` replaces the xLSTM with `TrueBatteryEncoder`, which has no parameters and no memory and outputs `z_t = b_t` from the observation. The actor and critic read the observation only, because observed mode already ends with `b_t`. This holds in rollout collection, PPO minibatch evaluation, value prediction and evaluation, because all of them go through the same policy methods.
- `encoder_objective=rl` with no encoder parameters means there is no episode buffer, no supervised encoder step, and nothing for the routed PPO losses to update. The speed head stays off.
- CPPO's collector, PPO losses and whole-stream minibatching are unchanged. The default `cppo.context_source=encoder` leaves A–F unchanged; a missing field in an older saved config also means `encoder` during validation and model construction.
- `validate_config` requires `observed` mode for the oracle. It rejects `full` (exact speed), hidden-battery modes, a supervised objective, the speed head, the Gaussian head or `context_dim` ≠ 1.
- The inherited W&B prefix is retained, and the name carries the discount: with discount 0.99 and reward scale 0.01, seed 0 is named `cppo_S0c_oracle_gamma0.99_rs0.01_s0`, in group `cppo_S0c_oracle_gamma0.99_rs0.01`.

`python -m unittest checks.step0_checks -v` covers the pre-training checks:
- The config differs from the base only in the battery input.
- The actor and critic inputs equal plain PPO's observed-mode features on the same observations. With the oracle's weights loaded, plain PPO gives the same values and actions, and the previous action has no effect.
- In rollouts with resets and terminal transitions, the observation's last column equals the true `b` from the env infos, for both `o_t` and its successor.
- The encoder group is empty, so only the actor and critic change.
- A saved model reloads with the oracle policy.
- The oracle rejects exact-speed `full` mode. `checks.grid_checks` also verifies that older A–F configs without `context_source` still validate and build an encoder policy.

A full-size smoke run showed `encoder: 0` parameters and actor/critic parameter counts identical to plain PPO's (67,844 / 67,585). A training round took about 0.2 s, so a 2M-step run should take minutes, not the 5.5–7 h of A–F. If a 0c run takes hours, the encoder is running and something is wrong.

```bash
# All nine runs at once, one process per condition and seed; logs and runs.json in runs_step0/<time>/
nohup python run_step0.py > run_step0.log 2>&1 &
python run_step0.py --conditions 0a 0b      # a subset

# Or a single run
python main_battery_plane_contextual_ppo.py +step0=cppo_oracle training.seed=0          # 0c
python ppo/main_battery_plane_ppo.py +experiment=S0a_observed_bs256 training.seed=0      # 0a
python ppo/main_battery_plane_ppo.py +experiment=S0b_observed_bs1024 training.seed=0     # 0b
```

`run_step0.py` starts up to one run per physical core with 2 threads each. A run is mostly single-threaded Python: on the 16-core Ryzen 9 9950X, 1 → 4 threads made a run only ~15% faster, while nine runs side by side took ~1.25× the wall time of one run alone (measured 2026-10-12).

The plain-PPO script used for 0a and 0b now lives in this repo under `ppo/`. It was copied from Env_BatteryPlane, with its p01–p04 configs, so a `git pull` sets up any server. It writes to `runs_ppo/<run id>/`, including the resolved `config.yaml`. `checks.step0_checks` verifies that 0a, 0b and 0c share the environment, evaluation grid, budget and PPO settings, and differ only in learner and minibatch size.

**Evaluation, fixed before starting.** Evaluate deterministically every 50k transitions on validation seeds 1000–1002 × starting battery 0.2–0.7, with speed noise 5 and the same seeded initial states. For each training seed, the primary score is the mean of scheduled validation returns from 1.5M through 2M transitions. Report all three seed scores and their mean/spread, together with the full learning curves, final return, completion, ground-crash, and battery-empty rates. Keep final-test seeds 2000–2002 untouched. Do not compare this late-training score directly with the historical final-only score of 212.

**Logging (identical for 0a, 0b and 0c).** Both training scripts (`main_battery_plane_contextual_ppo.py` and `ppo/main_battery_plane_ppo.py`) use `flight_logging.py`. All nine runs log to the `cppo` W&B project, next to A–F. Metric names follow A–F where those exist.
- **Every evaluation (50k):** `validation/*`, at the same step as `eval/mean_reward`:
  - Return, and completion, ground-crash and battery-empty rates.
  - Time in the reward band and at goal speed, time at the speed cap, speed, distance, battery used and left on completion.
  - Altitude (mean and lowest), throttle and lift, and action size and change.
  - `by_b0/b0_<pct>pct/*`: return, outcomes, behaviour, throttle and lift for each starting battery.
  - `by_b0/throttle_mean_range` and `lift_mean_range`. A value of 0 means the policy flies the same whatever its battery.
- **Every rollout:** `rollout/*`, the same outcomes and behaviour over the last 100 training flights. Also `train/optimizer_steps`, `train/grad_norm`, `train/grad_norm_max` and `train/grad_clipped_fraction`: the actor and critic gradient before clipping, which compares 40 large updates with 160 small ones directly.
- **Every 100k steps and at the end:** `episode_plots/validation` and `episode_plots/final_validation`, showing altitude, speed, battery and actions of the first validation flight at b0 = 0.2, 0.5 and 0.7.
- **At the end:** `final_validation/*` holds the same metrics for the final checkpoint. CPPO's `final_validation.json` also stores them under `flight_metrics`.
- Each metric is derived from the environment's own definitions (see the `flight_logging.py` docstring). They were written without the server code that produced A–F, so a few A–F metrics (for example `mean_abs_action`) may be defined slightly differently.

**How to read the results.** These are diagnostic patterns, not proofs from a single run. If the seed results are mixed, add seeds before changing the pipeline.

- **All three learn useful policies:** CPPO can exploit true battery. Investigate the learned encoder and its online interaction with the policy next; this does not yet prove which caused the A–F failures.
- **0b and 0c struggle while 0a succeeds:** the larger minibatches/fewer optimizer steps become a leading suspect. Test that setting further before expanding encoder training.
- **0c struggles while 0a and 0b succeed:** investigate remaining CPPO differences, including whole-stream minibatching and the collector. This is not automatically evidence of a code bug.
- **All three struggle:** first check reproduction of the original p04 setup and evaluation before drawing conclusions about encoders.
- **Other patterns:** inspect the per-seed curves and implementation checks; do not force them into an update-count or encoder explanation.

Review Step 0 before launching steps 1–5. No pilot dataset or offline encoder training is needed for this screen.

#### Step 0 results (2026-10-12/13)

Scores are the mean validation return from 1.5M to 2M steps, averaged over seeds 0–2. "Completed" is the share of validation flights finished at the final checkpoint. W&B groups in `duck13/cppo` carry the settings, e.g. `S0a_observed_bs256_gamma0.99_rs0.1`. The discount screens ran before this naming: no suffix for 0.999, then `_gamma0995`, `_gamma09` and `_gamma095`.

**CPPO can use the true battery.** In every screen below, 0c scored within the seed spread of plain PPO (0a, 0b). At discount 0.99 with reward scale 0.01 or 0.1, it had the best average of the three.

**Discount (reward scale 0.01).**

| Discount | 0a | 0b | 0c | Completed | 20%-battery flights failed |
|---|---:|---:|---:|---:|---:|
| 0.9 | 242 | 230 | 289 | 65–72% | 27 of 27 |
| 0.95 | 296 | 197 | 299 | 67–76% | 27 of 27 |
| **0.99** | 226 | 213 | 249 | **80–91%** | **19 of 27** |
| 0.995 | 160 | 114 | 131 | 67–89% | 20 of 27 |
| 0.999 | 169 | 148 | 130 | 74–85% | 22 of 27 |

A shorter look-ahead gives bolder, faster flying: more points and more time in the reward band, but more crashes. 0.995 and 0.999 trained worst, and with 0.999 CPPO stopped using the battery. **Chosen: 0.99**, which balances return and failed flights.

**Reward scale (discount 0.99).**

| Reward scale | 0a | 0b | 0c | Runs collapsed (score < 100) | Completed |
|---|---:|---:|---:|---:|---:|
| 1 (unscaled) | 265 | 240 | 210 | 2 of 9 | 67–85% |
| **0.1** | 276 | 275 | 282 | **0 of 9** | **85–89%** |
| 0.01 | 226 | 213 | 249 | 0 of 9 | 80–91% |
| 0.005 | 212 | 245 | 208 | 1 of 9 | 67–83% |
| 0.001 | 232 | 305 | 249 | 0 of 9 | 78–85% |

The critic predicts returns well at every scale (95–100% explained variance). The scale changes the size of the training steps instead: unscaled, every update's gradient was 26–700 times the clipping limit, which likely explains the collapses. 0.1 looked best on three seeds, so a confirmation added seeds 3–7 for 0.1 and 0.01, like `sb3_confirm1` (W&B groups ending in `_confirm1`; `python run_step0.py --reward-scales 0.1 0.01 --seeds 3 4 5 6 7 --study confirm1`).

**Confirmation, eight seeds (0–7).**

| Reward scale | 0a | 0b | 0c | All 24 runs | Completed | 30%-battery flights failed |
|---|---:|---:|---:|---:|---:|---:|
| **0.1** | 246 | 258 | 282 | **262** | **87%** | **1 of 72** |
| 0.01 | 213 | 243 | 265 | 240 | 84% | 14 of 72 |

0.1 scored higher in all three conditions and beat 0.01 on 16 of 24 same-seed pairs, by 22 points on average. The score gain is small and not certain: 16 of 24 would happen by chance about 8% of the time, and on seeds 3–7 alone the lead shrank from about 50 to 6 points. The clearer gain is safety: fewer failed 30%-battery flights and better weakest runs. 20%-battery flights failed equally often (56 of 72), and no run collapsed. **Chosen: reward scale 0.1.**

**Plain PPO with and without the battery (`sb3_screen1`, `sb3_confirm1`, 2026-10-08/09).** Plain SB3 PPO with 0b's settings (batch 1024), run from a script that is not in this repo, in hidden mode (no battery, no memory) and observed mode. Eight seeds (0–7) per row.

| Discount, reward scale | Hidden: return (completed) | Observed: return (completed) |
|---|---:|---:|
| 0.99, 0.01 | 36 (15%) | 247 (86%) |
| 0.99, 1 | 16 (22%) | 276 (81%) |
| 0.995, 0.01 | 100 (18%) | 108 (72%) |
| 0.999, 0.01 | 77 (29%) | 161 (89%) |

Blind agents run out of battery in 70–82% of flights and never finish a flight that starts at 20% or 30% battery. The observed runs repeat 0b at the same settings (247 vs 213, 276 vs 240, 108 vs 114, 161 vs 148) with the same patterns: 0.995 is the worst discount, and unscaled rewards occasionally collapse (1 of 8 runs).

**Experiment A rerun (discount 0.99, reward scale 0.1, 2M steps, seeds 0–2).** Config `config/rerun/A_point_baseline.yaml`, W&B group `cppo_A_point_baseline_gamma0.99_rs0.1`.

| | Seeds 0 / 1 / 2 | Average | Completed |
|---|---|---:|---:|
| A rerun, reward scale 0.1 | 106 / −3 / −37 | 22 | 0–33% |
| Original A, unscaled | −52 / 14 / −14 | −17 | 0–17% |
| CPPO with the true battery (0c) | 361 / 272 / 212 | 282 | 87% |

The new settings did not rescue A. It stays at the blind level, and the planes fly like blind agents: fast (36–47% of the time in the reward band), but they run out of battery in 67–100% of flights, and seed 0 flies the same at every battery. The encoder's battery estimate reaches R² 0.80 (typical error about 0.066, the noisy-oracle level of step 1), yet the policy gains almost nothing from it.

**Summary.**
- **CPPO's training works with a perfect battery.** 0c matches plain PPO with the battery: 249, against 247 for `sb3` observed at 0.99 and 0.01.
- **Knowing the battery is worth about 200 points.** At 0.99 and 0.01, plain PPO scores 247 with it and 36 without it. That gap is what an encoder can win back.
- **So the A–F failure lies in the encoder, or in training it together with the policy.** A–F's encoder runs scored around 50, near the blind level. The A rerun with the new settings (22) confirms it: A's unscaled rewards were not the cause.
- **Settings from here on:** discount 0.99 and reward scale 0.1 (confirmed on 8 seeds). The blind agent was measured at 0.99 and 0.01 (36); at 0.1 see the memoryless results below (77 with batch 1024, 115 with CPPO's learner). A blind agent reaches about 100 with discount 0.995.
- **Low-battery flights fail for every learner and setting**, even with the true battery. This is not a CPPO problem.
- **Runs are exactly reproducible.** 0c with unscaled rewards repeated its earlier scores (333, 240, 58), so the spread between seeds is the real noise.

**Memoryless agent with step 0's settings (2026-10-14).** The A rerun (22, 17% of flights completed, all failures empty batteries) looks like the memoryless agent: plain PPO in hidden mode scored 36 at reward scale 0.01 and 16 at reward scale 1, with 15–22% completed and the same failures (`sb3_screen1` / `sb3_confirm1`, 8 seeds each). But those runs used another reward scale, only one learner (plain PPO, batch 1024), and a script outside this repo. This control repeats step 0 with the battery hidden and no memory, so that A, the memoryless agent and the agents that see the battery differ in one thing at a time:

| Run | Learner | Sees the battery | Memory | Same as |
|---|---|---|---|---|
| 0a_blind | Plain PPO, batch 256 | No (hidden mode) | No | 0a without the battery |
| 0b_blind | Plain PPO, batch 1024 | No | No | 0b without the battery; the learner of the `sb3_*` runs |
| 0c_blind | CPPO with no context (`cppo.context_source=none`) | No | No | 0c without the battery, **and the A rerun without its encoder** |

- **Everything else as step 0 and the A rerun:** discount 0.99, reward scale 0.1, 2M steps, 8 environments × 512 steps, 10 epochs, the same networks. The actor and critic have 67,588 and 67,329 parameters, exactly plain PPO's in hidden mode.
- **Seeds 0–7**, like step 0's confirmation at reward scale 0.1. Memoryless runs vary a lot (one of the 8 `sb3` runs at 0.01 broke down, −295). Training reset seeds are 0–14 and validation seeds 1000–1002; the final test stays unused.
- **Scoring:** the same validation flights and scoring as everything else, i.e. the late score (1.5M–2M), and completion, battery-empty and ground-crash rates of the final model, overall and for each starting battery.
- **Implementation:** `step0_oracle.py` adds `NoContextEncoder` (`z_t = 0`, no parameters, no memory) and `NoContextPolicy`, whose actor and critic read the observation only (shared with the oracle in `ObservationOnlyPolicy`). Configs: `config/step0/cppo_blind.yaml`, `ppo/configs/experiment/S0a_blind_bs256.yaml`, `S0b_blind_bs1024.yaml`. `validate_config` requires hidden mode and rejects encoder settings with `context_source=none`.
- **Report:** `report_step0.py` accepts 0a_blind, 0b_blind and 0c_blind. They share A's protocol, so one report holds A and the three controls.
- **Checks (`checks.step0_checks`):**
  - Each control differs from its step 0 run only in the observation mode (and, for 0c_blind, the context source).
  - 0c_blind differs from the A rerun only in the context source and the encoder objective.
  - With the same weights, the memoryless CPPO gives exactly plain PPO's hidden-mode values and actions, and the previous action has no effect.
  - Only the actor and critic learn, and a saved model reloads as `NoContextPolicy`.
  - Mislabeled runs are rejected by the report.
- **W&B:** `S0a_blind_bs256_gamma0.99_rs0.1_s<seed>`, `S0b_blind_bs1024_…`, `cppo_S0c_blind_gamma0.99_rs0.1_s<seed>`, tag `blind`.

**Comparisons, fixed before running:**
1. **A vs. 0c_blind** (same learner): does the encoder add anything? Seeds 0–2 paired, and A's three seeds against 0c_blind's eight.
2. **0a, 0b, 0c vs. their blind versions** (same seeds 0–7, already run at reward scale 0.1): what knowing the battery is worth for each learner, the environment's main claim.
3. **0b_blind vs. the `sb3` runs** at 0.01 and 1: how much the reward scale matters without the battery.

```bash
# 3 controls x seeds 0-7 = 24 runs of 2M steps, all at once with 1 thread each, about 15-20 minutes
nohup python run_step0.py --conditions 0a_blind 0b_blind 0c_blind --seeds 0 1 2 3 4 5 6 7 --jobs 24 --threads 1 > run_blind.log 2>&1 &
# The launcher prints the report command for the 24 runs; add the A rerun to the same report with
#   --run A runs_cppo/d36o544n/config.yaml runs_cppo/d36o544n --run A runs_cppo/rxguqn34/config.yaml runs_cppo/rxguqn34 \
#   --run A runs_cppo/ho5et2ek/config.yaml runs_cppo/ho5et2ek
# Optional, to put A on 8 seeds too (hours): python run_step0.py --conditions A --seeds 3 4 5 6 7 --threads 4
```

**Memoryless agent results (2026-10-14).** Seeds 0–7 (A: 0–2), scored like step 0. One report holds A and the three controls (`runs_step0/20261014-020124/report_blind_and_A.json`); the step 0 rows come from the 8-seed confirmation at reward scale 0.1. W&B groups: `S0a_blind_bs256_gamma0.99_rs0.1`, `S0b_blind_bs1024_gamma0.99_rs0.1`, `cppo_S0c_blind_gamma0.99_rs0.1`.

| Agent | Sees the battery | Memory | Late score | Seeds | Completed | Battery ran out | Hit the ground |
|---|---|---|---:|---|---:|---:|---:|
| 0a: plain PPO, batch 256 | Yes | No | 246 | 185 – 349 | 88% | 6% | 6% |
| 0a_blind | No | No | 23 | −14 – 111 | 26% | 74% | 0% |
| 0b: plain PPO, batch 1024 | Yes | No | 258 | 212 – 304 | 86% | 5% | 9% |
| 0b_blind | No | No | 77 | 42 – 136 | 15% | 85% | 0% |
| 0c: CPPO, true battery | Yes | — | 282 | 212 – 361 | 86% | 4% | 10% |
| 0c_blind: CPPO, no context | No | No | 115 | 90 – 139 | 25% | 75% | 0% |
| **A rerun: CPPO, learned encoder** | No (infers it) | Yes | **22** | 106 / −3 / −37 | 17% | 83% | 0% |

Flights completed by starting battery (20 / 30 / 40 / 50 / 60 / 70%): 0c 21 / 96 / 100 / 100 / 100 / 100%; 0c_blind 0 / 0 / 0 / 0 / 50 / 100%; A 0 / 0 / 0 / 0 / 33 / 67%.

- **A is worse than the same learner without any context.** CPPO with no battery and no memory scores 115, all 8 seeds between 90 and 139. A scores 22. Seed by seed: equal on seed 0 (106 vs 110), far below on seeds 1 and 2 (−3 vs 139, −37 vs 90), below every memoryless seed. So in two of three seeds the encoder makes CPPO worse, not just useless.
- **A fails as many flights as the memoryless agents:** 17% completed against 15–26%. Every failure, in A and all three controls, is an empty battery. All of them lose every flight that starts at 40% or less.
- **Knowing the battery is worth 170–220 points with every learner** (246 vs 23, 258 vs 77, 282 vs 115), and completion rises from 15–26% to 86–88%: the environment's main claim, now at matched settings.
- **Reward scale 0.1 helps the memoryless agent too:** plain PPO with batch 1024 scores 77, against 36 at 0.01 and 16 at 1 (`sb3` runs). None of the 24 runs broke down.
- **Without the battery, CPPO's learner does best** (115, against 77 and 23 for plain PPO) and varies least across seeds. The memoryless level that A should beat is therefore 115, not the 16–36 used earlier.
- **Why A does worse is open.** Candidates:
  - The encoder changes after every update, so the actor and critic keep chasing a moving input. This is the motivation for a frozen encoder.
  - The estimate misleads in a consistent way: the average guess at take-off, then a clock (see the encoder-error section below).
  - With three seeds this can't be settled; A on seeds 3–7 would firm it up.

### 1. Bracket the encoder with controls

Run the same PPO in every condition and change only the context `z_t` the policy receives:

| Condition | `z_t` | What it answers |
|---|---|---|
| **Oracle** | true `b_t` | Can PPO use a perfect estimate? Done in step 0: yes, CPPO 282, plain PPO 246–258 (8 seeds) |
| **Noisy oracle** | `b_t` + noise, σ = 0.03 / 0.06 / 0.12 | How good does the estimate need to be? Done: better than 0.03; even 0.03 breaks learning |
| **Frozen encoder** (one per training mix) | pretrained, frozen | The main experiment |
| **Clock-only** | average battery at step `t` | Return from an "encoder" that learned only time |
| **No context** | none | Measured (`sb3_confirm1`): 36 at discount 0.99 / scale 0.01, 100 at 0.995 |

The σ values bracket what we measured: about 0.06 for the current encoders and 0.12 for the clock-only estimate.

How to read the results:
- **Oracle fails:** the problem is the policy or PPO, and better encoders won't help.
- **Oracle and noisy oracle (0.06) work, frozen encoder fails:** the encoder is the problem. Its errors hurt more than random noise of the same size, probably because they're correlated over time and largest early in the flight.
- **Oracle and frozen encoder both work:** the encoder is fine, and the failure comes from training it jointly with the policy. Either `z_t` shifting under the policy or CPPO's own update is to blame.

Random noise is gentler than real encoder error, so a passing noisy oracle is necessary but not sufficient.

**Noisy oracle (implemented 2026-10-13).** `step1_controls.NoisyBattery` replaces the visible battery of observed mode with `b_t + noise`: Gaussian, standard deviation `env.battery_noise`, drawn fresh every step and not clipped. Actor and critic never see the true `b_t`; the env infos keep it for logging. The noise has its own random stream (the flights stay those of step 0 until the actions differ), and every validation flight gets the same noise in every evaluation. Runs 1a, 1b and 1c repeat 0a, 0b and 0c with only this change (`ppo/configs/experiment/S1a_noisy_bs256.yaml`, `S1b_noisy_bs1024.yaml`, `config/step1/cppo_noisy_oracle.yaml`), so CPPO's oracle path reads the noisy battery as `z_t`. W&B names carry the noise, e.g. `cppo_S1c_noisy_oracle_noise0.06_gamma0.99_rs0.1_s0`. `report_step0.py` scores each noise level in its own report; `python -m unittest checks.step1_checks -v` covers the wrapper, both scripts and the report.

```bash
# 3 noise levels x 1a, 1b, 1c x 3 seeds = 27 runs, all at once with 1 thread each (20% faster than 16 x 2), about 15 minutes
nohup python run_step0.py --conditions 1a 1b 1c --battery-noises 0.03 0.06 0.12 --jobs 27 --threads 1 > run_step1.log 2>&1 &
```

**Noisy oracle results (2026-10-13).** Late scores (1.5M–2M), seeds 0–2, discount 0.99, reward scale 0.1; W&B groups `…_noise<σ>_gamma0.99_rs0.1`.

| | No noise (step 0) | σ = 0.03 | σ = 0.06 | σ = 0.12 |
|---|---:|---:|---:|---:|
| Plain PPO, batch 256 (1a) | 276 | 86 | 73 | 64 |
| Plain PPO, batch 1024 (1b) | 275 | 85 | 71 | 51 |
| CPPO (1c) | 282 | 108 | 122 | 53 |
| Flights completed | 85–89% | 7–31% | 2–22% | 6–17% |

- **Even σ = 0.03 breaks it.** Scores fall by about two thirds, close to agents that cannot see the battery (16–36). The planes fly fast (32–57% of the time in the reward band), run out of battery in 69–98% of flights, never finish a flight that starts at 20% or 30% battery, and react much less to the battery. Plain PPO and CPPO fail alike.
- **The noise hurts the learning more than the flying.** CPPO policies trained with the true battery and flown with σ = 0.03 still scored 181 on average (248 without noise) and completed 67–83% of flights. Policies trained with σ = 0.03 and flown with the true battery stayed bad (149, 17–33% completed). With a noisy battery during training, PPO does not learn to use it and settles on flying fast.
- **For the encoder:** its typical error (about 0.066) is twice the noise that already breaks learning, and the real encoder (A rerun, 22) did worse than random noise of the same size (71–122). An estimate must be better than about 0.03 before a policy can learn from it, at least with noise that changes every step; a real encoder's error changes smoothly. Worth testing next: learn with an accurate battery first and switch to the estimate afterwards (or start from an accurate frozen encoder), and give the policy the estimate's uncertainty or memory.

**Fly with noise, no more learning (2026-10-13).** The finished step 0 policies (trained with the true battery, reward scale 0.1, seeds 0–7) fly the validation flights seeing the battery plus noise; nothing is trained (`python step1_fly_with_noise.py --runs runs_step0/<launch>/runs.json ... --output runs_step0/fly_with_noise_rs0.1.json`, about 20 s). Final checkpoints:

| Noise when flying | 0a | 0b | 0c | Completed | Trained with this noise (1a/1b/1c) |
|---|---:|---:|---:|---:|---:|
| None | 251 | 236 | 250 | 86–88% | |
| 0.03 | 107 | 153 | 163 | 60–72% | 64–131, 7–31% completed |
| 0.06 | −28 | 9 | 23 | 32–41% | 48–124, 2–22% completed |
| 0.12 | −136 | −113 | −93 | 12–19% | 43–73, 6–17% completed |

- At 0.03 a trained policy keeps about half to two thirds of its score and completes far more flights than one trained with that noise. At 0.06, the encoder's typical error, its score falls to about zero.
- The policies keep trusting the reading (their throttle reacts to it as strongly as without noise), so a wrong reading leads to wrong actions. The losses are in flights that start with 20–40% battery; at 60–70% a few percent of error barely matters.
- So plugging an estimate into a policy trained with the true battery keeps at most part of the score at 3% error and almost nothing at 6%. Next: does continued training after the switch let the policy adapt (late switch), instead of falling back to fast blind flying (about 70–120)?

**Late switch (implemented 2026-10-13).** The finished step 0 policies keep training for 1M more steps while they see the battery plus noise (0.03 or 0.06; noise 0 is the control: the same extra training without noise). `training.init_model` continues a saved model, its weights and optimizer state, on the run's own environments; the scripts refuse a model whose run used other settings, apart from the noise, step budget, threads, evaluation frequency and naming (`step1_controls.switch_source_mismatches`). Steps and evaluations count from the switch. Runs 1a_switch, 1b_switch and 1c_switch (`ppo/configs/experiment/S1a_late_switch_bs256.yaml`, `S1b_late_switch_bs1024.yaml`, `config/step1/cppo_late_switch.yaml`) continue 0a, 0b and 0c with the same seed; `--from-runs` finds those models (reward scale 0.1, no noise). W&B names: `…_late_switch_…_noise<σ>_gamma0.99_rs0.1_s<seed>`. `report_step0.py` scores 0.5M–1M after the switch.

```bash
# 3 noise levels x 1a, 1b, 1c x seeds 0-2 = 27 runs of 1M steps, all at once, about 8 minutes
nohup python run_step0.py --conditions 1a_switch 1b_switch 1c_switch --battery-noises 0 0.03 0.06 --jobs 27 --threads 1 \
    --from-runs runs_step0/20261013-002244/runs.json runs_step0/20261013-042940/runs.json > run_switch.log 2>&1 &
```

**Late switch results (2026-10-13).** Mean validation return over 0.5M–1M steps after the switch, seeds 0–2 (W&B groups `…_late_switch_…_noise<σ>_gamma0.99_rs0.1`).

| | Control: no noise | Noise 0.03 | Noise 0.06 |
|---|---:|---:|---:|
| 0a (plain PPO, batch 256) | 274 | 125 | 22 |
| 0b (plain PPO, batch 1024) | 225 | 148 | 50 |
| 0c (CPPO) | 233 | **235** | 85 |
| *Flown with noise, no more training (0a / 0b / 0c)* | | *171 / 143 / 181* | *42 / 28 / 58* |
| *Trained with noise from the start (1a / 1b / 1c)* | | *86 / 85 / 108* | *73 / 71 / 122* |

- Extra training alone changes little: the control stays near its starting level (10–30 points lower).
- At noise 0.03, CPPO adapts completely: 235, as high as its control, all three seeds 209–267, 81% of flights completed. Plain PPO does not really adapt (125–148, one seed each broke down), but still beats training with noise from the start.
- At noise 0.06, the encoder's error level, the switch does not help: 22–85, 28–48% of flights completed, no better than training with noise from the start.
- The switched policies keep using the battery (their throttle reacts to it as much as without noise or more, and they do not speed up like blind agents); they fail in low-battery flights.
- So learning with the true battery first and then switching works for an estimate accurate to about 3%, at least with CPPO, but not at about 6%. Before building pilots and a frozen encoder, find the best possible accuracy for this task (enc1's ideal estimator): if even that is off by more than 3% early in flights, a better encoder alone cannot fix it, and the policy needs help with uncertainty (for example the estimate's variance as an input).

**Early noise (implemented 2026-10-13).** The reverse, closer to a learned encoder that becomes accurate during training: the policy learns from scratch seeing the battery plus noise until step 500k (`env.battery_noise_until`), then the true battery, in training and validation (`step1_controls.BatteryNoiseSwitch`; `train/battery_noise` logs the noise averaged over each rollout). Runs 1a_early, 1b_early and 1c_early (`ppo/configs/experiment/S1a_early_noise_bs256.yaml`, `S1b_early_noise_bs1024.yaml`, `config/step1/cppo_early_noise.yaml`) are 1a, 1b and 1c with that switch: 2M steps, scored on 1.5M–2M like step 0 (0a–0c: no noise ever; 1a–1c: noise throughout); the final validation sees the true battery. W&B names: `…_early_noise…_noise<σ>_until500000_gamma0.99_rs0.1_s<seed>`. It asks whether an encoder has time to become accurate (does a noisy start leave lasting damage?) and whether the policy then learns to use the accurate battery.

```bash
# 3 noise levels (0.12 is about the A rerun's early encoder error) x 1a, 1b, 1c x seeds 0-2 = 27 runs of 2M steps, about 15 minutes
nohup python run_step0.py --conditions 1a_early 1b_early 1c_early --battery-noises 0.03 0.06 0.12 --jobs 27 --threads 1 > run_early.log 2>&1 &
```

**Early noise results (2026-10-13).** Late scores (1.5M–2M), seeds 0–2 (W&B groups `…_early_noise…_noise<σ>_until500000_gamma0.99_rs0.1`).

| | No noise ever (step 0) | 0.03 until 0.5M | 0.06 until 0.5M | 0.12 until 0.5M | Noise throughout (0.03 / 0.06 / 0.12) |
|---|---:|---:|---:|---:|---:|
| 0a (plain PPO, batch 256) | 276 | 270 | 289 | 331 | 86 / 73 / 64 |
| 0b (plain PPO, batch 1024) | 275 | 301 | 286 | 280 | 85 / 71 / 51 |
| 0c (CPPO) | 282 | 334 | 306 | 282 | 108 / 122 / 53 |
| Flights completed | 85–89% | 85–94% | 83–89% | 78–91% | 2–31% |

- **No lasting damage.** After 500k noisy steps, even at 0.12 (about the A rerun's early encoder error), every learner ends as good as with the true battery from the start (270–334 against 275–282). Some runs end higher (CPPO at 0.03: 334, seeds 323–347), possibly from extra exploration, but three seeds cannot tell.
- **Fast recovery.** Right at the switch the policies score 40–137 with the true battery; 250k steps later they are at 202–384, close to step 0 at that point, and they have caught up by 1M. They then use the battery (throttle reacts to it as much as in step 0) and do not fly like blind agents.
- **The order matters.** Noisy first, then accurate: the policy fully adapts. Accurate first, then noisy (late switch) or noisy throughout: it fails at 0.06. So an encoder may be poor early in training; what matters is how accurate it ends up. Here the switch goes to the exact battery; next, end at 0.03 or 0.06 instead of 0, closer to a real encoder, to find how accurate the late estimate must be.

**In-flight noise (2026-10-13).** Closer to how an encoder behaves inside each flight: it can't know the battery at take-off and works it out as the flight goes on. In every flight, in training and validation, the battery is noisy early in the flight and exact from flight step 350 (the middle of a 700-step flight). There are two versions (`env.battery_noise_flight`): **sharp**, full noise until step 350 and then the exact battery, and **linear**, noise that shrinks evenly to 0 at step 350. Both use the same random draws as constant noise (`step1_controls.NoisyBattery`), so the only difference is how much of the noise is left at each step. Runs 1a/1b/1c_sharp and 1a/1b/1c_linear (`ppo/configs/experiment/S1a_inflight_noise_bs256.yaml`, `S1b_inflight_noise_bs1024.yaml`, `config/step1/cppo_inflight_noise.yaml`) are 1a, 1b and 1c with that profile: 2M steps, scored on 1.5M–2M like step 0. W&B names: `…_inflight…_<sharp|linear>350_noise<σ>_gamma0.99_rs0.1_s<seed>`. The question: if the battery is noisy only early in each flight, does the policy still learn, and does it fly safely until the battery becomes clear? Compare with noise throughout (1a–1c, which fail) and none (step 0).

```bash
# 2 profiles x 3 noise levels (0.06, 0.12, 0.25) x 1a, 1b, 1c x seeds 0-2 = 54 runs of 2M steps, 27 at a time, about 30-35 minutes
nohup python run_step0.py --conditions 1a_sharp 1b_sharp 1c_sharp 1a_linear 1b_linear 1c_linear --battery-noises 0.06 0.12 0.25 --jobs 27 --threads 1 > run_inflight.log 2>&1 &
```

**In-flight noise results (2026-10-13).** Late scores (1.5M–2M), seeds 0–2 (W&B groups `…_inflight…_<sharp|linear>350_noise<σ>_gamma0.99_rs0.1`). One report per noise level and profile (`step0_report_noise<σ>_<profile>.json`).

| | No noise (step 0) | Noise throughout (0.06 / 0.12) | Sharp 0.06 | Linear 0.06 | Sharp 0.12 | Linear 0.12 | Sharp 0.25 | Linear 0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0a (plain PPO, batch 256) | 276 | 73 / 64 | 257 | 284 | 53* | 300 | 129 | 244 |
| 0b (plain PPO, batch 1024) | 275 | 71 / 51 | 242 | 351 | 224 | 292 | 173 | 213 |
| 0c (CPPO) | 282 | 122 / 53 | 174 | 309 | 188 | 268 | 154 | 183 |
| Flights completed | 85–89% | 2–31% | 41–67% | 78–85% | 43–69% | 70–83% | 54–61% | 39–67% |

\* One seed broke down by 200k steps (full throttle, battery empty after about 100 steps); the other two score 228 and 229.

- **Noise only early in the flight hurts far less than noise all flight.** Every in-flight run beats noise throughout (51–122).
- **Fading noise (linear) is as good as the true battery** at 0.06 and 0.12 (268–351 against 275–282), even starting at 0.12, about the A rerun's early encoder error. At 0.25 it keeps 183–244.
- **Noise that stays full until mid-flight (sharp) costs a lot,** even at 0.06 (174–257). Flights starting at 30% fail almost always (25–27 of 27, against 1 of 27 with the true battery), and many at 40%: the battery runs out, mostly before step 350 (median 265–320 steps). With the true battery a policy starting low flies gently to save energy (throttle 0.08 at 20% charge against 0.67 at 70%). With sharp noise it can't tell 30% from 50% from one noisy reading, so it flies normally (throttle 0.70–0.96 at 20%), and the exact reading comes too late.
- **So what matters is how early in each flight the estimate becomes good.** A plane starting low must decide to save energy in about the first 250 steps. An estimate that starts poor and improves steadily, as a recurrent encoder's should, is fine; one that stays poor until mid-flight is not.
- Caveats: the noise is fresh at every step and these policies have no memory, so they can't average readings, unlike a real encoder whose error changes smoothly. Linear 0.06 scoring above the true battery (0b: 351) matches early noise, possibly extra exploration, but three seeds can't tell.
- Next: measure how the A rerun's encoder error changes over a flight, step by step: is it like sharp or linear, and how large is it around step 250? (Done, below.)

**The A rerun's encoder error over a flight (2026-10-13).** The in-flight noise runs say an estimate is good enough if it improves steadily during each flight. This checks whether the real encoder does. `step1_encoder_error.py` loads the three A rerun models (seeds 0–2) and records, at every step of the 18 validation flights (seeds 1000–1002, never the final test), the encoder's estimate z_t next to the true battery b_t. It does this for the 500k, 1M and 1.5M checkpoints and the final model, with two kinds of flights:

- **Own flights:** the A policy flies, exactly as in its validation. These are the kind of flights its encoder was trained on.
- **Pilot flights:** a good policy flies the plane and the A encoder only watches. The pilot is the step 0 CPPO policy with the true battery (0c, same training seed, completes 87% of these flights). The encoder reads the hidden observation and the pilot's actions, never the battery. These are the flights a policy that uses the battery would fly.
- **Cross pairs:** every final A encoder also watches every other seed's pilot, to tell whether a large error comes from the encoder or from the pilot's flights.

Safety checks: every flight runs a hidden-mode and an observed-mode copy of the env side by side with the same seed and actions. The encoder and the A policy read only the hidden one, only the pilot reads the observed one, and the script checks at every step that the hidden observation is the observed one without the battery. The final models' flights match the returns and lengths of the runs' own final validation exactly. `checks.step1_checks` tests that the encoder reads the pilot's actions on the pilot's exact flight.

```bash
# A rerun seeds 0-2 and the 0c runs of the same seeds (rs 0.1, discount 0.99); about 30 seconds, results in runs_step0/encoder_error.json
python step1_encoder_error.py --a-runs runs_cppo/d36o544n runs_cppo/rxguqn34 runs_cppo/ho5et2ek \
    --pilot-runs runs_cppo/n8ubl3ha runs_cppo/g9awmtpl runs_cppo/wvxnps47 --output runs_step0/encoder_error.json
```

Typical size of the error (root mean square of estimate − true battery over the flights still flying), final models, 3 seeds × 18 flights, next to the two in-flight noise profiles at 0.12:

| Flight step | 0 | 50 | 100 | 175 | 250 | 350 | 500 | 650 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Own flights** (A flies) | 0.17 | 0.12 | 0.05 | 0.03 | **0.02** | 0.04 | 0.02 | 0.04 |
| Own flights still flying (of 54) | 54 | 54 | 54 | 48 | 39 | 30 | 18 | 10 |
| **Pilot flights** (0c flies, A's encoder watches) | 0.17 | 0.16 | 0.13 | 0.14 | **0.16** | 0.15 | 0.14 | 0.14 |
| Pilot flights still flying (of 54) | 54 | 54 | 54 | 54 | 54 | 52 | 50 | 47 |
| Linear 0.12 (trained fine: 268–300) | 0.12 | 0.10 | 0.09 | 0.06 | 0.03 | 0 | 0 | 0 |
| Sharp 0.12 (hurt: 53–224) | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0 | 0 | 0 |
| Noise all flight 0.12 (failed: 51–64) | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |

Estimate / true battery, by starting battery (final models, mean over 3 seeds; "–" = no flight still flying):

| Start | Own: step 50 | Own: step 100 | Own: step 250 | Pilot: step 50 | Pilot: step 100 | Pilot: step 250 | Pilot: step 500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20% | 0.26 / 0.11 | 0.12 / 0.06 | – | 0.34 / 0.15 | 0.28 / 0.12 | 0.22 / 0.09 | 0.04 / 0.07 |
| 30% | 0.29 / 0.21 | 0.16 / 0.15 | 0.06 / 0.06 | 0.36 / 0.23 | 0.27 / 0.20 | 0.09 / 0.17 | 0.04 / 0.14 |
| 40% | 0.32 / 0.31 | 0.24 / 0.24 | 0.09 / 0.08 | 0.34 / 0.31 | 0.31 / 0.26 | 0.11 / 0.21 | 0.04 / 0.18 |
| 50% | 0.36 / 0.40 | 0.33 / 0.34 | 0.16 / 0.16 | 0.36 / 0.40 | 0.28 / 0.34 | 0.13 / 0.24 | 0.05 / 0.19 |
| 60% | 0.40 / 0.50 | 0.41 / 0.43 | 0.23 / 0.24 | 0.40 / 0.50 | 0.38 / 0.42 | 0.16 / 0.29 | 0.06 / 0.20 |
| 70% | 0.44 / 0.60 | 0.47 / 0.52 | 0.30 / 0.32 | 0.43 / 0.59 | 0.46 / 0.51 | 0.22 / 0.35 | 0.08 / 0.22 |

At take-off (step 0) the estimate is 0.44 for every flight in both cases.

Error during training (RMS at steps 50 / 100 / 250 / 500):

| Model | Own flights | Own flights completed | Pilot flights |
|---|---|---:|---|
| 500k | 0.15 / 0.09 / 0.04 / 0.04 | 6% | 0.22 / 0.23 / 0.14 / 0.14 |
| 1M | 0.15 / 0.12 / 0.18 / 0.14 | 9% | 0.19 / 0.20 / 0.13 / 0.12 |
| 1.5M | 0.13 / 0.07 / 0.11 / 0.08 | 4% | 0.16 / 0.14 / 0.14 / 0.12 |
| Final (2M) | 0.12 / 0.05 / 0.02 / 0.02 | 17% | 0.16 / 0.13 / 0.16 / 0.14 |

Cross pairs, final encoders (RMS at step 100 / step 250); the diagonal is the same-seed pairing above:

| | Pilot 0c s0 (18/18 completed) | Pilot 0c s1 (15/18) | Pilot 0c s2 (14/18) |
|---|---:|---:|---:|
| Encoder A s0 (late score 106) | 0.08 / 0.04 | 0.16 / 0.06 | 0.10 / 0.09 |
| Encoder A s1 (late score −3) | 0.07 / 0.16 | 0.19 / 0.18 | 0.10 / 0.23 |
| Encoder A s2 (late score −37) | 0.08 / 0.11 | 0.19 / 0.12 | 0.07 / 0.21 |

What it shows:

- **At take-off the encoder can't know, so it guesses the average.** It says 0.44 for every flight, an error of 0.17, exactly what not knowing the starting battery costs. No estimator can do better at step 0.
- **On its own flights, the error shrinks like the linear profile, only faster.** It is 0.05 by step 100 and 0.02–0.04 from step 175 on: below linear 0.12 from step 100 to 250, though it never becomes exact as linear does at step 350. The error is smooth, a steady offset rather than step-to-step jitter: the error 25 steps later is strongly correlated with the current one (0.87; 0.96 on pilot flights). The noisy oracle redraws its noise every step.
- **A still fails, because it flies hard from take-off, like a blind policy.** Its flights end with an empty battery at times set by the starting charge: on average 140–213 steps at 20%, 216–337 at 30%, 612–700 at 70%. Only 17% complete. By step 100, when the estimate becomes right, a plane that started at 20% has 6% left. The environment reveals the battery only while the throttle is at its cap, so this encoder learns the battery by spending it.
- **On a good pilot's flights the encoder is about as wrong as not knowing, for the whole flight.** The error stays at 0.13–0.17 from start to end and never becomes exact. From mid-flight on it thinks the battery is lower than it is (average offset −0.07 at step 250, −0.12 at step 500). At step 500 it says 4–8% for every flight while the truth is 7–22%. It expects the battery to drain as fast as in A's own hard flights. This resembles noise all flight at 0.12–0.25, where every learner failed, and is worse than sharp, which at least becomes exact at mid-flight.
- **It doesn't get better with training.** On pilot flights the error is 0.12–0.23 at every checkpoint. On own flights it moves with the policy: it is good at 500k, worse at 1M and 1.5M, and best at the end.
- **The encoder learned a clock, not the battery.** In A's flights the plane always flies about the same way, so the battery is basically "average guess minus a fixed amount per step". The encoder learned that rule instead of reading the battery from how the plane responds and counting the drain from the actions. When a policy flies differently, for example gently on a low battery, the rule is wrong.
- **The encoder that saw the widest range of flying generalizes best.** Seed 0's encoder reads every pilot's flights fairly well (0.04–0.09 at step 250), while seeds 1 and 2 are at 0.11–0.23. Seed 0's policy flies more gently and with more variety: its battery drops by 0.10–0.17 over the first 100 steps, against 0.14–0.21 for seeds 1 and 2. The pilots' battery drops by as little as 0.05. Seed 0 also has the best late score (106 against −3 and −37). The pilots' flights matter too: pilot s0's flights are easier to read than s2's. Three seeds only, though.
- **This is a chicken-and-egg problem.** The policy can only learn to save energy if the estimate is good on gentle flights, and the encoder only becomes good on the flights the current policy already flies.

Caveats:

- Part of the pilot-flight error can't be avoided. A plane flying gently never reaches the throttle cap, so its starting battery doesn't show in what the agent sees. The drift toward too-low estimates is avoidable, though: the drain follows from the actions, which the encoder reads.
- How much of the error is unavoidable is what enc1's ideal estimator would tell (measured below: about half on the pilot's flights).
- These are the A rerun's encoders (point estimate, supervised on A's own flights); C–F were not measured.

What this means for the plan: the encoder's problem is less its accuracy on the flights it was trained on than how narrow those flights are. This strongly supports step 3, training the encoder on varied flights (gentle and hard, random throttle, the 0c pilots' flights) so that it reads the battery from the plane's response and counts the drain from the actions instead of learning a clock. Before that, enc1's ideal estimator on the pilots' flights would show the best error possible at each flight step, the target for a pretrained encoder.

**The ideal estimator ("perfect calculator", 2026-10-14).** How much of the encoder's error is unavoidable? `step1_ideal_estimator.py` computes the best estimate any encoder could make from the same information.

**How it works:**
- `IdealBatteryEstimator` reads what the encoder reads: the hidden-mode observations and the executed actions, never the battery.
- It knows the exact physics and the training distribution of starting states.
- The unknowns are the starting battery (uniform 0.2–0.7) and the starting speed (uniform 0–30 m/s, read with noise of 5 m/s); altitude and climb rate are observed exactly.
- It tries a grid of 201 × 601 candidate starting states. Each is replayed through the environment's own `BatteryPlaneVec.step` with the executed actions, from the observed altitude and climb rate.
- Each candidate is weighed by how well it predicts the next speed reading and the exact climb rate. A candidate whose battery would be empty while the plane flies on is dropped.
- The estimate is the weighted mean battery, i.e. the posterior mean: the smallest possible expected squared error.

**Checks (`checks.step1_checks`):**
- With the true starting state as its only candidate, the replay equals the environment's battery to about 5×10⁻¹⁴.
- At take-off it gives the prior (0.45 ± 0.144).
- On scripted flights, full throttle reveals the battery within about 100 steps, while flying below the throttle limit reveals nothing for the whole flight.
- It refuses observations that contain the battery.
- On the validation flights its own uncertainty matches its errors (calibration 0.92 and 0.83, 1 = perfect), and the grid always held a candidate matching the climb rate.

It ran on the same 108 validation flights as the encoder-error measurement: A's own flights and the 0c pilots' flights, final models, seeds 0–2. Results are in `runs_step0/ideal_estimator.json`.

```bash
# About 2.5 minutes for the 108 flights with 24 processes
python step1_ideal_estimator.py --a-runs runs_cppo/d36o544n runs_cppo/rxguqn34 runs_cppo/ho5et2ek \
    --pilot-runs runs_cppo/n8ubl3ha runs_cppo/g9awmtpl runs_cppo/wvxnps47 --output runs_step0/ideal_estimator.json
```

Typical error at each flight step. The clock only knows how far into the flight it is (the mean battery of these flights at that step), so it is slightly flattering.

| Flight step | 0 | 25 | 50 | 100 | 175 | 250 | 350 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A flies:** clock | 0.17 | 0.17 | 0.17 | 0.16 | 0.13 | 0.10 | 0.08 | 0.05 |
| **A flies:** A's encoder | 0.17 | 0.16 | 0.12 | 0.05 | 0.03 | 0.02 | 0.04 | 0.02 |
| **A flies:** ideal estimator | 0.17 | 0.15 | 0.09 | 0.04 | 0.02 | 0.01 | 0.02 | 0.01 |
| **Good pilot flies:** clock | 0.17 | 0.16 | 0.15 | 0.13 | 0.11 | 0.09 | 0.07 | 0.06 |
| **Good pilot flies:** A's encoder | 0.17 | 0.18 | 0.16 | 0.13 | 0.14 | 0.16 | 0.15 | 0.14 |
| **Good pilot flies:** ideal estimator | 0.17 | 0.16 | 0.11 | 0.09 | 0.08 | 0.08 | 0.07 | 0.06 |

Good pilot's flights at step 250, by starting battery:

| Starting battery | 20% | 30% | 40% | 50% | 60% | 70% |
|---|---:|---:|---:|---:|---:|---:|
| Ideal estimator | 0.17 | 0.09 | 0.03 | 0.02 | 0.01 | 0.02 |
| A's encoder | 0.25 | 0.10 | 0.12 | 0.14 | 0.16 | 0.17 |

- **On its own flights, A's encoder is close to ideal.** From step 100 on it is within about 0.01 of the ideal and closes about 90% of the clock-to-ideal gap (62% at step 50). Reading the battery on familiar flights is not A's problem.
- **Part of the time it takes is physics.** Even the ideal estimator, on A's full-throttle flights, is 0.15 off at step 25 and 0.09 at step 50, and reaches 0.04 only at step 100. The battery shows slowly, through a noisy speed reading; nothing can know it in the first 1–2 seconds.
- **On the good pilot's flights, about half of the encoder's error is unavoidable** (0.08 against 0.16 at step 250).
  - **Unavoidable:** the low-battery flights (20%: 0.17, 30%: 0.09), where the pilot flies gently. Flying below the throttle limit hides the battery completely.
  - **The encoder's own:** from 40% up the ideal is within 0.01–0.03 by step 250, while the encoder is 0.12–0.17 off, worse than the clock. That is the clock it learned from A's flying.
- **The autogenic trap, measured.**
  - The flights where knowing the battery matters most (low battery, so save energy) are the ones where saving energy, i.e. flying gently, hides the battery.
  - An agent that can't see the battery therefore can't copy the oracle's behaviour: it must spend battery at full throttle to find out its charge, then save.
  - So the oracle's 282 is a ceiling a hidden-mode agent can't fully reach.
- **For the plan:**
  - An encoder trained on varied flights (step 3) can remove the encoder's own half of the error (on the pilot's flights from about 0.16 to 0.08 at step 250, and to 0.01–0.03 from 40% up).
  - On low-battery flights no encoder can help unless the policy learns to probe first, and the policy should get the estimate's uncertainty (the Gaussian head of C–F).
  - An easy alternative environment, `EASY_ENV_WARMUP.md`: a 5-second autopilot take-off at full throttle before the agent takes control, which reveals the battery (best possible error 0.05 at hand-over, against 0.12 after 2 s), with the starting battery raised so that the hand-over battery stays uniform on 0.2–0.7.
  - A further control: use the ideal estimator itself as the encoder and train PPO on top. That would give the best score any hidden-mode agent can reach, the honest target for an encoder. At full grid size it is too slow for 2M steps; a coarser grid would take roughly 2 hours per run.

### 2. Run every condition through plain PPO

A frozen encoder is a fixed function of the history. So it can sit in a VecEnv wrapper that:
- keeps one encoder state per environment,
- feeds it `[o_t, a_{t-1}]` each step,
- resets it at episode starts,
- and appends `z_t` to the observation.

Then every condition runs on the same SB3 PPO setup as the p01–p04 baselines. This removes the CPPO-specific differences, including the 4× fewer gradient steps and the recurrent minibatching. The oracle condition then reduces to the observed baseline, which gives a built-in check: it should reproduce about 212 final and about 360 at peak. Keep one CPPO frozen-encoder run as a bridge back to the A–F results.

**The context wrapper (implemented 2026-10-14).** `context_wrapper.py` holds `ContextVecEnv`, a VecEnv wrapper for plain PPO.

- **What it does:**
  - It keeps one memory of a frozen context source per environment.
  - At every step it feeds the source the new hidden-mode observation and the action just executed, clipped as the environment executes it.
  - Where an episode starts, the memory is reset and the previous action is zero, as in CPPO's collector.
  - It appends the context to every observation, including each episode's final observation (`terminal_observation`), which continues the old memory before the next episode starts fresh.
- **Settings:** the plain-PPO config's `context:` section (`ppo/configs/battery_plane_ppo.yaml`). The source is chosen with `context.source`:
  - `none`: the default, plain PPO unchanged.
  - `ideal`: the ideal estimator, made step-by-step (`IdealBatteryEstimator.start` / `update`, bit-identical to the whole-flight version). It appends the estimate, and its std with `context.include_std`. Training uses a coarser grid (101 × 301 candidates, climb-rate tolerance 0.01 m/s).
  - `encoder`: the frozen encoder of a saved Contextual_PPO model (`context.encoder_model`), giving z_t as that model's actor and critic read it. For example A's encoder, for the "freeze A's encoder, train a fresh policy" test.
  - `true_battery`: the true battery from the env infos, a check only.
- **Where it applies:**
  - Both scripts' validation environments wrap with the same source, so `report_step0.py` validates such runs correctly.
  - The report's protocol now includes the context, and every existing condition requires none.
  - `validate_ppo` requires hidden mode and rejects the noisy battery alongside a context. Contextual_PPO refuses a `context` section, since it has its own encoder.
- **Checks (`checks.step2_checks`):**
  - With the true battery, hidden mode plus the wrapper reproduces observed mode exactly: observations, rewards, episode ends and terminal observations, across automatic resets.
  - The ideal and encoder contexts equal the offline estimator and the encoder run over each episode.
  - The encoder stays frozen while PPO trains on top.
  - PPO saves, reloads and is validated through the wrapper.
  - Wrong settings are refused.
- **Speed:** a run alone trains at about 1,000 steps per second, but 32 side by side at first managed only about 40 each. With 8 × 30,000 candidates per run, every run's data went to main memory instead of the processor's cache.
  - The fix is pruning (`context.prune_below: 30`): candidates whose weight falls below 10⁻¹³ of the best one are dropped, and the rest move to a smaller set.
  - The estimates stay the same to about 10⁻¹², 6–14 times faster (`checks.step2_checks` compares pruned and exact).
  - With 32 runs side by side each now trains at about 420 steps per second early in training, and about 0.6 GB of memory per run.
  - Later in training, flights get longer and gentle flights keep more candidates, so the 32-run launch of the ideal-context runs took about 3.5 hours.


**Plain PPO with the ideal estimate: design (2026-10-14).** What is the best score an agent that can't see the battery can reach in the hard environment? Plain PPO in hidden mode reads the ideal estimator's battery estimate through the context wrapper. This is the best possible frozen encoder: it reads the battery from the physics on every kind of flight and never changes.

| Run | Learner | What PPO reads | Compared with |
|---|---|---|---|
| 2a_ideal | Plain PPO, batch 256 | Hidden observation + ideal estimate | 0a (true battery, 246) and 0a_blind (no battery, 23) |
| 2b_ideal | Plain PPO, batch 1024 | Hidden observation + ideal estimate | 0b (258) and 0b_blind (77) |
| 2a_ideal_std | Plain PPO, batch 256 | + the estimate's std | 2a_ideal: does knowing how sure the estimate is help? |
| 2b_ideal_std | Plain PPO, batch 1024 | + the estimate's std | 2b_ideal |

- **Settings:** as step 0 and the memoryless controls (discount 0.99, reward scale 0.1, 2M steps, seeds 0–7, validation seeds 1000–1002, scored on 1.5M–2M). The configs (`ppo/configs/experiment/S2*_ideal*`) are the memoryless configs plus `context.source=ideal`.
- **Training grid:** 101 × 301 candidates, climb-rate tolerance 0.01 m/s.
- **Report:** `report_step0.py` accepts the four conditions. The estimate-only and the estimate-plus-std runs go in separate reports, because their inputs differ.
- **Checks (`checks.step2_checks`):** the four configs differ from the memoryless ones only in the context, the launcher builds the same runs, and mislabeled runs are rejected.
- **W&B:** `S2a_ideal_bs256_gamma0.99_rs0.1_s<seed>`, `S2b_ideal_bs1024_…`, `S2a_ideal_std_bs256_…`, `S2b_ideal_std_bs1024_…`, tag `ideal_context`.

**Decision rules, fixed before running.** Gap closed = (ideal − memoryless) / (true battery − memoryless), per learner, on the same seeds.
- **Close to the true battery** (within the seed spread, or most of the gap closed): in practice the battery can be inferred well enough. The encoder is the bottleneck, so build the pilots and a frozen learned encoder (steps 3–4), with this score as their target.
- **Close to the memoryless agent:** even a perfect encoder doesn't help, because the remaining uncertainty and the need to probe dominate. Don't build pilots yet; rethink first (the easy environment of `EASY_ENV_WARMUP.md`, the std input, exploration).
- **In between:** report the gap closed, and look at where the ideal agent fails (by starting battery, early-flight behaviour, whether it probes).
- **std:** if the `_std` runs score clearly higher (beyond the seed spread), give the learned encoder an uncertainty output too (the Gaussian head of C–F).

```bash
# 4 conditions x seeds 0-7 = 32 runs of 2M steps, all at once with 1 thread each, about 3.5 hours
nohup python run_step0.py --conditions 2a_ideal 2b_ideal 2a_ideal_std 2b_ideal_std --seeds 0 1 2 3 4 5 6 7 --jobs 32 --threads 1 > run_ideal_ppo.log 2>&1 &
```

**Plain PPO with the ideal estimate: results (2026-10-14).** All 32 runs finished, none failed. Each run took 1.5–3.5 hours with 32 side by side (about 3.5 hours for the launch), not the 1.5 hours estimated from the first rollouts. Later in training flights get longer, and on gentle flights the estimator keeps many candidates alive.

The reports, one per variant: `runs_step0/20261014-044145/step0_report_ideal.json` and `step0_report_ideal_std.json`. W&B groups: `S2a_ideal_bs256_gamma0.99_rs0.1`, `S2b_ideal_bs1024_gamma0.99_rs0.1`, `S2a_ideal_std_bs256_gamma0.99_rs0.1`, `S2b_ideal_std_bs1024_gamma0.99_rs0.1`. The true-battery rows come from the 8-seed step 0 confirmation at reward scale 0.1, and the memoryless rows from the memoryless controls; all use the same settings, seeds 0–7 and scoring.

*Late score (1.5M–2M), mean over seeds 0–7 (spread across seeds = population std):*

| Plain PPO | True battery (0a / 0b) | No battery, no memory (0a_blind / 0b_blind) | Ideal estimate (2a / 2b_ideal) | Ideal estimate + std (2a / 2b_ideal_std) |
|---|---:|---:|---:|---:|
| Batch 256 | 246 (52) | 23 (41) | **283** (40) | **303** (19) |
| Batch 1024 | 258 (30) | 77 (32) | **323** (22) | **325** (16) |

*Per seed (0–7):*

| Run | Seeds 0–7 |
|---|---|
| 0a, true battery | 349, 246, 233, 231, 310, 219, 199, 185 |
| 2a_ideal | 243, 285, 312, 285, 222, 355, 304, 256 |
| 2a_ideal_std | 306, 300, 317, 307, 327, 320, 265, 281 |
| 0b, true battery | 268, 292, 265, 248, 256, 304, 212, 222 |
| 2b_ideal | 329, 323, 270, 332, 342, 344, 323, 321 |
| 2b_ideal_std | 348, 309, 304, 337, 309, 329, 319, 347 |

*Outcomes of the final models (18 validation flights per seed):*

| Run | Completed | Battery ran out | Hit the ground |
|---|---:|---:|---:|
| 0a / 0b (true battery) | 88% / 86% | 6% / 5% | 6% / 9% |
| 0a_blind / 0b_blind | 26% / 15% | 74% / 85% | 0% / 0% |
| 2a_ideal / 2b_ideal | 88% / 92% | 10% / 2% | 2% / 6% |
| 2a_ideal_std / 2b_ideal_std | 90% / 90% | 5% / 3% | 6% / 7% |

*By starting battery (20 / 30 / 40 / 50 / 60 / 70%), final models:*

| Run | Flights completed | Return | Mean throttle |
|---|---|---|---|
| 0a, true battery | 29 / 100 / 100 / 100 / 100 / 100% | −160 / 178 / 243 / 322 / 411 / 512 | −0.12 / 0.32 / 0.40 / 0.48 / 0.57 / 0.65 |
| 2a_ideal | 29 / 96 / 100 / 100 / 100 / 100% | −121 / 168 / 259 / 334 / 429 / 521 | 0.19 / 0.29 / 0.38 / 0.47 / 0.57 / 0.64 |
| 2a_ideal_std | 42 / 96 / 100 / 100 / 100 / 100% | −45 / 195 / 276 / 353 / 449 / 542 | 0.25 / 0.34 / 0.40 / 0.50 / 0.60 / 0.67 |
| 0b, true battery | 17 / 100 / 100 / 100 / 100 / 100% | −215 / 169 / 238 / 313 / 402 / 508 | −0.01 / 0.30 / 0.37 / 0.45 / 0.54 / 0.63 |
| 2b_ideal | 54 / 96 / 100 / 100 / 100 / 100% | 1 / 196 / 272 / 356 / 462 / 561 | 0.28 / 0.34 / 0.41 / 0.51 / 0.62 / 0.69 |
| 2b_ideal_std | 42 / 96 / 100 / 100 / 100 / 100% | −54 / 209 / 291 / 376 / 476 / 577 | 0.24 / 0.35 / 0.44 / 0.55 / 0.65 / 0.73 |
| 0a_blind / 0b_blind | 0–42% up to 60%, 67–100% at 70% | −202 to 380 / −241 to 510 | about 1.0 at 20–40%, falling to 0.8–0.9 at 70% |

*Gap closed* = (ideal − memoryless) / (true battery − memoryless), on the same seeds:

| | Ideal estimate | Ideal estimate + std |
|---|---:|---:|
| Batch 256 | 116% | 125% |
| Batch 1024 | 136% | 137% |

**Why above the true battery? The swap check.** `step2_swap_battery.py` flies the final 2a_ideal and 2b_ideal models over the validation flights twice: once with the estimate they were trained on, and once with the true battery swapped into the same input (`context.source=true_battery`). Results are in `runs_step0/20261014-044145/swap_true_battery.json`.

| Final models | Flown with the ideal estimate | Flown with the true battery instead |
|---|---:|---:|
| 2a_ideal (batch 256) | 265, 88% completed | **280**, 92% completed |
| 2b_ideal (batch 1024) | 308, 92% completed | **310**, 92% completed |

```bash
python step2_swap_battery.py --runs runs_step0/20261014-044145/runs.json --output runs_step0/20261014-044145/swap_true_battery.json
```

**What it shows:**

- **A perfect frozen encoder is enough in the hard environment.** Plain PPO reading the ideal estimate closes the whole gap between "no battery" (23 / 77) and "true battery" (246 / 258), and goes beyond it: 283 / 323, or 303 / 325 with the std. Completion rises from 15–26% to 88–92%. This holds although gentle flying hides the battery on low-battery flights: on 20%-battery flights the ideal-estimate agents complete 29–54%, against 17–29% with the true battery.
- **By the decision rule fixed before running, the encoder is the bottleneck.** A's learned encoder (22) and the memoryless agents (23–115) are far below what a good battery estimate makes possible with the same environment and PPO settings. A's failure comes from its encoder (what it learned, and learning it together with the policy), not from the environment being unsolvable for an agent that must infer its battery.
- **The estimate-trained policies are simply better pilots.** Flown with the true battery instead of the estimate, they score the same or slightly higher (280 vs 265, 310 vs 308), still above the agents trained on the true battery (246, 258).
  - So the estimate carried no information beyond the battery that the policies depend on: the advantage comes from training, not from what they read.
  - The difference shows most on low-battery flights. On 20%-battery flights the true-battery agents' mean throttle is negative (−0.12 and −0.01): they push the cruise motor backwards on average, which slows the plane and still costs battery. The estimate-trained agents fly forward gently (0.19–0.28) and complete more of those flights.
  - On high-battery flights they throttle about as hard or slightly harder (0.64–0.73 against 0.63–0.65 at 70%) and earn more there (521–577 against 508–512).
- **A likely reason, not yet tested: uncertainty early in each flight helps PPO learn.** The estimate starts every flight at the same guess (0.45) and becomes accurate as the flight goes on, so the policy first learns behaviour that works for every battery and then refines it. The in-flight noise runs showed the same pattern: with noise fading out by mid-flight, the agents scored 284 / 351 / 309 against 276 / 275 / 282 with the true battery from the start (seeds 0–2).
- **The std input helps a little at most.**
  - At batch 256: 303 against 283, better on 6 of 8 seeds, and much tighter across seeds (spread 19 against 40), with more 20%-battery flights completed (42% against 29%). But the difference is within about 1.3 standard errors.
  - At batch 1024: no difference (325 against 323, better on 4 of 8 seeds).
  - By the decision rule ("clearly higher beyond the seed spread") the std is optional. A learned encoder can start with a point estimate; an uncertainty output (the Gaussian head of C–F) remains worth trying.
- **Batch 1024 does better than batch 256** with the ideal estimate (323 against 283), as it did without memory (77 against 23), and varies less across seeds.

**Caveats:**
- The true-battery runs come from an earlier launch, with the same code path and settings; the swap check, which uses the same models and flights, supports the comparison.
- The ideal estimate is computed online on a coarser grid (101 × 301) with pruning, which matches the full computation to about 1e-12. It is still a bound for estimators with this physics model and prior, not for encoders with other information.
- The learner is plain PPO. CPPO with the ideal estimate was not run; CPPO's own learner did best without memory (115), so it may do at least as well.

**Next steps:**
1. **Pilots and an offline-trained frozen encoder** (steps 3–4). Plug it into the same wrapper (`context.source=encoder`) and compare with these runs. The target is to get close to the ideal estimator's error at each flight step on every pilot style (`step1_ideal_estimator.py`), and a PPO score near 283 / 323.
2. **Optional and cheap: A's own encoder, frozen, under a fresh plain PPO** (`context.source=encoder`, `context.encoder_model=runs_cppo/<A run>/final_model.zip`). Between A (22) and the ideal (283 / 323), this separates what A's encoder knows from the cost of training it together with the policy.
3. **Optional:** rerun 0a / 0b in one launch with the ideal runs, to rule out launch-to-launch differences. And test the "uncertainty helps training" idea directly, for example the true battery with noise fading out by mid-flight against the true battery alone, on 8 seeds.

### 3. Pilots

The physics sets the requirements for the data. The battery shows up in speed only while the cruise motor is at its cap. The environment's own report says one capped speed reading pins `b` to ±0.37, and about 14 capped steps get it to ±0.1. The encoder also has to carry its estimate through stretches where the motor is below the cap.

| Pilot | What it teaches the encoder | Why include it |
|---|---|---|
| Full throttle, low and after climbing | Battery from steady speed at the cap | Most information. It's what the collapsed CPPO runs do |
| Constant speed below the cap | Nothing about the battery | Negative control: the encoder should fall back to the clock estimate, not invent a battery |
| **Probe then hold:** cap for k steps, then hold a lower speed, repeat | Identify the battery, then remember it | **The most important pilot.** The C runs that throttled back had negative R², which is exactly this situation |
| Sinusoidal throttle, random amplitude, frequency and offset | Repeatedly crossing the cap | Partial information, arriving at varied times |
| Random throttle held for random durations, plus the "shaky" style | Broad coverage, actions unrelated to the battery | Closest to early PPO exploration |
| **Altitude profiles** on the lift motor | How the speed–battery relation changes with altitude | Drag depends on altitude, and the reward band is only reachable high up. Don't vary only the throttle |
| PPO checkpoints (hidden and observed, early/mid/late, stochastic) | The states a trained policy actually visits | Closest to what the encoder will face inside PPO |

**Watch out for battery leakage through actions.** If a pilot's actions depend on the true battery, the encoder can learn to read the battery from `a_{t-1}` instead of from the physics. Examples are the environment's `policy_budgeted`, which cuts throttle at a fixed battery level, and observed-mode PPO. That shortcut won't carry over, because the hidden policy's actions depend on `z`, not on the true battery.
- Pilots may read the observation, true speed or altitude, the clock, and randomness, but never `b` or `b0`. A pilot that holds speed below the cap doesn't leak, because below the cap the dynamics don't depend on the battery.
- Put battery-dependent pilots in a separate training mix. Check for leakage by testing that encoder on battery-blind pilots.

Three training mixes:
1. Full throttle only.
2. All battery-blind pilots.
3. Mix 2 plus the PPO checkpoints.

**The pilots (implemented 2026-10-14).** `pilots.py` flies BatteryPlane in many ways and records each flight as an encoder reads it: the hidden-mode observation o_t, the previous executed action a_{t-1} (zero at the start), and, as the training label only, the true battery b_t. The list above was revised after step 1 and the ideal estimator, with three goals: break the link between time and battery (A's encoder learned a clock), include the flights that matter (probing, then saving energy; gentle flying on a low battery), and close leaks.

| | Style | What it does |
|---|---|---|
| Keep | `probe_hold` | Full throttle for 1–10 s, then hold 25–50 m/s below the cap, repeated 1–4 times. The first probe is at take-off half the time |
| Keep | `steady_below` | A constant throttle or a held speed, always below the cap (throttle ≤ 0.55, so at most 11 m/s², under the 11.5 m/s² cap of an empty battery). The battery never shows: the encoder must learn to keep its best guess and count the drain |
| Keep | `random_hold` | Random throttle, reverse included, held for random durations; random lift. Each flight draws its own level and spread, so flights drain at different rates |
| Keep | `shaky` | A new random throttle and lift at every step around a per-flight level and spread |
| Keep, low weight | `wavy` | Sinusoidal throttle with random offset, amplitude, period (1–20 s) and phase |
| Keep, as control | `full_throttle` | Full throttle all flight, climbing to the ceiling: A-like data, only in the narrow mix |
| Change | `hold_speed` | Hold 45–65 m/s (the reward band), steering by the smoothed noisy speed. `battery_plane.policy_hold_goal` steers by the true speed, which would pass a cleaner battery signal through the actions than any agent has |
| Change | `cruise` | Hold the cheapest speed for the altitude, steering by the smoothed noisy speed (changed from `policy_cruise`) |
| Add | `endurance` | A full-throttle probe of 3–8 s at take-off, then gentle flying to the end: late-flight, low-battery data, which full-throttle pilots never reach |
| Add | `blind_ppo` | The memoryless PPO policies (0a_blind, 0b_blind, 0c_blind), stochastic half the time: real PPO flying with no battery information |
| Add | `replay_good` | Good pilots (true battery: 0a, 0b, 0c; ideal estimate: 2a/2b_ideal and their _std) fly a shadow plane with its own random battery and start, and their actions fly the recorded plane. The flying looks battery-aware, but the actions say nothing about the recorded plane's battery |
| Add | `switching` | Changes style mid-flight: segments of 30–250 steps, each flown by a fresh pilot of a style drawn uniformly from the battery-blind styles (the scripted ones, `blind_ppo`, `replay_good`). The whole plan (switch times and styles) is drawn at take-off, before anything is observed, and a segment's style never depends on the time or on any observation. So it can't teach "low battery → gentle flying" |
| Separate mix | `aware_good` | The same good pilots flying the recorded plane itself. **Leaks:** their actions depend on its true battery or on a battery estimate, so an encoder could read the battery from the actions |

**Common to all styles:**
- Every scripted style draws an altitude programme per flight (level, climb, the ceiling, changing targets, or low), steered by the exact observed altitude and climb rate, with a guard against hitting the ground.
- `policy_budgeted` is not used: it reads the true battery.

**What a pilot may read:**
- **Scripted pilots:** the hidden-mode observation (noisy speed, exact altitude and climb rate, time left), their own randomness, and physics formulas that hold no plane state (`PlanePhysics`: drag, the cheapest speed). They steer by an exponentially smoothed speed reading, never the true speed or the battery.
- **Policy pilots:** what their model was trained on. Memoryless pilots read the hidden observation; replayed and aware pilots read their plane's observed observation, or the hidden one plus their own ideal estimate (`PolicyDriver`). A policy must have been trained with the dataset's physics settings (`env.kwargs`).

**Seeds:** every flight has its own seed: style k uses 100,000 + k × 1,000,000 + i. Everything else random in a flight derives from it, so no two flights share noise and every flight is reproducible:
- the pilot's own randomness, `np.random.default_rng([flight seed, style, 17])`
- the k-th action-sampling seed of a stochastic policy pilot, flight seed × 64 + k (`sampling_seed`)
- the k-th shadow plane of a replayed pilot, 1,000,000,000 + flight seed × 64 + k (`shadow_seed`)
- the j-th segment of a switching flight, 10,000,000,000 + flight seed × 64 + j (`segment_seed`). The segment's pilot draws its randomness, sampling seeds and shadow planes from it, in ranges above every other seed

None of these touches the training (0–14, and 0–707 for the first reset of each environment), validation (1000–1002) or final-test (2000–2002) seeds, and `fly` refuses the held-out ones.

**Mixes (`pilots.MIXES`):**
- `narrow`: full throttle only (the A-like control).
- `main`: every battery-blind style.
- `main_plus_aware`: main plus `aware_good`, always tested on battery-blind pilots to catch an encoder that reads the battery from the actions.

**Checks (`checks.step3_checks`):**
- **No hidden channel:** a scripted pilot replayed on a recorded flight's hidden observations, without any plane, gives exactly the recorded actions.
- **Below-cap pilots never reach the cap,** and probes do; outside its probes `probe_hold` never does.
- **Records match the environment:** the hidden-mode observations and batteries equal those of a hidden-mode plane flown with the same actions.
- **Reproducible, and refuses held-out seeds.**
- **Replay doesn't leak:** a replayed good pilot's actions are identical whatever the recorded plane's battery (0.25 or 0.65), deterministic or stochastic. An aware pilot's are not, and it is flagged.
- **Memoryless pilots must be battery-blind,** and a policy trained with other physics settings is refused.
- **Seeds never overlap:** recorded, shadow, sampling and switching-segment seeds are unique and apart from the training, validation and final-test seeds. Every stochastic start, in every flight and style, uses its own seed: two stochastic flights fly different noise, and the same flight twice the same.
- **The leaking style is never in the narrow or main mix,** nor among the styles a switching flight changes between.
- **Switching never ties gentle flying to a low battery:**
  - the same flight with a 25% or a 65% battery gets the same plan
  - gentle styles are as common in segments that start after mid-flight as before (22.5% against 22.9%, expected 2/9)
  - the battery when a gentle segment starts equals the battery when a hard one starts (0.241 against 0.243, 300 flights)
- **The clock-only R² is right** on synthetic data.
- **End to end:** generation writes the expected files and is reproducible.

**The first dataset (`runs_encoder/pilot_flights/v1`, gitignored):** 19,500 flights with 19,500 distinct seeds, about 5 minutes with 30 processes, 208 MB. Its policy pilots are the 24 memoryless and 56 good models of the step 0, memoryless-control and ideal-estimate reports.

**What each style's `.npz` holds:**
- the inputs `observations` and `prev_actions`
- the label `batteries`
- `privileged_saturated`, `privileged_b0` and `privileged_vx0`, diagnostics that are never inputs
- seeds, lengths, outcomes and the observed starting altitude

`load_flights` returns only the inputs, the label, and each flight's style and seed (an identifier for splitting, never an input).

**Dataset v2 (`runs_encoder/pilot_flights/v2`)** adds 3,000 `switching` flights (mean length 569, 42% completed, 49% of steps at the cap, clock R² 0.32): 22,500 flights, 22,500 distinct seeds.
- `switching` comes last in the style list, so every other style kept its seeds.
- Its 12 other styles are bit-for-bit identical to v1, generated in a separate run: the generation is fully deterministic.
- Clock R² of the mixes in v2: narrow 0.51, main 0.20 (19,000 flights), main_plus_aware 0.21.

| Style | Flights | Mean length | Completed | Steps at the cap | Clock R² |
|---|---:|---:|---:|---:|---:|
| probe_hold | 3,000 | 630 | 64% | 32% | 0.28 |
| steady_below | 2,000 | 680 | 88% | 0% | 0.10 |
| random_hold | 2,000 | 526 | 27% | 44% | 0.37 |
| shaky | 1,000 | 611 | 55% | 22% | 0.29 |
| wavy | 500 | 610 | 57% | 22% | 0.25 |
| hold_speed | 1,500 | 517 | 20% | 74% | 0.41 |
| cruise | 1,000 | 699 | 99% | 2% | 0.04 |
| endurance | 1,500 | 686 | 92% | 16% | 0.12 |
| full_throttle (narrow) | 2,000 | 470 | 1% | 100% | 0.51 |
| blind_ppo | 1,500 | 493 | 18% | 89% | 0.51 |
| replay_good | 2,000 | 620 | 73% | 28% | 0.18 |
| switching (v2) | 3,000 | 569 | 42% | 49% | 0.32 |
| aware_good (separate) | 1,500 | 688 | 95% | 25% | 0.48 |
| **Mix narrow / main / main_plus_aware (v2)** | 2,000 / 19,000 / 20,500 | | | | **0.51 / 0.20 / 0.21** |

In v1, without `switching`, the main and main_plus_aware mixes had 16,000 and 17,500 flights, with clock R² 0.19 and 0.20.

- **The clock is much weaker in the main mix:** it explains 20% of the battery's spread, against 51% in the narrow control and about 45% in A–F's flights.
- **Two first versions of the random styles failed this test.** A shared drain rate let a clock explain 74% (`random_hold`) and 93% (`shaky`), until each flight drew its own throttle level and spread.
- **`blind_ppo` flies like A** (89% of steps at the cap, clock R² 0.51). It is still in the main mix, because a new pilot flies like that early in training.

```bash
# About 5 minutes with 30 processes; the policy pilots load their models from the reports
python pilots.py --output runs_encoder/pilot_flights/v2 --jobs 30 --policy-reports \
    runs_step0/20261014-020124/report_blind_and_A.json runs_step0/20261013-042940/step0_report_rs0.1_seeds0-7.json \
    runs_step0/20261014-044145/step0_report_ideal.json runs_step0/20261014-044145/step0_report_ideal_std.json
python -m unittest checks.step3_checks -v
```

**Next:** offline training of the encoder on these flights (step 4), measured against the ideal estimator at each flight step and for each style, then frozen under plain PPO through the context wrapper.

**Alternative training target: the ideal estimator as a teacher (optional).** Train the encoder to match the calculator's guess (and its uncertainty) instead of, or as well as, the true battery.
- **Why:** where the battery can't be known, such as gentle low-battery flights, the true battery is a noisy target. The calculator's guess is the clean version of what the encoder should output: the best estimate possible from the same inputs. An encoder trained on the true battery with squared error aims for the same thing on average, but learns it from noisier targets.
- **The catch:** it brings knowledge of the physics into training. The ideal estimator knows the exact dynamics, the prior and the speed noise. That is a design choice worth stating openly; the true-battery labels remain the default.
- **How:**
  - Run `IdealBatteryEstimator` over every pilot flight (hidden observations and executed actions) and store its mean and std per step next to the true battery. With pruning that is about 0.3–1 s per flight, minutes for the dataset with 30 processes.
  - Train a point head on the mean, or a Gaussian head on the mean and std (for example by matching the two Gaussians), alone or alongside the true-battery loss.
- **Compare:** encoders trained on the true battery, on the teacher and on both, measured against the ideal estimator at 1, 5 and 20 s into the flight for each pilot style, and then frozen under plain PPO.

### 4. Measure the encoder properly before plugging it in

- **Don't report pooled R².** The clock gets 0.45 of it for free. Report RMSE at 1 s, 5 s and 20 s into the flight, against the clock-only baseline and an ideal estimator. The enc1 study already computes prior and ideal baselines, so reuse that setup.
- **Cross-pilot matrix:** train on one mix, test on each pilot. enc1 suggests pilot variety matters: the xLSTM trained only on shaky flights closed 0–37% of the prior-to-ideal error gap, while the one trained on all flight styles closed 52–68%.
- **Closed-loop check:** after PPO training, measure the encoder's error on the PPO policy's own flights. That's where distribution shift shows up.
- **"Fully trained" means a plateau in validation loss.** For scale, enc1's xLSTM trained for about 7.5 h and its LSTM for about 12 min.

**Offline training (step 4, started 2026-10-15; still running).** `train_encoder.py` trains experiment A's encoder offline on the pilot flights of dataset v2 (main mix):
- **The encoder:** A's exactly. [o_t, a_{t-1}] → linear 6 → 128 → xLSTM stack of width 128 (an mLSTM block, then an sLSTM block) → MLP 128 → 128 → 128 → 1 (tanh) → sigmoid → b̂_t. It runs over whole flights from its initial state, step by step, as CPPO and the context wrapper run it.
- **Designs:** E1 is trained on the true battery b_t (squared error). E3 adds a second head on the same embedding that predicts the starting battery b0 at every step (squared error, weight 1). E3's extra head is dropped when the encoder is saved.
- **Split, by flight seed, the same for every run:** seed % 10 = 8 is validation (early stopping), 9 is test (scored once, with the best checkpoint), the rest is training. That gives 15,200 / 1,900 / 1,900 flights, every style in each.
- **Settings:** batches of 32 whole flights, padding masked, Adam at A's learning rate 3e-4, gradient clipping at 1.0. A run stops after 3 passes without a better validation error, or after 20.
- **Outputs, per run:** the best and last checkpoints, and `encoder_model.zip`, a Contextual_PPO model holding the best encoder for `context.source=encoder`.
- **Launch:** 3 seeds of each design side by side, 2 threads each, W&B project `battery-encoder` (groups `E1_point_main` and `E3_point_b0_main`).
- **Cost:** about 62 (E1) and 65 (E3) minutes per pass. The step-by-step xLSTM training is limited by memory speed: one run alone does about 7,700 flight steps per second, six together about 22,700 in total, and more threads or runs don't help.

**Status after 5 passes (validation, best pass so far):**

| | Typical error | At 1 s | At 5 s | At 20 s | Gap closed |
|---|---:|---:|---:|---:|---:|
| Clock-only estimate | 0.141 | 0.144 | 0.144 | 0.139 | 0% |
| **Floor (ideal estimator)** | **0.084** | 0.133 | 0.096 | 0.071 | 100% |
| E1, seeds 0–2 | 0.127–0.130 | 0.144–0.145 | 0.139–0.140 | 0.119–0.124 | 19–25% |
| E3, seeds 0–2 | 0.128–0.131 | 0.144–0.155 | 0.140–0.148 | 0.120–0.121 | 17–23% |

*Gap closed* = (clock − encoder) / (clock − floor).
- **The encoders have barely moved since pass 2** (best validation error 0.128–0.131 then, 0.127–0.131 now).
- **E3's starting-battery head has stayed at 0.136–0.139:** the same as guessing the average starting battery (0.144). The encoder does not identify the starting battery.

**The floor (`step4_floor.py`, `runs_encoder/step4/floor_v2_main`).** This is the best estimate possible on the same flights from the same inputs: the ideal estimator (`IdealBatteryEstimator`) over the 1,900 validation and 1,900 test flights. It reads hidden observations and executed actions only, knows the physics exactly, and is deterministic.

| | Validation | Test |
|---|---:|---:|
| Floor, typical error | **0.084** | **0.085** |
| Floor at 1 / 5 / 20 s | 0.133 / 0.096 / 0.071 | 0.136 / 0.096 / 0.072 |
| Clock-only estimate | 0.141 | 0.144 |
| Calibration: mean / median of (error / own std)² (1 and 0.45 when calibrated) | 0.86 / 0.42 | 0.88 / 0.40 |
| Largest climb-rate mismatch (tolerance 0.005 m/s) | 0.0016 m/s | 0.0016 m/s |

Per style (validation), the clock, the floor, and the best pass of E1 and E3 seed 0:

| Style | Clock | Floor | E1 s0 | E3 s0 |
|---|---:|---:|---:|---:|
| `blind_ppo` | 0.134 | 0.044 | 0.103 | 0.108 |
| `cruise` | 0.176 | 0.112 | 0.147 | 0.146 |
| `endurance` | 0.145 | 0.055 | 0.137 | 0.139 |
| `hold_speed` | 0.129 | 0.053 | 0.105 | 0.109 |
| `probe_hold` | 0.132 | 0.070 | 0.125 | 0.126 |
| `random_hold` | 0.134 | 0.075 | 0.123 | 0.123 |
| `replay_good` | 0.136 | 0.067 | 0.123 | 0.125 |
| `shaky` | 0.125 | 0.082 | 0.126 | 0.125 |
| `steady_below` | 0.160 | 0.147 | 0.150 | 0.149 |
| `switching` | 0.133 | 0.064 | 0.115 | 0.116 |
| `wavy` | 0.154 | 0.116 | 0.127 | 0.132 |

**What it shows:**
- **The flights hold much more than the encoders learned:** 0.084 against 0.13. The encoders closed only 17–25% of the gap between the clock and the floor.
- **One second into a flight, nobody knows much** (floor 0.133, the spread of starting batteries is 0.144). **By 5 s the ideal estimator knows a lot (0.096); the encoders are still at the clock (0.14).** They barely use the early speed readings.
- **The gap is largest where the battery shows at the cap and must be remembered:** `probe_hold` (0.125 against a floor of 0.070, about the clock's 0.132), `endurance` (0.137 against 0.055), `blind_ppo` and `hold_speed` (0.10–0.11 against about 0.05), `switching` (0.115 against 0.064).
- **Where the battery never shows (`steady_below`), the encoders are at the floor (0.150 against 0.147):** nothing more is possible there.
- **Counting the drain is partly learned:** on `cruise`, where only counting helps, 0.176 (clock) → 0.147, against a floor of 0.112.
- **So the encoder learned the clock and some drain counting, but not the second job:** reading the battery from the speed at the cap and holding on to it. That matches E3's head staying at the prior.
- **The floor itself is trustworthy:** its stated uncertainty matches its errors (median 0.40–0.42 against 0.45 for a calibrated estimate), and its grid is fine enough (climb-rate mismatch 0.0016 m/s).
- **Grid:** step 2's grid (101 × 301, with pruning) is the default. The full grid (201 × 601) gave the same estimates within 0.003 (mean 0.0002) on 160 flights, but took 12 times longer (126 s against 10 s) because its arrays spill out of the CPU caches and run at memory speed beside the training runs. The floor took 3.1 minutes with 16 processes.

**Diagnostics (built, not yet run): `step4_probes.py`.** These test whether the stall comes from xLSTM or from the training, on the same split and error measures:
- **Test tasks, each with a label computed from the flights:**
  - `count`: thrust used so far, dt/E × Σ(a_x² + a_y²) over the executed actions. It is fully determined by the inputs, so its floor is 0. Tests counting.
  - `find`: the starting battery, on full-throttle flights, where the speed gives it away. Tests reading the battery from the speed.
  - `battery`: the real task.
- **Architectures:** A's xLSTM, and PyTorch's GRU and LSTM (2 layers of width 128, A's head). The GRU and LSTM read whole flights in optimised code: about 4.4 and 2.8 minutes per pass over all training flights, against about an hour for the xLSTM.

```bash
python train_encoder.py --design E1 --seed 0 --output-root runs_encoder/step4      # E3: --design E3
python step4_floor.py --output runs_encoder/step4/floor_v2_main --jobs 16           # about 3 minutes
python step4_probes.py --arch gru --task battery --seed 0                           # tasks count, find, battery; archs xlstm, gru, lstm
python -m unittest checks.step4_checks -v
```

### 5. Protocol

- Use at least 3 seeds, preferably 5. Runs collapse unpredictably (D_s0, F_s0).
- Score each run by the evaluation curve, or the mean over late training, not only the final checkpoint. Observed PPO peaked around 360 at 600k steps and ended at 212.
- Keep the 2M-step budget and validation seeds 1000–1002. Leave the final-test seeds 2000–2002 untouched.
- Write the decision rules from sections 0 and 1 down before running anything.
- Optional extra condition: start from the pretrained encoder, then keep training it online the way CPPO does now. That shows whether online training helps or hurts once the encoder starts out good.

Steps 1–5 originally listed 8 conditions × 3 seeds = 24 PPO runs: oracle, three noisy oracles, clock-only, and three frozen encoders (one per mix).
- **Done:** the oracle (step 0's 0a, 0b and 0c, 8 seeds) and the noisy oracles (step 1, plus its switch, early-noise and in-flight variants). The ideal-estimate runs (step 2) were added; they compared against the 0a / 0b runs from an earlier launch with the same code path, and the swap check backs that comparison.
- **Still open:** the frozen encoders, and the clock-only control. The clock-only control is less urgent now: the memoryless controls and A's clock-like encoder already sit at the bottom of the range.

## Next steps

**Done:** the CPPO oracle bypass and step 0; step 1 (noisy oracle, fly-with-noise, late switch, early and in-flight noise, the A encoder's error over a flight, the ideal estimator); the memoryless controls; the context wrapper and plain PPO with the ideal estimate (step 2: the encoder is the bottleneck); the pilot generator and dataset v2 (step 3).

1. **Train the encoder offline on dataset v2 (step 4).** Running since 2026-10-15 (`train_encoder.py`, E1 and E3, 3 seeds each).
   - The xLSTM on the `main` mix, with an LSTM for comparison. Also train on `narrow` (the A-like control) and on `main_plus_aware` (the leak test).
   - Split by whole flights for validation. Keep some styles out entirely for the cross-pilot matrix.
   - Train until the validation loss plateaus. Reuse the enc1 training setup where it fits; its input is `pilots.load_flights` (observations, previous actions, true battery, style).
   - Save it so `FrozenEncoderContext` can load it. Done: each run saves `encoder_model.zip`, a Contextual_PPO model holding the best encoder.
2. **Measure it before plugging it in** (section 4). The floor is measured (0.084 on validation, step 4 results above); the encoders close only 17–25% of the gap so far. Next, the diagnostic probes (`step4_probes.py`: counting, finding, GRU / LSTM). Still to do:
   - Error at 1, 5 and 20 s and over the whole flight, for each style.
   - Compare against the clock-only estimate and the ideal estimator (`step1_ideal_estimator.py`) on the same flights.
   - Run the leak test: test the `main_plus_aware` encoder on battery-blind pilots.
3. **Freeze it under plain PPO**, with `context.source=encoder` and `context.encoder_model=...`. Use 8 seeds at step 0's settings (gamma 0.99, reward scale 0.1).
   - Compare against the ideal estimate (283 / 323), the true battery (246 / 258) and memoryless PPO (23 / 77).
   - Then the closed-loop check: the encoder's error on the PPO policy's own flights.
4. **Optional:**
   - Train on the ideal estimator as a teacher (section 3).
   - Freeze A's own encoder under plain PPO (step 2's next steps).
   - Add earlier PPO checkpoints (0.5M / 1M / 1.5M steps) or more seeds as pilots, if the encoder's error is limited by data.
   - Start from the pretrained encoder and keep training it online, as CPPO does (section 5).
