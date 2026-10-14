# Planned experiment: pretrained frozen encoder vs. policy

**Status:** step 0 and step 1's noisy oracle, fly-with-noise, late-switch, early-noise and in-flight-noise tests done, and the A rerun's encoder error measured over a flight (2026-10-13), the memoryless controls at step 0's settings, and the ideal estimator on the validation flights (2026-10-14; results below); the rest of steps 1–5 planned.

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

### 4. Measure the encoder properly before plugging it in

- **Don't report pooled R².** The clock gets 0.45 of it for free. Report RMSE at 1 s, 5 s and 20 s into the flight, against the clock-only baseline and an ideal estimator. The enc1 study already computes prior and ideal baselines, so reuse that setup.
- **Cross-pilot matrix:** train on one mix, test on each pilot. enc1 suggests pilot variety matters: the xLSTM trained only on shaky flights closed 0–37% of the prior-to-ideal error gap, while the one trained on all flight styles closed 52–68%.
- **Closed-loop check:** after PPO training, measure the encoder's error on the PPO policy's own flights. That's where distribution shift shows up.
- **"Fully trained" means a plateau in validation loss.** For scale, enc1's xLSTM trained for about 7.5 h and its LSTM for about 12 min.

### 5. Protocol

- Use at least 3 seeds, preferably 5. Runs collapse unpredictably (D_s0, F_s0).
- Score each run by the evaluation curve, or the mean over late training, not only the final checkpoint. Observed PPO peaked around 360 at 600k steps and ended at 212.
- Keep the 2M-step budget and validation seeds 1000–1002. Leave the final-test seeds 2000–2002 untouched.
- Write the decision rules from sections 0 and 1 down before running anything.
- Optional extra condition: start from the pretrained encoder, then keep training it online the way CPPO does now. That shows whether online training helps or hurts once the encoder starts out good.

Steps 1–5 currently list 8 conditions × 3 seeds = 24 PPO runs: oracle, three noisy oracles, clock-only, and three frozen encoders. Step 0 is a separate nine-run screen (0a, 0b and 0c, three seeds each). Review its results before committing to those later runs. Reuse 0a as the later oracle only if the wrapper and training/evaluation protocol are verified equivalent.

## Next steps

The enc1 setup on the server already covers the offline training step. The new pieces are:

1. **The CPPO oracle bypass.** Done: `cppo.context_source=oracle`, verified by `checks.step0_checks` (see Step 0).
2. **The step 0 runs.** Done: CPPO can use the true battery, with discount 0.99 and reward scale 0.1 (Step 0 results). The A rerun with these settings scored 22, near the blind level, so the encoder side is the problem. Next: step 1.
3. **The context wrapper.** If Step 0 supports proceeding, implement the frozen-encoder and control inputs for plain PPO.
4. **The pilot generator.**
5. **The remaining controls.** Verify whether Step 0's plain-PPO oracle can be reused; add the clock-only and noisy-oracle conditions when the follow-up scope is fixed.
