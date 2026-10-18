# Planned experiment: pretrained frozen encoder vs. policy

**Status:** step 0 and step 1's noisy oracle, fly-with-noise, late-switch, early-noise and in-flight-noise tests done, and the A rerun's encoder error measured over a flight (2026-10-13), the memoryless controls at step 0's settings, the ideal estimator on the validation flights, the context wrapper, and plain PPO with the ideal estimate (2026-10-14; results below), the pilots and their flights (step 3: dataset v2 with style switching, 2026-10-15), and offline training of the encoder on those flights (step 4: E1 and E3 running since 2026-10-15) with its floor measured, and diagnostic probes (2026-10-16: a GRU encoder learns the task, 0.100 against a floor of 0.085 and 0.144 for the clock), the xLSTM given the same time (it learns it too) and the 2 × 2 test of a ConvGRU front-end and dropping `tau` (2026-10-17: neither helps; the plain GRU reaches 0.097), and plain PPO with the frozen GRU (2026-10-17: 209 at batch 256, against 23 without memory, 246 with the true battery and 283 with the ideal estimate), and the closed-loop check (2026-10-17: on PPO's own flights the GRU overestimates the battery after probing, which caused the one collapse; results below), and joint training without pilots or freezing (J0-J5: Contextual PPO with a GRU encoder, 30 runs, 2026-10-17: the best rung, J1 with 16 times more encoder training, scores 114, no better than without memory; results below).

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

**Status after 6 passes (2026-10-15 22:45; validation, best pass so far):**

| | Typical error | At 1 s | At 5 s | At 20 s | Gap closed |
|---|---:|---:|---:|---:|---:|
| Clock-only estimate | 0.141 | 0.144 | 0.144 | 0.139 | 0% |
| **Floor (ideal estimator)** | **0.084** | 0.133 | 0.096 | 0.071 | 100% |
| E1, seeds 0–2 | 0.126–0.128 | 0.144 | 0.138–0.139 | 0.118–0.120 | 24–26% |
| E3, seeds 0–2 | 0.127–0.128 | 0.144–0.145 | 0.140–0.141 | 0.119–0.121 | 22–24% |

*Gap closed* = (clock − encoder) / (clock − floor).
- **The encoders have barely moved since pass 2** (best validation error 0.128–0.131 then, 0.126–0.128 now).
- **E1 seed 2 stopped first** (after pass 6, best pass 3, 6.3 h). Its test error is 0.129, against 0.144 for the clock and 0.085 for the floor: 26% of the gap, the same picture as on validation.
- **E3's starting-battery head has stayed at 0.136–0.137** (one pass at 0.149–0.152): the same as guessing the average starting battery (0.144). The encoder does not identify the starting battery.
- **Training error is as high as validation error** (about 0.131 against 0.127 at pass 5): the encoder underfits; it isn't memorising flights.

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

**Diagnostics (2026-10-15 22:18 to 2026-10-16 00:44): `step4_probes.py`.** These test whether the stall comes from xLSTM or from the training, on the same split and error measures:
- **Test tasks, each with a label computed from the flights:**
  - `count`: thrust used so far, dt/E × Σ(a_x² + a_y²) over the executed actions. It is fully determined by the inputs, so its floor is 0. Tests counting.
  - `find`: the starting battery, on full-throttle flights, where the speed gives it away. Tests reading the battery from the speed.
  - `battery`: the real task.
- **Architectures:** A's xLSTM, and PyTorch's GRU and LSTM (2 layers of width 128, A's head). The GRU and LSTM read whole flights in optimised code: about 4.4 and 2.8 minutes per pass over all training flights, against about an hour for the xLSTM.
- **Checks (`checks.step4_checks`):**
  - The count label adds exactly dt/E × (a_x² + a_y²) of the previous action at every step, from the actions only.
  - No encoder looks ahead: changing a flight after a step never changes the estimates up to it, for the xLSTM, GRU and LSTM. So no encoder can use how long a flight lasts, which on full-throttle flights gives the starting battery away.
  - Padding changes nothing, and runs are reproducible.
  - Four flights of every style of dataset v2, replayed in a hidden-mode plane with the same seed and actions, match the recorded inputs and labels within 3e-8.
- **Launch (22:18):** the GRU on all three tasks, seeds 0–2, 1 thread each at low priority (`nice`), beside the E1/E3 runs. A pass takes 8.1–8.7 minutes for `battery` and `count`.

**GRU probe results (all 12 runs finished 2026-10-16 00:44; test flights, best checkpoint):**

| Task | GRU, seeds 0 / 1 / 2 | Clock-only | Floor |
|---|---:|---:|---:|
| **`battery`** (the real task) | **0.101 / 0.100** / 0.127 | 0.144 | 0.085 |
| `battery` at 1 / 5 / 20 s, seeds 0–1 | 0.143–0.146 / 0.109–0.111 / 0.088–0.090 | 0.145 / 0.147 / 0.142 | 0.136 / 0.096 / 0.072 |
| `count` | 0.0024 / 0.0029 / 0.0022 | 0.083 | 0 |
| `find`, 20,000 full-throttle flights | 0.063 / 0.064 / 0.062 | 0.124 | about 0.057 |
| `find` at 5 / 20 s | 0.061–0.069 / 0.028–0.033 | 0.143 / 0.094 | about 0.060 / 0.022 |
| `find`, dataset v2 (1,600 flights; too short) | 0.124 / 0.123 / 0.122 | 0.122 | about 0.057 |

The `find` floor is the ideal estimator's posterior mean of b0 on the first 200 test flights, not yet on all 2,000.

**The GRU learned the real task.**
- Seeds 0 and 1 closed **73–75%** of the gap between the clock and the floor (test errors 0.101 and 0.100). The xLSTM runs that have finished closed 26% (E1 seed 2: 0.129, E3 seed 2: 0.129).
- **It "clicked" after a plateau.** Validation by pass:
  - seed 0: 0.138, 0.131, 0.129, 0.134, 0.127, 0.126, 0.124, 0.127, 0.121, **0.108**, 0.104, 0.108, 0.101, …
  - seed 1: 0.136, 0.130, 0.129, 0.130, 0.130, 0.126, 0.125, 0.125, 0.122, 0.116, 0.112, 0.106, 0.103, … 0.099
  - It sat near the clock for 8–9 passes (about 4,500 gradient steps), then dropped to about 0.10 within 3–4 passes.
- **Seed 2 never got there:** early stopping ended it at pass 7 (best pass 4), before the click. Three passes without improvement is too impatient for this task.
- **Seed 1 hit the 20-pass limit with its best result on the last pass:** it was still improving.
- A pass took about 7–8 minutes (1 thread, beside the xLSTM runs); the runs took 2.0–2.3 h.

**Per style (test):** the GRU now reads the battery at the cap and keeps it, where the xLSTM does not.

| Style | Clock | Floor | xLSTM (E1 s2) | GRU s0 | GRU s1 |
|---|---:|---:|---:|---:|---:|
| `blind_ppo` | 0.141 | 0.049 | 0.112 | 0.061 | 0.061 |
| `cruise` | 0.172 | 0.122 | 0.147 | 0.133 | 0.136 |
| `endurance` | 0.139 | 0.055 | 0.130 | 0.068 | 0.067 |
| `hold_speed` | 0.126 | 0.054 | 0.111 | 0.069 | 0.063 |
| `probe_hold` | 0.131 | 0.068 | 0.126 | 0.089 | 0.085 |
| `random_hold` | 0.129 | 0.072 | 0.114 | 0.102 | 0.102 |
| `replay_good` | 0.148 | 0.069 | 0.131 | 0.083 | 0.081 |
| `shaky` | 0.133 | 0.097 | 0.128 | 0.121 | 0.120 |
| `steady_below` | 0.173 | 0.144 | 0.147 | 0.147 | 0.149 |
| `switching` | 0.137 | 0.067 | 0.124 | 0.091 | 0.089 |
| `wavy` | 0.158 | 0.112 | 0.149 | 0.135 | 0.133 |

- The largest gains are on the styles where the battery shows at the cap: `blind_ppo`, `endurance`, `hold_speed`, `replay_good`, `probe_hold`, `switching`. There the GRU is within 0.01–0.025 of the floor.
- On `steady_below` (the battery never shows), everyone is at the floor.
- The GRU is furthest from the floor on `cruise`, `wavy`, `shaky` and `random_hold`. These are flights where the information is partial and has to be combined with exact drain counting.

**Both skills, alone, are learnable:**
- **Counting:** the GRU's error is 0.002–0.003, against 0.083 for the clock (the count ends at 0.21 on average, at most 0.70). It is almost exact.
- **Finding, given enough training:** on 20,000 full-throttle flights the GRU reached 0.062–0.064, near the floor (about 0.057), and 0.028–0.033 at 20 s (floor about 0.022). It also clicked: validation sat at 0.122–0.129 for 2–3 passes, then fell to about 0.08 in one pass.
- **On dataset v2's 1,600 flights,** `find` never got past the clock (below).

**`find` on dataset v2's 1,600 full-throttle flights (first runs):**
- **On `find`, the GRU learned only the clock:**
  - On full-throttle flights the clock knows something: a plane still flying late probably started with more battery, since low-battery flights run out first.
  - The GRU didn't use the speed, while the ideal estimator's estimate of b0 (its posterior mean over the starting batteries, on the same 200 test flights) is within 0.022 by 20 s.
- **But that `find` run was too short to judge:**
  - With 1,600 training flights a pass is only 50 gradient steps, and the runs stopped after 350–600 steps. E1 does 475 per pass.
  - With only 200 validation flights the error jumped by chance (0.122–0.134), so early stopping came too soon.

**`find` again, on 20,000 full-throttle flights (`runs_encoder/pilot_flights/ft20k`, launched 22:41).**
- `pilots.py --styles full_throttle --flights 20000` generated them in 73 seconds with 12 processes.
- **Checked:**
  - The first 2,000 are bit-for-bit those of dataset v2.
  - The 20,000 seeds (8,100,000–8,119,999) are distinct and overlap no other style. No held-out seed is among them.
  - The physics settings are the same, and the style doesn't leak.
- **Split:** 16,000 / 2,000 / 2,000, by the same rule. A pass is now about 500 gradient steps, like E1, and early stopping is decided on 2,000 validation flights.
- **W&B:** a run on a non-default dataset adds the dataset folder to its name: group `probe_gru_find_narrow_ft20k`, runs `..._s0/1/2`.
- **Result:** 0.062–0.064 on the test flights (table above); the runs took 1.2–1.5 h, about 6.5 minutes per pass.

**What the probes show:**
- **The task is learnable by a small recurrent encoder trained offline:** the GRU got about three quarters of the way from the clock to the best possible estimate.
- **The difficulty is a long plateau, not an impossible task.** Each encoder first learns the clock and counting. Learning to read the battery at the cap comes much later and suddenly, after about 4,500 gradient steps on the real task. With little data (`find` on 1,600 flights) or impatient early stopping (GRU seed 2, and possibly E1/E3), runs end on the plateau.
- **Whether the xLSTM would click too was open on 2026-10-16 at 00:48; it did (see "The xLSTM clicked too" below).**
  - Its curves so far look like the GRU's before the click: E1 seed 0 0.129, 0.129, 0.130, 0.134, 0.127, 0.126, **0.124** at pass 7, and E3 seed 0 0.123 at pass 7.
  - But it takes about an hour per pass, so the click would come after 9–10 hours if it comes at the same number of steps.
  - E1 seed 2 and E3 seed 2 already stopped, after 6 and 7 passes. On 2026-10-16 at 00:48, the other four were at passes 7–8.

**Next for step 4:**
1. **The GRU on `battery` with patient early stopping** (8 passes without improvement, up to 40 passes), 3 seeds: now the baseline cell of the 2 × 2 plan below.
2. **Give the xLSTM the same chance:** done (`train_encoder.py --resume`, and `find` on 20,000 flights): it clicked too (below).
3. **The `find` floor on all 2,000 test flights** of `ft20k`.
4. **Freeze the best encoder under plain PPO.** For a GRU, the context wrapper needs to load one (`FrozenEncoderContext` reads a Contextual_PPO model with the xLSTM today).

**The xLSTM clicked too (status 2026-10-16 20:28).** Given enough passes, the xLSTM learns the battery the same way as the GRU, after the same plateau:
- **Real task (validation, best pass):**
  - E1 seed 1: 0.100, pass 20 of 20
  - E3 seeds 0 / 1: 0.102 / 0.101
  - E1 seed 0: 0.105, stopped at pass 16
  - the resumed E1 / E3 seed 2: 0.102 / 0.101
  - the GRU: 0.099–0.101
- **The click:** most runs clicked after 9–11 passes, like the GRU. The two runs stopped early at passes 6–7 clicked once resumed with patience 8.
- **`find`, 20,000 full-throttle flights:** 0.061–0.067 after 15 passes, against 0.062–0.064 for the GRU.
- **Cost:** about 75–100 minutes per pass beside the other runs, against about 6–8 for the GRU.
- **So the xLSTM is a fit encoder:** as accurate as the GRU, but about 15 times slower to train.
- **E3's starting-battery head makes no clear difference.**

### Plan: a ConvGRU encoder, a 2 × 2 test, and learned encoders in PPO (2026-10-16)

**Objective.**
1. Find out whether better input processing helps a GRU encoder learn the battery faster (a shorter plateau) and closer to the ideal estimator. Today's GRU is about 0.05 away from the ideal estimate at every point of the flight. Test two changes separately:
   - dropping `tau` (time left) from the encoder's input
   - a short causal filter and a per-step MLP before the GRU
2. Make GRU-type encoders loadable by the context wrapper, so plain PPO can train with a frozen learned encoder: the GRU already trained, and the best cell of the grid.

**The grid (E1's target, 5 seeds per cell):**

| | `tau` kept | `tau` dropped |
|---|---|---|
| **GRU** (linear 6 → 128, GRU 2 × 128, A's head) | the patient baseline, which is also the patient GRU rerun | `tau` alone |
| **ConvGRU** (filter + MLP → GRU 2 × 128, A's head) | the front-end alone | both |

- **Shared settings:** E1's target and loss, dataset v2 main mix, the same split by flight seed, Adam 3e-4, batch 32, clipping 1.0, A's head (128 → 128 → 1, tanh, sigmoid).
- **Patient early stopping:** 8 passes without improvement, at most 40 passes.
- **Both rows keep the same GRU core,** so the comparison isolates the front-end. A small version (GRU 1 × 64) is an optional fifth cell.
- **ConvGRU front-end:**
  - inputs: altitude, noisy speed, climb rate, previous throttle and lift, plus `tau` in the kept column, plus a 0/1 channel marking real frames against start-of-flight padding
  - a causal Conv1d (window 16 = 0.8 s, 32 channels, SiLU)
  - a per-step MLP 32 → 128 → 128 (SiLU), then LayerNorm
  - then the GRU

**Decision rules, set before running:**
- Plateau length = passes until validation first falls below 0.115.
- Final quality = test error overall, at 1, 5 and 20 s, and per style, against the floor.
- A change helps if it improves either by more than the seed spread, in all seeds.
- If only the both-changes cell helps, report the two changes as working together.

**Implementation:**
1. **`recurrent_encoders.py`** (new; project glue, nothing in `contextual_ppo/` changes, so A–F is unaffected):
   - `GRUEncoder`: the probe GRU's exact architecture and parameter names, so the GRUs already trained load directly.
   - `ConvGRUEncoder`.
   - One interface:
     - full-flight mode for training: zero-pad the start, one Conv1d call, one GRU call
     - one-step mode for the wrapper: a rolling buffer of the last 15 frames plus the GRU state, zeroed at episode starts
     - the initial and reset state, and the "drop `tau`" option inside the encoder
   - Full-flight mode refuses sequences with an episode start in the middle.
   - `save_encoder` / `load_encoder`: an `encoder.pt` holding the architecture, its settings and the weights.
2. **`train_encoder.py`:**
   - `--arch {xlstm, gru, convgru}` (default `xlstm`: E1/E3 unchanged, names included), `--drop-tau`, `--hidden`, `--layers`.
   - The b0 head works for every architecture.
   - The new architectures save `encoder.pt`; the xLSTM keeps `encoder_model.zip`.
   - Names like `E1_convgru_notau_point_main_s0`.
   - A resume compares only the code files the original run recorded, so the xLSTM runs stay resumable.
3. **`context_wrapper.py`:** `context.source=encoder` with a `.pt` model steps the standalone encoder (one memory per environment, reset at episode starts, frozen). A `.zip` model keeps the xLSTM path.
4. **Export** the probe GRUs' best checkpoints to `encoder.pt`, so PPO can use the current best GRU at once.
5. **Checks:**
   - full flight and one step at a time agree within 1e-5, for both architectures, with and without `tau`
   - no look-ahead, including through the filter window
   - with `tau` dropped, `tau` never changes the output
   - padding changes nothing
   - the exported probe GRU reproduces its outputs exactly
   - the wrapper's estimates equal full-flight mode on recorded flights, and stay frozen while PPO trains
   - resume works for the new architectures and still for the xLSTM runs
   - distinct names, reproducible runs
6. Smoke runs and timing, then launch: 12 runs, 1 thread each, low priority.
7. Measure against the floor: overall, by flight time, per style, and the encoder-minus-calculator difference.
8. **Afterwards, separately:** plain PPO with the frozen encoder (`context.source=encoder`), 8 seeds at step 0's settings, validation seeds 1000–1002 only. Compare with the ideal estimate (283 / 323), the true battery (246 / 258) and memoryless PPO (23 / 77).

**Doubts:**
- **Dropping `tau` may change little:** a GRU can build its own clock by counting steps. If it matters, it will most likely show in plateau length.
- **The remaining gap may come from training length rather than input processing** (GRU seed 1 was still improving at pass 20). So the comparison is against the patient baseline, not the earlier runs.
- **Click timing varies between seeds;** 3 seeds may not separate small effects (5 per cell is cheap).
- **CPU:** 12 more runs beside xLSTM runs means about 30 of 32 threads, and everything slows down.
- **Start-of-flight padding:** the filter sees 15 padded frames at take-off. The 0/1 channel marks them, and the checks guarantee training and stepping treat them alike.
- **A lower error on pilot flights does not guarantee better PPO:** PPO's own flights differ. The frozen-PPO test and the closed-loop error on PPO's flights decide.
- **The floor stays valid with `tau` dropped:** the ideal estimator never needs it.

**The stopped xLSTM runs, finalized (2026-10-16 20:45).** The six runs still going were stopped by hand at 20:33 and finished with `--finalize`: the best checkpoint is scored on the test flights, and the W&B run gets its test summary and is marked finished.
- **Real task, test error:** E3 seed 1: 0.102 (best pass 18); E3 seed 2, resumed: 0.101 (best pass 17); E1 seed 2, resumed: 0.103 (best pass 17).
- **The runs that ended on their own (validation):** E1 seed 0: 0.105, E1 seed 1: 0.100, E3 seed 0: 0.102.
- **`find`, test error:** 0.066 / 0.061 / 0.062 (best passes 9 / 14 / 11 of 15), against 0.062–0.064 for the GRU.
- **The clock gets 0.144 on the real task and 0.124 on `find`.**
- `--finalize` (`train_encoder.py`, `step4_probes.py`) does what a run's natural end does. A check confirms it gives the same `result.json` and model as a run that finished on its own.

**The 2 × 2 plan, built (steps 1–6, 2026-10-16).**
- **The pieces:**
  - `recurrent_encoders.py`: `GRUEncoder`, `ConvGRUEncoder`, `encoder.pt`, and export of the probe GRUs
  - `train_encoder.py --arch gru | convgru --drop-tau`
  - the context wrapper loads an `encoder.pt`
  - the three probe GRUs are exported to `runs_encoder/step4_probes/probe_gru_battery_main_s{0,1,2}/encoder.pt`
- **Checks (80 in the project pass):**
  - whole flight and one step at a time agree within 1e-5, for both architectures, with and without `tau`
  - no look-ahead, including through the filter window
  - with `tau` dropped, `tau` never changes the output
  - the wrapper's estimates equal whole-flight estimates and stay frozen while PPO trains
  - an exported probe GRU gives exactly the probe's estimates
  - the new architectures train, resume exactly and finalize
  - an xLSTM run from before `--arch` still resumes
- **Sizes:** GRU 232,193 weights; ConvGRU 255,905 (255,393 without `tau`).
- **Timing:** with 20 runs at once (1 thread each), about 9–14 minutes per pass, against about 3.5 minutes alone. The grid should take about 5–7 hours, and at most about 10.
- **Planned launch:** 4 cells × 5 seeds, E1's target, patience 8, up to 40 passes, `--output-root runs_encoder/step4_grid`. Names: `E1_gru_point_main`, `E1_gru_notau_point_main`, `E1_convgru_point_main`, `E1_convgru_notau_point_main` (`_s0` to `_s4`).

**The 2 × 2 results (launched 2026-10-16 21:00 UTC, all 20 runs done by 2026-10-17 07:09 UTC).**
- **Timing:** 20 runs at once, 1 thread each, about 9–10 minutes per pass for the GRU cells and 12 for the ConvGRU cells.
- **Run lengths:** 7 of the 20 runs used all 40 passes: three ConvGRU without `tau`, two ConvGRU, and seed 1 of both GRU cells. The others stopped after 24–32.

| Cell (5 seeds) | Test error, mean (range) | Click at pass* | Gap closed** |
|---|---:|---:|---:|
| **GRU** (with `tau`) | **0.0972** (0.0955–0.0989) | 10–12 | **80%** |
| GRU, no `tau` | 0.0974 (0.0954–0.0986) | 9–13 | 79% |
| ConvGRU (with `tau`) | 0.0990 (0.0980–0.1001) | 13–19 | 77% |
| ConvGRU, no `tau` | 0.0986 (0.0977–0.1005) | 14–16 | 77% |
| Clock-only / floor (ideal estimator) | 0.144 / 0.085 | | 0% / 100% |

\*First pass with validation below 0.115. \*\*(clock − encoder) / (clock − floor), on the test flights.

**By the decision rules set before running:**
- **Dropping `tau` makes no difference,** neither in the final error (0.0972 against 0.0974; 0.0990 against 0.0986) nor in when the click comes. So `tau` was not what made the clock shortcut "sticky": the network builds its own clock by counting steps.
- **The ConvGRU front-end does not help:**
  - The click comes later in every seed (pass 13–19 against 9–13).
  - The final error is about 0.0015 higher, but the seed ranges overlap.
  - The filter and MLP add more to learn before the click and nothing after it.
- **No interaction:** both changes together don't help either.
- **Patience helps a little:** the patient GRU reaches 0.097, against 0.100 for the earlier impatient GRU probes. The longest-trained seed (seed 1, 40 passes) reaches 0.0955.

**Against the ideal estimator (test flights, mean of 5 seeds):**

| | 0–1 s | 1–5 s | 5–10 s | 10–20 s | 20–35 s | Whole flight |
|---|---:|---:|---:|---:|---:|---:|
| GRU minus ideal estimate (RMS) | 0.034 | 0.042 | 0.046 | 0.047 | 0.043 | **0.045** |
| ConvGRU minus ideal estimate | 0.038 | 0.047 | 0.050 | 0.050 | 0.045 | 0.048 |

- **Error at 1 / 5 / 20 s:** GRU 0.142–0.143 / 0.105–0.107 / 0.084–0.089, against 0.136 / 0.096 / 0.072 for the ideal estimator.
- **Per style, GRU against ideal:** `probe_hold` 0.081 / 0.068, `endurance` 0.064 / 0.055, `replay_good` 0.078 / 0.069, `cruise` 0.135 / 0.122, `steady_below` 0.147 / 0.144.
- **The GRU's error is about 14% above the floor.** Its own extra error shrank from about 0.05 (the probes) to 0.045. It stays roughly constant through the flight, while the ideal estimator's error keeps shrinking.
- **Conclusion:** better input processing is not the missing piece. The plain GRU is the best learned encoder. The remaining gap more likely needs a cleaner training signal (the ideal estimator as a teacher), more training or more data.

**The easy environment (EASY_ENV_WARMUP.md), considered 2026-10-17 and kept as plan B.**
- **It would probably help the encoder:** the forced 5-second full-throttle take-off is the `find` situation, which the GRU handles well (about 0.06–0.07 after 5 s, near the floor).
- **Why not now:**
  - PPO already did best with the ideal estimate, whose error early in each flight is just as large, so early uncertainty doesn't seem to block PPO.
  - The easy version needs the environment, the learner, new pilot flights and all baselines redone.
  - It weakens the autogenic claim.
- **When to switch:** if PPO with the frozen GRU falls well short of the ideal estimate, and the shortfall comes from early in each flight.

**Plain PPO with the frozen GRU (prepared 2026-10-17).** This tests whether a learned encoder is enough. It is step 2's test with the GRU's estimate in place of the ideal estimator's.
- **Conditions:** `4a_encoder` / `4b_encoder` (`ppo/configs/experiment/S4a_encoder_bs256.yaml`, `S4b_encoder_bs1024.yaml`): the memoryless 0a / 0b learners in hidden mode, with the frozen encoder's battery estimate appended (`context.source=encoder`, `context.encoder_model`, `context.encoder_tag`).
- **What else is the same:** a check confirms that everything except the context matches the memoryless runs and step 2's ideal-estimate runs: PPO settings, 2M steps, gamma 0.99, reward scale 0.1, 8 environments and the validation grid.
- **W&B names carry the encoder,** e.g. `S4b_encoder_bs1024_E1_gru_point_main_s1_gamma0.99_rs0.1_s0`.
- **Encoders, chosen by validation (not test):** the GRU cell's three best seeds, s1 (0.0945), s0 (0.0967) and s4 (0.0969).
- **Design (32 runs at once, 1 thread each):**
  - GRU s1 under both batch sizes, PPO seeds 0–7 (16 runs)
  - GRU s0 and s4 under batch 1024, seeds 0–7 (16 runs), to see how much the encoder seed matters
- **Seeds kept apart:**
  - The encoders trained only on pilot flights with seeds 100,000–12,102,999 (their training, validation and test splits).
  - The PPO runs use environment seeds 0–707 for training and 1000–1002 for validation. No flight is shared.
  - The final-test seeds 2000–2002 are used nowhere.
  - Some pilot flights were flown by PPO policies from earlier runs (seeds 0–7), but on their own environment seeds.
- **Code:**
  - `run_step0.py --encoder-models` runs each condition once per encoder file and puts each encoder in a report of its own.
  - `report_step0.py` scores `4a_encoder` / `4b_encoder` like step 0: the late mean over 1.5M–2M and a final validation that reloads the frozen encoder.
  - An encoder run refuses to start without both the encoder file and its name.
- **Checks:** 82 in the project pass. A smoke run trained both conditions end to end and recorded the encoder's absolute path, its name, batch 256 / 1024, hidden mode and a policy input of 5 values.
- **The frozen GRU's cost:** it slows environment stepping from about 24,000 to 16,000 steps per second. The runs should take about 30–60 minutes.
- **Compare with** (late score, batch 256 / 1024): ideal estimate 283 / 323, true battery 246 / 258, memoryless 23 / 77. Then measure the encoder's error on the PPO policies' own flights.

```bash
G=runs_encoder/step4_grid
nohup python run_step0.py --conditions 4a_encoder 4b_encoder --encoder-models $G/E1_gru_point_main_s1/encoder.pt \
  --seeds 0 1 2 3 4 5 6 7 --jobs 16 --threads 1 --output runs_step0/frozen_gru_a > run_frozen_gru_a.log 2>&1 &
nohup python run_step0.py --conditions 4b_encoder --encoder-models $G/E1_gru_point_main_s0/encoder.pt $G/E1_gru_point_main_s4/encoder.pt \
  --seeds 0 1 2 3 4 5 6 7 --jobs 16 --threads 1 --output runs_step0/frozen_gru_b > run_frozen_gru_b.log 2>&1 &
```

**Results: plain PPO with the frozen GRU (launched 2026-10-17 08:01 UTC).**
- **All 32 runs finished,** in 26–28 minutes each.
- **Reports:**
  - `runs_step0/frozen_gru_a/step0_report_encoder_E1_gru_point_main_s1.json`
  - `runs_step0/frozen_gru_b/step0_report_encoder_E1_gru_point_main_s0.json`
  - `runs_step0/frozen_gru_b/step0_report_encoder_E1_gru_point_main_s4.json`
- **Late score:** the mean validation return over 1.5M–2M steps, 8 PPO seeds each. The final validation is of the final checkpoint.

| Battery estimate | Batch | Late score (range) | Median | Final | Completed | Battery empty | Crashed |
|---|---:|---:|---:|---:|---:|---:|---:|
| None (memoryless) | 256 | 23 ± 42 (−14–111) | | 28 | 26% | 74% | 0% |
| None (memoryless) | 1024 | 77 ± 32 (42–136) | | 79 | 15% | 85% | 0% |
| **Frozen GRU s1** | **256** | **209 ± 36 (167–261)** | 210 | 204 | **88%** | 12% | 0% |
| Frozen GRU s1 | 1024 | 164 ± 184 (−294–330) | 218 | 157 | 73% | 22% | 5% |
| Frozen GRU s0 | 1024 | 179 ± 64 (69–277) | 192 | 204 | 77% | 16% | 7% |
| Frozen GRU s4 | 1024 | 120 ± 104 (−7–296) | 101 | 125 | 43% | 52% | 5% |
| True battery | 256 | 246 ± 52 (185–349) | 232 | 251 | 88% | 6% | 6% |
| True battery | 1024 | 258 ± 30 (212–304) | 261 | 236 | 86% | 5% | 9% |
| Ideal estimate | 256 | 283 ± 40 (222–355) | 285 | 265 | 88% | 10% | 2% |
| Ideal estimate | 1024 | 323 ± 22 (270–344) | 326 | 308 | 92% | 2% | 6% |

**Late score per PPO seed (seeds 0–7):**
- **GRU s1:**
  - batch 256: 168, 203, 255, 169, 216, 231, 167, 261
  - batch 1024: 330, 177, 123, 280, 262, 249, **−294**, 187
- **GRU s0, batch 1024:** 196, 145, 69, 277, 109, 221, 227, 187
- **GRU s4, batch 1024:** 57, −7, 244, 26, 296, 165, 146, 32

**What it shows:**
1. **Inferring the battery from the history helps a lot.** With the frozen GRU, plain PPO at batch 256 goes from 23 to **209** on all 8 seeds (167–261), and completes **88%** of the validation flights against 26% without memory. That is the same completion rate as with the true battery or the ideal estimate. This is the project's second goal shown with a learned encoder: an agent that must infer its hidden battery from its own history does far better than one that can't.
2. **Not yet as good as perfect inference.**
   - At batch 256, the GRU gets about 72% of the way from memoryless (23) to the ideal estimate (283), and about 83% of the way to the true battery (246).
   - It completes flights as often but earns less on the way.
   - The learner, settings and validation flights are identical, so the remaining gap is the encoder's.
3. **Batch 256 is reliable; batch 1024 is not.**
   - With GRU s1 at batch 1024, one seed collapsed (−294, no flight completed) and one reached only 123. The other six reached 177–330, up to the ideal estimate's level.
   - With the ideal estimate, batch 1024 was the better setting (323 against 283). The GRU's extra error seems to make the larger batch unstable.
4. **The encoder seed matters more than its error on the pilot flights suggests.**
   - GRU s0 and s4 had almost the same validation error (0.0967 and 0.0969), but very different PPO results: 179 against 120, and 77% against 43% completed.
   - The likely reason is that PPO's own flights differ from the pilot flights, so an encoder's error on them can differ even when its pilot-flight error doesn't.

**Next:**
1. **The closed-loop check:** each GRU's error on the PPO policies' own validation flights, against the ideal estimator on the same flights. It should explain the s4 / s0 / s1 differences and the batch-1024 collapses.
2. **Train the encoder on PPO-style flights:** add flights flown by these frozen-GRU PPO policies (as pilots, replayed or direct) to the encoder's data and retrain, closing the gap between the flights it learns from and the flights PPO flies.
3. **A cleaner target:** the ideal estimator as a teacher (section 3), against the steady extra error of about 0.045.

**The closed-loop check (2026-10-17, `step4_closed_loop.py`, `runs_step0/closed_loop_frozen_gru.json`).**
- **What it measures:** the 32 frozen-GRU PPO runs, each flown over its validation flights exactly as its final validation was. The flights reproduce the reports' returns and lengths.
- **What it records at every step:**
  - the GRU's estimate, which the policy read
  - the true battery, only for scoring
  - the ideal estimate from the same hidden observations and executed actions
- **The question:** the encoders were measured on pilot flights (open loop). Inside PPO, the encoder's estimate shapes the flights it then reads (closed loop).

| Group (8 runs each) | Late score | GRU error | Ideal error | GRU minus ideal | GRU bias* |
|---|---:|---:|---:|---:|---:|
| GRU s1, batch 256 | 209 | 0.126 | 0.110 | 0.053 | +0.021 |
| GRU s1, batch 1024 | 164 | 0.119 | 0.102 | 0.051 | +0.016 |
| GRU s0, batch 1024 | 179 | 0.106 | 0.092 | 0.043 | −0.011 |
| GRU s4, batch 1024 | 120 | 0.054 | 0.044 | 0.028 | +0.002 |

\*Mean of estimate − true battery: positive means the GRU thinks there is more battery than there is.

**How a policy flies decides how well anyone can know its battery.** The runs fall into three kinds:
1. **Policies that never probe** (GRU s1 batch 256 seeds 0, 3, 6; batch 1024 seeds 2, 7; GRU s0 seeds 0, 7; scores 123–196):
   - They fly below the cap all flight, so the battery never shows. Even the ideal estimator is off by about 0.16, about as much as knowing nothing; its errors are largest at starting batteries 0.2 and 0.7.
   - The GRU is at the ideal estimate (GRU minus ideal 0.007–0.017). Nothing more could be read.
   - They complete flights cautiously, with middling scores.
2. **Policies that probe** (most of the best runs, e.g. GRU s1 batch 256 seeds 1, 2, 4, 5, 7, scoring 203–261; s1 batch 1024 seeds 0, 3, 4, 5):
   - They fly at the cap for a while, and the ideal estimator gets within 0.02–0.07.
   - The GRU lags (GRU minus ideal 0.03–0.07) and **overestimates the battery after about 5 s, by +0.03 to +0.06.**
   - These best policies would likely do better still with an accurate encoder.
3. **Policies that fly fast** (GRU s4, 6 of 8 seeds; scores −7 to 165; 17–78% completed):
   - They fly hard and often run out, so their flights reveal the battery. The GRU is accurate and unbiased there (GRU minus ideal 0.02–0.05, bias about 0).
   - s4's poor scores are not caused by encoder error on these flights: its policies settled on flying fast. Whether that comes from how s4's estimates behaved during training or from the luck of 8 seeds, this check can't tell.

**The collapse (GRU s1, batch 1024, seed 6, −294):**
- On that policy's flights the GRU is furthest from the ideal estimate of any run (0.126), and it **overestimates the battery by +0.064 on average** (+0.063 at 5–10 s).
- The policy flew as if it had more battery than it did, and ran out on every flight.
- This is a closed-loop failure: on an unusual policy's flights, the encoder's error can grow large and one-sided.

**What it means:**
- **A run's encoder error doesn't predict its score** (correlation −0.08 over the 32 runs): the error depends on how the policy flies.
- **Within a group, the runs that probe score higher** (correlation of error with score −0.84 for GRU s1 at batch 256). Policies that learn to read their battery do best.
- **In closed loop, the GRU's weakness is overestimating the battery after probing.** That hurts the best policies and caused the collapse. Early in flights it slightly underestimates instead (−0.01 to −0.05 in the first second).
- **Fixes this points to:**
  1. Train the encoder on flights of these PPO policies: probe-then-glide flights and the collapsed run's. That is where it is biased.
  2. The ideal estimator as a teacher: its estimate is unbiased given what is visible, which targets the overestimation directly.

```bash
python train_encoder.py --design E1 --seed 0 --output-root runs_encoder/step4      # E3: --design E3
python step4_floor.py --output runs_encoder/step4/floor_v2_main --jobs 16           # about 3 minutes
python step4_probes.py --arch gru --task battery --seed 0                           # tasks count, find, battery; archs xlstm, gru, lstm
python pilots.py --output runs_encoder/pilot_flights/ft20k --styles full_throttle --flights 20000 --jobs 12   # about 1 minute
python step4_probes.py --arch gru --task find --seed 0 --data runs_encoder/pilot_flights/ft20k
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

### Joint training: Contextual PPO with a GRU encoder, no pilots, no frozen encoder (launched 2026-10-17)

**Question.** Experiment A trained its encoder together with the policy and failed (22). The offline encoders and the frozen-GRU test say a learned encoder can work. What does joint training need to work too?

**Diagnosis of why A failed (the loop):**
1. **The encoder trained far too little:** about 1,950 supervised steps of 16 episodes over a 1M-step run, against about 4,300–5,200 steps of 32 flights before the offline encoders left their plateau.
2. **Its data was narrow:** only the current policy's flights. The episode buffer (200,000 steps) drops the oldest episodes, so the varied early flights were soon gone.
3. **The policy stopped exploring** (`ent_coef` 0).
4. **The two locked each other in:** the clock-like estimate taught the policy to ignore it, so it flew the same narrow way, so the clock stayed "good enough".

**The ladder.** Each rung adds one change, at the A rerun's settings: hidden mode, gamma 0.99, reward scale 0.1, 2M steps, 5 seeds (0–4). Configs are in `config/joint/`; launcher conditions are `J0`–`J5`.

| Rung | Change | Question |
|---|---|---|
| J0 | A's encoder budget (8 supervised steps of 16 episodes per update, newest 200,000 steps kept), with a GRU | does the GRU alone change anything? |
| J1 | 32 steps of 32 episodes per update (about 15,600 in all, about 3× the offline plateau), every finished episode kept | was the training budget the problem? |
| J2 | J1 + entropy bonus 0.01 | does keeping exploration help? |
| J3 | J2 + Gaussian head (beta-NLL, beta 0.5); the policy reads the mean and log variance | does the belief let the policy learn when to probe? |
| J4 | J3 + curiosity about the battery: a training-reward bonus of 1.0 × the drop in the head's entropy from each observation to the next, falling linearly to 0 at 1M steps | does curiosity add anything? |
| J5 | J1 + a warm-up: no PPO update for the first 200,000 steps, so the untrained policy keeps flying in varied ways while the encoder trains on those flights (128 steps per update) | does a self-made warm-up help? |

**Compare with:**

| Reference | Score |
|---|---:|
| A rerun (xLSTM) | 22 |
| Memoryless Contextual PPO | 115 |
| Contextual PPO with the true battery | 282 |
| Plain PPO with the frozen GRU (batch 256) | 209 |
| Plain PPO with the ideal estimate | 283 / 323 |

**Code (only the new options; the xLSTM default, A–F and step 0–2 are unchanged):**
- `contextual_ppo/torch_layers.py`: `GRURolloutEncoder`, with the xLSTM encoder's interface (`step`, `unroll` over episodes or rollout streams with episode starts, the auxiliary head, point and Gaussian heads).
- `cppo.encoder_arch: xlstm | gru` and `cppo.gru` (128 wide, 2 layers) in the policy (saved with the model) and in the main script.
- The rollout buffer knows both architectures' state layout.
- `ContextualPPO(policy_warmup_steps, supervised_warmup_gradient_steps, curiosity_weight, curiosity_until)`, set by `cppo.policy_warmup_steps`, `cppo.supervised.warmup_gradient_steps` and `cppo.curiosity`, all off by default:
  - **The warm-up** skips the PPO epochs and trains only the encoder.
  - **The curiosity bonus** goes into the rollout buffer's intrinsic rewards. Stored rewards, episode returns and evaluation stay the environment's.
- `run_step0.py` / `report_step0.py`: conditions `J0`–`J5`, each in a report of its own, checked to be a supervised GRU encoder. The report's check for A now also refuses a non-xLSTM encoder.

**Checks (`checks.joint_checks`; 89 in the project pass):**
- **The GRU's paths agree within 1e-5:** a whole sequence, one step at a time, episodes padded at the end (each row's final state included), and episode starts inside a stream.
- **Re-encoding a rollout during the PPO update reproduces collection exactly:** values, log-probabilities and contexts, for the point (J1) and Gaussian (J3) heads, over several rollouts with episodes ending inside the streams.
- The model trains, saves and reloads with the GRU; old saved models keep the xLSTM.
- **Each rung differs from its parent by its one change only.**
- **The curiosity bonus is exactly weight × entropy drop,** in the intrinsic rewards only, reaching the advantages.
- **The warm-up leaves the actor and critic untouched** while the encoder trains, and PPO starts afterwards.
- **Seeds:** the 30 runs use training environment seeds 0–407, never the validation (1000–1002) or final-test (2000–2002) seeds. W&B names are distinct, ending in `_v2`, e.g. `cppo_J1_gru_budget_gamma0.99_rs0.1_v2_s0`.

**A performance bug found and fixed before the real launch:**
- A first launch at 13:34 used PyTorch's packed sequences for the padded episode batches. Their backward pass is very slow on CPU: 90 s per supervised step under load, against 3.7 s unpacked. J1 would have needed days.
- Since the padding is at the end and the GRU is causal, one unpacked call gives the same outputs on every real step. The state after each row's last real step comes from a cheap packed pass without gradient. Now 0.51 s per step on an idle machine.
- **The 20 runs of the first launch were stopped at about 10,000–30,000 steps.** Their W&B runs (names without `_v2`) are leftovers that can be deleted.

```bash
python run_step0.py --conditions J0 J1 J2 J3 J4 J5 --seeds 0 1 2 3 4 --jobs 30 --threads 1 --study v2 --output runs_step0/joint_v2
python -m unittest checks.joint_checks -v
```

**Results (launched 2026-10-17 12:03 UTC, all 30 runs done by 22:47 UTC).**
- **Run times:** J0 took 2.7–2.8 h; J1–J4 9.5–9.7 h; J5 10.1–10.2 h, with the 30 runs side by side, 1 thread each.
- **Reports:** `runs_step0/joint_v2/step0_report_joint_J0.json` … `_J5.json`.
- **Scoring:** the late score is the mean validation return over 1.5M–2M steps, 5 seeds (0–4). The final validation is of the final checkpoint. W&B groups are `cppo_J<k>_…_gamma0.99_rs0.1_v2`.

| Rung | Late score (median) | Seeds 0–4 | Final | Completed | Battery empty | Encoder on its own flights at the end (validation R², RMSE) |
|---|---:|---|---:|---:|---:|---|
| J0: A's budget, GRU | 45 ± 37 (55) | 102, 2, 55, 8, 55 | 40 | 14% | 86% | 0.84, 0.060 |
| **J1: 16× encoder training, all episodes kept** | **114 ± 38 (98)** | 139, 88, 70, 98, 174 | 121 | 23% | 77% | 0.84, 0.060 |
| J2: J1 + entropy bonus | 63 ± 26 (51) | 65, 51, 39, 112, 49 | 69 | 9% | 91% | 0.82, 0.062 |
| J3: J2 + Gaussian head | 70 ± 50 (81) | 142, 48, −8, 81, 88 | 71 | 31% | 68% | 0.82, 0.063 |
| J4: J3 + curiosity | 83 ± 49 (72) | 72, 62, 18, 97, 166 | 61 | 27% | 73% | 0.79, 0.068 |
| J5: J1 + warm-up | 94 ± 55 (100) | 4, 100, 66, 155, 145 | 103 | 26% | 74% | 0.70, 0.083 |

**For comparison** (late score, same settings):

| Reference | Score |
|---|---:|
| A rerun (xLSTM, 3 seeds) | 22 |
| Memoryless Contextual PPO | 115 |
| Contextual PPO with the true battery | 282 |
| Plain PPO with the frozen GRU, batch 256 | 209 |
| Plain PPO with the ideal estimate | 283 / 323 |

**What it shows:**
- **Joint training still doesn't work.** No rung beats Contextual PPO without memory (115), and all stay far below the frozen GRU trained on pilot flights (209) and the true battery (282). The planes still run out of battery on 68–91% of the validation flights.
- **More encoder training is the only change that helps:** J0 → J1, 45 → 114. Even so, it only reaches the memoryless level.
- **The GRU alone changes little:** J0 at 45 against A at 22, with a large spread (2–102).
- **The extras don't add anything on top of J1:**
  - The entropy bonus makes it worse (63).
  - The Gaussian head (70) and the curiosity bonus (83) don't recover J1's level, though they complete a few more flights (27–31%).
  - The warm-up (94) has a wide spread (4–155), and its encoder fits its own flights worst (R² 0.70).
- **The encoder again fits mostly a clock:** on its own policy's flights it reaches validation R² 0.79–0.84 (RMSE about 0.06). That is like experiment A, whose encoder looked accurate on its own flights while it had learned "battery falls a fixed amount per step".
- **The loop holds:** the policy flies fast and narrowly, so the flights stay narrow, so the clock stays good enough for the encoder, so the policy has no reason to use it. More encoder training, exploration, an uncertainty input, curiosity and a warm-up on random flights didn't break it within 2M steps.
- **What did break it:** training the encoder first on varied flights from many pilots, then freezing it (step 4: 209).
- **Possible next steps for joint training:**
  - Start Contextual PPO from the pretrained GRU (frozen at first, then trained online), testing whether online training keeps or loses what the pilots taught.
  - Mix pilot flights into the joint encoder's training data.

### Teacher targets: T1 and T2 (launched 2026-10-18 00:31 UTC)

**The question:** does the GRU get closer to the floor if it learns the ideal estimator's (the calculator's) belief instead of the true battery? Where the battery can't be known, the true battery is a noisy target; the calculator's mean is the clean version of the best possible guess (section 3's alternative target). It targets the GRU's steady extra error of about 0.045 and its overestimation after probing in closed loop.
- **T1:** the GRU's point head learns the calculator's mean (squared error).
- **T2:** a Gaussian head (a mean and a variance) learns the calculator's mean and spread.
  - Loss: the expected negative log-likelihood under the calculator's belief, 0.5 log v + (s² + (m − μ)²) / (2v), which is smallest at m = μ and v = s².
  - Beta-NLL with β = 1, so the mean learns as in T1, and T2 differs from T1 only by also learning the spread.
  - The calculator's std is floored at 0.005, its grid spacing in b0. Below that, the std reflects the grid, not the physics.
- **Everything else is the 2 × 2 grid's GRU cell** (the baseline, test 0.0972, 0.0955–0.0989):
  - the same flights and split
  - batch 32, learning rate 3e-4, patience 8, up to 40 passes, 1 thread
  - early stopping and every reported error against the true battery
- **Paired with the baseline:** T1 at seed s starts from the same weights as the baseline GRU at seed s and sees the batches in the same order, so only the target differs. T2's head has one more output, which changes its initial weights.
- **The catch, stated openly:** the targets bring knowledge of the physics into training (the exact dynamics, the prior and the speed noise).

**Teacher targets:** `step4_floor.py --splits train validation test` ran the calculator over all 19,000 main-mix flights of dataset v2 in 20 minutes with 30 processes (`runs_encoder/step4/teacher_v2_main`).
- **Validation and test estimates:** bit-identical to the floor's (`floor_v2_main`): validation 0.0841, test 0.0853.
- **On the training flights:** error 0.0841 against the true battery, bias −0.0006.
- **Calibration:** 1.09 on the training flights (mean squared error / its own variance).

**Code:**
- `recurrent_encoders.py`: an optional Gaussian head, `context_head="gaussian"` (mean through the sigmoid; variance softplus + 1e-6, as A's head).
  - `forward` and `step` still return the mean, so the context wrapper reads the mean.
  - `forward_with_variance` and `step_with_variance` return the variance too.
- `train_encoder.py`:
  - New options: `--target teacher --teacher <dir>`, `--context-head gaussian`, `--nll-beta`, `--teacher-std-floor`.
  - The validation metrics add the error against the calculator's mean and, for T2, its calibration and how far its std is from the calculator's.
  - Names: `E1_gru_teacher_point_main_s0` … `_s4` (T1) and `E1_gru_teacher_gauss_main_s0` … `_s4` (T2).
  - It records the teacher's path and SHA-256.
  - A teacher computed on other flights, or missing any flight, is refused.

**Checks (92 in the project pass):**
- With the true battery as the "teacher", training gives exactly E1's weights and validation errors.
- Replacing the validation and test flights' targets with nonsense changes neither T1's nor T2's weights, nor their validation errors against the battery. Only training flights' targets reach the loss.
- The Gaussian head's two modes agree within 1e-5, and it saves and loads.
- The loss is smallest at the calculator's belief, and β = 1 gives the mean half the squared error's gradient.
- T2 resumes exactly and finalizes as its natural end.
- Options that don't go together are refused.
- The old code (commit 71f3ffd) and the new code give bit-identical weights, validation errors and results for the battery-target GRU. The grid's GRU s0 gives its recorded validation error (0.096684) in the new code.
- A bug found and fixed during the checks: the teacher file was read once per flight (each read decompresses the whole array). It now loads in 0.3 s.

**Seeds:**
- The flights have seeds from 100,000 up. The environment's validation (1000–1002) and final-test (2000–2002) seeds appear nowhere.
- The calculator is deterministic and has no seed. Its prior is the environment's reset distribution, not fitted to any flights.
- The calculator's flag "the plane still flies" at step t rules out an empty battery. That is known at every decision step, because an empty battery ends the flight on that same step. At the last step it is not used.

**Timing:** with 10 runs side by side, extra threads don't help (5.3, 5.2 and 5.8 minutes per pass at 1, 2 and 3 threads each: memory-bound). With validation, about 5.5 minutes per pass. The baseline's runs took 24–40 passes, so expect about 2.2–3.7 hours.

```bash
R=runs_encoder/step4_teacher; TT=runs_encoder/step4/teacher_v2_main; mkdir -p $R/logs
for s in 0 1 2 3 4; do
  nohup python train_encoder.py --design E1 --arch gru --target teacher --teacher $TT --seed $s --patience 8 --max-epochs 40 --threads 1 --output-root $R > $R/logs/teacher_point_s$s.log 2>&1 &
  nohup python train_encoder.py --design E1 --arch gru --target teacher --teacher $TT --context-head gaussian --seed $s --patience 8 --max-epochs 40 --threads 1 --output-root $R > $R/logs/teacher_gauss_s$s.log 2>&1 &
done
```

**Compare (test flights):** the GRU cell 0.0972, the floor 0.0853 and the clock 0.1438, overall, at 1, 5 and 20 s, and per style. Also the RMS of the GRU's estimate minus the calculator's (the baseline's is 0.045). If a teacher encoder clearly beats the baseline, freeze it under plain PPO as in step 4 (209 at batch 256).

**Results (all 10 runs done by 2026-10-18 04:10 UTC, 3.0–3.5 h each).**
- **Run lengths:** 9 of 10 runs used all 40 passes. T1's best passes were 32–37. T2's were 27–40, and 3 of its seeds were still improving at pass 40.
- **Scoring:** best checkpoint by validation, test flights, against the true battery.

| Encoder (5 seeds) | Test error, mean (range) | Gap closed* | Minus the calculator** | Bias | At 1 / 5 / 20 s | Click at pass*** |
|---|---:|---:|---:|---:|---|---:|
| Clock only | 0.1438 | 0% | | | | |
| GRU cell (true battery) | 0.0972 (0.0955–0.0989) | 80% | 0.0445 | −0.0064 | 0.142 / 0.106 / 0.086 | 10–12 |
| **T1** (the calculator's mean) | **0.0933 (0.0928–0.0938)** | **86%** | **0.0349** | −0.0035 | 0.141 / 0.103 / 0.081 | **7–8** |
| T2 (mean and spread) | 0.0934 (0.0930–0.0942) | 86% | 0.0357 | −0.0059 | 0.141 / 0.103 / 0.082 | 11–14 |
| Calculator (the floor) | 0.0853 | 100% | 0 | −0.0007 | 0.136 / 0.096 / 0.072 | |

\*(clock − encoder) / (clock − floor). \*\*RMS of the encoder's estimate minus the calculator's, at every step. \*\*\*First pass with validation below 0.115.

**More training doesn't explain it.** Most of the baseline's runs stopped early (24–32 passes) because they had plateaued, while the teacher runs used up to 40. At the same pass, T1 is ahead in all 5 seeds every time (best validation error so far, mean of 5 seeds):

| By pass | 10 | 15 | 20 | 24 | 32 | 40 |
|---|---:|---:|---:|---:|---:|---:|
| GRU cell | 0.117 | 0.101 | 0.099 | 0.0975 | 0.0971 | 0.0969 |
| T1 | 0.100 | 0.096 | 0.0946 | 0.0940 | 0.0930 | 0.0925 |
| T2 | 0.124 | 0.101 | 0.0962 | 0.0946 | 0.0937 | 0.0930 |

**By flight time (test, RMS):**

| | 0–1 s | 1–5 s | 5–10 s | 10–20 s | 20–35 s |
|---|---:|---:|---:|---:|---:|
| GRU cell minus calculator | 0.034 | 0.042 | 0.046 | 0.047 | 0.043 |
| T1 minus calculator | 0.024 | 0.037 | 0.038 | 0.036 | 0.032 |
| GRU cell bias | −0.025 | −0.009 | −0.006 | −0.005 | −0.005 |
| T1 bias | −0.007 | −0.004 | −0.004 | −0.005 | −0.002 |

**By style (test, GRU cell / T1 / calculator):**
- `probe_hold` 0.081 / 0.075 / 0.068
- `wavy` 0.127 / 0.118 / 0.112
- `shaky` 0.118 / 0.111 / 0.097
- `random_hold` 0.098 / 0.094 / 0.072
- `switching` 0.086 / 0.081 / 0.068
- `endurance` 0.064 / 0.060 / 0.055
- `replay_good` 0.078 / 0.075 / 0.069
- `hold_speed` 0.061 / 0.058 / 0.054
- `blind_ppo` 0.059 / 0.057 / 0.049
- `cruise` 0.135 / 0.133 / 0.122
- `steady_below` 0.147 / 0.144 / 0.144

**T2's uncertainty (test):**
- **Calibrated:** squared error / its variance averages 0.91–1.02 (1 is perfect), median about 0.5.
- **It follows the calculator's std:** correlation 0.91–0.93.
- **It is a little wider** (mean std 0.081, against the calculator's 0.068), as it should be, since its errors are larger than the calculator's.
- **It predicts its own error** (correlation of std with the error's size 0.64–0.66) a little better than the calculator's std does (0.59–0.61).

**What it shows:**
- **The cleaner target helps,** consistently but modestly:
  - 0.0972 → 0.0933, and the seed ranges don't overlap (T1's worst seed beats the baseline's best).
  - A third of the remaining gap to the floor is closed (0.0119 → 0.0080).
  - The distance to the calculator falls by about 22% (0.045 → 0.035).
- **T1 finds the battery sooner** (the click at pass 7–8 against 10–12), keeps improving longer, and is better in every time window and every style. The largest gains are on the probing and changing flights.
- **The early-flight bias almost disappears** (−0.025 → −0.007 in the first second).
- **T2's mean is as good as T1's,** so learning the spread costs nothing in accuracy, and its uncertainty is calibrated. It clicks later, but has caught up by pass 20.
- **What is left:** the gap to the calculator stays about 0.03–0.037 through the flight. The calculator keeps sharpening late in the flight (0.072 after 20 s); T1 less so (0.081). The largest gaps are on `random_hold`, `switching` and `shaky`.
- **Caveats:**
  - These are pilot flights, open loop. The test that counts is plain PPO with the frozen encoder.
  - The closed-loop overestimation after probing has to be measured again on PPO's own flights.
  - The targets bring the physics into training.
- **Next:** freeze T1 under plain PPO (below). The encoder is chosen by validation: T1 seed 3, 0.0918. The validation errors were:
  - T1: s0 0.0925, s1 0.0928, s2 0.0925, s3 0.0918, s4 0.0927
  - T2: 0.0923–0.0941
  - GRU cell: 0.0945–0.0990
- **Later option:** T2 makes a second PPO test possible, with the policy also reading the encoder's uncertainty (the context wrapper would need to pass on the variance).

**Plain PPO with the frozen T1 encoder (launched 2026-10-18 04:55 UTC, all 16 runs done by 05:07 UTC).**
- **Design:** step 4's frozen-GRU design with T1 seed 3 (chosen by validation, 0.0918) as the encoder: `4a_encoder` (batch 256) and `4b_encoder` (batch 1024), PPO seeds 0–7, 16 runs at once, 1 thread each, about 12 minutes per run.
- **Only the encoder differs:** the resolved configs differ from the working GRU's runs only in `context.encoder_model`, `context.encoder_tag` and the W&B names, e.g. `S4a_encoder_bs256_E1_gru_teacher_point_main_s3_gamma0.99_rs0.1_s0`.
- **Seeds:** the same as the working GRU's runs: environment seeds 0–707 for training and 1000–1002 for validation. The encoder trained only on pilot flights with seeds from 100,000 up. The final-test seeds 2000–2002 are used nowhere.
- **Report:** `runs_step0/frozen_teacher_t1/step0_report_encoder_E1_gru_teacher_point_main_s3.json`. Closed-loop check: `runs_step0/closed_loop_frozen_t1.json`.

```bash
nohup python run_step0.py --conditions 4a_encoder 4b_encoder --encoder-models runs_encoder/step4_teacher/E1_gru_teacher_point_main_s3/encoder.pt \
  --seeds 0 1 2 3 4 5 6 7 --jobs 16 --threads 1 --output runs_step0/frozen_teacher_t1 > run_frozen_teacher_t1.log 2>&1 &
python step4_closed_loop.py --reports runs_step0/frozen_teacher_t1/step0_report_encoder_E1_gru_teacher_point_main_s3.json \
  --output runs_step0/closed_loop_frozen_t1.json --jobs 16
```

| Battery estimate | Batch | Late score ± std (median) | Seeds 0–7 | Final | Completed | Battery empty | Crashed |
|---|---|---:|---|---:|---:|---:|---:|
| Working GRU s1 | 256 | 209 ± 39 (210) | 168 203 255 169 216 231 167 261 | 204 | 88% | 12% | 0% |
| **T1 s3** | **256** | **291 ± 49 (312)** | 270 320 331 216 331 305 333 224 | 294 | 81% | 16% | 3% |
| Working GRU s1 | 1024 | 164 ± 196 (218) | 330 177 123 280 262 249 −294 187 | 157 | 73% | 22% | 5% |
| Working GRU s0 | 1024 | 179 ± 68 (192) | 196 145 69 277 109 221 227 187 | 204 | 77% | 16% | 7% |
| Working GRU s4 | 1024 | 120 ± 111 (101) | 57 −7 244 26 296 165 146 32 | 125 | 43% | 52% | 5% |
| **T1 s3** | **1024** | **262 ± 45 (270)** | 255 296 216 284 239 189 316 303 | 286 | 73% | 24% | 3% |

**For comparison** (late score, batch 256 / 1024): memoryless 23 / 77, true battery 246 / 258, the ideal estimate 283 / 323.

**The closed-loop check** (T1 on its own PPO policies' validation flights, next to the ideal estimator on the same flights):

| Encoder, batch | Encoder error | Ideal estimator | Encoder minus ideal | Bias | Bias at b0 = 0.2 / 0.7 |
|---|---:|---:|---:|---:|---|
| Working GRU s1, 256 | 0.126 | 0.110 | 0.053 | +0.021 | +0.20 / −0.13 |
| Working GRU s1, 1024 | 0.119 | 0.102 | 0.051 | +0.016 | +0.18 / −0.11 |
| Working GRU s0, 1024 | 0.106 | 0.092 | 0.043 | −0.011 | +0.15 / −0.12 |
| Working GRU s4, 1024 | 0.054 | 0.044 | 0.028 | +0.002 | +0.07 / −0.04 |
| **T1 s3, 256** | **0.051** | 0.046 | **0.016** | **+0.000** | +0.07 / −0.04 |
| **T1 s3, 1024** | **0.050** | 0.047 | **0.012** | −0.003 | +0.07 / −0.04 |

- **By flight time (T1, batch 256, encoder / ideal):** 0–1 s 0.167 / 0.167; 1–5 s 0.095 / 0.089; 5–10 s 0.034 / 0.028; 10–20 s 0.027 / 0.019; 20–35 s 0.024 / 0.016.
- **T1's bias after the first second:** between −0.006 and +0.006 in every window, at both batch sizes.
- **Correlation of a run's encoder error with its late score:** −0.06 over all 16 runs.

**What it shows:**
- **The teacher-trained encoder lifts PPO to the level of the ideal estimate:**
  - At batch 256: 209 → 291, about 3.7 standard errors. That is above the true battery (246) and on a level with the ideal estimate (283).
  - At batch 1024: 262, against 120–179 for the three working-GRU encoders. That is on a level with the true battery (258), still below the ideal estimate (323).
- **The batch-1024 instability is gone:** no seed collapses (the worst is 189, against −294 with GRU s1).
- **On PPO's own flights, T1 almost matches the ideal estimator:** 0.012–0.016 from it, against 0.028–0.053 for the working GRU encoders. That is a gap about three times smaller than on the pilot flights (0.035).
- **The overestimation that caused the collapse is gone.** The bias stays near zero after the first second, where the working GRU s1 overestimated by +0.02 to +0.03.
- **The policies probe and reveal the battery:** after 5 s the ideal estimator's error on their flights is only 0.013–0.028, and T1 follows it closely.
- **A small offline gain, a large gain in PPO.** Offline, T1 was only 0.004 better (0.0972 → 0.0933). But the noise-free target taught it to follow the ideal estimate closely right after the battery shows, which is exactly when PPO acts on it. The same GRU trained on the noisy true-battery labels lagged and overestimated there.
- **Like the ideal estimate in step 2, T1 beats the true battery at batch 256.** A good estimate of what can be known seems to be a better policy input than the exact battery.
- **Completion is a little lower than with GRU s1** (81% against 88% at batch 256) despite the much higher score. The T1 policies seem to fly more aggressively and earn more per flight.
- **Caveats:**
  - Only one T1 encoder (seed 3) was tested. With the working GRU the encoder seed mattered at batch 1024 (120–179).
  - The teacher brings the physics into training (the dynamics, the prior and the speed noise), so this encoder is not purely learned from experience.
- **Next:**
  - The encoder-seed check: T1 seeds 0 and 2 (the next best by validation, 0.0925) under batch 1024, PPO seeds 0–7 (16 runs, about 12 minutes).
  - Optionally, T2 with the policy also reading the encoder's uncertainty.

### The final test (2026-10-18)

The reserved final-test flights (seeds 2000–2002), used once, after all training and selection, on the frozen final checkpoints of the plain-PPO conditions:
- memoryless PPO (`0a_blind` / `0b_blind`)
- the true battery (`0a` / `0b`)
- the ideal estimate (`2a_ideal` / `2b_ideal`)
- the frozen GRU trained on true-battery labels (`4a_encoder` / `4b_encoder`: s1 at both batch sizes, s0 and s4 at batch 1024)
- the frozen GRU trained on the ideal estimate (T1 s3, both batch sizes)

That is 96 runs, 8 PPO seeds each.

**How:** `evaluate_final_test.py --reports` (new; its A–F `--grid` mode is unchanged).
- **Validation first:** every run is scored again on the validation flights, and must reproduce its report's final validation exactly. All 96 did, so these are the models that were reported.
- **Then the final test:** each run's own validation grid with only the seeds replaced by the reserved ones. For a Contextual PPO config that is its `final_test` grid. The grid is refused if those seeds overlap the run's validation or training seeds.
- **Once only:** each run keeps its result in its directory (`final_test.json`), which is reused, never overwritten.
- **Checks:** `checks.final_test_checks`.
- **A bug fixed on the way:** the first launch didn't limit numpy's BLAS threads in the workers, and the ideal estimator oversubscribed the server (load about 450). The script now gives every worker one thread, as `run_step0.py` does. That changed only the speed; the 17 runs finished before the fix had reproduced their validation.
- **Summary:** `runs_step0/final_test_all.json`.

```bash
python evaluate_final_test.py --reports runs_step0/20261014-020124/report_blind_and_A.json \
  runs_step0/20261013-042940/step0_report_rs0.1_seeds0-7.json runs_step0/20261014-044145/step0_report_ideal.json \
  runs_step0/frozen_gru_a/step0_report_encoder_E1_gru_point_main_s1.json runs_step0/frozen_gru_b/step0_report_encoder_E1_gru_point_main_s0.json \
  runs_step0/frozen_gru_b/step0_report_encoder_E1_gru_point_main_s4.json \
  runs_step0/frozen_teacher_t1/step0_report_encoder_E1_gru_teacher_point_main_s3.json \
  --conditions 0a_blind 0b_blind 0a 0b 2a_ideal 2b_ideal 4a_encoder 4b_encoder --output runs_step0/final_test_all.json --jobs 16
```

**Final test** (18 flights per run, mean ± std across 8 PPO seeds; the last column is the same checkpoints on the validation flights):

| Battery input | Batch | Final test (median) | Worst seed | Completed | Battery empty | Crashed | Validation |
|---|---|---:|---:|---:|---:|---:|---:|
| None (memoryless) | 256 | 49 ± 45 (28) | 5 | 25% | 75% | 0% | 28 |
| True battery | 256 | 283 ± 66 (278) | 209 | 90% | 4% | 6% | 251 |
| Ideal estimate | 256 | 316 ± 39 (320) | 262 | 97% | 3% | 0% | 265 |
| Frozen GRU, true-battery labels (s1) | 256 | 238 ± 40 (246) | 184 | 92% | 8% | 0% | 204 |
| **Frozen GRU, ideal estimate as teacher (T1 s3)** | 256 | **330 ± 46 (338)** | 250 | 86% | 10% | 3% | 294 |
| None (memoryless) | 1024 | 119 ± 65 (129) | −31 | 17% | 83% | 0% | 79 |
| True battery | 1024 | 280 ± 36 (288) | 225 | 92% | 1% | 7% | 236 |
| Ideal estimate | 1024 | 348 ± 12 (351) | 331 | 97% | 1% | 1% | 308 |
| Frozen GRU, true-battery labels (s1) | 1024 | 184 ± 206 (217) | −299 | 76% | 17% | 7% | 157 |
| Frozen GRU, true-battery labels (s0 / s4) | 1024 | 240 ± 86 / 152 ± 116 | 98 / 19 | 81% / 45% | 15% / 51% | 4% / 4% | 204 / 125 |
| **Frozen GRU, ideal estimate as teacher (T1 s3)** | 1024 | **319 ± 56 (326)** | 220 | 76% | 18% | 6% | 286 |

**Differences** (Welch's t-test and Mann–Whitney U, 8 seeds against 8):

| Comparison | Batch 256 | Batch 1024 |
|---|---|---|
| T1 − GRU on labels | +92 (p 0.0008 / 0.003), against s1 | +79 (p 0.05 / 0.07), against s0; +135 against s1 (p 0.11, its collapsed seed) |
| T1 − true battery | +47 (p 0.12 / 0.08) | +38 (p 0.13 / 0.13) |
| T1 − ideal estimate | +14 (p 0.53 / 0.44) | −29 (p 0.19 / 0.38) |
| GRU on labels − true battery | −45 (p 0.12), s1 | −40 (p 0.25), s0 |
| Ideal estimate − true battery | +33 (p 0.25) | +68 (p 0.001) |
| Frozen encoder − memoryless | GRU on labels +189, T1 +281 (both p < 0.001) | T1 +200 (p < 0.001); GRU on labels s0 +122 (p 0.007), s1 +65 (p 0.42), s4 +33 (p 0.50) |
| True battery − memoryless | +235 (p < 0.001) | +162 (p < 0.001) |

**Share of the gap from memoryless to the ideal estimate, final test:**
- batch 256: GRU on labels 71%, T1 105%, true battery 88%
- batch 1024: GRU on labels 53% (s0), T1 87%, true battery 71%

**By starting battery** (final test, mean return, completed in brackets):

| | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| Memoryless, 256 | −186 (0%) | −118 (0%) | −34 (4%) | 55 (12%) | 185 (38%) | 390 (96%) |
| True battery, 256 | −100 (42%) | 199 (100%) | 266 (100%) | 348 (100%) | 442 (100%) | 544 (100%) |
| Ideal estimate, 256 | 83 (79%) | 202 (100%) | 269 (100%) | 349 (100%) | 446 (100%) | 548 (100%) |
| GRU on labels s1, 256 | 55 (54%) | 231 (100%) | 243 (100%) | 266 (100%) | 303 (100%) | 330 (100%) |
| T1, 256 | −4 (38%) | 193 (79%) | 322 (100%) | 400 (100%) | 488 (100%) | 582 (100%) |
| Ideal estimate, 1024 | 149 (88%) | 213 (96%) | 288 (100%) | 370 (100%) | 476 (100%) | 590 (100%) |
| T1, 1024 | −125 (4%) | 160 (67%) | 303 (88%) | 425 (100%) | 525 (100%) | 625 (100%) |

**What it shows:**
- **The ranking and the gaps hold on unseen flights.** Every condition scores 20–51 points higher than on validation (the starting heights and speeds are drawn per seed, and these flights are a little easier), but about equally, so the differences carry over. T1 minus GRU at batch 256 is +90 on validation and +92 on the final test.
- **T1 is on a level with the ideal estimate:**
  - At batch 256: 330 against 316. That is 105% of the gap from memoryless to the ideal estimate.
  - At batch 1024: 319 against 348, 87%. The difference is not significant at 8 seeds.
  - T1 scores above the true battery at both batch sizes (+47 / +38), but not significantly.
- **T1 clearly beats the GRU trained on labels at batch 256** (+92, p < 0.001). At batch 1024 the gain is +79 to +135, with p 0.05–0.11.
- **The GRU trained on labels collapses again on new flights:** s1's collapsed seed scores −299. T1's worst seed is 220.
- **Different pilots, by starting battery:**
  - **The ideal-estimate and true-battery policies agree from 0.3 up.** The ideal-estimate policies save most of the 0.2 flights (79–88% completed, against 42–50% with the true battery), which is where their lead over the true battery comes from.
  - **T1's policies earn the most at high batteries** (582 / 625 at 0.7) but lose most of the lowest-battery flights.
  - **The GRU on labels' policies fly one careful style:** nearly all flights completed, little extra on high batteries (330 at 0.7).
  - **T1's encoder is the better one,** but its policies haven't learned the low-battery end as well as the ideal estimate's.
- **What the paper can claim from the final test:**
  - The environment is valid: knowing the battery is worth +162 to +235.
  - At batch 1024, two of the three GRU-on-labels encoders (s1, s4) are not significantly better than memoryless.
  - A memory agent whose encoder is trained offline (on true-battery labels, or better, on the ideal estimate) and then frozen gets 53–71% of the way to the ideal estimate with labels, and 87–105% with the teacher.

### Wrap-up experiments (2026-10-18)

**1. The final test of the Contextual PPO rows.** The same procedure as above (`evaluate_final_test.py --reports`, validation reproduced first, then the reserved seeds) for:
- Contextual PPO with the true battery (`0c`, 8 seeds)
- without context (`0c_blind`, 8 seeds)
- with its learned xLSTM encoder (the A rerun, 3 seeds)
- with a jointly trained GRU (J1, 5 seeds)

All 24 reproduced their validation.

| Contextual PPO | Final test (median) | Worst seed | Completed | Battery empty | Crashed | Validation late score |
|---|---:|---:|---:|---:|---:|---:|
| With the true battery (`0c`) | 286 ± 70 (291) | 187 | 89% | 3% | 8% | 282 |
| No context (`0c_blind`) | 155 ± 19 (154) | 120 | 27% | 73% | 0% | 115 |
| Its learned xLSTM encoder (A rerun, 3 seeds) | 42 ± 66 (78) | −33 | 17% | 83% | 0% | 22 |
| Joint training with a GRU (J1, 5 seeds) | 150 ± 43 (157) | 84 | 30% | 70% | 0% | 114 |

- **Contextual PPO with the true battery matches plain PPO with the true battery** (283 / 280).
- **Experiment A stays far below the memoryless control,** and J1 stays at the memoryless level, as on validation.

**2. T1's encoder seed.** T1 encoder seeds 0 and 2 (next best by validation, both 0.0925) frozen under plain PPO at batch 1024, PPO seeds 0–7, the design used for the GRU on labels.
- **Runs:** 16, about 11 minutes each, 16 at once.
- **Reports:** `runs_step0/frozen_teacher_t1_seeds/step0_report_encoder_E1_gru_teacher_point_main_s{0,2}.json`.
- **Final test:** the same procedure as above. All 136 final-tested runs are summarized in `runs_step0/final_test_complete_v2.json`, which supersedes `final_test_all.json`.

| Batch 1024, 8 PPO seeds | Validation late score | Final test (median) | Worst seed (final test) | Completed (final test) |
|---|---:|---:|---:|---:|
| T1 encoder s0 | 279 | 313 ± 48 (323) | 234 | 85% |
| T1 encoder s2 | 272 | 331 ± 23 (337) | 295 | 88% |
| T1 encoder s3 | 262 | 319 ± 56 (326) | 220 | 76% |
| GRU on labels s0 / s1 / s4 | 179 / 164 / 120 | 240 / 184 / 152 | 98 / −299 / 19 | 81% / 76% / 45% |
| True battery / ideal estimate | 258 / 323 | 280 / 348 | 225 / 331 | 92% / 97% |

- **The encoder seed hardly matters for T1:** 262–279 on validation and 313–331 on the final test across its three encoders, against 120–179 and 152–240 for the GRU on labels.
- **The headline result doesn't rest on one encoder.** All three T1 encoders are above the true battery (280) on the final test, below the ideal estimate (348), and none collapses (worst seeds 220–295).

**3. The swap test.** `step2_swap_battery.py`, now also for frozen-encoder runs, flies every frozen-encoder policy over the validation flights twice (never the final test): once with its own encoder, and once with the true battery in its place (`context.source=true_battery`, the same width). The own-encoder flights reproduce the reports' final validation (checked on one run: 301.9). That is 64 runs, in 22 seconds; the results are in `runs_step0/swap_true_battery_frozen.json`.

```bash
python step2_swap_battery.py --runs runs_step0/frozen_gru_a/runs.json runs_step0/frozen_gru_b/runs.json \
  runs_step0/frozen_teacher_t1/runs.json runs_step0/frozen_teacher_t1_seeds/runs.json \
  --output runs_step0/swap_true_battery_frozen.json --jobs 30
```

| Policies (8 PPO seeds each) | Own encoder | True battery instead | Change | Seeds better / worse | Wilcoxon p |
|---|---:|---:|---:|---|---:|
| T1 s3, batch 256 | 294 | 322 | +28 | 7 / 1 | 0.02 |
| T1 s3, batch 1024 | 286 | 324 | +38 | 7 / 1 | 0.02 |
| T1 s0, batch 1024 | 282 | 282 | 0 | 5 / 3 | 0.95 |
| T1 s2, batch 1024 | 288 | 293 | +5 | 4 / 4 | 0.95 |
| GRU on labels s1, batch 256 | 204 | 170 | −34 | 3 / 5 | 0.25 |
| GRU on labels s1, batch 1024 | 157 | 114 | −44 | 0 / 7 | 0.02 |
| GRU on labels s0, batch 1024 | 204 | 187 | −17 | 3 / 5 | 0.38 |
| GRU on labels s4, batch 1024 | 125 | 142 | +17 | 7 / 0 | 0.02 |
| **All T1 runs (32)** | | | **+18** | 23 / 9 | **0.008** |
| **All GRU-on-labels runs (32)** | | | **−20** | 13 / 17 | 0.08 |
| Step 2: ideal-estimate runs, batch 256 / 1024 | 265 / 308 | 280 / 310 | +15 / +2 | | |

**What it shows:**
- **The T1 policies are simply better pilots,** as step 2 found for the ideal-estimate policies.
  - Given the true battery they score as well or better (+18 on average, 23 of 32 runs better).
  - So their lead over plain PPO trained with the true battery doesn't come from extra information in the estimate. It comes from how they learned to fly.
  - The small gain with the true battery is what T1's remaining estimation error costs them.
- **The GRU-on-labels policies learned to read a distorted signal.** Given the true battery, most score worse (−20 on average; s1's policies lose in 7 of 8 seeds at batch 1024).
  - Their encoder's estimate is biased in a way that depends on the starting battery (the closed-loop check: +0.18 to +0.20 at 0.2, −0.11 to −0.13 at 0.7).
  - The policies adapted to that bias, so the truth is an unfamiliar input to them.
- **For the T1 policies, their encoder's estimate and the true battery are interchangeable inputs.** That is the practical sense in which T1 "learned the battery".

## Next steps

**Done:** the CPPO oracle bypass and step 0; step 1 (noisy oracle, fly-with-noise, late switch, early and in-flight noise, the A encoder's error over a flight, the ideal estimator); the memoryless controls; the context wrapper and plain PPO with the ideal estimate (step 2: the encoder is the bottleneck); the pilot generator and dataset v2 (step 3).

1. **Train the encoder offline on dataset v2 (step 4).** Running since 2026-10-15 (`train_encoder.py`, E1 and E3, 3 seeds each).
   - The xLSTM on the `main` mix, with an LSTM for comparison. Also train on `narrow` (the A-like control) and on `main_plus_aware` (the leak test).
   - Split by whole flights for validation. Keep some styles out entirely for the cross-pilot matrix.
   - Train until the validation loss plateaus. Reuse the enc1 training setup where it fits; its input is `pilots.load_flights` (observations, previous actions, true battery, style).
   - Save it so `FrozenEncoderContext` can load it. Done: each run saves `encoder_model.zip`, a Contextual_PPO model holding the best encoder.
2. **Measure it before plugging it in** (section 4). The floor is measured (0.084 on validation, step 4 results above); the encoders close only 22–26% of the gap so far. The diagnostic probes (`step4_probes.py`) show that a GRU learns the task, to 0.100–0.101 on the test flights (73–75% of the gap), after a long plateau; the xLSTM learns it too given the same time; the 2 × 2 test (2026-10-17) leaves the plain GRU best at 0.097 on the test flights (80% of the gap). Still to do:
   - Error at 1, 5 and 20 s and over the whole flight, for each style.
   - Compare against the clock-only estimate and the ideal estimator (`step1_ideal_estimator.py`) on the same flights.
   - Run the leak test: test the `main_plus_aware` encoder on battery-blind pilots.
3. **Freeze it under plain PPO**, with `context.source=encoder` and `context.encoder_model=...`. Use 8 seeds at step 0's settings (gamma 0.99, reward scale 0.1).
   - Compare against the ideal estimate (283 / 323), the true battery (246 / 258) and memoryless PPO (23 / 77).
   - Then the closed-loop check: the encoder's error on the PPO policy's own flights.
   - Done 2026-10-17 with the best GRU encoders (step 4 above): 209 ± 36 at batch 256 (88% of flights completed), against 23 without memory, 246 with the true battery and 283 with the ideal estimate. Batch 1024 is unstable with the GRU (one seed collapsed), and the encoder seed matters. The closed-loop check (done): on PPO's own flights the GRU overestimates the battery after probing (+0.03 to +0.06), which caused the collapse. Next: encoder training on these PPO policies' flights, and the ideal estimator as a teacher.
   - Done 2026-10-18 with the teacher-trained GRU (T1 s3, above): 291 ± 49 at batch 256 and 262 ± 45 at batch 1024, no collapse; on its policies' own flights it stays 0.012–0.016 from the ideal estimator, with no overestimation.
4. **Joint training (J0–J5, done 2026-10-17):** Contextual PPO with a GRU encoder trained together with the policy. The best rung, J1, scores 114, about the memoryless level, against 209 for the frozen GRU. Only more encoder training helped. Next options: start Contextual PPO from the pretrained GRU, or mix pilot flights into its encoder data. If the shortfall turns out to come from early-flight error, the easy environment is plan B.
5. **Optional:**
   - Train on the ideal estimator as a teacher (section 3). Done 2026-10-18 (T1 and T2, 5 seeds each, above): test 0.0933, against 0.0972 for the GRU cell and the 0.0853 floor; T2 matches T1 and its uncertainty is calibrated. T1 frozen under plain PPO (done 2026-10-18): 291 at batch 256 and 262 at batch 1024, against 209 / 164 for the working GRU, the true battery's 246 / 258 and the ideal estimate's 283 / 323; on its policies' own flights it is 0.012–0.016 from the ideal estimator, with no overestimation. Next: the encoder-seed check (T1 seeds 0 and 2 under batch 1024).
   - The final test (seeds 2000–2002, done 2026-10-18, above), batch 256 / 1024:
     - memoryless 49 / 119
     - true battery 283 / 280
     - ideal estimate 316 / 348
     - GRU on labels 238 / 184 (s0 240)
     - T1 330 / 319

     The ranking and gaps of the validation flights hold.
   - Freeze A's own encoder under plain PPO (step 2's next steps).
   - Add earlier PPO checkpoints (0.5M / 1M / 1.5M steps) or more seeds as pilots, if the encoder's error is limited by data.
   - Start from the pretrained encoder and keep training it online, as CPPO does (section 5).
