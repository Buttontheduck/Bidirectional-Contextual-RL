# Planned experiment: pretrained frozen encoder vs. policy

**Status:** planned (2026-10-12), not started.

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

Observed-mode plain PPO (p04) already provides encouraging evidence: 212 final return and about 360 at its peak. Repeat it alongside the new conditions under the same scoring rule. Those historical scores are reference points, not guaranteed reproduction targets.

**CPPO and plain PPO train the networks differently.** Both collect 4,096 transitions per rollout (8 environments × 512 steps), and both use each transition 10 times. They split those transitions into minibatches differently:

| | Minibatch | Optimizer steps per rollout | Optimizer steps over 2M transitions |
|---|---|---:|---:|
| Plain PPO (p01–p04) | 256 transitions, shuffled from all 8 environments | 160 | ≈ 78k |
| CPPO (A–F) | 1,024 transitions: two complete 512-step stretches, each from one environment | 40 | ≈ 19.5k |

The learning rate is the same, so CPPO makes a quarter as many weight updates, but not a quarter as many passes over the data. Its minibatches contain two environment streams, potentially spanning episode boundaries; plain PPO shuffles individual transitions across all eight streams.

**Primary runs: three conditions × three seeds = nine fresh runs.** All use 8 environments × 512 steps, 10 epochs, 2M requested environment transitions (rounded to complete rollouts), and training seeds 0, 1, 2.

Seeds, the same for all three conditions:
- Training env `i` starts from reset seed `seed + i`, so 0–9. SB3 re-seeds the envs when learning starts. The same seed in 0a, 0b and 0c starts from the same first flights.
- Validation uses seeds 1000–1002. The final-test seeds 2000–2002 stay unused.
- Both scripts reject training seeds that reach the held-out seeds, and `checks.grid_checks` verifies the real start seeds.
- Neighbouring training seeds share their first flight in 7 of the 8 envs, then diverge. This is negligible, and it is kept for comparability with A–F and p04.

| Run | Learner and battery input | Minibatch | Optimizer steps per rollout | Purpose |
|---|---|---|---:|---|
| 0a | Plain PPO, true battery (`observed` mode) | 256 shuffled transitions | 160 | Reproduce the original baseline |
| 0b | Plain PPO, true battery (`observed` mode) | 1,024 shuffled transitions | 40 | Match CPPO's batch size and optimizer-step count |
| 0c | CPPO, true battery replacing the encoder | 1,024 transitions (2 streams × 512 steps) | 40 | Test CPPO without estimation error or encoder drift |

- Every actor and critic receives the same five values: the four hidden-mode observations plus the true current battery, in the same order and scale. Speed remains noisy. There is no predicted mean, variance, or other encoder feature in 0c.
- Keep the environment, reward, actor/critic architecture, learning rate, discount, GAE settings, and other PPO settings identical. The intended differences are the learner and the minibatch settings shown above.
- Compare 0b with 0c to investigate differences beyond batch size and optimizer-step count, including shuffled transitions versus complete streams. Compare 0a with 0b to investigate the batch-size/update-count setting within plain PPO; these two effects are not separated by this comparison.
- The earlier 32-environment × 128-step condition is deferred. It changes trajectory collection and GAE bootstrapping as well as minibatches, so success would not establish that CPPO "only needed more updates."

**The 0c oracle path (implemented 2026-10-12).** Setting only `env.observation_mode=observed` would leave the encoder active and append its predictions to the visible battery, which is a different experiment. 0c instead uses the Hydra option `cppo.context_source=oracle`, set by `config/step0/cppo_oracle.yaml`:
- `step0_oracle.py` replaces the xLSTM with `TrueBatteryEncoder`, which has no parameters and no memory and outputs `z_t = b_t` from the observation. The actor and critic read the observation only, because observed mode already ends with `b_t`. This holds in rollout collection, PPO minibatch evaluation, value prediction and evaluation, because all of them go through the same policy methods.
- `encoder_objective=rl` with no encoder parameters means there is no episode buffer, no supervised encoder step, and nothing for the routed PPO losses to update. The speed head stays off.
- CPPO's collector, PPO losses and whole-stream minibatching are unchanged. The default `cppo.context_source=encoder` leaves A–F unchanged; a missing field in an older saved config also means `encoder` during validation and model construction.
- `validate_config` requires `observed` mode for the oracle. It rejects `full` (exact speed), hidden-battery modes, a supervised objective, the speed head, the Gaussian head or `context_dim` ≠ 1.
- The inherited W&B prefix is retained: seed 0 is named `cppo_S0c_oracle_s0`, in group `cppo_S0c_oracle`.

`python -m unittest checks.step0_checks -v` covers the pre-training checks:
- The config differs from the base only in the battery input.
- The actor and critic inputs equal plain PPO's observed-mode features on the same observations. With the oracle's weights loaded, plain PPO gives the same values and actions, and the previous action has no effect.
- In rollouts with resets and terminal transitions, the observation's last column equals the true `b` from the env infos, for both `o_t` and its successor.
- The encoder group is empty, so only the actor and critic change.
- A saved model reloads with the oracle policy.
- The oracle rejects exact-speed `full` mode. `checks.grid_checks` also verifies that older A–F configs without `context_source` still validate and build an encoder policy.

A full-size smoke run showed `encoder: 0` parameters and actor/critic parameter counts identical to plain PPO's (67,844 / 67,585). A training round took about 0.2 s, so a 2M-step run should take minutes, not the 5.5–7 h of A–F. If a 0c run takes hours, the encoder is running and something is wrong.

```bash
# All nine runs from this repo: the three conditions side by side, logs in runs_step0/
nohup bash run_step0.sh > run_step0.log 2>&1 &

# Or one condition at a time
python main_battery_plane_contextual_ppo.py -m +step0=cppo_oracle training.seed=0,1,2          # 0c
python ppo/main_battery_plane_ppo.py -m +experiment=S0a_observed_bs256 training.seed=0,1,2      # 0a
python ppo/main_battery_plane_ppo.py -m +experiment=S0b_observed_bs1024 training.seed=0,1,2     # 0b
```

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

### 1. Bracket the encoder with controls

Run the same PPO in every condition and change only the context `z_t` the policy receives:

| Condition | `z_t` | What it answers |
|---|---|---|
| **Oracle** | true `b_t` | Can PPO use a perfect estimate? |
| **Noisy oracle** | `b_t` + noise, σ = 0.03 / 0.06 / 0.12 | How good does the estimate need to be? |
| **Frozen encoder** (one per training mix) | pretrained, frozen | The main experiment |
| **Clock-only** | average battery at step `t` | Return from an "encoder" that learned only time |
| **No context** | none | Already measured: PPO hidden, 52.7 |

The σ values bracket what we measured: about 0.06 for the current encoders and 0.12 for the clock-only estimate.

How to read the results:
- **Oracle fails:** the problem is the policy or PPO, and better encoders won't help.
- **Oracle and noisy oracle (0.06) work, frozen encoder fails:** the encoder is the problem. Its errors hurt more than random noise of the same size, probably because they're correlated over time and largest early in the flight.
- **Oracle and frozen encoder both work:** the encoder is fine, and the failure comes from training it jointly with the policy. Either `z_t` shifting under the policy or CPPO's own update is to blame.

Random noise is gentler than real encoder error, so a passing noisy oracle is necessary but not sufficient.

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
2. **The step 0 runs.** Run the three conditions, then use the results to choose the next diagnostic.
3. **The context wrapper.** If Step 0 supports proceeding, implement the frozen-encoder and control inputs for plain PPO.
4. **The pilot generator.**
5. **The remaining controls.** Verify whether Step 0's plain-PPO oracle can be reused; add the clock-only and noisy-oracle conditions when the follow-up scope is fixed.
