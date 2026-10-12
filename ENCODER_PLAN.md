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
- Write the decision rules from section 1 down before running anything.
- Optional extra condition: start from the pretrained encoder, then keep training it online the way CPPO does now. That shows whether online training helps or hurts once the encoder starts out good.

That comes to 8 conditions × 3 seeds = 24 PPO runs: oracle, three noisy oracles, clock-only, and three frozen encoders.

## Next steps

The enc1 setup on the server already covers the offline training step. The new pieces are:

1. **The context wrapper.** Write this first, because the oracle run on its own already answers the most important question.
2. **The pilot generator.**
3. **The oracle and clock-only controls.**
