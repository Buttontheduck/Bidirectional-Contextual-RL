# CityLearn: buildings with batteries and solar panels

CityLearn simulates a small community of buildings connected to the electricity grid. Each building has a fixed electricity demand, may have solar panels and a battery, and in some datasets heat pumps, hot-water tanks, electric vehicles or occupants. An agent decides, every hour, how much each battery charges or discharges, so as to buy less electricity, at cheaper times, with lower carbon emissions and smaller peaks.

This document describes the environment as installed here, read from its source code and checked by running it. It uses the dataset `citylearn_challenge_2022_phase_1` as the reference.

**Status (22 October 2026): set aside for now.** We tried CityLearn as a second test of autogenic context, with one building whose battery charge is the hidden context. The experiments ran cleanly, but they did not give a usable answer:
- no learned controller (PPO up to 10 million steps, SAC up to 500,000) beat a simple hand-written rule that never looks at the charge;
- hiding the charge made no measurable difference.

So we could not tell whether knowing the charge is worth little in this task, or whether the learners were not yet good enough to use it. The designs, results and ideas for a later attempt are in [Experiments](#experiments) and [Where this leaves us](#where-this-leaves-us).

- **Version:** CityLearn 3.0.2. It is installed twice:
  - **`/root/venvs/citylearn`:** fully separate, with exactly the versions CityLearn asks for (numpy 1.26.4, gymnasium 0.28.1, scikit-learn 1.2.2, Stable-Baselines3 2.3.2, CPU PyTorch 2.14.1).
  - **`/root/venvs/carlos-citylearn`:** built on the `carlos` conda environment and seeing all its packages (numpy 2.4.6, gymnasium 0.29.1, Stable-Baselines3 2.9.0, xLSTM), plus CityLearn without its version limits, scikit-learn 1.9.1 and simplejson. **Use this one with Contextual PPO** ([below](#using-it-with-contextual-ppo)).
  - `carlos` itself is unchanged.
- **Source:** the `citylearn` package (`citylearn.py`, `building.py`, `energy_model.py`, `reward_function.py`, `cost_function.py`, `wrappers.py`, `base.py`).
- **Create it:**

```python
from citylearn.citylearn import CityLearnEnv     # run with /root/venvs/carlos-citylearn/bin/python
env = CityLearnEnv("citylearn_challenge_2022_phase_1", central_agent=True)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([env.action_space[0].sample()])
```

CityLearn downloads datasets from its GitHub repository on first use, into `~/.cache/citylearn/v3.0.2/`. It has 32 of them, among them:
- the challenges of 2020 to 2023;
- neighbourhoods in California, Texas, Vermont and Quebec;
- demonstrations of three-phase grids, electric vehicles and power outages.

## The reference dataset: citylearn_challenge_2022_phase_1

| | |
|---|---|
| Buildings | 5 |
| Time step | 1 hour |
| Data | one year, 8,760 hours (time steps 0–8759); one episode is that year by default, 8,759 transitions |
| Devices per building | a battery and solar panels; everything else (heating, cooling, hot water) is folded into a fixed "non-shiftable" load from the data |
| Control | one central agent for all buildings (`central_agent=True`), or one agent per building |

| Building | Battery capacity | Battery power | Battery efficiency | Solar panels |
|---|---:|---:|---:|---:|
| Building_1–3, 5 | 6.4 kWh | 5 kW | 0.9 | 4 kW |
| Building_4 | 6.4 kWh | 5 kW | 0.9 | 5 kW |

The batteries are the size of a home battery (a Tesla Powerwall-like 6.4 kWh / 5 kW). Everything except the batteries comes from data files: the buildings' loads, the solar generation per kW of panels, the weather, the electricity price and the grid's carbon intensity. The same year repeats in every episode.

## State space

The full state of the simulation, most of which the agent does not see:

| Part | Seen by the agent? |
|---|---|
| Time step (hour of the year) | partly: month, weekday and hour |
| Weather, price and carbon intensity now and their forecasts | yes |
| Each building's load and solar generation in the coming hour | yes (from the data, exact) |
| Each building's loads further ahead | no |
| Each battery's stored energy | **yes**, as its state of charge |
| Each battery's degraded capacity (wear) | no |
| Each battery's efficiency curve and maximum-power curve | no |
| Each building's net consumption in the hour just finished | yes |

## Observation space

With a central agent, **44 numbers**, not normalized: 24 shared by all buildings, then 4 for each of the 5 buildings.

| Shared entry | Unit | Range in this dataset |
|---|---|---|
| `month` | 1–12 | 1–12 |
| `day_type` | day of the week | 1–7 |
| `hour` | 1–24 | 1–24 |
| `outdoor_dry_bulb_temperature`, and `_predicted_1/2/3` | °C | 5.6–32.2 |
| `outdoor_relative_humidity`, and `_predicted_1/2/3` | % | 10–100 |
| `diffuse_solar_irradiance`, and `_predicted_1/2/3` | W/m² | 0–1,017 |
| `direct_solar_irradiance`, and `_predicted_1/2/3` | W/m² | 0–953 |
| `carbon_intensity` | kg CO₂ per kWh from the grid | 0.07–0.28 |
| `electricity_pricing`, and `_predicted_1/2/3` | $ per kWh | 0.21–0.54 (time-of-use tariff: 0.21, 0.22, 0.40, 0.50, 0.54) |

The `_predicted_1/2/3` entries are forecasts 6, 12 and 24 hours ahead. For temperature, checked here, they match the later actual values to within 0.6 °C.

| Entry per building | Unit | Meaning | Range (Building_1) |
|---|---|---|---|
| `non_shiftable_load` | kWh | the building's demand in the **coming** hour | 0.06–7.99 |
| `solar_generation` | kWh | the panels' output in the **coming** hour | 0–3.9 |
| `electrical_storage_soc` | 0–1 | the battery's stored energy as a share of its **nominal** capacity | 0–1 |
| `net_electricity_consumption` | kWh | what the building took from the grid (negative: gave to it) in the hour **just finished** | −8.6 to 13.0 |

**Net consumption:** net(t) = load(t) − solar(t) + battery charging(t), where battery charging is negative while discharging. After each step, the observation's load and solar are for the hour about to start, but its net consumption is for the hour just finished. So the battery's last energy flow follows from two consecutive observations: battery(t−1) = net in the new observation − load(t−1) + solar(t−1), with load and solar from the previous observation. Checked to machine precision.

**Normalized observation.** CityLearn's `NormalizedObservationWrapper` gives **47 numbers in [0, 1]**:
- month, weekday and hour each become a cosine and a sine, so 3 entries become 6;
- every other entry is scaled by its minimum and maximum over the dataset.

`StableBaselines3Wrapper` then turns the list-of-lists interface into plain arrays.

## Action space

With a central agent, **5 numbers in [−1, 1]**, one per battery:
- a > 0 charges, a < 0 discharges;
- the requested energy is a × rated power × 1 hour, so a = 1 asks for 5 kWh in the hour.

The battery then delivers what it can:
1. **Power limit,** set by the state of charge (the maximum-power curve, below). Near full, a battery charges much more slowly.
2. **Room limit:** charging stops at the battery's degraded capacity, discharging at empty (depth of discharge 1, so the whole battery can be used).
3. **Losses:** the round-trip efficiency depends on the power (the efficiency curve, below). CityLearn applies √efficiency on the way in and again on the way out.

The action the agent sends is therefore an upper bound. The executed energy is visible afterwards in the state of charge and the net consumption.

## The battery model

**State of charge** = stored energy / nominal capacity (6.4 kWh). It starts at **0** (empty) at every reset.

**Efficiency curve.** Efficiency is a function of the power used, as a share of rated power. When the dataset does not give the curve, as here, CityLearn draws it at random around the nominal efficiency, from the environment's seed. The maximum-power curve, the maximum power as a share of rated power at each state of charge, is drawn the same way. With this dataset's seed (2022):

| Building | Efficiency at power share 0 / ~0.3 / ~0.7 / ~0.8 / 1 | Maximum power at state of charge 0 / ~0.8 / 1 |
|---|---|---|
| Building_1 | 0.79 / 0.84 / 0.89 / 0.90 / 0.87 | 0.98 / 0.93 / 0.26 |
| Building_2 | 0.78 / 0.83 / 0.89 / 0.90 / 0.87 | 0.97 / 0.92 / 0.24 |
| Building_3 | 0.78 / 0.82 / 0.89 / 0.90 / 0.86 | 0.97 / 0.92 / 0.23 |
| Building_4 | 0.80 / 0.85 / 0.90 / 0.90 / 0.88 | 0.99 / 0.94 / 0.29 |
| Building_5 | 0.80 / 0.84 / 0.89 / 0.90 / 0.87 | 0.98 / 0.93 / 0.27 |

Above about 80% charge, the maximum power falls steeply: the last 20% charges at a quarter of the rated power.

**Wear.** Every step, the capacity shrinks by capacity-loss coefficient × nominal capacity × |energy moved| / (2 × current capacity). With this dataset's coefficient, 10⁻⁵, that is about 0.001% of capacity per full cycle. **Measured:** a year of daily cycling at half power takes each battery from 6.4 to 6.377 kWh (−0.37%). The state of charge stays relative to the nominal capacity, so a worn battery's state of charge tops out below 1.

## Reward and objective

**Default reward** (this dataset's), per step: reward = −Σ over buildings of max(net consumption, 0).
- It is minus the electricity bought from the grid in the hour just finished, in kWh.
- Electricity given to the grid earns nothing.
- With a central agent, one number for all buildings; per building with decentralized agents.

| Measured over the year | Return | Reward per step |
|---|---:|---:|
| No control (all actions 0) | −29,256 | mean −3.34, from −16.7 to 0 |
| A naive daily cycle (charge 8:00–16:00 at half power, discharge otherwise) | −35,440 | mean −4.05 |

CityLearn has other reward functions to choose from (`reward_function`): sharing rewards between buildings (`MARL`), penalizing unused solar (`SolarPenaltyReward`), thermal comfort (`ComfortReward`, for datasets with indoor temperature), electric-vehicle targets, and combinations.

**Objective, as CityLearn scores it.** `env.evaluate()` computes key performance indicators, each divided by its value without any control, so 1.0 means "no better than doing nothing" and lower is better. At the district level:

| Indicator | Meaning |
|---|---|
| `cost_total` | money paid for grid electricity, Σ max(net × price, 0) |
| `carbon_emissions_total` | emissions of grid electricity, Σ max(net × carbon intensity, 0) |
| `electricity_consumption_total` | grid electricity bought, Σ max(net, 0) |
| `daily_peak_average`, `all_time_peak_average` | the district's daily and yearly peak demand |
| `ramping_average` | how much the district's demand jumps from hour to hour |
| `daily/monthly_one_minus_load_factor_average` | how uneven demand is (1 − average / peak) |
| `zero_net_energy` | how far from producing as much as it uses |
| discomfort and unserved-energy indicators | zero here (no indoor temperature, no outages in this dataset) |

The naive daily cycle scores worse than doing nothing on every indicator: cost 1.03, emissions 1.21, grid electricity 1.22, daily peak 1.61, ramping 1.76. Charging and discharging at the wrong times only adds losses and peaks. The 2022 challenge ranked agents by a combination of cost and emissions (phase 1) and, later, grid indicators.

## Episodes, resets and seeds

- **Episode end:** when the data run out (`terminated = True` at the last time step). CityLearn never sets `truncated`, and `info` is always an empty dict.
- **The final step is special:** CityLearn executes the last hour, but it does not advance its clock or refresh the observation. The terminal observation still shows the previous hour's net consumption and state of charge, and any reward computed from it would repeat the previous hour. Read the last hour from the building's arrays by step count (cost and charge of hour t at index t), as `city_env.py` does.
- **Shorter episodes:** `episode_time_steps` (an int, or a list of (start, end) windows), with `rolling_episode_split` (every hour can start an episode) and `random_episode_split` (windows drawn at random instead of in order).
- **Reset:** every battery is emptied (state of charge 0), its wear is undone, and the window for the next episode is chosen.
- **Seed:** `reset(seed=s)` sets the environment's seed. With `random_episode_split`, episode n's window is drawn with seed s × (n + 1), where n counts the episodes this environment object has run. Two consequences:
  - **seed 0 draws the same window every time;**
  - **the same seed gives a different window depending on how many episodes came before.**
  - Reproducible validation therefore needs explicit windows.
- The battery curves are drawn once, from the schema's seed (2022), when the environment is built.

**Speed:** about 1,000 steps per second for the raw environment on one CPU thread (a year in 8.8 s).

## As a test of autogenic context

| | BatteryPlane | CityLearn 2022 phase 1, as shipped |
|---|---|---|
| What the agent's actions change | the battery charge | each battery's stored energy, and its wear |
| Hidden from the agent | the battery charge | only the wear and the batteries' efficiency and power curves |
| The agent's own battery level | hidden | **shown** (state of charge) |
| How large the hidden part is | large: it decides the whole flight | small: wear 0.4% a year; the curves are fixed per building |

As shipped, **the battery level, the part the agent's actions create, is in the observation.** The hidden parts are either negligible (wear) or fixed constants of each building (the curves), not something the agent's actions create. There is little for a context encoder to infer.

Design options that would create an autogenic context. The first was tried ([Experiments](#experiments)) and set aside without a clear answer; the others have not been tried:
- **Hide the state of charge.** The agent's own charging creates it. It can be inferred from the history: each step's battery flow follows from net consumption, load and solar, and the stored energy follows from adding those flows up, with losses. A memoryless agent cannot know it.
- **Make wear substantial and hide the degraded capacity.** It would grow with the agent's own cycling, and show only when charging to full, when the state of charge stops below 1. That is the same structure as BatteryPlane's thrust cap.
- **Draw the battery parameters (capacity, efficiency) at random each episode.** This gives a hidden context, but a fixed one, not created by the agent.

As learned from the MyoSuite elbow task, any of these should first pass a value-of-context gate: plain PPO with the context shown vs hidden, before any Contextual PPO budget.

## Using it with Contextual PPO

**The learner works with CityLearn unchanged.** `contextual_ppo/` needs no update. A smoke test on this dataset ran with both encoders, GRU and xLSTM:
- week-long episodes, 2 environments;
- supervised encoder with a 5-dimensional label;
- 1,008 steps each;
- training and recurrent `predict()` both worked.

**Which installation.** Use `/root/venvs/carlos-citylearn`.
- **The separate, version-exact installation cannot run the learner:** Stable-Baselines3 2.3.2 lacks `FloatSchedule`, and it has no `xlstm`.
- **On `carlos`'s newer packages, CityLearn behaves the same up to float rounding.** A whole year with a fixed random action sequence gives a return of −49,928.57 against −49,928.54 (0.00006%). Observations and rewards differ by at most 0.003 kWh per step, and the scores by at most 2.6 × 10⁻⁶. The 2023 phase 2 dataset, with neural-network building models, agrees as closely. The cause is numpy 2 keeping some arithmetic in float32 where numpy 1 used float64.

**What is needed, all of it inside `contextual_environments/city/`:**
1. **Wrappers:**
   - **Central agent, normalized:** `central_agent=True`, then CityLearn's `NormalizedObservationWrapper` and `StableBaselines3Wrapper`, giving 47 observations in [0, 1] and 5 actions in [−1, 1].
   - **The episode end:** CityLearn reports the end of the data as `terminated`, so PPO treats the last state as worthless.
     - **Year-long or open-ended episodes, without time information:** a wrapper must report the end as `truncated`, so that Stable-Baselines3 bootstraps from `terminal_observation`.
     - **Fixed short episodes with the time left in the observation** (as in the experiment below): `terminated` is right, because the end is part of the task. Report the starting and ending stored energy then, since the policy may empty the battery before the end.
   - **A label in `info`:** the supervised encoder reads its target from the reset info and from every step's info (`context_info_keys=(reset_key, step_key)`). CityLearn's `info` is empty, so a wrapper must add the hidden context there. A label of several numbers also needs `rollout_buffer_kwargs={"context_dim": D}` and the policy's `context_dim = D`.
2. **Episode windows:**
   - **Shorter episodes** (for example a week, 168 steps), because the encoder's episode buffer stores and trains on whole episodes. A year is 8,759 steps.
   - **Explicit, separate time windows for training, validation and the final test,** for example held-out weeks. The data are one fixed year, and CityLearn's random windows depend on the episode count, with seed 0 giving no variety.
3. **A training script** like `myo/train_ppo.py`: builds `ContextualPPO` (or plain PPO), validation on fixed windows, W&B logging. The reward needs scaling: about −3 per step, down to −17.
4. **A hidden context,** chosen from the options above. Without one, Contextual PPO has nothing to infer.

The executed battery energy can be smaller than the action asks (power limit, full, empty). The encoder reads the action as sent, as in BatteryPlane, and the gap shows in the next observation.

## Experiments

### Setup: one building, its battery, the charge as the context

The task is in [city_env.py](city_env.py), training in [train_ppo.py](train_ppo.py), evaluation in [evaluate.py](evaluate.py), and the checks in [city_checks.py](city_checks.py). All are self-contained and run with `/root/venvs/carlos-citylearn/bin/python`.

| | |
|---|---|
| Building | Building_1 of `citylearn_challenge_2022_phase_1`: a 6.4 kWh / 5 kW battery, 4 kW of solar panels |
| Episode | **48 hours** (48 steps), starting at a **random hour**, with a **random starting charge**, uniform in [0, 1] |
| Reward | minus the money paid for grid electricity in the hour: −max(net consumption × price, 0), in $. Exports earn nothing. This is CityLearn's cost indicator, per step. |
| Action | 1 number in [−1, 1], the battery's charge (+) or discharge (−) as a share of its rated power |
| Observation, **charge observed** | CityLearn's normalized observation of the building (31 numbers in [0, 1]) plus the **time left** in the episode (1 at the start, 0 at the end): 32 numbers |
| Observation, **charge hidden** | the same without the state of charge: 31 numbers, otherwise identical |
| Episode end | `terminated` after 48 steps; with the time left observed, the end is part of the task |

**Prices.** The tariff is cheap all day ($0.21, or $0.22 from June to September) except for one expensive block from 16:00 to 20:00: $0.50 on most days, $0.54 on summer weekdays, $0.40 on summer weekends. The mean is $0.273 per kWh.

**Training, validation and final test never mix.** The year's weeks are split, spread over the seasons:
- **validation weeks:** 11, 24, 37, 50;
- **final-test weeks:** 5, 18, 31, 44;
- **training weeks:** the rest.

Training episodes start at any of 6,984 hours whose 48 hours stay in training weeks.

Validation is **80 fixed scenarios**, each a (start hour, starting charge) pair, 20 per validation week, each inside its week. The mean starting charge is 0.50. A scenario sets the same start and starting charge for both agents. The 80 final-test scenarios are reserved and have not been used.

**Measures per policy,** on the 80 validation scenarios:
- **Cost** per 48-hour episode.
- **Energy-adjusted cost:** cost + (starting stored energy − ending stored energy) × $0.273/kWh, the tariff's mean price. Spending the free starting charge, or emptying the battery just before the end, then does not count as a saving. Starting and ending stored energy are reported too.
- **Saturation:**
  - the share of hours that end with the battery empty (charge ≤ 2%) or full (≥ 95%);
  - the median hours until it is first empty or full;
  - the share of episodes that get there within 6 hours.
- **Predictability of the charge:** a gradient-boosted regressor learns the charge from the hidden-setting observation, on 300 training scenarios flown by the same policy, and is scored (R²) on the validation flights. A second regressor uses the clock alone (hour, weekday, month, time left). A high R² means a memoryless agent effectively knows the charge.

**Checks** ([city_checks.py](city_checks.py), all pass):
- the hidden observation differs from the observed one only by the charge, with the same rewards for the same actions;
- a scenario fixes the start and the starting charge;
- episodes last 48 steps, with the time left counting down;
- each step's reward is the cost of its hour, and an episode's rewards add up to CityLearn's own cost total;
- the energy accounting matches the battery;
- training, validation and final-test hours never mix.

### 1. Charge observed vs hidden, plain PPO, 2M steps (22 October 2026)

**Training.** Plain Stable-Baselines3 PPO with BatteryPlane's baseline settings:
- 8 environments × 512 steps per update, minibatches of 256, 10 epochs;
- learning rate 3 × 10⁻⁴, γ = 0.99, GAE λ = 0.95, clip 0.2, entropy 0;
- actor and critic with two hidden layers of 256 (tanh);
- rewards in $, unscaled.

**5 seeds (0–4) per setting, 2 million steps,** validation every 50,000 steps. Each run took 20 minutes, with 10 side by side. W&B project `contextual-city`, groups `PPO_B1_h48_cost_seeCharge_rs1` and `PPO_B1_h48_cost_hidden_rs1`.

**Hand-written rules first** (a preliminary diagnostic of what the charge can be worth, not an upper bound: the rules may all be weak). Rules see the coming hour's load, solar and price, and the tariff schedule; only the last also reads the charge.

| Rule | Cost | Energy-adjusted cost | End energy (start 3.21 kWh) | Hours empty / full | R², hidden observation | R², clock alone |
|---|---:|---:|---:|---:|---:|---:|
| No battery use | $11.89 | $11.89 | 3.21 | 3% / 5% | — | — |
| Self-consumption: charge from leftover solar, cover any demand from the battery | **$7.07** | **$7.39** | 2.02 | 51% / 19% | 0.84 | 0.62 |
| Time-of-use: charge from leftover solar, cover demand only in expensive hours | $7.48 | $7.65 | 2.61 | 32% / 23% | 0.69 | 0.55 |
| Time-of-use plus grid charging at full power in the 6 cheap hours before an expensive block | $8.04 | $8.19 | 2.67 | 29% / 29% | 0.72 | 0.69 |
| The same, with grid charging paced by the charge to be full when the block starts (charge-aware) | $7.39 | $7.54 | 2.67 | 29% / 24% | 0.71 | 0.63 |

The best rule is blind, and it beats the charge-aware one. The battery refuses on its own when it is full or empty, so a blind rule loses little by asking at the wrong time. With a single flat-priced expensive block per day, there is little to plan.

**PPO results** (80 validation scenarios, mean ± standard deviation over 5 seeds):

| Policy | Cost | Energy-adjusted cost | End energy (start 3.21 kWh) | Hours empty / full | Median hours to empty or full | R², hidden observation | R², clock alone |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO, charge observed | $7.74 ± 0.07 | $8.23 ± 0.08 | 1.45 | 59% / 18% | 1.4 | 0.80 | 0.68 |
| PPO, charge hidden | $7.66 ± 0.08 | $8.16 ± 0.09 | 1.40 | 60% / 15% | 1.4 | 0.88 | 0.78 |

**The gate: hiding the charge costs nothing measurable.**

| Hidden − observed | Gap | Bootstrap 95% CI | Welch 95% CI, p | Scenarios where hidden pays more |
|---|---:|---|---|---:|
| Cost | −$0.08 (−1.0%) | −$0.18 to +$0.01 | −$0.20 to +$0.04, p = 0.17 | 34% |
| Energy-adjusted cost | −$0.07 (−0.8%) | −$0.17 to +$0.03 | −$0.20 to +$0.07, p = 0.29 | 39% |

The hidden agents are, if anything, slightly cheaper.

**Why the charge carries little information here.** The battery is almost always at a limit:
- both agents leave it empty about 60% of hours and full another 15–18%;
- they reach empty or full within 1–2 hours (median) in over 99% of episodes, so the random starting charge is erased almost at once;
- after that, the charge is largely predictable from what the hidden agent sees: R² 0.88 on the hidden agents' own flights, and 0.78 from the clock alone.

**The episode boundary affects both agents equally.** Both spend about 1.8 kWh of the free starting energy (ending at about 1.4 kWh), which the adjusted cost charges at about $0.49 per episode. It does not explain the comparison.

**But PPO is still weak, so this does not show that the charge is useless.**
- Both PPO agents are about 10% worse than the blind self-consumption rule on energy-adjusted cost ($8.16–8.23 vs $7.39).
- Their validation cost was still falling at 2M steps:

| Validation energy-adjusted cost, mean of 5 seeds | 0.5M | 1M | 1.5M | 2M |
|---|---:|---:|---:|---:|
| Observed | $9.15 | $8.48 | $8.35 | $8.22 |
| Hidden | $8.85 | $8.35 | $8.19 | $8.15 |

- Their best checkpoints are at 1.45–2M steps.

A learner that has not converged may simply not use the charge yet, so the next step was to train longer (experiment 2).

```bash
/root/venvs/carlos-citylearn/bin/python contextual_environments/city/city_checks.py -v
/root/venvs/carlos-citylearn/bin/python contextual_environments/city/train_ppo.py --seed 0 --observe-charge     # observed
/root/venvs/carlos-citylearn/bin/python contextual_environments/city/train_ppo.py --seed 0                      # hidden
OMP_NUM_THREADS=1 /root/venvs/carlos-citylearn/bin/python contextual_environments/city/evaluate.py --rules --runs <run dirs> --output runs/gate/gate_h48_cost.json
```

Set `OMP_NUM_THREADS=1` for the evaluation. Its regressors otherwise each use every core, and parallel workers slow to a crawl.

### 2. The same, for 10 million steps (22 October 2026)

**Design.** Experiment 1 unchanged, but for 10 million steps instead of 2 million, with validation and a checkpoint every 250,000 steps. Same 5 seeds per setting. Each run took about 102 minutes, with 10 side by side. W&B groups `PPO_B1_h48_cost_seeCharge_rs1_10M` and `PPO_B1_h48_cost_hidden_rs1_10M`.

Training is deterministic: up to 2 million steps these runs reproduce experiment 1 exactly ($8.216 observed and $8.153 hidden, energy-adjusted, at 2M in both).

**Learning curves** (validation energy-adjusted cost, mean of 5 seeds):

| | 2M | 4M | 6M | 8M | 10M (final) |
|---|---:|---:|---:|---:|---:|
| Observed | $8.22 | $8.07 | $8.06 | $8.04 | $8.03 |
| Hidden | $8.15 | $8.14 | $8.19 | $8.18 | $8.09 |

Both levelled off after about 4 million steps.

**Results** (80 validation scenarios, final models, mean ± standard deviation over 5 seeds):

| Policy | Cost | Energy-adjusted cost | End energy (start 3.21 kWh) | Hours empty / full | Median hours to empty or full | R², hidden observation | R², clock alone |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO, charge observed | $7.55 ± 0.04 | $8.03 ± 0.08 | 1.45 | 59% / 16% | 1.8 | 0.83 | 0.72 |
| PPO, charge hidden | $7.60 ± 0.07 | $8.09 ± 0.09 | 1.44 | 60% / 15% | 1.7 | 0.87 | 0.77 |
| *Blind self-consumption rule* | *$7.07* | *$7.39* | *2.02* | *51% / 19%* | *3* | *0.84* | *0.62* |

| Hidden − observed | Gap | Bootstrap 95% CI | Welch 95% CI, p | Scenarios where hidden pays more |
|---|---:|---|---|---:|
| Cost | +$0.05 (+0.7%) | −$0.01 to +$0.13 | −$0.05 to +$0.15, p = 0.24 | 66% |
| Energy-adjusted cost | +$0.06 (+0.7%) | −$0.04 to +$0.17 | −$0.09 to +$0.20, p = 0.39 | 66% |

**What it says:**
- **Still no measurable cost of hiding the charge.** With longer training the observed agents pulled slightly ahead: the hidden ones now pay more in 66% of scenarios, against 34% at 2M. But the gap, 0.7%, is not significant.
- **Neither setting beats the blind rule.** The bootstrap 95% CI of the mean energy-adjusted cost is $7.96–8.10 observed and $8.02–8.18 hidden, against the rule's $7.39.
- **The battery still sits at a limit about 75% of the time,** and its charge stays largely predictable from what a hidden agent sees.

### 3. A stronger learner: SAC with the charge observed (22 October 2026)

**The question.** Is the charge information unhelpful, or is PPO simply failing to use it? If a stronger learner that sees the charge beats the blind rule, the observed-vs-hidden comparison can be repeated with it. If it also loses to the rule, the learning problem remains unresolved, and nothing can be concluded about the charge.

**Design.** The same task as experiments 1–2: Building_1, cost reward, 48-hour episodes from a random hour, random starting charge, time left observed, the same 80 validation scenarios. The final-test scenarios are still unused.
- **SAC** ([train_sac.py](train_sac.py)), Stable-Baselines3 2.9.0, unmodified, with its standard settings:
  - one environment, replay buffer of 1 million transitions, batch 256;
  - learning rate 3 × 10⁻⁴, γ = 0.99, τ = 0.005;
  - one gradient step per environment step, entropy coefficient tuned automatically;
  - actor and two critics with two hidden layers of 256 (ReLU).

  The one change from Stable-Baselines3's defaults is 10,000 random steps before learning starts, instead of 100.
- **Charge observed, 5 seeds (0–4), 500,000 steps,** validation and a checkpoint every 25,000 steps. Each run took 84 minutes (SAC runs at about 120 steps per second, against PPO's 1,500–1,700). W&B group `SAC_B1_h48_cost_seeCharge_rs1`.
- **Decision rule, fixed in advance:** SAC beats the rule if the bootstrap 95% CI of its mean energy-adjusted cost (final models, 5 seeds) lies entirely below the self-consumption rule's $7.39.

**Results** (80 validation scenarios, final models):

| Policy | Cost | Energy-adjusted cost (95% CI of the mean) | End energy | Hours empty / full | Median hours to empty or full | R², hidden observation | R², clock alone |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAC, charge observed | $7.36 ± 0.11 | $7.94 ± 0.14 ($7.81–8.06) | 1.09 | 54% / 17% | 2.2 | 0.86 | 0.80 |
| *Blind self-consumption rule* | *$7.07* | *$7.39* | *2.02* | *51% / 19%* | *3* | *0.84* | *0.62* |

Per seed, energy-adjusted: $8.06, $7.71, $7.95, $8.11, $7.85. SAC minus the rule is +$0.55 energy-adjusted (+7%) and +$0.29 raw (+4%). SAC is cheaper than the rule in 16% of scenarios (20% on raw cost).

| Validation energy-adjusted cost, mean of 5 seeds | 25k | 50k | 100k | 200k | 300k | 400k | 500k |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAC, charge observed | $8.76 | $8.52 | $8.24 | $8.20 | $8.09 | $8.01 | $7.93 |

**Verdict: SAC does not beat the rule, so the learning problem remains unresolved.**
- SAC is clearly the stronger learner. After 500,000 steps it is ahead of PPO after 10 million ($7.94 against $8.03), and it was still improving when it stopped, with its best checkpoints at 400,000–500,000 steps.
- Part of its gap to the rule comes from the episode boundary. The training reward is the raw cost and gives no value to energy left at the end, so it is rational for a learned agent to empty the battery before the episode ends: SAC ends at 1.09 kWh, the rule at 2.02. But SAC also loses on raw cost, its own objective, by 4%, so the boundary is not the whole story.

```bash
/root/venvs/carlos-citylearn/bin/python contextual_environments/city/train_sac.py --seed 0 --observe-charge --steps 500000
OMP_NUM_THREADS=1 /root/venvs/carlos-citylearn/bin/python contextual_environments/city/evaluate.py --rules --runs <SAC run dirs> --output runs/gate/sac_seeCharge_vs_rule.json
```

### Where this leaves us

We have set this test aside for now. It was worth trying: CityLearn's battery charge is a natural autogenic context, created by the agent's own charging and hidden by simply leaving it out of the observation. The pipeline worked as intended. Observed and hidden agents saw the same scenarios, the episode boundary was accounted for, and every number was checked against CityLearn's own accounting. But the experiments did not give a clear answer.

| Experiment | Question | Outcome |
|---|---|---|
| Rules | What could the charge be worth? (preliminary) | the best rule ignores the charge |
| 1. PPO, 2M steps | Does hiding the charge cost anything? | no measurable cost (−1.0%), but PPO is 10% worse than the blind rule |
| 2. PPO, 10M steps | Does longer training change that? | no measurable cost (+0.7%); PPO levels off 9% above the blind rule |
| 3. SAC, 500k steps | Can a stronger learner beat the blind rule? | no: 7% above it, though still improving |

**What we learned:**
- **The battery spends about three quarters of its hours empty or full,** under every policy tried. It reaches a limit within 1–3 hours of the start, so the random starting charge is soon forgotten. After that, the charge is largely predictable from what a hidden agent sees (R² 0.83–0.88). That alone may make the charge worth little in this task.
- **The tariff makes a blind rule strong.** With one flat-priced expensive block per day, and a battery that simply refuses when it is full or empty, a rule loses little by not knowing the charge.
- **Neither learner reached the rule.** Until one does, the observed-vs-hidden comparison cannot separate "the charge is not useful" from "the learner cannot use it yet".

**Ideas for a later attempt** (none tried):
- **Train on the energy-adjusted cost,** with a credit for energy left at the end, so the training objective matches the evaluation and emptying the battery before the end no longer pays.
- **Train SAC longer** (it was still improving), for example 1.5 million steps.
- **Residual learning:** SAC learns corrections on top of the self-consumption rule, starting at the rule's level. The observed-vs-hidden comparison then asks whether the charge helps improve on the rule.
- **Make the charge carry more information:**
  - a larger battery relative to the building's load, so it rarely sits empty or full;
  - a tariff with several price levels per day, so that how much charge is left matters for when to spend it.

## References

- J. R. Vázquez-Canteli, J. Kämpf, G. Henze, Z. Nagy. *CityLearn v1.0: An OpenAI Gym environment for demand response with deep reinforcement learning.* BuildSys 2019.
- K. Nweye et al. *CityLearn v2: energy-flexible, resilient, occupant-centric, and carbon-aware management of grid-interactive communities.* Journal of Building Performance Simulation, 2024.
- Documentation: https://www.citylearn.net. Source: https://github.com/citylearn-project/CityLearn
