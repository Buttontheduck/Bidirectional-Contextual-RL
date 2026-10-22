# Gym4ReaL MicroGridEnv: a household battery that wears out

Gym4ReaL is a suite of environments for testing reinforcement learning under real-world conditions (Salaorni et al., 2025). Its **MicroGridEnv** is a household with solar panels and a battery, connected to the electricity market:
- every hour the household uses electricity and the panels produce some;
- the agent decides how much of the difference goes into or comes out of the battery, and the rest is sold to or bought from the grid;
- the battery is a detailed digital twin that heats up and **wears out with use and with time**;
- the reward is the money earned from trading minus the cost of that wear.

This document describes the environment as installed here, read from its source code and checked by running it.

**Results so far (22 October 2026)** (see [Experiments](#experiments)):
- **The project's pretrained PPO reproduces in the unchanged environment and beats every rule-based baseline:** +€21–27 per household per year over the best rule.
- **Retrained with the same recipe,** PPO gains nothing measurable from seeing the battery's state of health: +€7.3 per household per year (+2.3%), with a 95% CI of −€7.6 to +€21.3.

- **Source:** https://github.com/Daveonwave/gym4ReaL, commit `774d5ce` (3 July 2025), cloned to `/root/venvs/src/gym4ReaL`. It is not a pip package (its `pyproject.toml` is an unfilled template), so both Python environments reach it through a `.pth` file.
- **Installed twice:**
  - **`/root/venvs/gym4real`:** fully separate, with the repository's pinned versions: gymnasium 1.1.1, numpy 2.2.4, pandas 2.2.3, scipy 1.15.2, PyYAML 6.0.2. It uses Python 3.11; the repository names 3.12, but the code needs only 3.10.
  - **`/root/venvs/carlos-gym4real`:** built on the `carlos` conda environment and seeing all its packages (Stable-Baselines3 2.9, xLSTM, numpy 2.4.6, gymnasium 0.29.1), plus pandas 2.2.3 and joblib. **Use this one with Contextual PPO.** Over 5,000 steps with the same household and actions, it reproduces the separate installation **exactly**: every observation, reward and battery state is bit-identical.
  - **Gym4ReaL does not run with pandas 3** (`carlos` has 3.0.5): pandas 3 returns read-only arrays, and the market simulator scales them in place. Hence pandas 2.2.3 in this environment only; `carlos` itself is unchanged.
- **Create it.** The configuration paths are relative to the repository, so run from its folder or `os.chdir` there first:

```python
import os; os.chdir("/root/venvs/src/gym4ReaL")          # run with /root/venvs/carlos-gym4real/bin/python
from gym4real.envs.microgrid.utils import parameter_generator
from gym4real.envs.microgrid.env import MicroGridEnv
env = MicroGridEnv(settings=parameter_generator())        # or gymnasium.make("gym4real/microgrid-v0", settings=...) after `import gym4real`
obs, info = env.reset(options={"eval_profile": "0"})      # household "0"; without the option a household is drawn at random
obs, reward, terminated, truncated, info = env.step([0.5])
```

`parameter_generator(world_options="gym4real/envs/microgrid/world_test.yaml")` gives the test year instead.

## The household and its data

| Part | Data | In numbers |
|---|---|---|
| Electricity demand | 398 measured Italian households, hourly, **one year** that repeats: 370 for training (columns 0–369), 28 for testing (370–397) | yearly mean 37–824 W per household (median 283 W), up to 3,972 W in an hour |
| Solar panels | a 3 kW installation in northern Italy, hourly (renewables.ninja) | training 2015–2018, test 2019; mean 466 W, up to 2,208 W |
| Electricity prices | Italian market (GME), hourly. **Ask:** the buying price, including taxes and fees. **Bid:** the selling price. | ask 0.093–0.257 €/kWh (mean 0.140), bid 0.006–0.170 €/kWh (mean 0.053) |
| Ambient temperature | daily | training −0.8 to 29.7 °C (mean 14.7), test 1.7 to 28.5 °C |

Buying costs on average 2.6 times what selling earns, so using your own solar energy is worth more than selling it.

| Battery (an NMC lithium-ion pack) | |
|---|---|
| Nominal voltage, capacity | 350.4 V, 60 Ah (about 21 kWh) |
| Voltage range | 288.0–398.4 V |
| Usable charge | 20%–100% of the current capacity (about 16.8 kWh when new) |
| Rated life | 3,000 cycles at 80% depth of discharge |
| Replacement cost | €3,000 |
| End of life | state of health 0.6 |
| Start (default) | charge 20%, health 1.0, temperature equal to the day's ambient temperature |

## Time and episodes

- **One step is one hour.**
- **Default episode:** it always starts on 1 January 2015 at 01:00 and runs to the end of the training data, **35,063 steps (4 years)**. The household is drawn at random.
- **The episode ends:**
  - **terminated:** when the battery's health falls to 0.6, its end of life;
  - **truncated:** when the solar or price data run out. The environment prints "Generation history is run out-of-data" then.
- **Measured:** with the default settings no battery reached end of life within the 4 years. Health ended at 0.90–0.93 under every simple policy tried, so every episode ended with the data.
- **Test configuration:** 2019 only, 8,760 steps, with the 28 test households.

## State space

The full state of the simulation, most of which the agent does not see:

| Part | Seen by the agent (default)? |
|---|---|
| Battery charge (SoC, a share of the current capacity) | yes |
| Battery temperature | yes |
| **Battery state of health (SoH)** and with it the current capacity (SoH × 60 Ah) | **no** (optional) |
| Accumulated battery damage: running mean charge and temperature, counted cycles | no |
| Battery voltage, current, internal resistances, open-circuit voltage | no |
| This hour's demand and solar output | no: the agent sees the **previous** hour's |
| Which household this is | no |
| Prices now (ask and bid) | yes |
| Time of day and time of year | yes |
| Ambient temperature | no |

## Observation space

**10 numbers, not normalized:**

| Entry | Unit | Meaning |
|---|---|---|
| `temperature` | K | battery temperature now |
| `soc` | 0.2–1 | battery charge, as a share of its current (worn) capacity |
| `demand` | W | the household's mean demand in the **previous** hour |
| `generation` | W | the panels' mean output in the **previous** hour |
| `ask` | €/Wh | buying price this hour |
| `bid` | €/Wh | selling price this hour |
| `sin/cos_day_of_year` | −1 to 1 | time of year |
| `sin/cos_seconds_of_day` | −1 to 1 | time of day |

Adding `soh` to the `observations` list in the world file (or setting `settings["soh"] = True`) inserts the **state of health** after `demand`, giving 11 numbers.

The observation space bounds demand, generation and prices only from below (0 to ∞).

## Action space

**One number in [0, 1]:** the share of this hour's **margin** (solar output − demand) that goes through the battery. The rest is traded with the grid.

| Hour | a = 0 | a = 1 |
|---|---|---|
| Surplus (more sun than demand) | sell all of it | store all of it |
| Deficit (more demand than sun) | buy all of it | take all of it from the battery |

- **The agent cannot charge from the grid or sell from the battery.** It only decides how to split the household's own surplus or deficit.
- **The battery takes or gives at most what keeps its charge within 20%–100%.** The rest of the request is traded and penalized (the clipping term below). There is no other power limit.
- The action is a share of the **actual** margin of this hour, which the agent does not see: it sees the previous hour's demand and solar.

## The battery model

The battery is the digital twin ErNESTO-DT (cited in the paper as Salaorni, 2023), with four parts:

| Part | Model |
|---|---|
| Electrical | Thevenin equivalent circuit (series resistance, one resistor-capacitor pair, open-circuit voltage), its parameters looked up by temperature and charge; driven by power |
| Thermal | one thermal mass (138,240 J/K) heated by the battery's losses and cooled toward the ambient temperature |
| Charge | counted from the current, relative to the **current** capacity, kept within 20%–100% |
| Aging | Bolun's model (Xu et al., 2018): calendar aging plus cycle aging; cycles counted by rainflow as they happen |

**How the battery wears** (configuration `aging.yaml`). An accumulated damage d grows with two stresses:
- **Calendar aging,** all the time: 4.14 × 10⁻¹⁰ per second × e^(1.04 × (mean charge − 0.5)) × e^(0.0693 × (mean temperature − 298.15 K) × 298.15 / mean temperature). The means are over the episode so far. **A battery kept full ages faster.**
- **Cycle aging:** every charge–discharge cycle adds a stress that grows with its depth, its mean charge and the temperature.

Health is then SoH = start health − [1 − 0.0575 e^(−121 d) − 0.9425 e^(−d)].
- The first term is the growth of a surface film (SEI), which happens mostly early. **Per unit of stress, a fresh battery loses health about 8 times faster than one past its first ~5% of loss.**
- The battery's capacity is SoH × 60 Ah, so **wear shrinks how much energy a "full" battery holds,** and the observed charge is measured against that shrunk capacity.
- The damage d starts from zero at every reset, whatever the starting health.

**Measured with the default settings,** first year, three households:

| Household (yearly mean demand) | Policy | Health after 1 year | Charge: mean, share of hours at 20% / at 100% |
|---|---|---:|---|
| any | a = 0 (battery unused, kept at 20%) | 0.971 | 0.20, 100% / 0% |
| low (128 W) | a = 1 (battery first) | 0.948 | 0.97, 0% / 35% |
| median (283 W) | a = 1 | 0.950 | 0.82, 3% / 21% |
| high (521 W) | a = 1 | 0.957 | 0.52, 22% / 7% |

**Calendar aging dominates.** An unused battery at 20% charge still loses about 3% of its health a year. Using it mostly adds calendar aging at a high charge, and less from the cycles themselves.

## Reward and objective

**Objective:** earn as much as possible from trading over the battery's life, net of the battery's wear. The paper treats it as an infinite-horizon task.

**Reward per step:** r = r_trad / 1.0208 + r_deg + 0.1 × r_clip / 3,972.

| Term | Value | Unit |
|---|---|---|
| r_trad | energy traded × price: sold energy × bid (> 0), bought energy × ask (< 0) | € per hour |
| r_deg | −(health lost this step) × €3,000 / (1 − 0.6), i.e. **€75 per percentage point of health** | € |
| r_clip | −(requested battery power − power the battery could take) | W |

- The divisors are the configured normalizations: €1.0208 is the largest possible hourly trade (3,972 W × 0.257 €/kWh), and 3,972 W the largest possible clipping. The weights are 1, 1 and 0.1.
- **The terms are on different scales.** Trading is divided by about €1, so it stays roughly in euros. Wear is not normalized at all.
- **Wear is charged even when the battery is idle:** calendar aging costs about €215 a year with a = 0.
- The step `info` gives each term raw (`pure_rewards`), normalized and weighted, plus a full battery snapshot.

**Measured: full default episodes** (household "0", low demand, 4 years):

| Policy | Return | Trading | Wear cost | Health at the end |
|---|---:|---:|---:|---:|
| a = 0: only the market (battery unused) | −343 | +€172 | −€511 | 0.932 |
| a = 1: battery first | −502 | +€488 | −€724 | 0.904 |
| a = 0.5 | −527 | +€330 | −€723 | 0.904 |
| random | −524 | +€330 | −€722 | 0.904 |

**Measured: first year, three households:**

| Household | Return a = 0 | Return a = 1 | Trading a = 0 / a = 1 | Wear a = 0 / a = 1 |
|---|---:|---:|---|---|
| low demand (128 W) | **−118** | −317 | +€99 / +€152 | −€215 / −€391 |
| median (283 W) | **−260** | −351 | −€46 / +€74 | −€215 / −€378 |
| high (521 W) | −489 | **−468** | −€280 / −€103 | −€215 / −€331 |

**The battery earns more from trading, but often costs more in wear than it earns.** For low- and median-demand households, leaving it unused gives the best of these simple policies. For high-demand households, using it pays slightly. The repository's rule-based baselines are exactly these: only the market (a = 0), battery first (a = 1), 50-50 (a = 0.5), and random.

## Resets, seeds and options

- **Household:** drawn at every reset from numpy's **global** random generator, which the constructor seeds with 42. `reset(seed=...)` does not change the draw. `reset(options={"eval_profile": "<id>"})` picks a household.
  - Several copies in one process share the global stream.
  - Copies in separate processes all draw the same sequence, since each constructor reseeds with 42.
- **`random_data_init=True`:** starts at a random hour, but the start is wrapped into the **first year** of the data, so only the time of year varies. The solar and price data always continue from 2015.
- **`random_battery_init=True`:** draws a random starting state from the configured bounds:
  - health 0.6–1 (examples: 0.62, 0.69, 0.85);
  - charge 0.2–1;
  - temperature 0–50 °C.

  A battery starting near 0.6 reaches end of life soon after.
- **Other settings** (`parameter_generator` arguments): reward weights, reward normalization on or off, end-of-life threshold, step length, replacement cost, a price spread factor.

**Speed:** about 2,500–5,000 steps per second on one CPU thread (a 4-year episode in 6–15 s).

## What can be hidden

| Hidden quantity | Hidden by default? | Who changes it | How much it varies | Does it matter for the best action? | Can it be inferred from the history? |
|---|---|---|---|---|---|
| **State of health, and the battery's current capacity** | **yes** (`soh` optional) | **the agent**, through calendar aging (its average charge) and cycle aging (its cycles), plus time | 3–5% per year by default; with `random_battery_init`, a random start in 0.6–1 | through the capacity (how much a full battery holds), the end of life at 0.6, and the wear cost: a fresh battery wears up to 8× faster per unit of stress | yes: the charge moves by energy / capacity, and last hour's energy flow follows from the observed previous-hour demand and solar and the agent's own action |
| Accumulated stress (running mean charge and temperature, counted cycles) | yes, always | the agent | grows over the episode | it sets future calendar and cycle aging | partly: the agent can track its own charge history |
| State of charge | no (observed); could be removed | the agent | 20%–100% within hours | yes: how much can still be stored or used | yes, as above (as for CityLearn's battery charge) |
| Battery temperature | no (observed); could be removed | mostly the weather (−0.7 to 29.7 °C under every policy tried); little from the agent | daily and seasonal | weakly, through wear | partly, from time of year |
| Which household this is | yes | nobody (fixed per episode) | yearly mean 37–824 W | yes: for low-demand households the battery barely pays; for high-demand ones it does | yes, from the observed demand history |
| This hour's demand and solar output | yes (the previous hour is shown) | the data | hour to hour | yes | partly, from the previous hours |
| Ambient temperature | yes | the data | daily | weakly | partly |

The `info` dict carries the privileged values a supervised encoder could learn from: health, current capacity, damage, calendar and cycle aging, temperature, voltage, current. These are never in the observation unless requested.

## As a test of autogenic context

The battery's **state of health** is the natural autogenic context here:
- the agent's own use creates it, through its average charge and its cycles;
- it is hidden by default;
- it limits what the agent can do, since a worn battery holds less;
- it can be inferred from the history.

That is the same structure as BatteryPlane's battery. The measurements above suggest caution, though, following the lessons of the MyoSuite and CityLearn attempts in this folder:
- **It changes slowly.** 3–5% of health per year by default. Within any one day the context is nearly constant, and with the default fixed start, health is largely a function of how long the episode has run.
- **The wear cost does not depend on the household's choices alone.** Calendar aging, charged even for an unused battery, dominates the cost. The best simple policy for most households is not to use the battery at all, which leaves little room for any context to matter.
- **`random_battery_init` makes health a large hidden context per episode** (0.6–1). But a battery starting near 0.6 ends soon, and the damage model restarts at every reset, so the start health and the wear rate are decoupled.

As learned from MyoSuite and CityLearn, any design here should first pass a value-of-context gate:
1. check that a learner beats the rule-based baselines;
2. then compare that learner with health shown vs hidden;

before any Contextual PPO budget. Both steps were run in the unchanged environment ([Experiments](#experiments)). The learner beats the rules, but showing health gives no measurable benefit.

## Experiments

Both experiments use the unchanged environment through `/root/venvs/carlos-gym4real/bin/python`:
- [evaluate_checkpoint.py](evaluate_checkpoint.py) evaluates any PPO model and the rule baselines as the project's notebook does;
- [train_ppo.py](train_ppo.py) trains PPO with the recipe of the project's pretrained model.

Results and models are in `runs/` (ignored by Git), and learning curves in the W&B project `contextual-microgrid`.

**Evaluation protocol,** the project's notebook `examples/microgrid/benchmarks.ipynb`, unchanged:
- the test configuration (`world_test.yaml`: the 2019 data), with `parameter_generator`'s defaults;
- all **28 test households** (profiles 370–397), one full year each (8,759 hours);
- rule baselines: random, only market (a = 0, battery unused), battery first (a = 1), 50-50 (a = 0.5);
- PPO run through `make_vec_env` and `model.predict(obs)`:
  - **sampled actions,** Stable-Baselines3's default, as in the notebook; here over 5 evaluation seeds, which differ by less than €1;
  - plus a **deterministic** run;
- **score:** per household, the year's total trading reward plus total wear reward, both raw, in € (higher is better).

The two additions for reproducibility are a seeded random baseline and seeded PPO sampling. The final-test notion of the main project does not apply here: the notebook's test households are the benchmark's own held-out set, and no choice was tuned on them.

### 1. The project's pretrained PPO, in the unchanged environment (22 October 2026)

**The checkpoints** (`examples/microgrid/trained_models/`):
- **`PPO_trained`:** 14,025,600 steps, which is 100 four-year episodes on each of 4 environments. Recipe in experiment 2. Saved with Stable-Baselines3 2.3.2 and gymnasium 0.28.1; it loads unchanged in Stable-Baselines3 2.9.0.
- **`PPO_quick`:** the training tutorial's run, default PPO, 5 episodes on each of 4 environments (701,280 steps).
- Neither ships with observation-normalization statistics. The notebook evaluates them on raw observations, and so does this evaluation.

**Results** (28 test households, score per household per year; ± is the notebook's 95% half-width, 1.96 × std / √28):

| Policy | Score | Trading | Wear | Health after the year |
|---|---:|---:|---:|---:|
| **PPO_trained, deterministic** | **−€307 ± 36** | −€30 | −€278 | 0.963 |
| PPO_quick, deterministic | −€310 ± 40 | −€71 | −€239 | 0.968 |
| **PPO_trained, sampled (as in the notebook)** | **−€313 ± 34** | −€26 | −€287 | 0.962 |
| PPO_quick, sampled | −€325 ± 37 | −€59 | −€266 | 0.965 |
| Only market: the best rule | −€334 ± 43 | −€117 | −€217 | 0.971 |
| Battery first | −€356 ± 26 | +€12 | −€368 | 0.952 |
| Random | −€410 ± 29 | −€52 | −€358 | 0.953 |
| 50-50 | −€410 ± 29 | −€52 | −€358 | 0.953 |

**Against the best rule, household by household** (bootstrap 95% CI over the 28 households):

| Policy | Gain over "only market" | Households where PPO is better |
|---|---|---:|
| PPO_trained, sampled | **+€21.5** (+€9.1 to +€33.0) | 86% |
| PPO_trained, deterministic | +€26.9 (+€15.7 to +€37.1) | 89% |
| PPO_quick, sampled | +€9.4 (€0.0 to +€17.7) | 71% |
| PPO_quick, deterministic | +€24.6 (+€19.5 to +€28.7) | 96% |

**What it says:**
- **The pretrained agent reproduces and beats every rule,** matching the paper's conclusion that "PPO achieves higher profit than rule-based strategies".
- **It wins by using the battery in moderation.** Compared with only market, it earns €87 more from trading and pays €60 more in wear. Battery first trades best but wears the battery most.
- **The gain varies across households,** from −€59 to +€68. That fits the paper's remark that PPO "has a large variance".
- **Two small mismatches with the paper:** its text says testing used 2020 data and training used 8 environments, while the repository's test data is 2019 and the checkpoint records 4 environments. This evaluation follows the repository as shipped.

### 2. Battery health shown vs hidden, with PPO_trained's recipe (22 October 2026)

**Question:** with a learner that beats the rules, does seeing the battery's state of health help?

**Design:**
- **Environment unchanged:** `world_train.yaml` through `parameter_generator`'s defaults:
  - 2015–2018 data;
  - one of the 370 training households drawn at every reset;
  - fixed start, so health starts at 1.0 every episode;
  - reward normalization on;
  - built with `make_vec_env` as the notebooks do.
- **Health shown:** the state of health added to the observation (`settings["soh"] = True`, 11 numbers).
- **Health hidden:** the default 10 numbers, the observation `PPO_trained` saw.
- **Recipe of `PPO_trained`,** read from its checkpoint:

| Setting | Value |
|---|---|
| Algorithm, network | Stable-Baselines3 PPO, `MlpPolicy` defaults: actor and critic with two hidden layers of 64, tanh, orthogonal initialization |
| Environments × steps per update | 4 × 2,048 |
| Minibatch, epochs | 64, 10 |
| γ, GAE λ, clip | 0.99, 0.95, 0.2 |
| Entropy, value-loss weight, gradient clipping | 0.01, 0.5, 0.5 |
| Learning rate | linear from 10⁻³ to 0 |
| Budget | 100 four-year episodes per environment = 14,025,600 steps |

- **5 seeds (0–4) per setting,** 10 runs side by side on one thread each. Each took 2.0–2.4 hours, at about 1,800 steps per second. Checkpoints every 10 episodes. W&B groups `PPO_MG_recipe_seeSoH` and `PPO_MG_recipe_hidden`.
- **Fixed before the results:** the final models are evaluated with the protocol above. The primary measure is health shown minus hidden in mean score with deterministic actions, the training seed as the unit (bootstrap 95% CI and Welch's t-test). Sampled actions are reported too.

**Results** (final models, deterministic actions, score per household per year):

| | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| **Health shown** | −€302 | −€303 | −€306 | −€326 | −€334 | **−€314** |
| **Health hidden** | −€305 | −€325 | −€320 | −€337 | −€319 | **−€321** |

| Shown − hidden | Gap | Bootstrap 95% CI | Welch 95% CI, p |
|---|---:|---|---|
| Deterministic actions (primary) | **+€7.3 (+2.3%)** | −€7.6 to +€21.3 | −€12.2 to +€26.9, p = 0.41 |
| Sampled actions (as in the notebook) | +€2.5 (+0.8%) | −€8.9 to +€14.6 | −€13.9 to +€18.9, p = 0.72 |

| Against "only market" (−€334), deterministic | Gain | Households better |
|---|---|---:|
| Health shown | +€20.2 (+€9.7 to +€30.3) | 89% |
| Health hidden | +€12.9 (+€2.5 to +€22.9) | 75% |

**What it says:**
- **Showing health gives no measurable benefit.** The gap is +€7.3 per household per year, with a confidence interval that includes zero.
- **Seeds matter more than health.** Seeds differ by about €30 within each setting, and the two ranges overlap almost completely. With the seeds averaged, the shown agents score higher in every household. But that is one overall shift spread across all households, not 28 separate pieces of evidence: the unit is the training seed, and there are only 5 per setting.
- **Seeds differ mainly in how much they use the battery.** Some trade more and pay more wear (health shown, seeds 3–4: trading −€14 to −€18, wear −€308 to −€319). Others the reverse (health hidden, seed 1: trading −€97, wear −€228). The totals end up close, with health 0.956–0.970 after the test year.
- **The learner is not the bottleneck.** Both settings beat the best rule.
- **The reproduction works.** The hidden runs average −€321 against the published checkpoint's −€307, within their seed range (−€305 to −€337). The published model is one seed.
- **An interim look agreed:** at 8.4–11.2M steps the two settings averaged −€316 (shown) and −€315 (hidden).

**Why health does not help here:**
- **It is nearly predictable.** It starts at 1.0 every episode and falls smoothly, by only about 3.5% over the test year, so it is close to a function of the time elapsed.
- **Its effects are small.** It shrinks the capacity by about 4% within a year, and the wear price per unit of health is constant.
- **The upper end of the CI,** about +7%, leaves room for a modest benefit, but there is no evidence of one.

**Possible next steps** (not run):
- **Random starting health** (`random_battery_init`), so health differs a lot between episodes, where knowing it should matter most. This changes the environment, and also randomizes the starting charge and temperature.
- **More seeds,** to narrow the CI. The spread between seeds is large.

```bash
/root/venvs/carlos-gym4real/bin/python contextual_environments/microgrid/evaluate_checkpoint.py --output contextual_environments/microgrid/runs/checkpoint_evaluation.json
/root/venvs/carlos-gym4real/bin/python contextual_environments/microgrid/train_ppo.py --seed 0 --show-soh    # health shown
/root/venvs/carlos-gym4real/bin/python contextual_environments/microgrid/train_ppo.py --seed 0               # health hidden
OMP_NUM_THREADS=1 /root/venvs/carlos-gym4real/bin/python contextual_environments/microgrid/evaluate_checkpoint.py --runs <run dirs> --output contextual_environments/microgrid/runs/gate_recipe.json
```

## References

- D. Salaorni, V. De Paola, S. Delpero, G. Dispoto, P. Bonetti, A. Russo, G. Calcagno, F. Trovò, M. Papini, A. M. Metelli, M. Mussi, M. Restelli. *Gym4ReaL: A Suite for Benchmarking Real-World Reinforcement Learning.* arXiv:2507.00257, 2025.
- B. Xu, A. Oudalov, A. Ulbig, G. Andersson, D. S. Kirschen. *Modeling of lithium-ion battery degradation for cell life assessment.* IEEE Transactions on Smart Grid, 2018. (The "Bolun" aging model.)
- Source code: https://github.com/Daveonwave/gym4ReaL
