# myoFatiElbowPose1D6MRandom-v0: elbow posing with muscle fatigue

A MyoSuite task: a simulated human arm must bend its elbow to a randomly chosen angle and hold it there, using six muscles. In the `myoFati` version every muscle also tires as it works and recovers as it rests, and that fatigue is never shown to the agent. This document describes the environment exactly as installed here, read from its source code and checked by running it.

**Verdict (22 October 2026): this task is not a useful test of autogenic context.** Plain PPO does as well with the fatigue hidden as with it shown, so there is nothing for a context encoder to win. A redesigned version (a new target every 2 s for a minute, with a 5 kg weight in the hand) failed a value-of-context gate too: seeing the fatigue is worth at most 4% against the best memoryless policy, and only about 1.4% comes from acting on it. The evidence is in [Experiments](#experiments) and the reasons are in [Why it is not a useful test of autogenic context](#why-it-is-not-a-useful-test-of-autogenic-context).

- **Version:** MyoSuite 3.0.0 with MuJoCo 3.16.0, installed in the virtual environment `/root/venvs/carlos-myo`. That environment is built on the `carlos` conda environment and sees all its packages; `carlos` itself is unchanged.
- **What it is made of:** the task `myoElbowPose1D6MRandom-v0` (class `PoseEnvV0`, file `myosuite/envs/myo/tasks/basic/arm/pose.py`), the arm model `myoelbow_1dof6muscles.xml`, and the `FatigueWrapper` (`myosuite/envs/wrappers.py`), which adds the fatigue model `CumulativeFatigue` (`myosuite/core/muscle_conditions.py`).
- **Create it:**

```python
from myosuite.utils import gym          # run with /root/venvs/carlos-myo/bin/python
env = gym.make("myoFatiElbowPose1D6MRandom-v0")
obs, info = env.reset(seed=0)
```

Leave `MUJOCO_GL` unset on the server: training needs no rendering, and `MUJOCO_GL=egl` fails because the server has no EGL graphics library.

## The arm

| | |
|---|---|
| Joint | one hinge, the elbow (`r_elbow_flex`), from 0 to 2.269 rad (0° to 130°) |
| Muscles | six Hill-type muscles, each pulling through a tendon |
| Simulation step | 0.002 s; the agent acts every 10 steps, so one control step is **0.02 s (50 per second)** |
| Episode | **100 control steps = 2 seconds** |

| Muscle | Role | Peak force |
|---|---|---:|
| TRIlong (triceps, long head) | straightens the elbow (extensor) | 770 N |
| TRIlat (triceps, lateral head) | extensor | 604 N |
| TRImed (triceps, medial head) | extensor | 604 N |
| BIClong (biceps, long head) | bends the elbow (flexor) | 608 N |
| BICshort (biceps, short head) | flexor | 424 N |
| BRA (brachialis) | flexor | 976 N |

Each muscle's activation follows its command with MuJoCo's activation dynamics: 10 ms to activate, 40 ms to relax.

## State space

The full state of the simulation, most of which the agent does not see:

| Part | Size | Seen by the agent? |
|---|---:|---|
| Elbow angle and angular speed | 2 | yes |
| Muscle activations (MuJoCo's activation state) | 6 | yes |
| Target angle | 1 | indirectly, through the angle error |
| Fatigue: for each muscle the active, resting and fatigued shares (MA, MR, MF, summing to 1) | 3 × 6 | **no** |

The fatigue is the hidden context.

## Observation space

9 numbers, not normalized:

| Entry | Size | Meaning |
|---|---:|---|
| `qpos` | 1 | elbow angle (rad) |
| `qvel` | 1 | angular speed × 0.02 s, i.e. the change of angle per control step |
| `pose_err` | 1 | target angle − elbow angle (rad); with `qpos` it gives the target |
| `act` | 6 | the muscles' current activation in [0, 1]. Under fatigue this equals the active share MA, so it cannot exceed what the tired muscle still has |

## Action space

6 numbers in [−1, 1], one per muscle. Each goes through a fixed sigmoid, excitation = 1 / (1 + e^(−5 (a − 0.5))), before it reaches the muscle:

| Action a | −1 | −0.42 | 0 | 0.5 | 1 |
|---|---:|---:|---:|---:|---:|
| Excitation | 0.0006 | 0.01 (the "rest" threshold) | 0.076 | 0.5 | **0.924** |

So the strongest possible command is 0.924, not 1. The fatigue model then decides how much of that excitation the muscle can deliver (below), and the result drives MuJoCo's muscle.

## Reward and objective

**Objective:** bring the elbow to the target angle quickly and hold it there for the rest of the 2 seconds, without wasting muscle effort. The task counts as solved while the elbow is within 0.175 rad (10°) of the target.

**Reward per control step**, with d = |target angle − elbow angle| in radians:

reward = 1 × (−d) + 4 × bonus + 1 × (−‖act‖ / 6) + 50 × penalty

| Term | Value | Range here |
|---|---|---|
| pose | −d | −2.27 to 0 |
| bonus | 1 if d < 0.175 rad (10°), plus 1 more if d < 0.2625 rad (15°) | 0, 1 or 2, weighted × 4: up to 8 |
| act_reg | −(length of the 6 activations) / 6 | −0.38 to 0 (activations reach at most 0.924) |
| penalty | −1 if d > 2π | never, the joint can't get that far |

- **Best possible:** about 8 per step (on target with almost no effort), so about 800 per episode.
- **A random policy** averages about 4.2 per step, around 180 per episode (measured), because random targets and starting angles are often close to each other.
- Also reported in `info`: `pose`, `bonus`, `penalty`, `act_reg`, `sparse` (= pose), `solved`.

**Episode end:** after 100 steps (time limit, reported as truncated). It can also terminate early if d > 2π (impossible here) or if the simulation becomes numerically unstable (`MjInstabilityTerminationWrapper`).

## Reset

At every reset:
1. **Target angle:** uniform in [0, 2.27] rad.
2. **Starting elbow angle:** uniform over the joint range [0, 2.269] rad, independent of the target; speed 0; muscle activations 0.
3. **Fatigue:** by default every muscle starts fresh: MA = 0, MR = 1, MF = 0. Two options of the `FatigueWrapper` change this:
   - `fatigue_reset_random=True`: each muscle draws its own random state. Two numbers nf, ap are drawn uniformly in [0, 1]; MA = nf · ap, MR = nf · (1 − ap), MF = 1 − nf. The fatigued share MF is then uniform in [0, 1], independently for every muscle (measured over 2,000 resets: mean 0.49–0.51 per muscle, minimum 0.0, maximum 1.0).
   - `fatigue_reset_vec=[...]`: a given fatigued share MF for every muscle (with MR = 1 − MF and MA = 0).

   Switch them on the wrapper itself; the outer wrapper forwards method calls but not attribute assignments:

```python
fatigue = env
while type(fatigue).__name__ != "FatigueWrapper":
    fatigue = fatigue.env
fatigue.set_fatigue_reset_random(True)          # or: fatigue.fatigue_reset_vec = numpy.full(6, 0.8)
state = fatigue.muscle_fatigue                  # .MA, .MR, .MF: the hidden fatigue, one value per muscle
```

## The fatigue model (3CC-r)

The three-compartment controller with rest recovery (Xia & Frey Law 2008; Looft et al. 2018; Rakshit et al. 2021). Every muscle's motor units are split into three shares that always add up to 1:

| Share | Meaning |
|---|---|
| MA, active | the units that are working now; this is what the muscle actually delivers |
| MR, resting | fresh units, available to be recruited |
| MF, fatigued | tired units, unavailable until they recover |

The command after the sigmoid is the target load TL. Every control step (dt = 0.02 s):

- dMA/dt = C − F · MA
- dMR/dt = −C + rR · MF
- dMF/dt = F · MA − rR · MF

In words:
- **C moves units between resting and active** so that MA follows the command TL, with MuJoCo-like activation and relaxation rates. It can only recruit units that are resting: once MR runs out, MA cannot rise further. A muscle can therefore deliver at most 1 − MF.
- **Working tires units:** active units become fatigued at rate F.
- **Rest recovers them:** fatigued units return to resting at rate R, sped up r times while the muscle rests (command TL ≤ 0.01, i.e. action a ≤ −0.42).
- **The muscle receives MA,** the fatigue-limited activation, in place of the command.

Per-muscle rates, by functional muscle group (MyoSuite's table `MUSCLE_FATIGUE_PARAMS`):

| Group | Muscles | F, fatigue (per s) | R, recovery (per s) | r, rest multiplier | Recovery at rest r · R (per s) |
|---|---|---:|---:|---:|---:|
| Elbow extensors | TRIlong, TRIlat, TRImed | 0.01559 | 0.00125 | 25.52 | 0.032 (time constant about 31 s) |
| Elbow flexors | BIClong, BICshort, BRA | 0.01703 | 0.00494 | 4.68 | 0.023 (about 43 s) |

**How fast it changes (measured).** Starting fresh, a full command for the whole 2-second episode fatigues only 2.8% (extensors) and 3.1% (flexors) of each muscle. Tiring a muscle substantially takes about a minute of hard work, and recovering takes tens of seconds of rest. The fatigue model alone, at the strongest command (excitation 0.924), then at rest:

| | 2 s | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|---:|
| Fatigued share MF at full command, extensors / flexors | 0.03 / 0.03 | 0.14 / 0.15 | 0.37 / 0.37 | 0.59 / 0.57 | 0.80 / 0.72 |
| Then at rest (after the 120 s), extensors / flexors | | 0.58 / 0.57 | 0.31 / 0.36 | 0.12 / 0.18 | |

**When the fatigue shows (measured).** Hold a constant command for 0.5 s:

| Starting fatigued share MF | Command a (excitation) | Observed activation | Capacity left 1 − MF |
|---:|---|---:|---:|
| 0.0 | 1.0 (0.924) | 0.924 | 0.993 |
| 0.8 | 1.0 (0.924) | **0.199** | 0.199 |
| 0.0 | 0.3 (0.269) | 0.269 | 0.998 |
| 0.8 | 0.3 (0.269) | **0.199** | 0.199 |

Below its remaining capacity a tired muscle follows the command exactly as a fresh one does, and the fatigue is invisible. Only a command above the capacity reveals it: the observed activation stops at 1 − MF.

## As a test of autogenic context

| | BatteryPlane | myoFatiElbowPose1D6MRandom |
|---|---|---|
| Hidden context | the battery charge (1 number) | the fatigued share of each muscle (6 numbers, plus their active and resting shares) |
| Who changes it | only the agent's own thrust | only the agent's own muscle effort, plus recovery at rest |
| What it limits | the cruise motor's maximum thrust | each muscle's maximum activation (1 − MF) |
| When it shows | only while the motor runs at its limit | only while a command exceeds the muscle's remaining capacity |
| Recovers? | no | yes, at rest |
| Change within one episode | large (it can drain completely) | tiny with these settings (at most about 3% in 2 s) |
| Reset | random starting charge | **fresh muscles by default**; random with `fatigue_reset_random=True` |

The structure is the same as BatteryPlane's: the agent's effort creates the hidden context, the context caps what the agent can do, and it shows only when the agent pushes against the cap. The time scales differ, though:

- **Default settings:** every episode starts fresh and fatigue barely builds up in 2 seconds, so there is almost no context to infer. An agent can ignore fatigue and lose almost nothing.
- **With `fatigue_reset_random=True`:** every episode starts with a hidden, random fatigue in each muscle, which the agent can only discover by commanding above the remaining capacity. Within the 2 seconds its own effort changes that fatigue very little, so the context is mostly given at reset: closer to a hidden fixed context than to BatteryPlane's, which the agent drains as it flies.
- **To make the agent's own effort matter within an episode** (the autogenic part), the episodes would have to be much longer, the fatigue rates F faster, or the fatigue kept from one episode to the next.

The experiments below test the default settings, random starting fatigue and 60 s episodes.

## Experiments

### Setup shared by all runs

All runs use [train_ppo.py](train_ppo.py) in this folder: plain Stable-Baselines3 PPO with a memoryless policy (it sees only the current observation). The script imports nothing from the rest of the repository.

| PPO setting | Value |
|---|---|
| Environments × steps per update | 8 × 512 = 4,096 steps |
| Minibatch, epochs | 256, 10 |
| Learning rate, discount γ, GAE λ | 3 × 10⁻⁴, 0.99, 0.95 |
| Clip range, entropy, value-loss weight, gradient clipping | 0.2, 0, 0.5, 0.5 |
| Actor and critic | separate, two hidden layers of 256, tanh, orthogonal initialization |
| Reward scale | × 0.1 for training only (`rs0.1` in the run names); every return reported here is in the environment's own units |
| Budget | 2 million steps per run, 3 seeds (0, 1, 2), one CPU thread per run |

These are the settings of BatteryPlane's plain-PPO baseline 0a, unchanged. The reward scale was chosen for BatteryPlane and not tuned here.

**Seeds.**
- Training environment i of seed s starts from reset seed s + i, so seeds 0–2 use reset seeds 0–9.
- Validation uses reset seeds 1000–1029.
- Seeds 2000–2029 are reserved for a final test and have not been used.
- A reset seed fixes the target angle, the starting angle and, when it is random, the starting fatigue. [elbow_checks.py](elbow_checks.py) checks this, that training and validation draws differ, that the hidden observation contains no fatigue, and that the episode lengths are right.

**Validation.** Every 50,000 steps (every 100,000 for 60 s episodes), the current policy flies 30 episodes from reset seeds 1000–1029 with deterministic actions. The tables show the final model. Measures:
- **Return:** the sum of rewards over the episode.
- **Episodes ending on target:** the share of episodes whose last step is within 10° of the target (the task's "solved").
- **Steps on target:** the share of all steps within 10° of the target. The arm starts away from the target and needs a few steps to get there, so a perfect policy scores a little under 100% in 2 s episodes.
- **Final error:** the angle error at the last step, in degrees.

```bash
/root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0      # leave MUJOCO_GL unset
/root/venvs/carlos-myo/bin/python contextual_environments/myo/elbow_checks.py -v
```

Runs, logs and models are in `runs/` (ignored by Git). Learning curves are in the W&B project `contextual-myo`.

### 1. Plain PPO without fatigue

**Question:** does plain PPO learn the elbow task at all? This is the task without the fatigue wrapper, `myoElbowPose1D6MRandom-v0`, with 2 s episodes. W&B group `PPO_ElbowPose1D6MRandom_rs0.1`.

Validation over training, as the mean of the 3 seeds:

| Training steps | Return | Episodes ending on target | Steps on target | Final error |
|---:|---:|---:|---:|---:|
| 0 (untrained) | 181.0 | 23% | 22.9% | 29.3° |
| 50,000 | 742.3 | 100% | 93.2% | 1.9° |
| 100,000 | 768.2 | 100% | 97.1% | 1.4° |
| 500,000 | 769.2 | 100% | 97.2% | 3.0° |
| 1,000,000 | 771.2 | 100% | 97.3% | 1.6° |
| 2,000,000 | **772.3 ± 0.4** | **100%** | **97.2%** | **0.9°** |

Final policies, per seed:

| Seed | Return | Episodes ending on target | Steps on target | Final error |
|---|---:|---:|---:|---:|
| 0 | 772.8 | 100% | 97.2% | 0.7° |
| 1 | 771.8 | 100% | 97.2% | 1.3° |
| 2 | 772.3 | 100% | 97.3% | 0.8° |

Each run took 8 minutes.

**Answer: yes, easily.**
- Every seed ends all 30 validation episodes on target at the first check, 50,000 steps (2.5% of the budget).
- From 100,000 steps on, every check ends 100% on target, and the return creeps up only slightly (768 → 772).
- About 772 out of roughly 800 is close to the ceiling, because the arm needs a few steps to swing over at the start.
- The seeds barely differ (771.8–772.8).

Reaching and holding an angle is easy for a memoryless policy. Any failure on the fatigue versions below comes from the fatigue, not from the reaching itself.

### 2. Plain PPO with fatigue, and with 60 s episodes

**Questions:**
- Does hidden fatigue stop plain PPO?
- Does it matter whether the muscles start fresh or randomly tired?
- Does a longer episode, in which the agent's own effort can build up fatigue, change the answer?
- Does hiding the fatigue cost anything, compared with showing it?

**Settings.** All are 3 seeds × 2 million steps, with the shared setup above.

| Setting | Task | Starting fatigue | Episode | Fatigue in the observation | W&B group |
|---|---|---|---|---|---|
| No fatigue, 2 s (section 1) | `myoElbowPose1D6MRandom-v0` | none | 2 s | — | `PPO_ElbowPose1D6MRandom_rs0.1` |
| Fresh, 2 s | `myoFatiElbowPose1D6MRandom-v0` | fresh (MF = 0) | 2 s | hidden | `PPO_FatiElbowPose1D6MRandom_fresh_rs0.1` |
| Random, 2 s | same | random (MF uniform in [0, 1] per muscle) | 2 s | hidden | `..._randomMF_rs0.1` |
| Random, 2 s, shown | same | random | 2 s | **shown** (control) | `..._randomMF_seeMF_rs0.1` |
| No fatigue, 60 s | `myoElbowPose1D6MRandom-v0` | none | 60 s | — | `PPO_ElbowPose1D6MRandom_T3000_rs0.1` |
| Fresh, 60 s | `myoFatiElbowPose1D6MRandom-v0` | fresh | 60 s | hidden | `..._fresh_T3000_rs0.1` |
| Random, 60 s | same | random | 60 s | hidden | `..._randomMF_T3000_rs0.1` |
| Random, 60 s, shown | same | random | 60 s | **shown** (control) | `..._randomMF_seeMF_T3000_rs0.1` |

- **Why 60 s (3,000 steps):** that is the fatigue model's own time scale (tiring takes about a minute, recovery at rest 30–45 s), and 2 million steps still give about 670 episodes. The target stays the same for the whole episode, so the arm reaches it and then holds it.
- **The two controls:**
  - *No fatigue, 60 s* separates the effect of the longer episode from the effect of fatigue.
  - *Shown* appends the six fatigued shares MF to the observation, like BatteryPlane's observed mode. It shows how much knowing the fatigue is worth to the same memoryless learner.
- **A reference without training:** the three no-fatigue policies of section 1, flown unchanged on each setting ([evaluate_ppo.py](evaluate_ppo.py)). This is a policy that has never felt fatigue.

Each 2 s run took 14 minutes and each 60 s run 17–19 minutes, with 21 runs side by side.

**Results**, final validation, mean ± standard deviation over the 3 seeds:

| Setting | Return | Per step | Episodes ending on target | Steps on target | Final error |
|---|---:|---:|---:|---:|---:|
| No fatigue, 2 s | 772.3 ± 0.4 | 7.72 | 100% | 97.2% | 0.9° |
| Fresh, 2 s | 771.1 ± 0.1 | 7.71 | 100% | 97.2% | 1.2° |
| Random, 2 s | 687.1 ± 6.0 | 6.87 | 91.1% | 85.1% | 3.6° |
| Random, 2 s, shown | 685.6 ± 1.4 | 6.86 | 90.0% | 84.3% | 4.5° |
| *Random, 2 s, no-fatigue policy* | *668.4 ± 4.2* | *6.68* | *88.9%* | *82.0%* | *3.6°* |
| No fatigue, 60 s | 23,868 ± 6 | 7.96 | 100% | 99.9% | 0.5° |
| Fresh, 60 s | 23,841 ± 9 | 7.95 | 100% | 99.9% | 1.0° |
| Random, 60 s | 21,000 ± 248 | 7.00 | 85.6% | 84.1% | 5.0° |
| Random, 60 s, shown | 20,441 ± 200 | 6.81 | 85.6% | 82.1% | 5.3° |
| *Random, 60 s, no-fatigue policy* | *21,633 ± 323* | *7.21* | *86.7%* | *88.2%* | *3.0°* |

The no-fatigue policy also loses nothing with fresh muscles: 770.8 in 2 s and 23,846 in 60 s. Every run reached its final level early: within 2% of its final return by 100,000 steps (no fatigue and fresh) or by 150,000–450,000 steps (random).

**What the results say:**
1. **Fresh muscles: fatigue makes no difference**, even over 60 s. Holding a pose takes little effort, so the muscles end the minute only 3% fatigued on average (the most-used muscle reaches about 50%), and the arm stays on target.
2. **Random starting fatigue costs about 11%.** Return per step falls from 7.7 to 6.9 in 2 s episodes, and from 8.0 to 7.0 in 60 s episodes.
3. **Showing the fatigue does not help.** In 2 s episodes the hidden and shown policies are equal (687 vs 686). In 60 s episodes the shown policies are lower (20,441 vs 21,000): they start worse (75.7% vs 82.6% of steps on target in the first 10 s) and catch up over the minute.
4. **A policy that never felt fatigue does almost as well.** It scores 668 in 2 s episodes, and in 60 s episodes even a little more than the trained ones (21,633). The 60 s policies saw 30 times fewer episodes, and so 30 times fewer targets, in the same budget.
5. **In 60 s episodes the muscles recover while the arm holds.** The average fatigued share falls from 0.51 at the start to 0.25 at the end.

**Does hiding the fatigue cost anything?** The training seed is the unit, and the gap is shown minus hidden:

| Episodes | Hidden | Shown | Shown − hidden | 95% CI, bootstrap | 95% CI, Welch | Welch p |
|---|---:|---:|---:|---|---|---:|
| 2 s | 687.1 | 685.6 | −1.5 (−0.2%) | −1.4% to +0.7% | −2.7% to +2.3% | 0.77 |
| 60 s | 21,000 | 20,441 | −559 (−2.7%) | −4.3% to −1.0% | −5.7% to +0.4% | 0.07 |

- With 3 seeds per setting the bootstrap interval (20,000 resamples) is coarse, so the wider Welch interval is the safer bound.
- **In 2 s episodes, hiding the fatigue costs at most about 2% of the return.**
- **In 60 s episodes it costs nothing:** the shown policies are the lower ones.
- For comparison, hiding the battery in BatteryPlane costs the same plain PPO 91% of its validation return (23 vs 246 at minibatch 256, README).
- A larger test (8 seeds, 1,000 evaluation episodes) was started and then cancelled: it could only have narrowed these bounds, not changed the answer. Its partial runs are in `runs/cancelled/`.

**Where the failures come from: mostly impossible episodes.** [reachability.py](reachability.py) drives the tired arm as hard as it goes, each way, in each of the 30 validation episodes: all flexors at the strongest command with the extensors at rest, then the reverse. If the target lies more than 10° beyond the angles reached, no policy can end on target.

- **2 of the 30 episodes are impossible** (seeds 1013 and 1015). The tired flexors cannot lift the forearm against gravity: target 129° with at most 118° reachable, and target 123° with at most 85°.
- Every policy fails both of them, and they account for most of the failures:

| Policy | Ends on target, all 30 episodes | Of the 28 reachable | Of the 2 impossible |
|---|---:|---:|---:|
| Random, 2 s | 91.1% | 97.6% | 0% |
| Random, 2 s, shown | 90.0% | 96.4% | 0% |
| No-fatigue policy, 2 s | 88.9% | 95.2% | 0% |
| Random, 60 s | 85.6% | 91.7% | 0% |
| Random, 60 s, shown | 85.6% | 91.7% | 0% |
| No-fatigue policy, 60 s | 86.7% | 92.9% | 0% |

The remaining failures are nearly the same episodes for every policy:

| Seed | Target | Start | Reachable | Flexor strength left | Random 2 s | Shown 2 s | No-fatigue 2 s | Random 60 s | Shown 60 s | No-fatigue 60 s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1013 | 129° | 97° | 0–118°, impossible | 40% | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 1015 | 123° | 93° | 0–85°, impossible | 11% | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 1024 | 128° | 59° | 0–120° | 42% | 2/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 1026 | 122° | 13° | 0–116° | 37% | 2/3 | 3/3 | 2/3 | 0/3 | 1/3 | 0/3 |
| 1004 | 5° | 64° | 14–127° | 65% (extensors 8%) | 3/3 | 3/3 | 3/3 | 2/3 | 2/3 | 3/3 |
| 1020 | 4° | 115° | 0–122° | 44% | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |

"Flexor strength left" is the flexors' remaining capacity 1 − MF, weighted by each muscle's peak force. The table counts the seeds that end on target.

- **1024 and 1026 are borderline:** the target lies just beyond the highest angle the tired flexors reach, within the 10° margin, so ending there needs the flexors close to their limit. In 2 s that sometimes works. In 60 s no policy lasts the minute: holding near the limit tires the flexors further (in 1024 the long biceps head goes from 11% to 59% fatigued), and the arm sinks.
- **Only 1004 shows a real difference between policies:** a nearly straight target with almost exhausted extensors. The policies trained on 60 s episodes reach it only after 30–40 s, whereas the no-fatigue policy gets there.

```bash
F=myoFatiElbowPose1D6MRandom-v0
/root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0 --env-id $F --fatigue-reset random                       # random, 2 s
/root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0 --env-id $F --fatigue-reset random --observe-fatigue     # shown
/root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0 --env-id $F --episode-steps 3000 --eval-every 100000     # fresh, 60 s
/root/venvs/carlos-myo/bin/python contextual_environments/myo/evaluate_ppo.py --models <section 1 runs>/final_model.zip --env-id $F --fatigue-reset random --output runs/reference/nofat_policy_on_fati_random_T100.json
/root/venvs/carlos-myo/bin/python contextual_environments/myo/reachability.py --runs <run dirs> --references <evaluate_ppo outputs> --output runs/reference/reachability.json
```

### 3. A redesigned task: repeated, demanding targets (gate experiment)

**Why.** Feedback on sections 1–2 suggested repeated, demanding targets within one episode, with some choice about which muscles to use or when to rest. Faster fatigue alone could mainly create more impossible episodes. It also suggested a check before any large training budget: do two different fatigue states, under otherwise similar observed conditions, favour different actions? If one feedback strategy works across fatigue states, Contextual PPO has little to win, even with perfect inference.

**The gate, fixed in advance.** Memoryless PPO with the fatigue shown is an upper bound for any learner that has to infer the fatigue, Contextual PPO included. So the gap shown − hidden is the most perfect inference could win.
- **Pass:** the shown runs beat the hidden ones by at least 10% of the hidden return, and the 95% bootstrap CI of the gap lies above 0.
- **Otherwise:** pass on the task.

**First finding: the arm needs a load.** The elbow model is light, so even high targets take little effort. Held angle at a given flexor excitation, with the extensors at rest:

| Flexor excitation | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 0.924 (full) |
|---|---:|---:|---:|---:|---:|---:|---:|
| No load | 85° | 107° | 114° | 119° | 123° | 127° | 130° |
| 5 kg in the hand | 19° | 35° | 81° | 98° | 105° | 113° | 119° |

Without a load, the no-fatigue policy flew 60 s of targets between 90° and 130°, a new one every 2 s. Its flexors ended only 18% fatigued on average (41% at most), and it stayed on target 99% of the time. Repeated targets alone do not make fatigue matter.

**The task.**
- MyoSuite's `myoFatiElbowPose1D6MRandom-v0`, with its fatigue model and rates unchanged.
- 60 s episodes, with **a new target every 2 s** (30 per episode), uniform in 0°–110°.
- **A 5 kg weight in the hand,** acting at the wrist at every control step.
- A fresh arm reaches every target: at most 119° at full effort. Holding takes about 30% of the flexors' strength at 81°, 40% at 98°, 50% at 105° and 70% at 113°.
- Low targets let the muscles rest (the arm hangs on the joint stop), so there is a real choice about when to spend strength. The target sequence is drawn from a generator seeded at each reset, so a reset seed fixes it.
- **Two variants:** fresh muscles at the start (fatigue only from the agent's own effort), and MyoSuite's random starting fatigue.

**Training.** Each variant has fatigue hidden or shown, 3 seeds, 4 million steps. γ = 0.999, so the agent looks about 20 s ahead, across several targets. The reward scale is 0.01, which keeps the critic's targets the same size as in sections 1–2: 8 / (1 − 0.999) × 0.01 = 80. Validation runs every 500,000 steps. Each run took 22 minutes, with 12 side by side. Options: `--target-every 100 --target-range 0 110 --load-kg 5 --gamma 0.999 --reward-scale 0.01`. W&B groups: `PPO_FatiElbowPose1D6MRandom_{fresh,randomMF}[_seeMF]_T3000_rep100_t0-110_load5_g0.999_rs0.01`.

**The gate evaluation** ([fatigue_gate.py](fatigue_gate.py)):
- Every final model flies the same 100 episodes, from reset seeds 1000–1099.
- Each seeing policy also flies them with a decoy fatigue:
  - **another episode's fatigue:** episode k is shown the fatigue the same policy had in episode k + 1 at the same step, a real fatigue trajectory from a different history;
  - **fresh muscles:** MF = 0 throughout.
- If a seeing policy loses little with a decoy, its actions do not depend on the fatigue.

**Results, final models** (100 episodes, mean ± standard deviation over 3 seeds):

| Variant | Hidden | Shown | Shown − hidden | 95% CI, bootstrap | 95% CI, Welch | Gate |
|---|---:|---:|---:|---|---|---|
| Fresh start | 22,893 ± 42 | 22,924 ± 28 | +0.1% | −0.1% to +0.4% | −0.3% to +0.6% (p = 0.45) | fails |
| Random starting fatigue | 19,259 ± 437 | 21,759 ± 146 | **+13.0%** | +10.5% to +15.8% | +6.8% to +19.1% (p = 0.009) | passes |

| Seeing policies flown with | Fresh start: loss | Random start: loss (per seed) |
|---|---:|---:|
| Another episode's fatigue | 0.0% | **1.4%** (137, 575, 234) |
| Fresh muscles shown | 0.1% | 3.8% (142, 2,540, −194) |

**Fresh start: no gap.** Even with the weight, the agent's own effort leaves the muscles only 9% fatigued on average at the end of the minute (49% for the most-used muscle), because the low targets let them recover. Hidden and shown play the same, and the decoys change nothing.

**Random start: the 13% is mostly the hidden learner getting worse.**
- The seeing policies barely depend on what fatigue they are shown: another episode's fatigue costs them only 1.4%. So the 13% cannot be the value of acting on the fatigue.
- The learning curves (validation, 30 episodes) show where it comes from. Every hidden seed peaks at the first check and then declines, while its end-of-episode fatigue rises: the memoryless policies learn to push harder and tire themselves. The seeing policies improve and let the muscles recover during the minute (end fatigue 0.32 against 0.50 hidden, from 0.51 at the start).

| Validation return | 0.5M | 1M | 2M | 3M | 4M |
|---|---:|---:|---:|---:|---:|
| Hidden, mean of 3 seeds | 20,671 | 20,145 | 19,556 | 19,197 | 18,769 |
| Shown, mean of 3 seeds | 19,784 | 20,789 | 21,125 | 21,409 | 21,586 |

**The deciding check: shown against the best hidden policy.**
- The hidden runs were retrained for 1 million steps, saving a checkpoint every 100,000 steps (`--save-checkpoints`). They reproduce the 4M runs exactly up to 500,000 steps.
- Each seed's best checkpoint by validation (seeds 1000–1029) is at 500k, 800k and 400k steps (validation 21,032, 20,986, 20,813).
- Those checkpoints and the seeing policies then flew 100 episodes not used for the selection, reset seeds 1030–1129 (`fatigue_gate.py --hidden-model best --first-seed 1030`):

| Random starting fatigue | Return | Steps on target |
|---|---:|---:|
| Hidden, best checkpoint | 20,880 ± 139 | 88.5% |
| Shown, final model | 21,714 ± 158 | 91.9% |
| Shown, flown with another episode's fatigue | 21,418 | |
| **Shown − best hidden** | **+834 (+4.0%)**, bootstrap CI +2.8% to +5.1%, Welch CI +2.0% to +6.0% (p = 0.005) | |

**The gate fails for both variants, so we pass on this task.**
- **Against the best memoryless policy,** seeing the fatigue is worth 4%: real (p = 0.005) but small. Seeing the battery in BatteryPlane is worth 91%.
- **Most of even that 4% does not come from acting on the fatigue.** Fed another episode's fatigue, the seeing policies still score 21,418, above the best hidden policy (20,880). Only about 1.4 of the 4 points come from knowing the right fatigue when choosing actions. The rest is a training effect: seeing the fatigue while learning helps PPO, probably because the critic predicts returns better, and keeps the policy from degrading.
- **On the feedback's question:** different fatigue states hardly call for different actions here, and one feedback strategy works across them. Even perfect inference could win Contextual PPO at most about 4%, of which about 1–2% would come from the inference itself.

```bash
F=myoFatiElbowPose1D6MRandom-v0; G="--env-id $F --episode-steps 3000 --target-every 100 --target-range 0 110 --load-kg 5 --gamma 0.999 --reward-scale 0.01"
/root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0 $G --fatigue-reset random --steps 4000000 --eval-every 500000                     # hidden
/root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0 $G --fatigue-reset random --observe-fatigue --steps 4000000 --eval-every 500000   # shown
/root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0 $G --fatigue-reset random --steps 1000000 --eval-every 100000 --save-checkpoints --suffix _ckpt1M
/root/venvs/carlos-myo/bin/python contextual_environments/myo/fatigue_gate.py --hidden <hidden runs> --shown <shown runs> --output runs/gate/gate_randomMF.json
/root/venvs/carlos-myo/bin/python contextual_environments/myo/fatigue_gate.py --hidden <ckpt1M runs> --shown <shown runs> --hidden-model best --first-seed 1030 --output runs/gate/gate_randomMF_besthidden.json
```

## Why it is not a useful test of autogenic context

A test of autogenic context needs four things. BatteryPlane was built to have all of them:
1. **The agent's own actions create the hidden context** during the episode. The thrust drains the battery.
2. **The context changes what the best action is.** Whether to climb into the reward band depends on the battery.
3. **The current observation does not reveal it,** so it has to be inferred from the history. The battery shows only while the motor runs at its limit.
4. **As a result, an agent that does not know the context loses clearly** against one that does. Hiding the battery costs plain PPO 91% of its return.

myoFatiElbowPose1D6MRandom fails the first two, and so the fourth.

- **Its own effort hardly creates any context.**
  - With MyoSuite's default fresh start, holding a pose takes so little effort that after a full minute the muscles are only 3% fatigued on average.
  - PPO with fresh fatigue scores the same as PPO without fatigue: 771.1 vs 772.3 in 2 s, and 23,841 vs 23,868 in 60 s.
  - With random starting fatigue there is a context, but it is handed out at reset rather than made by the agent. During the episode it mostly fades by recovery (from 0.51 to 0.25 over 60 s), not by the agent's work.
- **The context barely changes the best action.**
  - The policy controls one angle and sees its error. Tired muscles make the arm respond more weakly; the error stays, and the policy pushes harder, which draws on whatever strength is left. Six muscles share one joint, so when one muscle is tired the others in its group carry the same command.
  - The best action is therefore set by the error, which the policy sees, and not by the fatigue.
  - A policy trained without any fatigue already keeps most of the score (668 vs 687 in 2 s).
- **Where fatigue does decide the outcome, knowing it does not help.**
  - The failures are physically impossible episodes (2 of 30), or borderline ones at the tired flexors' limit that no policy can hold for a minute.
  - Knowing the fatigue does not make an impossible target reachable.
- **So nothing is lost by not knowing.**
  - Hiding the fatigue costs at most about 2% of the return in 2 s episodes, and nothing in 60 s episodes.
  - There is no gap for a context encoder to close. A Contextual PPO result on this task could not show whether the encoder infers the context, because a memoryless agent already does as well.

**The deeper reason:** nothing in the task asks for a decision before the fatigue shows. BatteryPlane's agent must choose whether to climb before the battery reveals itself, and a wrong guess costs reward later. The elbow task has one fixed target per episode and no reason to save strength for later, so the right move is always the same: reduce the error you can see.

**Making the task more demanding does not fix this (section 3).** A new target every 2 s for a minute, with a 5 kg weight in the hand, gives the agent a reason to save strength. Still:
- with fresh muscles at the start, its own effort leaves them only 9% fatigued by the end of the minute, and seeing the fatigue changes nothing;
- with random starting fatigue, seeing it is worth 4% against the best memoryless policy, and the seeing policies lose only 1.4% when shown another episode's fatigue.

The fatigue-aware policies mostly differ in how they learn, not in what they do.

**What else could make it useful.** Each of these changes the task further. Only the first was tried (section 3), and it failed the gate:
- **Several demanding targets per episode,** so that strength spent now is missing later. Tried with a 5 kg load.
- **Fatigue carried over from one episode to the next,** so that the agent's past effort becomes the hidden context, as in BatteryPlane.
- **Faster fatigue rates F,** so that the context changes within seconds rather than minutes.
- **A cost or a risk that depends on fatigue and is paid before the fatigue shows,** like BatteryPlane's climb, which pays off only with a well-charged battery.

**Scope.** These conclusions hold for the tasks as tested:
- MyoSuite's standard reward and fatigue rates;
- 2 s or 60 s episodes, with one target or a new one every 2 s;
- no load, or 5 kg in the hand;
- fresh or random starting fatigue;
- plain PPO with γ 0.99 or 0.999 and 3 seeds per setting.

## References

- V. Caggiano, H. Wang, G. Durandau, M. Sartori, V. Kumar. *MyoSuite: A contact-rich simulation suite for musculoskeletal motor control.* L4DC 2022. https://sites.google.com/view/myosuite
- T. Xia, L. A. Frey Law. *A theoretical approach for modeling peripheral muscle fatigue and recovery.* Journal of Biomechanics, 2008.
- J. M. Looft, N. Herkert, L. Frey-Law. *Modification of a three-compartment muscle fatigue model to predict peak torque decline during intermittent tasks.* Journal of Biomechanics, 2018.
- R. Rakshit, Y. Xiang, J. Yang. *Functional muscle group- and sex-specific parameters for a three-compartment controller muscle fatigue model applied to isometric contractions.* Journal of Biomechanics, 2021.
- L. A. Frey-Law, J. M. Looft, J. Heitsman. *A three-compartment muscle fatigue model accurately predicts joint-specific maximum endurance times for sustained isometric tasks.* Journal of Biomechanics, 2012.
