# Contextual PPO experiment grid

| Test | Context head | Beta | Actor and critic inputs | Speed loss weight |
|---|---|---:|---|---:|
| A | Point, MSE | — | Observation + mean | Off |
| B | Point, MSE | — | Observation + mean | 0.1 |
| C | Gaussian | 0 | Observation + mean + log variance | Off |
| D | Gaussian | 0.5 | Observation + mean + log variance | Off |
| E | Selected Gaussian | C or D | Observation + mean | Off |
| F | Selected Gaussian | C or D | Observation + mean + log variance | 0.1 |

Three seeds (0, 1, 2) per configuration give 18 runs. E still learns its variance
through the Gaussian loss; neither actor nor critic receives that variance.
All runs train fresh encoders and policies. The mean-only comparison controls the
loss and inputs, but closed-loop policies can collect different training data.

The current repository settings are preserved: 1M environment transitions per run,
8 DummyVecEnv environments, 512 rollout steps, 10 PPO epochs, and the 128-wide
mLSTM + sLSTM encoder. This is 18 runs, each rounded up to a complete PPO rollout.
Use `--steps 2000000` if the intended budget is 2M. Do not mix budgets in one grid.

## Run on the Ryzen 9950X

From the repository root in the active `carlos` environment:

```bash
python run_grid.py --dry-run
python run_grid.py --smoke --jobs 2 --threads 1
python run_grid.py --jobs 4 --threads 4
```

The production command runs A-D (12 runs), selects C or D, then runs E-F (6 runs).
Selection uses the mean final-checkpoint validation return across the three
training seeds. Higher wins; an exact tie chooses C. It selects one beta for all
E/F seeds. `selection.json` records the scores and rule. Intermediate best models
are saved for inspection, but are not used by this selection or final-test protocol.

The 9950X has 16 physical cores and 32 logical CPUs. Four jobs with four CPU threads
each is an initial benchmark setting. The launcher also limits BLAS thread pools
and sets PyTorch inter-op threads to one. Eight environments within each job are
still executed by DummyVecEnv; they are not eight additional worker processes.
Do not start 12 runs with four threads each as the default.

There is no measured guarantee of 12-16 hours. Compare this setup with eight jobs
and two threads in a separate output directory using representative episode lengths.
The tiny smoke test checks correctness, not production runtime. It uses shortened
episodes, small networks, 32 transitions, and disabled W&B, in `runs_cppo/grid_smoke/`.

Keep the production launcher in a persistent terminal such as tmux so that an SSH
disconnect does not end it. It prints each job's log path as it starts. W&B retains
your existing configuration; use `--wandb-mode offline` if desired.

## Stages, reuse and files

```bash
python run_grid.py --stage screen --jobs 4 --threads 4
python run_grid.py --stage followup --jobs 4 --threads 4
```

Both commands use `runs_cppo/grid/` unless `--output` is supplied. Repeating the same
command reuses completed runs with intact checkpoints and validation results.
Source code, package versions, seeds, budget, smoke settings and thread count must
match the saved manifest; otherwise choose a new output directory. Concurrency can
change when resuming. Failures stop the queue and active children; unsuccessful
attempts are retained and restart from scratch on the next launch. Checkpoint
continuation and automatic import of older runs are not implemented.

Each attempt stores its resolved config, metadata, checkpoints, validation curve,
final model and `final_validation.json`. Each seed/configuration has a `result.json`
pointer to the successful attempt. `grid_results.json` is written after all six
configurations finish. CPU runtime and actual transition count are recorded.

## Reserved final test

Validation uses seeds 1000-1002. Final testing reserves seeds 2000-2002 with the same
battery/noise grid. The training script rejects overlap between the validation,
final-test and initial training seeds. The encoder episode-buffer validation split
is a separate diagnostic; it does not select C versus D in this protocol.

After completing all training and selection, evaluate frozen final checkpoints:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python evaluate_final_test.py --grid runs_cppo/grid
```

This writes per-run `final_test.json` files and `final_test_results.json` for the
whole grid. Existing test results are reused. Do not use these scores to tune the
configurations or choose beta; if you do, they cease to be an untouched final test.

The training entry point imports the repository's `contextual_ppo/` directly, as in
the pasted server version. There is no need to link that directory into SB3.

```bash
python -m unittest checks.grid_checks -v
```

## Results: first grid (run 2026-10-08)

All 18 runs finished on the server, at **2M transitions** per run (`--steps 2000000`, actual 2,002,944). All share one source hash (`6c78f5e9…`). The data comes from W&B `duck13/cppo`, groups `cppo_<A–F>_…`.

These runs were produced by a version of the training script that logs `final_validation/*` metrics; the committed `main_battery_plane_contextual_ppo.py` logs `validation/final_mean_reward` instead.

All numbers below are for the deterministic final checkpoint on the validation grid: seeds 1000–1002 × starting battery 0.2–0.7, speed noise 5. That is 18 episodes per run. Final-test results are written to JSON on the server and are not included here.

Every episode ends in exactly one of three ways:
- **completed:** reached `T`;
- **ground crash:** altitude ≤ 0;
- **battery empty:** charge ≤ 0.

The environment never truncates, so the three rates sum to 100%.

### Selection

| | s0 | s1 | s2 | Mean |
|---|---:|---:|---:|---:|
| C (β = 0) | 100.2 | 126.4 | 36.3 | **87.6** |
| D (β = 0.5) | −189.4 | 14.9 | 8.2 | −55.4 |

C was selected, so E and F ran with `selected_gaussian.nll_beta=0.0` (groups `…_beta0.0`). C also wins on the median (100.2 vs 8.2), so D's single crashed seed doesn't decide the choice.

### Overall (3 seeds × 18 episodes = 54 episodes per experiment)

| Exp | Return s0 / s1 / s2 | Mean return | Completed | Ground crash | Battery empty |
|---|---|---:|---:|---:|---:|
| A point | −56.6 / −14.6 / −14.7 | −28.6 | 11% | 6% | 83% |
| B point + speed | 36.1 / −14.7 / −14.2 | 2.4 | 22% | 0% | 78% |
| **C Gaussian β=0** | 100.2 / 126.4 / 36.3 | **87.6** | **33%** | 0% | 67% |
| D Gaussian β=0.5 | −189.4 / 14.9 / 8.2 | −55.4 | 11% | 33% | 56% |
| E mean only | 45.7 / −14.7 / −14.6 | 5.5 | 22% | 0% | 78% |
| F Gaussian + speed | −150.2 / −13.2 / −1.0 | −54.8 | 11% | 22% | 67% |

### By starting battery (3 seeds × 3 episodes = 9 episodes per cell)

Mean return:

| Exp | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| A | −213.9 | −152.4 | −88.1 | −17.4 | 32.7 | 267.3 |
| B | −197.3 | −139.8 | −84.2 | −30.8 | 122.9 | 343.5 |
| C | −194.2 | −121.4 | −39.5 | 141.3 | 299.5 | 440.0 |
| D | −191.8 | −146.9 | −103.7 | −62.5 | −23.3 | 195.6 |
| E | −196.4 | −137.8 | −81.1 | −25.6 | 126.1 | 347.6 |
| F | −210.5 | −138.1 | −97.8 | −58.8 | −20.9 | 197.4 |

Completed:

| Exp | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| A | 0% | 0% | 0% | 0% | 0% | 67% |
| B | 0% | 0% | 0% | 0% | 33% | 100% |
| C | 0% | 0% | 0% | 33% | 67% | 100% |
| D | 0% | 0% | 0% | 0% | 0% | 67% |
| E | 0% | 0% | 0% | 0% | 33% | 100% |
| F | 0% | 0% | 0% | 0% | 0% | 67% |

Ground crash:

| Exp | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| A¹ | 0–22% | 0–22% | 0–22% | 0–22% | 0–22% | 11–33% |
| B | 0% | 0% | 0% | 0% | 0% | 0% |
| C | 0% | 0% | 0% | 0% | 0% | 0% |
| D | 33% | 33% | 33% | 33% | 33% | 33% |
| E | 0% | 0% | 0% | 0% | 0% | 0% |
| F² | 0–22% | 0–33% | 0–33% | 11–33% | 0–33% | 11–33% |

Battery empty:

| Exp | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|
| A¹ | 78–100% | 78–100% | 78–100% | 78–100% | 78–100% | 0–22% |
| B | 100% | 100% | 100% | 100% | 67% | 0% |
| C | 100% | 100% | 100% | 67% | 33% | 0% |
| D | 67% | 67% | 67% | 67% | 67% | 0% |
| E | 100% | 100% | 100% | 100% | 67% | 0% |
| F² | 78–100% | 67–100% | 67–100% | 67–89% | 67–100% | 0–22% |

W&B logs ground-crash and battery-empty rates for each run as a whole only; per starting battery it logs just return and completion. Each run either has no ground crashes or crashes in every episode, except A_s0 and F_s0, so the split by battery level is exact everywhere else. For those two runs the episode plot (`episode_plots/final_validation`) shows one episode at 0.2, 0.5 and 0.7, so their cells are ranges:

1. A_s0 crashed into the ground in 3 of its 18 episodes, at least one of them at 0.7. Its other 15 episodes ended with an empty battery.
2. F_s0 had 12 ground crashes and 6 empty batteries. The plot shows an empty battery at 0.2 and ground crashes at 0.5 and 0.7.

### Per run

| Run | Return | Completed | Ground | Battery | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_s0 | −56.6 | 0% | 17% | 83% | −246.9 | −177.4 | −95.9 | 9.6 | 57.2 | 114.0 |
| A_s1 | −14.6 | 17% | 0% | 83% | −197.3 | −139.8 | −84.3 | −30.9 | 20.6 | 344.0 |
| A_s2 | −14.7 | 17% | 0% | 83% | −197.3 | −139.9 | −84.3 | −31.0 | 20.5 | 343.9 |
| B_s0 | 36.1 | 33% | 0% | 67% | −197.3 | −139.9 | −84.3 | −31.0 | 327.1 | 342.0 |
| B_s1 | −14.7 | 17% | 0% | 83% | −197.3 | −139.9 | −84.3 | −31.0 | 20.5 | 343.9 |
| B_s2 | −14.2 | 17% | 0% | 83% | −197.2 | −139.6 | −83.9 | −30.5 | 21.1 | 344.6 |
| C_s0 | 100.2 | 50% | 0% | 50% | −192.5 | −122.9 | −38.2 | 312.1 | 318.6 | 324.0 |
| C_s1 | 126.4 | 17% | 0% | 83% | −193.8 | −103.9 | −0.4 | 128.6 | 259.5 | 668.3 |
| C_s2 | 36.3 | 33% | 0% | 67% | −196.2 | −137.4 | −79.8 | −16.9 | 320.5 | 327.8 |
| D_s0 | −189.4 | 0% | 100% | 0% | −190.0 | −189.4 | −189.2 | −189.2 | −189.2 | −189.2 |
| D_s1 | 14.9 | 17% | 0% | 83% | −192.1 | −124.1 | −58.1 | 5.1 | 65.1 | 393.5 |
| D_s2 | 8.2 | 17% | 0% | 83% | −193.2 | −127.3 | −63.9 | −3.4 | 54.2 | 382.6 |
| E_s0 | 45.7 | 33% | 0% | 67% | −194.5 | −133.5 | −74.8 | −14.6 | 336.6 | 355.2 |
| E_s1 | −14.7 | 17% | 0% | 83% | −197.3 | −139.9 | −84.3 | −31.0 | 20.6 | 343.8 |
| E_s2 | −14.6 | 17% | 0% | 83% | −197.3 | −139.9 | −84.3 | −31.0 | 21.2 | 343.8 |
| F_s0 | −150.2 | 0% | 67% | 33% | −240.8 | −143.8 | −138.5 | −132.3 | −125.8 | −119.9 |
| F_s1 | −13.2 | 17% | 0% | 83% | −196.8 | −138.9 | −83.0 | −29.4 | 22.6 | 346.1 |
| F_s2 | −1.0 | 17% | 0% | 83% | −194.1 | −131.7 | −71.8 | −14.7 | 40.4 | 366.1 |

### Reference on the same grid

| Policy | Return | Completed | Ground crash | Battery empty |
|---|---:|---:|---:|---:|
| Vanilla PPO, `full` (battery visible, exact speed) | 338.7 | 78% | 22% | 0% |
| Vanilla PPO, `observed` (battery visible, noisy speed) | 212.2 | 83% | 6% | 11% |
| Vanilla PPO, `hidden` (no memory) | 52.7 | 22% | 0% | 78% |
| Vanilla PPO, `clean` (battery hidden, exact speed) | −14.4 | 17% | 0% | 83% |

These are the final checkpoints from `Env_BatteryPlane/runs_ppo` (2M steps, 3 seeds), evaluated locally on 2026-10-12. Observed PPO peaked at about 360 around 600k steps, then fell to 212 by 2M.

The scripted policies in `python battery_plane.py` start from a fixed position (y0 = 100, vx0 = 15), so they are only a rough guide: the oracle averages 444, the best battery-blind rule 278, climb then full throttle 153, and cruise-only 132.

### Observations

- **No run beats cruise-only (132).** The best run is C_s1 at 126.4.
- **No run finishes an episode that starts at battery 0.2–0.4.** All of those episodes end with an empty battery, apart from the ground crashes in A_s0, D_s0 and F_s0.
- **10 of 18 runs show the same outcome pattern.** These are A_s1, A_s2, B_s1, B_s2, D_s1, D_s2, E_s1, E_s2, F_s1 and F_s2. They complete only from 0.7, spend no time in the reward band, and keep the cruise motor at its cap about 100% of the time.
  - Seven of them (A_s1, A_s2, B_s1, B_s2, E_s1, E_s2, F_s1) have near-identical per-battery returns: −197.3, −139.9, −84.3, −31.0, ≈ 21, ≈ 344.
  - PPO in `clean` mode reproduces those returns exactly. Its flights hold both actions pinned at the bounds: cruise motor +1, lift motor −1. The plane never climbs, sinks from its 64–85 m start, and tops out at 41–48 m/s, below the 52 m/s band.
  - D_s1, D_s2 and F_s2 are slightly faster variants of the same pattern.
- **Three different behaviours:**
  - C_s0, C_s2, E_s0 and B_s0 finish from 0.5 or 0.6 upward.
  - C_s1 alone climbs and sprints: 668 at 0.7, 43% of the time in the band.
  - D_s0 (every episode a ground crash) and F_s0 (67% ground crashes) collapsed.
- **Three seeds can't separate the experiments.** C's seeds are the top 3 when ranked against A's (one-sided permutation p = 0.05). C against E gives p ≈ 0.10. The speed head shows no consistent effect.
- **Encoder R² overstates the battery estimate.** A clock-only estimate already gets pooled R² 0.45 on the collapsed runs' flights. Those runs' encoders reach 0.83–0.87 in closed loop, about 70–77% of the way from the clock to a perfect estimate. C_s0, C_s2, D_s0 and F_s0 have negative closed-loop R². The follow-up experiment is in `ENCODER_PLAN.md`.
