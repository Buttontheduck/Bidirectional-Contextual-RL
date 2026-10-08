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
