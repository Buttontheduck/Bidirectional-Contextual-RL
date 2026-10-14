# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Contextual PPO (CPPO): SB3 PPO with a recurrent xLSTM context encoder, trained on BatteryPlane in `hidden` observation mode. The agent has to infer its own hidden battery charge `b_t` from the history `[o_t, a_{t-1}]`. The research question is an experiment grid A–F (`GRID_PLAN.md`) comparing point vs. Gaussian context estimates, beta-NLL, variance-as-input, and an auxiliary speed head.

## Environment and commands

Use the `carlos` conda env (Python 3.11): `conda activate carlos`. The `python` on the default PATH is base anaconda, so it won't work. `carlos` has no pytest. Training runs on CPU.

```bash
# Single training run (base config = experiment A); Hydra overrides on the command line
python main_battery_plane_contextual_ppo.py +experiment=C_gaussian_nll training.seed=1 wandb.mode=disabled
python main_battery_plane_contextual_ppo.py -m +experiment=A_point_baseline,B_point_speed training.seed=0,1,2

# Full grid: A-D, select C or D on validation, then E-F (see GRID_PLAN.md)
python run_grid.py --dry-run
python run_grid.py --smoke --jobs 2 --threads 1          # tiny correctness check, W&B disabled, runs_cppo/grid_smoke/
python run_grid.py --jobs 4 --threads 4                  # production; --stage screen|followup to split
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python evaluate_final_test.py --grid runs_cppo/grid

# Config/grid checks (tracked in git)
python -m unittest checks.grid_checks -v
python -m unittest checks.grid_checks.GridChecks.test_held_out_seeds_cannot_overlap -v

# Step 0 of ENCODER_PLAN.md: every (condition, seed) as its own process, all at once (16 cores x 2 threads on the 9950X)
nohup python run_step0.py > run_step0.log 2>&1 &     # logs and runs.json in runs_step0/<time>/; --conditions 0a 0b, --dry-run
python run_step0.py --smoke                          # tiny budget, W&B disabled, runs under runs_step0/<time>-smoke/
nohup python run_step0.py --conditions A --threads 4 > run_A.log 2>&1 &   # experiment A rerun (config/rerun/), ~4.5 h
nohup python run_step0.py --conditions 1a 1b 1c --battery-noises 0.03 0.06 0.12 --jobs 27 --threads 1 > run_step1.log 2>&1 &   # step 1 noisy oracle, ~15 min
python step1_fly_with_noise.py --runs runs_step0/<launch>/runs.json --reward-scale 0.1 --output runs_step0/fly_with_noise.json  # step 0 policies flown with noise
nohup python run_step0.py --conditions 1a_switch 1b_switch 1c_switch --battery-noises 0 0.03 0.06 --jobs 27 --threads 1 --from-runs <step 0 runs.json files> > run_switch.log 2>&1 &   # late switch, ~8 min
nohup python run_step0.py --conditions 1a_early 1b_early 1c_early --battery-noises 0.03 0.06 0.12 --jobs 27 --threads 1 > run_early.log 2>&1 &   # early noise, ~15 min
nohup python run_step0.py --conditions 1a_sharp 1b_sharp 1c_sharp 1a_linear 1b_linear 1c_linear --battery-noises 0.06 0.12 0.25 --jobs 27 --threads 1 > run_inflight.log 2>&1 &   # in-flight noise, ~30-35 min
python step1_encoder_error.py --a-runs <A rerun dirs> --pilot-runs <0c dirs, same seeds> --output runs_step0/encoder_error.json  # A encoder's error over a flight, ~30 s
nohup python run_step0.py --conditions 0a_blind 0b_blind 0c_blind --seeds 0 1 2 3 4 5 6 7 --jobs 24 --threads 1 > run_blind.log 2>&1 &   # memoryless controls at step 0's settings, ~15-20 min
python step1_ideal_estimator.py --a-runs <A rerun dirs> --pilot-runs <0c dirs, same seeds> --output runs_step0/ideal_estimator.json  # best possible battery estimate on the same flights, ~3 min
python warmup_information.py                         # EASY_ENV_WARMUP.md: what a forced full-throttle take-off reveals, ~2 min
python -m unittest checks.step1_checks -v
python main_battery_plane_contextual_ppo.py +step0=cppo_oracle training.seed=0       # one 0c run: CPPO, true battery
python ppo/main_battery_plane_ppo.py +experiment=S0a_observed_bs256 training.seed=0   # one 0a run (S0b_observed_bs1024: 0b)
python -m unittest checks.step0_checks -v
python -m unittest checks.logging_checks -v          # flight_logging.py: metric definitions, both learners log the same

# Learner unit tests: plain functions; each file's __main__ runs every test_* in it
python contextual_ppo/test_contextual_ppo.py
python -c "from stable_baselines3.contextual_ppo.test_buffer import test_no_mutable_aliasing as t; t()"

# Environment design rules and value-of-context report for the defaults
python battery_plane.py
```

`.gitignore` excludes `test_*.py`, so the `contextual_ppo/test_*.py` files exist only locally. They import `stable_baselines3.contextual_ppo`, which resolves through a symlink (below). Their optional BatteryPlane checks read `battery_plane.py` from `$BATTERY_PLANE_DIR` (default `../Env_BatteryPlane`).

## Symlinks (load-bearing)

- `~/anaconda3/envs/carlos/lib/python3.11/site-packages/stable_baselines3/contextual_ppo` → this repo's `contextual_ppo/`
- `../Env_BatteryPlane/battery_plane.py` → this repo's `battery_plane.py`
- `../Env_BatteryPlane/flight_logging.py` → this repo's `flight_logging.py` (only for Env_BatteryPlane's own copy of the plain-PPO script; step 0 uses this repo's `ppo/`)

Edits here also change the sibling `Env_BatteryPlane` project. The training script itself does not need the links: it puts the repo root on `sys.path` and imports top-level `contextual_ppo` and `battery_plane`.

## Architecture

There are three layers, and the dependencies point one way only:

1. **`battery_plane.py`**: the environment. `BatteryPlaneVec` is the numpy physics core. `BatteryPlaneEnv` is the gymnasium wrapper (`.core` exposes the Vec core). `iter_eval_resets` expands an evaluation grid (seeds × initial battery × speed noise × y0 × vx0) into reset kwargs. Its module docstring is the spec for the dynamics, reward, observation modes, and reset options. Always build envs with `bp.BatteryPlaneEnv(observation_mode=..., **kwargs)`, never `gym.make("BatteryPlane-v0")`: in `carlos` that id resolves to an older, buggy copy installed inside gymnasium (`gymnasium/envs/classic_control/battery_plane.py`). The info keys `b`, `b0`, `cap` and `saturated` are privileged. They may serve as training labels or diagnostics, but never as policy or encoder inputs.

2. **`contextual_ppo/`**: the environment-agnostic learner. It must never import `battery_plane`. It mirrors SB3's layout:
   - `ContextualPPO` (`contextual_ppo.py`) → `ContextualOnPolicyAlgorithm` (`on_policy_algorithm.py`) → SB3 `OnPolicyAlgorithm`. The collector carries per-env encoder memory and the previous *executed* (clipped) action across steps and rollouts, and resets them at episode starts. It also handles truncation bootstrapping and `terminal_observation` successors.
   - `ContextualActorCriticPolicy` (`policies.py`): a shared encoder produces `z_t`; separate actor and critic MLPs read `[o_t, z_t]`. `parameter_groups()` splits the parameters into `encoder`/`actor`/`value` for gradient routing. `predict()` is recurrent: callers must pass back `state` and `episode_start` (SB3's `evaluate_policy` already does).
   - `XLSTMRolloutEncoder` (`torch_layers.py`): an xLSTM block stack (official `xlstm` package) plus a context head (`point`, or `gaussian` → `[mu, sigma2]`) plus an optional auxiliary head.
   - Two buffers. `ContextualRolloutBuffer` (`buffers.py`) holds PPO data as complete per-env streams, so each minibatch is a set of whole streams and `n_minibatches` must divide `n_envs`. These are re-encoded from the stored start-of-rollout state, with optional `bptt_len`. `SupervisedContextualEpisodeBuffer` (`episode_buffers.py`) holds complete episodes of `(o_t, a_{t-1}, label_t)`, with a train/validation split, for the supervised encoder.
   - `encoder_objective`. With `"supervised"` (used by every experiment), the PPO epochs train only the actor and critic while the encoder stays frozen; then `supervised_gradient_steps` encoder steps fit `z_t` to `b_t` (MSE, or beta-NLL for the Gaussian head, plus `weight × auxiliary MSE`). With `"rl"`, the PPO losses are routed to the encoder: critic loss + `encoder_actor_loss_weight × actor loss`.
   - `Contextual_PPO_project_plan.md` is the detailed design doc. Its references to `Env_BatteryPlane/ppo/...` paths and to `c01`–`c04` experiments predate the move into this repo. The current experiments are A–F in `config/experiment/`.

3. **Project glue**:
   - `main_battery_plane_contextual_ppo.py` is the Hydra entry point (`config/cppo.yaml`). It provides `validate_config` (cross-field constraints, enforced before any logging or worker starts), `build_model` (maps `cfg.cppo.*` to `ContextualPPO`/`policy_kwargs`), env factories, the `TrueSpeedInfo` wrapper (adds `vx_ratio` to infos as the velocity-head target), `ScheduledEvalResets` (replays a fixed eval grid), and W&B/TensorBoard logging.
   - `ppo/main_battery_plane_ppo.py` (configs in `ppo/configs/`) is plain SB3 PPO, copied from Env_BatteryPlane: the p01–p04 baselines and step 0's runs 0a/0b (`+experiment=S0a_observed_bs256` / `S0b_observed_bs1024`). It imports `battery_plane.py` and `flight_logging.py` from the repo root and writes to `runs_ppo/<run id>/`, including `config.yaml`.
   - `cppo.reward_scale` / `ppo.reward_scale` multiply the rewards the learner trains on (gymnasium `TransformReward` outside each training env's `Monitor`, so episode returns, evaluation and reports stay in env units). Step 0 uses 0.1 in all three conditions (screened 1–0.001, confirmed against 0.01 on 8 seeds), with gamma 0.99, chosen from discount screens of 0.9–0.999. The step 0 W&B names and groups are built from both settings, e.g. `S0a_observed_bs256_gamma0.99_rs0.01_s0` (earlier discount screens: no suffix for 0.999, `_gamma0995`, `_gamma09`, `_gamma095`); `run_step0.py --reward-scales ...` runs the nine runs once per scale, and `--study confirm1` appends `_confirm1` to the names and groups of follow-up seeds; the base configs keep 1.0 and 0.99, so A–F and p01–p04 are unchanged. `report_step0.py` rejects runs whose PPO settings, reward scale included, differ.
   - `step0_oracle.py` holds `TrueBatteryEncoder` (no parameters, no state, `z_t = b_t` read from the observed-mode observation) and `TrueBatteryPolicy` (the actor and critic read the observation only). `cppo.context_source=oracle` selects them. `NoContextPolicy` (`cppo.context_source=none`, hidden mode only, `config/step0/cppo_blind.yaml`) is the memoryless control: `z_t = 0`, so the actor and critic read the hidden-mode observation only, like plain PPO; runs 0a_blind/0b_blind/0c_blind are 0a/0b/0c in hidden mode, and 0c_blind is the A rerun without its encoder. The default is `encoder`, and `config/step0/` sits outside the A–F grid.
   - `step1_controls.py` holds step 1's controls: `NoisyBattery` (the noisy oracle) adds Gaussian noise of std `env.battery_noise` to the observed-mode battery, with its own random stream, reseeded at seeded resets. Runs 1a/1b/1c (`config/step1/`, `ppo/configs/experiment/S1*`) are 0a/0b/0c with that noise. The late switch (1a_switch/1b_switch/1c_switch) continues the finished step 0 models with noise via `training.init_model` (both training scripts; `switch_source_mismatches` refuses a model trained with other settings). Early noise (1a_early/1b_early/1c_early) trains from scratch with noise until `env.battery_noise_until`, then the true battery (`BatteryNoiseSwitch` callback, DummyVecEnv only). In-flight noise (1a/1b/1c_sharp, 1a/1b/1c_linear) makes the noise of every flight go away at flight step `env.battery_noise_flight_step` (350), at once (`sharp`) or evenly (`linear`), set by `env.battery_noise_flight`. `step1_encoder_error.py` records the A rerun encoder's estimate against the true battery at every step of the validation flights, flown by A itself or by a 0c pilot (the encoder only watches, reading a hidden-mode copy of the env). `step1_ideal_estimator.py` (`IdealBatteryEstimator`) computes the best possible estimate on those flights: the posterior mean of `b_t` over a grid of starting batteries and speeds, each replayed through `BatteryPlaneVec.step` with the executed actions, from hidden-mode observations only.
   - `flight_logging.py` holds algorithm-agnostic SB3 callbacks that log flight diagnostics from the env infos and executed actions, shared with the plain-PPO script through a symlink:
     - `FlightLoggingEvalCallback`: `validation/*` with per-starting-battery `by_b0/*`, and `episode_plots/*`
     - `RolloutFlightLogger`: `rollout/*`
     - `GradNormLogger`: `train/grad_norm*` and `train/optimizer_steps`. It wraps `torch.nn.utils.clip_grad_norm_` only while the model trains.
     - `FlightRecorder` and `evaluate_flights`: the final validation (`final_validation/*`)
   - Each run writes to `runs_cppo/<wandb run id>/`, or to `paths.run_dir`: the resolved config, metadata, checkpoints, best model, `final_model.zip`, and `final_validation.json`. `wandb/` is at the repo root. All of these are gitignored.

## Experiment protocol invariants

- `config/cppo.yaml` *is* experiment A. Each `config/experiment/*.yaml` (`# @package _global_`) changes only `context_head`, `context_variance_input`, `supervised.nll_beta`, `velocity_head.enabled`, and the W&B naming. `checks/grid_checks.py` asserts this, so changing a shared setting in a single experiment breaks that check on purpose.
- E and F need `selected_gaussian.nll_beta` set to 0.0 (C wins) or 0.5 (D wins); `validate_config` rejects null and any other value. `run_grid.py` picks the winner from the mean final-checkpoint validation return across seeds (a tie goes to C) and writes `selection.json`.
- The seeds are disjoint by construction, and `validate_config` checks it:
  - Training env copy `i` starts training from reset seed `seed+i`, so 0–9 for seeds 0–2. The env factories reset with `seed*100+i`, but SB3 re-seeds the VecEnv when learning starts.
  - Validation uses 1000–1002, and the reserved final test uses 2000–2002.
  - `checks.grid_checks` verifies the real start seeds for both CPPO and plain PPO.
  - Neighbouring training seeds share one starting flight in 7 of the 8 envs (seed `s` env `i+1` matches seed `s+1` env `i`). The streams diverge once those flights end at different steps. This is kept so runs stay comparable with A–F and p04.
  - Only `evaluate_final_test.py` may use the `final_test` grid. Never use it for tuning or selection.
- `run_grid.py` writes a `manifest.json` containing a SHA-256 of the main script, `battery_plane.py`, `run_grid.py`, every config YAML, and the non-test `contextual_ppo/*.py`, plus package versions, seeds, steps, and threads. After any code or config change, existing grid output directories refuse to resume, so use a new `--output`. Completed runs are reused only through their `result.json` pointers.
- Thread limits: `run_grid.py` sets the BLAS env vars for each child, and the training script sets `torch.set_num_threads(training.num_threads)` and interop threads = 1.
