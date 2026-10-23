# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Contextual PPO (CPPO): SB3 PPO with a recurrent xLSTM context encoder, trained on BatteryPlane in `hidden` observation mode. The agent has to infer its own hidden battery charge `b_t` from the history `[o_t, a_{t-1}]`. The research question is an experiment grid A–F (`GRID_PLAN.md`) comparing point vs. Gaussian context estimates, beta-NLL, variance-as-input, and an auxiliary speed head.

## Branches

`main` holds the hard version, the project page (`docs/`) and the figures (`figures/`). This branch, `easy-env`, also holds the easy version: the take-off and transition options of `battery_plane.py`, the learner's preludes (the encoder reads the take-off before every episode) and its window training (`cppo.supervised.sequence`: whole episodes, plain chunks, R2D2 stored states), their checks and launch options. `EASY_ENV_WARMUP.md` records its plan and results on both branches.

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
python evaluate_final_test.py --reports <report_step0 reports> --conditions 0a 0b 2a_ideal 4a_encoder ... --output runs_step0/final_test_all.json --jobs 16   # final test of step 0-4 runs (seeds 2000-2002): validation reproduced first, then each run's grid with the reserved seeds; final_test.json per run, reused
python -m unittest checks.final_test_checks -v
python figures/collect_data.py && python figures/make_figures.py   # paper figures: numbers from the saved results into figures/data/ (~3 min), plots into figures/out/
python figures/make_figures_person.py && python figures/env_snapshots.py   # the project page's figures into figures/out_person/ (env_snapshots copies its PNGs to docs/assets/figures/); Roboto from font-roboto, see requirements.txt

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
python battery_plane.py --warmup-steps 100 --transition-steps 80   # the easy version's design report: 5 s full-throttle take-off, then 4 s back to the mission's starting speed
python -m unittest checks.easy_env_checks -v         # the easy version: exact hand-over battery and state, nothing about b in the hand-over or in the autopilot's actions
nohup python run_step0.py --conditions 0a 0b 0c 0a_blind 0b_blind 0c_blind --seeds 0 1 2 3 4 5 6 7 --jobs 32 --threads 1 --warmup-steps 100 --transition-steps 80 --output runs_step0/easy_e1_trans80 > run_easy_e1_trans80.log 2>&1 &   # E1 on the easy version (W&B names end in _warm100_trans80)
nohup python run_step0.py --conditions J0 J1 --seeds 0 1 2 3 4 --jobs 10 --threads 3 --warmup-steps 100 --transition-steps 80 --output runs_step0/easy_e3 > run_easy_e3.log 2>&1 &   # E3: Contextual PPO with the GRU encoder trained online, reading the take-off as a prelude
python -m unittest checks.prelude_checks -v          # the learner's preludes: start_episodes, collection, episode buffer, PPO's cached contexts, validation
nohup python run_step0.py --conditions J1_chunk J1_r2d2 --seeds 0 1 2 --jobs 6 --threads 5 --warmup-steps 100 --output runs_step0/easy_e3_windows > run_easy_e3_windows.log 2>&1 &   # J1 with the encoder trained on windows: plain chunks / R2D2 stored states
python -m unittest checks.sequence_checks -v         # what the end-to-end encoder trains on: whole episodes, plain chunks, R2D2 stored-state windows
nohup python run_step0.py --conditions 2a_ideal 2b_ideal 2a_ideal_std 2b_ideal_std --seeds 0 1 2 3 4 5 6 7 --jobs 32 --threads 1 > run_ideal_ppo.log 2>&1 &   # step 2: plain PPO with the ideal estimate, ~3.5 h
python step2_swap_battery.py --runs runs_step0/<launch>/runs.json --output runs_step0/<launch>/swap_true_battery.json  # the ideal-estimate or frozen-encoder policies flown with the true battery instead (several runs.json files)
python -m unittest checks.step1_checks -v
python -m unittest checks.step2_checks -v          # context_wrapper.py: frozen battery estimate appended for plain PPO
python pilots.py --output runs_encoder/pilot_flights/<name> --jobs 30 --policy-reports <report_step0 reports>  # step 3: pilot flights for the offline encoder, ~5 min
python -m unittest checks.step3_checks -v          # pilots.py: no hidden channel, below-cap pilots, replay without leaks
python train_encoder.py --design E1 --seed 0 --output-root runs_encoder/step4   # step 4: A's encoder trained offline on the pilot flights (E3: + b0 head); 6 runs x 2 threads side by side: ~1 h per epoch each (memory-bound), early stopping
python train_encoder.py --design E3 --seed 0 --smoke                             # tiny run, W&B disabled, runs_encoder/step4_smoke/
python train_encoder.py --resume runs_encoder/step4/E1_point_main_s2 --patience 8 --max-epochs 40   # continue a finished run exactly where it stopped, into <name>_resumed (W&B group <group>_resumed)
python train_encoder.py --finalize runs_encoder/step4/<run>   # a run stopped by hand (between epochs): what its natural end does (result.json, model, W&B summary); step4_probes.py --finalize too
python train_encoder.py --design E1 --arch convgru --drop-tau --seed 0 --patience 8 --max-epochs 40 --output-root runs_encoder/step4_grid   # step 4's 2 x 2 plan: --arch gru | convgru (recurrent_encoders.py), saves encoder.pt
python recurrent_encoders.py --export-probe runs_encoder/step4_probes/probe_gru_battery_main_s1 --output <run dir>/encoder.pt   # a probe GRU as an encoder.pt for the context wrapper
python step4_floor.py --output runs_encoder/step4/floor_v2_main --jobs 16   # the floor: ideal estimator on the encoders' validation/test flights, ~4 min
python step4_floor.py --splits train validation test --output runs_encoder/step4/teacher_v2_main --jobs 30   # teacher targets: the ideal estimator on every main-mix flight, ~25 min
python train_encoder.py --design E1 --arch gru --target teacher --teacher runs_encoder/step4/teacher_v2_main --seed 0 --patience 8 --max-epochs 40 --output-root runs_encoder/step4_teacher   # T1: the GRU learns the ideal estimator's mean; + --context-head gaussian: T2, its mean and std
nohup python run_step0.py --conditions J0 J1 J2 J3 J4 J5 --seeds 0 1 2 3 4 --jobs 30 --threads 1 --study v2 --output runs_step0/joint_v2 > run_joint_v2.log 2>&1 &   # joint training with a GRU encoder (config/joint/)
python -m unittest checks.joint_checks -v          # GRURolloutEncoder, re-encoding = collection, the J rungs, curiosity, warm-up
nohup python run_step0.py --conditions M1a_lstm M1b_lstm M2a_win16 M2b_win16 M2a_win64 M2b_win64 --seeds 0 1 2 3 4 5 6 7 --jobs 32 --threads 1 --warmup-steps 100 --output runs_step0/memory_takeoff > run_memory_takeoff.log 2>&1 &   # memory baselines (PAPER_ASSESSMENT.md): sb3-contrib RecurrentPPO (LSTM) and plain PPO on windows of 16 / 64 steps; windows ~1-2 h, RecurrentPPO ~3.5-4 h with the fused LSTM (memory.fused_lstm=false: sb3-contrib's own step-by-step loop, ~15x slower under full load); on the take-off version RecurrentPPO replays the take-off, so run_step0.py scales its budget and validations by 800/700
python -m unittest checks.memory_baseline_checks -v   # memory_baselines.py: the window, RecurrentPPO's input, the take-off replay and its loss mask, the report's schedules
python step4_closed_loop.py --reports <frozen-encoder report_step0 reports> --output runs_step0/closed_loop_frozen_gru.json   # step 4's closed-loop check: a frozen encoder's estimate on its PPO policies' own validation flights, next to the ideal estimate, ~1 min
python step4_probes.py --arch gru --task battery --seed 0     # step 4 diagnostics: tasks count (thrust used so far), find (b0, full-throttle flights), battery; archs xlstm, gru, lstm (GRU ~4 min per epoch); another --data adds its folder to the W&B names
python pilots.py --output runs_encoder/pilot_flights/ft20k --styles full_throttle --flights 20000 --jobs 12   # 20,000 full-throttle flights for the find probe (first 2,000 = v2's), ~1 min
python -m unittest checks.step4_checks -v          # train_encoder.py: split by flight seed, inputs, padding, reproducible, loads in the context wrapper; step4_floor.py; step4_probes.py
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
   - `XLSTMRolloutEncoder` (`torch_layers.py`): an xLSTM block stack (official `xlstm` package) plus a context head (`point`, or `gaussian` → `[mu, sigma2]`) plus an optional auxiliary head. `GRURolloutEncoder` is the same interface with a GRU (`cppo.encoder_arch: gru`, `cppo.gru`; default `xlstm`); its padded-episode path uses one unpacked GRU call (packed sequences are very slow to backpropagate on CPU).
   - Preludes (`prelude_info_keys`, `prelude_label_key`, off by default; the training script sets them for a learned encoder when `env.kwargs.warmup_steps > 0`): at every episode start the encoder first reads the episode's prelude from the reset info (`policy.start_episodes`; BatteryPlane's take-off), and the episode buffer stores it with its labels. Supervised only: PPO then uses the contexts cached at collection (`evaluate_actions_from_contexts`) instead of replaying the encoder, which is exact because the encoder steps come after the PPO epochs. Validation and the final test predict through `PreludePredictor`, since SB3's `evaluate_policy` never shows the reset info.
   - What the supervised encoder steps train on (`cppo.supervised.sequence`, `supervised_sequence`): `episode` (default: whole episodes from their start, full BPTT), `chunk` (plain chunking: `chunk_batch_size` windows of `chunk_len` steps from a blank memory) or `stored_state` (R2D2: windows started from the encoder memory the episode buffer saved at collection, `burn_in` steps without loss or gradient, then `chunk_len` trained steps). The validation fit is always on whole episodes. Conditions J1_chunk and J1_r2d2 (`config/joint/`).
   - Joint-training options of `ContextualPPO`, off by default: `policy_warmup_steps` (no PPO update until then; the encoder trains on the initial policy's flights, `supervised_warmup_gradient_steps` per update) and `curiosity_weight` / `curiosity_until` (a training-reward bonus of weight × the drop in the Gaussian head's entropy, into the rollout buffer's intrinsic rewards, fading linearly to 0). Configs: `config/joint/` (J0–J5).
   - Two buffers. `ContextualRolloutBuffer` (`buffers.py`) holds PPO data as complete per-env streams, so each minibatch is a set of whole streams and `n_minibatches` must divide `n_envs`. These are re-encoded from the stored start-of-rollout state, with optional `bptt_len`. `SupervisedContextualEpisodeBuffer` (`episode_buffers.py`) holds complete episodes of `(o_t, a_{t-1}, label_t)`, with a train/validation split, for the supervised encoder.
   - `encoder_objective`. With `"supervised"` (used by every experiment), the PPO epochs train only the actor and critic while the encoder stays frozen; then `supervised_gradient_steps` encoder steps fit `z_t` to `b_t` (MSE, or beta-NLL for the Gaussian head, plus `weight × auxiliary MSE`). With `"rl"`, the PPO losses are routed to the encoder: critic loss + `encoder_actor_loss_weight × actor loss`.
   - `Contextual_PPO_project_plan.md` is the detailed design doc. Its references to `Env_BatteryPlane/ppo/...` paths and to `c01`–`c04` experiments predate the move into this repo. The current experiments are A–F in `config/experiment/`.

3. **Project glue**:
   - `main_battery_plane_contextual_ppo.py` is the Hydra entry point (`config/cppo.yaml`). It provides `validate_config` (cross-field constraints, enforced before any logging or worker starts), `build_model` (maps `cfg.cppo.*` to `ContextualPPO`/`policy_kwargs`), env factories, the `TrueSpeedInfo` wrapper (adds `vx_ratio` to infos as the velocity-head target), `ScheduledEvalResets` (replays a fixed eval grid), and W&B/TensorBoard logging.
   - `ppo/main_battery_plane_ppo.py` (configs in `ppo/configs/`) is plain SB3 PPO, copied from Env_BatteryPlane: the p01–p04 baselines and step 0's runs 0a/0b (`+experiment=S0a_observed_bs256` / `S0b_observed_bs1024`). It imports `battery_plane.py` and `flight_logging.py` from the repo root and writes to `runs_ppo/<run id>/`, including `config.yaml`.
   - `cppo.reward_scale` / `ppo.reward_scale` multiply the rewards the learner trains on (gymnasium `TransformReward` outside each training env's `Monitor`, so episode returns, evaluation and reports stay in env units). Step 0 uses 0.1 in all three conditions (screened 1–0.001, confirmed against 0.01 on 8 seeds), with gamma 0.99, chosen from discount screens of 0.9–0.999. The step 0 W&B names and groups are built from both settings, e.g. `S0a_observed_bs256_gamma0.99_rs0.01_s0` (earlier discount screens: no suffix for 0.999, `_gamma0995`, `_gamma09`, `_gamma095`); `run_step0.py --reward-scales ...` runs the nine runs once per scale, and `--study confirm1` appends `_confirm1` to the names and groups of follow-up seeds; the base configs keep 1.0 and 0.99, so A–F and p01–p04 are unchanged. `report_step0.py` rejects runs whose PPO settings, reward scale included, differ.
   - `step0_oracle.py` holds `TrueBatteryEncoder` (no parameters, no state, `z_t = b_t` read from the observed-mode observation) and `TrueBatteryPolicy` (the actor and critic read the observation only). `cppo.context_source=oracle` selects them. `NoContextPolicy` (`cppo.context_source=none`, hidden mode only, `config/step0/cppo_blind.yaml`) is the memoryless control: `z_t = 0`, so the actor and critic read the hidden-mode observation only, like plain PPO; runs 0a_blind/0b_blind/0c_blind are 0a/0b/0c in hidden mode, and 0c_blind is the A rerun without its encoder. The default is `encoder`, and `config/step0/` sits outside the A–F grid.
   - `step1_controls.py` holds step 1's controls: `NoisyBattery` (the noisy oracle) adds Gaussian noise of std `env.battery_noise` to the observed-mode battery, with its own random stream, reseeded at seeded resets. Runs 1a/1b/1c (`config/step1/`, `ppo/configs/experiment/S1*`) are 0a/0b/0c with that noise. The late switch (1a_switch/1b_switch/1c_switch) continues the finished step 0 models with noise via `training.init_model` (both training scripts; `switch_source_mismatches` refuses a model trained with other settings). Early noise (1a_early/1b_early/1c_early) trains from scratch with noise until `env.battery_noise_until`, then the true battery (`BatteryNoiseSwitch` callback, DummyVecEnv only). In-flight noise (1a/1b/1c_sharp, 1a/1b/1c_linear) makes the noise of every flight go away at flight step `env.battery_noise_flight_step` (350), at once (`sharp`) or evenly (`linear`), set by `env.battery_noise_flight`. `step1_encoder_error.py` records the A rerun encoder's estimate against the true battery at every step of the validation flights, flown by A itself or by a 0c pilot (the encoder only watches, reading a hidden-mode copy of the env). `step1_ideal_estimator.py` (`IdealBatteryEstimator`) computes the best possible estimate on those flights: the posterior mean of `b_t` over a grid of starting batteries and speeds, each replayed through `BatteryPlaneVec.step` with the executed actions, from hidden-mode observations only.
   - `context_wrapper.py` (step 2 of ENCODER_PLAN.md): `ContextVecEnv` appends a frozen battery estimate to the observations of plain PPO, with one memory per environment, reset at episode starts. `context.source` in the plain-PPO config: `none` (default), `ideal` (the ideal estimator, step-by-step), `encoder` (a frozen learned encoder: a saved Contextual_PPO model's xLSTM from a `.zip`, or a standalone GRU / ConvGRU from an `encoder.pt`) or `true_battery` (a check: reproduces observed mode). Both scripts' validation environments wrap with the same source.
   - `memory_baselines.py` (PAPER_ASSESSMENT.md, Tier 1 item 1): standard memory learners at step 0's settings, chosen by `memory.kind` in the plain-PPO config. `HistoryWindow` (`window`): plain PPO on the last k steps of `[o_t, a_{t-1}, valid]`, filled with the take-off at reset on the easy version. `RecurrentInput` + `MaskedRecurrentPPO` (`recurrent`, `ppo.policy=MlpLstmPolicy`): sb3-contrib's RecurrentPPO reading `[o_t, a_{t-1}]`; on the easy version the take-off is replayed as the first steps of every episode with an autopilot input, and those steps are masked out of every PPO loss (`flight_logging.py` skips their `autopilot` infos). Conditions M1a/M1b_lstm and M2a/M2b_win16/win64 (`ppo/configs/experiment/M*.yaml`); `report_step0.py` scores each in a report of its own.
   - `recurrent_encoders.py` (step 4): standalone battery encoders, `GRUEncoder` (step4_probes.py's GRU, same parameter names) and `ConvGRUEncoder` (a causal 16-frame Conv1d with a valid-frame channel and a per-step MLP before the GRU), with `drop_tau`. `forward` runs whole flights (training), `step` one frame with the memory in a state dict (the context wrapper); they agree within 1e-5. `context_head="gaussian"` adds a variance (`forward_with_variance`, `step_with_variance`; `forward`/`step` still return the mean). `save_encoder`/`load_encoder` use `encoder.pt`. `train_encoder.py --arch gru | convgru` trains them (names like `E1_convgru_notau_point_main_s0`); the default `--arch xlstm` is A's encoder, unchanged.
   - `pilots.py` (step 3 of ENCODER_PLAN.md) flies BatteryPlane in many styles (probe-then-hold, below-cap, random, speed-holding by the noisy speed, endurance, memoryless PPO, good pilots replayed on a shadow plane, style switching with a plan drawn at take-off, and the leaking aware pilots) and records flights as an encoder reads them: hidden observation, previous executed action, and the true battery as label. Scripted pilots read only hidden observations; one .npz per style plus manifest.json (clock-only R² per style and mix) under runs_encoder/pilot_flights/ (gitignored).
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
