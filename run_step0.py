"""Step 0 of ENCODER_PLAN.md: runs 0a, 0b and 0c in parallel, one training process per condition and seed.

Every (condition, seed) pair is its own process, so the nine runs (3 conditions x seeds 0-2) train at the same time
instead of one after another. A run is mostly single-threaded Python (environment steps and per-step policy calls):
on the 16-core / 32-thread Ryzen 9 9950X, going from 1 to 4 PyTorch threads made a run only ~15% faster, and nine
runs side by side with 2 threads each took ~1.25x the wall time of one run alone (2026-10-12). Hence the defaults:
up to one run per physical core (--jobs) and 2 threads per run (--threads).

    conda activate carlos
    nohup python run_step0.py > run_step0.log 2>&1 &    # all nine runs; keeps running if the SSH session ends
    python run_step0.py --conditions 0a 0b               # only the plain-PPO runs
    python run_step0.py --dry-run                        # print the commands without starting anything
    python run_step0.py --smoke                          # tiny budget, W&B off, everything under runs_step0/
    python run_step0.py --reward-scales 1 0.1 0.001      # the nine runs once per reward scale
    python run_step0.py --seeds 3 4 5 6 7 --study confirm1   # more seeds, W&B names and groups end in _confirm1
    python run_step0.py --conditions A --threads 4       # experiment A (learned encoder) rerun with step 0's settings
    python run_step0.py --conditions 1a 1b 1c --battery-noises 0.03 0.06 0.12   # step 1: noisy oracle
    python run_step0.py --conditions 1a_switch 1b_switch 1c_switch --battery-noises 0 0.03 0.06 \
        --from-runs runs_step0/<step 0 launch>/runs.json ...      # step 1: late switch from the step 0 models
    python run_step0.py --conditions 1a_early 1b_early 1c_early --battery-noises 0.03 0.06 0.12   # step 1: early noise
    python run_step0.py --conditions 1a_sharp 1b_sharp 1c_sharp 1a_linear 1b_linear 1c_linear \
        --battery-noises 0.06 0.12 0.25                     # step 1: in-flight noise (gone by flight step 350)
    python run_step0.py --conditions 2a_ideal 2b_ideal 2a_ideal_std 2b_ideal_std --seeds 0 1 2 3 4 5 6 7 \
        --jobs 32 --threads 1                               # step 2: plain PPO with the ideal battery estimate
    python run_step0.py --conditions 0a_blind 0b_blind 0c_blind --seeds 0 1 2 3 4 5 6 7   # memoryless controls
    python run_step0.py --conditions J0 J1 J2 J3 --seeds 0 1 2 3 4 --jobs 20 --threads 1   # joint training with a GRU encoder (also J4, J5)
    python run_step0.py --conditions 4a_encoder 4b_encoder --encoder-models runs_encoder/step4_grid/E1_gru_point_main_s1/encoder.pt \
        --seeds 0 1 2 3 4 5 6 7 --jobs 32 --threads 1      # step 4: plain PPO with a frozen learned encoder, once per model
    python run_step0.py --conditions M2a_win16 M2b_win16 M2a_win64 M2b_win64 M1a_lstm M1b_lstm --seeds 0 1 2 3 4 5 6 7 \
        --jobs 16 --threads 1                               # memory baselines: windows of 16 / 64 steps, RecurrentPPO (LSTM)

Each launch gets its own folder runs_step0/<time>/ with one log per run (0a_s0.log, ...) and runs.json, which maps
every run to its run directory (runs_ppo/<id>/ or runs_cppo/<id>/). At the end the launcher prints the
report_step0.py command for the finished runs, one per reward scale. The W&B names carry the discount and the reward
scale (e.g. S0a_observed_bs256_gamma0.99_rs0.1_s0), taken from the configs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

from run_grid import child_environment

ROOT = Path(__file__).resolve().parent
# Condition -> training script, its experiment config and its learner's config section
CONDITIONS = {
    "0a": ("ppo/main_battery_plane_ppo.py", "+experiment=S0a_observed_bs256", "ppo"),
    "0b": ("ppo/main_battery_plane_ppo.py", "+experiment=S0b_observed_bs1024", "ppo"),
    "0c": ("main_battery_plane_contextual_ppo.py", "+step0=cppo_oracle", "cppo"),
    # Experiment A (learned xLSTM encoder, hidden mode) with step 0's settings; hours per run, so never a default
    "A": ("main_battery_plane_contextual_ppo.py", "+rerun=A_point_baseline", "cppo"),
    # Step 1, noisy oracle: 0a, 0b and 0c seeing the battery plus noise (env.battery_noise)
    "1a": ("ppo/main_battery_plane_ppo.py", "+experiment=S1a_noisy_bs256", "ppo"),
    "1b": ("ppo/main_battery_plane_ppo.py", "+experiment=S1b_noisy_bs1024", "ppo"),
    "1c": ("main_battery_plane_contextual_ppo.py", "+step1=cppo_noisy_oracle", "cppo"),
    # Step 1, late switch: the finished 0a, 0b and 0c models keep training with noise on the battery (0 = control)
    "1a_switch": ("ppo/main_battery_plane_ppo.py", "+experiment=S1a_late_switch_bs256", "ppo"),
    "1b_switch": ("ppo/main_battery_plane_ppo.py", "+experiment=S1b_late_switch_bs1024", "ppo"),
    "1c_switch": ("main_battery_plane_contextual_ppo.py", "+step1=cppo_late_switch", "cppo"),
    # Step 1, early noise: 0a, 0b and 0c seeing the battery plus noise until step 500k, then the true battery
    "1a_early": ("ppo/main_battery_plane_ppo.py", "+experiment=S1a_early_noise_bs256", "ppo"),
    "1b_early": ("ppo/main_battery_plane_ppo.py", "+experiment=S1b_early_noise_bs1024", "ppo"),
    "1c_early": ("main_battery_plane_contextual_ppo.py", "+step1=cppo_early_noise", "cppo"),
    # Step 1, in-flight noise: noise early in every flight, the exact battery from flight step 350 (sharp or linear)
    "1a_sharp": ("ppo/main_battery_plane_ppo.py", "+experiment=S1a_inflight_noise_bs256", "ppo"),
    "1b_sharp": ("ppo/main_battery_plane_ppo.py", "+experiment=S1b_inflight_noise_bs1024", "ppo"),
    "1c_sharp": ("main_battery_plane_contextual_ppo.py", "+step1=cppo_inflight_noise", "cppo"),
    "1a_linear": ("ppo/main_battery_plane_ppo.py", "+experiment=S1a_inflight_noise_bs256", "ppo"),
    "1b_linear": ("ppo/main_battery_plane_ppo.py", "+experiment=S1b_inflight_noise_bs1024", "ppo"),
    "1c_linear": ("main_battery_plane_contextual_ppo.py", "+step1=cppo_inflight_noise", "cppo"),
    # Memoryless controls: 0a, 0b and 0c in hidden mode, without the battery and without memory (0c_blind: A without
    # its encoder)
    "0a_blind": ("ppo/main_battery_plane_ppo.py", "+experiment=S0a_blind_bs256", "ppo"),
    "0b_blind": ("ppo/main_battery_plane_ppo.py", "+experiment=S0b_blind_bs1024", "ppo"),
    "0c_blind": ("main_battery_plane_contextual_ppo.py", "+step0=cppo_blind", "cppo"),
    # Step 2, ideal context: 0a_blind and 0b_blind with the ideal estimator's battery estimate appended (and its std)
    "2a_ideal": ("ppo/main_battery_plane_ppo.py", "+experiment=S2a_ideal_bs256", "ppo"),
    "2b_ideal": ("ppo/main_battery_plane_ppo.py", "+experiment=S2b_ideal_bs1024", "ppo"),
    "2a_ideal_std": ("ppo/main_battery_plane_ppo.py", "+experiment=S2a_ideal_std_bs256", "ppo"),
    "2b_ideal_std": ("ppo/main_battery_plane_ppo.py", "+experiment=S2b_ideal_std_bs1024", "ppo"),
    "J0": ("main_battery_plane_contextual_ppo.py", "+joint=J0_gru_a_budget", "cppo"),
    "J1": ("main_battery_plane_contextual_ppo.py", "+joint=J1_gru_budget", "cppo"),
    "J2": ("main_battery_plane_contextual_ppo.py", "+joint=J2_gru_budget_entropy", "cppo"),
    "J3": ("main_battery_plane_contextual_ppo.py", "+joint=J3_gru_budget_entropy_gaussian", "cppo"),
    "J4": ("main_battery_plane_contextual_ppo.py", "+joint=J4_gru_budget_entropy_gaussian_curiosity", "cppo"),
    "J5": ("main_battery_plane_contextual_ppo.py", "+joint=J5_gru_budget_warmup", "cppo"),
    "4a_encoder": ("ppo/main_battery_plane_ppo.py", "+experiment=S4a_encoder_bs256", "ppo"),
    "4b_encoder": ("ppo/main_battery_plane_ppo.py", "+experiment=S4b_encoder_bs1024", "ppo"),
    # Memory baselines (PAPER_ASSESSMENT.md, Tier 1 item 1): 0a_blind / 0b_blind with sb3-contrib's RecurrentPPO (an LSTM)
    # or with a window of the last 16 or 64 steps (memory_baselines.py)
    "M1a_lstm": ("ppo/main_battery_plane_ppo.py", "+experiment=M1a_lstm_bs256", "ppo"),
    "M1b_lstm": ("ppo/main_battery_plane_ppo.py", "+experiment=M1b_lstm_bs1024", "ppo"),
    "M2a_win16": ("ppo/main_battery_plane_ppo.py", "+experiment=M2a_win16_bs256", "ppo"),
    "M2b_win16": ("ppo/main_battery_plane_ppo.py", "+experiment=M2b_win16_bs1024", "ppo"),
    "M2a_win64": ("ppo/main_battery_plane_ppo.py", "+experiment=M2a_win64_bs256", "ppo"),
    "M2b_win64": ("ppo/main_battery_plane_ppo.py", "+experiment=M2b_win64_bs1024", "ppo"),
}
# Extra overrides of a condition (the in-flight configs default to the sharp profile)
CONDITION_OVERRIDES = {name: ["env.battery_noise_flight=linear"] for name in ("1a_linear", "1b_linear", "1c_linear")}
STEP0_CONDITIONS = ["0a", "0b", "0c"]
NOISY_CONDITIONS = ["1a", "1b", "1c"]
# Late switch -> the step 0 run whose final model it continues (same seed)
SWITCH_SOURCES = {"1a_switch": "0a", "1b_switch": "0b", "1c_switch": "0c"}
EARLY_NOISE_CONDITIONS = ["1a_early", "1b_early", "1c_early"]
INFLIGHT_CONDITIONS = ["1a_sharp", "1b_sharp", "1c_sharp", "1a_linear", "1b_linear", "1c_linear"]
BLIND_CONDITIONS = ["0a_blind", "0b_blind", "0c_blind"]
# Ideal context -> its report group: the estimate alone, or with its std (report_step0.py never mixes the two)
IDEAL_CONDITIONS = {"2a_ideal": "ideal", "2b_ideal": "ideal", "2a_ideal_std": "ideal_std", "2b_ideal_std": "ideal_std"}
# Frozen learned encoders (step 4): every run once per --encoder-models file, each encoder in a report of its own
ENCODER_CONDITIONS = ["4a_encoder", "4b_encoder"]
# Joint training with a GRU encoder (config/joint/): each rung in a report of its own (their settings differ)
JOINT_CONDITIONS = [name for name in CONDITIONS if name.startswith("J")]
# Memory baselines: each condition in a report of its own (their inputs differ)
MEMORY_CONDITIONS = [name for name in CONDITIONS if name.startswith("M")]
# Installation check: two rollouts and two evaluations per run (keys shared by both training scripts)
SMOKE_OVERRIDES = ["training.total_timesteps=8192", "evaluation.frequency=4096", "evaluation.plots.frequency=8192",
                   "checkpoint.frequency=8192"]


def physical_cores() -> int:
    """Physical cores this process may use; its logical CPUs where the topology can't be read."""
    cpus = os.sched_getaffinity(0)
    cores = set()
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        for name in ("core_cpus_list", "thread_siblings_list"):
            if (topology / name).is_file():
                cores.add((topology / name).read_text().strip())
                break
        else:
            return len(cpus)
    return len(cores)


def launch_name(condition: str, seed: int, reward_scale: float | None = None, battery_noise: float | None = None,
                encoder: Path | None = None) -> str:
    """Name of one run's log: 0a_s0, with the reward scale, battery noise and frozen encoder the launch sets, e.g.
    1c_noise0.06_s0 or 4b_encoder_E1_gru_point_main_s1_s0."""
    parts = [condition]
    if reward_scale is not None:
        parts.append(f"rs{reward_scale:g}")
    if battery_noise is not None:
        parts.append(f"noise{battery_noise:g}")
    if encoder is not None:
        parts.append(encoder_tag(encoder))
    return "_".join([*parts, f"s{seed}"])


def encoder_tag(path: Path) -> str:
    """An encoder file's name in run names: its run directory's name for encoder.pt / encoder_model.zip, else its stem."""
    path = Path(path)
    name = path.parent.name if path.name in ("encoder.pt", "encoder_model.zip", "final_model.zip") else path.stem
    return re.sub(r"[^A-Za-z0-9_.-]", "-", name)


def switch_sources(runs_files: list[Path], reward_scale: float) -> dict[tuple[str, int], Path]:
    """
    Final models a late switch can continue: (step 0 condition, seed) -> final_model.zip of the finished runs in the
    launchers' runs.json files that trained with this reward scale and without battery noise. A condition and seed found
    twice is an error.
    """
    from omegaconf import OmegaConf

    sources: dict[tuple[str, int], Path] = {}
    for path in runs_files:
        for row in json.loads(Path(path).read_text()):
            if row["condition"] not in STEP0_CONDITIONS or row["status"] != "done":
                continue
            run_dir = Path(row["run_dir"])
            cfg = OmegaConf.load(run_dir / "config.yaml")
            learner = cfg.cppo if "cppo" in cfg else cfg.ppo
            if float(learner.get("reward_scale", 1.0)) != reward_scale or float(cfg.env.get("battery_noise", 0.0)) != 0.0:
                continue
            key = (row["condition"], int(cfg.training.seed))
            if key in sources:
                raise ValueError(f"Two step 0 runs for {key}: {sources[key].parent} and {run_dir}")
            sources[key] = run_dir / "final_model.zip"
    return sources


def command_for(condition: str, seed: int, threads: int, output: Path, smoke: bool = False,
                reward_scale: float | None = None, study: str | None = None,
                battery_noise: float | None = None, init_model: Path | None = None, encoder: Path | None = None) -> list[str]:
    script, experiment, learner = CONDITIONS[condition]
    command = [
        sys.executable, str(ROOT / script), experiment, f"training.seed={seed}",
        *CONDITION_OVERRIDES.get(condition, []), f"training.num_threads={threads}", "training.num_interop_threads=1",
        # Hydra's job log; the training scripts never change directory
        f"hydra.run.dir={json.dumps(str(output / 'hydra' / launch_name(condition, seed, reward_scale, battery_noise, encoder)))}",
    ]
    if encoder is not None:
        # Step 4: the frozen encoder, and its name in the W&B name and group
        command += [f"context.encoder_model={json.dumps(str(Path(encoder).resolve()))}", f"context.encoder_tag={encoder_tag(encoder)}"]
    if reward_scale is not None:
        # The configs build the W&B name and group from it
        command.append(f"{learner}.reward_scale={reward_scale:g}")
    if battery_noise is not None:
        # The step 1 configs build the W&B name and group from it
        command.append(f"env.battery_noise={battery_noise:g}")
    if init_model is not None:
        # Late switch: the step 0 model this run continues
        command.append(f"training.init_model={json.dumps(str(init_model))}")
    if study:
        # Appended to the W&B name and group, e.g. ..._rs0.1_confirm1_s3
        command.append(f"wandb.study=_{study}")
    if smoke:
        command += [*SMOKE_OVERRIDES, f"paths.output_dir={json.dumps(str(output / 'runs'))}"]
        if condition in EARLY_NOISE_CONDITIONS:
            # The switch must fall inside the smoke budget: halfway, before the second evaluation
            command.append("env.battery_noise_until=4096")
        if condition == "J4":
            # The curiosity bonus fades out inside the smoke budget
            command.append("cppo.curiosity.until=8192")
        if condition == "J5":
            # The warm-up ends halfway: one update without PPO, then one with it
            command.append("cppo.policy_warmup_steps=4096")
    return command


class Launcher:
    def __init__(self, args):
        self.args = args
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.children: set[subprocess.Popen] = set()
        # (step 0 condition, seed) -> the final model a late switch continues
        self.sources = getattr(args, "sources", {})

    def init_model(self, condition: str, seed: int) -> Path | None:
        return self.sources[(SWITCH_SOURCES[condition], seed)] if condition in SWITCH_SOURCES else None

    def stop(self) -> None:
        self.stopping.set()
        with self.lock:
            children = list(self.children)
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def run_one(self, condition: str, seed: int, reward_scale: float | None = None,
                battery_noise: float | None = None, encoder: Path | None = None) -> dict:
        name = launch_name(condition, seed, reward_scale, battery_noise, encoder)
        identity = {"encoder_model": None if encoder is None else str(Path(encoder).resolve()),
                    "encoder_tag": None if encoder is None else encoder_tag(encoder)}
        log = self.args.output / f"{name}.log"
        environment = child_environment(self.args.threads)
        if self.args.smoke or self.args.wandb_mode is not None:
            environment["WANDB_MODE"] = "disabled" if self.args.smoke else self.args.wandb_mode
        started = time.monotonic()
        with log.open("x") as stream:
            with self.lock:
                if self.stopping.is_set():
                    return {"condition": condition, "seed": seed, "reward_scale": reward_scale,
                            "battery_noise": battery_noise, **identity, "status": "not started", "log": str(log)}
                process = subprocess.Popen(
                    command_for(condition, seed, self.args.threads, self.args.output, self.args.smoke, reward_scale,
                                self.args.study, battery_noise, self.init_model(condition, seed), encoder),
                    cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                )
                self.children.add(process)
            print(f"START {name} | log: {log}", flush=True)
            try:
                code = process.wait()
            finally:
                with self.lock:
                    self.children.discard(process)
        minutes = (time.monotonic() - started) / 60
        # Both training scripts print the final model path last, inside the run directory
        found = re.search(r"Saved final model to (\S+?)/final_model(?:\.zip)?", log.read_text(errors="replace"))
        run_dir = found.group(1) if code == 0 and found else None
        status = "done" if run_dir else "failed"
        print(f"{status.upper()} {name} | {minutes:.1f} min | "
              + (f"run: {run_dir}" if run_dir else f"exit {code}, see {log}"), flush=True)
        return {"condition": condition, "seed": seed, "reward_scale": reward_scale, "battery_noise": battery_noise, **identity,
                "init_model": None if self.init_model(condition, seed) is None else str(self.init_model(condition, seed)),
                "status": status, "exit_code": code, "minutes": round(minutes, 2),
                "log": str(log), "run_dir": run_dir}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=STEP0_CONDITIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--reward-scales", type=float, nargs="+", default=None,
                        help="Run every condition and seed once per reward scale (default: the configs' reward scale)")
    parser.add_argument("--battery-noises", type=float, nargs="+", default=None,
                        help="Step 1 conditions only: run every condition and seed once per noise std (default: the "
                             "configs' 0.06); 0, the control without noise, only for the late switch")
    parser.add_argument("--from-runs", type=Path, nargs="+", default=None,
                        help="Late switch: runs.json files of the step 0 launches whose final models it continues")
    parser.add_argument("--source-reward-scale", type=float, default=0.1,
                        help="Late switch: the reward scale of the step 0 models to continue (default 0.1)")
    parser.add_argument("--encoder-models", type=Path, nargs="+", default=None,
                        help="Step 4's frozen-encoder conditions only: run every condition and seed once per encoder file "
                             "(an encoder.pt of recurrent_encoders.py or a Contextual_PPO .zip)")
    parser.add_argument("--jobs", type=int, default=physical_cores(), help="Runs at the same time (default: physical cores)")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads per run (default: 2)")
    parser.add_argument("--study", default=None,
                        help="Label appended to the W&B names and groups, e.g. confirm1 for more seeds of a screen")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--smoke", action="store_true", help="Tiny budget, W&B disabled, run directories in --output")
    parser.add_argument("--output", type=Path, help="Folder for the logs and runs.json (default: runs_step0/<time>)")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without starting anything")
    args = parser.parse_args(argv)
    if min(args.jobs, args.threads) < 1:
        parser.error("--jobs and --threads must be positive")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Seeds must be distinct non-negative integers and conditions distinct")
    if args.study is not None and not re.fullmatch(r"[A-Za-z0-9]+", args.study):
        parser.error("--study must be letters and digits, e.g. confirm1")
    scales = args.reward_scales
    if scales is not None and (min(scales) <= 0 or len({f"{scale:g}" for scale in scales}) != len(scales)):
        parser.error("Reward scales must be distinct and positive")
    noises = args.battery_noises
    if noises is not None:
        if min(noises) < 0 or len({f"{noise:g}" for noise in noises}) != len(noises):
            parser.error("Battery noises must be distinct and non-negative")
        step1 = NOISY_CONDITIONS + list(SWITCH_SOURCES) + EARLY_NOISE_CONDITIONS + INFLIGHT_CONDITIONS
        if not set(args.conditions) <= set(step1):
            parser.error(f"--battery-noises applies to the step 1 conditions {step1} only")
        if 0 in noises and not set(args.conditions) <= set(SWITCH_SOURCES):
            parser.error("Battery noise 0 is the late switch's control; the noisy oracle and early noise need noise > 0")
    encoding = [condition for condition in args.conditions if condition in ENCODER_CONDITIONS]
    if encoding and not args.encoder_models:
        parser.error(f"{encoding} need --encoder-models: the frozen encoders to train on top of")
    if args.encoder_models:
        if not set(args.conditions) <= set(ENCODER_CONDITIONS):
            parser.error(f"--encoder-models applies to the frozen-encoder conditions {ENCODER_CONDITIONS} only")
        for path in args.encoder_models:
            if not path.is_file() or path.suffix not in (".pt", ".zip"):
                parser.error(f"--encoder-models: {path} is not an encoder file (.pt or .zip)")
        tags = [encoder_tag(path) for path in args.encoder_models]
        if len(set(tags)) != len(tags):
            parser.error(f"--encoder-models: two encoders would share a run name ({tags})")
    switching = [condition for condition in args.conditions if condition in SWITCH_SOURCES]
    args.sources = {}
    if switching:
        if not args.from_runs:
            parser.error("The late switch needs --from-runs: the step 0 launches whose final models it continues")
        try:
            args.sources = switch_sources(args.from_runs, args.source_reward_scale)
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"--from-runs: {error}")
        missing = [(SWITCH_SOURCES[c], seed) for c in switching for seed in args.seeds if (SWITCH_SOURCES[c], seed) not in args.sources]
        if missing:
            parser.error(f"--from-runs has no finished step 0 run (reward scale {args.source_reward_scale:g}, no noise) for {missing}")
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if args.smoke else "")
    args.output = (args.output or ROOT / "runs_step0" / stamp).resolve()
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    # Seeds outer: with fewer job slots than runs, every condition still progresses
    scales, noises, encoders = args.reward_scales or [None], args.battery_noises or [None], args.encoder_models or [None]
    runs = [(condition, seed, scale, noise, encoder) for scale in scales for noise in noises for encoder in encoders
            for seed in args.seeds for condition in args.conditions]
    at_once = min(args.jobs, len(runs))
    described = "" if args.reward_scales is None else f" x reward scales {[f'{scale:g}' for scale in scales]}"
    described += "" if args.battery_noises is None else f" x battery noises {[f'{noise:g}' for noise in noises]}"
    described += "" if args.encoder_models is None else f" x encoders {[encoder_tag(e) for e in encoders]}"
    print(f"{len(runs)} runs ({' '.join(args.conditions)} x seeds {args.seeds}{described}): {at_once} at once x "
          f"{args.threads} threads | {physical_cores()} physical cores, {os.cpu_count()} hardware threads", flush=True)
    if at_once * args.threads > os.cpu_count():
        print(f"Note: {at_once * args.threads} threads exceed the {os.cpu_count()} hardware threads", flush=True)
    if args.dry_run:
        launcher = Launcher(args)
        for condition, seed, scale, noise, encoder in runs:
            print(shlex.join(command_for(condition, seed, args.threads, args.output, args.smoke, scale, args.study, noise,
                                         launcher.init_model(condition, seed), encoder)))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"Logs: {args.output}", flush=True)

    launcher = Launcher(args)

    def terminate(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    pool = ThreadPoolExecutor(max_workers=at_once)
    futures = [pool.submit(launcher.run_one, condition, seed, scale, noise, encoder)
               for condition, seed, scale, noise, encoder in runs]
    try:
        results = [future.result() for future in as_completed(futures)]
    except BaseException:
        launcher.stop()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        signal.signal(signal.SIGTERM, previous)

    results.sort(key=lambda result: (-(result["reward_scale"] or 0), result["battery_noise"] or 0, result["encoder_tag"] or "",
                                     result["condition"], result["seed"]))
    (args.output / "runs.json").write_text(json.dumps(results, indent=2) + "\n")
    done = [result for result in results if result["status"] == "done"]
    failed = len(results) - len(done)
    print(f"{len(done)} of {len(results)} runs finished" + (f", {failed} FAILED" if failed else ""), flush=True)
    if not args.smoke:
        for line in report_commands(done, scales, noises, args.output):
            print(line, flush=True)
    return 1 if failed else 0


def report_commands(done: list[dict], scales: list, noises: list, output: Path) -> list[str]:
    """
    report_step0.py compares runs with the same settings: one command per reward scale, battery noise, in-flight
    profile and context, e.g. "Report (battery noise 0.06): python report_step0.py --run 1a ... --output
    .../step0_report_noise0.06.json".
    """
    def variant(result):
        condition = result["condition"]
        if condition in INFLIGHT_CONDITIONS:
            return condition.split("_")[1]
        if condition in ENCODER_CONDITIONS:
            return f"encoder_{result['encoder_tag']}"
        if condition in JOINT_CONDITIONS:
            return f"joint_{condition}"
        if condition in MEMORY_CONDITIONS:
            return f"memory_{condition}"
        return IDEAL_CONDITIONS.get(condition)

    encoder_variants = sorted({variant(r) for r in done
                               if r["condition"] in ENCODER_CONDITIONS + JOINT_CONDITIONS + MEMORY_CONDITIONS})
    lines = []
    for scale in scales:
        for noise in noises:
            for flight in (None, "sharp", "linear", "ideal", "ideal_std", *encoder_variants):
                group = [r for r in done if r["reward_scale"] == scale and r["battery_noise"] == noise
                         and variant(r) == flight]
                if group:
                    lines.append(report_command(group, scale, noise, flight, output))
    return lines


def report_command(group: list[dict], scale, noise, flight, output: Path) -> str:
    labels, suffix = [], ""
    if scale is not None:
        labels.append(f"reward scale {scale:g}")
        suffix += f"_rs{scale:g}"
    if noise is not None:
        labels.append(f"battery noise {noise:g}")
        suffix += f"_noise{noise:g}"
    if flight is not None:
        labels.append({"ideal": "ideal context", "ideal_std": "ideal context with its std"}.get(
            flight, f"frozen encoder {flight[len('encoder_'):]}" if flight.startswith("encoder_")
            else f"joint training {flight[len('joint_'):]}" if flight.startswith("joint_")
            else f"memory baseline {flight[len('memory_'):]}" if flight.startswith("memory_") else f"{flight} in-flight noise"))
        suffix += f"_{flight}"
    label = f" ({', '.join(labels)})" if labels else ""
    report = " ".join(f"--run {r['condition']} {r['run_dir']}/config.yaml {r['run_dir']}" for r in group)
    return f"Report{label}: python report_step0.py {report} --output {output / f'step0_report{suffix}.json'}"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Stopped: the running training processes were terminated.", file=sys.stderr)
        sys.exit(130)
