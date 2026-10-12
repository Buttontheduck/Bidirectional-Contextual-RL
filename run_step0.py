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

Each launch gets its own folder runs_step0/<time>/ with one log per run (0a_s0.log, ...) and runs.json, which maps
every run to its run directory (runs_ppo/<id>/ or runs_cppo/<id>/). At the end the launcher prints the
report_step0.py command for the finished runs.
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
# Condition -> training script and its experiment config
CONDITIONS = {
    "0a": ("ppo/main_battery_plane_ppo.py", "+experiment=S0a_observed_bs256"),
    "0b": ("ppo/main_battery_plane_ppo.py", "+experiment=S0b_observed_bs1024"),
    "0c": ("main_battery_plane_contextual_ppo.py", "+step0=cppo_oracle"),
}
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


def command_for(condition: str, seed: int, threads: int, output: Path, smoke: bool = False) -> list[str]:
    script, experiment = CONDITIONS[condition]
    command = [
        sys.executable, str(ROOT / script), experiment, f"training.seed={seed}",
        f"training.num_threads={threads}", "training.num_interop_threads=1",
        # Hydra's job log; the training scripts never change directory
        f"hydra.run.dir={json.dumps(str(output / 'hydra' / f'{condition}_s{seed}'))}",
    ]
    if smoke:
        command += [*SMOKE_OVERRIDES, f"paths.output_dir={json.dumps(str(output / 'runs'))}"]
    return command


class Launcher:
    def __init__(self, args):
        self.args = args
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.children: set[subprocess.Popen] = set()

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

    def run_one(self, condition: str, seed: int) -> dict:
        name = f"{condition}_s{seed}"
        log = self.args.output / f"{name}.log"
        environment = child_environment(self.args.threads)
        if self.args.smoke or self.args.wandb_mode is not None:
            environment["WANDB_MODE"] = "disabled" if self.args.smoke else self.args.wandb_mode
        started = time.monotonic()
        with log.open("x") as stream:
            with self.lock:
                if self.stopping.is_set():
                    return {"condition": condition, "seed": seed, "status": "not started", "log": str(log)}
                process = subprocess.Popen(
                    command_for(condition, seed, self.args.threads, self.args.output, self.args.smoke),
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
        return {"condition": condition, "seed": seed, "status": status, "exit_code": code, "minutes": round(minutes, 2),
                "log": str(log), "run_dir": run_dir}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--jobs", type=int, default=physical_cores(), help="Runs at the same time (default: physical cores)")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads per run (default: 2)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--smoke", action="store_true", help="Tiny budget, W&B disabled, run directories in --output")
    parser.add_argument("--output", type=Path, help="Folder for the logs and runs.json (default: runs_step0/<time>)")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without starting anything")
    args = parser.parse_args(argv)
    if min(args.jobs, args.threads) < 1:
        parser.error("--jobs and --threads must be positive")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Seeds must be distinct non-negative integers and conditions distinct")
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if args.smoke else "")
    args.output = (args.output or ROOT / "runs_step0" / stamp).resolve()
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    # Seeds outer: with fewer job slots than runs, every condition still progresses
    runs = [(condition, seed) for seed in args.seeds for condition in args.conditions]
    at_once = min(args.jobs, len(runs))
    print(f"{len(runs)} runs ({' '.join(args.conditions)} x seeds {args.seeds}): {at_once} at once x {args.threads} "
          f"threads | {physical_cores()} physical cores, {os.cpu_count()} hardware threads", flush=True)
    if at_once * args.threads > os.cpu_count():
        print(f"Note: {at_once * args.threads} threads exceed the {os.cpu_count()} hardware threads", flush=True)
    if args.dry_run:
        for condition, seed in runs:
            print(shlex.join(command_for(condition, seed, args.threads, args.output, args.smoke)))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"Logs: {args.output}", flush=True)

    launcher = Launcher(args)

    def terminate(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, terminate)
    pool = ThreadPoolExecutor(max_workers=at_once)
    futures = [pool.submit(launcher.run_one, condition, seed) for condition, seed in runs]
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

    results.sort(key=lambda result: (result["condition"], result["seed"]))
    (args.output / "runs.json").write_text(json.dumps(results, indent=2) + "\n")
    done = [result for result in results if result["status"] == "done"]
    failed = len(results) - len(done)
    print(f"{len(done)} of {len(results)} runs finished" + (f", {failed} FAILED" if failed else ""), flush=True)
    if done and not args.smoke:
        report = " ".join(f"--run {r['condition']} {r['run_dir']}/config.yaml {r['run_dir']}" for r in done)
        print(f"Report: python report_step0.py {report} --output {args.output / 'step0_report.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Stopped: the running training processes were terminated.", file=sys.stderr)
        sys.exit(130)
