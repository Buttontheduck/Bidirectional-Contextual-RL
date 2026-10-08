"""Run A-D, select one Gaussian using validation across seeds, then run E-F."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "A": "A_point_baseline", "B": "B_point_speed", "C": "C_gaussian_nll",
    "D": "D_gaussian_beta_nll", "E": "E_gaussian_mean_only", "F": "F_gaussian_speed",
}
SMOKE_OVERRIDES = [
    "training.n_envs=2", "cppo.n_steps=8", "cppo.n_minibatches=2", "cppo.n_epochs=1",
    "cppo.net_arch=[16]", "cppo.context_net_arch=[8]", "cppo.velocity_head.net_arch=[8]",
    "cppo.xlstm.embedding_dim=16", "cppo.xlstm.mlstm_block.mlstm.qkv_proj_blocksize=4",
    "cppo.xlstm.mlstm_block.mlstm.num_heads=2", "cppo.xlstm.slstm_block.slstm.num_heads=2",
    "cppo.supervised.episode_batch_size=2", "cppo.supervised.gradient_steps=1",
    "+env.kwargs.T=4", "evaluation.frequency=16", "cppo.verbose=0",
    "evaluation.reset_grid.eval_seeds=[1000,1001]", "evaluation.reset_grid.eval_initial_battery=[0.3,0.6]",
]


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def source_digest() -> str:
    files = [ROOT / "main_battery_plane_contextual_ppo.py", ROOT / "battery_plane.py", ROOT / "run_grid.py"]
    files += list((ROOT / "config").rglob("*.yaml"))
    files += [p for p in (ROOT / "contextual_ppo").glob("*.py") if not p.name.startswith("test_")]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def child_environment(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def command_for(args, letter: str, seed: int, run_dir: Path, beta: float | None = None) -> list[str]:
    command = [
        sys.executable, str(ROOT / "main_battery_plane_contextual_ppo.py"),
        f"+experiment={EXPERIMENTS[letter]}", f"training.seed={seed}",
        f"training.total_timesteps={args.steps}", "training.device=cpu",
        f"training.num_threads={args.threads}", "training.num_interop_threads=1",
        "evaluation.render=false", f"paths.run_dir={json.dumps(str(run_dir))}",
    ]
    if args.wandb_mode is not None:
        command.append(f"wandb.mode={args.wandb_mode}")
    if letter in "EF":
        if beta not in (0.0, 0.5):
            raise ValueError("E/F need the Gaussian setting selected from C/D validation")
        command.append(f"selected_gaussian.nll_beta={beta}")
    if args.smoke:
        command.extend(SMOKE_OVERRIDES)
        command.append("wandb.mode=disabled")
    return command


def completed(output: Path, letter: str, seed: int, beta=None):
    pointer = output / f"{letter}_s{seed}" / "result.json"
    if not pointer.exists():
        return None
    result = read_json(pointer)
    if result["experiment"] != letter or result["seed"] != seed or result["selected_beta"] != beta:
        raise ValueError(f"Result identity mismatch: {pointer}")
    run_dir = output / result["run_dir"]
    if not (run_dir / "final_model.zip").is_file():
        raise ValueError(f"Completed checkpoint is missing: {run_dir}")
    validation = read_json(run_dir / "final_validation.json")
    if not math.isfinite(validation["mean_reward"]) or validation != result["validation"]:
        raise ValueError(f"Completed validation result is invalid or changed: {run_dir}")
    return result


def choose_gaussian(results: list[dict], seeds: list[int]) -> dict:
    scores = {}
    for letter in "CD":
        rows = [r for r in results if r["experiment"] == letter]
        if len(rows) != len(seeds) or sorted(r["seed"] for r in rows) != sorted(seeds):
            raise ValueError(f"Need exactly one completed {letter} result for every seed")
        values = [r["validation"]["mean_reward"] for r in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Selection requires finite validation returns")
        scores[letter] = statistics.mean(values)
    chosen = "D" if scores["D"] > scores["C"] else "C"
    return {
        "selected_experiment": chosen, "nll_beta": 0.5 if chosen == "D" else 0.0,
        "criterion": "Highest mean final-checkpoint validation return across all training seeds; ties choose C",
        "seeds": seeds, "mean_validation_return": scores,
    }


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another grid launcher is using {output}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Runner:
    def __init__(self, args):
        self.args = args
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.children = set()

    def stop(self):
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

    def run_one(self, letter, seed, beta):
        if self.stopping.is_set():
            raise RuntimeError("Grid stopped")
        old = completed(self.args.output, letter, seed, beta)
        if old is not None:
            print(f"REUSE {letter} seed {seed}", flush=True)
            return old
        folder = self.args.output / f"{letter}_s{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        attempt = f"attempt-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir = folder / attempt
        log = folder / f"{attempt}.log"
        command = command_for(self.args, letter, seed, run_dir, beta)
        started = time.monotonic()
        print(f"START {letter} seed {seed} | log: {log}", flush=True)
        with log.open("x") as stream:
            with self.lock:
                if self.stopping.is_set():
                    raise RuntimeError("Grid stopped")
                process = subprocess.Popen(command, cwd=ROOT, env=child_environment(self.args.threads),
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.children.add(process)
            try:
                code = process.wait()
            finally:
                with self.lock:
                    self.children.discard(process)
        if code:
            raise RuntimeError(f"{letter} seed {seed} failed (exit {code}); inspect {log}")
        validation = read_json(run_dir / "final_validation.json")
        if not math.isfinite(validation["mean_reward"]) or not (run_dir / "final_model.zip").is_file():
            raise RuntimeError(f"Run lacks a valid final result: {run_dir}")
        if validation["timesteps"] < self.args.steps:
            raise RuntimeError(f"Run did not complete its training budget: {run_dir}")
        result = {
            "experiment": letter, "seed": seed, "selected_beta": beta,
            "run_dir": str(run_dir.relative_to(self.args.output)), "validation": validation,
            "elapsed_seconds": time.monotonic() - started,
        }
        save_json(folder / "result.json", result)
        print(f"DONE {letter} seed {seed} | validation {validation['mean_reward']:.3f} "
              f"| {result['elapsed_seconds'] / 60:.1f} minutes", flush=True)
        return result

    def batch(self, letters: str, beta=None):
        pool = ThreadPoolExecutor(max_workers=self.args.jobs)
        futures = [pool.submit(self.run_one, letter, seed, beta) for letter in letters for seed in self.args.seeds]
        try:
            return [f.result() for f in as_completed(futures)]
        except BaseException:
            self.stop()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "screen", "followup"], default="all")
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent training processes (initial 9950X setting)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads per training process")
    parser.add_argument("--steps", type=int, default=None, help="Per-run transitions; default 1M (32 for smoke)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, help="Grid state and artifacts; rerun the same command to reuse completed runs")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--smoke", action="store_true", help="Tiny models/episodes/budget for installation checks only")
    parser.add_argument("--dry-run", action="store_true", help="Describe the grid without starting jobs or writing files")
    args = parser.parse_args(argv)
    args.steps = args.steps if args.steps is not None else (32 if args.smoke else 1_000_000)
    if min(args.jobs, args.threads, args.steps) < 1 or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Jobs, threads and steps must be positive; seeds must be distinct non-negative integers")
    args.output = (args.output or ROOT / "runs_cppo" / ("grid_smoke" if args.smoke else "grid")).resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"{args.jobs} simultaneous runs x {args.threads} threads; {args.steps:,} steps/run; seeds {args.seeds}")
    print(f"A-D: {4 * len(args.seeds)} runs. Select C/D on validation. E-F: {2 * len(args.seeds)} runs.")
    print("Final-test episodes are reserved for evaluate_final_test.py.")
    if args.dry_run:
        return
    identity = {"source_sha256": source_digest(), "seeds": args.seeds, "steps": args.steps,
                "threads": args.threads, "smoke": args.smoke,
                "versions": {name: version(name) for name in (
                    "torch", "stable-baselines3", "xlstm", "numpy", "gymnasium", "hydra-core", "omegaconf",
                )}}
    with output_lock(args.output):
        manifest = args.output / "manifest.json"
        if manifest.exists():
            if read_json(manifest) != identity:
                raise ValueError("This output folder has different code/settings. Use a new --output directory.")
        else:
            if any(args.output.glob("*_s*")):
                raise ValueError("Output contains untracked runs; use a fresh --output directory")
            save_json(manifest, identity)
        runner = Runner(args)
        def terminate(signum, frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            if args.stage == "followup":
                screening = [completed(args.output, letter, seed) for letter in "ABCD" for seed in args.seeds]
                if any(result is None for result in screening):
                    raise ValueError("Finish the screening stage in this output directory first")
            else:
                screening = runner.batch("ABCD")
            selection = choose_gaussian(screening, args.seeds)
            selection["source_sha256"] = identity["source_sha256"]
            save_json(args.output / "selection.json", selection)
            print(f"SELECT {selection['selected_experiment']} | beta={selection['nll_beta']} "
                  f"| validation means {selection['mean_validation_return']}", flush=True)
            if args.stage != "screen":
                followup = runner.batch("EF", beta=selection["nll_beta"])
                save_json(args.output / "grid_results.json", {
                    "selection": selection, "smoke": args.smoke,
                    "runs": sorted(screening + followup, key=lambda r: (r["experiment"], r["seed"])),
                })
                print(f"All {6 * len(args.seeds)} runs completed. Results: {args.output / 'grid_results.json'}")
        except BaseException:
            runner.stop()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Completed runs are reusable; unfinished runs restart in new attempt directories.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
