"""Step 4 of ENCODER_PLAN.md, diagnostics: why the offline encoder stalls near the clock.

E1 and E3 (train_encoder.py) close only about 20% of the gap between the clock-only estimate and the floor
(step4_floor.py). These runs separate the encoder's two jobs, and the architecture from the training, on the same
flights, split (by flight seed) and error measures as train_encoder.py.

Tasks (--task), each a label computed from the recorded flights:
    count    the thrust used so far, c_t = dt/E * sum over the executed actions before o_t of (a_x^2 + a_y^2): on the
             scale of the battery's drain (at most 0.001 per step), and fully determined by the inputs, so its floor
             is 0. Tests counting: adding up small amounts over hundreds of steps.
    find     the starting battery b0, the same at every step of a flight. On full-throttle flights (--mix narrow, the
             default) the speed at the cap gives it away: the ideal estimator is within a few percent by 20 s. Tests
             reading the battery from the speed.
    battery  the true battery b_t: the real task (E1's), to compare architectures on it.
Architectures (--arch):
    xlstm    A's encoder (train_encoder.make_encoder), stepped through each flight in a Python loop (slow)
    gru, lstm  PyTorch's built-in recurrent layers over whole flights (fast): an input projection to the width, --layers
             layers of width --hidden, and A's head (width -> 128 -> 128 -> 1, tanh, sigmoid)
Both read [o_t, a_{t-1}] (hidden-mode observation, previous executed action) from their initial state at the start of
every flight. Padding at the end of shorter flights is masked; the recurrent layers are causal, so it never changes
the real steps' outputs.

The training seed sets only the initial weights and the batch order. Each epoch is one pass over the training flights
(--train-flights: an evenly spaced subset, for the slow xLSTM); early stopping on the validation error, the test flights
scored once with the best checkpoint, next to the clock-only estimate (the mean label at each flight step).

Outputs (<output root>/<name>/; runs_encoder/ is gitignored): config.json, metrics.jsonl, best.pt, last.pt, result.json.
W&B: project battery-encoder, group probe_<arch>_<task>_<mix>[_<dataset folder, if not v2>][_n<train flights>][_lr<rate>],
run <group>_s<seed>.

    python step4_probes.py --arch gru --task battery --seed 0
    python step4_probes.py --arch xlstm --task count --train-flights 3000 --seed 0
    python step4_probes.py --arch gru --task find --seed 0 --smoke        # tiny run, W&B disabled
"""
from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

import train_encoder as T

TASKS = ("count", "find", "battery")
ARCHS = ("xlstm", "gru", "lstm")
DEFAULT_MIX = {"count": "main", "find": "narrow", "battery": "main"}
DEFAULT_LEARNING_RATE = 3e-4
HASHED_CODE = ("step4_probes.py", *T.HASHED_CODE)


def task_labels(task: str, data: dict, dt: float, E: float) -> list[np.ndarray]:
    """The label at every step of every flight (data: pilots.load_flights output). Only the battery labels are
    privileged; count reads the inputs, find the battery at t = 0."""
    if task == "battery":
        return [np.asarray(b, np.float32) for b in data["batteries"]]
    if task == "find":
        return [np.full(len(b), b[0], np.float32) for b in data["batteries"]]
    # prev_actions[t] is a_{t-1}, zero at t = 0: c_0 = 0 and c_t adds a_{t-1}
    return [(dt / E * np.cumsum((np.asarray(p, np.float64) ** 2).sum(axis=1))).astype(np.float32) for p in data["prev_actions"]]


def make_model(arch: str, cfg, hidden: int, layers: int):
    """xLSTM: A's encoder; GRU / LSTM: an input projection, the recurrent layers and A's head."""
    import torch as th
    from torch import nn

    if arch == "xlstm":
        return T.make_encoder(cfg, b0_head=False)
    from stable_baselines3.common.torch_layers import create_mlp

    import main_battery_plane_contextual_ppo as training

    activation = training.ACTIVATIONS[str(cfg.cppo.activation_fn).lower()]

    class RecurrentEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observation_dim, self.action_dim = 4, 2
            self.input_projection = nn.Linear(6, hidden)
            self.rnn = (nn.GRU if arch == "gru" else nn.LSTM)(hidden, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Sequential(*create_mlp(hidden, 1, list(cfg.cppo.context_net_arch), activation), nn.Sigmoid())

        def forward(self, observations: th.Tensor, prev_actions: th.Tensor) -> th.Tensor:
            """[B, L, 4] and [B, L, 2] from the start of each flight -> estimates [B, L]"""
            outputs, _ = self.rnn(self.input_projection(th.cat([observations, prev_actions], dim=-1)))
            return self.head(outputs)[..., 0]

    return RecurrentEncoder()


def predict(model, arch: str, batch: dict):
    if arch == "xlstm":
        return T.predict(model, batch, b0_head=False)[0]
    return model(batch["observations"], batch["prev_actions"])


def evaluate(model, arch: str, flights: T.Flights, rows: np.ndarray, steps_at: dict[str, int], batch_size: int = 256) -> dict:
    import torch as th

    model.eval()
    tally = T.ErrorTally(steps_at)
    order = rows[np.argsort(flights.lengths[rows], kind="stable")]
    with th.no_grad():
        for start in range(0, len(order), batch_size):
            part = order[start:start + batch_size]
            estimates = predict(model, arch, flights.batch(part))
            for j, i in enumerate(part):
                tally.add(estimates[j, :flights.lengths[i]].numpy() - flights.batteries[i], flights.styles[i])
    model.train()
    return tally.summary()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Initial weights and batch order (not the split)")
    parser.add_argument("--data", type=Path, default=T.DEFAULT_DATA, help="pilots.py output folder")
    parser.add_argument("--mix", default=None, help="A mix of pilots.MIXES (default: narrow for find, main otherwise)")
    parser.add_argument("--train-flights", type=int, default=None, help="An evenly spaced subset of the training flights")
    parser.add_argument("--output-root", type=Path, default=T.PROJECT_ROOT / "runs_encoder" / "step4_probes")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Adam; E1's by default")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=128, help="GRU / LSTM width")
    parser.add_argument("--layers", type=int, default=2, help="GRU / LSTM layers")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-flights", type=int, default=None, help="Only some flights of each split (checks)")
    parser.add_argument("--wandb-mode", default=None, help="online, offline or disabled (default: WANDB_MODE)")
    parser.add_argument("--smoke", action="store_true", help="48 flights per split, 2 epochs, W&B disabled")
    args = parser.parse_args(argv)
    if args.smoke:
        args.max_flights, args.max_epochs, args.wandb_mode, args.log_every = args.max_flights or 48, 2, "disabled", 1
        args.output_root = args.output_root.with_name(args.output_root.name + "_smoke")
    if min(args.batch_size, args.max_epochs, args.patience, args.threads, args.hidden, args.layers) < 1:
        parser.error("--batch-size, --max-epochs, --patience, --threads, --hidden and --layers must be at least 1")
    if args.train_flights is not None and args.train_flights < 1:
        parser.error("--train-flights must be at least 1")
    mix = args.mix or DEFAULT_MIX[args.task]

    import torch as th
    import wandb

    import battery_plane as bp
    import main_battery_plane_contextual_ppo as training
    import pilots

    th.set_num_threads(args.threads)
    if th.get_num_interop_threads() != 1:
        th.set_num_interop_threads(1)
    # Another dataset than v2 adds its folder name, so W&B names and run directories stay distinct
    data_tag = "" if args.data.resolve() == T.DEFAULT_DATA.resolve() else f"_{args.data.name}"
    group = (f"probe_{args.arch}_{args.task}_{mix}{data_tag}" + (f"_n{args.train_flights}" if args.train_flights else "")
             + (f"_lr{args.learning_rate:g}" if args.learning_rate != DEFAULT_LEARNING_RATE else ""))
    name = f"{group}_s{args.seed}"
    run_dir = args.output_root / name
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"{run_dir} already holds a run: choose another --output-root")

    cfg = T.encoder_config([f"training.seed={args.seed}"])
    env_kwargs = training.env_kwargs_from(cfg)
    manifest = json.loads((args.data / "manifest.json").read_text())
    if dict(manifest["env_kwargs"]) != env_kwargs:
        parser.error(f"The flights were recorded with env kwargs {manifest['env_kwargs']}, A's config uses {env_kwargs}")
    core = bp.BatteryPlaneVec(n_envs=1, observation_mode="hidden", seed=0, **env_kwargs)
    steps_at = {f"{s}s": int(round(s / core.dt)) for s in T.SECONDS_AT}
    flights, _ = T.load(args.data, mix, args.max_flights)
    leaking = [s for s in pilots.MIXES[mix] if manifest["styles"][s]["leaks"]]
    data = {"observations": flights.observations, "prev_actions": flights.prev_actions, "batteries": flights.batteries,
            "styles": flights.styles, "seeds": flights.seeds}
    # From here on, Flights.batteries holds the task's labels (the true battery only for --task battery)
    flights = T.Flights({**data, "batteries": task_labels(args.task, data, core.dt, core.E)})
    rows = {split: flights.indices(split) for split in ("train", "validation", "test")}
    if args.train_flights is not None:
        # The run's name says _n<train flights>: refuse a subset that isn't one
        if args.train_flights >= len(rows["train"]):
            parser.error(f"--train-flights {args.train_flights} is not smaller than the {len(rows['train'])} training flights")
        rows["train"] = rows["train"][np.unique(np.linspace(0, len(rows["train"]) - 1, args.train_flights).round().astype(int))]
    if any(len(r) == 0 for r in rows.values()):
        parser.error(f"Empty split: { {k: len(v) for k, v in rows.items()} }")
    clock = T.clock_estimate(flights, rows["train"])

    th.manual_seed(args.seed)
    model = make_model(args.arch, cfg, args.hidden, args.layers)
    model.train()
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng([args.seed, 4])

    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "arch": args.arch, "task": args.task, "name": name, "group": group, "seed": args.seed, "mix": mix,
        "styles": list(pilots.MIXES[mix]), "leaking_styles": leaking, "data": str(args.data), "train_flights": args.train_flights,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "max_grad_norm": args.max_grad_norm,
        "max_epochs": args.max_epochs, "patience": args.patience, "threads": args.threads, "max_flights": args.max_flights,
        "hidden": args.hidden if args.arch != "xlstm" else None, "layers": args.layers if args.arch != "xlstm" else None,
        "steps_at": steps_at, "parameters": int(sum(p.numel() for p in model.parameters())),
        "split": {"rule": f"train_encoder.split_of (flight seed % {T.SPLIT_MODULUS})",
                  "flights": {k: int(len(v)) for k, v in rows.items()},
                  "steps": {k: int(flights.lengths[v].sum()) for k, v in rows.items()}},
        "code_sha256": {path: T.sha256(T.PROJECT_ROOT / path) for path in HASHED_CODE},
        "data_manifest_sha256": T.sha256(args.data / "manifest.json"),
        "git": T.git_state(), "versions": {pkg: version(pkg) for pkg in ("torch", "xlstm", "numpy")},
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    run = wandb.init(project=T.WANDB_PROJECT, name=name, group=group, tags=["step4", "probe", args.arch, args.task, mix],
                     config=settings, mode=args.wandb_mode, dir=str(T.PROJECT_ROOT))
    print(f"{name}: {settings['split']['flights']} flights, {settings['parameters']:,} parameters, run directory {run_dir}", flush=True)

    best, best_epoch, stale, step = float("inf"), 0, 0, 0
    started = time.time()
    with open(run_dir / "metrics.jsonl", "w") as metrics:
        for epoch in range(1, args.max_epochs + 1):
            epoch_start, losses = time.time(), []
            for rows_batch in T.length_batches(rows["train"], flights.lengths, args.batch_size, rng):
                batch = flights.batch(rows_batch)
                loss = T.masked_mse(predict(model, args.arch, batch), batch["labels"], batch["mask"])
                optimizer.zero_grad()
                loss.backward()
                grad_norm = float(th.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
                optimizer.step()
                step += 1
                losses.append(loss.item())
                if not np.isfinite(losses[-1]):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                if step % args.log_every == 0:
                    run.log({"train/mse": losses[-1], "train/grad_norm": grad_norm, "train/epoch": epoch}, step=step)
            validation = evaluate(model, args.arch, flights, rows["validation"], steps_at)
            improved = validation["mse"] < best
            if improved:
                best, best_epoch, stale = validation["mse"], epoch, 0
                th.save({"state_dict": model.state_dict(), "settings": settings, "epoch": epoch}, run_dir / "best.pt")
            else:
                stale += 1
            th.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step},
                    run_dir / "last.pt")
            row = {"epoch": epoch, "step": step, "train_mse": float(np.mean(losses)), "validation": validation,
                   "best_epoch": best_epoch, "epoch_seconds": time.time() - epoch_start}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            run.log({"validation/rmse": validation["rmse"], "validation/mse": validation["mse"], "train/epoch_mse": row["train_mse"],
                     "epoch": epoch, "epoch_seconds": row["epoch_seconds"],
                     **{f"validation/rmse_at_{k}": v for k, v in validation["rmse_at"].items()},
                     **{f"validation/by_style/{k}": v for k, v in validation["rmse_by_style"].items()}}, step=step)
            print(f"epoch {epoch}: validation rmse {validation['rmse']:.4f} (at 1/5/20 s: "
                  + " / ".join(f"{v:.4f}" for v in validation["rmse_at"].values())
                  + f"){' best' if improved else ''}, {row['epoch_seconds'] / 60:.1f} min", flush=True)
            if stale >= args.patience:
                print(f"No better validation error for {args.patience} epochs: stopping", flush=True)
                break

    model.load_state_dict(th.load(run_dir / "best.pt", weights_only=False)["state_dict"])
    result = {
        "name": name, "group": group, "task": args.task, "arch": args.arch, "best_epoch": best_epoch, "epochs": epoch,
        "steps": step, "hours": (time.time() - started) / 3600,
        "validation": evaluate(model, args.arch, flights, rows["validation"], steps_at),
        "test": evaluate(model, args.arch, flights, rows["test"], steps_at),
        "clock_validation": T.evaluate_clock(clock, flights, rows["validation"], steps_at),
        "clock_test": T.evaluate_clock(clock, flights, rows["test"], steps_at),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    run.summary.update({"best_epoch": best_epoch, "test/rmse": result["test"]["rmse"],
                        **{f"test/rmse_at_{k}": v for k, v in result["test"]["rmse_at"].items()},
                        "clock_test/rmse": result["clock_test"]["rmse"]})
    run.finish()
    print(f"{name}: best epoch {best_epoch}, test rmse {result['test']['rmse']:.4f} (clock only "
          f"{result['clock_test']['rmse']:.4f}), {result['hours']:.2f} h", flush=True)


if __name__ == "__main__":
    main()
