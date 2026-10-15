"""Step 4 of ENCODER_PLAN.md: train the battery encoder offline on the pilot flights, then save it for plain PPO.

The encoder is experiment A's (config/cppo.yaml: the same xLSTM blocks and widths, the point head with a sigmoid). It
reads [o_t, a_{t-1}] (the hidden-mode observation and the previous executed action, zero at the start) over complete
flights from its initial state, step by step, as Contextual_PPO and the context wrapper run it. The label is the true
battery b_t of each step (privileged: a target, never an input). Designs:
    E1  the point head, squared error on b_t
    E3  E1 plus a second head that predicts the flight's starting battery b0 at every step (squared error times
        --b0-weight). It only shapes training: the saved model drops it, and PPO reads the b_t estimate only.

Flights are split by their seed, the same way for every design and training seed: seed % 10 == 8 is validation (early
stopping and the best checkpoint), seed % 10 == 9 is test (scored once, at the end, with the best checkpoint), the rest
is training. The training seed sets only the encoder's initial weights and the order of the batches.

Each epoch is one pass over the training flights, in batches of whole flights of similar length (padding is masked:
it neither advances the memory nor counts in the loss). Training stops when the validation error of b_t has not
improved for --patience epochs, or after --max-epochs. Errors are reported overall, at 1, 5 and 20 s into the flight
and per style, next to the clock-only estimate (the mean training battery at each flight step).

Outputs (<output root>/<name>/; runs_encoder/ is gitignored):
    config.json         settings, split sizes, code and data hashes
    metrics.jsonl       one line per epoch
    best_encoder.pt     the best encoder's weights (with the b0 head for E3) and its settings
    last_encoder.pt     the last epoch's weights and optimizer state
    encoder_model.zip   a Contextual_PPO model holding the best encoder, for context.source=encoder (its actor and
                        critic are untrained and unused)
    result.json         validation and test errors of the best encoder and of the clock-only estimate

    python train_encoder.py --design E1 --seed 0 --output-root runs_encoder/step4
    python train_encoder.py --design E3 --seed 0 --smoke        # tiny correctness run, W&B disabled
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

DESIGNS = {"E1": {"b0_head": False, "label": "point"}, "E3": {"b0_head": True, "label": "point_b0"}}
SPLIT_MODULUS, VALIDATION_REMAINDER, TEST_REMAINDER = 10, 8, 9
SECONDS_AT = (1, 5, 20)
DEFAULT_DATA = PROJECT_ROOT / "runs_encoder" / "pilot_flights" / "v2"
WANDB_PROJECT = "battery-encoder"
HASHED_CODE = ("train_encoder.py", "pilots.py", "main_battery_plane_contextual_ppo.py", "config/cppo.yaml",
               "contextual_ppo/torch_layers.py", "contextual_ppo/policies.py", "battery_plane.py")


def split_of(seed: int) -> str:
    """The split of a flight, from its seed alone: the same for every design and training seed."""
    remainder = int(seed) % SPLIT_MODULUS
    return "validation" if remainder == VALIDATION_REMAINDER else "test" if remainder == TEST_REMAINDER else "train"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_config(overrides=()):
    """Experiment A's config (config/cppo.yaml): the encoder settings, and the environment the model is saved with."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "config")):
        return compose(config_name="cppo", overrides=["cppo.context_head=point", "cppo.velocity_head.enabled=false",
                                                      *overrides])


def make_encoder(cfg, b0_head: bool):
    """A's encoder, built with the arguments Contextual_PPO's policy uses (policies.make_encoder), on hidden-mode
    observations (4 values) and actions (2); with a b0 head (one output) for E3."""
    from omegaconf import OmegaConf

    from contextual_ppo.torch_layers import XLSTMRolloutEncoder
    import main_battery_plane_contextual_ppo as training

    cppo = cfg.cppo
    return XLSTMRolloutEncoder(
        observation_dim=4, action_dim=2, context_dim=int(cppo.context_dim),
        xlstm_config=OmegaConf.to_container(cppo.xlstm, resolve=True), projection_net_arch=list(cppo.context_net_arch),
        activation_fn=training.ACTIVATIONS[str(cppo.activation_fn).lower()], context_activation=cppo.context_activation,
        context_head="point", auxiliary_dim=1 if b0_head else None,
        auxiliary_net_arch=list(cppo.velocity_head.net_arch) if b0_head else None)


class Flights:
    """The flights of a mix as per-flight arrays (inputs, the label b_t, and b0 = b_t at t = 0), with styles, seeds
    and splits."""

    def __init__(self, data: dict) -> None:
        self.observations = data["observations"]
        self.prev_actions = data["prev_actions"]
        self.batteries = data["batteries"]
        self.styles = data["styles"]
        self.seeds = data["seeds"]
        self.lengths = np.array([len(b) for b in self.batteries])
        self.splits = np.array([split_of(seed) for seed in self.seeds])
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Flight seeds repeat: the split would put copies of a flight on both sides")

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == split)

    def batch(self, rows) -> dict:
        """Padded tensors of the given flights: inputs [B, L, .], labels [B, L], b0 [B], mask [B, L]."""
        import torch as th

        length = int(self.lengths[rows].max())
        obs, prev = np.zeros((len(rows), length, 4), np.float32), np.zeros((len(rows), length, 2), np.float32)
        labels, mask = np.zeros((len(rows), length), np.float32), np.zeros((len(rows), length), bool)
        for j, i in enumerate(rows):
            n = self.lengths[i]
            obs[j, :n], prev[j, :n], labels[j, :n], mask[j, :n] = self.observations[i], self.prev_actions[i], self.batteries[i], True
        b0 = np.array([self.batteries[i][0] for i in rows], np.float32)
        return {"observations": th.from_numpy(obs), "prev_actions": th.from_numpy(prev), "labels": th.from_numpy(labels),
                "b0": th.from_numpy(b0), "mask": th.from_numpy(mask)}


def load(directory: Path, mix: str, max_flights: int | None = None) -> tuple[Flights, dict]:
    """The flights of a mix and the dataset's manifest. max_flights: the first flights of each split (smoke runs)."""
    import pilots

    manifest = json.loads((directory / "manifest.json").read_text())
    styles = pilots.MIXES[mix]
    missing = [s for s in styles if s not in manifest["styles"]]
    if missing:
        raise ValueError(f"{directory} has no flights of {missing}")
    data = pilots.load_flights(directory, styles)
    if max_flights is not None:
        keep = []
        for split in ("train", "validation", "test"):
            rows = [i for i, seed in enumerate(data["seeds"]) if split_of(seed) == split]
            # Evenly spaced over the split, so every style is represented
            keep += sorted({rows[int(j)] for j in np.linspace(0, len(rows) - 1, min(max_flights, len(rows)))})
        data = {key: [values[i] for i in sorted(keep)] for key, values in data.items()}
    return Flights(data), manifest


def length_batches(rows: np.ndarray, lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Random batches of whole flights of similar length: shuffle, sort chunks of 16 batches by length, cut, shuffle."""
    rows = rng.permutation(rows)
    batches = []
    chunk = 16 * batch_size
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def predict(encoder, batch: dict, b0_head: bool):
    """The encoder over whole flights from its initial state: b_t estimates [B, L] and b0 estimates [B, L] (E3)."""
    state = encoder.initial_state(batch["observations"].shape[0])
    if b0_head:
        contexts, b0, _ = encoder.unroll_with_auxiliary(state, batch["observations"], batch["prev_actions"], batch["mask"])
        return contexts[..., 0], b0[..., 0]
    contexts, _ = encoder.unroll(state, batch["observations"], batch["prev_actions"], batch["mask"])
    return contexts[..., 0], None


def masked_mse(prediction, target, mask):
    mask = mask.to(prediction.dtype)
    return ((prediction - target) ** 2 * mask).sum() / mask.sum()


def clock_estimate(flights: Flights, rows: np.ndarray) -> np.ndarray:
    """The clock-only estimate: the mean battery at each flight step over the given (training) flights still flying;
    after the longest, its last value."""
    longest = int(flights.lengths[rows].max())
    total, count = np.zeros(longest), np.zeros(longest)
    for i in rows:
        n = flights.lengths[i]
        total[:n] += flights.batteries[i]
        count[:n] += 1
    return total / count


class ErrorTally:
    """Squared errors of an estimator over flights: overall, at given flight steps, and per style."""

    def __init__(self, steps_at: dict[str, int]) -> None:
        self.steps_at = steps_at
        self.sum, self.count = 0.0, 0
        self.at = {name: [0.0, 0] for name in steps_at}
        self.style: dict[str, list] = {}
        self.b0_sum = 0.0

    def add(self, errors: np.ndarray, style: str, b0_errors: np.ndarray | None = None) -> None:
        """errors: one flight's estimate - label at each step."""
        squared = errors.astype(np.float64) ** 2
        self.sum += squared.sum()
        self.count += len(squared)
        for name, step in self.steps_at.items():
            if step < len(squared):
                self.at[name][0] += squared[step]
                self.at[name][1] += 1
        tally = self.style.setdefault(style, [0.0, 0])
        tally[0] += squared.sum()
        tally[1] += len(squared)
        if b0_errors is not None:
            self.b0_sum += (b0_errors.astype(np.float64) ** 2).sum()

    def summary(self, with_b0: bool = False) -> dict:
        rmse = lambda s, n: float(np.sqrt(s / n)) if n else None
        out = {"mse": self.sum / self.count, "rmse": rmse(self.sum, self.count),
               "rmse_at": {name: rmse(*value) for name, value in self.at.items()},
               "flights_at": {name: value[1] for name, value in self.at.items()},
               "rmse_by_style": {style: rmse(*value) for style, value in sorted(self.style.items())}}
        if with_b0:
            out["b0_rmse"] = rmse(self.b0_sum, self.count)
        return out


def evaluate(encoder, flights: Flights, rows: np.ndarray, b0_head: bool, steps_at: dict[str, int], batch_size: int = 256) -> dict:
    """Errors of the encoder on the given flights (no gradient)."""
    import torch as th

    encoder.eval()
    tally = ErrorTally(steps_at)
    order = rows[np.argsort(flights.lengths[rows], kind="stable")]
    with th.no_grad():
        for start in range(0, len(order), batch_size):
            part = order[start:start + batch_size]
            batch = flights.batch(part)
            estimates, b0 = predict(encoder, batch, b0_head)
            for j, i in enumerate(part):
                n = flights.lengths[i]
                b0_errors = None if b0 is None else b0[j, :n].numpy() - flights.batteries[i][0]
                tally.add(estimates[j, :n].numpy() - flights.batteries[i], flights.styles[i], b0_errors)
    encoder.train()
    return tally.summary(with_b0=b0_head)


def evaluate_clock(clock: np.ndarray, flights: Flights, rows: np.ndarray, steps_at: dict[str, int]) -> dict:
    tally = ErrorTally(steps_at)
    for i in rows:
        n = flights.lengths[i]
        estimate = clock[np.minimum(np.arange(n), len(clock) - 1)]
        tally.add(estimate - flights.batteries[i], flights.styles[i])
    return tally.summary()


def save_policy_model(cfg, encoder, path: Path) -> None:
    """A Contextual_PPO model (A's settings, hidden mode) holding the encoder without its b0 head, loadable by
    context_wrapper.FrozenEncoderContext. Building the model reseeds torch: call it after training."""
    import main_battery_plane_contextual_ppo as training

    env = training.make_train_env(cfg)
    model = training.build_model(cfg, env)
    shared = {key: value for key, value in encoder.state_dict().items() if not key.startswith("auxiliary_projection.")}
    # strict: the saved encoder must be exactly the trained one, every weight in place
    model.policy.encoder.load_state_dict(shared, strict=True)
    model.save(path)
    env.close()


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", choices=sorted(DESIGNS), required=True)
    parser.add_argument("--seed", type=int, required=True, help="Initial weights and batch order (not the split)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="pilots.py output folder")
    parser.add_argument("--mix", default="main", help="A mix of pilots.MIXES")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs_encoder" / "step4")
    parser.add_argument("--batch-size", type=int, default=32, help="Whole flights per gradient step")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam; A's learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3, help="Epochs without a better validation error before stopping")
    parser.add_argument("--b0-weight", type=float, default=1.0, help="E3: weight of the b0 head's squared error")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50, help="Gradient steps between training-loss logs")
    parser.add_argument("--max-flights", type=int, default=None, help="Only the first flights of each split")
    parser.add_argument("--wandb-mode", default=None, help="online, offline or disabled (default: WANDB_MODE)")
    parser.add_argument("--smoke", action="store_true", help="48 flights per split, 2 epochs, W&B disabled")
    args = parser.parse_args(argv)
    if args.smoke:
        args.max_flights, args.max_epochs, args.wandb_mode, args.log_every = args.max_flights or 48, 2, "disabled", 1
        args.output_root = args.output_root.with_name(args.output_root.name + "_smoke")
    if args.batch_size < 1 or args.max_epochs < 1 or args.patience < 1 or args.threads < 1:
        parser.error("--batch-size, --max-epochs, --patience and --threads must be at least 1")
    if not args.b0_weight > 0:
        parser.error("--b0-weight must be positive")

    from omegaconf import OmegaConf
    import torch as th
    import wandb

    import battery_plane as bp
    import main_battery_plane_contextual_ppo as training
    import pilots

    th.set_num_threads(args.threads)
    if th.get_num_interop_threads() != 1:
        th.set_num_interop_threads(1)     # only once per process, before any parallel work
    design = DESIGNS[args.design]
    b0_head = design["b0_head"]
    group = f"{args.design}_{design['label']}_{args.mix}"
    name = f"{group}_s{args.seed}"
    run_dir = args.output_root / name
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"{run_dir} already holds a run: choose another --output-root")

    cfg = encoder_config([f"training.seed={args.seed}"])
    flights, manifest = load(args.data, args.mix, args.max_flights)
    env_kwargs = training.env_kwargs_from(cfg)
    if dict(manifest["env_kwargs"]) != env_kwargs:
        parser.error(f"The flights were recorded with env kwargs {manifest['env_kwargs']}, A's config uses {env_kwargs}")
    leaking = [s for s in pilots.MIXES[args.mix] if manifest["styles"][s]["leaks"]]
    dt = bp.BatteryPlaneVec(n_envs=1, observation_mode="hidden", seed=0, **env_kwargs).dt
    steps_at = {f"{s}s": int(round(s / dt)) for s in SECONDS_AT}
    rows = {split: flights.indices(split) for split in ("train", "validation", "test")}
    if any(len(r) == 0 for r in rows.values()):
        parser.error(f"Empty split: { {k: len(v) for k, v in rows.items()} }")
    clock = clock_estimate(flights, rows["train"])

    th.manual_seed(args.seed)
    encoder = make_encoder(cfg, b0_head)
    encoder.train()
    optimizer = th.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng([args.seed, 4])

    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "design": args.design, "b0_head": b0_head, "name": name, "group": group, "seed": args.seed, "mix": args.mix,
        "styles": list(pilots.MIXES[args.mix]), "leaking_styles": leaking, "data": str(args.data),
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "max_grad_norm": args.max_grad_norm,
        "max_epochs": args.max_epochs, "patience": args.patience, "b0_weight": args.b0_weight if b0_head else None,
        "threads": args.threads, "max_flights": args.max_flights, "steps_at": steps_at,
        "split": {"rule": f"flight seed % {SPLIT_MODULUS}: {VALIDATION_REMAINDER} validation, {TEST_REMAINDER} test, rest train",
                  "flights": {k: int(len(v)) for k, v in rows.items()},
                  "steps": {k: int(flights.lengths[v].sum()) for k, v in rows.items()}},
        "encoder": {"xlstm": OmegaConf.to_container(cfg.cppo.xlstm, resolve=True), "context_net_arch": list(cfg.cppo.context_net_arch),
                    "activation_fn": str(cfg.cppo.activation_fn), "context_activation": cfg.cppo.context_activation,
                    "b0_net_arch": list(cfg.cppo.velocity_head.net_arch) if b0_head else None,
                    "parameters": int(sum(p.numel() for p in encoder.parameters()))},
        "code_sha256": {path: sha256(PROJECT_ROOT / path) for path in HASHED_CODE},
        "data_sha256": {"manifest.json": sha256(args.data / "manifest.json"),
                        **{f"{s}.npz": sha256(args.data / f"{s}.npz") for s in pilots.MIXES[args.mix]}},
        "git": git_state(), "versions": {pkg: version(pkg) for pkg in ("torch", "xlstm", "numpy", "stable-baselines3")},
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    run = wandb.init(project=WANDB_PROJECT, name=name, group=group, tags=["step4", args.design, args.mix],
                     config=json.loads((run_dir / "config.json").read_text()), mode=args.wandb_mode, dir=str(PROJECT_ROOT))
    print(f"{name}: {settings['split']['flights']} flights, {settings['encoder']['parameters']:,} parameters, "
          f"run directory {run_dir}", flush=True)

    best, best_epoch, stale, step = float("inf"), 0, 0, 0
    started = time.time()
    with open(run_dir / "metrics.jsonl", "w") as metrics:
        for epoch in range(1, args.max_epochs + 1):
            epoch_start, losses = time.time(), []
            for rows_batch in length_batches(rows["train"], flights.lengths, args.batch_size, rng):
                batch = flights.batch(rows_batch)
                estimates, b0 = predict(encoder, batch, b0_head)
                b_loss = masked_mse(estimates, batch["labels"], batch["mask"])
                loss = b_loss
                if b0_head:
                    b0_loss = masked_mse(b0, batch["b0"][:, None].expand_as(b0), batch["mask"])
                    loss = b_loss + args.b0_weight * b0_loss
                optimizer.zero_grad()
                loss.backward()
                grad_norm = float(th.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm))
                optimizer.step()
                step += 1
                losses.append(b_loss.item())
                if not np.isfinite(loss.item()):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                if step % args.log_every == 0:
                    log = {"train/b_mse": losses[-1], "train/grad_norm": grad_norm, "train/epoch": epoch}
                    if b0_head:
                        log["train/b0_mse"] = b0_loss.item()
                    run.log(log, step=step)
                    print(f"epoch {epoch} step {step}: b_t mse {losses[-1]:.5f}, {time.time() - epoch_start:.0f} s into the epoch", flush=True)
            validation = evaluate(encoder, flights, rows["validation"], b0_head, steps_at)
            improved = validation["mse"] < best
            if improved:
                best, best_epoch, stale = validation["mse"], epoch, 0
                th.save({"state_dict": encoder.state_dict(), "settings": settings, "epoch": epoch}, run_dir / "best_encoder.pt")
            else:
                stale += 1
            th.save({"state_dict": encoder.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step},
                    run_dir / "last_encoder.pt")
            row = {"epoch": epoch, "step": step, "train_b_mse": float(np.mean(losses)), "validation": validation,
                   "best_epoch": best_epoch, "epoch_seconds": time.time() - epoch_start}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            log = {"validation/b_rmse": validation["rmse"], "validation/b_mse": validation["mse"], "train/epoch_b_mse": row["train_b_mse"],
                   "epoch": epoch, "epoch_seconds": row["epoch_seconds"],
                   **{f"validation/b_rmse_at_{k}": v for k, v in validation["rmse_at"].items()},
                   **{f"validation/by_style/{k}": v for k, v in validation["rmse_by_style"].items()}}
            if b0_head:
                log["validation/b0_rmse"] = validation["b0_rmse"]
            run.log(log, step=step)
            print(f"epoch {epoch}: validation b_t rmse {validation['rmse']:.4f} (at 1/5/20 s: "
                  + " / ".join(f"{v:.4f}" for v in validation["rmse_at"].values())
                  + f"){' best' if improved else ''}, {row['epoch_seconds'] / 60:.1f} min", flush=True)
            if stale >= args.patience:
                print(f"No better validation error for {args.patience} epochs: stopping", flush=True)
                break

    checkpoint = th.load(run_dir / "best_encoder.pt", weights_only=False)
    encoder.load_state_dict(checkpoint["state_dict"])
    result = {
        "name": name, "group": group, "best_epoch": best_epoch, "epochs": epoch, "steps": step,
        "hours": (time.time() - started) / 3600,
        "validation": evaluate(encoder, flights, rows["validation"], b0_head, steps_at),
        "test": evaluate(encoder, flights, rows["test"], b0_head, steps_at),
        "clock_validation": evaluate_clock(clock, flights, rows["validation"], steps_at),
        "clock_test": evaluate_clock(clock, flights, rows["test"], steps_at),
        "test_seeds": [int(flights.seeds[i]) for i in rows["test"]],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_policy_model(cfg, encoder, run_dir / "encoder_model.zip")
    run.summary.update({"best_epoch": best_epoch, "test/b_rmse": result["test"]["rmse"],
                        **{f"test/b_rmse_at_{k}": v for k, v in result["test"]["rmse_at"].items()},
                        "clock_test/b_rmse": result["clock_test"]["rmse"]})
    run.finish()
    print(f"{name}: best epoch {best_epoch}, test b_t rmse {result['test']['rmse']:.4f} (clock only "
          f"{result['clock_test']['rmse']:.4f}), {result['hours']:.2f} h", flush=True)


if __name__ == "__main__":
    main()
