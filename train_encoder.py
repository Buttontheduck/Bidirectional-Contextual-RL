"""Step 4 of ENCODER_PLAN.md: train the battery encoder offline on the pilot flights, then save it for plain PPO.

The encoder is experiment A's (config/cppo.yaml: the same xLSTM blocks and widths, the point head with a sigmoid). It
reads [o_t, a_{t-1}] (the hidden-mode observation and the previous executed action, zero at the start) over complete
flights from its initial state, step by step, as Contextual_PPO and the context wrapper run it. The label is the true
battery b_t of each step (privileged: a target, never an input). Designs:
    E1  the point head, squared error on b_t
    E3  E1 plus a second head that predicts the flight's starting battery b0 at every step (squared error times
        --b0-weight). It only shapes training: the saved model drops it, and PPO reads the b_t estimate only.
--arch (ENCODER_PLAN.md, step 4's 2 x 2 plan): xlstm (the default, A's encoder, everything above) or a standalone encoder
of recurrent_encoders.py, gru or convgru, with --drop-tau (the encoder ignores tau, time left), --hidden, --layers,
--head-arch and, for convgru, --conv-channels and --conv-window. These save encoder.pt (recurrent_encoders.save_encoder,
read by the context wrapper) instead of encoder_model.zip, and their names carry the architecture, e.g.
E1_convgru_notau_point_main_s0; the xLSTM's names are unchanged.

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

--resume continues a finished run from its last checkpoint with more patience or more epochs, into a new run directory
<name>_resumed (W&B group <group>_resumed): the same settings, data and model code, the weights and optimizer state of
its last epoch, and the batch order replayed up to there, so it continues as the uninterrupted run would have. Its
metrics.jsonl starts with the original run's epochs, and its best checkpoint starts as the original's.

    python train_encoder.py --design E1 --seed 0 --output-root runs_encoder/step4
    python train_encoder.py --design E3 --seed 0 --smoke        # tiny correctness run, W&B disabled
    python train_encoder.py --resume runs_encoder/step4/E1_point_main_s2 --patience 8 --max-epochs 40
    python train_encoder.py --finalize runs_encoder/step4/E3_point_b0_main_s1     # a run stopped by hand

--finalize does for a run stopped by hand (between epochs, so its checkpoints match metrics.jsonl) what its natural end
does: result.json from the best checkpoint (marked stopped_by_hand), encoder_model.zip, and the W&B run's summary.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
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
               "contextual_ppo/torch_layers.py", "contextual_ppo/policies.py", "battery_plane.py", "recurrent_encoders.py")
ARCHS = ("xlstm", "gru", "convgru")
# The architecture settings a run records; runs from before --arch were A's xLSTM
ARCH_SETTINGS = ("arch", "drop_tau", "hidden", "layers", "head_arch", "conv_channels", "conv_window")
XLSTM_ARCH = {"arch": "xlstm", "drop_tau": False, "hidden": None, "layers": None, "head_arch": None, "conv_channels": None,
              "conv_window": None}
ARCH_DEFAULTS = {"hidden": 128, "layers": 2, "head_arch": [128, 128], "conv_channels": 32, "conv_window": 16}


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
    if not hasattr(encoder, "unroll"):
        # recurrent_encoders: causal over whole flights, so the end padding of shorter flights never reaches a real step
        estimates, b0 = encoder(batch["observations"], batch["prev_actions"])
        return estimates, b0 if b0_head else None
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


def arch_settings(settings: dict) -> dict:
    """The architecture settings of a run (its config.json or the parsed options), A's xLSTM for runs from before --arch."""
    return {key: settings.get(key, XLSTM_ARCH[key]) for key in ARCH_SETTINGS}


def arch_tag(arch: dict) -> str:
    """The architecture's part of a run name: empty for A's xLSTM, e.g. _convgru_notau, plus any non-default size."""
    if arch["arch"] == "xlstm":
        return ""
    tag = f"_{arch['arch']}" + ("_notau" if arch["drop_tau"] else "")
    if (arch["hidden"], arch["layers"]) != (ARCH_DEFAULTS["hidden"], ARCH_DEFAULTS["layers"]):
        tag += f"_h{arch['hidden']}l{arch['layers']}"
    if list(arch["head_arch"]) != ARCH_DEFAULTS["head_arch"]:
        tag += "_head" + "-".join(str(x) for x in arch["head_arch"])
    if arch["arch"] == "convgru" and (arch["conv_channels"], arch["conv_window"]) != (ARCH_DEFAULTS["conv_channels"], ARCH_DEFAULTS["conv_window"]):
        tag += f"_c{arch['conv_channels']}w{arch['conv_window']}"
    return tag


def make_model(cfg, arch: dict, b0_head: bool):
    """A's xLSTM (make_encoder), or a GRU / ConvGRU of recurrent_encoders.py with a b0 head for E3."""
    if arch["arch"] == "xlstm":
        return make_encoder(cfg, b0_head)
    from recurrent_encoders import build_encoder

    kwargs = {"drop_tau": arch["drop_tau"], "hidden": arch["hidden"], "layers": arch["layers"],
              "head_arch": list(arch["head_arch"]), "aux_dim": 1 if b0_head else None}
    if arch["arch"] == "convgru":
        kwargs.update(conv_channels=arch["conv_channels"], window=arch["conv_window"])
    return build_encoder(arch["arch"], **kwargs)


def save_trained(cfg, encoder, arch: dict, run_dir: Path, name: str) -> None:
    """The best encoder for PPO: encoder_model.zip (a Contextual_PPO model) for the xLSTM, encoder.pt otherwise."""
    if arch["arch"] == "xlstm":
        save_policy_model(cfg, encoder, run_dir / "encoder_model.zip")
    else:
        from recurrent_encoders import save_encoder

        save_encoder(encoder, run_dir / "encoder.pt", {"name": name})


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


def epoch_log(row: dict, b0_head: bool) -> dict:
    """The W&B log of one metrics.jsonl row."""
    validation = row["validation"]
    log = {"validation/b_rmse": validation["rmse"], "validation/b_mse": validation["mse"], "train/epoch_b_mse": row["train_b_mse"],
           "epoch": row["epoch"], "epoch_seconds": row["epoch_seconds"],
           **{f"validation/b_rmse_at_{k}": v for k, v in validation["rmse_at"].items()},
           **{f"validation/by_style/{k}": v for k, v in validation["rmse_by_style"].items()}}
    if b0_head:
        log["validation/b0_rmse"] = validation["b0_rmse"]
    return log


# Settings a resumed run takes from the original run (they define the training it continues)
RESUMED_SETTINGS = ("design", "seed", "data", "mix", "batch_size", "learning_rate", "max_grad_norm", "b0_weight", "max_flights",
                    *ARCH_SETTINGS)
# Code a resumed run must share with the original (train_encoder.py itself may have gained options since)
RESUME_SAME_CODE = tuple(path for path in HASHED_CODE if path != "train_encoder.py")


def wandb_run_id(name: str) -> str | None:
    """The id of the latest local W&B run whose console log starts with this run's name line, or None."""
    for folder in sorted((PROJECT_ROOT / "wandb").glob("run-*"), reverse=True):
        log = folder / "files" / "output.log"
        if log.is_file():
            with open(log, errors="replace") as f:
                if any(line.startswith(f"{name}: ") for _, line in zip(range(200), f)):
                    return folder.name.rsplit("-", 1)[-1]
    return None


def finish_wandb_run(name: str, summary: dict, mode: str | None) -> str:
    """Add the summary of a finished run to its W&B run (resumed by id) and mark it finished; returns what happened."""
    import wandb

    if mode == "disabled":
        return "W&B disabled"
    run_id = wandb_run_id(name)
    if run_id is None:
        return f"no local W&B run found for {name}"
    run = wandb.init(project=WANDB_PROJECT, id=run_id, resume="must", mode=mode, dir=str(PROJECT_ROOT))
    run.summary.update(summary)
    run.finish()
    return f"W&B run {run_id} finished"


def stopped_run_state(run_dir: Path, last_name: str, best_name: str) -> tuple[dict, list[dict], dict, dict, dict]:
    """settings, metrics rows, best row and the last and best checkpoints of a run stopped by hand, checked to agree."""
    import torch as th

    if (run_dir / "result.json").exists():
        raise ValueError(f"{run_dir} has already finished (result.json)")
    settings = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
    last = th.load(run_dir / last_name, weights_only=False)
    best = th.load(run_dir / best_name, weights_only=False)
    best_row = min(rows, key=lambda r: r["validation"]["mse"])
    if last["epoch"] != len(rows) or best["epoch"] != best_row["epoch"]:
        raise ValueError(f"{run_dir}: checkpoints (last epoch {last['epoch']}, best {best['epoch']}) don't match "
                         f"metrics.jsonl ({len(rows)} epochs, best {best_row['epoch']})")
    return settings, rows, best_row, last, best


def finalize_run(run_dir: Path, wandb_mode: str | None = None) -> dict:
    """What a run's natural end does, for a run stopped by hand: result.json from the best checkpoint, encoder_model.zip
    and the W&B summary. The best checkpoint must reproduce its recorded validation error."""
    settings, rows, best_row, last, best = stopped_run_state(run_dir, "last_encoder.pt", "best_encoder.pt")
    cfg = encoder_config([f"training.seed={settings['seed']}"])
    flights, _ = load(Path(settings["data"]), settings["mix"], settings["max_flights"])
    rows_split = {split: flights.indices(split) for split in ("train", "validation", "test")}
    clock = clock_estimate(flights, rows_split["train"])
    steps_at, b0_head = settings["steps_at"], settings["b0_head"]
    arch = arch_settings(settings)
    encoder = make_model(cfg, arch, b0_head)
    encoder.load_state_dict(best["state_dict"])
    validation = evaluate(encoder, flights, rows_split["validation"], b0_head, steps_at)
    if not np.isclose(validation["rmse"], best_row["validation"]["rmse"], rtol=0, atol=1e-5):
        raise ValueError(f"{run_dir}: the best checkpoint gives validation {validation['rmse']:.6f}, metrics.jsonl "
                         f"{best_row['validation']['rmse']:.6f}")
    # This session's epochs: after the resume point for a resumed run (as a resumed run's natural end counts them)
    first = settings.get("resumed_from", {}).get("epochs", 0)
    result = {
        "name": settings["name"], "group": settings["group"], "best_epoch": best_row["epoch"], "epochs": len(rows),
        "steps": last["step"], "hours": sum(r["epoch_seconds"] for r in rows[first:]) / 3600,
        "validation": validation, "test": evaluate(encoder, flights, rows_split["test"], b0_head, steps_at),
        "clock_validation": evaluate_clock(clock, flights, rows_split["validation"], steps_at),
        "clock_test": evaluate_clock(clock, flights, rows_split["test"], steps_at),
        "test_seeds": [int(flights.seeds[i]) for i in rows_split["test"]],
        "stopped_by_hand": True, "finalized": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_trained(cfg, encoder, arch, run_dir, settings["name"])
    result["wandb"] = finish_wandb_run(settings["name"], {
        "best_epoch": result["best_epoch"], "test/b_rmse": result["test"]["rmse"], "stopped_by_hand": True,
        **{f"test/b_rmse_at_{k}": v for k, v in result["test"]["rmse_at"].items()},
        "clock_test/b_rmse": result["clock_test"]["rmse"]}, wandb_mode)
    return result


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", choices=sorted(DESIGNS), help="Required unless --resume")
    parser.add_argument("--seed", type=int, help="Initial weights and batch order (not the split); required unless --resume")
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
    parser.add_argument("--arch", choices=ARCHS, default="xlstm", help="A's xLSTM, or recurrent_encoders.py's GRU / ConvGRU")
    parser.add_argument("--drop-tau", action="store_true", help="gru / convgru: the encoder ignores tau (time left)")
    parser.add_argument("--hidden", type=int, default=ARCH_DEFAULTS["hidden"], help="gru / convgru: GRU width")
    parser.add_argument("--layers", type=int, default=ARCH_DEFAULTS["layers"], help="gru / convgru: GRU layers")
    parser.add_argument("--head-arch", type=int, nargs="+", default=ARCH_DEFAULTS["head_arch"], help="gru / convgru: head layers")
    parser.add_argument("--conv-channels", type=int, default=ARCH_DEFAULTS["conv_channels"], help="convgru: filter channels")
    parser.add_argument("--conv-window", type=int, default=ARCH_DEFAULTS["conv_window"], help="convgru: filter window (frames)")
    parser.add_argument("--log-every", type=int, default=50, help="Gradient steps between training-loss logs")
    parser.add_argument("--max-flights", type=int, default=None, help="Only the first flights of each split")
    parser.add_argument("--wandb-mode", default=None, help="online, offline or disabled (default: WANDB_MODE)")
    parser.add_argument("--smoke", action="store_true", help="48 flights per split, 2 epochs, W&B disabled")
    parser.add_argument("--resume", type=Path, default=None, help="A finished run directory to continue (see above)")
    parser.add_argument("--finalize", type=Path, default=None, help="A run directory stopped by hand to finish (see above)")
    args = parser.parse_args(argv)
    if args.finalize is not None:
        import torch as th

        th.set_num_threads(args.threads)
        try:
            result = finalize_run(args.finalize, args.wandb_mode)
        except ValueError as error:
            parser.error(str(error))
        print(f"{result['name']}: finalized, best epoch {result['best_epoch']} of {result['epochs']}, test b_t rmse "
              f"{result['test']['rmse']:.4f} (clock only {result['clock_test']['rmse']:.4f}); {result['wandb']}", flush=True)
        return
    previous = history = None
    if args.resume is not None:
        if args.smoke:
            parser.error("--resume and --smoke exclude each other")
        if not (args.resume / "result.json").is_file():
            parser.error(f"{args.resume} has not finished (no result.json): wait for it, or it failed")
        previous = json.loads((args.resume / "config.json").read_text())
        history = [json.loads(line) for line in open(args.resume / "metrics.jsonl")]
        previous_values = {**{key: previous.get(key) for key in RESUMED_SETTINGS}, **arch_settings(previous)}
        for key in RESUMED_SETTINGS:
            option = key.replace("_", "-")
            value, before = getattr(args, key), previous_values[key]
            if value is not None and value != parser.get_default(key) and str(value) != str(before):
                parser.error(f"--{option} {value}: --resume continues with the original run's --{option} ({before})")
            setattr(args, key, Path(before) if key == "data" else before)
        if args.b0_weight is None:
            args.b0_weight = parser.get_default("b0_weight")
        if args.max_epochs <= len(history):
            parser.error(f"--max-epochs {args.max_epochs}: the run already did {len(history)} epochs")
        best_row = min(history, key=lambda r: r["validation"]["mse"])
        if len(history) - best_row["epoch"] >= args.patience:
            parser.error(f"Under --patience {args.patience} the run would already have stopped (best epoch {best_row['epoch']} "
                         f"of {len(history)}): give more patience")
    elif args.design is None or args.seed is None:
        parser.error("--design and --seed are required unless --resume")
    sizes = ("hidden", "layers", "head_arch", "conv_channels", "conv_window")
    if args.arch == "xlstm" and (args.drop_tau or any(getattr(args, k) not in (None, parser.get_default(k)) for k in sizes)):
        parser.error("--drop-tau, --hidden, --layers, --head-arch and --conv-* are for --arch gru / convgru (the xLSTM is A's)")
    if args.arch != "xlstm" and min([args.hidden, args.layers, *args.head_arch]) < 1:
        parser.error("--hidden, --layers and --head-arch must be at least 1")
    if args.arch == "convgru" and (args.conv_channels < 1 or args.conv_window < 2):
        parser.error("--conv-channels must be at least 1 and --conv-window at least 2")
    arch = arch_settings(vars(args)) if args.arch != "xlstm" else dict(XLSTM_ARCH)
    if arch["arch"] == "gru":
        arch.update(conv_channels=None, conv_window=None)
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
    group = f"{args.design}{arch_tag(arch)}_{design['label']}_{args.mix}" + ("_resumed" if previous else "")
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
    encoder = make_model(cfg, arch, b0_head)
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
        **arch,
        "encoder": ({"xlstm": OmegaConf.to_container(cfg.cppo.xlstm, resolve=True), "context_net_arch": list(cfg.cppo.context_net_arch),
                     "activation_fn": str(cfg.cppo.activation_fn), "context_activation": cfg.cppo.context_activation,
                     "b0_net_arch": list(cfg.cppo.velocity_head.net_arch) if b0_head else None}
                    if arch["arch"] == "xlstm" else {"kwargs": encoder.kwargs})
                   | {"parameters": int(sum(p.numel() for p in encoder.parameters()))},
        "code_sha256": {path: sha256(PROJECT_ROOT / path) for path in HASHED_CODE},
        "data_sha256": {"manifest.json": sha256(args.data / "manifest.json"),
                        **{f"{s}.npz": sha256(args.data / f"{s}.npz") for s in pilots.MIXES[args.mix]}},
        "git": git_state(), "versions": {pkg: version(pkg) for pkg in ("torch", "xlstm", "numpy", "stable-baselines3")},
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    if previous:
        # Only the files the original run recorded (recurrent_encoders.py came later; the xLSTM doesn't use it)
        changed = [path for path in RESUME_SAME_CODE
                   if path in previous["code_sha256"] and previous["code_sha256"][path] != settings["code_sha256"][path]]
        if changed or previous["data_sha256"] != settings["data_sha256"]:
            run_dir.rmdir()
            parser.error(f"{args.resume} was trained with other {'code: ' + ', '.join(changed) if changed else 'data'}")
        settings["resumed_from"] = {"run_dir": str(args.resume), "name": previous["name"], "epochs": len(history),
                                    "hours": json.loads((args.resume / "result.json").read_text())["hours"],
                                    "code_sha256": previous["code_sha256"]}
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    run = wandb.init(project=WANDB_PROJECT, name=name, group=group, tags=["step4", args.design, args.mix, arch["arch"]] + (["notau"] if arch["drop_tau"] else []) + (["resumed"] if previous else []),
                     config=json.loads((run_dir / "config.json").read_text()), mode=args.wandb_mode, dir=str(PROJECT_ROOT))
    print(f"{name}: {settings['split']['flights']} flights, {settings['encoder']['parameters']:,} parameters, "
          f"run directory {run_dir}", flush=True)

    best, best_epoch, stale, step, first_epoch = float("inf"), 0, 0, 0, 1
    if previous:
        last = th.load(args.resume / "last_encoder.pt", weights_only=False)
        if last["epoch"] != len(history):
            parser.error(f"{args.resume}: last_encoder.pt is from epoch {last['epoch']}, metrics.jsonl has {len(history)}")
        encoder.load_state_dict(last["state_dict"])
        optimizer.load_state_dict(last["optimizer"])
        # The batch order of the epochs already done, drawn again so the next epoch's is the uninterrupted run's
        for _ in range(last["epoch"]):
            length_batches(rows["train"], flights.lengths, args.batch_size, rng)
        best_row = min(history, key=lambda r: r["validation"]["mse"])
        best, best_epoch, step, first_epoch = best_row["validation"]["mse"], best_row["epoch"], last["step"], last["epoch"] + 1
        stale = len(history) - best_epoch
        shutil.copyfile(args.resume / "best_encoder.pt", run_dir / "best_encoder.pt")
        for row in history:
            run.log(epoch_log(row, b0_head), step=row["step"])
        print(f"Resuming {previous['name']} after epoch {last['epoch']} (best epoch {best_epoch}), patience {args.patience}, "
              f"up to epoch {args.max_epochs}", flush=True)
    started = time.time()
    with open(run_dir / "metrics.jsonl", "w") as metrics:
        for row in history or []:
            metrics.write(json.dumps(row) + "\n")
        for epoch in range(first_epoch, args.max_epochs + 1):
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
            run.log(epoch_log(row, b0_head), step=step)
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
    save_trained(cfg, encoder, arch, run_dir, name)
    run.summary.update({"best_epoch": best_epoch, "test/b_rmse": result["test"]["rmse"],
                        **{f"test/b_rmse_at_{k}": v for k, v in result["test"]["rmse_at"].items()},
                        "clock_test/b_rmse": result["clock_test"]["rmse"]})
    run.finish()
    print(f"{name}: best epoch {best_epoch}, test b_t rmse {result['test']['rmse']:.4f} (clock only "
          f"{result['clock_test']['rmse']:.4f}), {result['hours']:.2f} h", flush=True)


if __name__ == "__main__":
    main()
