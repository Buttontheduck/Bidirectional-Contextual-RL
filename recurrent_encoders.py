"""Step 4 of ENCODER_PLAN.md: standalone recurrent battery encoders (GRU, ConvGRU), for offline training
(train_encoder.py --arch gru | convgru) and for plain PPO through the context wrapper (context_wrapper.py, an encoder.pt).

Every encoder reads [o_t, a_{t-1}] (the hidden-mode observation [y, v_x + noise, v_y, tau] and the previous executed
action, zero at a flight's start) from a zero state at the start of every flight, and outputs the battery estimate
b_hat_t in (0, 1), plus an optional auxiliary output (E3's b0 head). It runs two ways, which agree within float rounding
(checks.step4_checks):
    forward(observations, prev_actions)        whole flights [B, L, .] from their first step, for training. Causal: the
                                               padding at the end of shorter flights never changes the real steps.
    step(state, observations, prev_actions)    one frame per row, the memory carried in state, for the context wrapper
drop_tau removes tau (o_t[3], time left) inside the encoder, so the encoder sees only y, v_x + noise, v_y and the action.
context_head="gaussian" (teacher T2 of ENCODER_PLAN.md) makes the head's last layer output two values, the estimate
(the mean, through the sigmoid) and its variance (softplus + 1e-6, as A's Gaussian head). forward and step still return
the estimate; forward_with_variance and step_with_variance return the variance too.

    GRUEncoder      input projection (in -> hidden), a GRU (layers x hidden), head (hidden -> head_arch -> 1, tanh,
                    sigmoid). With tau and A's head (128, 128) it is step4_probes.py's GRU, parameter names included,
                    so a probe checkpoint loads directly (export_probe).
    ConvGRUEncoder  a causal Conv1d over the last `window` frames (the inputs plus a valid channel, 1 for real frames
                    and 0 for the zero padding before a flight's first frame; SiLU), a per-step MLP (conv channels ->
                    hidden -> hidden, SiLU, LayerNorm), then the same GRU and head. In one-step mode the last window - 1
                    frames are kept in the state, zero (and not valid) at a flight's start, as in the padded full flight.

save_encoder / load_encoder: encoder.pt = {"arch", "kwargs", "state_dict", "info"}. Project glue: nothing here imports
battery_plane or changes contextual_ppo.

    python recurrent_encoders.py --export-probe runs_encoder/step4_probes/probe_gru_battery_main_s1 --output <encoder.pt>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as th
from torch import nn
from torch.nn import functional as F

OBSERVATION_DIM, ACTION_DIM = 4, 2
TAU_INDEX = 3
DEFAULT_HEAD_ARCH = (128, 128)        # A's context head (config/cppo.yaml cppo.context_net_arch), tanh
CONTEXT_HEADS = ("point", "gaussian")
MIN_VARIANCE = 1e-6                   # A's Gaussian head (contextual_ppo.torch_layers.MIN_CONTEXT_VARIANCE)


def mlp_head(width: int, head_arch, sigmoid: bool, outputs: int = 1) -> nn.Sequential:
    """width -> head_arch -> outputs with tanh between layers (stable_baselines3's create_mlp layout), then a sigmoid."""
    layers, last = [], width
    for size in head_arch:
        layers += [nn.Linear(last, size), nn.Tanh()]
        last = size
    layers.append(nn.Linear(last, outputs))
    if sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class RecurrentBatteryEncoder(nn.Module):
    """The shared interface (see the module docstring). Subclasses build `features` steps and keep their state in a dict."""

    ARCH = ""

    def __init__(self, drop_tau: bool = False, hidden: int = 128, layers: int = 2, head_arch=DEFAULT_HEAD_ARCH,
                 aux_dim: int | None = None, context_head: str = "point") -> None:
        super().__init__()
        if aux_dim not in (None, 1):
            raise ValueError("aux_dim must be None or 1")
        if context_head not in CONTEXT_HEADS:
            raise ValueError(f"context_head must be one of {CONTEXT_HEADS}, got {context_head!r}")
        self.observation_dim, self.action_dim = OBSERVATION_DIM, ACTION_DIM
        self.drop_tau, self.hidden, self.layers = bool(drop_tau), int(hidden), int(layers)
        self.input_dim = OBSERVATION_DIM - int(self.drop_tau) + ACTION_DIM
        self.context_head = context_head
        self.rnn = nn.GRU(self.hidden, self.hidden, num_layers=self.layers, batch_first=True)
        # Gaussian: the last layer outputs the mean and the variance, each through its own activation (heads)
        self.head = (mlp_head(self.hidden, head_arch, sigmoid=True) if context_head == "point"
                     else mlp_head(self.hidden, head_arch, sigmoid=False, outputs=2))
        self.aux_head = None if aux_dim is None else mlp_head(self.hidden, head_arch, sigmoid=False)
        self.kwargs = {"drop_tau": self.drop_tau, "hidden": self.hidden, "layers": self.layers,
                       "head_arch": [int(x) for x in head_arch], "aux_dim": aux_dim, "context_head": context_head}

    def inputs(self, observations: th.Tensor, prev_actions: th.Tensor) -> th.Tensor:
        """[..., 4] and [..., 2] -> [..., input_dim]: tau dropped if drop_tau."""
        if observations.shape[-1] != OBSERVATION_DIM or prev_actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected hidden-mode observations ({OBSERVATION_DIM}) and actions ({ACTION_DIM}), got "
                             f"{observations.shape[-1]} and {prev_actions.shape[-1]}")
        if self.drop_tau:
            observations = observations[..., :TAU_INDEX]
        return th.cat([observations, prev_actions], dim=-1)

    def heads(self, memory: th.Tensor) -> tuple[th.Tensor, th.Tensor | None, th.Tensor | None]:
        """The estimate, its variance (Gaussian head; else None) and the auxiliary output (else None)."""
        out = self.head(memory)
        if self.context_head == "point":
            estimate, variance = out[..., 0], None
        else:
            estimate, variance = th.sigmoid(out[..., 0]), F.softplus(out[..., 1]) + MIN_VARIANCE
        return estimate, variance, None if self.aux_head is None else self.aux_head(memory)[..., 0]

    def outputs(self, memory: th.Tensor) -> tuple[th.Tensor, th.Tensor | None]:
        estimate, _, aux = self.heads(memory)
        return estimate, aux

    def sequence_memory(self, observations: th.Tensor, prev_actions: th.Tensor) -> th.Tensor:
        memory, _ = self.rnn(self.sequence_features(self.inputs(observations, prev_actions)))
        return memory

    def step_memory(self, state: dict, observations: th.Tensor, prev_actions: th.Tensor) -> tuple[th.Tensor, dict]:
        features, new_state = self.step_features(state, self.inputs(observations, prev_actions))
        memory, h = self.rnn(features[:, None, :], state["h"])
        return memory[:, 0], {**new_state, "h": h}

    def forward(self, observations: th.Tensor, prev_actions: th.Tensor) -> tuple[th.Tensor, th.Tensor | None]:
        """Whole flights from their first step: [B, L, 4], [B, L, 2] -> estimates [B, L] and auxiliary outputs [B, L]."""
        return self.outputs(self.sequence_memory(observations, prev_actions))

    def forward_with_variance(self, observations: th.Tensor, prev_actions: th.Tensor):
        """As forward, plus the variance [B, L] (None with the point head): estimates, variances, auxiliary outputs."""
        return self.heads(self.sequence_memory(observations, prev_actions))

    def step(self, state: dict, observations: th.Tensor, prev_actions: th.Tensor) -> tuple[th.Tensor, th.Tensor | None, dict]:
        """One frame per row: [B, 4], [B, 2] -> estimates [B], auxiliary outputs [B] and the new state."""
        memory, new_state = self.step_memory(state, observations, prev_actions)
        return (*self.outputs(memory), new_state)

    def step_with_variance(self, state: dict, observations: th.Tensor, prev_actions: th.Tensor):
        """As step, plus the variance [B] (None with the point head): estimates, variances, auxiliary outputs, state."""
        memory, new_state = self.step_memory(state, observations, prev_actions)
        return (*self.heads(memory), new_state)

    def initial_state(self, batch: int) -> dict:
        param = next(self.parameters())
        return {"h": param.new_zeros(self.layers, batch, self.hidden)}

    @staticmethod
    def _batch_mask(mask: th.Tensor, value: th.Tensor, key: str) -> th.Tensor:
        # h: [layers, B, hidden]; window: [B, channels, frames]
        return mask[None, :, None] if key == "h" else mask[:, None, None]

    def select_state(self, mask, state_if_true: dict, state_if_false: dict) -> dict:
        """Per row: state_if_true where mask, else state_if_false."""
        mask = th.as_tensor(mask, dtype=th.bool)
        return {key: th.where(self._batch_mask(mask, value, key), value, state_if_false[key])
                for key, value in state_if_true.items()}

    def reset_state(self, state: dict, mask) -> dict:
        """The rows in mask back to the initial state (an episode starts)."""
        mask = th.as_tensor(mask, dtype=th.bool)
        return self.select_state(mask, {key: th.zeros_like(value) for key, value in state.items()}, state)

    def sequence_features(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def step_features(self, state: dict, x: th.Tensor) -> tuple[th.Tensor, dict]:
        raise NotImplementedError


class GRUEncoder(RecurrentBatteryEncoder):
    ARCH = "gru"

    def __init__(self, drop_tau: bool = False, hidden: int = 128, layers: int = 2, head_arch=DEFAULT_HEAD_ARCH,
                 aux_dim: int | None = None, context_head: str = "point") -> None:
        super().__init__(drop_tau, hidden, layers, head_arch, aux_dim, context_head)
        self.input_projection = nn.Linear(self.input_dim, self.hidden)

    def sequence_features(self, x: th.Tensor) -> th.Tensor:
        return self.input_projection(x)

    def step_features(self, state: dict, x: th.Tensor) -> tuple[th.Tensor, dict]:
        return self.input_projection(x), {}


class ConvGRUEncoder(RecurrentBatteryEncoder):
    ARCH = "convgru"

    def __init__(self, drop_tau: bool = False, hidden: int = 128, layers: int = 2, head_arch=DEFAULT_HEAD_ARCH,
                 aux_dim: int | None = None, conv_channels: int = 32, window: int = 16, context_head: str = "point") -> None:
        super().__init__(drop_tau, hidden, layers, head_arch, aux_dim, context_head)
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window, self.channels = int(window), self.input_dim + 1           # + the valid channel
        self.conv = nn.Conv1d(self.channels, int(conv_channels), kernel_size=self.window)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(int(conv_channels), self.hidden), nn.SiLU(),
                                 nn.Linear(self.hidden, self.hidden), nn.SiLU(), nn.LayerNorm(self.hidden))
        self.kwargs.update({"conv_channels": int(conv_channels), "window": self.window})

    @staticmethod
    def with_valid(x: th.Tensor) -> th.Tensor:
        return th.cat([x, th.ones_like(x[..., :1])], dim=-1)

    def sequence_features(self, x: th.Tensor) -> th.Tensor:
        # [B, L, C] -> [B, C, L], window - 1 zero frames (not valid) before the first: output t sees frames t-15 .. t
        frames = F.pad(self.with_valid(x).transpose(1, 2), (self.window - 1, 0))
        return self.mlp(self.conv(frames).transpose(1, 2))

    def step_features(self, state: dict, x: th.Tensor) -> tuple[th.Tensor, dict]:
        frames = th.cat([state["window"], self.with_valid(x)[:, :, None]], dim=2)      # [B, C, window]
        return self.mlp(self.conv(frames)[:, :, 0]), {"window": frames[:, :, 1:]}

    def initial_state(self, batch: int) -> dict:
        state = super().initial_state(batch)
        state["window"] = state["h"].new_zeros(batch, self.channels, self.window - 1)
        return state


ARCHITECTURES = {"gru": GRUEncoder, "convgru": ConvGRUEncoder}


def build_encoder(arch: str, **kwargs) -> RecurrentBatteryEncoder:
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {sorted(ARCHITECTURES)}, got {arch!r}")
    return ARCHITECTURES[arch](**kwargs)


def save_encoder(encoder: RecurrentBatteryEncoder, path: Path, info: dict | None = None) -> None:
    th.save({"arch": encoder.ARCH, "kwargs": encoder.kwargs, "state_dict": encoder.state_dict(), "info": info or {}}, path)


def load_encoder(path: Path) -> RecurrentBatteryEncoder:
    """An encoder.pt, on the CPU, in evaluation mode."""
    saved = th.load(path, map_location="cpu", weights_only=False)
    encoder = build_encoder(saved["arch"], **saved["kwargs"])
    encoder.load_state_dict(saved["state_dict"], strict=True)
    return encoder.eval()


def export_probe(run_dir: Path, output: Path) -> RecurrentBatteryEncoder:
    """A step4_probes.py GRU trained on the battery (its best checkpoint) as an encoder.pt."""
    settings = json.loads((run_dir / "config.json").read_text())
    if settings["arch"] != "gru" or settings["task"] != "battery":
        raise ValueError(f"{run_dir}: only a GRU probe of the battery task exports (arch {settings['arch']}, task {settings['task']})")
    encoder = GRUEncoder(drop_tau=False, hidden=settings["hidden"], layers=settings["layers"], head_arch=DEFAULT_HEAD_ARCH)
    best = th.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    encoder.load_state_dict(best["state_dict"], strict=True)
    result = json.loads((run_dir / "result.json").read_text()) if (run_dir / "result.json").is_file() else {}
    save_encoder(encoder, output, {"exported_from": str(run_dir), "name": settings["name"], "best_epoch": best["epoch"],
                                   "test_rmse": result.get("test", {}).get("rmse")})
    return encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-probe", type=Path, required=True, help="A step4_probes.py run directory (GRU, battery)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists")
    export_probe(args.export_probe, args.output)
    print(f"Exported {args.export_probe} to {args.output}")


if __name__ == "__main__":
    main()
