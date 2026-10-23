"""BatteryPlane flight diagnostics for W&B, shared by the CPPO and plain-PPO training scripts.

Step 0 of ENCODER_PLAN.md compares CPPO (0c) with plain PPO (0a, 0b), so both scripts log the same flight metrics
through these algorithm-agnostic SB3 callbacks. They read the env infos (b0, b, y, vx, saturated, mission_complete)
and the executed actions: privileged diagnostics, never model inputs. Scalars go through the SB3 logger
(TensorBoard, which W&B syncs), under the names of the A-F runs where those exist.

- FlightLoggingEvalCallback: the flights of every evaluation (validation/*), logged with eval/mean_reward, and an
  episode plot every plot_every environment steps (episode_plots/validation)
- RolloutFlightLogger: the last 100 training flights (rollout/*), logged with every rollout
- GradNormLogger: the actor/critic gradient norm before clipping, per update (train/grad_norm*)
- FlightRecorder / evaluate_flights / log_final_flights: the same flight metrics for the final validation

Per flight, from the true state in the infos:
    outcome          exclusive: battery_empty (b <= 0) before ground_crash (y <= 0), then completed (t == T)
    time_in_band     share of steps with speed >= v_min (s_band of BatteryPlaneVec.value_of_context)
    time_at_goal     share of steps with speed >= the speed of 90% of the main reward (its s_goal)
    saturation_rate  share of steps whose cruise command exceeded the battery's thrust cap (info["saturated"])
    distance_m       sum of speed x dt, the env's own update of x
    throttle / lift  mean executed action a_x / a_y; by_b0/*_range is its spread over the starting charges
                     (0: the policy flies the same whatever its battery)
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Image

# Outcome -> name of its rate (the A-F names)
OUTCOME_RATES = {"completed": "completion_rate", "ground_crash": "ground_crash_rate",
                 "battery_empty": "battery_depleted_rate", "truncated": "truncated_rate"}
# Logged per starting charge (validation/by_b0/b0_<pct>pct/*)
BY_B0_METRICS = ("return_mean", "completion_rate", "ground_crash_rate", "battery_depleted_rate", "time_in_band",
                 "saturation_rate", "altitude_mean", "throttle_mean", "lift_mean", "battery_used_mean")
# Logged over the recent training flights (rollout/*); SB3 already logs ep_rew_mean and ep_len_mean
ROLLOUT_METRICS = ("completion_rate", "ground_crash_rate", "battery_depleted_rate", "final_battery_completed_mean",
                   "time_in_band", "time_at_goal", "saturation_rate", "speed_mean", "altitude_mean", "throttle_mean",
                   "battery_used_mean")
# Output formats that skip a key: W&B (through TensorBoard) still gets it, the console table doesn't
NOT_ON_CONSOLE = ("stdout", "log", "json", "csv")


def terminal_outcome(info: Mapping[str, Any]) -> str:
    """One outcome per flight; battery depletion takes precedence on simultaneous failures, as in the env UI."""
    if float(info["b"]) <= 0:
        return "battery_empty"
    if float(info["y"]) <= 0:
        return "ground_crash"
    if info["mission_complete"]:
        return "completed"
    if info.get("TimeLimit.truncated", False):
        return "truncated"
    raise ValueError("Evaluation episode ended without a recognized BatteryPlane outcome")


class Flight:
    """
    Running sums of one flight, from the info and the executed action of each step.

    :param core: The env's BatteryPlaneVec (v_min, v_goal, dt)
    :param keep_trace: Also keep (time, altitude, speed, charge, throttle, lift) of every step, for plots
    """

    def __init__(self, core: Any, keep_trace: bool = False) -> None:
        self.v_min = float(core.v_min)
        self.v_goal_90 = float(core.v_min + (core.v_goal - core.v_min) * math.sqrt(0.9))
        self.dt = float(core.dt)
        self.trace: list[tuple[float, ...]] | None = [] if keep_trace else None
        self.steps = 0
        self.reward = self.in_band = self.at_goal = self.saturated = 0.0
        self.speed = self.altitude = self.throttle = self.lift = self.abs_action = self.action_change = 0.0
        self.altitude_min = math.inf
        self.previous: tuple[float, float] | None = None
        self.b0 = self.b = math.nan

    def add(self, info: Mapping[str, Any], reward: float, action: Sequence[float]) -> None:
        if info.get("autopilot"):
            return   # a replayed take-off step (memory_baselines.RecurrentInput): not part of the mission
        vx, y = float(info["vx"]), float(info["y"])
        throttle, lift = float(action[0]), float(action[1])
        self.steps += 1
        self.reward += float(reward)
        self.in_band += vx >= self.v_min
        self.at_goal += vx >= self.v_goal_90
        self.saturated += bool(info["saturated"])
        self.speed += vx
        self.altitude += y
        self.altitude_min = min(self.altitude_min, y)
        self.throttle += throttle
        self.lift += lift
        self.abs_action += (abs(throttle) + abs(lift)) / 2
        if self.previous is not None:
            self.action_change += (abs(throttle - self.previous[0]) + abs(lift - self.previous[1])) / 2
        self.previous = (throttle, lift)
        self.b0, self.b = float(info["b0"]), float(info["b"])
        if self.trace is not None:
            self.trace.append((self.steps * self.dt, y, vx, self.b, throttle, lift))

    def finish(self, info: Mapping[str, Any]) -> dict[str, Any]:
        """The flight's metrics, given the info of its last step."""
        n = self.steps
        return {
            "b0": self.b0, "outcome": terminal_outcome(info), "return": self.reward, "length": n,
            "time_in_band": self.in_band / n, "time_at_goal": self.at_goal / n, "saturation_rate": self.saturated / n,
            "speed_mean": self.speed / n, "altitude_mean": self.altitude / n, "altitude_min": self.altitude_min,
            "distance_m": self.speed * self.dt, "battery_used": self.b0 - self.b, "final_battery": self.b,
            "throttle_mean": self.throttle / n, "lift_mean": self.lift / n, "mean_abs_action": self.abs_action / n,
            "mean_action_change": self.action_change / max(n - 1, 1),
        }


def summarize(flights: Iterable[Mapping[str, Any]], by_b0: bool = True) -> dict[str, float]:
    """
    Metrics of a set of flights (finite floats only, so they also fit strict JSON), and with ``by_b0`` the same
    per starting charge (``by_b0/b0_<pct>pct/<metric>``) plus the spread of throttle and lift over the charges.
    """
    flights = list(flights)

    def mean(key: str, group: list) -> float:
        return float(np.mean([flight[key] for flight in group]))

    def describe(group: list) -> dict[str, float]:
        out = {"return_mean": mean("return", group), "return_std": float(np.std([f["return"] for f in group])),
               "length_mean": mean("length", group)}
        for outcome, name in OUTCOME_RATES.items():
            out[name] = sum(flight["outcome"] == outcome for flight in group) / len(group)
        completed = [flight for flight in group if flight["outcome"] == "completed"]
        if completed:
            out["final_battery_completed_mean"] = mean("final_battery", completed)
        for key in ("time_in_band", "time_at_goal", "saturation_rate", "speed_mean", "altitude_mean", "altitude_min",
                    "throttle_mean", "lift_mean", "mean_abs_action", "mean_action_change"):
            out[key] = mean(key, group)
        out["distance_mean_m"] = mean("distance_m", group)
        out["battery_used_mean"] = mean("battery_used", group)
        return out

    metrics = describe(flights)
    if by_b0:
        groups: dict[float, list] = {}
        for flight in flights:
            groups.setdefault(flight["b0"], []).append(flight)
        per_charge = {b0: describe(group) for b0, group in sorted(groups.items())}
        for b0, values in per_charge.items():
            for key in BY_B0_METRICS:
                metrics[f"by_b0/b0_{100 * b0:g}pct/{key}"] = values[key]
        if len(per_charge) > 1:
            for key in ("throttle_mean", "lift_mean"):
                values = [charge[key] for charge in per_charge.values()]
                metrics[f"by_b0/{key}_range"] = max(values) - min(values)
    return metrics


def plot_flights(traces: Mapping[float, tuple[list, str]], core: Any, title: str) -> np.ndarray:
    """Altitude, speed (with the reward band), charge and actions over time, one colour per starting charge."""
    # No pyplot: no global figure state, no GUI backend
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 8), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(4, 1, sharex=True)
    for b0, (trace, outcome) in sorted(traces.items()):
        time_s, altitude, speed, charge, throttle, lift = np.asarray(trace).T
        (line,) = axes[0].plot(time_s, altitude, label=f"b0 = {b0:g} ({outcome.replace('_', ' ')})")
        color = line.get_color()
        axes[1].plot(time_s, speed, color=color)
        axes[2].plot(time_s, charge, color=color)
        axes[3].plot(time_s, throttle, color=color)
        axes[3].plot(time_s, lift, color=color, linestyle=":")
    for speed in (core.v_min, core.v_goal):
        axes[1].axhline(speed, color="grey", linestyle="--", linewidth=0.8)
    for ax, label in zip(axes, ("altitude (m)", "speed (m/s)", "battery", "throttle -, lift :"), strict=True):
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


class FlightRecorder:
    """
    Every flight of an ``evaluate_policy`` run, through its callback (``record``), on a single evaluation env.

    :param core: The env's BatteryPlaneVec
    :param plot_batteries: Starting charges whose first flight is kept for ``plot``
    """

    def __init__(self, core: Any, plot_batteries: Iterable[float] = ()) -> None:
        self.core = core
        self.plot_batteries = [float(b0) for b0 in plot_batteries]
        self.flights: list[dict[str, Any]] = []
        self.traces: dict[float, tuple[list, str]] = {}
        self._current: Flight | None = None

    def record(self, locals_: dict[str, Any], globals_: dict[str, Any] | None = None) -> None:
        """``evaluate_policy`` callback, called at every step of each counted episode."""
        if locals_["n_envs"] != 1:
            raise ValueError("FlightRecorder needs a single evaluation environment")
        if self._current is None:
            self._current = Flight(self.core, keep_trace=bool(self.plot_batteries))
        info = locals_["info"]
        self._current.add(info, locals_["reward"], locals_["actions"][locals_["i"]])
        if locals_["done"]:
            flight = self._current.finish(info)
            self.flights.append(flight)
            b0 = next((b0 for b0 in self.plot_batteries if math.isclose(b0, flight["b0"]) and b0 not in self.traces), None)
            if b0 is not None:
                self.traces[b0] = (self._current.trace, flight["outcome"])
            self._current = None

    def metrics(self) -> dict[str, float]:
        return summarize(self.flights)

    def plot(self, title: str) -> np.ndarray | None:
        return plot_flights(self.traces, self.core, title) if self.traces else None


def evaluate_flights(
    model: Any, env: Any, n_episodes: int, deterministic: bool, core: Any, plot_batteries: Iterable[float] = ()
) -> tuple[list[float], list[int], FlightRecorder]:
    """``evaluate_policy`` on a single-env VecEnv, with every flight recorded."""
    recorder = FlightRecorder(core, plot_batteries)
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=deterministic,
                                       return_episode_rewards=True, callback=recorder.record)
    return rewards, lengths, recorder  # type: ignore[return-value]


def log_final_flights(run: Any, recorder: FlightRecorder, timesteps: int) -> None:
    """The final validation in the W&B summary (final_validation/*) and its episode plot."""
    import wandb

    for key, value in recorder.metrics().items():
        run.summary[f"final_validation/{key}"] = value
    image = recorder.plot(f"Final validation flights ({timesteps:,} steps)")
    if image is not None:
        run.log({"episode_plots/final_validation": wandb.Image(image)})


class FlightLoggingEvalCallback(EvalCallback):
    """
    ``EvalCallback`` that also logs the flights of every evaluation (``validation/*``) in the same dump as
    ``eval/mean_reward``, and every ``plot_every`` environment steps an episode plot of the first flight at each
    charge in ``plot_batteries`` (``episode_plots/validation``). The evaluation env must be a single env.

    :param flight_core: The evaluation env's BatteryPlaneVec
    :param plot_every: Environment steps between episode plots (a multiple of the evaluation frequency), None for none
    :param plot_batteries: Starting charges to plot
    """

    def __init__(self, *args: Any, flight_core: Any, plot_every: int | None = None,
                 plot_batteries: Iterable[float] = (), **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.flight_core = flight_core
        self.plot_every = plot_every
        self.plot_batteries = tuple(plot_batteries)
        self._recorder: FlightRecorder | None = None

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            plot = bool(self.plot_every) and self.num_timesteps % self.plot_every == 0  # type: ignore[operator]
            self._recorder = FlightRecorder(self.flight_core, self.plot_batteries if plot else ())
        return super()._on_step()

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)
        recorder = self._recorder
        if recorder is None:
            return
        recorder.record(locals_, globals_)
        # The last flight ends before EvalCallback records eval/mean_reward and dumps: same step, same dump
        if locals_["done"] and len(recorder.flights) == self.n_eval_episodes:
            self._recorder = None
            for key, value in recorder.metrics().items():
                self.logger.record(f"validation/{key}", value, exclude=NOT_ON_CONSOLE if "/" in key else None)
            image = recorder.plot(f"Validation flights at {self.num_timesteps:,} steps")
            if image is not None:
                self.logger.record("episode_plots/validation", Image(image, "HWC"), exclude=NOT_ON_CONSOLE)


class RolloutFlightLogger(BaseCallback):
    """
    ``rollout/*`` flight metrics over the last ``window`` training flights (SB3's ``ep_rew_mean`` window), logged
    with every rollout, from the step infos and executed actions of every training env.

    :param flight_core: A BatteryPlaneVec with the training physics (v_min, v_goal, dt)
    :param window: Number of recent flights
    """

    def __init__(self, flight_core: Any, window: int = 100) -> None:
        super().__init__()
        self.flight_core = flight_core
        self.window = window

    def _on_training_start(self) -> None:
        self._flights = [Flight(self.flight_core) for _ in range(self.training_env.num_envs)]
        self._finished: deque[dict[str, Any]] = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        infos, dones = self.locals["infos"], self.locals["dones"]
        rewards, actions = self.locals["rewards"], self.locals["clipped_actions"]
        for i, info in enumerate(infos):
            self._flights[i].add(info, rewards[i], actions[i])
            if dones[i]:
                self._finished.append(self._flights[i].finish(info))
                self._flights[i] = Flight(self.flight_core)
        return True

    def _on_rollout_end(self) -> None:
        if self._finished:
            metrics = summarize(self._finished, by_b0=False)
            for key in ROLLOUT_METRICS:
                if key in metrics:
                    self.logger.record(f"rollout/{key}", metrics[key])


class GradNormLogger(BaseCallback):
    """
    Gradient norm of the actor and critic before clipping, over the optimizer steps of each update. PPO and
    ContextualPPO both clip them with ``torch.nn.utils.clip_grad_norm_`` and drop the norm it returns; this callback
    wraps that function only while the model trains (from the end of a rollout to the start of the next) and logs,
    with the update's other ``train/*`` metrics: ``train/grad_norm`` (mean), ``train/grad_norm_max``,
    ``train/grad_clipped_fraction`` (share of steps above ``max_grad_norm``) and ``train/optimizer_steps``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._original: Any = None
        self._norms: list[float] = []

    def _on_rollout_end(self) -> None:
        self._restore()
        original = torch.nn.utils.clip_grad_norm_
        norms: list[float] = []

        def recording(parameters: Any, max_norm: float, *args: Any, **kwargs: Any) -> torch.Tensor:
            total = original(parameters, max_norm, *args, **kwargs)
            norms.append(float(total))
            return total

        self._original, self._norms = original, norms
        torch.nn.utils.clip_grad_norm_ = recording

    def _on_rollout_start(self) -> None:
        self._restore()

    def _on_training_end(self) -> None:
        self._restore()

    def _on_step(self) -> bool:
        return True

    def _restore(self) -> None:
        if self._original is None:
            return
        torch.nn.utils.clip_grad_norm_ = self._original
        self._original = None
        norms = np.asarray(self._norms)
        self.logger.record("train/optimizer_steps", len(norms))
        if len(norms):
            self.logger.record("train/grad_norm", float(norms.mean()))
            self.logger.record("train/grad_norm_max", float(norms.max()))
            self.logger.record("train/grad_clipped_fraction", float((norms > self.model.max_grad_norm).mean()))
