"""Step 3 of ENCODER_PLAN.md: pilots that fly BatteryPlane to make training flights for an offline battery encoder.

The A rerun's encoder learned a clock from the only flights it saw, its own hard flying. These pilots fly in many ways
so that the battery can't be predicted from the time alone, and so that the flights that matter (probing, then saving
energy; gentle flying on a low battery) are in the data. Every flight is recorded as an encoder reads it: the hidden-mode
observation o_t, the previous executed action a_{t-1} (zero at the start) and, as the training label only, the true
battery b_t.

What a pilot may read (no leak of the battery into its actions, which the encoder reads):
    scripted pilots   the hidden-mode observation [y, vx + noise, vy, tau], their own past actions and randomness, and
                      physics formulas (drag, the cheapest speed) that hold no plane state. They steer by a smoothed
                      noisy speed reading, never the true speed (which would pass a cleaner battery signal through the
                      actions than any agent has) and never the battery
    blind_ppo         memoryless PPO policies trained in hidden mode (0a_blind, 0b_blind, 0c_blind): battery-blind
    replay_good       good pilots (true battery: 0a, 0b, 0c; ideal estimate: 2a_ideal, 2b_ideal and their _std) fly a
                      shadow plane with its own random battery and start, and their actions fly the recorded plane: the
                      flying looks battery-aware, but the actions say nothing about the recorded plane's battery
    aware_good        the same good pilots flying the recorded plane itself. LEAKS: their actions depend on its true
                      battery (0a, 0b, 0c) or on a battery estimate (2a/2b_ideal), so an encoder could read the battery
                      from the actions. Only in the separate mix main_plus_aware, always checked on battery-blind pilots

Pilot styles (ENCODER_PLAN.md step 3, "keep / change / add"):
    probe_hold     full throttle for 1-10 s (a probe), then hold a speed below the cap, repeated 1-4 times
    steady_below   constant throttle or a held speed, always below the cap (it never reveals the battery: the encoder
                   must keep its best guess and count the drain)
    random_hold    random throttle (including reverse) held for random durations, random lift
    shaky          a new random action at every step
    wavy           sinusoidal throttle (low weight)
    hold_speed     hold a speed of 45-65 m/s (the reward band), steering by the noisy speed (changed from
                   battery_plane.policy_hold_goal, which steers by the true speed)
    cruise         hold the cheapest speed for the altitude, steering by the noisy speed (changed from policy_cruise)
    endurance      a probe at take-off, then gentle flying to the end (late-flight, low-battery data)
    full_throttle  full throttle all flight, climbing to the ceiling: A-like data, the narrow control only
    blind_ppo, replay_good, aware_good   see above
Every scripted pilot also draws an altitude programme per flight (level, climb, ceiling, changing targets, low), steered
by the exact observed altitude and climb rate, with a guard against hitting the ground.

Mixes (MIXES): narrow (full_throttle), main (all battery-blind styles) and main_plus_aware (main + aware_good).

    python pilots.py --output runs_encoder/pilot_flights/v1 --policy-reports <step 0 / memoryless / ideal reports>

Project glue: it imports battery_plane, like the training scripts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

import battery_plane as bp

# Throttle that never reaches the cruise motor's cap: 0.55 x a_max = 11 m/s^2 < h_min = 11.5 m/s^2, the cap of an empty
# battery (checked against the env in PlanePhysics)
BELOW_CAP_THROTTLE = 0.55
# Validation and final-test reset seeds: never used for a pilot flight
HELD_OUT_SEEDS = frozenset({1000, 1001, 1002, 2000, 2001, 2002})
# Seeds of the recorded flights: style k uses FIRST_SEED + k * SEED_STRIDE + i (unique per flight, never a held-out
# seed). Everything else random in a flight derives from its seed, so no two flights share noise: the pilot's own
# randomness (np.random.default_rng([flight seed, style, 17])), the k-th action-sampling seed of a stochastic policy
# pilot (sampling_seed) and the k-th shadow plane of a replayed pilot (shadow_seed), k < SEEDS_PER_FLIGHT
FIRST_SEED = 100_000
SEED_STRIDE = 1_000_000
SEEDS_PER_FLIGHT = 64
SHADOW_SEED_BASE = 1_000_000_000

def sampling_seed(flight_seed: int, k: int) -> int:
    """Torch seed of the k-th start of a stochastic policy pilot in a flight: distinct for every flight and k."""
    if not 0 <= k < SEEDS_PER_FLIGHT:
        raise ValueError(f"At most {SEEDS_PER_FLIGHT} policy starts per flight")
    return flight_seed * SEEDS_PER_FLIGHT + k


SEGMENT_SEED_BASE = 10_000_000_000


def segment_seed(flight_seed: int, j: int) -> int:
    """
    Seed of the j-th segment of a switching flight: the segment's pilot draws its randomness, action-sampling seeds and
    shadow planes from it. Distinct for every flight and j, and above every flight and shadow seed.
    """
    if not 0 <= j < SEEDS_PER_FLIGHT:
        raise ValueError(f"At most {SEEDS_PER_FLIGHT} segments per flight")
    return SEGMENT_SEED_BASE + flight_seed * SEEDS_PER_FLIGHT + j


def shadow_seed(flight_seed: int, k: int) -> int:
    """Reset seed of the k-th shadow plane of a replayed flight: distinct for every flight and k, above every flight seed."""
    if not 0 <= k < SEEDS_PER_FLIGHT:
        raise ValueError(f"At most {SEEDS_PER_FLIGHT} shadow flights per flight")
    return SHADOW_SEED_BASE + flight_seed * SEEDS_PER_FLIGHT + k


SCRIPTED_STYLES = ("probe_hold", "steady_below", "random_hold", "shaky", "wavy", "hold_speed", "cruise", "endurance",
                   "full_throttle")
POLICY_STYLES = ("blind_ppo", "replay_good", "aware_good")
# Styles whose actions depend on the recorded plane's battery or on a battery estimate: never in the main mixes
LEAKING_STYLES = frozenset({"aware_good"})
# switching comes last, so adding it left every other style's seeds unchanged
STYLES = SCRIPTED_STYLES + POLICY_STYLES + ("switching",)
# Styles a switching flight changes between (the policy ones only with policies); never a leaking style
SWITCH_STYLES = ("probe_hold", "steady_below", "random_hold", "shaky", "wavy", "hold_speed", "cruise", "endurance",
                 "full_throttle", "blind_ppo", "replay_good")
# For the check that switching never ties gentle flying to a low battery
GENTLE_STYLES, HARD_STYLES = frozenset({"steady_below", "cruise"}), frozenset({"full_throttle", "hold_speed"})
# Default flights per style (about 20k flights in all)
DEFAULT_FLIGHTS = {"probe_hold": 3000, "steady_below": 2000, "random_hold": 2000, "shaky": 1000, "wavy": 500,
                   "hold_speed": 1500, "cruise": 1000, "endurance": 1500, "full_throttle": 2000, "blind_ppo": 1500,
                   "replay_good": 2000, "aware_good": 1500, "switching": 3000}
MIXES = {
    "narrow": ("full_throttle",),
    "main": ("probe_hold", "steady_below", "random_hold", "shaky", "wavy", "hold_speed", "cruise", "endurance",
             "blind_ppo", "replay_good", "switching"),
    "main_plus_aware": ("probe_hold", "steady_below", "random_hold", "shaky", "wavy", "hold_speed", "cruise",
                        "endurance", "blind_ppo", "replay_good", "switching", "aware_good"),
}
assert not (set(MIXES["narrow"]) | set(MIXES["main"]) | set(SWITCH_STYLES)) & LEAKING_STYLES
# Report conditions (report_step0.py) whose models fly as policy pilots
BLIND_CONDITIONS = ("0a_blind", "0b_blind", "0c_blind")
GOOD_CONDITIONS = ("0a", "0b", "0c", "2a_ideal", "2b_ideal", "2a_ideal_std", "2b_ideal_std")
OUTCOMES = ("completed", "battery_empty", "ground_crash", "truncated")


# ---------------------------------------------------------------------------------------------------------- physics


class PlanePhysics:
    """
    Physics formulas for the scripted pilots: drag and the cheapest level-flight speed at an altitude, and the constants.
    It holds no plane state, so a pilot can't read the battery or the true speed through it.
    """

    def __init__(self, env_kwargs: dict | None = None) -> None:
        core = bp.BatteryPlaneVec(n_envs=1, observation_mode="hidden", seed=0, **dict(env_kwargs or {}))
        self.a_max, self.y_max, self.y_ref, self.v_ref = core.a_max, core.y_max, core.y_ref, core.v_ref
        self.T, self.dt, self.g = core.T, core.dt, core.g
        self._k_D, self._H, self._q_star = core.k_D, core.H, core.q_star()
        if BELOW_CAP_THROTTLE * core.a_max >= core.h_min:
            raise ValueError("BELOW_CAP_THROTTLE must stay below the cap of an empty battery")

    def drag(self, y: float, v: float) -> float:
        return self._k_D * math.exp(-y / self._H) * v * abs(v)

    def v_star(self, y: float) -> float:
        """Cheapest level-flight speed at altitude y."""
        return math.sqrt(self._q_star / math.exp(-y / self._H))

    def hold_throttle(self, v_target: float, v: float, y: float, low: float, high: float, gain: float = 2.0) -> float:
        """Cruise throttle that drives the speed estimate v to v_target (drag feed-forward plus a proportional term)."""
        return float(np.clip((self.drag(y, v) + gain * (v_target - v)) / self.a_max, low, high))

    def decode(self, obs) -> tuple[float, float, float, float]:
        """Altitude (m), speed reading (m/s, noisy), climb rate (m/s) and time left from a hidden-mode observation."""
        return float(obs[0]) * self.y_ref, float(obs[1]) * self.v_ref, float(obs[2]) * self.v_ref, float(obs[3])


class SpeedFilter:
    """Exponential smoothing of the noisy speed reading: what a pilot steers by instead of the true speed."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha, self.v = alpha, 0.0

    def reset(self, reading: float) -> float:
        self.v = reading
        return self.v

    def update(self, reading: float) -> float:
        self.v += self.alpha * (reading - self.v)
        return self.v


class AltitudeProgram:
    """
    The lift command of one flight, steered by the exact observed altitude and climb rate: level flight, a climb to an
    altitude, the ceiling, changing altitude targets, or low flight. A guard pulls up near the ground.
    """

    KINDS = ("level", "climb", "ceiling", "targets", "low")

    def __init__(self, physics: PlanePhysics, rng: np.random.Generator, kinds=KINDS, guard: bool = True) -> None:
        self.physics, self.guard = physics, guard
        self.kind = str(rng.choice(kinds))
        self.schedule: list[tuple[int, float]] = []
        if self.kind == "climb":
            self.schedule = [(0, float(rng.uniform(100.0, 195.0)))]
        elif self.kind == "ceiling":
            self.schedule = [(0, physics.y_max + 5.0)]
        elif self.kind == "low":
            self.schedule = [(0, float(rng.uniform(25.0, 60.0)))]
        elif self.kind == "targets":
            t = 0
            while t < physics.T:
                self.schedule.append((t, float(rng.uniform(40.0, 195.0))))
                t += int(rng.integers(100, 301))

    def target(self, t: int) -> float | None:
        current = None
        for start, value in self.schedule:
            if start <= t:
                current = value
        return current

    def lift(self, y: float, vy: float, t: int) -> float:
        if self.guard and y < 15.0 and vy < 0.5:
            return 1.0
        target = self.target(t)
        if target is None:
            return 0.0
        return float(np.clip((0.5 * (target - y) - 1.5 * vy) / (self.physics.a_max / 2), -1.0, 1.0))


def guarded(lift: float, y: float, vy: float) -> float:
    """The ground guard for random lift commands."""
    return 1.0 if y < 15.0 and vy < 0.5 else lift


# ----------------------------------------------------------------------------------------------------------- pilots


class Pilot:
    """
    A way of flying. reset() starts a flight from its first observation, act() returns the action for observation o_t
    at flight step t. Scripted pilots read only hidden-mode observations; ``observation`` says which observation the
    generator passes ("hidden", or "observed" with the battery for a pilot that LEAKS).
    """

    name = ""
    #: Its actions depend on the recorded plane's battery or a battery estimate: never in the main mix
    leaks = False
    #: The observation it reads: "hidden" [y, vx + noise, vy, tau] or "observed" (with the battery)
    observation = "hidden"

    def __init__(self, physics: PlanePhysics) -> None:
        self.physics = physics

    def reset(self, rng: np.random.Generator, obs: np.ndarray, flight_seed: int = 0) -> None:
        """A new flight: its random stream, first observation and seed (from which any further seed derives)."""
        self.rng, self.flight_seed = rng, flight_seed

    def act(self, obs: np.ndarray, t: int) -> np.ndarray:
        raise NotImplementedError


class FullThrottle(Pilot):
    """Full throttle all flight, climbing to the ceiling: the A-like narrow control."""

    name = "full_throttle"

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.altitude = AltitudeProgram(self.physics, rng, kinds=("ceiling",))

    def act(self, obs, t):
        y, _, vy, _ = self.physics.decode(obs)
        return np.array([1.0, self.altitude.lift(y, vy, t)], np.float32)


class ProbeHold(Pilot):
    """Full-throttle probes of 1-10 s, each followed by a held speed below the cap; 1-4 probes per flight."""

    name = "probe_hold"

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.probes: list[tuple[int, int]] = []
        t = 0 if rng.random() < 0.5 else int(rng.integers(0, 251))
        for _ in range(int(rng.integers(1, 5))):
            length = int(rng.integers(20, 201))
            self.probes.append((t, t + length))
            t += length + int(rng.integers(60, 301))
        self.v_hold = float(rng.uniform(25.0, 50.0))
        self.speed = SpeedFilter()
        self.speed.reset(self.physics.decode(obs)[1])
        self.altitude = AltitudeProgram(self.physics, rng)

    def probing(self, t: int) -> bool:
        return any(start <= t < end for start, end in self.probes)

    def act(self, obs, t):
        y, reading, vy, _ = self.physics.decode(obs)
        v = self.speed.update(reading)
        throttle = 1.0 if self.probing(t) else self.physics.hold_throttle(self.v_hold, v, y, -BELOW_CAP_THROTTLE,
                                                                           BELOW_CAP_THROTTLE)
        return np.array([throttle, self.altitude.lift(y, vy, t)], np.float32)


class SteadyBelow(Pilot):
    """A constant throttle, or a held speed, always below the cap: the battery never shows."""

    name = "steady_below"

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.constant = float(rng.uniform(0.05, BELOW_CAP_THROTTLE)) if rng.random() < 0.5 else None
        self.v_hold = float(rng.uniform(20.0, 45.0))
        self.speed = SpeedFilter()
        self.speed.reset(self.physics.decode(obs)[1])
        self.altitude = AltitudeProgram(self.physics, rng)

    def act(self, obs, t):
        y, reading, vy, _ = self.physics.decode(obs)
        v = self.speed.update(reading)
        throttle = self.constant if self.constant is not None else self.physics.hold_throttle(
            self.v_hold, v, y, -BELOW_CAP_THROTTLE, BELOW_CAP_THROTTLE)
        return np.array([throttle, self.altitude.lift(y, vy, t)], np.float32)


class RandomHold(Pilot):
    """
    Random throttle (reverse included) held for random durations; random lift. Each flight draws its own throttle
    level, spread and segment lengths, so flights drain the battery at different rates (one shared rate would make the
    battery predictable from the clock). Full forward or reverse 20% of the time.
    """

    name = "random_hold"

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.level, self.spread = float(rng.uniform(-0.2, 0.9)), float(rng.uniform(0.1, 1.0))
        self.longest = int(rng.integers(10, 151))
        self.altitude = AltitudeProgram(self.physics, rng) if rng.random() < 0.5 else None
        self.throttle_until = self.lift_until = -1
        self.throttle = self.lift_command = 0.0

    def act(self, obs, t):
        y, _, vy, _ = self.physics.decode(obs)
        rng = self.rng
        if t >= self.throttle_until:
            if rng.random() < 0.2:
                self.throttle = float(rng.choice([-1.0, 1.0]))
            else:
                self.throttle = float(np.clip(self.level + self.spread * rng.uniform(-1.0, 1.0), -1.0, 1.0))
            self.throttle_until = t + int(rng.integers(5, self.longest + 1))
        if self.altitude is not None:
            lift = self.altitude.lift(y, vy, t)
        else:
            if t >= self.lift_until:
                self.lift_command = float(rng.uniform(-1.0, 1.0))
                self.lift_until = t + int(rng.integers(10, 151))
            lift = guarded(self.lift_command, y, vy)
        return np.array([self.throttle, lift], np.float32)


class Shaky(Pilot):
    """A new random throttle and lift at every step around a per-flight level and spread (with the ground guard)."""

    name = "shaky"

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.level, self.spread = float(rng.uniform(-0.3, 0.8)), float(rng.uniform(0.1, 1.0))
        self.lift_spread = float(rng.uniform(0.2, 1.0))

    def act(self, obs, t):
        y, _, vy, _ = self.physics.decode(obs)
        throttle = float(np.clip(self.level + self.spread * self.rng.uniform(-1.0, 1.0), -1.0, 1.0))
        lift = float(self.lift_spread * self.rng.uniform(-1.0, 1.0))
        return np.array([throttle, guarded(lift, y, vy)], np.float32)


class Wavy(Pilot):
    """Sinusoidal throttle with a random offset, amplitude, period (1-20 s) and phase."""

    name = "wavy"

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.offset, self.amplitude = float(rng.uniform(-0.2, 0.8)), float(rng.uniform(0.1, 0.8))
        self.period, self.phase = float(rng.uniform(20.0, 400.0)), float(rng.uniform(0.0, 2 * math.pi))
        self.altitude = AltitudeProgram(self.physics, rng)

    def act(self, obs, t):
        y, _, vy, _ = self.physics.decode(obs)
        throttle = float(np.clip(self.offset + self.amplitude * math.sin(2 * math.pi * t / self.period + self.phase), -1, 1))
        return np.array([throttle, self.altitude.lift(y, vy, t)], np.float32)


class HoldSpeed(Pilot):
    """
    Hold 45-65 m/s, steering by the smoothed noisy speed (battery_plane.policy_hold_goal steers by the true speed). On a
    low battery the cap stops it from holding the speed, which reveals the battery as it would for a real agent.
    """

    name = "hold_speed"

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.v_target = float(rng.uniform(45.0, 65.0))
        self.speed = SpeedFilter()
        self.speed.reset(self.physics.decode(obs)[1])
        kinds = ("ceiling", "climb") if rng.random() < 0.7 else AltitudeProgram.KINDS
        self.altitude = AltitudeProgram(self.physics, rng, kinds=kinds)

    def act(self, obs, t):
        y, reading, vy, _ = self.physics.decode(obs)
        throttle = self.physics.hold_throttle(self.v_target, self.speed.update(reading), y, 0.0, 1.0)
        return np.array([throttle, self.altitude.lift(y, vy, t)], np.float32)


class Cruise(Pilot):
    """Hold the cheapest speed for the altitude, steering by the smoothed noisy speed (changed from policy_cruise)."""

    name = "cruise"

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.speed = SpeedFilter()
        self.speed.reset(self.physics.decode(obs)[1])
        self.altitude = AltitudeProgram(self.physics, rng)

    def act(self, obs, t):
        y, reading, vy, _ = self.physics.decode(obs)
        throttle = self.physics.hold_throttle(self.physics.v_star(y), self.speed.update(reading), y, 0.0, 1.0)
        return np.array([throttle, self.altitude.lift(y, vy, t)], np.float32)


class Endurance(Pilot):
    """A full-throttle probe of 3-8 s at take-off, then gentle flying to the end: cruise, or a speed below the cap."""

    name = "endurance"

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.probe = int(rng.integers(60, 161))
        self.cruise = rng.random() < 0.5
        self.v_hold = float(rng.uniform(25.0, 45.0))
        self.speed = SpeedFilter()
        self.speed.reset(self.physics.decode(obs)[1])
        self.altitude = AltitudeProgram(self.physics, rng, kinds=("level", "climb"))

    def act(self, obs, t):
        y, reading, vy, _ = self.physics.decode(obs)
        v = self.speed.update(reading)
        if t < self.probe:
            throttle = 1.0
        elif self.cruise:
            throttle = self.physics.hold_throttle(self.physics.v_star(y), v, y, 0.0, 1.0)
        else:
            throttle = self.physics.hold_throttle(self.v_hold, v, y, -BELOW_CAP_THROTTLE, BELOW_CAP_THROTTLE)
        return np.array([throttle, self.altitude.lift(y, vy, t)], np.float32)


SCRIPTED_PILOTS = {cls.name: cls for cls in (FullThrottle, ProbeHold, SteadyBelow, RandomHold, Shaky, Wavy, HoldSpeed,
                                             Cruise, Endurance)}


# -------------------------------------------------------------------------------------------------- policy pilots


class PolicyDriver:
    """
    A saved model (plain PPO or Contextual_PPO) that flies from observed-mode observations. It reads what it was trained
    on: the observed observation (the true battery: 0a, 0b, 0c), the hidden observation (0a_blind, 0b_blind,
    0c_blind), or the hidden observation plus its own ideal estimate (2a_ideal, 2b_ideal and their _std), which it
    computes from the hidden observations and its executed actions as the context wrapper does.
    """

    def __init__(self, run_dir: str | Path) -> None:
        from omegaconf import OmegaConf

        from context_wrapper import context_source_from

        self.run_dir = Path(run_dir)
        cfg = OmegaConf.load(self.run_dir / "config.yaml")
        self.contextual = "cppo" in cfg
        self.mode = str(cfg.env.observation_mode)
        if self.mode not in ("observed", "hidden"):
            raise ValueError(f"{run_dir}: pilots fly observed- or hidden-mode policies, not {self.mode}")
        self.source = context_source_from(cfg)
        if self.contextual:
            from contextual_ppo import ContextualPPO

            self.model = ContextualPPO.load(str(self.run_dir / "final_model.zip"), device="cpu")
        else:
            from stable_baselines3 import PPO

            self.model = PPO.load(str(self.run_dir / "final_model.zip"), device="cpu")
        self.estimator = None
        if self.source == "ideal":
            from step1_ideal_estimator import IdealBatteryEstimator

            prune = cfg.context.get("prune_below")
            env_kwargs = dict(OmegaConf.to_container(cfg.env.kwargs, resolve=True)) if cfg.env.get("kwargs") else {}
            self.estimator = IdealBatteryEstimator(env_kwargs, b_points=int(cfg.context.b_points),
                                                   vx_points=int(cfg.context.vx_points),
                                                   vy_tolerance=float(cfg.context.vy_tolerance),
                                                   prune_below=None if prune is None else float(prune))
            self.include_std = bool(cfg.context.include_std)
        elif self.source != "none":
            raise ValueError(f"{run_dir}: unsupported context source {self.source}")
        #: Whether its actions depend on the battery or a battery estimate of the plane it flies
        self.battery_aware = self.mode == "observed" or self.source == "ideal"
        #: The BatteryPlane arguments it was trained with
        self.env_kwargs = dict(OmegaConf.to_container(cfg.env.kwargs, resolve=True)) if cfg.env.get("kwargs") else {}

    def start(self, observed_obs: np.ndarray, deterministic: bool, seed: int) -> None:
        """
        A new flight for this policy. ``seed`` seeds its action sampling (torch's generator) when it flies stochastically,
        so that a flight is reproducible whatever the worker flew before it.
        """
        import torch

        self.deterministic, self.state, self.first, self.last_action = deterministic, None, True, None
        if not deterministic:
            torch.manual_seed(seed)
        if self.estimator is not None:
            self.context = self.estimator.start(np.asarray(observed_obs[:4], float))

    def act(self, observed_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(observed_obs, np.float32)
        if self.estimator is not None and self.last_action is not None:
            self.context = self.estimator.update(self.last_action, obs[:4].astype(float))
        x = obs if self.mode == "observed" else obs[:4]
        if self.estimator is not None:
            x = np.concatenate([x, np.array(self.context[:2 if self.include_std else 1], np.float32)])
        if self.contextual:
            action, self.state = self.model.predict(x[None], state=self.state, episode_start=np.array([self.first]),
                                                    deterministic=self.deterministic)
        else:
            action, _ = self.model.predict(x[None], deterministic=self.deterministic)
        self.first = False
        self.last_action = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
        return self.last_action


class BlindPPO(Pilot):
    """Memoryless PPO policies trained in hidden mode, stochastic half of the time: battery-blind PPO flying."""

    name = "blind_ppo"

    def __init__(self, physics, drivers: list[PolicyDriver]) -> None:
        super().__init__(physics)
        if not drivers or any(d.battery_aware for d in drivers):
            raise ValueError("blind_ppo needs battery-blind policies (hidden mode, no context)")
        self.drivers = drivers

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.driver = self.drivers[int(rng.integers(len(self.drivers)))]
        # Hidden-mode policies read the first four columns only
        self.driver.start(np.asarray(obs, np.float32), deterministic=bool(rng.random() < 0.5),
                          seed=sampling_seed(flight_seed, 0))

    def act(self, obs, t):
        return self.driver.act(np.asarray(obs, np.float32))


class ReplayGood(Pilot):
    """
    A good pilot flies a shadow plane with its own random battery and start; its actions fly the recorded plane. When
    the shadow flight ends, a new shadow flight starts. The actions never depend on the recorded plane.
    """

    name = "replay_good"

    def __init__(self, physics, drivers: list[PolicyDriver], env_kwargs: dict | None = None) -> None:
        super().__init__(physics)
        if not drivers:
            raise ValueError("replay_good needs good policies")
        self.drivers, self.env_kwargs = drivers, dict(env_kwargs or {})

    def _new_shadow_flight(self) -> None:
        k = self.shadow_flights
        self.shadow_obs, _ = self.shadow.reset(seed=shadow_seed(self.flight_seed, k))
        self.driver.start(self.shadow_obs, self.deterministic, seed=sampling_seed(self.flight_seed, k))
        self.shadow_flights += 1

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.driver = self.drivers[int(rng.integers(len(self.drivers)))]
        self.deterministic = bool(rng.random() < 0.7)
        self.shadow = bp.BatteryPlaneEnv(observation_mode="observed", **self.env_kwargs)
        self.shadow_flights = 0
        self._new_shadow_flight()

    def act(self, obs, t):
        action = self.driver.act(self.shadow_obs)
        self.shadow_obs, _, terminated, truncated, _ = self.shadow.step(action)
        if terminated or truncated:
            self._new_shadow_flight()
        return action


class AwareGood(Pilot):
    """
    LEAKS: a good pilot flying the recorded plane itself, from its observed observation (the true battery, or the
    hidden part plus its own ideal estimate). Only for the separate mix main_plus_aware.
    """

    name = "aware_good"
    leaks = True
    observation = "observed"

    def __init__(self, physics, drivers: list[PolicyDriver]) -> None:
        super().__init__(physics)
        if not drivers:
            raise ValueError("aware_good needs good policies")
        self.drivers = drivers

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.driver = self.drivers[int(rng.integers(len(self.drivers)))]
        self.driver.start(np.asarray(obs, np.float32), deterministic=bool(rng.random() < 0.7),
                          seed=sampling_seed(flight_seed, 0))

    def act(self, obs, t):
        return self.driver.act(np.asarray(obs, np.float32))


class Switching(Pilot):
    """
    Changes style in the middle of the flight: segments of 30-250 steps, each flown by a fresh pilot of a style drawn
    uniformly from ``styles``. The whole plan (every switch time and every style) is drawn at take-off from the
    flight's random stream, before anything is observed, and a segment's style never depends on the time or on any
    observation. So which style flies, gentle or hard, has nothing to do with the battery: the flights can't teach
    "low battery -> gentle flying". Each segment's pilot gets its own seed (segment_seed), so no two segments share
    randomness, action-sampling noise or a shadow plane; it counts flight steps from its own start.
    """

    name = "switching"

    def __init__(self, physics, pilots: dict[str, Pilot]) -> None:
        super().__init__(physics)
        if not pilots or any(p.leaks or p.observation != "hidden" for p in pilots.values()):
            raise ValueError("switching needs battery-blind pilots that read the hidden observation")
        self.pilots, self.styles = pilots, tuple(pilots)

    def reset(self, rng, obs, flight_seed=0):
        super().reset(rng, obs, flight_seed)
        self.plan: list[tuple[int, str]] = []
        t = 0
        while t < self.physics.T:
            self.plan.append((t, str(rng.choice(self.styles))))
            t += int(rng.integers(30, 251))
        if len(self.plan) > SEEDS_PER_FLIGHT:
            raise ValueError("Too many segments for the per-flight seeds")
        self.segment = -1

    def act(self, obs, t):
        if self.segment + 1 < len(self.plan) and t >= self.plan[self.segment + 1][0]:
            self.segment += 1
            seed = segment_seed(self.flight_seed, self.segment)
            style = self.plan[self.segment][1]
            self.pilot = self.pilots[style]
            self.pilot.reset(np.random.default_rng([seed, STYLES.index(style), 23]), obs, seed)
        return self.pilot.act(obs, t - self.plan[self.segment][0])


# ------------------------------------------------------------------------------------------------------- flights


def fly(pilot: Pilot, seed: int, rng: np.random.Generator, env_kwargs: dict | None = None) -> dict:
    """
    One recorded flight. The plane runs in observed mode, whose first four columns are the hidden-mode observation
    (the same noise draws); the record keeps only those, and the pilot sees the battery only if it LEAKS.

    :return: observations [K+1, 4] (o_0 .. o_K, the last one after the final action), prev_actions [K+1, 2] (a_{t-1},
        zero at t = 0), batteries [K+1] (the label b_t, privileged), saturated [K+1] (the action before o_t hit the cap;
        privileged, diagnostics only), and the flight's seed, start state (privileged), length K and outcome
    """
    from flight_logging import terminal_outcome

    if seed in HELD_OUT_SEEDS:
        raise ValueError(f"Seed {seed} is a validation or final-test seed")
    env = bp.BatteryPlaneEnv(observation_mode="observed", **dict(env_kwargs or {}))
    obs, info = env.reset(seed=seed)
    start = {"b0": float(info["b0"]), "y0": float(env.core.y[0]), "vx0": float(env.core.vx[0])}
    view = (lambda o: o) if pilot.observation == "observed" else (lambda o: o[:4])
    pilot.reset(rng, view(obs), seed)
    observations, prev_actions = [obs[:4].copy()], [np.zeros(2, np.float32)]
    batteries, saturated = [float(info["b0"])], [False]
    for t in range(env.core.T):
        action = np.clip(np.asarray(pilot.act(view(obs), t), np.float32).reshape(2), -1.0, 1.0)
        obs, _, terminated, truncated, info = env.step(action)
        observations.append(obs[:4].copy())
        prev_actions.append(action)
        batteries.append(float(info["b"]))
        saturated.append(bool(info["saturated"]))
        if terminated or truncated:
            break
    return {"seed": seed, **start, "length": len(prev_actions) - 1, "outcome": terminal_outcome(info),
            "observations": np.array(observations, np.float32), "prev_actions": np.array(prev_actions, np.float32),
            "batteries": np.array(batteries, np.float32), "saturated": np.array(saturated, bool)}


_DRIVERS: dict[str, PolicyDriver] = {}


def drivers_for(run_dirs: list[str], env_kwargs: dict | None = None) -> list[PolicyDriver]:
    """Loaded policies, cached per worker process; each must have been trained with the dataset's physics."""
    for run_dir in run_dirs:
        if run_dir not in _DRIVERS:
            _DRIVERS[run_dir] = PolicyDriver(run_dir)
        if _DRIVERS[run_dir].env_kwargs != dict(env_kwargs or {}):
            raise ValueError(f"{run_dir} was trained with env.kwargs {_DRIVERS[run_dir].env_kwargs}, the flights use "
                             f"{dict(env_kwargs or {})}")
    return [_DRIVERS[run_dir] for run_dir in run_dirs]


def make_pilot(style: str, physics: PlanePhysics, policies: dict[str, list[str]], env_kwargs: dict | None = None) -> Pilot:
    if style in SCRIPTED_PILOTS:
        return SCRIPTED_PILOTS[style](physics)
    if style == "blind_ppo":
        return BlindPPO(physics, drivers_for(policies.get("blind", []), env_kwargs))
    if style == "replay_good":
        return ReplayGood(physics, drivers_for(policies.get("good", []), env_kwargs), env_kwargs)
    if style == "aware_good":
        return AwareGood(physics, drivers_for(policies.get("good", []), env_kwargs))
    if style == "switching":
        # The policy styles join when their policies are given
        usable = [s for s in SWITCH_STYLES if s in SCRIPTED_PILOTS or (s == "blind_ppo" and policies.get("blind"))
                  or (s == "replay_good" and policies.get("good"))]
        return Switching(physics, {s: make_pilot(s, physics, policies, env_kwargs) for s in usable})
    raise ValueError(f"Unknown pilot style {style!r}; one of {STYLES}")


def flight_seed(style: str, index: int, first_seed: int = FIRST_SEED) -> int:
    return first_seed + STYLES.index(style) * SEED_STRIDE + index


def fly_batch(style: str, indices: list[int], policies: dict, env_kwargs: dict, first_seed: int) -> list[dict]:
    """A batch of one style's flights (one worker task); the pilot's randomness is seeded by the flight."""
    import torch

    torch.set_num_threads(1)
    physics = PlanePhysics(env_kwargs)
    pilot = make_pilot(style, physics, policies, env_kwargs)
    flights = []
    for index in indices:
        seed = flight_seed(style, index, first_seed)
        rng = np.random.default_rng([seed, STYLES.index(style), 17])
        flights.append(fly(pilot, seed, rng, env_kwargs))
    return flights


# --------------------------------------------------------------------------------------------- storage and checks


def clock_r2(batteries: list[np.ndarray]) -> float:
    """
    How much of the battery's spread a clock explains: R^2 of the best estimate that knows only the flight step (the
    mean battery of the flights still flying at that step). A-F's flights: about 0.45.
    """
    longest = max(len(b) for b in batteries)
    sums, counts = np.zeros(longest), np.zeros(longest)
    for b in batteries:
        sums[:len(b)] += b
        counts[:len(b)] += 1
    clock = sums / np.maximum(counts, 1)
    values = np.concatenate(batteries)
    residual = sum(float(((b - clock[:len(b)]) ** 2).sum()) for b in batteries)
    total = float(((values - values.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float("nan")


def save_style(directory: Path, style: str, flights: list[dict]) -> dict:
    """One style's flights as one .npz (flights concatenated; flight_offsets[i]:flight_offsets[i+1] is flight i)."""
    offsets = np.cumsum([0] + [len(f["batteries"]) for f in flights])
    np.savez_compressed(
        directory / f"{style}.npz",
        observations=np.concatenate([f["observations"] for f in flights]),
        prev_actions=np.concatenate([f["prev_actions"] for f in flights]),
        # The training label (privileged: never an encoder input)
        batteries=np.concatenate([f["batteries"] for f in flights]),
        # Privileged diagnostics, never inputs: the action before o_t hit the cap; the starting battery and true speed
        privileged_saturated=np.concatenate([f["saturated"] for f in flights]),
        privileged_b0=np.array([f["b0"] for f in flights], np.float32),
        privileged_vx0=np.array([f["vx0"] for f in flights], np.float32),
        flight_offsets=offsets.astype(np.int64),
        seeds=np.array([f["seed"] for f in flights], np.int64),
        y0=np.array([f["y0"] for f in flights], np.float32),
        lengths=np.array([f["length"] for f in flights], np.int64),
        outcomes=np.array([OUTCOMES.index(f["outcome"]) for f in flights], np.int64),
    )
    batteries = [f["batteries"] for f in flights]
    steps = int(offsets[-1])
    throttle = np.concatenate([f["prev_actions"][1:, 0] for f in flights])
    return {"flights": len(flights), "steps": steps, "mean_length": float(np.mean([f["length"] for f in flights])),
            "outcomes": {o: sum(f["outcome"] == o for f in flights) / len(flights) for o in OUTCOMES},
            "clock_r2": clock_r2(batteries),
            "cap_fraction": float(np.concatenate([f["saturated"][1:] for f in flights]).mean()),
            "throttle_mean": float(throttle.mean()), "throttle_std": float(throttle.std()),
            "low_battery_fraction": float((np.concatenate(batteries) < 0.15).mean())}


def load_flights(directory: str | Path, styles) -> dict:
    """The flights of the given styles (or a mix name), as lists of per-flight arrays, with each flight's style."""
    directory = Path(directory)
    styles = MIXES[styles] if isinstance(styles, str) else tuple(styles)
    out = {"observations": [], "prev_actions": [], "batteries": [], "styles": []}
    for style in styles:
        with np.load(directory / f"{style}.npz") as data:
            # Each access to an .npz array decompresses all of it: read every array once
            offsets, arrays = data["flight_offsets"], {key: data[key] for key in ("observations", "prev_actions", "batteries")}
        for i in range(len(offsets) - 1):
            part = slice(offsets[i], offsets[i + 1])
            for key, values in arrays.items():
                out[key].append(values[part])
            out["styles"].append(style)
    return out


def policies_from_reports(paths: list[Path]) -> dict[str, list[str]]:
    """Run directories of the memoryless (blind) and good policies, from report_step0.py reports."""
    found = {"blind": [], "good": []}
    for path in paths:
        for run in json.loads(Path(path).read_text())["runs"]:
            group = "blind" if run["condition"] in BLIND_CONDITIONS else "good" if run["condition"] in GOOD_CONDITIONS else None
            if group and run["run_directory"] not in found[group]:
                found[group].append(run["run_directory"])
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Folder for one .npz per style and manifest.json")
    parser.add_argument("--styles", nargs="+", default=list(STYLES), choices=STYLES)
    parser.add_argument("--flights", type=int, default=None, help="Flights per style (default: DEFAULT_FLIGHTS)")
    parser.add_argument("--policy-reports", type=Path, nargs="*", default=[],
                        help="report_step0.py reports whose models fly the policy pilots (blind: 0a/0b/0c_blind; good: "
                             "0a, 0b, 0c, 2a/2b_ideal and their _std)")
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--jobs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=50, help="Flights per worker task")
    args = parser.parse_args()
    policies = policies_from_reports(args.policy_reports)
    for style, group in (("blind_ppo", "blind"), ("replay_good", "good"), ("aware_good", "good")):
        if style in args.styles and not policies[group]:
            parser.error(f"{style} needs --policy-reports with {group} policies")
    counts = {style: args.flights or DEFAULT_FLIGHTS[style] for style in args.styles}
    last = max(flight_seed(style, counts[style] - 1, args.first_seed) for style in args.styles)
    if args.first_seed <= 2002 or last >= SHADOW_SEED_BASE:
        parser.error("Flight seeds must lie above the validation and test seeds and below the shadow seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    env_kwargs: dict = {}
    print(f"{sum(counts.values())} flights: " + ", ".join(f"{s} {n}" for s, n in counts.items()), flush=True)
    started = time.time()
    tasks = [(style, list(range(i, min(i + args.batch, counts[style])))) for style in args.styles
             for i in range(0, counts[style], args.batch)]
    flights: dict[str, list[dict]] = {style: [] for style in args.styles}
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [(style, pool.submit(fly_batch, style, indices, policies, env_kwargs, args.first_seed))
                   for style, indices in tasks]
        for style, future in futures:
            flights[style].extend(future.result())
    styles = {}
    for style in args.styles:
        flights[style].sort(key=lambda f: f["seed"])
        styles[style] = {"leaks": style in LEAKING_STYLES, **save_style(args.output, style, flights[style])}
        print(f"  {style:13s} {styles[style]['flights']:5d} flights, mean length {styles[style]['mean_length']:5.0f}, "
              f"completed {styles[style]['outcomes']['completed']:.0%}, at the cap {styles[style]['cap_fraction']:.0%} "
              f"of steps, clock R^2 {styles[style]['clock_r2']:.2f}", flush=True)
    mixes = {}
    for mix, members in MIXES.items():
        if all(m in flights for m in members):
            mixes[mix] = {"styles": list(members),
                          "clock_r2": clock_r2([f["batteries"] for m in members for f in flights[m]]),
                          "flights": sum(len(flights[m]) for m in members)}
            print(f"  mix {mix:15s} {mixes[mix]['flights']:6d} flights, clock R^2 {mixes[mix]['clock_r2']:.2f}", flush=True)
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - started, 1),
                "env_kwargs": env_kwargs, "first_seed": args.first_seed, "seed_stride": SEED_STRIDE,
                "seeds_per_flight": SEEDS_PER_FLIGHT, "shadow_seed_base": SHADOW_SEED_BASE,
                "held_out_seeds": sorted(HELD_OUT_SEEDS), "styles": styles, "mixes": mixes,
                "policies": policies, "below_cap_throttle": BELOW_CAP_THROTTLE,
                "record": "observations o_t [4] (hidden), prev_actions a_{t-1} [2] (0 at t=0), batteries b_t (label, "
                          "privileged), privileged_* (diagnostics, never inputs), per flight o_0..o_K"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
