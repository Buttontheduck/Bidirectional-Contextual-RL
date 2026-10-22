"""One CityLearn building with one battery, its charge as the context: the environment of the observed-vs-hidden test.

Self-contained (imports nothing from the rest of the repository). Run with /root/venvs/carlos-citylearn/bin/python.

- Building_1 of citylearn_challenge_2022_phase_1: a 6.4 kWh / 5 kW battery and 4 kW of solar panels, hourly steps.
- Episodes of 48 hours, from a random start hour, with a random starting charge (uniform in [0, 1]).
- Reward: minus the money paid for grid electricity in the hour, -max(net consumption x price, 0) in $ (exports earn
  nothing), CityLearn's cost KPI per step.
- Observation: CityLearn's normalized observation of the building (31 numbers in [0, 1]) plus the time left in the
  episode (1 at the start, 0 at the end). The observed setting keeps the battery's state of charge; the hidden setting
  drops it and is otherwise identical.
- Time: the year's weeks are split, spread over the seasons, into training weeks, validation weeks (11, 24, 37, 50) and
  final-test weeks (5, 18, 31, 44). Training episodes start anywhere their 48 hours stay in training weeks; validation
  and final-test scenarios are fixed lists of (start hour, starting charge) inside their weeks. The final-test list is
  reserved.
- info: the privileged charge ("charge", the state of charge) and stored energy, for diagnostics and for later
  supervised encoders; never part of the observation.
"""
from __future__ import annotations

import logging
import warnings

import gymnasium
import numpy as np

DATASET = "citylearn_challenge_2022_phase_1"
BUILDING = "Building_1"
EPISODE_STEPS = 48
HOURS, WEEK = 8760, 168
VALIDATION_WEEKS = (11, 24, 37, 50)
TEST_WEEKS = (5, 18, 31, 44)
CHEAP_PRICE, EXPENSIVE_PRICE = 0.25, 0.35          # between the tariff's cheap ($0.21-0.22) and expensive ($0.40-0.54) levels


def split_of_hour(hour: int) -> str:
    week = hour // WEEK
    return "validation" if week in VALIDATION_WEEKS else "test" if week in TEST_WEEKS else "train"


TRAIN_STARTS = np.array([s for s in range(HOURS - EPISODE_STEPS)
                         if all(split_of_hour(h) == "train" for h in range(s, s + EPISODE_STEPS + 1))])


def held_out_scenarios(weeks: tuple[int, ...], n: int, seed: int) -> list[tuple[int, float]]:
    """n fixed (start hour, starting charge) pairs, spread evenly over the given weeks, each episode inside its week."""
    rng = np.random.default_rng(seed)
    return [(int(weeks[k % len(weeks)] * WEEK + rng.integers(0, WEEK - EPISODE_STEPS)), float(rng.uniform(0.0, 1.0)))
            for k in range(n)]


VALIDATION_SCENARIOS = held_out_scenarios(VALIDATION_WEEKS, 80, seed=1000)
TEST_SCENARIOS = held_out_scenarios(TEST_WEEKS, 80, seed=2000)      # reserved: never used for training, tuning or selection


class CityBattery(gymnasium.Env):
    """Building_1 with its battery: 48-hour episodes, cost reward, charge observed or hidden.

    reset(options={"scenario": (start, charge)}) plays a given scenario; otherwise a training scenario is drawn from
    the env's generator (seeded by reset(seed=...)).
    """

    metadata = {"render_modes": []}

    def __init__(self, observe_charge: bool):
        from citylearn.citylearn import CityLearnEnv
        from citylearn.wrappers import NormalizedObservationWrapper

        logging.getLogger().setLevel(logging.WARNING)
        warnings.filterwarnings("ignore", module="citylearn")
        self.observe_charge = observe_charge
        self.city = CityLearnEnv(DATASET, buildings=[BUILDING], central_agent=True)
        self.city.reset()                                   # the default episode is the whole year: read the series once
        self.building = self.city.buildings[0]
        self.year_price = np.asarray(self.building.pricing.electricity_pricing, dtype=float).copy()
        self.battery = self.building.electrical_storage
        self.capacity, self.power = float(self.battery.capacity), float(self.battery.nominal_power)
        self.normalized = NormalizedObservationWrapper(self.city)
        names = list(self.normalized.observation_names[0])
        self.keep = [i for i, name in enumerate(names) if observe_charge or name != "electrical_storage_soc"]
        self.observation_names = [names[i] for i in self.keep] + ["time_left"]
        self.raw_names = list(self.city.observation_names[0])
        self.observation_space = gymnasium.spaces.Box(0.0, 1.0, (len(self.observation_names),), dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.t, self.start, self.initial_charge = 0, 0, 0.0

    def raw(self) -> dict[str, float]:
        """The building's current raw (unnormalized) observation, by name; for rules and checks."""
        return dict(zip(self.raw_names, np.asarray(self.city.observations[0], dtype=float)))

    def charge(self) -> float:
        """The battery's state of charge now: the starting charge before the first step, then after the last executed
        hour (CityLearn writes hour t at index t; on the final step it does not advance its clock or refresh the
        observation, so the arrays are read by this env's own step count)."""
        return float(self.battery.soc[max(self.t - 1, 0)])

    def _observation(self, normalized) -> np.ndarray:
        values = np.asarray(normalized[0], dtype=np.float32)[self.keep]
        time_left = (EPISODE_STEPS - self.t) / EPISODE_STEPS
        return np.clip(np.append(values, time_left), 0.0, 1.0).astype(np.float32)

    def _info(self, cost: float | None = None) -> dict:
        info = {"charge": self.charge(), "energy": self.charge() * self.capacity, "start": self.start,
                "initial_charge": self.initial_charge, "hour_of_year": self.start + self.t}
        if cost is not None:
            info["cost"] = cost
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            start, charge = options["scenario"]
        else:
            start, charge = int(self.np_random.choice(TRAIN_STARTS)), float(self.np_random.uniform(0.0, 1.0))
        self.start, self.initial_charge, self.t = int(start), float(charge), 0
        self.city.episode_time_steps = [(self.start, self.start + EPISODE_STEPS)]
        self.battery.initial_soc = self.initial_charge
        normalized, _ = self.normalized.reset()
        return self._observation(normalized), self._info()

    def step(self, action):
        normalized, _, terminated, truncated, _ = self.normalized.step([np.asarray(action, dtype=float).reshape(-1)])
        hour = self.t                                       # the hour just executed, indexed from the episode start
        self.t += 1
        cost = max(float(self.building.net_electricity_consumption_cost[hour]), 0.0)
        return self._observation(normalized), -cost, bool(terminated), bool(truncated), self._info(cost)


# --- hand-written rules: a preliminary diagnostic of what the charge can be worth, not an upper bound ---------------

def _hours_to_expensive(env: CityBattery) -> int | None:
    """Hours from now to the next expensive hour inside the episode (0 if now), or None if none is left."""
    now = env.start + env.t
    for k in range(EPISODE_STEPS - env.t):
        if env.year_price[now + k] >= EXPENSIVE_PRICE:
            return k
    return None


def rule_action(name: str, env: CityBattery) -> np.ndarray:
    """One rule's action. Rules see the raw observation of the coming hour (load, solar, price) and the tariff schedule;
    only "aware_grid" also reads the charge."""
    raw = env.raw()
    surplus = raw["solar_generation"] - raw["non_shiftable_load"]        # kWh in the coming hour, >0: more solar than load
    follow = float(np.clip(surplus / env.power, -1.0, 1.0))              # charge the surplus, or cover the deficit
    expensive = raw["electricity_pricing"] >= EXPENSIVE_PRICE
    if name == "idle":
        a = 0.0
    elif name == "self_consumption":                                     # follow the surplus/deficit at every hour
        a = follow
    elif name == "time_of_use":                                          # charge any surplus; discharge only when expensive
        a = follow if (surplus > 0 or expensive) else 0.0
    elif name in ("grid_before_peak", "aware_grid"):
        # as time_of_use, plus charging from the grid in the cheap hours before an expensive block inside the episode
        a = follow if (surplus > 0 or expensive) else 0.0
        ahead = _hours_to_expensive(env)
        if not expensive and ahead is not None and ahead <= 6:
            if name == "grid_before_peak":
                a = 1.0                                                  # full power: the battery refuses when full
            else:                                                        # pace the charge to be full when the block starts
                missing = (1.0 - env.charge()) * env.capacity
                a = max(a, float(np.clip(missing / max(ahead, 1) / env.power, 0.0, 1.0)))
    else:
        raise ValueError(name)
    return np.array([a], dtype=np.float32)


RULES = ("idle", "self_consumption", "time_of_use", "grid_before_peak", "aware_grid")


def fly(env: CityBattery, act, scenarios: list[tuple[int, float]], record_observations: bool = False) -> list[dict]:
    """Plays each scenario; act(env, observation) returns the action. Per episode: cost, starting and ending stored
    energy, the charge after every step, and (if asked) the hidden-setting observations."""
    episodes = []
    for scenario in scenarios:
        obs, info = env.reset(options={"scenario": scenario})
        charges, observations, cost, done = [info["charge"]], [obs], 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(act(env, obs))
            cost -= reward
            charges.append(info["charge"])
            observations.append(obs)
            done = terminated or truncated
        episode = {"start": scenario[0], "initial_charge": scenario[1], "cost": cost,
                   "start_energy": charges[0] * env.capacity, "end_energy": charges[-1] * env.capacity,
                   "charges": charges}
        if record_observations:
            episode["observations"] = np.array(observations)
        episodes.append(episode)
    return episodes
