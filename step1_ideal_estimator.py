"""Step 1 of ENCODER_PLAN.md: the ideal battery estimator ("perfect calculator") on the validation flights.

The encoder-error measurement (step1_encoder_error.py) found that the A rerun's encoder reads the battery well on
its own hard flights but stays about 0.15 off on a good pilot's gentle flights. This asks how much of that is
unavoidable: what is the best estimate any encoder could make from the same information?

IdealBatteryEstimator knows the exact physics (BatteryPlaneVec) and the training distribution of starting states.
It reads what the encoder reads, the hidden-mode observations and the executed actions, and computes the posterior
mean of b_t: the estimate with the smallest expected squared error. The unknowns are the starting battery and the
starting speed (the speed reading is noisy); altitude and climb rate are observed exactly. It is compared with the
A encoder and a clock (the mean battery of the flights still flying at that step) on the flights of
step1_encoder_error.py: A's own flights and those of the 0c pilot of the same seed (final models, validation seeds
1000-1002 only, never the final test).

    python step1_ideal_estimator.py --a-runs runs_cppo/<A s0> runs_cppo/<A s1> runs_cppo/<A s2> \
        --pilot-runs runs_cppo/<0c s0> runs_cppo/<0c s1> runs_cppo/<0c s2> --output runs_step0/ideal_estimator.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

import battery_plane as bp

VIEWS = ("own", "pilot")
STEPS = (0, 25, 50, 100, 175, 250, 350, 500, 650)
BATTERIES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


class IdealBatteryEstimator:
    """
    The posterior mean of the battery b_t given the hidden-mode observations o_0..o_t and the executed actions
    a_0..a_{t-1} (what the encoder reads at step t), under the exact physics and the training distribution of
    starting states.

    Unknown: the starting battery, uniform on the env's b0_range, and the starting speed, uniform on vx0_range,
    read with Gaussian noise of std sigma. Known exactly (up to float32 rounding): altitude, climb rate and time.
    Candidates: a grid of (starting battery, starting speed). At every step each candidate is replayed through
    BatteryPlaneVec.step with the executed action, from the observed altitude and climb rate, and weighed by how well it
    predicts the next observation: the speed reading (Gaussian, std sigma) and the climb rate (Gaussian, std
    vy_tolerance; the climb rate is exact, the tolerance only absorbs the grid's spacing). A candidate whose battery is
    empty while the plane flies on is impossible. The estimate is the weighted mean of the candidates' batteries.

    :param env_kwargs: BatteryPlaneVec arguments of the run (training.env_kwargs_from(cfg))
    :param b_values: Starting batteries to try (default: b_points evenly over b0_range)
    :param vx_values: Starting speeds to try (default: vx_points evenly over vx0_range)
    :param vy_tolerance: Std of the climb-rate match, m/s
    :param prune_below: Drop the candidates whose log weight falls this far below the best one (e.g. 30: weights under
        1e-13 of the best), so that only the plausible ones are replayed; None keeps every candidate (exact, slower)
    """

    def __init__(self, env_kwargs: dict | None = None, b_values=None, vx_values=None, b_points: int = 201,
                 vx_points: int = 601, vy_tolerance: float = 0.005, prune_below: float | None = None) -> None:
        env_kwargs = dict(env_kwargs or {})
        core = bp.BatteryPlaneVec(n_envs=1, observation_mode="hidden", seed=0, **env_kwargs)
        if core.soc_choices is not None:
            raise ValueError("The ideal estimator assumes a continuous uniform starting battery (soc_choices=None)")
        if core.sigma <= 0:
            raise ValueError("The ideal estimator needs a noisy speed reading (hidden mode)")
        b_values = np.linspace(*core.b0_range, b_points) if b_values is None else np.asarray(b_values, float)
        vx_values = np.linspace(*core.vx0_range, vx_points) if vx_values is None else np.asarray(vx_values, float)
        b0, vx0 = np.meshgrid(b_values, vx_values, indexing="ij")
        self.b0, self.vx0 = b0.ravel(), vx0.ravel()
        self.sigma, self.vy_tolerance = core.sigma, float(vy_tolerance)
        self.y_ref, self.v_ref = core.y_ref, core.v_ref
        if prune_below is not None and not prune_below > 0:
            raise ValueError("prune_below must be positive or None")
        self.prune_below, self.env_kwargs = prune_below, env_kwargs
        # One plane per candidate, without speed noise (only the candidates' true states are used); a pruned flight
        # replays a smaller copy holding the remaining candidates
        self.full_sim = self.sim = self._planes(len(self.b0))
        # Steps at which the plane flew on although every candidate's battery was empty (grid rounding right before a
        # battery-empty ending): the constraint is skipped there instead of ruling out everything
        self.constraint_skips = 0

    def _planes(self, n: int):
        sim = bp.BatteryPlaneVec(n_envs=n, observation_mode="hidden", seed=0, auto_reset=False, **self.env_kwargs)
        sim.sigma = 0.0
        return sim

    def _prune(self) -> None:
        """Keep the candidates within prune_below of the best; move them to a smaller copy once half are gone."""
        keep = self.log_w >= self.log_w.max() - self.prune_below
        if keep.sum() > len(keep) // 2:
            return
        sim = self._planes(int(keep.sum()))
        sim.b0[:], sim.b[:], sim.vx[:] = self.sim.b0[keep], self.sim.b[keep], self.sim.vx[keep]
        self.sim, self.log_w = sim, self.log_w[keep]

    def _check(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=float)
        if obs.shape != (4,):
            raise ValueError("The ideal estimator reads hidden-mode observations [y, vx + noise, vy, tau], never the battery")
        return obs

    def _posterior(self) -> tuple[float, float]:
        w = np.exp(self.log_w - self.log_w.max())
        w /= w.sum()
        mean = float(w @ self.sim.b)
        return mean, float(np.sqrt(max(w @ (self.sim.b - mean) ** 2, 0.0)))

    def start(self, observation) -> tuple[float, float]:
        """
        A new flight: weigh every candidate by the first speed reading.

        :param observation: o_0, the hidden-mode observation at the reset
        :return: The posterior mean and std of b_0
        """
        obs = self._check(observation)
        sim = self.sim = self.full_sim
        sim.b0[:], sim.b[:], sim.vx[:] = self.b0, self.b0, self.vx0
        self.log_w = -0.5 * ((obs[1] * self.v_ref - sim.vx) / self.sigma) ** 2
        self.last, self.t, self.vy_mismatch = obs, 0, None
        return self._posterior()

    def update(self, action, observation, alive: bool = True) -> tuple[float, float]:
        """
        One step: replay every candidate with the executed action from the last observed altitude and climb rate,
        and weigh it by the new observation.

        :param action: a_t, the executed action after the last observation
        :param observation: o_{t+1}
        :param alive: False for the final observation of a flight (the plane may have just run out of battery),
            True while it flies on, which rules out every candidate with an empty battery
        :return: The posterior mean and std of b_{t+1}
        """
        obs = self._check(observation)
        sim = self.sim
        n = sim.n
        y, vy = self.last[0] * self.y_ref, self.last[2] * self.v_ref
        sim.y[:], sim.vy[:], sim.t[:] = y, vy, self.t
        sim.step(np.broadcast_to(np.asarray(action, dtype=float), (n, 2)))
        vy_error = obs[2] * self.v_ref - sim.vy
        self.vy_mismatch = float(np.abs(vy_error).min())
        self.log_w = (self.log_w - 0.5 * ((obs[1] * self.v_ref - sim.vx) / self.sigma) ** 2
                      - 0.5 * (vy_error / self.vy_tolerance) ** 2)
        if alive:
            # The plane flies on: an empty battery is impossible
            empty = sim.b <= 0.0
            if empty.all():
                self.constraint_skips += 1
            else:
                self.log_w[empty] = -np.inf
        if not np.isfinite(self.log_w.max()):
            raise RuntimeError(f"No candidate explains the flight at step {self.t + 1}")
        self.last, self.t = obs, self.t + 1
        mean_std = self._posterior()
        if self.prune_below is not None:
            self._prune()
        return mean_std

    def estimate(self, observations, actions) -> dict:
        """
        A whole flight at once (start, then one update per action).

        :param observations: [K, 4] hidden-mode observations o_0..o_{K-1} of one flight (the plane alive at each)
        :param actions: [K-1, 2] (or [K, 2]) executed actions, a_t taken after o_t
        :return: estimates [K] (posterior mean of b_t), stds [K] (posterior std), vy_mismatch [K-1] (the smallest
            climb-rate error of any candidate at each step: well below vy_tolerance when the grid is fine enough)
        """
        obs = np.asarray(observations, dtype=float)
        acts = np.asarray(actions, dtype=float)
        if obs.ndim != 2 or obs.shape[1] != 4:
            raise ValueError("The ideal estimator reads hidden-mode observations [y, vx + noise, vy, tau], never the battery")
        if acts.ndim != 2 or acts.shape[1] != 2 or len(acts) < len(obs) - 1:
            raise ValueError("Need one executed action after every observation but the last")
        mean, std = self.start(obs[0])
        estimates, stds, mismatch = [mean], [std], []
        for t in range(1, len(obs)):
            mean, std = self.update(acts[t - 1], obs[t])
            estimates.append(mean)
            stds.append(std)
            mismatch.append(self.vy_mismatch)
        return {"estimates": estimates, "stds": stds, "vy_mismatch": mismatch}

def estimate_flight(env_kwargs: dict, observations: list, actions: list, grid: dict) -> dict:
    return IdealBatteryEstimator(env_kwargs, **grid).estimate(observations, actions)


def record_flights(a_dir: str, pilot_dir: str, view: str) -> dict:
    """The final A model's validation flights (own) or the 0c pilot's with A's encoder watching (pilot)."""
    from omegaconf import OmegaConf

    import main_battery_plane_contextual_ppo as training
    import step1_encoder_error

    cfg = OmegaConf.load(Path(a_dir) / "config.yaml")
    env_kwargs = training.env_kwargs_from(cfg)
    # The estimator assumes the speed noise of the validation flights
    sigma = bp.BatteryPlaneVec(n_envs=1, observation_mode="hidden", seed=0, **env_kwargs).sigma
    if [float(x) for x in cfg.evaluation.reset_grid.eval_vx_obs_noise] != [sigma]:
        raise ValueError(f"Validation speed noise {list(cfg.evaluation.reset_grid.eval_vx_obs_noise)} != the estimator's {sigma}")
    result = step1_encoder_error.measure(a_dir, pilot_dir, "final_model.zip", view, record=True)
    result["env_kwargs"] = env_kwargs
    return result


def rms(errors) -> float | None:
    errors = np.asarray(errors, float)
    return float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None


def summarize(flights: list[dict]) -> dict:
    """Error (estimate - true battery) of the clock, the encoder and the ideal estimator at given flight steps."""
    by_step = {}
    for k in STEPS:
        alive = [f for f in flights if f["length"] > k]
        batteries = np.array([f["batteries"][k] for f in alive])
        by_step[str(k)] = {
            "n": len(alive),
            # The best estimate that knows only the flight step (in-sample: the mean battery of these flights)
            "clock": rms(batteries - batteries.mean()) if len(alive) else None,
            "encoder": rms([f["estimates"][k] - f["batteries"][k] for f in alive]),
            "ideal": rms([f["ideal"][k] - f["batteries"][k] for f in alive]),
            "ideal_std": float(np.sqrt(np.mean([f["ideal_std"][k] ** 2 for f in alive]))) if alive else None,
        }
    by_b0 = {}
    for b0 in BATTERIES:
        group = [f for f in flights if f["b0"] == b0]
        by_b0[f"{b0:g}"] = {str(k): {"battery": float(np.mean([f["batteries"][k] for f in alive])),
                                     "encoder": float(np.mean([f["estimates"][k] for f in alive])),
                                     "ideal": float(np.mean([f["ideal"][k] for f in alive]))}
                            for k in STEPS if (alive := [f for f in group if f["length"] > k])}
    z2 = np.concatenate([((np.array(f["ideal"]) - np.array(f["batteries"])) / np.maximum(np.array(f["ideal_std"]), 1e-6)) ** 2
                         for f in flights])
    return {"by_step": by_step, "by_b0": by_b0,
            # 1 when the ideal estimator's own uncertainty matches its errors
            "ideal_calibration": float(np.mean(z2)),
            "max_vy_mismatch": float(max(max(f["vy_mismatch"], default=0.0) for f in flights))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-runs", nargs="+", required=True, help="A rerun directories (runs_cppo/<id>)")
    parser.add_argument("--pilot-runs", nargs="+", required=True, help="0c run directories of the same training seeds")
    parser.add_argument("--b-points", type=int, default=201, help="Starting batteries tried (default 201: steps of 0.0025)")
    parser.add_argument("--vx-points", type=int, default=601, help="Starting speeds tried (default 601: steps of 0.05 m/s)")
    parser.add_argument("--vy-tolerance", type=float, default=0.005, help="Std of the climb-rate match, m/s")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from omegaconf import OmegaConf

    import report_step0

    pilots = {}
    for directory in args.pilot_runs:
        cfg = OmegaConf.load(Path(directory) / "config.yaml")
        report_step0.evaluation_protocol(cfg, "0c")
        pilots[int(cfg.training.seed)] = directory
    jobs = []
    for directory in args.a_runs:
        cfg = OmegaConf.load(Path(directory) / "config.yaml")
        report_step0.evaluation_protocol(cfg, "A")
        seed = int(cfg.training.seed)
        if seed not in pilots:
            parser.error(f"No 0c pilot with training seed {seed}")
        jobs += [(directory, pilots[seed], view) for view in VIEWS]
    grid = {"b_points": args.b_points, "vx_points": args.vx_points, "vy_tolerance": args.vy_tolerance}
    print(f"{len(args.a_runs)} A runs x {VIEWS}, validation flights only; ideal estimator with "
          f"{args.b_points} x {args.vx_points} candidates, climb-rate tolerance {args.vy_tolerance:g} m/s", flush=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.jobs, mp_context=context) as pool:
        results = list(pool.map(record_flights, *zip(*jobs)))
        flights = [(r, f) for r in results for f in r["flights"]]
        print(f"{len(flights)} flights recorded; running the ideal estimator", flush=True)
        ideals = list(pool.map(estimate_flight, [r["env_kwargs"] for r, _ in flights], [f["observations"] for _, f in flights],
                               [f["actions"] for _, f in flights], [grid] * len(flights)))
    for (_, flight), ideal in zip(flights, ideals):
        flight.update(ideal=ideal["estimates"], ideal_std=ideal["stds"], vy_mismatch=ideal["vy_mismatch"])
        del flight["observations"], flight["actions"]
    summaries = {view: summarize([f for r in results if r["view"] == view for f in r["flights"]]) for view in VIEWS}
    args.output.write_text(json.dumps({"steps": STEPS, "grid": grid, "summaries": summaries, "runs": results},
                                      allow_nan=False) + "\n")
    for view, summary in summaries.items():
        print(f"== {view} flights (calibration {summary['ideal_calibration']:.2f}, "
              f"largest climb-rate mismatch {summary['max_vy_mismatch']:.4f} m/s)", flush=True)
        for row in ("clock", "encoder", "ideal", "ideal_std"):
            cells = " ".join(f"{k}:{summary['by_step'][str(k)][row]:.3f}" if summary["by_step"][str(k)][row] is not None
                             else f"{k}:-" for k in STEPS)
            print(f"  {row:9s} {cells}", flush=True)


if __name__ == "__main__":
    main()
