"""Step 1 of ENCODER_PLAN.md: how the A rerun's encoder error changes over a flight, step by step.

The in-flight noise runs showed that a battery estimate is enough if it improves steadily during the flight (linear),
but not if it stays poor until mid-flight (sharp). This measures which of the two the learned encoder of the A rerun
(experiment A with step 0's settings, config/rerun/) looks like: its estimate z_t of the battery b_t at every step of
the validation flights (seeds 1000-1002, never the final test), at its checkpoints and final model.

Two kinds of flights:
  own    the A policy flies (hidden mode), as in its validation: the flights its encoder was trained on
  pilot  a step 0 CPPO policy with the true battery (0c, same training seed) flies the plane, and the A encoder only
         rides along: it reads the hidden-mode observation and the pilot's executed actions, never the battery.
         The flights a policy that can use the battery would fly.

Every flight runs a hidden-mode and an observed-mode env in lockstep with the same seed and actions: the encoder and
the A policy read the hidden one, only the pilot reads the observed one, and the script checks at every step that the
hidden observation is the observed one without its battery column. The final models' flights are checked against
the returns and lengths of the runs' own final validation (final_validation.json). Cross pairs (final models only) let
every A encoder ride along with every pilot, to tell the encoder's share of the error from the pilot's.

    python step1_encoder_error.py --a-runs runs_cppo/<A s0> runs_cppo/<A s1> runs_cppo/<A s2> \
        --pilot-runs runs_cppo/<0c s0> runs_cppo/<0c s1> runs_cppo/<0c s2> --output runs_step0/encoder_error.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

CHECKPOINTS = ("checkpoints/cppo_500000_steps.zip", "checkpoints/cppo_1000000_steps.zip",
               "checkpoints/cppo_1500000_steps.zip", "final_model.zip")
VIEWS = ("own", "pilot")
# Flight steps of the summary (the observation index: 0 at take-off, 350 = mid-flight)
STEPS = (0, 25, 50, 100, 175, 250, 350, 500, 650)
WINDOWS = ((0, 50), (50, 150), (150, 250), (250, 350), (350, 500), (500, 700))
BATTERIES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def fly(encoder_policy, pilot_policy, env_kwargs: dict, reset: dict) -> dict:
    """
    One validation flight: the encoder's estimate and the true battery at every flight step.

    :param encoder_policy: The A policy, whose encoder is measured (and which flies when there is no pilot)
    :param pilot_policy: The policy that flies, reading the observed-mode observation, or None
    """
    import torch as th

    import battery_plane as bp
    from contextual_ppo import ContextualPolicyState

    hidden = bp.BatteryPlaneEnv(observation_mode="hidden", **env_kwargs)
    observed = bp.BatteryPlaneEnv(observation_mode="observed", **env_kwargs)
    hidden_obs, info = hidden.reset(seed=reset["seed"], options=dict(reset["options"]))
    observed_obs, _ = observed.reset(seed=reset["seed"], options=dict(reset["options"]))
    state = encoder_policy.initial_state(1)
    pilot_state = pilot_policy.initial_state(1) if pilot_policy is not None else None
    battery, estimates, batteries, total = info["b0"], [], [], 0.0
    while True:
        if not np.array_equal(hidden_obs, observed_obs[:-1]):
            raise RuntimeError("The hidden observation must be the observed one without the battery")
        with th.no_grad():
            _, _, _, latents, state = encoder_policy.forward(th.as_tensor(hidden_obs[None]), state, deterministic=True)
            if pilot_policy is not None:
                _, _, _, _, pilot_state = pilot_policy.forward(th.as_tensor(observed_obs[None]), pilot_state, deterministic=True)
                # The encoder's memory carries the action that was executed: the pilot's
                state = ContextualPolicyState(state.encoder_state, pilot_state.prev_actions)
        estimates.append(float(latents[0, 0]))
        batteries.append(float(battery))
        action = state.prev_actions[0].numpy()
        hidden_obs, reward, terminated, truncated, info = hidden.step(action)
        observed_obs, observed_reward, *_ = observed.step(action)
        if observed_reward != reward:
            raise RuntimeError("The two envs must fly the same flight")
        total += float(reward)
        battery = info["b"]
        if terminated or truncated:
            break
    from flight_logging import terminal_outcome

    return {"seed": reset["seed"], "b0": reset["options"]["b0"], "return": total, "length": len(estimates),
            "outcome": terminal_outcome(info), "estimates": estimates, "batteries": batteries}


def measure(a_dir: str, pilot_dir: str, checkpoint: str, view: str) -> dict:
    """The flights of one A checkpoint, flown by itself (own) or by the 0c pilot of the same seed (pilot)."""
    import torch as th
    from omegaconf import OmegaConf

    import main_battery_plane_contextual_ppo as training
    import report_step0
    from contextual_ppo.torch_layers import XLSTMRolloutEncoder

    th.set_num_threads(1)
    cfg = OmegaConf.load(Path(a_dir) / "config.yaml")
    model = training.ContextualPPO.load(Path(a_dir) / checkpoint, device="cpu")
    if not isinstance(model.policy.encoder, XLSTMRolloutEncoder) or model.policy.context_head != "point":
        raise ValueError(f"Not experiment A's point encoder: {a_dir}")
    pilot = None
    if view == "pilot":
        pilot_cfg = OmegaConf.load(Path(pilot_dir) / "config.yaml")
        pilot = training.ContextualPPO.load(Path(pilot_dir) / "final_model.zip", device="cpu")
        if not isinstance(pilot.policy, training.TrueBatteryPolicy):
            raise ValueError(f"Not a 0c pilot: {pilot_dir}")
        if training.env_kwargs_from(pilot_cfg) != training.env_kwargs_from(cfg):
            raise ValueError("The pilot was trained on other physics")
    schedule = list(training.iter_eval_resets(cfg.evaluation.reset_grid))
    flights = [fly(model.policy, pilot.policy if pilot else None, training.env_kwargs_from(cfg), reset) for reset in schedule]
    # The flights of a final model are those of its own final validation (this check needs the same flights)
    reference_dir = a_dir if view == "own" else pilot_dir
    if checkpoint == "final_model.zip":
        saved = json.loads((Path(reference_dir) / "final_validation.json").read_text())
        if ([f["length"] for f in flights] != saved["episode_lengths"]
                or not np.allclose([f["return"] for f in flights], saved["episode_rewards"], atol=1e-6)):
            raise RuntimeError(f"These flights differ from the final validation of {reference_dir}")
        report_step0.evaluation_protocol(cfg, "A")
    return {"a_run": a_dir, "pilot_run": pilot_dir if view == "pilot" else None, "seed": int(cfg.training.seed),
            "checkpoint": checkpoint, "timesteps": int(model.num_timesteps), "view": view, "flights": flights}


def errors_at(flights: list[dict], step: int) -> np.ndarray:
    return np.array([f["estimates"][step] - f["batteries"][step] for f in flights if f["length"] > step])


def summarize(flights: list[dict]) -> dict:
    """Error (estimate - true battery) over the flight: RMS and mean at given steps and in windows, and per starting battery."""
    def stats(errors):
        return {"rms": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
                "bias": float(np.mean(errors)) if len(errors) else None, "n": int(len(errors))}

    by_step = {str(k): stats(errors_at(flights, k)) for k in STEPS}
    windows = {}
    for start, end in WINDOWS:
        errors = np.concatenate([np.array(f["estimates"][start:end]) - np.array(f["batteries"][start:end]) for f in flights])
        windows[f"{start}-{end}"] = stats(errors)
    by_b0 = {}
    for b0 in BATTERIES:
        group = [f for f in flights if f["b0"] == b0]
        by_b0[f"{b0:g}"] = {
            "estimate_at": {str(k): float(np.mean([f["estimates"][k] for f in group if f["length"] > k]))
                            for k in STEPS if any(f["length"] > k for f in group)},
            "battery_at": {str(k): float(np.mean([f["batteries"][k] for f in group if f["length"] > k]))
                           for k in STEPS if any(f["length"] > k for f in group)},
            "completion_rate": sum(f["outcome"] == "completed" for f in group) / len(group),
        }
    # Smooth or jittery: correlation of the error with itself 25 steps later, within flights
    pairs = [(np.array(f["estimates"]) - np.array(f["batteries"])) for f in flights if f["length"] > 50]
    lagged = np.concatenate([e[:-25] for e in pairs]), np.concatenate([e[25:] for e in pairs])
    return {"by_step": by_step, "windows": windows, "by_b0": by_b0,
            "error_correlation_25_steps": float(np.corrcoef(*lagged)[0, 1]),
            "completion_rate": sum(f["outcome"] == "completed" for f in flights) / len(flights)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a-runs", nargs="+", required=True, help="A rerun directories (runs_cppo/<id>)")
    parser.add_argument("--pilot-runs", nargs="+", required=True, help="0c run directories of the same training seeds")
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
        jobs += [(directory, pilots[seed], checkpoint, view) for checkpoint in CHECKPOINTS for view in VIEWS]
    # Cross pairs: every final A encoder riding along with every pilot (is it the encoder or the pilot's flights?)
    seeds = {directory: int(OmegaConf.load(Path(directory) / "config.yaml").training.seed) for directory in args.a_runs}
    cross = [(directory, pilot, "final_model.zip", "pilot") for directory in args.a_runs for seed, pilot in sorted(pilots.items())
             if seed != seeds[directory]]
    print(f"{len(args.a_runs)} A runs x {len(CHECKPOINTS)} checkpoints x {VIEWS}, plus {len(cross)} cross pairs, "
          "validation flights only", flush=True)
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(measure, *zip(*jobs)))
        cross_results = list(pool.map(measure, *zip(*cross))) if cross else []
    summaries = {}
    for checkpoint in CHECKPOINTS:
        for view in VIEWS:
            group = [r for r in results if r["checkpoint"] == checkpoint and r["view"] == view]
            summaries[f"{view}/{checkpoint}"] = {"seeds": sorted(r["seed"] for r in group),
                                                 **summarize([f for r in group for f in r["flights"]])}
    pilot_seeds = {directory: seed for seed, directory in pilots.items()}
    pairs = {}
    for r in [r for r in results if r["view"] == "pilot" and r["checkpoint"] == "final_model.zip"] + cross_results:
        pairs[f"A s{r['seed']} x 0c s{pilot_seeds[r['pilot_run']]}"] = {
            str(k): float(np.sqrt(np.mean(errors_at(r["flights"], k) ** 2))) for k in STEPS}
    pairs = dict(sorted(pairs.items()))
    args.output.write_text(json.dumps({"steps": STEPS, "summaries": summaries, "pairs": pairs, "runs": results,
                                       "cross_runs": cross_results}, allow_nan=False) + "\n")
    for key, summary in summaries.items():
        steps = " ".join(f"{k}:{summary['by_step'][str(k)]['rms']:.3f}" if summary["by_step"][str(k)]["rms"] is not None
                         else f"{k}:-" for k in STEPS)
        print(f"{key:45s} RMS error at flight step {steps} | completed {summary['completion_rate']:.0%}", flush=True)
    for pair, by_step in pairs.items():
        print(f"final encoder {pair}: RMS error at flight step " + " ".join(f"{k}:{by_step[str(k)]:.3f}" for k in STEPS))


if __name__ == "__main__":
    main()
