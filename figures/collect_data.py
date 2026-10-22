"""Collect the data of the paper figures (figures/make_figures.py) into figures/data/, from the saved results.

Nothing is trained; the only computation is the offline encoders' error at every flight step on the pilot test flights
(dataset v2, main mix, the test split of train_encoder.py), with each run's best checkpoint:
    clock          the mean training battery at each flight step (train_encoder.clock_estimate)
    ideal          the ideal estimator (runs_encoder/step4/teacher_v2_main, the floor)
    gru_labels     the GRU trained on the true battery (runs_encoder/step4_grid/E1_gru_point_main_s0..s4)
    gru_teacher    the GRU trained on the ideal estimate, T1 (runs_encoder/step4_teacher/E1_gru_teacher_point_main_s0..s4)
    xlstm          A's xLSTM trained on the true battery, given enough passes (runs_encoder/step4/E1_point_main_s0,
                   E1_point_main_s1, E1_point_main_resumed_s2)
Everything else is read from the result files (final test, closed-loop checks, swap test, reports, training metrics).

    python figures/collect_data.py          # about 5 minutes
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(__file__).resolve().parent / "data"
ENC = ROOT / "runs_encoder"
STEP0 = ROOT / "runs_step0"
TEACHER = ENC / "step4" / "teacher_v2_main"
GRU_LABELS = [ENC / "step4_grid" / f"E1_gru_point_main_s{s}" for s in range(5)]
GRU_TEACHER = [ENC / "step4_teacher" / f"E1_gru_teacher_point_main_s{s}" for s in range(5)]
XLSTM = [ENC / "step4" / name for name in ("E1_point_main_s0", "E1_point_main_s1", "E1_point_main_resumed_s2")]
FINAL_TEST = STEP0 / "final_test_complete_v2.json"


def flights_with_teacher():
    import train_encoder as T

    flights, _ = T.load(T.DEFAULT_DATA, "main")
    T.load_teacher(TEACHER, flights, T.DEFAULT_DATA, "main", T.TEACHER_STD_FLOOR)
    return flights


def per_step(errors: list[np.ndarray]) -> dict:
    """RMSE and mean error (bias) at each flight step over the flights still flying, and their number."""
    longest = max(len(e) for e in errors)
    sq, total, n = np.zeros(longest), np.zeros(longest), np.zeros(longest)
    for e in errors:
        e = e.astype(np.float64)
        sq[:len(e)] += e ** 2
        total[:len(e)] += e
        n[:len(e)] += 1
    return {"rmse": np.sqrt(sq / n).tolist(), "bias": (total / n).tolist(), "flights": n.astype(int).tolist()}


def encoder_errors(run_dir: str) -> dict:
    """One run's best encoder on the test flights: errors against the true battery at every step."""
    import torch as th

    import recurrent_encoders as R
    import train_encoder as T

    th.set_num_threads(2)
    flights = flights_with_teacher()
    rows = flights.indices("test")
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / "config.json").read_text())
    if (run_dir / "encoder.pt").is_file():
        encoder = R.load_encoder(run_dir / "encoder.pt")
        b0_head = False
    else:
        cfg = T.encoder_config([f"training.seed={settings['seed']}"])
        b0_head = settings["b0_head"]
        encoder = T.make_model(cfg, T.arch_settings(settings), b0_head)
        encoder.load_state_dict(th.load(run_dir / "best_encoder.pt", weights_only=False)["state_dict"])
        encoder.eval()
    errors = []
    order = rows[np.argsort(flights.lengths[rows], kind="stable")]
    with th.no_grad():
        for start in range(0, len(order), 128):
            part = order[start:start + 128]
            estimates, _ = T.predict(encoder, flights.batch(part), b0_head)
            errors += [estimates[j, :flights.lengths[i]].numpy() - flights.batteries[i] for j, i in enumerate(part)]
    result = json.loads((run_dir / "result.json").read_text())
    curve = per_step(errors)
    rmse = float(np.sqrt(np.mean(np.concatenate(errors).astype(np.float64) ** 2)))
    if not np.isclose(rmse, result["test"]["rmse"], atol=2e-4):
        raise RuntimeError(f"{run_dir}: test error {rmse:.5f}, result.json {result['test']['rmse']:.5f}")
    return {"run": run_dir.name, "test_rmse": rmse, **curve}


def offline_curves() -> dict:
    import train_encoder as T

    flights = flights_with_teacher()
    train, test = flights.indices("train"), flights.indices("test")
    clock = T.clock_estimate(flights, train)
    out = {"dt": 0.05,
           "clock": per_step([clock[np.minimum(np.arange(flights.lengths[i]), len(clock) - 1)] - flights.batteries[i] for i in test]),
           "ideal": per_step([flights.teacher_means[i] - flights.batteries[i] for i in test])}
    runs = {"gru_labels": GRU_LABELS, "gru_teacher": GRU_TEACHER, "xlstm": XLSTM}
    jobs = [str(r) for group in runs.values() for r in group]
    with ProcessPoolExecutor(len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        results = dict(zip(jobs, pool.map(encoder_errors, jobs)))
    for name, group in runs.items():
        out[name] = [results[str(r)] for r in group]
    return out


def training_curves() -> dict:
    """Validation error per pass (each run's metrics.jsonl; a resumed run's starts with the original's passes)."""
    def curve(run):
        rows = [json.loads(line) for line in open(run / "metrics.jsonl")]
        return {"run": run.name, "validation_rmse": [r["validation"]["rmse"] for r in rows]}
    floor = json.loads((ENC / "step4" / "floor_v2_main" / "floor.json").read_text())
    return {"gru_labels": [curve(r) for r in GRU_LABELS], "gru_teacher": [curve(r) for r in GRU_TEACHER],
            "xlstm": [curve(r) for r in XLSTM],
            "clock_validation": json.loads((GRU_LABELS[0] / "result.json").read_text())["clock_validation"]["rmse"],
            "floor_validation": floor["validation"]["rmse"]}


def architecture() -> dict:
    """Offline test error of every encoder run (figure 7), with the clock and the floor."""
    def test_rmse(run):
        return json.loads((run / "result.json").read_text())["test"]["rmse"]
    grid = ENC / "step4_grid"
    floor = json.loads((ENC / "step4" / "floor_v2_main" / "floor.json").read_text())
    first = json.loads((GRU_LABELS[0] / "result.json").read_text())
    return {
        "clock": first["clock_test"]["rmse"], "floor": floor["test"]["rmse"],
        "rows": [
            {"label": "xLSTM, stopped early (patience 3)", "kind": "xlstm",
             "values": [test_rmse(ENC / "step4" / n) for n in ("E1_point_main_s2", "E3_point_b0_main_s2")]},
            {"label": "xLSTM, enough passes", "kind": "xlstm",
             "values": [test_rmse(ENC / "step4" / n) for n in ("E1_point_main_s0", "E1_point_main_s1", "E1_point_main_resumed_s2",
                                                                 "E3_point_b0_main_s0", "E3_point_b0_main_s1", "E3_point_b0_main_resumed_s2")]},
            {"label": "GRU", "kind": "gru_labels", "values": [test_rmse(grid / f"E1_gru_point_main_s{s}") for s in range(5)]},
            {"label": "GRU, no time-left input", "kind": "gru_labels", "values": [test_rmse(grid / f"E1_gru_notau_point_main_s{s}") for s in range(5)]},
            {"label": "ConvGRU", "kind": "gru_labels", "values": [test_rmse(grid / f"E1_convgru_point_main_s{s}") for s in range(5)]},
            {"label": "ConvGRU, no time-left input", "kind": "gru_labels", "values": [test_rmse(grid / f"E1_convgru_notau_point_main_s{s}") for s in range(5)]},
            {"label": "GRU, ideal-estimate teacher (T1)", "kind": "gru_teacher", "values": [test_rmse(r) for r in GRU_TEACHER]},
        ]}


def final_test() -> dict:
    """Per condition: the seed scores, and returns and completion by starting battery (final test)."""
    data = json.loads(FINAL_TEST.read_text())
    out = {"summary": data["summary"], "by_b0": {}}
    for name in data["summary"]:
        rows = [r for r in data["runs"] if r["condition"] + (f" ({r['encoder']})" if r["encoder"] else "") == name]
        by = {}
        for r in rows:
            test = r["final_test"]
            for reset, ret, outcome in zip(test["reset_schedule"], test["episode_rewards"], test["episode_outcomes"]):
                by.setdefault(f"{reset['options']['b0']:g}", []).append((ret, outcome == "completed"))
        out["by_b0"][name] = {b: {"return": float(np.mean([v[0] for v in vals])), "completed": float(np.mean([v[1] for v in vals])),
                                  "flights": len(vals)} for b, vals in sorted(by.items())}
        # Per training seed: the mean return and the share completed at each starting battery (3 test flights each)
        per_seed = {}
        for r in sorted(rows, key=lambda r: r["seed"]):
            test = r["final_test"]
            for reset, ret, outcome in zip(test["reset_schedule"], test["episode_rewards"], test["episode_outcomes"]):
                cell = per_seed.setdefault(f"{reset['options']['b0']:g}", {}).setdefault(r["seed"], [])
                cell.append((ret, outcome == "completed"))
        out.setdefault("by_b0_seed", {})[name] = {
            b: {"seeds": sorted(cells), "return": [float(np.mean([v[0] for v in cells[k]])) for k in sorted(cells)],
                "completed": [float(np.mean([v[1] for v in cells[k]])) for k in sorted(cells)]}
            for b, cells in sorted(per_seed.items())}
    return out


def closed_loop() -> dict:
    """Bias (estimate - battery) by starting battery, for the encoder and the ideal estimator on the same PPO flights."""
    out = {}
    for path in (STEP0 / "closed_loop_frozen_gru.json", STEP0 / "closed_loop_frozen_t1.json"):
        data = json.loads(path.read_text())
        for run in data["runs"]:
            key = f"{run['encoder']}|{run['batch_size']}"
            group = out.setdefault(key, {"encoder": run["encoder"], "batch_size": run["batch_size"], "flights": []})
            group["flights"] += data["flights"][run["run_directory"]]
    result = {}
    for key, group in out.items():
        by = {}
        for f in group["flights"]:
            b = np.array(f["batteries"])
            entry = by.setdefault(f"{f['b0']:g}", {"enc": [], "ideal": []})
            entry["enc"].append(np.array(f["estimates"]) - b)
            entry["ideal"].append(np.array(f["ideal"]) - b)
        result[key] = {"encoder": group["encoder"], "batch_size": group["batch_size"],
                       "by_b0": {b: {"encoder_bias": float(np.concatenate(v["enc"]).mean()), "ideal_bias": float(np.concatenate(v["ideal"]).mean()),
                                     "encoder_rmse": float(np.sqrt(np.mean(np.concatenate(v["enc"]) ** 2))),
                                     "ideal_rmse": float(np.sqrt(np.mean(np.concatenate(v["ideal"]) ** 2)))}
                                 for b, v in sorted(by.items())}}
    return result


def swap() -> dict:
    data = json.loads((STEP0 / "swap_true_battery_frozen.json").read_text())
    out = {}
    for r in data["runs"]:
        out.setdefault(r["condition"], {}).setdefault(str(r["seed"]), {})[r["source"]] = r["mean_reward"]
    return out


def joint_and_grid() -> dict:
    """Validation late scores of the joint-training rungs and their references (figure 8), and the A-F grid's final
    validation returns per seed (GRID_PLAN.md, 2M steps, unscaled rewards: not comparable with the rest)."""
    def late(path, condition):
        rows = [r for r in json.loads(Path(path).read_text())["runs"] if r["condition"] == condition]
        return [r["late_mean_reward"] for r in sorted(rows, key=lambda r: r["seed"])]
    joint = {f"J{k}": late(STEP0 / "joint_v2" / f"step0_report_joint_J{k}.json", f"J{k}") for k in range(6)}
    blind = STEP0 / "20261014-020124" / "report_blind_and_A.json"
    references = {"0c_blind": late(blind, "0c_blind"), "A": late(blind, "A"),
                  "0c": late(STEP0 / "20261013-042940" / "step0_report_rs0.1_seeds0-7.json", "0c")}
    grid = {"A": [-56.6, -14.6, -14.7], "B": [36.1, -14.7, -14.2], "C": [100.2, 126.4, 36.3], "D": [-189.4, 14.9, 8.2],
            "E": [45.7, -14.7, -14.6], "F": [-150.2, -13.2, -1.0]}
    return {"joint": joint, "references": references, "grid": grid}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, make in (("final_test", final_test), ("closed_loop", closed_loop), ("swap", swap), ("training_curves", training_curves),
                       ("architecture", architecture), ("joint_and_grid", joint_and_grid), ("offline_curves", offline_curves)):
        (DATA / f"{name}.json").write_text(json.dumps(make()) + "\n")
        print(f"{name}.json", flush=True)


if __name__ == "__main__":
    main()
