"""Evaluate frozen final checkpoints on reserved episodes after completing the grid."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from main_battery_plane_contextual_ppo import (
    ContextualPPO, configure_cpu_threads, evaluate_schedule, validate_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True, help="Completed run_grid.py output directory")
    args = parser.parse_args()
    grid = args.grid.resolve()
    results = json.loads((grid / "grid_results.json").read_text())
    summary = []
    for run in results["runs"]:
        directory = grid / run["run_dir"]
        report = directory / "final_test.json"
        if report.exists():
            result = json.loads(report.read_text())
            print(f"REUSE final test: {run['experiment']} seed {run['seed']}")
        else:
            cfg = OmegaConf.load(directory / "config.yaml")
            validate_config(cfg)
            configure_cpu_threads(cfg)
            model = ContextualPPO.load(directory / "final_model.zip", device="cpu")
            result = evaluate_schedule(model, cfg, cfg.final_test.reset_grid)
            result["split"] = "final_test"
            with report.open("x") as stream:
                stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        summary.append({"experiment": run["experiment"], "seed": run["seed"], **result})
        print(f"{run['experiment']} seed {run['seed']}: {result['mean_reward']:.3f}")
    (grid / "final_test_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
