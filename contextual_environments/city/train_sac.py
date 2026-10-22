"""Stable-Baselines3 SAC on one CityLearn building with its battery (city_env.py), the charge observed or hidden: is the
charge information unhelpful, or is PPO failing to use it?

Self-contained. Run with /root/venvs/carlos-citylearn/bin/python:

    /root/venvs/carlos-citylearn/bin/python contextual_environments/city/train_sac.py --seed 0 --observe-charge
    ... --steps 5000 --learning-starts 1000 --eval-every 2500 --wandb-mode disabled                 # smoke test

Same task as train_ppo.py: Building_1, cost reward in $ (unscaled), 48-hour episodes from a random hour with a random
starting charge, the time left in the observation, training episodes from training weeks only, validation on the 80
fixed validation scenarios (deterministic actions). The final-test scenarios are reserved and not used here.

SAC settings (Stable-Baselines3's standard ones): one environment, replay buffer of 1M transitions, 10,000 random
steps before learning, batch 256, learning rate 3e-4, gamma 0.99, tau 0.005, one gradient step per environment step,
entropy coefficient tuned automatically (target entropy -1 for the one-number action), actor and two critics with two
hidden layers of 256 (ReLU).

Writes to contextual_environments/city/runs/<run name>_<time>/ (gitignored): config.json, validation.json,
checkpoints/model_<steps>.zip at every validation, final_model.zip, summary.json, TensorBoard logs (synced to W&B
project contextual-city unless --wandb-mode disabled).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observe-charge", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="contextual-city")
    parser.add_argument("--suffix", default="", help="appended to the W&B group and name")
    args = parser.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(args.threads))
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from train_ppo import env_factory, validate

    torch.set_num_threads(args.threads)
    group = f"SAC_B1_h48_cost_{'seeCharge' if args.observe_charge else 'hidden'}_rs{args.reward_scale:g}{args.suffix}"
    name = f"{group}_s{args.seed}"
    run_dir = HERE / "runs" / f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    config = {**vars(args), "algo": "sac", "name": name, "group": group,
              "sac": {"learning_rate": 3e-4, "buffer_size": 1_000_000, "learning_starts": args.learning_starts,
                      "batch_size": 256, "tau": 0.005, "gamma": 0.99, "train_freq": 1, "gradient_steps": 1,
                      "ent_coef": "auto", "net_arch": [256, 256], "activation": "relu"}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(project=args.wandb_project, name=name, group=group, config=config, mode=args.wandb_mode,
                         sync_tensorboard=True, dir=str(run_dir))

    env = DummyVecEnv([env_factory(args.observe_charge, args.reward_scale)])
    sac = config["sac"]
    model = SAC("MlpPolicy", env, learning_rate=sac["learning_rate"], buffer_size=sac["buffer_size"],
                learning_starts=sac["learning_starts"], batch_size=sac["batch_size"], tau=sac["tau"], gamma=sac["gamma"],
                train_freq=sac["train_freq"], gradient_steps=sac["gradient_steps"], ent_coef=sac["ent_coef"],
                policy_kwargs={"net_arch": sac["net_arch"]}, seed=args.seed,
                tensorboard_log=str(run_dir / "tensorboard"), device="cpu", verbose=1)

    history = []

    class Validation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 0

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                result = validate(self.model, args.observe_charge)
                history.append({"timesteps": int(self.num_timesteps), **result})
                for key, value in result.items():
                    if isinstance(value, float):
                        self.logger.record(f"validation/{key}", value)
                (run_dir / "validation.json").write_text(json.dumps(history, indent=2) + "\n")
                self.model.save(run_dir / "checkpoints" / f"model_{self.num_timesteps}")
                print(f"[validation] {self.num_timesteps:>9,d} steps: cost {result['cost']:.3f}, adjusted "
                      f"{result['adjusted_cost']:.3f}, end energy {result['end_energy']:.2f} kWh, empty "
                      f"{result['share_empty']:.0%}, full {result['share_full']:.0%}", flush=True)
            return True

    started = time.monotonic()
    model.learn(total_timesteps=args.steps, callback=Validation(), tb_log_name="SAC", log_interval=50)
    model.save(run_dir / "final_model")
    final = validate(model, args.observe_charge)
    history.append({"timesteps": int(model.num_timesteps), "final": True, **final})
    (run_dir / "validation.json").write_text(json.dumps(history, indent=2) + "\n")
    summary = {"final_validation": {k: v for k, v in final.items() if isinstance(v, float)},
               "minutes": (time.monotonic() - started) / 60, "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if run is not None:
        for key, value in summary["final_validation"].items():
            run.summary[f"final_validation/{key}"] = value
        run.finish()


if __name__ == "__main__":
    main()
