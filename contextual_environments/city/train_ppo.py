"""Plain Stable-Baselines3 PPO on one CityLearn building with its battery (city_env.py), the charge observed or hidden.

Self-contained. Run with /root/venvs/carlos-citylearn/bin/python:

    /root/venvs/carlos-citylearn/bin/python contextual_environments/city/train_ppo.py --seed 0 --observe-charge   # observed
    /root/venvs/carlos-citylearn/bin/python contextual_environments/city/train_ppo.py --seed 0                    # hidden
    ... --steps 20000 --wandb-mode disabled                                                                       # smoke test

PPO settings are BatteryPlane's plain-PPO baseline (0a): 8 environments x 512 steps per update, minibatches of 256,
10 epochs, learning rate 3e-4, gamma 0.99, GAE 0.95, clip 0.2, entropy 0, value loss 0.5, gradient clipping 0.5, actor and
critic with two hidden layers of 256 (tanh), orthogonal initialization. Rewards are in $ (minus the cost of the hour),
unscaled (--reward-scale 1): returns are about -7 to -12 per episode.

Seeds: training environment i starts from reset seed seed + i; training episodes come only from training weeks.
Validation: every --eval-every steps, the 80 fixed validation scenarios (validation weeks), deterministic actions;
logged: cost, energy-adjusted cost, starting and ending stored energy, hours empty and full. The final-test scenarios
are reserved and not used here.

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

import numpy as np

HERE = Path(__file__).resolve().parent


def env_factory(observe_charge: bool, reward_scale: float):
    def _init():
        from gymnasium.wrappers import TransformReward
        from stable_baselines3.common.monitor import Monitor

        from city_env import CityBattery

        env = Monitor(CityBattery(observe_charge=observe_charge))          # episode returns in $
        if reward_scale != 1.0:
            env = TransformReward(env, lambda reward: reward_scale * reward)
        return env

    return _init


def validate(model, observe_charge: bool) -> dict:
    from city_env import VALIDATION_SCENARIOS, CityBattery, fly
    from evaluate import summarize

    env = CityBattery(observe_charge=observe_charge)
    episodes = fly(env, lambda env, obs: model.predict(obs, deterministic=True)[0], VALIDATION_SCENARIOS)
    return summarize(episodes)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observe-charge", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="contextual-city")
    parser.add_argument("--suffix", default="", help="appended to the W&B group and name, e.g. _10M")
    args = parser.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(args.threads))
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(args.threads)
    group = f"PPO_B1_h48_cost_{'seeCharge' if args.observe_charge else 'hidden'}_rs{args.reward_scale:g}{args.suffix}"
    name = f"{group}_s{args.seed}"
    run_dir = HERE / "runs" / f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    config = {**vars(args), "name": name, "group": group,
              "ppo": {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 256, "n_epochs": 10, "gamma": 0.99,
                      "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
                      "net_arch": [256, 256], "activation": "tanh", "ortho_init": True, "log_std_init": 0.0}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(project=args.wandb_project, name=name, group=group, config=config, mode=args.wandb_mode,
                         sync_tensorboard=True, dir=str(run_dir))

    env = DummyVecEnv([env_factory(args.observe_charge, args.reward_scale) for _ in range(args.n_envs)])
    ppo = config["ppo"]
    model = PPO("MlpPolicy", env, learning_rate=ppo["learning_rate"], n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
                n_epochs=ppo["n_epochs"], gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"], clip_range=ppo["clip_range"],
                ent_coef=ppo["ent_coef"], vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"], seed=args.seed,
                policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "activation_fn": torch.nn.Tanh,
                               "ortho_init": True, "log_std_init": 0.0},
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
    model.learn(total_timesteps=args.steps, callback=Validation(), tb_log_name="PPO")
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
