"""PPO on Gym4ReaL's MicroGridEnv with the recipe of the project's pretrained PPO_trained, the battery's state of health
shown or hidden: the value-of-context gate for battery health.

Run from anywhere with /root/venvs/carlos-gym4real/bin/python (the script changes into the Gym4ReaL folder, whose
configuration paths are relative):

    /root/venvs/carlos-gym4real/bin/python contextual_environments/microgrid/train_ppo.py --seed 0 --show-soh   # health shown
    /root/venvs/carlos-gym4real/bin/python contextual_environments/microgrid/train_ppo.py --seed 0              # hidden
    ... --random-battery-init      # Gym4ReaL's random starting battery state (health 0.6-1) at every reset
    ... --episodes 1 --wandb-mode disabled                                                                     # smoke test

The environment is unchanged: world_train.yaml through parameter_generator's defaults (2015-2018 data, the 370
training households drawn at every reset, fixed start, reward normalization on), built with make_vec_env as the
project's notebooks do. Shown adds the state of health to the observation (settings["soh"] = True, 11 numbers);
hidden is the default 10, the observation PPO_trained was trained on.

PPO_trained's recipe, read from its checkpoint: Stable-Baselines3 PPO, MlpPolicy with its defaults (actor and critic
with two hidden layers of 64, tanh, orthogonal initialization), 4 environments x 2,048 steps per update, minibatch 64,
10 epochs, gamma 0.99, GAE 0.95, clip 0.2, entropy 0.01, value loss 0.5, gradient clipping 0.5, advantages normalized,
learning rate falling linearly from 1e-3 to 0, and 100 episodes per environment: 35,064 x 4 x 100 = 14,025,600 steps.

Writes to contextual_environments/microgrid/runs/<run name>_<time>/ (gitignored): config.json, checkpoints every 10
episodes per environment, final_model.zip, summary.json, TensorBoard logs (synced to W&B project contextual-microgrid
unless --wandb-mode disabled).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
GYM4REAL = os.environ.get("GYM4REAL_DIR", "/root/venvs/src/gym4ReaL")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show-soh", action="store_true")
    parser.add_argument("--random-battery-init", action="store_true",
                        help="Gym4ReaL's random_battery_init: a random starting battery state (health 0.6-1, charge "
                             "0.2-1, temperature 0-50 C) at every reset")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100, help="episodes per environment (PPO_trained: 100)")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="contextual-microgrid")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(args.threads))
    warnings.filterwarnings("ignore")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env

    torch.set_num_threads(args.threads)
    suffix = ("_randinit" if args.random_battery_init else "") + args.suffix
    group = f"PPO_MG_recipe{suffix}_{'seeSoH' if args.show_soh else 'hidden'}"
    name = f"{group}_s{args.seed}"
    run_dir = HERE / "runs" / f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)

    os.chdir(GYM4REAL)
    import gym4real  # noqa: F401  (registers gym4real/microgrid-v0)
    from gym4real.envs.microgrid.utils import parameter_generator

    params = parameter_generator(world_options="gym4real/envs/microgrid/world_train.yaml",
                                 random_battery_init=True if args.random_battery_init else None)
    params["soh"] = args.show_soh
    envs = make_vec_env("gym4real/microgrid-v0", n_envs=args.n_envs, env_kwargs={"settings": params})
    episode_steps = len(envs.get_attr("generation")[0])
    total = episode_steps * args.n_envs * args.episodes
    config = {**vars(args), "name": name, "group": group, "gym4real_commit": "774d5ce",
              "observation_size": int(envs.observation_space.shape[0]), "total_timesteps": total,
              "ppo": {"policy": "MlpPolicy (pi/vf [64, 64], tanh)", "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
                      "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
                      "max_grad_norm": 0.5, "learning_rate": "linear 1e-3 -> 0"}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(project=args.wandb_project, name=name, group=group, config=config, mode=args.wandb_mode,
                         sync_tensorboard=True, dir=str(run_dir))

    model = PPO("MlpPolicy", envs, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, learning_rate=lambda p: 1e-3 * p,
                seed=args.seed, device="cpu", tensorboard_log=str(run_dir / "tensorboard"), verbose=0)
    checkpoints = CheckpointCallback(save_freq=max(episode_steps * 10, 1), save_path=str(run_dir / "checkpoints"),
                                     name_prefix="model")
    started = time.monotonic()
    model.learn(total_timesteps=total, callback=checkpoints, tb_log_name="PPO", log_interval=5)
    model.save(run_dir / "final_model")
    summary = {"minutes": (time.monotonic() - started) / 60, "num_timesteps": int(model.num_timesteps), "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
