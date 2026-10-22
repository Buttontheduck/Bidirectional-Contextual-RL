"""Plain Stable-Baselines3 PPO on MyoSuite's elbow task, with or without muscle fatigue.

Self-contained: it imports nothing from the rest of the repository. Run it with the MyoSuite environment (built on the
carlos conda env):

    /root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0                    # no fatigue, 2 s
    /root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0 --env-id myoFatiElbowPose1D6MRandom-v0 --fatigue-reset random
    /root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0 --env-id myoFatiElbowPose1D6MRandom-v0 --episode-steps 3000
    /root/venvs/carlos-myo/bin/python contextual_environments/myo/train_ppo.py --seed 0 --steps 20000 --wandb-mode disabled   # smoke test

Options of the task:
- --episode-steps: the time limit in control steps of 0.02 s (default: the registration's 100 = 2 s).
- --fatigue-reset (fatigue tasks only): fresh muscles at every reset (MyoSuite's default) or a random starting fatigue
  per muscle (FatigueWrapper's fatigue_reset_random: the fatigued share MF uniform in [0, 1]), drawn from the env's own
  generator, so a reset seed fixes it.
- --observe-fatigue (fatigue tasks only): a control that appends the six muscles' fatigued shares MF to the observation,
  like BatteryPlane's observed mode. Without it the fatigue stays hidden.
- --target-every N --target-range LO HI: repeated targets. Every N control steps (and at the start) a new target angle,
  uniform in [LO, HI] degrees, from a generator seeded at each reset by the env's own; the arm and the fatigue carry on.
- --load-kg M: the hand holds a weight of M kg: its weight acts at the wrist site at every control step.
- --gamma: the discount (default 0.99).
- --save-checkpoints: the model at every validation, in checkpoints/model_<steps>.zip (fatigue_gate.py --hidden-model best).

PPO settings are BatteryPlane's plain-PPO baseline (0a): 8 environments x 512 steps per update, minibatches of 256,
10 epochs, learning rate 3e-4, gamma 0.99, GAE 0.95, clip 0.2, entropy 0, value loss 0.5, gradient clipping 0.5, actor and
critic with two hidden layers of 256 (tanh), orthogonal initialization, rewards x 0.1 for training only (Monitor records
the environment's own returns).

Seeds: training environment i starts from reset seed seed + i (SB3 seeds the VecEnv when learning starts), so 0-9 for
seeds 0-2. Validation: every --eval-every steps, 30 episodes with reset seeds 1000-1029 and deterministic actions. The
final-test seeds 2000-2029 are reserved and not used here.

Per validation: mean return and return per step, the share of episodes solved at their last step (elbow within the
task's threshold, 0.175 rad = 10 deg), the share of all steps solved (and per 10 s window in longer episodes), the mean
final angle error in degrees, and on fatigue tasks the fatigued shares at the start and end; every episode's numbers go
to validation.json.

Writes to contextual_environments/myo/runs/<run name>_<time>/ (gitignored): config.json, validation.json (every
validation), final_model.zip, summary.json, and TensorBoard logs (synced to W&B project contextual-myo unless
--wandb-mode disabled).
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
VALIDATION_SEEDS = list(range(1000, 1030))
FINAL_TEST_SEEDS = list(range(2000, 2030))     # reserved: never used for training, tuning or selection
WINDOW = 500                                   # control steps per window in longer episodes (10 s)


def find_fatigue(env):
    """The FatigueWrapper inside env, or None; set its options on it directly, the outer wrapper does not pass them on."""
    while type(env).__name__ != "FatigueWrapper":
        if not hasattr(env, "env"):
            return None
        env = env.env
    return env


def _hand_load_wrapper():
    import gymnasium
    import mujoco

    class HandLoad(gymnasium.Wrapper):
        """A weight held in the hand: its gravity force acts at the wrist site, recomputed at every control step."""

        def __init__(self, env, load_kg: float):
            super().__init__(env)
            model = env.unwrapped.model
            self.load_kg, self.site = load_kg, model.site("wrist").id
            self.body = int(model.site_bodyid[self.site])

        def apply_load(self):
            model, data = self.env.unwrapped.model, self.env.unwrapped.data
            force = np.zeros(model.nv)
            mujoco.mj_applyFT(model, data, np.array([0.0, 0.0, -9.81 * self.load_kg]), np.zeros(3),
                              data.site_xpos[self.site].copy(), self.body, force)
            data.qfrc_applied[:] = force

        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            self.apply_load()
            return result

        def step(self, action):
            self.apply_load()
            return self.env.step(action)

    return HandLoad


def _repeated_targets_wrapper():
    import gymnasium

    class RepeatedTargets(gymnasium.Wrapper):
        """A new target angle every `every` control steps (and at the start), uniform in `range_deg`; the observation
        returned at a switch already shows the new target."""

        def __init__(self, env, every: int, range_deg: tuple[float, float]):
            super().__init__(env)
            self.every, self.low, self.high = every, np.radians(range_deg[0]), np.radians(range_deg[1])
            self.rng, self.t = None, 0

        def new_target(self, obs):
            task = self.env.unwrapped
            task.target_jnt_value = np.array([self.rng.uniform(self.low, self.high)])
            return np.asarray(task.get_obs(), dtype=obs.dtype)

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            self.rng = np.random.default_rng(int(self.env.unwrapped.np_random.integers(2**63)))
            self.t = 0
            return self.new_target(obs), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.t += 1
            if self.t % self.every == 0 and not (terminated or truncated):
                obs = self.new_target(obs)
            return obs, reward, terminated, truncated, info

    return RepeatedTargets


def make_env(env_id: str, episode_steps: int | None = None, fatigue_reset: str = "fresh", observe_fatigue: bool = False,
             target_every: int | None = None, target_range: tuple[float, float] | None = None, load_kg: float = 0.0):
    """One MyoSuite environment (gymnasium API, time limit from its registration unless episode_steps is given)."""
    import gymnasium
    from myosuite.utils import gym

    env = gym.make(env_id) if episode_steps is None else gym.make(env_id, max_episode_steps=episode_steps)
    fatigue = find_fatigue(env)
    if fatigue is None and (fatigue_reset != "fresh" or observe_fatigue):
        raise ValueError(f"{env_id} has no fatigue: --fatigue-reset random and --observe-fatigue need a myoFati task")
    if fatigue is not None:
        fatigue.set_fatigue_reset_random(fatigue_reset == "random")
    if load_kg:
        env = _hand_load_wrapper()(env, load_kg)
    if target_every is not None:
        env = _repeated_targets_wrapper()(env, target_every, tuple(target_range))
    if observe_fatigue:
        class ObservedFatigue(gymnasium.ObservationWrapper):
            """Appends the muscles' fatigued shares MF (hidden otherwise) to the observation."""

            def __init__(self, inner):
                super().__init__(inner)
                low, high = inner.observation_space.low, inner.observation_space.high
                n = fatigue.muscle_fatigue.MF.shape[0]
                self.observation_space = gymnasium.spaces.Box(np.concatenate([low, np.zeros(n)]),
                                                              np.concatenate([high, np.ones(n)]),
                                                              dtype=inner.observation_space.dtype)

            def observation(self, observation):
                return np.concatenate([observation, fatigue.muscle_fatigue.MF]).astype(self.observation_space.dtype)

        env = ObservedFatigue(env)
    return env


def train_env_factory(env_options: dict, reward_scale: float):
    def _init():
        from gymnasium.wrappers import TransformReward
        from stable_baselines3.common.monitor import Monitor

        env = Monitor(make_env(**env_options))              # episode returns in the environment's units
        if reward_scale != 1.0:
            env = TransformReward(env, lambda reward: reward_scale * reward)
        return env

    return _init


def evaluate(model, env_options: dict, seeds: list[int]) -> dict:
    """Deterministic episodes from the given reset seeds."""
    env = make_env(**env_options)
    fatigue = find_fatigue(env)
    episodes = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        start_mf = fatigue.muscle_fatigue.MF.tolist() if fatigue is not None else None
        total, solved, done, info = 0.0, [], False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            solved.append(bool(np.asarray(info["solved"]).item()))
            done = terminated or truncated
        episodes.append({
            "seed": seed, "return": total, "steps": len(solved), "solved_at_end": solved[-1],
            "solved_share_of_steps": float(np.mean(solved)),
            "solved_by_window": ([float(np.mean(solved[i:i + WINDOW])) for i in range(0, len(solved), WINDOW)]
                                 if len(solved) > WINDOW else None),
            "final_error_deg": float(np.degrees(np.abs(np.asarray(info["obs_dict"]["pose_err"])).max())),
            "start_mf": start_mf, "end_mf": fatigue.muscle_fatigue.MF.tolist() if fatigue is not None else None})
    env.close()
    returns = [e["return"] for e in episodes]
    result = {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
              "mean_reward_per_step": float(np.mean([e["return"] / e["steps"] for e in episodes])),
              "solved_at_end": float(np.mean([e["solved_at_end"] for e in episodes])),
              "solved_share_of_steps": float(np.mean([e["solved_share_of_steps"] for e in episodes])),
              "final_error_deg": float(np.mean([e["final_error_deg"] for e in episodes]))}
    if episodes[0]["solved_by_window"] is not None:
        result["solved_by_window"] = np.mean([e["solved_by_window"] for e in episodes], axis=0).tolist()
    if fatigue is not None:
        result.update({"start_mf_mean": float(np.mean([e["start_mf"] for e in episodes])),
                       "end_mf_mean": float(np.mean([e["end_mf"] for e in episodes])),
                       "end_mf_max": float(np.max([e["end_mf"] for e in episodes]))})
    return {**result, "episodes": episodes}


def run_name(env_id: str, episode_steps: int | None, fatigue_reset: str, observe_fatigue: bool, reward_scale: float,
             target_every: int | None = None, target_range: tuple[float, float] | None = None, load_kg: float = 0.0,
             gamma: float = 0.99) -> str:
    """The W&B group, e.g. PPO_ElbowPose1D6MRandom_rs0.1, PPO_FatiElbowPose1D6MRandom_randomMF_T3000_rs0.1 or
    PPO_FatiElbowPose1D6MRandom_fresh_T3000_rep100_t0-110_load5_g0.999_rs0.01."""
    parts = ["PPO", env_id.removeprefix("myo").removesuffix("-v0")]
    if env_id.startswith("myoFati"):
        parts.append("randomMF" if fatigue_reset == "random" else "fresh")
    if observe_fatigue:
        parts.append("seeMF")
    if episode_steps is not None:
        parts.append(f"T{episode_steps}")
    if target_every is not None:
        parts += [f"rep{target_every}", f"t{target_range[0]:g}-{target_range[1]:g}"]
    if load_kg:
        parts.append(f"load{load_kg:g}")
    if gamma != 0.99:
        parts.append(f"g{gamma:g}")
    return "_".join(parts + [f"rs{reward_scale:g}"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default="myoElbowPose1D6MRandom-v0")
    parser.add_argument("--episode-steps", type=int, default=None)
    parser.add_argument("--fatigue-reset", choices=["fresh", "random"], default="fresh")
    parser.add_argument("--observe-fatigue", action="store_true")
    parser.add_argument("--target-every", type=int, default=None)
    parser.add_argument("--target-range", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--load-kg", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save-checkpoints", action="store_true", help="save the model at every validation")
    parser.add_argument("--suffix", default="", help="appended to the W&B group and name, e.g. _ckpt1M")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default="contextual-myo")
    args = parser.parse_args(argv)
    if args.seed + args.n_envs > min(VALIDATION_SEEDS):
        parser.error("The training reset seeds (seed + i) would reach the validation seeds")
    if not args.env_id.startswith("myoFati") and (args.fatigue_reset != "fresh" or args.observe_fatigue):
        parser.error("--fatigue-reset random and --observe-fatigue need a myoFati task")
    if (args.target_every is None) != (args.target_range is None):
        parser.error("--target-every and --target-range go together")

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(args.threads)
    env_options = {"env_id": args.env_id, "episode_steps": args.episode_steps, "fatigue_reset": args.fatigue_reset,
                   "observe_fatigue": args.observe_fatigue, "target_every": args.target_every,
                   "target_range": args.target_range, "load_kg": args.load_kg}
    group = run_name(args.env_id, args.episode_steps, args.fatigue_reset, args.observe_fatigue, args.reward_scale,
                     args.target_every, args.target_range, args.load_kg, args.gamma) + args.suffix
    name = f"{group}_s{args.seed}"
    run_dir = HERE / "runs" / f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    config = {**vars(args), "name": name, "group": group, "validation_seeds": VALIDATION_SEEDS,
              "ppo": {"learning_rate": 3e-4, "n_steps": 512, "batch_size": args.batch_size, "n_epochs": 10, "gamma": args.gamma,
                      "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
                      "net_arch": [256, 256], "activation": "tanh", "ortho_init": True, "log_std_init": 0.0}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(project=args.wandb_project, name=name, group=group, config=config, mode=args.wandb_mode,
                         sync_tensorboard=True, dir=str(run_dir))

    env = DummyVecEnv([train_env_factory(env_options, args.reward_scale) for _ in range(args.n_envs)])
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
                result = evaluate(self.model, env_options, VALIDATION_SEEDS)
                history.append({"timesteps": int(self.num_timesteps), **result})
                if args.save_checkpoints:
                    self.model.save(run_dir / "checkpoints" / f"model_{self.num_timesteps}")
                for key, value in result.items():
                    if isinstance(value, float):
                        self.logger.record(f"validation/{key}", value)
                for i, value in enumerate(result.get("solved_by_window", [])):
                    self.logger.record(f"validation/solved_window_{i * WINDOW // 50}s", value)
                (run_dir / "validation.json").write_text(json.dumps(history, indent=2) + "\n")
                print(f"[validation] {self.num_timesteps:>9,d} steps: return {result['mean_return']:7.1f}, solved at end "
                      f"{result['solved_at_end']:.0%}, steps solved {result['solved_share_of_steps']:.0%}, "
                      f"final error {result['final_error_deg']:.1f} deg", flush=True)
            return True

    started = time.monotonic()
    model.learn(total_timesteps=args.steps, callback=Validation(), tb_log_name="PPO")
    model.save(run_dir / "final_model")
    final = evaluate(model, env_options, VALIDATION_SEEDS)
    history.append({"timesteps": int(model.num_timesteps), "final": True, **final})
    (run_dir / "validation.json").write_text(json.dumps(history, indent=2) + "\n")
    late = [h["mean_return"] for h in history if h["timesteps"] >= 0.75 * args.steps and not h.get("final")]
    summary = {"final_validation": {k: v for k, v in final.items() if k != "episodes"},
               "late_mean_return": float(np.mean(late)) if late else None, "minutes": (time.monotonic() - started) / 60,
               "run_dir": str(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if run is not None:
        for key, value in summary["final_validation"].items():
            if isinstance(value, float):
                run.summary[f"final_validation/{key}"] = value
        run.summary["late_mean_return"] = summary["late_mean_return"]
        run.finish()


if __name__ == "__main__":
    main()
