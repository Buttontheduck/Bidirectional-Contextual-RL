"""
BatteryPlane v3 -- Gymnasium environment with an AUTOGENIC battery context.

A 2D point-mass plane with a lift motor, a cruise motor, a wing and a battery.
The battery charge b is the context. It drifts only because of the agent's own thrust, and it sets the
cruise motor's thrust cap. Nothing outside the agent moves it: an autogenic context in the sense of
Biedenkapp (2026) -- endogenous, agent-driven, evolving smoothly within the episode.

State:        y, v_x, v_y, b, tau          (b = charge in [0,1], tau = time left in [0,1])
Observation:  [y/y_ref, (v_x + noise)/v_ref, v_y/v_ref, tau]   + [b] when b is visible (see modes)
Action:       (a_x, a_y) in [-1, 1]^2      a_x -> cruise motor, a_y -> climb around level flight

Air / wing:   rho(y) = exp(-y/H)             L = min(k_L rho max(v_x,0)^2, g)  D = k_D rho v_x |v_x|   (a wing flying backwards does not lift)
Lift motor:   ay = clip(g - L + (a_max/2) a_y, 0, a_max)            a_y = 0 is level flight at any speed
Cruise motor: ax = clip(a_max a_x, -cap, +cap),  cap = h_min + (h_max - h_min) * b      (follows the CURRENT b)
Motion:       v_x += dt (ax - D);  v_y += dt (ay + L - g - c v_y);  y += dt v_y;  ceiling: y <= y_max, v_y <= 0
Battery:      b -= (ax^2 + ay^2) / g^2 * dt / E           (hovering costs 1/E per second; delivered thrust, after the cap)
Reward:       clip((v_x - v_min) / (v_goal - v_min), 0, 1)^2  +  dense_w * clip(v_x / v_goal, 0, 1)^2,   minus C on failure
Terminated:   y <= 0 or b <= 0 (failure, -C)  or  t == T (mission complete, no penalty).
Truncated:    never. tau is observed, so the horizon is part of the MDP and t == T is a real terminal state
              (Pardo et al. 2018): bootstrapping from it would regress toward a state that is never a source
              state in the replay buffer. terminal_at_T=False restores the v1 behaviour (truncated at T).

Observation modes -- a 2 x 2: is b visible?  x  is the speed reading noisy?
    hidden     b hidden,  noisy speed     THE TEST: infer your own charge from how the plane responds to you
    clean      b hidden,  exact speed     isolates inference from filtering
    observed   b visible, noisy speed     context-aware policy pi(s, c)
    full       b visible, exact speed     plain MDP oracle

Information about b reaches a hidden-mode agent only through v_x, and only while the throttle is saturated
(below the cap the dynamics do not depend on b at all). Holding v_goal with a slack cap hides b: the agent
must carry its estimate forward from its own action history. Probing = running at the cap = spending charge,
so observing the context moves it. The b, b0, cap and saturated entries of info are privileged diagnostics
for logging and evaluation; never feed them to a hidden-mode policy or encoder.

Design rules (describe() checks the last three by SIMULATION -- static inequalities no longer hold under a
moving cap):
    v_min >= v*(y_max)                 efficient cruise never earns, at any altitude
    h_max < D_ground(v_min)            the reward band is unreachable at ground level even fully charged
    b_goal inside b0_range             full reward needs a healthy battery (b_goal = charge whose cap equals D(v_goal))
    always-sprint dies for the bottom of b0_range        rationing has to be learned, not just "command max"
    a full battery holds v_goal for a substantial part of the mission      the reward is exploitable at all
value_of_context() adds the check that matters for a context benchmark: how much return a policy that KNOWS b
earns beyond the best policy that cannot see b. If that gap is small, no algorithm can show a hidden-vs-full
difference. Rule of thumb used to pick b0_range: put its top where always-sprint starts to survive, so every
episode needs rationing (with E=100 that is b0 = 0.7).

Defaults (validated 2026-09-12): T=700 x 0.05 s = 35 s, E=100 hover-s, cap 11.5..18, band 52..62, sigma=5,
b0 ~ U[0.2, 0.7], dense_w=1.0, crash_penalty=300. All five rules pass; value of context ~36% of the oracle return.
(History: E=130 / b0 ~ U[0.2, 1] gave a ~9% gap and no gap at all above b0 = 0.55; E=80 broke the last rule.)

Evaluation controls (no change to the policy or learning algorithm):
    obs, info = env.reset(seed=0, options={"b0": 0.2, "vx_obs_noise": 2.0, "y0": 100.0, "vx0": 15.0})
    # b0 is the exact starting charge; vx_obs_noise is the speed noise std in m/s; y0/vx0 fix the start state.
    # Omitting an option restores its constructor default (random charge / random start state by default).
    # "clean" and "full" modes always use zero speed noise.

To connect these controls to an evaluation script's argparse parser:
    from gymnasium.envs.classic_control.battery_plane import add_eval_args, iter_eval_resets
    add_eval_args(parser)  # before parser.parse_args()
    args = parser.parse_args()
    for reset_kwargs in iter_eval_resets(args):
        obs, info = env.reset(**reset_kwargs)
        # Run one complete episode here with the existing evaluation policy.

Example script arguments (8 episodes; seeds vary fastest):
    --eval-seeds 0 1 --eval-initial-battery 0.2 0.9 --eval-vx-obs-noise 2 5 --eval-y0 100
The lists can have any length. The episode count is the product of their lengths.
Episode order: battery (outermost), noise, y0, vx0, seed (innermost).
"""
import math
import numpy as np

try:                                   # gymnasium is only needed for the Env wrapper and register()
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                    # the numpy core below works without it
    gym = None

# name -> (b visible, speed noise on)
OBS_MODES = {"hidden": (False, True), "clean": (False, False), "observed": (True, True), "full": (True, False)}


def add_eval_args(parser):
    """Add optional evaluation grid arguments to an existing argparse parser.

    Only the evaluation loop consuming iter_eval_resets(args) uses these settings.
    Without battery/noise arguments, the environment's constructor defaults apply.
    Returns the parser; does not parse arguments or run any episodes.
    """
    import argparse

    def charge(value):
        value = float(value)
        if not (np.isfinite(value) and 0.0 < value <= 1.0):
            raise argparse.ArgumentTypeError("initial battery must be finite and in (0, 1]")
        return value

    def noise(value):
        value = float(value)
        if not (np.isfinite(value) and value >= 0.0):
            raise argparse.ArgumentTypeError("vx_obs_noise must be finite and non-negative")
        return value

    def seed(value):
        value = int(value)
        if value < 0:
            raise argparse.ArgumentTypeError("evaluation seeds must be non-negative integers")
        return value

    group = parser.add_argument_group("BatteryPlane evaluation")
    group.add_argument("--eval-seeds", type=seed, nargs="+", default=[0],
                       help="Seeds to repeat for every battery/noise combination (default: 0).")
    group.add_argument("--eval-initial-battery", type=charge, nargs="+", default=None,
                       help="Exact initial charges in (0, 1]; omitted: use the configured charge distribution.")
    group.add_argument("--eval-vx-obs-noise", type=noise, nargs="+", default=None,
                       help="Speed noise standard deviations in m/s; omitted: use the configured noise.")
    group.add_argument("--eval-y0", type=float, nargs="+", default=None,
                       help="Exact start altitudes in m (0 < y0 <= y_max); omitted: random start altitude.")
    group.add_argument("--eval-vx0", type=float, nargs="+", default=None,
                       help="Exact start speeds in m/s; omitted: random start speed.")
    return parser


def iter_eval_resets(args):
    """Yield Gymnasium reset kwargs for each battery, noise, y0, vx0, seed combination.

    Accepts arguments from add_eval_args() or any object / mapping with the same
    attribute names (eval_seeds, eval_initial_battery, eval_vx_obs_noise, and the
    optional eval_y0, eval_vx0). Seeds vary fastest, so the same seeds are reused
    for every condition; battery varies slowest. Each yielded dictionary starts one
    episode; the caller runs its evaluation policy until terminated or truncated.
    """
    from itertools import product

    def get(name):
        value = args.get(name) if hasattr(args, "get") else getattr(args, name, None)
        return list(value) if value is not None else [None]

    batteries, noises, y0s, vx0s = get("eval_initial_battery"), get("eval_vx_obs_noise"), get("eval_y0"), get("eval_vx0")
    seeds = get("eval_seeds")
    if seeds == [None]:
        raise ValueError("eval_seeds must be a non-empty list of seeds")
    for b0, noise, y0, vx0, seed in product(batteries, noises, y0s, vx0s, seeds):
        options = {}
        if b0 is not None:
            options["b0"] = b0
        if noise is not None:
            options["vx_obs_noise"] = noise
        if y0 is not None:
            options["y0"] = y0
        if vx0 is not None:
            options["vx0"] = vx0
        yield {"seed": seed, "options": options}


# ====================================================================================== scripted policies
# All of them read the TRUE state (privileged). They are baselines and sanity checks, not agents.
def _climb(env, margin=5.0):
    return np.where(env.y < env.y_max - margin, 1.0, 0.0)


def policy_command_max(env):
    """Climb at full lift, then command maximum cruise thrust forever. The cap does the rest."""
    return np.stack([np.ones(env.n), _climb(env)], axis=1)


def policy_hold_goal(env, gain=2.0, overshoot=0.5):
    """Climb, then throttle to hold v_goal (+ a hair, so the clipped reward is exactly 1; never pays to brake)."""
    ax = (env.drag(env.y, env.vx) + gain * (env.v_goal + overshoot - env.vx)) / env.a_max
    return np.stack([np.clip(ax, 0.0, 1.0), _climb(env)], axis=1)


def policy_cruise(env, gain=2.0):
    """Survive: level flight at the cheapest speed v*(y) for the current altitude."""
    ax = (env.drag(env.y, env.vx) + gain * (env.v_star(env.y) - env.vx)) / env.a_max
    return np.stack([np.clip(ax, 0.0, 1.0), np.zeros(env.n)], axis=1)


def policy_budgeted(b_stop):
    """Oracle-ish rationing: hold v_goal while b > b_stop, then cruise. Uses the true charge."""
    def pol(env):
        sprint = (env.b > b_stop)[:, None]
        return np.where(sprint, policy_hold_goal(env), policy_cruise(env))
    pol.__name__ = f"budgeted(b_stop={b_stop})"
    return pol


# ====================================================================================== vectorized core
class BatteryPlaneVec:
    """Vectorized numpy core (n planes in parallel). The gym wrapper below uses n=1."""

    def __init__(
        self,
        n_envs: int = 1,
        observation_mode: str = "hidden",
        # context: initial charge. Continuous by default; pass soc_choices for a discrete set (e.g. (0.2, 0.9)).
        b0_range=(0.2, 0.7),            # top = where always-sprint starts to survive (see value_of_context)
        soc_choices=None,
        soc_probs=None,
        # time
        T: int = 700,                   # steps
        dt: float = 0.05,               # seconds per step  (episode = 35 s)
        # physics
        g: float = 10.0,
        a_max: float = 20.0,            # lift motor max accel, and the largest cruise command
        c_y: float = 0.25,              # vertical damping
        y_max: float = 200.0,           # ceiling
        H: float = None,                # density scale height; default y_max/ln2 -> 50% density at the ceiling
        k_L: float = 0.015,             # wing lift coefficient
        k_D: float = 0.008,             # parasitic drag coefficient
        # battery -> cruise motor cap
        h_min: float = 11.5,            # cap at b = 0
        h_max: float = 18.0,            # cap at b = 1
        cap_on: str = "b",              # "b": cap follows the current charge (autogenic, default)
                                        # "b0": cap frozen at the starting charge (allogenic control for ablations)
        E_hover_s: float = 100.0,       # seconds a full battery can hover (130 -> 100: a full battery must ration too)
        # reward
        v_min: float = 52.0,
        v_goal: float = 62.0,
        dense_w: float = 1.0,           # dense speed term, gives SAC a gradient below the band; 0 to disable
        crash_penalty: float = 300.0,
        # observation
        vx_obs_noise: float = 5.0,      # std of the speed reading (m/s); forced to 0 in "clean" and "full"
        y_ref: float = 200.0,
        v_ref: float = 60.0,
        # initial state
        y0_range=(30.0, 120.0),
        vx0_range=(0.0, 30.0),
        seed=None,                      # None -> fresh entropy; int for reproducibility
        auto_reset: bool = True,        # True: finished planes restart inside step() (vectorized training)
                                        # False: step() returns the final observation (Gymnasium semantics)
        terminal_at_T: bool = True,     # True: t == T is terminated (time-aware agent). False: truncated (v1 behaviour)
    ):
        self._kw = dict(observation_mode=observation_mode, b0_range=b0_range, soc_choices=soc_choices,
                        soc_probs=soc_probs, T=T, dt=dt, g=g, a_max=a_max, c_y=c_y, y_max=y_max, H=H, k_L=k_L,
                        k_D=k_D, h_min=h_min, h_max=h_max, cap_on=cap_on, E_hover_s=E_hover_s, v_min=v_min,
                        v_goal=v_goal, dense_w=dense_w, crash_penalty=crash_penalty, vx_obs_noise=vx_obs_noise,
                        y_ref=y_ref, v_ref=v_ref, y0_range=y0_range, vx0_range=vx0_range, terminal_at_T=terminal_at_T)
        # ---- validation
        if observation_mode not in OBS_MODES:
            raise ValueError(f"observation_mode must be one of {list(OBS_MODES)}, got {observation_mode!r}")
        if cap_on not in ("b", "b0"):
            raise ValueError("cap_on must be 'b' (autogenic, default) or 'b0' (frozen cap, allogenic control)")
        if not (isinstance(n_envs, (int, np.integer)) and n_envs >= 1):
            raise ValueError("n_envs must be a positive integer")
        if not (isinstance(T, (int, np.integer)) and T >= 1):
            raise ValueError("T must be a positive integer")
        for name, v in (("dt", dt), ("g", g), ("a_max", a_max), ("y_max", y_max), ("E_hover_s", E_hover_s),
                        ("y_ref", y_ref), ("v_ref", v_ref), ("k_L", k_L), ("k_D", k_D), ("h_min", h_min)):
            if not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be positive and finite")
        if H is not None and not (np.isfinite(H) and H > 0):
            raise ValueError("H must be positive and finite")
        if not (0 <= v_min < v_goal and np.isfinite(v_goal)):
            raise ValueError("require 0 <= v_min < v_goal")
        if not (h_max >= h_min):
            raise ValueError("require h_max >= h_min")
        if not (c_y >= 0 and dense_w >= 0 and crash_penalty >= 0 and vx_obs_noise >= 0):
            raise ValueError("c_y, dense_w, crash_penalty and vx_obs_noise must be non-negative")
        if not np.isfinite(vx_obs_noise):
            raise ValueError("vx_obs_noise must be finite")
        if soc_choices is None:
            lo, hi = float(b0_range[0]), float(b0_range[1])
            if not (0.0 < lo <= hi <= 1.0):
                raise ValueError("b0_range must satisfy 0 < lo <= hi <= 1")
            self.soc_choices, self.soc_probs = None, None
        else:
            self.soc_choices = np.asarray(soc_choices, float)
            if self.soc_choices.ndim != 1 or np.any(self.soc_choices <= 0) or np.any(self.soc_choices > 1):
                raise ValueError("soc_choices must be a 1-D list of charges in (0, 1]")
            self.soc_probs = (np.full(len(self.soc_choices), 1.0 / len(self.soc_choices)) if soc_probs is None
                              else np.asarray(soc_probs, float))
            if len(self.soc_probs) != len(self.soc_choices) or abs(self.soc_probs.sum() - 1.0) > 1e-6:
                raise ValueError("soc_probs must match soc_choices and sum to 1")
        # ---- attributes
        self.n = n_envs
        self.auto_reset = auto_reset
        self.terminal_at_T = terminal_at_T
        self.observation_mode = observation_mode
        self.observe_soc, noisy = OBS_MODES[observation_mode]
        self.sigma = float(vx_obs_noise) if noisy else 0.0
        self._default_sigma = self.sigma
        self.b0_range = (float(b0_range[0]), float(b0_range[1]))
        self.T, self.dt = int(T), float(dt)
        self.g, self.a_max, self.c_y, self.y_max = g, a_max, c_y, y_max
        self.H = H if H is not None else y_max / math.log(2.0)
        self.k_L, self.k_D = k_L, k_D
        self.h_min, self.h_max, self.cap_on, self.E = h_min, h_max, cap_on, E_hover_s
        self.v_min, self.v_goal, self.dense_w, self.C = v_min, v_goal, dense_w, crash_penalty
        self.y_ref, self.v_ref = y_ref, v_ref
        self.y0_range, self.vx0_range = y0_range, vx0_range
        self.rng = np.random.default_rng(seed)
        self.obs_dim = 5 if self.observe_soc else 4
        z = np.zeros(self.n)
        self.y, self.vx, self.vy, self.b, self.b0 = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
        self.x = z.copy()                       # horizontal position: display only, never observed
        self.last = {}                          # last step's delivered thrust / forces / reward, for rendering
        self.t = np.zeros(self.n, dtype=np.int64)
        self.reset()

    # ---------------------------------------------------------------- physics helpers
    def rho(self, y):
        return np.exp(-np.asarray(y, float) / self.H)

    def h_lim(self, b):
        return self.h_min + (self.h_max - self.h_min) * np.clip(b, 0.0, 1.0)

    def lift(self, y, vx):
        # a wing flying backwards does not lift (without max(.,0) backward flight got free lift from v_x^2)
        return np.minimum(self.k_L * self.rho(y) * np.maximum(np.asarray(vx, float), 0.0) ** 2, self.g)

    def drag(self, y, vx):
        vx = np.asarray(vx, float)
        return self.k_D * self.rho(y) * vx * np.abs(vx)

    def q_star(self):
        return self.g * self.k_L / (self.k_L ** 2 + self.k_D ** 2)

    def v_star(self, y):
        """Cheapest level-flight speed at altitude y."""
        return np.sqrt(self.q_star() / self.rho(y))

    def hold_cost(self, y, vx):
        """Battery drain per second (in hover units) to hold (y, vx) in level flight."""
        L, D = self.lift(y, vx), np.abs(self.drag(y, vx))
        return ((self.g - L) ** 2 + D ** 2) / self.g ** 2

    def v_steady(self, b, y=None):
        """Speed the cap at charge b can hold in level flight at altitude y (default: the ceiling)."""
        y = self.y_max if y is None else y
        return np.sqrt(self.h_lim(b) / (self.k_D * self.rho(y)))

    def t_exhaust(self, b0):
        """Closed form: seconds until b = 0 when running at the cap with the lift motor off (wing carries the plane).
        db/dt = -cap(b)^2 / (g^2 E)  ->  t = E g^2 / slope * (1/h_min - 1/cap(b0))."""
        slope = self.h_max - self.h_min
        if slope <= 0:
            return self.E * self.g ** 2 * np.asarray(b0, float) / self.h_min ** 2
        return self.E * self.g ** 2 / slope * (1.0 / self.h_min - 1.0 / self.h_lim(b0))

    # ---------------------------------------------------------------- gym plumbing
    def _obs(self):
        tau = 1.0 - self.t / self.T
        vx_obs = self.vx + self.rng.normal(0.0, self.sigma, size=self.n) if self.sigma > 0 else self.vx
        cols = [self.y / self.y_ref, vx_obs / self.v_ref, self.vy / self.v_ref, tau]
        if self.observe_soc:
            cols.append(self.b)
        return np.stack(cols, axis=1).astype(np.float32)

    def _reset_idx(self, idx):
        k = len(idx)
        if k == 0:
            return
        self.y[idx] = self.rng.uniform(*self.y0_range, size=k)
        self.vx[idx] = self.rng.uniform(*self.vx0_range, size=k)
        self.vy[idx] = 0.0
        if self.soc_choices is None:
            self.b0[idx] = self.rng.uniform(*self.b0_range, size=k)
        else:
            self.b0[idx] = self.rng.choice(self.soc_choices, size=k, p=self.soc_probs)
        self.b[idx] = self.b0[idx]
        self.t[idx] = 0
        self.x[idx] = 0.0

    def reset(self, b0=None, y0=None, vx0=None, *, vx_obs_noise=None):
        """Override starting charge/state (scalar or one value per plane) and speed noise.

        vx_obs_noise is a finite, non-negative scalar std in m/s for all planes.
        It applies until the next explicit reset; automatic vector resets retain it.
        Omitting it restores the constructor's noise. Clean/full modes stay noiseless.
        """
        sigma = self._default_sigma
        if vx_obs_noise is not None:
            noise = np.asarray(vx_obs_noise, dtype=float)
            if noise.ndim != 0 or not np.isfinite(noise) or noise < 0:
                raise ValueError("vx_obs_noise must be a finite, non-negative scalar")
            sigma = float(noise) if OBS_MODES[self.observation_mode][1] else 0.0
        self._reset_idx(np.arange(self.n))
        if b0 is not None:
            b0 = np.asarray(b0, float)
            if not (np.all(np.isfinite(b0)) and np.all(b0 > 0) and np.all(b0 <= 1)):
                raise ValueError("b0 must be a finite charge in (0, 1]")
            self.b0[:] = b0
            self.b[:] = b0
        if y0 is not None:
            y0 = np.asarray(y0, float)
            if not (np.all(np.isfinite(y0)) and np.all(y0 > 0) and np.all(y0 <= self.y_max)):
                raise ValueError("y0 must lie above the ground and at or below y_max")
            self.y[:] = y0
        if vx0 is not None:
            vx0 = np.asarray(vx0, float)
            if not np.all(np.isfinite(vx0)):
                raise ValueError("vx0 must be finite")
            self.vx[:] = vx0
        self.sigma = sigma
        self.last = {}
        return self._obs()

    def step(self, action):
        a = np.asarray(action, float).reshape(self.n, 2)
        if not np.all(np.isfinite(a)):
            raise ValueError("actions must be finite")
        a = np.clip(a, -1.0, 1.0)
        L, D = self.lift(self.y, self.vx), self.drag(self.y, self.vx)
        # motors: lift motor centred on level flight; cruise motor capped by the charge
        ay = np.clip(self.g - L + 0.5 * self.a_max * a[:, 1], 0.0, self.a_max)
        cap = self.h_lim(self.b if self.cap_on == "b" else self.b0)
        ax_cmd = self.a_max * a[:, 0]
        ax = np.clip(ax_cmd, -cap, cap)
        saturated = np.abs(ax_cmd) > cap
        alive_b = self.b > 0.0
        ax, ay = np.where(alive_b, ax, 0.0), np.where(alive_b, ay, 0.0)
        # battery: pay for delivered thrust (commanding past the cap is free: you just don't get it)
        self.b = np.clip(self.b - (ax ** 2 + ay ** 2) / self.g ** 2 * self.dt / self.E, 0.0, 1.0)
        # motion (semi-implicit Euler)
        self.vx += self.dt * (ax - D)
        self.vy += self.dt * (ay + L - self.g - self.c_y * self.vy)
        self.y += self.dt * self.vy
        self.x += self.dt * self.vx
        hit = self.y > self.y_max
        self.y = np.where(hit, self.y_max, self.y)
        self.vy = np.where(hit, np.minimum(self.vy, 0.0), self.vy)
        self.t += 1
        # reward
        r = np.clip((self.vx - self.v_min) / (self.v_goal - self.v_min), 0.0, 1.0) ** 2
        r = r + self.dense_w * np.clip(self.vx / self.v_goal, 0.0, 1.0) ** 2
        failed = (self.y <= 0.0) | (self.b <= 0.0)
        timeout = self.t >= self.T
        mission_complete = timeout & ~failed
        r = np.where(failed, r - self.C, r)
        if self.terminal_at_T:
            terminated, truncated = failed | timeout, np.zeros(self.n, dtype=bool)
        else:
            terminated, truncated = failed, timeout & ~failed
        self.last = {"ax": ax.copy(), "ay": ay.copy(), "cap": np.array(cap, float).reshape(self.n).copy(),
                     "L": L.copy(), "D": D.copy(), "r": r.copy(), "saturated": saturated.copy(),
                     "dead_battery": (self.b <= 0.0).copy(), "ground": (self.y <= 0.0).copy(),
                     "mission_complete": mission_complete.copy()}
        obs = self._obs()                                   # observation of the FINAL state (noise drawn once)
        # privileged diagnostics: for logging/evaluation only, never for a hidden-mode policy
        info = {"b0": self.b0.copy(), "b": self.b.copy(), "cap": self.last["cap"], "y": self.y.copy(),
                "vx": self.vx.copy(), "crashed": failed.copy(), "saturated": saturated,
                "mission_complete": mission_complete, "final_obs": obs.copy()}
        done = terminated | truncated
        if self.auto_reset and done.any():
            self._reset_idx(np.where(done)[0])
            obs[done] = self._obs()[done]                   # those planes now report their new episode's first obs
        return obs, r.astype(np.float32), terminated, truncated, info

    # ---------------------------------------------------------------- scripted rollouts and design check
    def simulate(self, policy, b0s, y0=100.0, vx0=15.0, seed=0):
        """One episode per entry of b0s (same start state), driven by a scripted policy(env) -> (n, 2) actions.
        Returns arrays: ret, t_dead (nan = survived), s_goal / s_band (seconds at >= 90% main reward / in the band), b_end."""
        b0s = np.atleast_1d(np.asarray(b0s, float))
        env = BatteryPlaneVec(n_envs=len(b0s), auto_reset=False, seed=seed, **self._kw)
        env.reset(b0=b0s, y0=y0, vx0=vx0)
        n = env.n
        alive = np.ones(n, dtype=bool)
        ret, t_dead, s_goal, s_band, b_end = np.zeros(n), np.full(n, np.nan), np.zeros(n), np.zeros(n), np.zeros(n)
        v90 = env.v_min + (env.v_goal - env.v_min) * math.sqrt(0.9)          # speed at which the main reward is 0.9
        for t in range(env.T):
            _, r, term, trunc, info = env.step(policy(env))
            ret[alive] += r[alive]
            s_goal[alive] += (env.vx[alive] >= v90) * env.dt
            s_band[alive] += (env.vx[alive] >= env.v_min) * env.dt
            finished = alive & (term | trunc)
            died = finished & info["crashed"]
            t_dead[died] = (t + 1) * env.dt
            b_end[finished] = env.b[finished]
            alive &= ~finished
            if not alive.any():
                break
        return {"b0": b0s, "ret": ret, "t_dead": t_dead, "s_goal": s_goal, "s_band": s_band, "b_end": b_end}

    def describe(self, b0_grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), b_stops=(0.05, 0.1, 0.15, 0.2, 0.3)):
        ym, dur = self.y_max, self.T * self.dt
        rho_m = float(self.rho(ym))
        v_star_0, v_star_m = float(self.v_star(0.0)), float(self.v_star(ym))
        D_min_m, D_goal_m = float(abs(self.drag(ym, self.v_min))), float(abs(self.drag(ym, self.v_goal)))
        D_min_0 = float(abs(self.drag(0.0, self.v_min)))
        slope = self.h_max - self.h_min
        b_goal = (D_goal_m - self.h_min) / slope if slope > 0 else (0.0 if D_goal_m <= self.h_min else math.inf)
        b_close = (D_min_m - self.h_min) / slope if slope > 0 else (0.0 if D_min_m <= self.h_min else math.inf)
        v_ground_full = math.sqrt(self.h_max / self.k_D)
        lo, hi = self.b0_range if self.soc_choices is None else (float(self.soc_choices.min()), float(self.soc_choices.max()))
        b_mid = 0.5 * (lo + hi)
        dv_db = slope / (2.0 * math.sqrt(float(self.h_lim(b_mid)) * self.k_D * rho_m)) if slope > 0 else 0.0
        k_id = (self.sigma / (0.1 * dv_db)) ** 2 if (self.sigma > 0 and dv_db > 0) else 0.0
        # --- simulations
        fine = np.round(np.arange(0.05, 1.0001, 0.01), 2)
        sim_fine = self.simulate(policy_command_max, fine)
        died = ~np.isnan(sim_fine["t_dead"])
        b_death = float(fine[died].max()) + 0.01 if died.any() else None       # smallest surviving b0 (always-sprint)
        res_max = self.simulate(policy_command_max, b0_grid)
        res_cru = self.simulate(policy_cruise, b0_grid)
        res_bud = {bs: self.simulate(policy_budgeted(bs), b0_grid) for bs in b_stops}
        bud_ret = np.stack([res_bud[bs]["ret"] for bs in b_stops])             # (n_stops, n_b0)
        best = bud_ret.argmax(axis=0)
        ok = lambda c: "OK" if c else "VIOLATED"
        lines = [
            f"episode {self.T} steps x {self.dt}s = {dur:.0f}s | ceiling {ym:.0f} m, density at ceiling {rho_m:.2f} | "
            f"mode={self.observation_mode}, speed noise std={self.sigma:g} m/s, cap follows {'current b' if self.cap_on == 'b' else 'b0 (frozen)'}",
            f"context: b0 ~ " + (f"U[{lo}, {hi}]" if self.soc_choices is None else f"{self.soc_choices.tolist()} p={self.soc_probs.tolist()}")
            + f" | budget {lo * self.E:.0f}..{hi * self.E:.0f} hover-s (hovering the whole mission costs {dur:.0f})",
            f"cheapest cruise v*: {v_star_0:.1f} m/s at ground, {v_star_m:.1f} at ceiling, costs {float(self.hold_cost(0, v_star_0)):.2f} hover-units/s "
            f"-> surviving the mission costs ~{float(self.hold_cost(0, v_star_0)) * dur / self.E:.2f} of b",
            f"reward band {self.v_min:g}..{self.v_goal:g} m/s;  v_min >= v*(ceiling): {ok(self.v_min >= v_star_m)}",
            f"cap = {self.h_min:g} + {slope:g} b  (drag at ceiling: D(v_min)={D_min_m:.1f}, D(v_goal)={D_goal_m:.1f})",
            f"band unreachable at ground even fully charged: max ground speed {v_ground_full:.1f} < v_min={self.v_min:g}: {ok(v_ground_full < self.v_min)}",
            f"at the ceiling the cap holds v_goal for b >= {b_goal:.2f} (inside b0 range: {ok(lo < b_goal < hi)}); "
            f"the band closes below b = {b_close:.2f}" + ("" if b_close > 0 else " (never: even an empty battery touches it)"),
            f"steady speed at the ceiling vs charge: " + ", ".join(f"b={b:.1f}->{float(self.v_steady(b)):.1f}" for b in (0.0, 0.25, 0.5, 0.75, 1.0)),
            f"closed-form exhaustion at the cap from b0={lo}: {float(self.t_exhaust(lo)):.1f}s, from b0={hi}: {float(self.t_exhaust(hi)):.1f}s (mission {dur:.0f}s)",
            f"always-sprint (climb, then command max) survives the mission only for b0 >= "
            + (f"{b_death:.2f}" if b_death is not None else "any") + f" -> rationing needed inside the b0 range: {ok(b_death is not None and lo < b_death <= hi + 1e-9)}",
            f"identification: dv_steady/db = {dv_db:.1f} m/s per unit b at b={b_mid:.2f}; one speed reading pins b to +-{(self.sigma / dv_db if dv_db > 0 else 0):.2f}; "
            + (f"~{k_id:.0f} saturated steps of memory for +-0.1" if k_id > 0 else "exact speed: b is exact from 2 consecutive readings while saturated"),
            "",
            "scripted rollouts from y0=100 m, vx0=15 m/s  (return | dead at | seconds at >=90% reward / in band | b at end):",
            f"{'b0':>5} | {'always-sprint':^38} | {'best budgeted (hold v_goal, then cruise)':^46} | {'cruise-only':^14}",
        ]
        for i, b0 in enumerate(b0_grid):
            m, cr, bb = res_max, res_cru, res_bud[b_stops[best[i]]]
            dead = lambda r: f"{r['t_dead'][i]:5.1f}s" if not np.isnan(r["t_dead"][i]) else "  -   "
            lines.append(f"{b0:5.2f} | {m['ret'][i]:7.1f} {dead(m)} {m['s_goal'][i]:5.1f}/{m['s_band'][i]:5.1f} b={m['b_end'][i]:.2f}"
                         f" | {bb['ret'][i]:7.1f} {dead(bb)} {bb['s_goal'][i]:5.1f}/{bb['s_band'][i]:5.1f} b={bb['b_end'][i]:.2f} (b_stop={b_stops[best[i]]:.2f})"
                         f" | {cr['ret'][i]:7.1f} {dead(cr)}")
        full = res_bud[b_stops[best[-1]]]["s_goal"][-1]
        lines.append(f"a full battery earns >= 90% reward for {full:.1f}s of {dur:.0f}s under the best budgeted policy: "
                     f"{ok(full >= 0.2 * dur)} (rule: >= 20% of the mission)")
        return "\n".join(lines)


    # ---------------------------------------------------------------- value of context (the benchmark check)
    def value_of_context(self, b0_grid=None, y0=100.0, vx0=15.0, seed=0, verbose=True):
        """How much is KNOWING the charge worth? The one number a context benchmark must have.

        Three families of hand-written policies fly from the same start state (y0, vx0) for every b0 in b0_grid:
            oracle  reads the TRUE charge: hold speed v while b > b_stop, then fly at the cheapest speed v*(y)
            blind   never reads b: same rule, but switches on the clock (t > t_stop) or when its OWN SPEED SAGS
                    below v_sag -- the speed sag is the only battery signal a blind pilot has. One rule for all b0
                    (a blind agent cannot pick a rule per episode), so we keep the single best rule over the grid.
            sprint  climb, then full throttle forever: what a careless blind agent does
            cruise  never climb, fly the cheapest speed: the survival floor, return available with no knowledge at all
        All track speed exactly (deadbeat throttle) and cut the lift motor at 150 m to coast up to the ceiling.

        value of context = (mean oracle - mean blind) / mean oracle.  Rule of thumb: a benchmark needs it well above
        ~20%. If it is small, no learning algorithm can show a hidden-vs-full difference, whatever the encoder.
        Returns a dict (b0, oracle, blind, sprint, cruise, sprint_crash, value_of_context, blind_rule).
        """
        if b0_grid is None:
            if self.soc_choices is None:
                lo, hi = self.b0_range
                b0_grid = np.round(np.arange(lo, hi + 1e-9, 0.05), 2)
            else:
                b0_grid = np.sort(self.soc_choices)
        b0_grid = np.atleast_1d(np.asarray(b0_grid, float))

        def throttle(env, v_target):                 # thrust that lands exactly on v_target next step (clipped)
            return np.clip((env.drag(env.y, env.vx) + (v_target - env.vx) / env.dt) / env.a_max, 0.0, 1.0)

        def coast_climb(env, cut=150.0):             # full lift until `cut`, then coast the rest of the way up
            return (env.y < cut) * 1.0

        def oracle(v, b_stop):
            def pol(env):
                v_t = np.where(env.b > b_stop, v, env.v_star(env.y))
                return np.stack([throttle(env, v_t), coast_climb(env)], 1)
            return pol

        def blind_clock(v, t_stop):
            def pol(env):
                v_t = np.where(env.t * env.dt < t_stop, v, env.v_star(env.y))
                return np.stack([throttle(env, v_t), coast_climb(env)], 1)
            return pol

        def blind_sag(v, v_sag):
            def pol(env):
                latch = getattr(env, "_voc_latch", None)
                if latch is None:
                    latch = np.zeros(env.n, bool)
                latch = latch | ((env.vx < v_sag) & (env.y > env.y_max - 10.0))    # once the speed sags, give up sprinting
                env._voc_latch = latch
                v_t = np.where(latch, env.v_star(env.y), v)
                return np.stack([throttle(env, v_t), coast_climb(env)], 1)
            return pol

        def sprint(env):
            return np.stack([np.ones(env.n), coast_climb(env)], 1)

        def cruise(env):
            return np.stack([throttle(env, env.v_star(env.y)), np.zeros(env.n)], 1)

        def ret(policy):
            r = self.simulate(policy, b0_grid, y0=y0, vx0=vx0, seed=seed)
            return r["ret"], ~np.isnan(r["t_dead"])

        speeds = (56.0, 58.0, 60.0, 62.0, 64.0, 70.0)
        orc = np.stack([ret(oracle(v, bs))[0] for v in speeds for bs in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6)])
        blind_pool, names = [], []
        for v in speeds:
            for ts in (5, 10, 15, 20, 25, 35):
                blind_pool.append(ret(blind_clock(v, ts))[0]); names.append(f"hold {v:g} m/s until t = {ts} s, then cruise")
            for vs in (50, 52, 55, 57, 59):
                blind_pool.append(ret(blind_sag(v, vs))[0]); names.append(f"hold {v:g} m/s until speed sags below {vs}, then cruise")
        blind_pool = np.stack(blind_pool)
        i_best = int(blind_pool.mean(1).argmax())
        oracle_r, blind_r = orc.max(0), blind_pool[i_best]
        sprint_r, sprint_crash = ret(sprint)
        cruise_r, _ = ret(cruise)
        voc = (oracle_r.mean() - blind_r.mean()) / max(abs(oracle_r.mean()), 1e-9)
        out = {"b0": b0_grid, "oracle": oracle_r, "blind": blind_r, "sprint": sprint_r, "cruise": cruise_r,
               "sprint_crash": sprint_crash, "value_of_context": float(voc), "blind_rule": names[i_best]}
        if verbose:
            row = lambda name, arr: f"  {name:8s}" + " ".join(f"{v:7.0f}" for v in arr)
            print(f"value of context  (hand-written policies from y0={y0:g} m, vx0={vx0:g} m/s; "
                  f"E={self.E:g}, dense_w={self.dense_w:g}, C={self.C:g}, b0 grid from the configured range)")
            print("  b0      " + " ".join(f"{b:7.2f}" for b in b0_grid))
            print(row("oracle", oracle_r) + "   knows b")
            print(row("blind", blind_r) + f"   best single blind rule: {names[i_best]}")
            print(row("sprint", sprint_r) + "   crashes: " + "".join("X" if c else "." for c in sprint_crash))
            print(row("cruise", cruise_r) + "   never climbs (floor)")
            surv = b0_grid[~sprint_crash]
            print(f"  mean: oracle {oracle_r.mean():.0f} | blind {blind_r.mean():.0f} | sprint {sprint_r.mean():.0f} | cruise {cruise_r.mean():.0f}"
                  f"   ->  VALUE OF CONTEXT = {100 * voc:.0f}% of the oracle return"
                  + ("  (OK: > 20%)" if voc > 0.2 else "  (LOW: < 20%, the benchmark can not separate hidden from full)"))
            print(f"  always-sprint survives only for b0 >= {surv.min():.2f}" if len(surv) else "  always-sprint never survives",
                  "-> the top of b0_range should sit about there (episodes above it need no context)")
        return out


# ====================================================================================== gymnasium wrapper
if gym is not None:

    class BatteryPlaneEnv(gym.Env):
        """Single-plane Gymnasium env.

        observation_mode: "hidden" (default), "clean", "observed", "full"  -- see the module docstring.
        reset(seed=0, options={"b0": 0.9, "vx_obs_noise": 2.0}) sets the exact starting
        charge and speed noise for that episode; "y0", "vx0" are also accepted.
        Omitted options use constructor defaults. Clean/full modes always have zero noise.
        add_eval_args() and iter_eval_resets() connect a CLI evaluation grid to reset().
        info carries privileged diagnostics (b0, b, cap, saturated, mission_complete): log them, never feed them
        to a hidden-mode agent.
        """
        metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
        W, H, PW = 960, 540, 300                  # window and right-panel width

        def __init__(self, observation_mode: str = "hidden", seed=None, render_mode=None, **kw):
            super().__init__()
            # auto_reset=False: step() returns the observation of the final state; the caller must call reset().
            self.core = BatteryPlaneVec(n_envs=1, observation_mode=observation_mode, seed=seed, auto_reset=False, **kw)
            self.observation_mode = observation_mode
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.core.obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            if render_mode is not None and render_mode not in self.metadata["render_modes"]:
                raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None, got {render_mode!r}")
            self.render_mode = render_mode
            self.metadata = dict(self.metadata, render_fps=max(1, int(round(1.0 / self.core.dt))))
            self._needs_reset = True
            self.screen = None; self.clock = None; self.font = None; self.font_big = None
            self._ep_return = 0.0; self._last_obs = None; self._status = ""
            from collections import deque
            self._trail = deque(maxlen=max(1, int(8.0 / self.core.dt)))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.core.rng = np.random.default_rng(seed)
            options = options or {}
            obs = self.core.reset(b0=options.get("b0"), y0=options.get("y0"), vx0=options.get("vx0"),
                                  vx_obs_noise=options.get("vx_obs_noise"))
            self._needs_reset = False
            self._ep_return = 0.0; self._last_obs = obs[0]; self._status = ""; self._trail.clear()
            if self.render_mode == "human":
                self.render()
            return obs[0], {"b0": float(self.core.b0[0])}

        def step(self, action):
            if self._needs_reset:
                raise gym.error.ResetNeeded("Episode is over: call reset() before step().")
            obs, r, term, trunc, info = self.core.step(np.asarray(action, dtype=float)[None, :])
            terminated, truncated = bool(term[0]), bool(trunc[0])
            self._needs_reset = terminated or truncated
            out = {"b0": float(info["b0"][0]), "b": float(info["b"][0]), "cap": float(info["cap"][0]),
                   "y": float(info["y"][0]), "vx": float(info["vx"][0]), "crashed": bool(info["crashed"][0]),
                   "saturated": bool(info["saturated"][0]), "mission_complete": bool(info["mission_complete"][0])}
            self._ep_return += float(r[0]); self._last_obs = obs[0]
            if out["crashed"]:
                self._status = "BATTERY EMPTY" if self.core.last["dead_battery"][0] else "CRASHED"
            elif out["mission_complete"]:
                self._status = "MISSION COMPLETE"
            elif truncated:
                self._status = "TRUNCATED"
            if self.render_mode == "human":
                self.render()
            return obs[0], float(r[0]), terminated, truncated, out

        def describe(self):
            return self.core.describe()

        def simulate(self, *args, **kwargs):
            return self.core.simulate(*args, **kwargs)

        def value_of_context(self, *args, **kwargs):
            return self.core.value_of_context(*args, **kwargs)

        # ------------------------------------------------------------ rendering (pygame, like gymnasium's classic control)
        def render(self):
            if self.render_mode is None:
                gym.logger.warn("No render_mode set; pass render_mode='human' or 'rgb_array' to the constructor.")
                return None
            try:
                import pygame
            except ImportError as e:
                raise gym.error.DependencyNotInstalled("pygame is not installed: pip install pygame") from e
            if self.screen is None:
                pygame.font.init()
                if self.render_mode == "human":
                    pygame.display.init()
                    self.screen = pygame.display.set_mode((self.W, self.H))
                    pygame.display.set_caption("BatteryPlane v3")
                else:
                    self.screen = pygame.Surface((self.W, self.H))
                self.font = pygame.font.Font(None, 22); self.font_big = pygame.font.Font(None, 54)
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.Surface((self.W, self.H))
            self._draw(pygame, surf)
            self.screen.blit(surf, (0, 0))
            if self.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return None
                self.clock.tick(self.metadata["render_fps"])
                pygame.display.flip()
                return None
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2)).copy()

        def _draw(self, pg, surf):
            c = self.core
            y, vx, vy, b, b0, x = (float(c.y[0]), float(c.vx[0]), float(c.vy[0]), float(c.b[0]), float(c.b0[0]), float(c.x[0]))
            y_draw = min(max(y, 0.0), c.y_max)                     # the physics clamps y at the ceiling; the ground can be crossed
            last = c.last if c.last else {"ax": [0.0], "ay": [0.0], "cap": [float(c.h_lim(b))], "L": [0.0], "D": [0.0],
                                          "r": [0.0], "saturated": [False]}
            ax, ay, cap, L, D, r = (float(last["ax"][0]), float(last["ay"][0]), float(last["cap"][0]), float(last["L"][0]),
                                    float(last["D"][0]), float(last["r"][0]))
            sat = bool(last["saturated"][0])
            # ---- layout: the world gets headroom above the ceiling so nothing drawn on the plane pokes into the void
            world_w = self.W - self.PW
            top, ground = 120, self.H - 60
            px_per_m = (ground - top) / c.y_max
            def sy(alt): return ground - alt * px_per_m
            def sky(rho):                                           # paler with altitude (thinner air)
                t = 1.0 - rho
                return (int(150 + 100 * t), int(200 + 50 * t), int(240 + 15 * t))
            surf.set_clip(pg.Rect(0, 0, world_w, self.H))          # the world never spills into the panel
            n_bands = 24
            for i in range(n_bands):
                a_lo, a_hi = c.y_max * i / n_bands, c.y_max * (i + 1) / n_bands
                pg.draw.rect(surf, sky(float(c.rho(a_lo))), (0, int(sy(a_hi)), world_w, int(math.ceil((a_hi - a_lo) * px_per_m)) + 1))
            # beyond the ceiling: greyed, hatched sky. It is a model limit, not a wall, and the plane's force arrows may
            # overlap it while the plane is pressed against the ceiling.
            base = sky(float(c.rho(c.y_max)))
            beyond = tuple(int(0.55 * v + 0.45 * w) for v, w in zip(base, (205, 208, 218)))
            pg.draw.rect(surf, beyond, (0, 0, world_w, top))
            hatch = tuple(max(0, v - 16) for v in beyond)
            for k in range(-top, world_w, 22):
                pg.draw.line(surf, hatch, (k, top), (k + top, 0), 1)
            # ---- ground with scrolling marks every 25 m so forward speed is visible
            pg.draw.rect(surf, (96, 140, 84), (0, ground, world_w, self.H - ground))
            plane_sx = 260
            for k in range(-20, 40):
                gx = plane_sx + (k * 25.0 - (x % 25.0)) * 3.0                          # 3 px per metre horizontally
                if 0 <= gx < world_w:
                    pg.draw.line(surf, (60, 100, 50), (int(gx), ground), (int(gx), ground + 12), 2)
            # ---- altitude ticks
            for alt in range(0, int(c.y_max) + 1, 50):
                pg.draw.line(surf, (110, 110, 130), (0, int(sy(alt))), (8, int(sy(alt))), 1)
                surf.blit(self.font.render(f"{alt}", True, (60, 60, 80)), (11, int(sy(alt)) - 8))
            # ---- flight trail (last 8 s), scrolling with the world; drawn before the ceiling line so the line stays crisp
            self._trail.append((x, y_draw))
            n = len(self._trail)
            for i in range(1, n):
                (x0, y0), (x1, y1) = self._trail[i - 1], self._trail[i]
                sx0, sx1 = plane_sx + (x0 - x) * 3.0, plane_sx + (x1 - x) * 3.0
                if sx1 < 0:
                    continue
                shade = int(90 + 140 * (1 - i / n))
                pg.draw.line(surf, (shade, shade, shade + 20), (sx0, sy(y0)), (sx1, sy(y1)), 2)
            # ---- ceiling line and label (the label sits to the right of the plane; nothing drawn on the plane reaches it)
            pg.draw.line(surf, (90, 95, 120), (0, top), (world_w, top), 2)
            lab = self.font.render(f"ceiling {c.y_max:.0f} m  (air density {float(c.rho(c.y_max)):.2f} of ground)", True, (70, 75, 100))
            surf.blit(lab, (world_w - lab.get_width() - 10, top - lab.get_height() - 4))
            # ---- plane (triangle pointing right, pitched by climb angle)
            cx, cy = plane_sx, sy(y_draw)
            pitch = math.atan2(vy, max(abs(vx), 5.0))
            def rot(dx, dy):
                return (cx + dx * math.cos(pitch) + dy * math.sin(pitch), cy - (-dx * math.sin(pitch) + dy * math.cos(pitch)))
            body = [rot(36, 0), rot(-26, -13), rot(-14, 0), rot(-26, 13)]
            # lift motor flame (down) and cruise motor flame (backwards), length = delivered thrust
            if ay > 0:
                pg.draw.line(surf, (255, 140, 40), rot(-6, 0), rot(-6, -8 - 40 * ay / c.a_max), 6)
            if ax > 0:
                pg.draw.line(surf, (255, 90, 30), rot(-24, 0), rot(-24 - 50 * ax / c.a_max, 0), 6)
            elif ax < 0:
                pg.draw.line(surf, (255, 90, 30), rot(34, 0), rot(34 + 50 * (-ax) / c.a_max, 0), 6)
            # wing lift (blue arrow up) scaled by the fraction of the weight it carries
            if L > 0.5:
                pg.draw.line(surf, (40, 90, 220), rot(0, 6), rot(0, 6 + 36 * L / c.g), 4)
            pg.draw.polygon(surf, (30, 60, 100) if not sat else (170, 40, 40), body)
            pg.draw.polygon(surf, (255, 255, 255), body, 2)
            # ---- speed bar (bottom of world area): v*, v_min, v_goal markers, true speed (solid) and what the agent sees (hollow)
            bx0, bx1, by = 60, world_w - 20, self.H - 28
            vmax = max(c.v_goal * 1.3, 80.0)
            def bxv(v): return bx0 + (bx1 - bx0) * min(max(v, 0.0), vmax) / vmax
            pg.draw.rect(surf, (70, 70, 70), (bx0, by - 6, bx1 - bx0, 12))
            pg.draw.rect(surf, (230, 200, 60), (bxv(c.v_min), by - 6, bxv(c.v_goal) - bxv(c.v_min), 12))   # reward band
            for v, lab_, col in ((float(c.v_star(y_draw)), "v*", (40, 90, 220)), (c.v_min, "v_min", (120, 90, 0)), (c.v_goal, "v_goal", (120, 90, 0))):
                pg.draw.line(surf, col, (bxv(v), by - 12), (bxv(v), by + 8), 2)
                surf.blit(self.font.render(lab_, True, col), (bxv(v) - 10, by + 9))
            pg.draw.polygon(surf, (255, 255, 255), [(bxv(vx), by - 14), (bxv(vx) - 7, by - 24), (bxv(vx) + 7, by - 24)])
            if self._last_obs is not None and c.sigma > 0:
                vs = float(self._last_obs[1]) * c.v_ref
                pg.draw.polygon(surf, (255, 255, 255), [(bxv(vs), by - 14), (bxv(vs) - 7, by - 24), (bxv(vs) + 7, by - 24)], 2)
            surf.blit(self.font.render("speed", True, (230, 230, 230)), (8, by - 8))
            if self._status:
                col = (60, 220, 90) if self._status == "MISSION COMPLETE" else (255, 70, 60)
                txt = self.font_big.render(self._status, True, col)
                surf.blit(txt, ((world_w - txt.get_width()) // 2, self.H // 2 - 20))
            surf.set_clip(None)
            # ---- right panel (true state: a diagnostic view, not what a hidden-mode agent sees)
            pg.draw.rect(surf, (28, 30, 38), (world_w, 0, self.PW, self.H))
            px = world_w + 14
            seen = self._last_obs[1] * c.v_ref if self._last_obs is not None else vx
            b_note = "visible" if c.observe_soc else "hidden"
            lines = [
                (f"BatteryPlane v3  [{self.observation_mode}]", (255, 255, 255)),
                (f"time left   {max(0, c.T - int(c.t[0])) * c.dt:5.1f} s", (220, 220, 220)),
                (f"altitude    {y:6.1f} m", (220, 220, 220)),
                (f"speed       {vx:6.1f} m/s  (agent sees {seen:5.1f})", (220, 220, 220)),
                (f"climb rate  {vy:6.1f} m/s", (220, 220, 220)),
                (f"battery     {100 * b:5.1f} %  (start {100 * b0:3.0f} %, {b_note})", (220, 220, 220)),
                (f"cruise thrust {ax:5.1f} / cap {cap:4.1f}" + ("   AT CAP" if sat else ""), (255, 120, 120) if sat else (220, 220, 220)),
                (f"lift motor  {ay:5.1f}    wing lift {L:4.1f} of {c.g:.0f}", (220, 220, 220)),
                (f"drag        {D:5.1f}    v* here {float(c.v_star(y_draw)):4.1f} m/s", (220, 220, 220)),
                (f"reward      {r:+6.3f}   return {self._ep_return:7.1f}", (230, 200, 60)),
            ]
            for i, (txt, col) in enumerate(lines):
                surf.blit(self.font.render(txt, True, col), (px, 14 + 24 * i))
            bw, bh, bxp, byp = self.PW - 28, 18, px, 14 + 24 * len(lines) + 6
            pg.draw.rect(surf, (70, 70, 70), (bxp, byp, bw, bh))
            pg.draw.rect(surf, (60, 200, 90) if b > 0.3 else (230, 80, 60), (bxp, byp, int(bw * b), bh))
            surf.blit(self.font.render("battery (true)", True, (200, 200, 200)), (bxp, byp + bh + 2))
            leg = [("orange: motor thrust", (255, 120, 40)), ("blue: wing lift", (80, 130, 255)),
                   ("yellow band: reward speeds", (230, 200, 60)), ("red plane: cruise motor at its cap", (255, 120, 120)),
                   ("hollow marker: speed the agent sees", (230, 230, 230)), ("hatched: above the model ceiling", (170, 175, 190))]
            for i, (txt, col) in enumerate(leg):
                surf.blit(self.font.render(txt, True, col), (px, byp + bh + 30 + 22 * i))

        def close(self):
            if self.screen is not None:
                import pygame
                if self.render_mode == "human":
                    pygame.display.quit()
            self.screen = None; self.clock = None; self.font = None; self.font_big = None

    def register(prefix: str = "BatteryPlane", version: str = "v3"):
        """Register one id per observation mode (idempotent). No max_episode_steps: the env has its own clock (tau)
        and ends at T by itself. Returns the ids."""
        ids = {f"{prefix}-{version}": "hidden", f"{prefix}Clean-{version}": "clean",
               f"{prefix}Observed-{version}": "observed", f"{prefix}Full-{version}": "full"}
        for env_id, mode in ids.items():
            if env_id not in gym.registry:
                gym.register(id=env_id, entry_point=BatteryPlaneEnv, kwargs={"observation_mode": mode})
        return list(ids)


if __name__ == "__main__":                 # python battery_plane.py  -> design rules + value of context for the defaults
    _env = BatteryPlaneVec()
    print(_env.describe())
    print()
    _env.value_of_context()
