"""Why the agent should climb: two snapshots of BatteryPlane and the top speed by altitude, for the project page.

The same plane with the same battery flies at full cruise throttle for 8 s, once low (20 m, thick air) and once climbing
to the ceiling (200 m, half the air density). Only up high does it reach the reward band: drag falls with the air
density, so the same thrust holds a higher speed. Rendered with the environment's own renderer (the world area only:
altitude ticks, the ceiling, the trail and the speed bar with the yellow reward band); test seed 2000, no training.

The chart is the closed form behind it: the speed the cruise motor can hold in level flight,
v = sqrt(cap(b) / (k_D rho(y))), cap(b) = 11.5 + 6.5 b, rho(y) = exp(-y / H), against altitude for three batteries,
with the reward band 52-62 m/s.

    PYTHONPATH=<dir with font_roboto> python figures/env_snapshots.py
Outputs: figures/out_person/env_snapshot_{low,high}.png, figures/out_person/fig_top_speed_by_altitude.{pdf,png,svg},
and copies of the PNGs in docs/assets/figures/.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import battery_plane as bp  # noqa: E402

OUT, PAGE = HERE / "out_person", ROOT / "docs" / "assets" / "figures"
SEED, B0, SECONDS = 2000, 0.6, 8.0
WORLD_WIDTH = bp.BatteryPlaneEnv.W - bp.BatteryPlaneEnv.PW       # the renderer's world area, without the side panel


def snapshot(y0: float, climb_to: float | None) -> tuple[np.ndarray, dict]:
    """The last frame of an 8 s full-throttle flight from y0 (climbing to climb_to first, if given), and its state."""
    env = bp.BatteryPlaneEnv(observation_mode="hidden", render_mode="rgb_array")
    env.reset(seed=SEED, options={"b0": B0, "y0": y0, "vx0": 40.0})
    frame = env.render()
    core = env.core
    for _ in range(int(SECONDS / core.dt)):
        lift = 0.6 if climb_to is not None and core.y[0] < climb_to - 4.0 else 0.0
        env.step(np.array([1.0, lift], np.float32))
        frame = env.render()                                        # every step: the renderer draws the trail
    state = {"altitude": float(core.y[0]), "speed": float(core.vx[0]), "battery": float(core.b[0])}
    env.close()
    return frame[:, :WORLD_WIDTH], state


def save_png(array: np.ndarray, name: str) -> Path:
    from PIL import Image

    OUT.mkdir(exist_ok=True)
    path = OUT / name
    Image.fromarray(array).save(path, optimize=True)
    return path


def top_speed_chart() -> Path:
    import matplotlib.pyplot as plt
    from make_figures_person import AXIS, GRID, INK2, MUTED, save   # the page figures' style (Roboto, colours)

    core = bp.BatteryPlaneVec(auto_reset=False)
    altitude = np.linspace(0.0, core.y_max, 401)
    batteries = [(0.2, "#86b6ef"), (0.5, "#2a78d6"), (1.0, "#104281")]   # one hue, light to dark (validated ordinal)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.axvspan(core.v_min, core.v_goal, color="#e6c83c", alpha=0.28, linewidth=0, zorder=0)
    ax.text(0.5 * (core.v_min + core.v_goal), 6, "reward\nband", ha="center", va="bottom", fontsize=7.5, color=INK2)
    for b, colour in batteries:
        speed = core.v_steady(b, altitude)
        ax.plot(speed, altitude, color=colour, linewidth=2.0, zorder=3)
        enters = altitude[np.argmax(speed >= core.v_min)]          # the lowest altitude where the band is reachable
        ax.plot([core.v_min], [enters], "o", color=colour, markersize=5, markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        ax.text(speed[-1] + 0.6, core.y_max, f"{b:.0%}", ha="left", va="center", fontsize=7.5, color=INK2)
        ax.text(core.v_min - 0.8, enters, f"{enters:.0f} m", ha="right", va="center", fontsize=6.8, color=MUTED)
    ax.text(70.6, core.y_max + 13, "battery", ha="left", va="bottom", fontsize=7, color=MUTED)
    ax.set_xlim(36, 74)
    ax.set_ylim(0, core.y_max + 8)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Top speed at full cruise thrust (m/s)")
    ax.set_ylabel("Altitude (m)")
    ax.spines["left"].set_visible(True)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(core.y_max, color=AXIS, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    save(fig, "fig_top_speed_by_altitude")
    return OUT / "fig_top_speed_by_altitude.png"


def main() -> None:
    low, low_state = snapshot(20.0, None)
    high, high_state = snapshot(120.0, 200.0)
    for name, state in (("low", low_state), ("high", high_state)):
        print(f"{name}: altitude {state['altitude']:.0f} m, speed {state['speed']:.1f} m/s, battery {state['battery']:.2f}")
    paths = [save_png(low, "env_snapshot_low.png"), save_png(high, "env_snapshot_high.png"), top_speed_chart()]
    PAGE.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy(path, PAGE / path.name)
        print(PAGE / path.name)


if __name__ == "__main__":
    main()
