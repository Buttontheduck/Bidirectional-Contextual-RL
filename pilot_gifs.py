"""GIFs of the pilots of pilots.py: every style flown on the final-test seeds 2000-2002, rendered with BatteryPlaneEnv's
renderer, into videos/pilots/<style>/<style>_seed<seed>.gif.

The flights are flown as pilots.py flies them: the same pilot classes, the same randomness rule for each pilot
(numpy generator [seed, style index, 17]) and, for the policy pilots (blind_ppo, replay_good, aware_good and the policy
parts of switching), the PPO policies listed in dataset v2's manifest. pilots.fly refuses the validation and final-test
seeds so that no training flight uses them; these flights only make videos (nothing trains or is scored on them), so
this script flies them itself, step by step as pilots.fly does.

Non-leaking pilots fly a hidden-mode plane. Its observation is the observed mode's first four values with the same noise
draws, so the flight is the one pilots.fly would record, and the panel correctly says the battery is hidden. The
leaking aware_good reads the battery, so it flies an observed-mode plane. The panel's "battery (true)" bar is the
viewer's, not the pilot's.

Every step is rendered (the trail needs every step); every --every-th frame goes into the GIF, scaled by --scale and
played at 20 frames per second, so --every 2 is twice real time. The last frame holds for 2 s. A caption bar on top
names the pilot, the seed, the starting battery and the outcome.

    python pilot_gifs.py                                    # 13 styles x seeds 2000-2002, in parallel
    python pilot_gifs.py --styles probe_hold --seeds 2000   # one GIF
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")            # pygame renders off screen
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (2000, 2001, 2002)                            # the final-test seeds (config/cppo.yaml final_test)
DEFAULT_MANIFEST = PROJECT_ROOT / "runs_encoder" / "pilot_flights" / "v2" / "manifest.json"
FPS = 20
OUTCOME_TEXT = {"completed": "completed", "battery_empty": "battery empty", "ground_crash": "ground crash",
                "truncated": "truncated"}


def fly_frames(style: str, seed: int, policies: dict, every: int, scale: float) -> tuple[list[np.ndarray], dict]:
    """One flight of a pilot style on a seed: the frames kept for the GIF, and the flight's summary."""
    import torch
    from PIL import Image

    import battery_plane as bp
    from flight_logging import terminal_outcome
    import pilots

    torch.set_num_threads(1)
    pilot = pilots.make_pilot(style, pilots.PlanePhysics({}), policies, {})
    leaks = pilot.observation == "observed"
    env = bp.BatteryPlaneEnv(observation_mode="observed" if leaks else "hidden", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    b0 = float(info["b0"])
    pilot.reset(np.random.default_rng([seed, pilots.STYLES.index(style), 17]), obs, seed)
    size = (int(round(env.W * scale)), int(round(env.H * scale)))
    frames, total = [], 0.0
    for t in range(env.core.T + 1):
        frame = env.render()
        if t % every == 0:
            frames.append(np.asarray(Image.fromarray(frame).resize(size, Image.LANCZOS)))
        if t == env.core.T:
            break
        action = np.clip(np.asarray(pilot.act(obs, t), np.float32).reshape(2), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            frames.append(np.asarray(Image.fromarray(env.render()).resize(size, Image.LANCZOS)))
            break
    env.close()
    return frames, {"style": style, "seed": seed, "b0": b0, "leaks": leaks, "steps": t + 1, "seconds": (t + 1) * env.core.dt,
                    "outcome": terminal_outcome(info), "return": total}


def caption(summary: dict) -> str:
    seen = "reads the battery (leaks)" if summary["leaks"] else "battery hidden"
    return (f"pilot: {summary['style']}  |  seed {summary['seed']}  |  start battery {summary['b0']:.0%}  |  {seen}  |  "
            f"{OUTCOME_TEXT.get(summary['outcome'], summary['outcome'])} at {summary['seconds']:.1f} s, return {summary['return']:.0f}")


def make_gif(job: tuple) -> dict:
    """Fly, caption and save one GIF; returns its summary with the file and its size."""
    style, seed, policies, every, scale, output = job
    frames, summary = fly_frames(style, seed, policies, every, scale)
    path = Path(output) / style / f"{style}_seed{seed}.gif"
    return {**summary, "file": str(path), "frames": len(frames), "megabytes": save_gif(frames, caption(summary), path, scale)}


def save_gif(frames: list[np.ndarray], text: str, path: Path, scale: float) -> float:
    """frames (RGB arrays) under a caption bar, as a looping GIF at FPS with one shared palette and the last frame held
    for 2 s; returns the file's size in MB."""
    from PIL import Image, ImageDraw, ImageFont

    height, width = frames[0].shape[:2]
    bar = max(22, int(round(36 * scale)))
    size = max(11, int(round(18 * scale)))
    font = ImageFont.load_default(size=size)
    while font.getlength(text) > width - 16 and size > 8:      # shrink until the caption fits the frame
        size -= 1
        font = ImageFont.load_default(size=size)
    images = []
    for frame in frames:
        canvas = Image.new("RGB", (width, height + bar), (24, 26, 34))
        canvas.paste(Image.fromarray(frame), (0, bar))
        ImageDraw.Draw(canvas).text((8, bar // 2), text, fill=(235, 235, 235), font=font, anchor="lm")
        images.append(canvas)
    # One palette for the whole GIF (no flicker between frames), from frames spread over the flight
    samples = [images[i] for i in np.linspace(0, len(images) - 1, min(12, len(images))).astype(int)]
    sheet = Image.new("RGB", (width, (height + bar) * len(samples)))
    for k, image in enumerate(samples):
        sheet.paste(image, (0, k * (height + bar)))
    palette = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    paletted = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    durations = [1000 // FPS] * (len(paletted) - 1) + [2000]
    path.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(path, save_all=True, append_images=paletted[1:], duration=durations, loop=0, optimize=True, disposal=1)
    return path.stat().st_size / 2 ** 20


def main() -> None:
    import pilots

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--styles", nargs="+", default=list(pilots.STYLES), choices=pilots.STYLES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Dataset manifest naming the policy pilots")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "videos" / "pilots")
    parser.add_argument("--every", type=int, default=2, help="Keep every n-th step (20 fps playback: n x real time)")
    parser.add_argument("--scale", type=float, default=2 / 3, help="Frame scale (960 x 540 at 1)")
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    policies = json.loads(args.manifest.read_text())["policies"]
    jobs = [(style, seed, policies, args.every, args.scale, str(args.output)) for style in args.styles for seed in args.seeds]
    with ProcessPoolExecutor(min(args.jobs, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(make_gif, jobs))
    for r in results:
        print(f"{r['style']:13s} seed {r['seed']}: start {r['b0']:.0%}, {r['outcome']:13s} at {r['seconds']:5.1f} s, "
              f"return {r['return']:6.1f} | {r['frames']} frames, {r['megabytes']:.1f} MB -> {r['file']}", flush=True)
    print(f"{len(results)} GIFs, {sum(r['megabytes'] for r in results):.0f} MB in total", flush=True)


if __name__ == "__main__":
    main()
