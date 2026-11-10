"""Assemble screenshot frames into a small looping GIF (used by frontend/scripts/demo-gif.mjs).

Usage:  python scripts/assemble_gif.py <frames dir> <out.gif>

Frames are scaled to 960 px wide and share one 256-colour palette, so the map's colours stay steady
from frame to frame. The last frame holds longer before the loop restarts.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

WIDTH, FRAME_MS, HOLD_MS = 960, 420, 1800


def main(frames_dir: str, out: str) -> None:
    paths = sorted(Path(frames_dir).glob("*.png"))
    if not paths:
        raise SystemExit(f"no frames in {frames_dir}")
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        frames.append(im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.Resampling.LANCZOS))
    # one palette for every frame, from a sample of frames plus the exact data colours of both themes,
    # so the four congestion shades never merge into one
    data_colours = [
        "#dd9347",
        "#d9692b",
        "#c93a26",
        "#8f1d17",
        "#8a93a3",
        "#2f5f9e",
        "#e0a458",
        "#e27d3e",
        "#dd4f35",
        "#b42b20",
        "#5f6d86",
        "#76a3da",
    ]
    picks = frames[:: max(1, len(frames) // 4)]
    h = frames[0].height
    sheet = Image.new("RGB", (WIDTH, h * len(picks) + 40 * len(data_colours)))
    for i, f in enumerate(picks):
        sheet.paste(f, (0, i * h))
    for i, c in enumerate(data_colours):
        sheet.paste(Image.new("RGB", (WIDTH, 40), c), (0, h * len(picks) + 40 * i))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    durations = [FRAME_MS] * (len(quantized) - 1) + [HOLD_MS]
    quantized[0].save(
        out, save_all=True, append_images=quantized[1:], duration=durations, loop=0, optimize=True
    )
    print(f"wrote {out} ({len(quantized)} frames, {Path(out).stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
