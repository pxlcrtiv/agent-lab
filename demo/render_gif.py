#!/usr/bin/env python3
"""Render demo/session.txt into demo/agent-lab-demo.gif.

Pure PIL animation: progressive terminal-style frames (Monaco, green on
dark), each held a few ticks for readability. Needs Pillow only -- the
library itself stays dependency-free.

Usage: python3 demo/render_gif.py [--hold 6] [--fps 15] [--width 720]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (13, 17, 23)
FG = (168, 255, 176)
DIM = (110, 160, 118)
FONT_PATH = "/System/Library/Fonts/Monaco.ttf"

CANVAS_W = 960
CANVAS_H = 540
MARGIN = 28
LINESPACE = 8


def load_font() -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, 15)


def pick_size(font: ImageFont.FreeTypeFont, lines: list[str], max_w: int) -> ImageFont.FreeTypeFont:
    """Shrink the font until the longest line fits the canvas."""
    size = 15
    while size > 9:
        f = ImageFont.truetype(FONT_PATH, size)
        if max(f.getlength(ln) for ln in lines) <= max_w:
            return f
        size -= 1
    return ImageFont.truetype(FONT_PATH, size)


def draw_frame(draw: ImageDraw, font: ImageFont.FreeTypeFont, lines: list[str],
               dim_prefix: str = None) -> None:
    draw.rectangle([0, 0, CANVAS_W, CANVAS_H], fill=BG)
    y = MARGIN
    for ln in lines:
        color = FG
        if dim_prefix and ln.startswith(dim_prefix):
            color = DIM
        draw.text((MARGIN, y), ln, font=font, fill=color)
        y += font.size + LINESPACE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=6)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=720)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    session = (root / "demo" / "session.txt").read_text(encoding="utf-8").splitlines()
    font = pick_size(load_font(), session, CANVAS_W - 2 * MARGIN)

    base = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    frames: list[Image.Image] = []
    for i in range(len(session)):
        draw = ImageDraw.Draw(base)
        draw_frame(draw, font, session[: i + 1], dim_prefix="$ ")
        frames.append(base.copy())

    held: list[Image.Image] = []
    for f in frames:
        held.extend([f] * args.hold)

    out_path = root / "demo" / "agent-lab-demo.gif"
    scaled = [f.resize((args.width, int(CANVAS_H * args.width / CANVAS_W)), Image.LANCZOS) for f in held]
    scaled[0].save(
        out_path,
        save_all=True,
        append_images=scaled[1:],
        duration=max(1, int(1000 / args.fps)),
        loop=0,
        optimize=True,
    )
    print("wrote %s (%d frames, %.1fs at %dfps)" % (
        out_path, len(scaled), len(scaled) / args.fps, args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())