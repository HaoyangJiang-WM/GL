#!/usr/bin/env python3
"""Overlay a new title on the case-5 comparison PNG.

This is useful when the original plotting script is unavailable.  It edits the
PNG directly by covering the existing top title band and drawing a new title.

Usage from repository root:

    python scripts/patch_case5_figure_title.py

or customize:

    python scripts/patch_case5_figure_title.py \
        --input figures/case5_all_comparisons.png \
        --output figures/case5_all_comparisons.png \
        --title "OT-FM C2F-SVGD-FM: Case-5 Velocity Comparisons" \
        --top 95 --font-size 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def draw_centered_title(draw: ImageDraw.ImageDraw, width: int, band_h: int, title: str, font, fill: str) -> None:
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = max(0, (width - tw) // 2)
    y = max(0, (band_h - th) // 2 - 2)
    draw.text((x, y), title, fill=fill, font=font)


def patch_title(
    input_path: Path,
    output_path: Path,
    title: str,
    top: int,
    font_size: int,
    bg: str,
    fg: str,
    mode: str,
) -> None:
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    font = load_font(font_size)

    if mode == "cover":
        out = img.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle([0, 0, w, top], fill=bg)
        draw_centered_title(draw, w, top, title, font, fg)
    elif mode == "prepend":
        out = Image.new("RGB", (w, h + top), bg)
        out.paste(img, (0, top))
        draw = ImageDraw.Draw(out)
        draw_centered_title(draw, w, top, title, font, fg)
    else:
        raise ValueError(f"unknown mode: {mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    print(f"saved {output_path}  size={out.size}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="figures/case5_all_comparisons.png")
    ap.add_argument("--output", default="figures/case5_all_comparisons.png")
    ap.add_argument("--title", default="OT-FM C2F-SVGD-FM: Case-5 Velocity Comparisons")
    ap.add_argument("--top", type=int, default=95, help="height of the top title band to cover/prepend")
    ap.add_argument("--font-size", type=int, default=42)
    ap.add_argument("--bg", default="white")
    ap.add_argument("--fg", default="black")
    ap.add_argument("--mode", choices=["cover", "prepend"], default="cover")
    args = ap.parse_args()

    patch_title(
        input_path=Path(args.input),
        output_path=Path(args.output),
        title=args.title,
        top=args.top,
        font_size=args.font_size,
        bg=args.bg,
        fg=args.fg,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
