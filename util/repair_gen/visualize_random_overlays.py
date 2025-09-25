#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_random_overlays.py

Run from dataset root (e.g., gt_datasets_20250915). Operates ONLY on:
  ./negative/random/random_*/output.json + original_image.jpg

Draws:
- defect contours (labels NOT starting with 'rps_points:') in CYAN, sharp corners
- rps_points:10   (laser_cut)      in RED, rounded caps/joins
- rps_points:11   (ITO_removal)    in YELLOW @ 0.5 alpha, rounded caps/joins
- rps_points:110  (U-left / C)     in YELLOW @ 0.5 alpha, same width as ITO_removal
- rps_points:112  (U-right / rev C)in YELLOW @ 0.5 alpha, same width as ITO_removal

Saves: repair_image.jpg (overwrites if exists)

Usage:
  python visualize_random_overlays.py
  python visualize_random_overlays.py --root . --verbose
  python visualize_random_overlays.py --defect-open
"""

import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from colorama import init as colorama_init, Fore, Style

# ---- Colorized CLI tags ----
colorama_init(autoreset=True)
OK   = Fore.GREEN + "[OK]" + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
ERR  = Fore.RED + "[ERR]" + Style.RESET_ALL
INFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL

# ---- Colors & widths (keep identical to your other script) ----
DEFECT_COLOR      = (0, 255, 255)     # cyan
LASER_COLOR       = (255, 0, 0)       # red
ITO_RGB           = (255, 255, 0)     # yellow
ITO_ALPHA_FLOAT   = 0.5               # 0.5 alpha

DEF_WIDTH         = 4                 # defect
LASER_WIDTH       = 7                 # rps:10
ITO_WIDTH         = 20                # rps:11, 110, 112

# ---- Helpers ----

def clamp_point(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    xi = max(0, min(int(round(x)), w - 1))
    yi = max(0, min(int(round(y)), h - 1))
    return xi, yi

def draw_filled_circle(draw: ImageDraw.ImageDraw, center: Tuple[int, int],
                       radius: int, color: Tuple[int, int, int, int]) -> None:
    if radius <= 0:
        return
    cx, cy = center
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(bbox, fill=color, outline=color)

def draw_polyline_round(draw: ImageDraw.ImageDraw, pts: List[Tuple[int, int]],
                        color: Tuple[int, int, int, int], width: int, closed: bool=False) -> None:
    """Rounded joins by stamping circles at vertices + a line."""
    if len(pts) < 2:
        if len(pts) == 1:
            draw_filled_circle(draw, pts[0], max(1, width // 2), color)
        return
    if closed:
        draw.line(pts + [pts[0]], fill=color, width=width)
    else:
        draw.line(pts, fill=color, width=width)
    r = max(1, width // 2)
    for p in pts:
        draw_filled_circle(draw, p, r, color)

def draw_polyline_sharp(draw: ImageDraw.ImageDraw, pts: List[Tuple[int, int]],
                        color: Tuple[int, int, int, int], width: int, closed: bool=False) -> None:
    if len(pts) < 2:
        if len(pts) == 1:
            draw.point(pts[0], fill=color)
        return
    if closed:
        draw.line(pts + [pts[0]], fill=color, width=width)
    else:
        draw.line(pts, fill=color, width=width)

def load_output_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_contours(contours: List[Dict[str, Any]]):
    """
    Returns:
      laser_segs: list[[p1,p2]] for rps_points:10
      ito_segs:   list[[p1,p2]] for rps_points:11
      u110:       list[[p1,p3]] (diagonal corners)
      u112:       list[[p1,p3]] (diagonal corners)
      defect_polylines: list[list[pt]]
    """
    laser, ito, u110, u112, defect = [], [], [], [], []
    for item in contours:
        label = str(item.get("label", ""))
        pts = item.get("points", [])
        if label.startswith("rps_points:"):
            if label == "rps_points:10" and len(pts) >= 2:
                laser.append(pts[:2])
            elif label == "rps_points:11" and len(pts) >= 2:
                ito.append(pts[:2])
            elif label == "rps_points:110" and len(pts) >= 2:
                u110.append(pts[:2])
            elif label == "rps_points:112" and len(pts) >= 2:
                u112.append(pts[:2])
            # ignore 111/113 by default
        else:
            if len(pts) >= 2:
                defect.append(pts)
    return laser, ito, u110, u112, defect

def u_shape_segments(p1: Tuple[int,int], p3: Tuple[int,int], kind: str):
    """
    Build segments for U-left (110) or U-right (112) using p1, p3 as diagonal corners.
    Matches your previous logic:
      p2 = (p3.x, p1.y)
      p4 = (p1.x, p3.y)
    - 110 (U-left / C):   draw p1->p2, p2->p3, p3->p4 (missing right side)
    - 112 (U-right / rC): draw p1->p2, p3->p4, p4->p1 (missing left side)
    Returns list of polylines (each as [ptA, ptB]) to draw.
    """
    p2 = (p3[0], p1[1])
    p4 = (p1[0], p3[1])
    if kind == "110":
        return [[p1, p2], [p2, p3], [p3, p4]]
    elif kind == "112":
        return [[p1, p2], [p3, p4], [p4, p1]]
    else:
        return []

def process_sample(sample_dir: str, defect_open: bool, verbose: bool=False) -> bool:
    img_path = os.path.join(sample_dir, "original_image.jpg")
    out_json = os.path.join(sample_dir, "output.json")
    if not os.path.isfile(img_path):
        if verbose: print(f"{WARN} {sample_dir}: missing original_image.jpg")
        return False
    if not os.path.isfile(out_json):
        if verbose: print(f"{WARN} {sample_dir}: missing output.json")
        return False

    try:
        base = Image.open(img_path).convert("RGB")
    except Exception as e:
        if verbose: print(f"{WARN} {sample_dir}: failed to open original_image.jpg: {e}")
        return False

    data = load_output_json(out_json)
    contours = data.get("contours", [])

    w, h = base.size

    # Split groups
    laser_segs, ito_segs, u110_pairs, u112_pairs, defect_lines = group_contours(contours)

    # Prepare drawing contexts
    base_rgba = base.convert("RGBA")
    draw_base = ImageDraw.Draw(base_rgba)

    # Semi-transparent overlay for all yellow ops (11, 110, 112)
    yellow_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_yellow = ImageDraw.Draw(yellow_overlay)

    # colors
    red_rgba   = (*LASER_COLOR, 255)
    cyan_rgba  = (*DEFECT_COLOR, 255)
    yellow_a   = (*ITO_RGB, int(round(ITO_ALPHA_FLOAT * 255)))

    # Draw defects (sharp)
    for pts in defect_lines:
        ipts = [clamp_point(x, y, w, h) for (x, y) in pts]
        draw_polyline_sharp(draw_base, ipts, color=cyan_rgba, width=DEF_WIDTH,
                            closed=(not defect_open and len(ipts) >= 3))

    # Draw laser cuts (rounded, opaque) rps:10
    for seg in laser_segs:
        p1 = clamp_point(seg[0][0], seg[0][1], w, h)
        p2 = clamp_point(seg[1][0], seg[1][1], w, h)
        draw_polyline_round(draw_base, [p1, p2], color=red_rgba, width=LASER_WIDTH, closed=False)

    # Draw ITO removal (rounded, semi-transparent) rps:11
    for seg in ito_segs:
        p1 = clamp_point(seg[0][0], seg[0][1], w, h)
        p2 = clamp_point(seg[1][0], seg[1][1], w, h)
        draw_polyline_round(draw_yellow, [p1, p2], color=yellow_a, width=ITO_WIDTH, closed=False)

    # Draw U-left (110) and U-right (112) with SAME style as ITO removal
    for pair in u110_pairs:
        p1 = clamp_point(pair[0][0], pair[0][1], w, h)
        p3 = clamp_point(pair[1][0], pair[1][1], w, h)
        for seg in u_shape_segments(p1, p3, "110"):
            draw_polyline_round(draw_yellow, seg, color=yellow_a, width=ITO_WIDTH, closed=False)

    for pair in u112_pairs:
        p1 = clamp_point(pair[0][0], pair[0][1], w, h)
        p3 = clamp_point(pair[1][0], pair[1][1], w, h)
        for seg in u_shape_segments(p1, p3, "112"):
            draw_polyline_round(draw_yellow, seg, color=yellow_a, width=ITO_WIDTH, closed=False)

    # Composite yellow overlay onto base
    composed = Image.alpha_composite(base_rgba, yellow_overlay)

    # Save
    save_path = os.path.join(sample_dir, "repair_image.jpg")
    composed.convert("RGB").save(save_path, quality=95)
    print(f"{OK} Wrote {os.path.relpath(save_path)}")
    return True

def main():
    ap = argparse.ArgumentParser(description="Visualize contours on original_image.jpg for negative/random samples.")
    ap.add_argument("--root", type=str, default=".", help="Dataset root (run from gt_datasets_20250915).")
    ap.add_argument("--defect-open", action="store_true", help="Draw defect polylines open (not closed).")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    random_root = os.path.join(os.path.abspath(args.root), "negative", "random")
    if not os.path.isdir(random_root):
        print(f"{ERR} Not found: {random_root}")
        return

    sample_dirs = sorted([p for p in glob(os.path.join(random_root, "random_*")) if os.path.isdir(p)])
    if args.verbose:
        print(f"{INFO} Found {len(sample_dirs)} random sample(s) under {random_root}")

    processed = 0
    for sdir in sample_dirs:
        ok = process_sample(sdir, defect_open=args.defect_open, verbose=args.verbose)
        if ok:
            processed += 1

    print(f"{OK} Processed {processed} sample(s).")

if __name__ == "__main__":
    main()
