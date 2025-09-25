#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_negative_augmentor.py

Generate augmented negative samples by randomly applying shift/scale/delete
to selected rps_points labels in positive samples.

NEW:
- Deduplicate rps_points (same label + same coords) before augmenting.
- Keep U-shapes (110/112) and their vertical "cap" 11 in sync: when either
  is shifted/scaled, both receive the *same* transform so rectangles stay closed.

Usage (examples):
  python random_negative_augmentor.py --n 10
  python random_negative_augmentor.py --n 5 --include 10,11,110,112 --ops shift,scale --shift-std 1.5 --scale-std 0.03
  python random_negative_augmentor.py --n 20 --p-delete 0.2 --seed 42

Notes:
- At least one augmentation (shift/scale/delete) is guaranteed per sample.
- By default, only rps_points: {10,11,110,112} are considered.
- 111 and 113 can be added via --include if desired.
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from glob import glob
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from colorama import init as colorama_init, Fore, Style

# ---------- Pretty CLI tags ----------
colorama_init(autoreset=True)
OK = Fore.GREEN + "[OK]" + Style.RESET_ALL
ERR = Fore.RED + "[ERR]" + Style.RESET_ALL
INFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL

# ---------- Helpers ----------

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def list_positive_samples(root: str) -> List[str]:
    """Return list of sample directories that contain output.json and original_image.jpg."""
    candidates = []
    pos_root = os.path.join(root, "positive")
    if not os.path.isdir(pos_root):
        return candidates
    for cls in sorted(os.listdir(pos_root)):
        cls_dir = os.path.join(pos_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for sample in sorted(os.listdir(cls_dir)):
            sdir = os.path.join(cls_dir, sample)
            if not os.path.isdir(sdir):
                continue
            if os.path.isfile(os.path.join(sdir, "output.json")) and os.path.isfile(os.path.join(sdir, "original_image.jpg")):
                candidates.append(sdir)
    return candidates

def parse_rps_type(label: str) -> str:
    """
    Return the type string after 'rps_points:' or '' if not an rps_points label.
    e.g., 'rps_points:110' -> '110'
    """
    if not label.startswith("rps_points:"):
        return ""
    return label.split(":", 1)[1].strip()

def clamp_point(x: float, y: float, w: int, h: int) -> Tuple[float, float]:
    return float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))

def points_center(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)

def axis_of_segment(p1: Tuple[float, float], p2: Tuple[float, float]) -> str:
    """Return 'h' (horizontal), 'v' (vertical), or 'diag'."""
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dy < 1e-6 and dx >= dy:
        return "h"
    if dx < 1e-6 and dy > dx:
        return "v"
    return "h" if dx >= dy else "v"

def gaussian(mu: float, sigma: float) -> float:
    return float(np.random.normal(mu, sigma))

# ---------- Geometric transforms ----------

def transform_line_with_params(
    pts: List[List[float]],
    image_w: int,
    image_h: int,
    cx: float, cy: float,
    sx: float, sy: float,
    dx: float, dy: float
) -> List[List[float]]:
    """Apply the same (centered) scale+shift to a 2-point segment."""
    if len(pts) < 2:
        return pts
    def apply(p):
        x, y = float(p[0]), float(p[1])
        x = cx + (x - cx) * sx + dx
        y = cy + (y - cy) * sy + dy
        return clamp_point(x, y, image_w, image_h)
    q1 = apply(pts[0]); q2 = apply(pts[1])
    return [[round(q1[0], 2), round(q1[1], 2)],
            [round(q2[0], 2), round(q2[1], 2)]]

def transform_line_auto(
    pts: List[List[float]],
    image_w: int,
    image_h: int,
    do_shift: bool,
    do_scale: bool,
    shift_std: float,
    scale_std: float,
    axis_specific: bool = True,
) -> Tuple[List[List[float]], Dict[str, float]]:
    """
    Transform a 2-point segment with optional shift and scale (returns params used).
    If axis_specific=True, scale along major axis only.
    """
    if len(pts) < 2:
        return pts, {"cx": 0, "cy": 0, "sx": 1, "sy": 1, "dx": 0, "dy": 0}
    p1 = (float(pts[0][0]), float(pts[0][1]))
    p2 = (float(pts[1][0]), float(pts[1][1]))
    cx, cy = points_center(p1, p2)
    dx = gaussian(0.0, shift_std) if do_shift else 0.0
    dy = gaussian(0.0, shift_std) if do_shift else 0.0
    if do_scale:
        if axis_specific:
            ax = axis_of_segment(p1, p2)
            if ax == "h":
                sx = 1.0 + gaussian(0.0, scale_std); sy = 1.0
            elif ax == "v":
                sx = 1.0; sy = 1.0 + gaussian(0.0, scale_std)
            else:
                sx = 1.0 + gaussian(0.0, scale_std)
                sy = 1.0 + gaussian(0.0, scale_std)
        else:
            sx = 1.0 + gaussian(0.0, scale_std)
            sy = 1.0 + gaussian(0.0, scale_std)
    else:
        sx, sy = 1.0, 1.0
    new_pts = transform_line_with_params(pts, image_w, image_h, cx, cy, sx, sy, dx, dy)
    return new_pts, {"cx": cx, "cy": cy, "sx": sx, "sy": sy, "dx": dx, "dy": dy}

def transform_rect_with_params(
    pts: List[List[float]],
    image_w: int,
    image_h: int,
    cx: float, cy: float,
    sx: float, sy: float,
    dx: float, dy: float
) -> List[List[float]]:
    """Apply the same (centered) scale+shift to a diag-pair rectangle (110/112)."""
    if len(pts) < 2:
        return pts
    def apply(p):
        x, y = float(p[0]), float(p[1])
        x = cx + (x - cx) * sx + dx
        y = cy + (y - cy) * sy + dy
        return clamp_point(x, y, image_w, image_h)
    p1 = apply(pts[0]); p3 = apply(pts[1])
    return [[round(p1[0], 2), round(p1[1], 2)],
            [round(p3[0], 2), round(p3[1], 2)]]

def rect_center(pts: List[List[float]]) -> Tuple[float, float]:
    p1 = (float(pts[0][0]), float(pts[0][1]))
    p3 = (float(pts[1][0]), float(pts[1][1]))
    return points_center(p1, p3)

# ---------- Dedup & pairing ----------

def _norm_point(p: List[float], ndigits: int = 3) -> Tuple[float, float]:
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits))

def _unordered_pair_key(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return tuple(sorted([a, b]))

def dedup_rps_contours(contours: List[Dict[str, Any]], ndigits: int = 3) -> Tuple[List[Dict[str, Any]], int]:
    """
    Remove exact duplicate rps_points entries (same label + same coords).
    For 2-point items, treat endpoints as unordered.
    """
    seen = set()
    out = []
    removed = 0
    for item in contours:
        label = str(item.get("label", ""))
        pts = item.get("points", [])
        if not label.startswith("rps_points:") or len(pts) < 2:
            out.append(item)
            continue
        if len(pts) == 2:
            a = _norm_point(pts[0], ndigits)
            b = _norm_point(pts[1], ndigits)
            key_pts = _unordered_pair_key(a, b)
        else:
            key_pts = tuple(_norm_point(p, ndigits) for p in pts)
        key = (label, key_pts)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(item)
    return out, removed

def _match_cap_for_u(
    u_pts: List[List[float]], kind: str, all_contours: List[Dict[str, Any]],
    eps: float = 1.0
) -> Optional[int]:
    """
    Find an index of rps_points:11 that acts as the closing vertical cap for a U.
    For 110 (U-left, missing left side), cap is vertical at x1: (x1,y1)-(x1,y3)
    For 112 (U-right, missing right side), cap is vertical at x3: (x3,y1)-(x3,y3)
    """
    if len(u_pts) < 2:
        return None
    x1, y1 = float(u_pts[0][0]), float(u_pts[0][1])
    x3, y3 = float(u_pts[1][0]), float(u_pts[1][1])
    if kind == "110":
        pA = (x1, y1); pB = (x1, y3)
    elif kind == "112":
        pA = (x3, y1); pB = (x3, y3)
    else:
        return None

    def close(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return (abs(a[0] - b[0]) <= eps) and (abs(a[1] - b[1]) <= eps)

    for j, item in enumerate(all_contours):
        if str(item.get("label", "")) != "rps_points:11":
            continue
        pts = item.get("points", [])
        if len(pts) < 2:
            continue
        q1 = (float(pts[0][0]), float(pts[0][1]))
        q2 = (float(pts[1][0]), float(pts[1][1]))
        if (close(pA, q1) and close(pB, q2)) or (close(pA, q2) and close(pB, q1)):
            return j
    return None

def build_groups(contours: List[Dict[str, Any]]) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Group indices for paired augmentation:
      - [u_index, cap_index] for each (110/112, 11) match.
      - Singletons [i] for all other rps_points.
    Returns:
      groups: leader_index -> [member_indices...]
      leader_of: index -> leader_index
    """
    n = len(contours)
    used = set()
    groups: Dict[int, List[int]] = {}
    leader_of: Dict[int, int] = {}

    for i, it in enumerate(contours):
        if i in used:
            continue
        label = str(it.get("label", ""))
        t = parse_rps_type(label)
        if t in ("110", "112"):
            cap_idx = _match_cap_for_u(it.get("points", []), t, contours, eps=1.0)
            if cap_idx is not None and cap_idx != i and cap_idx not in used:
                leader = min(i, cap_idx)
                members = [i, cap_idx] if i == leader else [cap_idx, i]
                groups[leader] = members
                leader_of[i] = leader
                leader_of[cap_idx] = leader
                used.add(i); used.add(cap_idx)
                continue
        # fallback: singleton if not grouped
        groups[i] = [i]
        leader_of[i] = i
        used.add(i)

    return groups, leader_of

# ---------- Main augmentation engine ----------

def augment_one_sample(
    sample_dir: str,
    include_types: List[str],
    allowed_ops: List[str],
    image_w: int,
    image_h: int,
    shift_std: float,
    scale_std: float,
    p_delete: float,
    max_attempts: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (new_output_json, meta_json). Ensures at least one augmentation occurred.
    May raise ValueError if the sample has no eligible rps_points.
    """
    output_path = os.path.join(sample_dir, "output.json")
    data = read_json(output_path)
    orig_contours = data.get("contours", [])

    # 1) Dedup rps_points
    contours, removed_dups = dedup_rps_contours(orig_contours, ndigits=3)

    # 2) Collect indices of eligible rps
    rps_idx = []
    for i, item in enumerate(contours):
        t = parse_rps_type(item.get("label", ""))
        if t and t in include_types and len(item.get("points", [])) >= 2:
            rps_idx.append(i)

    if not rps_idx:
        raise ValueError("No eligible rps_points of the requested types in this sample.")

    # 3) Build pairing groups (110/112 with their cap 11 if present)
    groups, leader_of = build_groups(contours)

    # Attempt augmentation until at least one op happens
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        new_contours: List[Dict[str, Any]] = []
        ops_log: List[Dict[str, Any]] = []
        augmented_any = False

        # choose some rps indices to *touch* (we'll promote to group-level)
        n_to_touch = random.randint(1, len(rps_idx))
        touch_set = set(random.sample(rps_idx, n_to_touch))
        # promote to leaders
        touch_leaders = set(leader_of[i] for i in touch_set)

        # iterate in original order; process groups at their leader
        processed = set()
        for i, item in enumerate(contours):
            if i in processed:
                continue

            label = str(item.get("label", ""))
            t = parse_rps_type(label)

            # Non-rps: passthrough
            if not t:
                new_contours.append(item)
                processed.add(i)
                continue

            leader = leader_of.get(i, i)
            if leader != i:
                # will be handled when leader iteration comes
                continue

            members = groups[leader]  # 1 or 2 indices
            # Is this group touched?
            group_touched = leader in touch_leaders or any(m in touch_set for m in members)

            # Per-member delete decisions (independent)
            will_delete = {}
            if group_touched and "delete" in allowed_ops:
                for m in members:
                    # deletion chance only for the member that is touched directly; optional: allow for both
                    will_delete[m] = (m in touch_set) and (random.random() < p_delete)
            else:
                for m in members:
                    will_delete[m] = False

            # Shift/scale decision: if ANY member is touched → apply SAME transform to BOTH
            want_shift = group_touched and ("shift" in allowed_ops)
            want_scale = group_touched and ("scale" in allowed_ops)

            # Compute transform if needed
            group_params = None
            if want_shift or want_scale:
                # Use the rectangle if present to define center; otherwise any member line.
                rect_idx = None
                rect_type = None
                for m in members:
                    mt = parse_rps_type(contours[m].get("label", ""))
                    if mt in ("110", "112"):
                        rect_idx = m; rect_type = mt; break

                if rect_idx is not None:
                    # center from rectangle diagonal
                    rect_pts = contours[rect_idx].get("points", [])
                    cx, cy = rect_center(rect_pts)
                else:
                    # center from first member's segment
                    seg = contours[members[0]].get("points", [])
                    if len(seg) >= 2:
                        p1 = (float(seg[0][0]), float(seg[0][1]))
                        p2 = (float(seg[1][0]), float(seg[1][1]))
                        cx, cy = points_center(p1, p2)
                    else:
                        cx, cy = (0.0, 0.0)

                dx = gaussian(0.0, shift_std) if want_shift else 0.0
                dy = gaussian(0.0, shift_std) if want_shift else 0.0
                sx = 1.0 + gaussian(0.0, scale_std) if want_scale else 1.0
                sy = 1.0 + gaussian(0.0, scale_std) if want_scale else 1.0
                group_params = {"cx": cx, "cy": cy, "sx": sx, "sy": sy, "dx": dx, "dy": dy}

            # Apply to each member in this group (respect deletion)
            for m in members:
                it_m = contours[m]
                lbl_m = str(it_m.get("label", ""))
                typ_m = parse_rps_type(lbl_m)
                pts_m = it_m.get("points", [])

                if will_delete.get(m, False):
                    ops_log.append({
                        "label": lbl_m, "op": "delete",
                        "params": {}, "paired_group": members
                    })
                    augmented_any = True
                    # don't append this member
                    continue

                if group_params is None:
                    # untouched (or only deletions happened for other members)
                    new_contours.append(it_m)
                    continue

                # Shift/scale with group_params (same for both members)
                cx = group_params["cx"]; cy = group_params["cy"]
                sx = group_params["sx"]; sy = group_params["sy"]
                dx = group_params["dx"]; dy = group_params["dy"]

                if typ_m in ("110", "112"):
                    new_pts = transform_rect_with_params(pts_m, image_w, image_h, cx, cy, sx, sy, dx, dy)
                else:
                    # lines (10/11) — use same 2D params (not axis-specific) so caps stay closed
                    new_pts = transform_line_with_params(pts_m, image_w, image_h, cx, cy, sx, sy, dx, dy)

                new_item = dict(it_m)
                new_item["points"] = new_pts
                new_contours.append(new_item)
                augmented_any = True

            # Log the shared transform once per group (if any)
            if group_params is not None:
                ops_log.append({
                    "label_group": [contours[m]["label"] for m in members],
                    "op": "+".join(filter(None, ["shift" if want_shift else "", "scale" if want_scale else ""])) or "none",
                    "shared_transform": group_params,
                    "members": members
                })

            processed.update(members)

        if augmented_any:
            new_output = dict(data)
            new_output["contours"] = new_contours
            meta = {
                "origin": {
                    "class": os.path.basename(os.path.dirname(sample_dir)),
                    "sample_id": os.path.basename(sample_dir),
                    "relative_path": os.path.relpath(sample_dir, start=os.getcwd())
                },
                "rps_pool": include_types,
                "allowed_ops": allowed_ops,
                "params": {
                    "shift_std_px": shift_std,
                    "scale_std": scale_std,
                    "p_delete": p_delete
                },
                "ops_applied": ops_log,
                "dedup_removed": removed_dups,
                "seed": random.getstate()[1][0] if hasattr(random, "getstate") else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            return new_output, meta
        # else: retry

    raise RuntimeError("Failed to produce at least one augmentation after multiple attempts.")

# ---------- IO helpers ----------

def next_random_index(out_root: str) -> int:
    """Find next random_<idx> index under out_root (negative/random)."""
    os.makedirs(out_root, exist_ok=True)
    existing = glob(os.path.join(out_root, "random_*"))
    mx = 0
    for p in existing:
        m = re.search(r"random_(\d+)$", p)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1

def copy_inheritance(src_sample_dir: str, dst_dir: str) -> None:
    """Copy masks/ and original_image.jpg from src to dst."""
    os.makedirs(dst_dir, exist_ok=True)
    # original image
    src_img = os.path.join(src_sample_dir, "original_image.jpg")
    if os.path.isfile(src_img):
        shutil.copy2(src_img, os.path.join(dst_dir, "original_image.jpg"))
    # masks folder
    src_masks = os.path.join(src_sample_dir, "masks")
    dst_masks = os.path.join(dst_dir, "masks")
    if os.path.isdir(src_masks):
        if os.path.isdir(dst_masks):
            shutil.rmtree(dst_masks)
        shutil.copytree(src_masks, dst_masks)

def detect_image_size(sample_dir: str, default_w: int, default_h: int) -> Tuple[int, int]:
    """
    Try to read original_image.jpg to get (W, H). If not possible, return defaults.
    (We keep it simple; default is usually 512x512 per your note.)
    """
    return (default_w, default_h)

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Random negative sample augmentor for rps_points.")
    parser.add_argument("--root", type=str, default=".", help="Dataset root (default: current directory).")
    parser.add_argument("--n", type=int, default=1, help="Number of negative samples to generate.")
    parser.add_argument("--include", type=str, default="10,11,110,112",
                        help="Comma-separated rps_points types to include (e.g., '10,11,110,112').")
    parser.add_argument("--ops", type=str, default="shift,delete",
                        help="Allowed operations (subset of: shift,scale,delete).")
    parser.add_argument("--shift-std", type=float, default=15.0,
                        help="Gaussian std (in pixels) for shifts (mean=0).")
    parser.add_argument("--scale-std", type=float, default=0.4,
                        help="Gaussian std for scaling (mean=1.0).")
    parser.add_argument("--p-delete", type=float, default=0.15,
                        help="Probability to delete a selected rps contour.")
    parser.add_argument("--img-w", type=int, default=512, help="Image width if not detectable.")
    parser.add_argument("--img-h", type=int, default=512, help="Image height if not detectable.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    dataset_root = os.path.abspath(args.root)
    pos_samples = list_positive_samples(dataset_root)
    if not pos_samples:
        print(f"{ERR} No positive samples found under {os.path.join(dataset_root, 'positive')}")
        sys.exit(1)

    include_types = [s.strip() for s in args.include.split(",") if s.strip()]
    allowed_ops = [s.strip().lower() for s in args.ops.split(",") if s.strip()]
    for op in allowed_ops:
        if op not in ("shift", "scale", "delete"):
            print(f"{ERR} Unknown operation in --ops: {op}")
            sys.exit(1)

    out_root = os.path.join(dataset_root, "negative", "random")
    start_idx = next_random_index(out_root)

    print(f"{INFO} Positive pool: {len(pos_samples)} samples")
    print(f"{INFO} rps include types: {include_types}")
    print(f"{INFO} allowed ops: {allowed_ops}")
    print(f"{INFO} shift_std={args.shift_std}px, scale_std={args.scale_std}, p_delete={args.p_delete}")
    print(f"{INFO} Output to: {out_root} (starting at random_{start_idx})")

    made = 0
    tries = 0
    max_global_tries = args.n * 20  # safety to avoid infinite loops if pool is odd

    while made < args.n and tries < max_global_tries:
        tries += 1
        sample_dir = random.choice(pos_samples)

        # Determine image size (fallback to CLI)
        w, h = detect_image_size(sample_dir, args.img_w, args.img_h)

        try:
            new_output, meta = augment_one_sample(
                sample_dir=sample_dir,
                include_types=include_types,
                allowed_ops=allowed_ops,
                image_w=w, image_h=h,
                shift_std=args.shift_std,
                scale_std=args.scale_std,
                p_delete=args.p_delete
            )
        except ValueError:
            # No eligible rps in this sample, pick another
            continue
        except RuntimeError as e:
            print(f"{WARN} {e} @ {sample_dir}")
            continue

        # Prepare dest dir
        idx = start_idx + made
        dst_dir = os.path.join(out_root, f"random_{idx}")
        os.makedirs(dst_dir, exist_ok=False)

        # Inherit files
        copy_inheritance(sample_dir, dst_dir)

        # Save output + meta
        write_json(new_output, os.path.join(dst_dir, "output.json"))
        write_json(meta, os.path.join(dst_dir, "meta.json"))

        print(f"{OK} Generated: {os.path.relpath(dst_dir, start=dataset_root)} "
              f"from {os.path.relpath(sample_dir, start=dataset_root)} "
              f"(ops={len(meta['ops_applied'])}, dedup_removed={meta['dedup_removed']})")
        made += 1

    if made < args.n:
        print(f"{WARN} Requested {args.n}, generated {made}. "
              f"Consider relaxing include types or ops.")


if __name__ == "__main__":
    main()
