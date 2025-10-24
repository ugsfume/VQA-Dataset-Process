#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coding_sample_matcher.py

Run **from the coding samples root** (where coding_1, coding_2, ... live).
For each sample directory that contains `repair_image.jpg`, do:

  1) Load the provided template image (large full layout PNG).
  2) Template-match the sample's `repair_image.jpg` inside the full layout
     (multi-scale, grayscale, NO rotations).
  3) Using the matched bbox, crop every component mask from the template's
     `masks/` and save them into `<sample>/masks/` (created if missing).
  4) Append/merge match info into `<sample>/meta.json`.

By default, cropped masks are saved at their cropped native size. You can
optionally letterbox them to N×N via `--normalize-size N`.

Usage:
  python coding_sample_matcher.py \
      --template /path/to/50UD \
      [--normalize-size 0] [--min-scale 0.9 --max-scale 1.1 --scale-steps 11] \
      [--score-threshold 0.6] [--save-matched-vis] [--force]

Expected template layout:
  <TEMPLATE_DIR>/<Class>.png
  <TEMPLATE_DIR>/masks/*.png
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

from colorama import init as colorama_init, Fore, Style, Back
colorama_init(autoreset=True)
LBL_WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
LBL_SKIP = Fore.MAGENTA + "[SKIP]" + Style.RESET_ALL
LBL_OK   = Fore.GREEN  + "[OK]"   + Style.RESET_ALL
LBL_INFO = Back.BLUE   + "[INFO]" + Style.RESET_ALL
LBL_DONE = Back.YELLOW + "[DONE]" + Style.RESET_ALL

# -------------------------------
# Image helpers
# -------------------------------
def letterbox(img: np.ndarray, target: int) -> np.ndarray:
    """Pad to target x target with aspect ratio preserved, black background."""
    if target <= 0:
        return img
    if img.ndim == 2:
        h, w = img.shape
        ch = 1
    else:
        h, w = img.shape[:2]
        ch = img.shape[2]
    if h == 0 or w == 0:
        return np.zeros((target, target, ch), dtype=img.dtype) if ch != 1 else np.zeros((target, target), dtype=img.dtype)
    scale = min(target / w, target / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((target, target, ch), dtype=img.dtype) if ch != 1 else np.zeros((target, target), dtype=img.dtype)
    x0 = (target - nw) // 2
    y0 = (target - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def multi_scale_match(
    full_bgr: np.ndarray,
    tmpl_bgr: np.ndarray,
    scales: List[float],
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Dict:
    """
    Match 'tmpl_bgr' inside 'full_bgr' at multiple scales (grayscale).
    Returns dict with score, scale, bbox (x1,y1,x2,y2), w,h, method.
    """
    full_gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
    H, W = full_gray.shape
    th0, tw0 = tmpl_gray.shape

    best = {"score": -1.0, "scale": None, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "w": 0, "h": 0, "method": "TM_CCOEFF_NORMED"}
    for s in scales:
        tw = int(round(tw0 * s))
        th = int(round(th0 * s))
        if tw < 8 or th < 8 or tw > W or th > H:
            continue
        t_s = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        res = cv2.matchTemplate(full_gray, t_s, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best["score"]:
            tlx, tly = max_loc
            best.update({
                "score": float(max_val), "scale": float(s),
                "x1": int(tlx), "y1": int(tly),
                "w": int(tw), "h": int(th),
                "x2": int(tlx + tw - 1), "y2": int(tly + th - 1),
            })
    return best

def draw_match(full_bgr: np.ndarray, match: Dict, out_path: str) -> None:
    vis = full_bgr.copy()
    x1, y1, x2, y2 = match["x1"], match["y1"], match["x2"], match["y2"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 225, 0), thickness=3)
    label = f"score={match['score']:.3f} scale={match['scale']:.3f}"
    cv2.putText(vis, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)

# -------------------------------
# Core ops
# -------------------------------
def build_scales(min_scale: float, max_scale: float, steps: int) -> List[float]:
    steps = max(2, int(steps))
    arr = np.linspace(min_scale, max_scale, steps)
    return [round(float(s), 3) for s in arr]

def resolve_template_assets(template_dir: Path, template_image: Optional[Path], template_masks: Optional[Path]) -> Tuple[Path, Path]:
    """Return (template_image_path, masks_dir)."""
    tdir = template_dir.resolve()
    if template_image is None:
        # Prefer <basename>.png, else first *.png in root
        preferred = tdir / f"{tdir.name}.png"
        if preferred.is_file():
            template_image = preferred
        else:
            pngs = [p for p in tdir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
            if len(pngs) == 1:
                template_image = pngs[0]
            else:
                raise FileNotFoundError(f"Cannot uniquely determine template PNG in {tdir}. "
                                        f"Use --template-image to specify explicitly.")
    if template_masks is None:
        candidate = tdir / "masks"
        if candidate.is_dir():
            template_masks = candidate
        else:
            raise FileNotFoundError(f"Template masks directory not found at {candidate}. "
                                    f"Use --template-masks to specify explicitly.")
    return template_image.resolve(), template_masks.resolve()

def crop_and_write_masks(template_masks_dir: Path, out_masks_dir: Path,
                         box: Tuple[int, int, int, int], normalize_size: int) -> int:
    """Crop every *.png in template_masks_dir to bbox (inclusive) and write to out_masks_dir."""
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = box
    written = 0
    for mfile in sorted(template_masks_dir.glob("*.png")):
        m = cv2.imread(str(mfile), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        H, W = m.shape[:2]
        cx1 = max(0, min(x1, W - 1))
        cy1 = max(0, min(y1, H - 1))
        cx2 = max(0, min(x2, W - 1))
        cy2 = max(0, min(y2, H - 1))
        if cx2 < cx1 or cy2 < cy1:
            crop = np.zeros((1, 1), dtype=np.uint8)
        else:
            crop = m[cy1:cy2 + 1, cx1:cx2 + 1]
        # binarize (keep crisp masks)
        crop = (crop > 127).astype(np.uint8) * 255
        if normalize_size and normalize_size > 0:
            crop = letterbox(crop, normalize_size)
        cv2.imwrite(str(out_masks_dir / mfile.name), crop)
        written += 1
    return written

def update_meta(meta_path: Path, updates: Dict) -> None:
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    # Shallow merge for top-level; replace 'match' if present
    meta.update(updates)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def process_sample(sample_dir: Path,
                   template_png: Path,
                   template_masks_dir: Path,
                   scales: List[float],
                   score_threshold: float,
                   normalize_size: int,
                   save_matched_vis: bool,
                   force: bool) -> Tuple[bool, str]:
    """Return (ok, message)."""
    img_path = sample_dir / "repair_image.jpg"
    if not img_path.is_file():
        return False, "repair_image.jpg not found"

    out_masks_dir = sample_dir / "masks"
    if out_masks_dir.exists() and not force:
        # if folder exists and has any pngs, skip unless --force
        has_png = any(p.suffix.lower() == ".png" for p in out_masks_dir.iterdir())
        if has_png:
            return False, "masks/ exists (use --force to overwrite)"

    full_bgr = cv2.imread(str(template_png), cv2.IMREAD_COLOR)
    samp_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if full_bgr is None:
        return False, f"cannot read template image: {template_png}"
    if samp_bgr is None:
        return False, f"cannot read sample image: {img_path}"

    match = multi_scale_match(full_bgr, samp_bgr, scales=scales)
    if match["score"] < score_threshold or match["scale"] is None:
        return False, f"low score {match['score']:.3f} < {score_threshold}"

    # --- Crop template RGB with clamped bbox ---
    H, W = full_bgr.shape[:2]
    sh, sw = samp_bgr.shape[:2]
    x1 = max(0, min(match["x1"], W - 1))
    y1 = max(0, min(match["y1"], H - 1))
    x2 = max(0, min(match["x2"], W - 1))
    y2 = max(0, min(match["y2"], H - 1))
    if x2 < x1 or y2 < y1:
        return False, "invalid bbox after clamping"

    crop_bgr = full_bgr[y1:y2 + 1, x1:x2 + 1]
    ch, cw = crop_bgr.shape[:2]

    # Ensure cropped size matches the search image size; resize if necessary
    resized = False
    if (ch, cw) != (sh, sw):
        interp = cv2.INTER_AREA if (ch > sh or cw > sw) else cv2.INTER_LINEAR
        crop_bgr = cv2.resize(crop_bgr, (sw, sh), interpolation=interp)
        ch, cw = crop_bgr.shape[:2]
        resized = True

    # Save cropped RGB as original_image.jpg
    cv2.imwrite(str(sample_dir / "original_image.jpg"), crop_bgr)

    # Write masks (use the original matched bbox to align with crop)
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    written = crop_and_write_masks(
        template_masks_dir, out_masks_dir,
        (match["x1"], match["y1"], match["x2"], match["y2"]),
        normalize_size=normalize_size
    )

    # Optional vis
    if save_matched_vis:
        draw_match(full_bgr, match, str(sample_dir / "matched.png"))

    # Update meta
    updates = {
        "template_image": str(template_png.resolve()),
        "template_masks_dir": str(template_masks_dir.resolve()),
        "normalize_size": int(normalize_size),
        "match": {
            "method": match["method"],
            "score": float(match["score"]),
            "scale": float(match["scale"]),
            "bbox": {
                "x1": int(match["x1"]), "y1": int(match["y1"]),
                "x2": int(match["x2"]), "y2": int(match["y2"]),
                "w": int(match["w"]), "h": int(match["h"]),
            },
        },
        "cropped_original_image": {
            "path": "original_image.jpg",
            "crop_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_size_before_resize": [int(ch if not resized else (y2 - y1 + 1)),
                                        int(cw if not resized else (x2 - x1 + 1))],
            "search_size": [int(sh), int(sw)],
            "resized_to_search": bool(resized),
        },
    }
    if save_matched_vis:
        updates["matched_vis"] = "matched.png"
    update_meta(sample_dir / "meta.json", updates)

    msg = f"masks={written}, score={match['score']:.3f}, scale={match['scale']:.3f}, saved original_image.jpg"
    if resized:
        msg += " (resized to match repair_image.jpg size)"
    return True, msg


# -------------------------------
# Main
# -------------------------------
def find_samples(root: Path) -> List[Path]:
    """Find dirs containing repair_image.jpg (recursively)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "repair_image.jpg" in filenames:
            out.append(Path(dirpath))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser(description="Crop per-sample masks by matching coding samples to a single template.")
    ap.add_argument("--template", required=True, type=str, help="Path to template directory containing <Class>.png and masks/")
    ap.add_argument("--template-image", default=None, type=str, help="(Optional) Explicit path to template PNG if multiple exist.")
    ap.add_argument("--template-masks", default=None, type=str, help="(Optional) Explicit path to template masks dir if not at ./masks")
    ap.add_argument("--normalize-size", type=int, default=0, help="Letterbox cropped masks to NxN (0 = no letterbox).")
    ap.add_argument("--min-scale", type=float, default=0.9, help="Min scale for multi-scale match.")
    ap.add_argument("--max-scale", type=float, default=1.1, help="Max scale for multi-scale match.")
    ap.add_argument("--scale-steps", type=int, default=11, help="Number of scales inclusive.")
    ap.add_argument("--score-threshold", type=float, default=0.7, help="Reject matches below this score.")
    ap.add_argument("--save-matched-vis", action="store_true", help="Save matched.png into each sample.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing masks/ if present.")
    args = ap.parse_args()

    root = Path.cwd()
    template_dir = Path(args.template)
    tmpl_img_arg = Path(args.template_image).expanduser().resolve() if args.template_image else None
    tmpl_masks_arg = Path(args.template_masks).expanduser().resolve() if args.template_masks else None

    template_png, template_masks_dir = resolve_template_assets(template_dir, tmpl_img_arg, tmpl_masks_arg)
    scales = build_scales(args.min_scale, args.max_scale, args.scale_steps)

    print(f"{LBL_INFO} Samples root : {root}")
    print(f"{LBL_INFO} Template PNG : {template_png}")
    print(f"{LBL_INFO} Template masks: {template_masks_dir}")
    print(f"{LBL_INFO} Scales       : {scales[0]} .. {scales[-1]} ({len(scales)} steps)")
    print(f"{LBL_INFO} Threshold    : {args.score_threshold}")
    print(f"{LBL_INFO} Normalize NxN: {args.normalize_size if args.normalize_size > 0 else 'no'}")

    samples = find_samples(root)
    if not samples:
        print(f"{LBL_WARN} No samples with repair_image.jpg found under current directory.")
        return

    ok_cnt = 0
    for sdir in samples:
        ok, msg = process_sample(
            sdir, template_png, template_masks_dir, scales,
            score_threshold=args.score_threshold,
            normalize_size=args.normalize_size,
            save_matched_vis=args.save_matched_vis,
            force=args.force,
        )
        status = LBL_OK if ok else LBL_SKIP
        print(f"{status} {sdir.name}: {msg}")
        if ok:
            ok_cnt += 1

    print(f"{LBL_DONE} Processed {len(samples)} sample(s). Successful: {ok_cnt}, Skipped/Failed: {len(samples) - ok_cnt}")

if __name__ == "__main__":
    main()
