"""
Create per-sample combined color mask images for VQA training.

For each sample folder (both augmentation and defect):
  <class>/aug/<sample>/masks/*.png
  defect/<class>_defect/<sample>/masks/*.png
Combines a selected subset of component masks into one RGB image (color_mask.png)
with layered priority (top -> bottom):
  Source, Drain, TFT, VIA_Hole, Mesh, Mesh_Hole, Com, Data, Gate, ITO

Notes:
- Only masks whose (normalized) filename matches one of those component names
  are considered; others are ignored.
- Randomly assigns distinct colors (from provided palette) to the PRESENT
  components per sample (no reuse within a sample).
- Overlays bottom components first (ITO bottom) then up to Source so top
  components visually overwrite lower ones where overlapping.
- Appends color assignment info into the sample's meta.json under key "color_mask".
- Skips samples already having color_mask.png unless --force.

Run (from with_label root):
  python src/color_mask.py
Optional args:
  --root PATH            (default .)
  --force                overwrite existing color_mask.png & meta mapping
  --seed SEED            reproducibility
  --dry-run              do not write files, just report
  --verbose

Dependencies: Pillow, numpy
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

# Ordered top->bottom (higher index = deeper layer)
COMPONENT_ORDER_TOP_TO_BOTTOM = [
    "Source", "Drain", "TFT", "VIA_Hole", "Mesh", "Mesh_Hole",
    "Com", "Data", "Gate", "ITO"
]

# Color palette (RGB) with Chinese labels
COLOR_PALETTE: List[Tuple[Tuple[int,int,int], str]] = [
    ((31,119,180), "蓝色"),
    ((255,127,14), "橙色"),
    ((44,160,44), "绿色"),
    ((214,39,40), "红色"),
    ((148,103,189), "紫色"),
    ((140,86,75), "棕色"),
    ((227,119,194), "粉色"),
    ((127,127,127), "灰色"),
    ((188,189,34), "橄榄色"),
    ((23,190,207), "青色"),
]

TARGET_SET_CANON = {c.lower(): c for c in COMPONENT_ORDER_TOP_TO_BOTTOM}

def norm_component_name(name: str) -> str:
    n = name.strip().replace("-", "_").replace(" ", "_")
    return n.lower()

def discover_sample_dirs(root: Path) -> List[Path]:
    sample_dirs = []
    # Class augmentation samples
    for class_dir in root.iterdir():
        if not class_dir.is_dir() or class_dir.name.startswith(".") or class_dir.name == "defect":
            continue
        aug_dir = class_dir / "aug"
        if aug_dir.is_dir():
            for s in aug_dir.iterdir():
                if s.is_dir():
                    sample_dirs.append(s)
    # Defect samples
    defect_root = root / "defect"
    if defect_root.is_dir():
        for class_def in defect_root.iterdir():
            if not class_def.is_dir():
                continue
            for s in class_def.iterdir():
                if s.is_dir():
                    sample_dirs.append(s)
    return sample_dirs

def load_mask(path: Path) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("L")
        return np.array(img)
    except Exception:
        return None

def get_canvas_size(sample_dir: Path, meta: dict) -> Tuple[int,int]:
    # Prefer meta new_width/new_height
    if meta:
        for k in ("new_width", "final_width", "width"):
            if k in meta:
                w = meta.get(k)
                h = meta.get("new_height") or meta.get("final_height") or meta.get("height")
                if isinstance(w,int) and isinstance(h,int):
                    return w,h
        # Try transforms.final_crop_resize.*
        try:
            fmap = meta["transforms"]["final_crop_resize"]
            w = fmap.get("final_width")
            h = fmap.get("final_height")
            if isinstance(w,int) and isinstance(h,int):
                return w,h
        except Exception:
            pass
    # Fallback: find any mask or augmented image
    for cand in ["aug_image.png", "aug_image.jpg"]:
        p = sample_dir / cand
        if p.is_file():
            with Image.open(p) as im:
                return im.size
    # Search first mask
    mask_dir = sample_dir / "masks"
    if mask_dir.is_dir():
        for f in mask_dir.iterdir():
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                with Image.open(f) as im:
                    return im.size
    # Default
    return 512,512

def pick_color_assignments(present_components: List[str], rng: random.Random) -> Dict[str, Dict]:
    choices = COLOR_PALETTE.copy()
    rng.shuffle(choices)
    assignments = {}
    for comp, (rgb, zh) in zip(present_components, choices):
        assignments[comp] = {
            "rgb": list(rgb),
            "hex": "#{:02X}{:02X}{:02X}".format(*rgb),
            "color_name_zh": zh
        }
    return assignments

def build_color_mask(sample_dir: Path, force: bool, rng: random.Random, dry: bool, verbose: bool) -> bool:
    mask_dir = sample_dir / "masks"
    if not mask_dir.is_dir():
        if verbose:
            print(f"[SKIP] {sample_dir}: no masks dir")
        return False

    meta_path = sample_dir / "meta.json"
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            if verbose:
                print(f"[WARN] {sample_dir}: meta.json parse failed")

    out_path = sample_dir / "color_mask.png"
    if out_path.exists() and not force:
        if verbose:
            print(f"[SKIP] {sample_dir}: color_mask.png exists (use --force to overwrite)")
        return False

    # Collect candidate masks
    component_masks = {}
    for f in mask_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        base = f.stem
        canon = norm_component_name(base)
        if canon in TARGET_SET_CANON:
            comp_name = TARGET_SET_CANON[canon]
            component_masks[comp_name] = f

    if not component_masks:
        if verbose:
            print(f"[SKIP] {sample_dir}: no target component masks found")
        return False

    # Order from bottom to top: reverse of given top->bottom
    ordered_components = [c for c in COMPONENT_ORDER_TOP_TO_BOTTOM if c in component_masks]
    if not ordered_components:
        if verbose:
            print(f"[SKIP] {sample_dir}: no ordered components after filtering")
        return False

    width, height = get_canvas_size(sample_dir, meta)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign colors randomly (unique per sample)
    assignments = pick_color_assignments(ordered_components, rng)

    # Paint: bottom first (reverse of top list)
    for comp in reversed(ordered_components):  # bottom to top
        mask_path = component_masks[comp]
        mask_arr = load_mask(mask_path)
        if mask_arr is None:
            if verbose:
                print(f"[WARN] {sample_dir}: failed to load {mask_path.name}")
            continue
        # Resize if dimension mismatch
        if mask_arr.shape[1] != width or mask_arr.shape[0] != height:
            mask_arr = np.array(Image.fromarray(mask_arr).resize((width, height), resample=Image.NEAREST))
        bin_mask = mask_arr > 128  # treat >128 as foreground
        rgb = assignments[comp]["rgb"]
        # Overlay: overwrite underlying colors (since this comp is above)
        canvas[bin_mask] = rgb

    if not dry:
        Image.fromarray(canvas).save(out_path)
        # Update meta.json
        meta.setdefault("color_mask", {})
        meta["color_mask"] = {
            "generated": True,
            "file": "color_mask.png",
            "components_order_top_to_bottom": ordered_components,
            "assigned_colors": assignments
        }
        try:
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[ERR] {sample_dir}: failed writing meta.json update ({e})")
            return False
    if verbose:
        print(f"[OK] {sample_dir}: created color_mask.png ({len(ordered_components)} comps)")
    return True

def main():
    ap = argparse.ArgumentParser(description="Generate combined color mask images.")
    ap.add_argument("--root", default=".", help="with_label root directory.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing color_mask.png.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for color assignment.")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files; just log.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    root = Path(args.root).resolve()
    sample_dirs = discover_sample_dirs(root)
    if args.verbose:
        print(f"[INFO] Found {len(sample_dirs)} sample dirs.")

    made = 0
    for sdir in sample_dirs:
        if build_color_mask(
            sample_dir=sdir,
            force=args.force,
            rng=rng,
            dry=args.dry_run,
            verbose=args.verbose
        ):
            made += 1

    print(f"[DONE] Generated {made} color mask(s). {'(dry-run)' if args.dry_run else ''}")

if __name__ == "__main__":
    main()