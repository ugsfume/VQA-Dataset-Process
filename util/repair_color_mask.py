#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_color_mask.py

For each sample in the reorganized repair dataset:
  <root>/{negative,positive}/<category>/<category>_n/

- Combine component masks from "masks/" into one RGB image (color_mask.png)
  using COMPONENT_ORDER_TOP_TO_BOTTOM (top->bottom) and a fixed palette,
  randomly assigned per sample without reuse.
- Overlay contours from "output.json" on top of the color mask with fixed
  colors and a defined layer order:
      top:   defect
             laser_cut          (rps_points:10)
             ITO_removal        (rps_points:11)
             revC_ITO_removal   (rps_points:110)
      bottom: C_ITO_removal     (rps_points:112)
- Save/merge color assignments & contour style into "meta.json".

Contours are rounded (fillet-like corners) for RPS overlays, but NOT for
'defect' contours (which use sharp joins/caps).

Now also saves Chinese color names for contour styles in meta.json
(similar to component color assignments).

Dependencies: Pillow, numpy, colorama
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import random
import numpy as np
from PIL import Image, ImageDraw

# --- Colorized CLI labels (colorama) ---
from colorama import init as colorama_init, Fore, Style, Back
colorama_init(autoreset=True)
LBL_WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
LBL_SKIP = Fore.MAGENTA + "[SKIP]" + Style.RESET_ALL
LBL_DRY  = Fore.BLUE   + "[DRY]"  + Style.RESET_ALL
LBL_OK   = Fore.GREEN  + "[OK]"   + Style.RESET_ALL
LBL_INFO = Back.BLUE   + "[INFO]" + Style.RESET_ALL
LBL_DONE = Back.YELLOW + "[DONE]" + Style.RESET_ALL

# ---------------------- Config: components & palette ----------------------

# Ordered top->bottom (higher = deeper)
COMPONENT_ORDER_TOP_TO_BOTTOM = [
    "Source", "Drain", "TFT", "VIA_Hole", "Mesh", "Mesh_Hole",
    "Com", "Data", "Gate", "ITO"
]

# Color palette (RGB) with Chinese labels (same scheme as before)
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

# Canonical name normalizer (filenames like "Mesh_Hole.jpg", "A-Com.jpg" etc.)
TARGET_SET_CANON = {c.lower(): c for c in COMPONENT_ORDER_TOP_TO_BOTTOM}

# ---------------------- Contour mapping & defaults ------------------------

# rps label -> aka name
RPS_MAP = {
    "rps_points:10":  "laser_cut",
    "rps_points:11":  "ITO_removal",
    "rps_points:110": "revC_ITO_removal",
    "rps_points:112": "C_ITO_removal",
}

# Default colors for contours (RGB)
DEFAULT_CONTOUR_COLORS = {
    "defect":            (0, 0, 139),      # 深蓝色
    "laser_cut":         (128, 0, 0),      # 栗色     (rps:10)
    "ITO_removal":       (250, 250, 51),   # 柠檬黄色 (rps:11)
    "revC_ITO_removal":  (250, 250, 51),   # 柠檬黄色 (rps:110)
    "C_ITO_removal":     (250, 250, 51),   # 柠檬黄色 (rps:112)

    # "revC_ITO_removal":  (255, 127, 80),   # 珊瑚色   (rps:110)
    # "C_ITO_removal":     (0, 255, 0),      # 酸橙色   (rps:112)
}

# Chinese names for the above defaults
DEFAULT_CONTOUR_COLOR_NAMES_ZH = {
    "defect":            "深蓝色",
    "laser_cut":         "栗色",
    "ITO_removal":       "柠檬黄色",
    "revC_ITO_removal":  "柠檬黄色",
    "C_ITO_removal":     "柠檬黄色",
}

# Draw order bottom->top (we render in this order)
CONTOUR_LAYER_ORDER_BOTTOM_TO_TOP = [
    "C_ITO_removal",
    "revC_ITO_removal",
    "ITO_removal",
    "defect",
    "laser_cut",
]

# ---------------------- Helpers ----------------------

def norm_component_name(name: str) -> str:
    n = name.strip().replace("-", "_").replace(" ", "_")
    return n.lower()

def parse_color(s: Optional[str], default_rgb: Tuple[int,int,int]) -> Tuple[int,int,int]:
    if not s:
        return default_rgb
    s = s.strip()
    if s.startswith("#"):
        s = s.lstrip("#")
        if len(s) == 6:
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
            return (r,g,b)
        raise ValueError(f"Invalid hex color: #{s}")
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid RGB tuple: {s}")
        vals = []
        for p in parts:
            if "." in p:
                v = float(p); v = max(0.0, min(1.0, v)); vals.append(int(round(v*255)))
            else:
                vals.append(max(0, min(255, int(p))))
        return tuple(vals)  # type: ignore
    # named colors not supported here to avoid ambiguity; accept only hex or r,g,b
    raise ValueError(f"Unsupported color format: {s}")

def rgb_to_hex(rgb: Tuple[int,int,int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def contour_color_name_zh(aka: str, rgb: Tuple[int,int,int]) -> str:
    """
    If the rgb matches our default for this aka exactly, return the known Chinese name.
    Otherwise, mark as custom color.
    """
    if aka in DEFAULT_CONTOUR_COLORS and tuple(rgb) == DEFAULT_CONTOUR_COLORS[aka]:
        return DEFAULT_CONTOUR_COLOR_NAMES_ZH.get(aka, "自定义色")
    return "自定义色"

def discover_samples(root: Path) -> List[Path]:
    out = []
    for polarity in ("negative", "positive"):
        pdir = root / polarity
        if not pdir.is_dir(): continue
        for cat in sorted(p for p in pdir.iterdir() if p.is_dir()):
            for sample in sorted(s for s in cat.iterdir() if s.is_dir()):
                out.append(sample)
    return out

def load_mask(path: Path) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("L")  # grayscale
        return np.array(img)
    except Exception:
        return None

def get_canvas_size(sample_dir: Path) -> Tuple[int,int]:
    """Prefer repair_image/original_image, else any mask, else 512x512."""
    for name in ("repair_image.jpg", "repair_image.png", "original_image.jpg", "original_image.png"):
        p = sample_dir / name
        if p.is_file():
            with Image.open(p) as im:
                return im.size
    masks_dir = sample_dir / "masks"
    if masks_dir.is_dir():
        for f in masks_dir.iterdir():
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
                with Image.open(f) as im:
                    return im.size
    return (512, 512)

def pick_color_assignments(present_components: List[str], rng: random.Random) -> Dict[str, Dict[str, Any]]:
    choices = COLOR_PALETTE.copy()
    rng.shuffle(choices)
    assignments: Dict[str, Dict[str, Any]] = {}
    for comp, (rgb, zh) in zip(present_components, choices):
        assignments[comp] = {
            "rgb": list(rgb),
            "hex": rgb_to_hex(rgb),
            "color_name_zh": zh
        }
    return assignments

def ensure_meta(sample_dir: Path) -> Dict[str, Any]:
    meta_path = sample_dir / "meta.json"
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_meta(sample_dir: Path, meta: Dict[str, Any]) -> None:
    (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def clamp_point(x: float, y: float, w: int, h: int) -> Tuple[int,int]:
    xi = max(0, min(int(round(x)), w - 1))
    yi = max(0, min(int(round(y)), h - 1))
    return xi, yi

# ---------------------- Rounded & sharp drawing primitives ----------------------

def _draw_filled_circle(draw: ImageDraw.ImageDraw, center: Tuple[int,int],
                        radius: int, color: Tuple[int,int,int]) -> None:
    if radius <= 0:
        return
    cx, cy = center
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(bbox, fill=color, outline=color)

def draw_polyline_round(draw: ImageDraw.ImageDraw, pts: List[Tuple[int,int]],
                        color: Tuple[int,int,int], width: int, closed: bool=False) -> None:
    """
    Draw a polyline (or polygon if closed=True) with rounded caps and rounded joins.
    Achieved by drawing the base line(s) and stamping filled circles at each vertex.
    """
    if len(pts) < 2:
        if len(pts) == 1:
            _draw_filled_circle(draw, pts[0], max(1, width // 2), color)
        return

    # Draw the connected line(s)
    if closed:
        draw.line(pts + [pts[0]], fill=color, width=width)
    else:
        draw.line(pts, fill=color, width=width)

    # Round caps & joins by drawing circles at vertices
    r = max(1, width // 2)
    for p in pts:
        _draw_filled_circle(draw, p, r, color)

def draw_polyline_sharp(draw: ImageDraw.ImageDraw, pts: List[Tuple[int,int]],
                        color: Tuple[int,int,int], width: int, closed: bool=False) -> None:
    """
    Draw a polyline (or closed polygonal line) with default Pillow joins/caps (non-rounded).
    No extra circles at vertices, so corners remain sharp.
    """
    if len(pts) < 2:
        if len(pts) == 1:
            draw.point(pts[0], fill=color)
        return
    if closed:
        draw.line(pts + [pts[0]], fill=color, width=width)
    else:
        draw.line(pts, fill=color, width=width)

def draw_c_open_rectangle_round(draw: ImageDraw.ImageDraw, tl: Tuple[int,int], br: Tuple[int,int],
                                open_side: str, color: Tuple[int,int,int], width: int) -> None:
    """
    Draw a rectangle with one open side, using rounded caps/joins on drawn sides.
    open_side in {"right","left"} as per existing semantics.
    """
    x1, y1 = tl; x2, y2 = br
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    top    = [(x1, y1), (x2, y1)]
    right  = [(x2, y1), (x2, y2)]
    bottom = [(x1, y2), (x2, y2)]
    left   = [(x1, y1), (x1, y2)]

    if open_side == "right":
        segs = [top, left, bottom]
    elif open_side == "left":
        segs = [top, right, bottom]
    else:
        segs = [top, right, bottom, left]

    for seg in segs:
        draw_polyline_round(draw, seg, color=color, width=width, closed=False)

# ---------------------- Drawing ----------------------

def draw_component_masks(sample_dir: Path, canvas: np.ndarray, rng: random.Random, verbose: bool=False
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    """Paint bottom->top; return updated canvas, assignment meta, present components."""
    masks_dir = sample_dir / "masks"
    if not masks_dir.is_dir():
        if verbose: print(f"{LBL_SKIP} {sample_dir}: no masks/")
        return canvas, {}, []

    # Discover candidate masks by canonical name
    component_files: Dict[str, Path] = {}
    for f in masks_dir.iterdir():
        if not f.is_file(): continue
        if f.suffix.lower() not in (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"): continue
        canon = norm_component_name(f.stem)
        if canon in TARGET_SET_CANON:
            component = TARGET_SET_CANON[canon]
            component_files[component] = f

    present = [c for c in COMPONENT_ORDER_TOP_TO_BOTTOM if c in component_files]
    if not present:
        if verbose: print(f"{LBL_WARN} {sample_dir}: no target component masks found")
        return canvas, {}, []

    # Assign colors per sample
    assignments = pick_color_assignments(present, rng)

    # Paint bottom->top (reverse of top->bottom list)
    h, w = canvas.shape[0], canvas.shape[1]
    for comp in reversed(present):
        mpath = component_files[comp]
        mask = load_mask(mpath)
        if mask is None:
            if verbose: print(f"{LBL_WARN} {sample_dir}: failed to load {mpath.name}")
            continue
        # Resize if needed
        if mask.shape[1] != w or mask.shape[0] != h:
            mask = np.array(Image.fromarray(mask).resize((w, h), resample=Image.NEAREST))
        # Thresholding: include pixels with grayscale >= 128
        bin_mask = mask >= 128
        rgb = assignments[comp]["rgb"]
        canvas[bin_mask] = rgb

    # Pack meta
    color_mask_meta = {
        "generated": True,
        "file": "color_mask.png",
        "components_order_top_to_bottom": present,
        "assigned_colors": assignments
    }
    return canvas, color_mask_meta, present

# (Kept for API compatibility, used for rounded segments)
def draw_line(draw: ImageDraw.ImageDraw, p1: Tuple[int,int], p2: Tuple[int,int],
              color: Tuple[int,int,int], width: int):
    draw_polyline_round(draw, [p1, p2], color=color, width=width, closed=False)

def draw_c_open_rectangle(draw: ImageDraw.ImageDraw, tl: Tuple[int,int], br: Tuple[int,int],
                          open_side: str, color: Tuple[int,int,int], width: int):
    draw_c_open_rectangle_round(draw, tl, br, open_side=open_side, color=color, width=width)

def draw_contours(sample_dir: Path, img_pil: Image.Image,
                  widths: Dict[str,int], colors: Dict[str,Tuple[int,int,int]],
                  defect_closed: bool = True, verbose: bool=False
) -> Dict[str,int]:
    """Draw overlays on top of img_pil. Returns counts per aka."""
    out_counts = {k: 0 for k in ["defect","laser_cut","ITO_removal","revC_ITO_removal","C_ITO_removal"]}
    ojson = sample_dir / "output.json"
    if not ojson.is_file():
        if verbose: print(f"{LBL_WARN} {sample_dir}: missing output.json; skipping contours")
        return out_counts

    try:
        data = json.loads(ojson.read_text(encoding="utf-8"))
        contours = data.get("contours", [])
    except Exception as e:
        if verbose: print(f"{LBL_WARN} {sample_dir}: failed to parse output.json: {e}")
        return out_counts

    w, h = img_pil.size
    draw = ImageDraw.Draw(img_pil)

    # Group items by aka and draw in bottom->top order
    grouped: Dict[str, List[Dict[str,Any]]] = {k: [] for k in out_counts.keys()}
    others: List[Dict[str,Any]] = []  # defect labels (TSCOK, TSPTC, ...)
    for item in contours:
        label = str(item.get("label",""))
        if label in RPS_MAP:
            aka = RPS_MAP[label]
            grouped[aka].append(item)
        elif label.startswith("rps_points:") and (label not in RPS_MAP):
            # Unknown rps type: ignore but warn
            if verbose: print(f"{LBL_WARN} {sample_dir}: unrecognized rps label '{label}'")
        else:
            others.append(item)  # defect contour

    # Draw in specified order
    for aka in CONTOUR_LAYER_ORDER_BOTTOM_TO_TOP:
        if aka == "defect":
            items = others
        else:
            items = grouped.get(aka, [])
        color = colors[aka]
        width = widths[aka]
        for it in items:
            pts = it.get("points", [])
            # Convert to clamped ints
            ipts = [clamp_point(x, y, w, h) for (x,y) in pts]
            if aka == "laser_cut" or aka == "ITO_removal":
                if len(ipts) >= 2:
                    # Rounded-ended segment
                    draw_polyline_round(draw, [ipts[0], ipts[1]], color=color, width=width, closed=False)
                    out_counts[aka] += 1
            elif aka == "C_ITO_removal" or aka == "revC_ITO_removal":
                if len(ipts) >= 2:
                    tl = (min(ipts[0][0], ipts[1][0]), min(ipts[0][1], ipts[1][1]))
                    br = (max(ipts[0][0], ipts[1][0]), max(ipts[0][1], ipts[1][1]))
                    open_side = "right" if aka == "C_ITO_removal" else "left"
                    draw_c_open_rectangle_round(draw, tl, br, open_side=open_side, color=color, width=width)
                    out_counts[aka] += 1
            elif aka == "defect":
                # Polyline with SHARP corners/caps (no rounding)
                if len(ipts) >= 2:
                    draw_polyline_sharp(
                        draw,
                        ipts,
                        color=color,
                        width=width,
                        closed=(defect_closed and len(ipts) >= 3)
                    )
                    out_counts["defect"] += 1

    return out_counts

# ---------------------- Main per-sample processing ------------------------

def process_sample(sample_dir: Path, rng: random.Random, force: bool,
                   widths: Dict[str,int], colors: Dict[str,Tuple[int,int,int]],
                   defect_closed: bool, dry_run: bool, verbose: bool
) -> bool:
    width, height = get_canvas_size(sample_dir)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    out_path = sample_dir / "color_mask.png"
    if out_path.exists() and not force:
        if verbose: print(f"{LBL_SKIP} {sample_dir}: color_mask.png exists (use --force)")
        return False

    # 1) Paint components
    canvas, color_mask_meta, present_components = draw_component_masks(sample_dir, canvas, rng, verbose=verbose)

    # 2) Overlay contours
    img = Image.fromarray(canvas, mode="RGB")
    counts = draw_contours(sample_dir, img, widths=widths, colors=colors,
                           defect_closed=defect_closed, verbose=verbose)

    if dry_run:
        if verbose:
            print(f"{LBL_DRY} Would write: {out_path}")
        return True

    # Save final image
    img.save(out_path)

    # Update meta.json
    meta = ensure_meta(sample_dir)
    meta.setdefault("color_mask", {})
    if color_mask_meta:
        meta["color_mask"] = color_mask_meta

    meta.setdefault("contours", {})
    meta["contours"]["layer_order_top_to_bottom"] = list(reversed(CONTOUR_LAYER_ORDER_BOTTOM_TO_TOP))

    # Build enriched styles (include Chinese name & hex)
    styles: Dict[str, Dict[str, Any]] = {}
    for aka in ["defect", "laser_cut", "ITO_removal", "revC_ITO_removal", "C_ITO_removal"]:
        rgb = tuple(colors[aka])
        styles[aka] = {
            "color_rgb": list(rgb),
            "hex": rgb_to_hex(rgb),
            "color_name_zh": contour_color_name_zh(aka, rgb),
            "width": widths[aka],
        }
    meta["contours"]["styles"] = styles

    meta["contours"]["counts"] = counts
    meta["contours"]["defect_closed"] = defect_closed

    save_meta(sample_dir, meta)
    if verbose: print(f"{LBL_OK} {sample_dir}: color_mask.png + meta.json updated")
    return True

# ---------------------- CLI ------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate color mask with contour overlays for repair dataset.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Dataset root (gt_datasets_20250915).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing color_mask.png section.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for component color assignment.")
    ap.add_argument("--defect-open", action="store_true",
                    help="Draw defect polylines open (do NOT close last->first). Default: closed.")
    ap.add_argument("--dry-run", action="store_true", help="Log actions without writing files.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")

    # Contour widths
    ap.add_argument("--width-defect", type=int, default=4)
    ap.add_argument("--width-laser-cut", type=int, default=7)          # rps:10
    ap.add_argument("--width-ito-removal", type=int, default=20)       # rps:11
    ap.add_argument("--width-revc-ito-removal", type=int, default=20)  # rps:110
    ap.add_argument("--width-c-ito-removal", type=int, default=20)     # rps:112

    # Contour color overrides
    ap.add_argument("--color-defect", type=str, default=None)            # e.g., "0,0,139" or "#00008B"
    ap.add_argument("--color-laser-cut", type=str, default=None)         # e.g., "#FF7F50"
    ap.add_argument("--color-ito-removal", type=str, default=None)       # "#FFFF00"
    ap.add_argument("--color-revc-ito-removal", type=str, default=None)  # "#00FF00"
    ap.add_argument("--color-c-ito-removal", type=str, default=None)     # "0,100,0"

    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Build colors & widths maps
    colors = {
        "defect":           parse_color(args.color_defect,           DEFAULT_CONTOUR_COLORS["defect"]),
        "laser_cut":        parse_color(args.color_laser_cut,        DEFAULT_CONTOUR_COLORS["laser_cut"]),
        "ITO_removal":      parse_color(args.color_ito_removal,      DEFAULT_CONTOUR_COLORS["ITO_removal"]),
        "revC_ITO_removal": parse_color(args.color_revc_ito_removal, DEFAULT_CONTOUR_COLORS["revC_ITO_removal"]),
        "C_ITO_removal":    parse_color(args.color_c_ito_removal,    DEFAULT_CONTOUR_COLORS["C_ITO_removal"]),
    }
    widths = {
        "defect":           max(1, args.width_defect),
        "laser_cut":        max(1, args.width_laser_cut),
        "ITO_removal":      max(1, args.width_ito_removal),
        "revC_ITO_removal": max(1, args.width_revc_ito_removal),
        "C_ITO_removal":    max(1, args.width_c_ito_removal),
    }

    root = args.root.resolve()
    samples = discover_samples(root)
    if args.verbose:
        print(f"{LBL_INFO} Found {len(samples)} sample dirs under {root}")

    processed = 0
    for s in samples:
        ok = process_sample(
            sample_dir=s,
            rng=rng,
            force=args.force,
            widths=widths,
            colors=colors,
            defect_closed=not args.defect_open,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if ok: processed += 1

    print(f"{LBL_DONE} Processed {processed} sample(s).{' (dry-run)' if args.dry_run else ''}")

if __name__ == "__main__":
    main()
