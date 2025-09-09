"""
Generate per-sample dual-image VQA (qa_mask.jsonl) using:
  - aug_image.png   (RGB original)
  - color_mask.png  (colored component mask)
Derive ground-truth answers from binary masks + meta.json color assignments.

Writes qa_mask.jsonl inside each sample folder.

Usage:
  python mask_qa_gt_generator.py -r /path/to/with_label --min-area 100
Optional filters:
  --only SAMPLE_NAME ...
  --aug-only / --defect-only
"""

import os
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
import cv2
import random
from colorama import init, Fore, Style

init(autoreset=True)
OK = Fore.GREEN + "[OK]" + Style.RESET_ALL
ERR = Fore.RED + "[ERR]" + Style.RESET_ALL
TOTAL = Fore.CYAN + "[TOTAL]" + Style.RESET_ALL

# Palette Chinese color names (for reverse color negatives)
PALETTE_COLOR_NAMES_ZH = ["蓝色","橙色","绿色","红色","紫色","棕色","粉色","灰色","橄榄色","青色"]

# Components allowed for COLOR QA (must be present + assigned color)
COLOR_COMPONENTS = [
    "Source", "Drain", "TFT", "VIA_Hole", "Mesh", "Mesh_Hole",
    "Com", "Data", "Gate", "ITO"
]

# Exclusions
COUNT_EXCLUDE = {"pxl_ito", "com_hole", "gate", "ito"}
BBOX_EXCLUDE = {"pxl_ito", "com_hole", "gate", "ito"}
COORD_EXCLUDE = {"pxl_ito", "com_hole"}  # excluded from reverse-window targets

def to_posix(p: str) -> str:
    return p.replace("\\", "/")

def _round_to_28_nearest(x: int) -> int:
    down = (x // 28) * 28
    up = ((x + 27) // 28) * 28
    return up if (x - down) > (up - x) else down  # ties -> down

def _scale_bboxes_to_size(bboxes: List[List[int]], w: int, h: int, rw: int, rh: int) -> List[List[int]]:
    if w <= 0 or h <= 0:
        return bboxes
    sx = rw / float(w)
    sy = rh / float(h)
    out: List[List[int]] = []
    for x1, y1, x2, y2 in bboxes:
        nx1 = int(round(x1 * sx))
        ny1 = int(round(y1 * sy))
        nx2 = int(round(x2 * sx))
        ny2 = int(round(y2 * sy))
        nx1 = max(0, min(nx1, rw - 1))
        ny1 = max(0, min(ny1, rh - 1))
        nx2 = max(0, min(nx2, rw - 1))
        ny2 = max(0, min(ny2, rh - 1))
        if nx2 < nx1: nx1, nx2 = nx2, nx1
        if ny2 < ny1: ny1, ny2 = ny2, ny1
        out.append([nx1, ny1, nx2, ny2])
    return out

def _resize_mask_nn(mask: np.ndarray, rw: int, rh: int) -> np.ndarray:
    resized = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8) * 255

def list_aug_folders(root: str) -> List[str]:
    out = []
    for cls in os.scandir(root):
        if not cls.is_dir() or cls.name.startswith('.') or cls.name == 'defect':
            continue
        aug_dir = os.path.join(cls.path, "aug")
        if not os.path.isdir(aug_dir):
            continue
        for s in os.scandir(aug_dir):
            if s.is_dir() and not s.name.startswith('.'):
                out.append(s.path)
    return out

def list_defect_folders(root: str) -> List[str]:
    out = []
    defect_root = os.path.join(root, "defect")
    if not os.path.isdir(defect_root):
        return out
    for cls in os.scandir(defect_root):
        if not cls.is_dir() or cls.name.startswith('.'):
            continue
        for s in os.scandir(cls.path):
            if s.is_dir() and not s.name.startswith('.'):
                out.append(s.path)
    return out

def load_rgb_and_masks(sample_dir: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    img_path = os.path.join(sample_dir, "aug_image.png")
    rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Missing aug_image.png at {sample_dir}")
    masks_dir = os.path.join(sample_dir, "masks")
    masks: Dict[str, np.ndarray] = {}
    if os.path.isdir(masks_dir):
        for fname in sorted(os.listdir(masks_dir)):
            if not fname.lower().endswith(".png"):
                continue
            path = os.path.join(masks_dir, fname)
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = (m > 127).astype(np.uint8) * 255
            masks[fname] = m
    return rgb, masks

def connected_components_bboxes(mask: np.ndarray, min_area: int) -> Tuple[List[List[int]], np.ndarray]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    bin01 = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    keep = []
    bboxes: List[List[int]] = []
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        x1, y1 = x, y
        x2, y2 = x + w - 1, y + h - 1
        bboxes.append([x1, y1, x2, y2])
        keep.append(lbl)
    if not keep:
        filtered = np.zeros_like(mask, dtype=np.uint8)
    else:
        filtered = np.isin(labels, keep).astype(np.uint8) * 255
    return bboxes, filtered

def comp_label_from_mask_filename(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0]

def slug(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_").lower()

def load_meta(sample_dir: str) -> dict:
    mp = os.path.join(sample_dir, "meta.json")
    if not os.path.isfile(mp):
        return {}
    try:
        with open(mp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def images_list(root: str, sample_dir: str, include_mask: bool) -> List[str]:
    rgb = to_posix(os.path.relpath(os.path.join(sample_dir, "aug_image.png"), root))
    if include_mask and os.path.isfile(os.path.join(sample_dir, "color_mask.png")):
        cm = to_posix(os.path.relpath(os.path.join(sample_dir, "color_mask.png"), root))
        return [rgb, cm]
    return [rgb]

def build_defect_record(sample_name: str, imgs: List[str], w: int, h: int, defect_flag: bool) -> Dict:
    return {
        "id": f"{sample_name}_defect_flag",
        "images": imgs,
        "meta": {"aug_size": [w, h], "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)]},
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像（仅用于结构参考，不含缺陷）。问题：原始图像中是否存在缺陷？"
            },
            {"from": "gpt", "value": "是" if defect_flag else "否"}
        ]
    }

def build_count_record(sample_name: str, imgs: List[str], label: str, w: int, h: int, count: int) -> Dict:
    return {
        "id": f"{sample_name}_{slug(label)}_count",
        "images": imgs,
        "meta": {"aug_size": [w, h], "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)], "component": label},
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。针对组件类型："
                         f"{label}。\n问题：该类型在掩膜中可见数量是多少？（请交叉参考两张图以确保一致性）"
            },
            {"from": "gpt", "value": str(count)}
        ]
    }

def build_color_record(sample_name: str, imgs: List[str], label: str, w: int, h: int, color_name: str) -> Dict:
    return {
        "id": f"{sample_name}_{slug(label)}_color",
        "images": imgs,
        "meta": {"aug_size": [w, h], "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)], "component": label, "color_source": "mask"},
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。针对组件类型："
                         f"{label}。\n问题：若该类型存在，请返回其掩膜颜色的中文名称；否则返回 null。"
            },
            {"from": "gpt", "value": color_name}
        ]
    }

def build_bbox_record(sample_name: str, imgs: List[str], label: str, w: int, h: int,
                      bboxes_scaled: List[List[int]], present: bool) -> Dict:
    bbox_list = [{"bbox_2d": bb, "label": label} for bb in bboxes_scaled] if present else None
    bbox_json = json.dumps(bbox_list, ensure_ascii=False, separators=(",", ":")) if bbox_list is not None else "null"
    return {
        "id": f"{sample_name}_{slug(label)}_bboxes",
        "images": imgs,
        "meta": {"aug_size": [w, h], "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)], "component": label},
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。针对组件类型："
                    f"{label}。\n问题：若存在，请返回所有该部件在图像像素坐标系中的边界框；若不存在则返回 null。"
                    "\n注意：请综合参考两张图像进行判断"
                )
            },
            {"from": "gpt", "value": bbox_json}
        ]
    }

# NEW: single-component reverse window builder (uses one bbox as the window)
def build_reverse_window_single_record(
    sample_name: str,
    imgs: List[str],
    w: int,
    h: int,
    window: Tuple[int,int,int,int],
    label: str
) -> Dict:
    x1, y1, x2, y2 = window
    wid = f"{x1}_{y1}_{x2}_{y2}"
    return {
        "id": f"{sample_name}_rev_window_{wid}",
        "images": imgs,
        "meta": {
            "aug_size": [w, h],
            "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)],
            "reverse": "window_to_component_single",
            "window": [x1, y1, x2, y2]
        },
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。"
                    f"指定窗口区域：[x1={x1},y1={y1},x2={x2},y2={y2}]。\n"
                    "问题：该窗口内显示的主要组件类型是什么？"
                    "\n注意：请综合参考两张图像进行判断"
                )
            },
            {"from": "gpt", "value": label if label else "null"}
        ]
    }

def process_sample(root: str, sample_dir: str, min_area: int,
                   rev_color_neg_max: int,
                   rev_coord_per_sample: int,
                   rng: random.Random) -> int:
    rgb, masks = load_rgb_and_masks(sample_dir)
    h, w = rgb.shape[:2]
    rw, rh = _round_to_28_nearest(w), _round_to_28_nearest(h)

    sample_name = os.path.basename(sample_dir)
    meta = load_meta(sample_dir)
    defect_flag = bool(meta.get("defect", False))
    color_assign = meta.get("color_mask", {}).get("assigned_colors", {}) or {}

    has_color_mask = os.path.isfile(os.path.join(sample_dir, "color_mask.png"))
    imgs = images_list(root, sample_dir, include_mask=has_color_mask)

    records: List[Dict] = []

    # Precompute resized masks only if needed elsewhere (keeping for parity)
    component_masks_binary_resized: Dict[str, np.ndarray] = {}

    # Accumulate bboxes per-component (scaled to rounded size) for reverse-window sampling
    per_comp_bboxes_scaled: Dict[str, List[List[int]]] = {}

    # Iterate components
    for fname, mask in sorted(masks.items()):
        label = comp_label_from_mask_filename(fname)
        s = slug(label)

        # For potential reverse-window check (not strictly necessary now, but kept)
        if s not in COORD_EXCLUDE:
            component_masks_binary_resized[label] = (_resize_mask_nn(mask, rw, rh) > 0)

        # BBoxes & count on original; then scale to rounded
        bboxes_orig, _ = connected_components_bboxes(mask, min_area)
        count = len(bboxes_orig)
        present = count > 0
        bboxes_scaled = _scale_bboxes_to_size(bboxes_orig, w, h, rw, rh) if present else []

        # Save for reverse-window if allowed
        if s not in COORD_EXCLUDE and bboxes_scaled:
            per_comp_bboxes_scaled.setdefault(label, []).extend(bboxes_scaled)

        # COUNT
        if s not in COUNT_EXCLUDE:
            records.append(build_count_record(sample_name, imgs, label, w, h, count))

        # COLOR
        if present and label in COLOR_COMPONENTS:
            assign = color_assign.get(label)
            if assign:
                cn = assign.get("color_name_zh")
                if cn:
                    records.append(build_color_record(sample_name, imgs, label, w, h, cn))

        # BBOX QA
        if s not in BBOX_EXCLUDE:
            records.append(build_bbox_record(sample_name, imgs, label, w, h, bboxes_scaled, present))

    # Reverse Color (unchanged)
    color_to_component: Dict[str,str] = {}
    for comp, info in color_assign.items():
        cn = info.get("color_name_zh")
        if cn:
            color_to_component[cn] = comp

    for cn, comp in color_to_component.items():
        records.append({
            "id": f"{sample_name}_rev_color_{slug(cn)}",
            "images": imgs,
            "meta": {
                "aug_size": [w, h],
                "aug_size_rounded": [rw, rh],
                "reverse": "color_to_component",
                "color_name": cn,
                "is_positive": True
            },
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。指定颜色："
                        f"{cn}。\n问题：该颜色在掩膜中对应的组件类型是什么？如无对应类型请返回 null。"
                    )
                },
                {"from": "gpt", "value": comp}
            ]
        })

    unused = [c for c in PALETTE_COLOR_NAMES_ZH if c not in color_to_component]
    rng.shuffle(unused)
    for cn in unused[:max(0, rev_color_neg_max)]:
        records.append({
            "id": f"{sample_name}_rev_color_{slug(cn)}",
            "images": imgs,
            "meta": {
                "aug_size": [w, h],
                "aug_size_rounded": [rw, rh],
                "reverse": "color_to_component",
                "color_name": cn,
                "is_positive": False
            },
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。指定颜色："
                        f"{cn}。\n问题：该颜色在掩膜中对应的组件类型是什么？如无对应类型请返回 null。"
                        "请仅输出组件类型名称或 null。"
                    )
                },
                {"from": "gpt", "value": "null"}
            ]
        })

    # --- NEW Reverse Window (single component): sample from actual bboxes (rounded coords) ---
    # Flatten all candidate (label, bbox) excluding COORD_EXCLUDE labels
    candidates: List[Tuple[str, List[int]]] = []
    for label, bbs in per_comp_bboxes_scaled.items():
        for bb in bbs:
            candidates.append((label, bb))

    if candidates and rev_coord_per_sample > 0:
        rng.shuffle(candidates)
        for label, bb in candidates[:rev_coord_per_sample]:
            records.append(build_reverse_window_single_record(sample_name, imgs, w, h, tuple(bb), label))

    # Write qa_mask.jsonl
    out_path = os.path.join(sample_dir, "qa_mask.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

    print(f"{OK} {sample_dir}: wrote qa_mask.jsonl with {len(records)} entries "
          f"(components={len(masks)}, color_mask={'yes' if has_color_mask else 'no'})")
    return len(records)

def main():
    ap = argparse.ArgumentParser(description="Generate dual-image mask-based QA (qa_mask.jsonl).")
    ap.add_argument("-r", "--root", default=".", help="with_label root directory.")
    ap.add_argument("--only", nargs="*", help="Specific sample folder names to process.")
    ap.add_argument("--min-area", type=int, default=100, help="Minimum area (pixels) for counting instances.")
    ap.add_argument("--aug-only", action="store_true", help="Process only augmented samples.")
    ap.add_argument("--defect-only", action="store_true", help="Process only defect samples.")
    # Reverse QA controls
    ap.add_argument("--rev-color-neg-max", type=int, default=10, help="Max negative unused-color reverse color QAs per sample.")
    ap.add_argument("--rev-coord-per-sample", type=int, default=5, help="Number of reverse window QAs per sample (sampled from true bboxes).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    rng = random.Random(args.seed)

    sample_dirs: List[str] = []
    if not args.defect_only:
        sample_dirs.extend(list_aug_folders(root))
    if not args.aug_only:
        sample_dirs.extend(list_defect_folders(root))

    if args.only:
        wanted = set(args.only)
        sample_dirs = [p for p in sample_dirs if os.path.basename(p) in wanted]

    total_records = 0
    processed = 0
    for sdir in sample_dirs:
        try:
            n = process_sample(
                root, sdir, args.min_area,
                args.rev_color_neg_max,
                args.rev_coord_per_sample,
                rng
            )
            total_records += n
            processed += 1
        except Exception as e:
            print(f"{ERR} {sdir}: {e}")

    print(f"{TOTAL} Samples processed: {processed}  QA lines: {total_records}")

if __name__ == "__main__":
    main()
