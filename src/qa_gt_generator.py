#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa_gt_generator.py

Generate per-sample VQA for the Domain Dataset:

  - qa_dual.jsonl : questions refer to both
        1) aug_image.png   (RGB original)
        2) color_mask.png  (colored component mask)

  - qa_mask.jsonl : questions refer ONLY to
        1) color_mask.png  (colored component mask)

Both files contain the same logical QA set (same ids/components/bboxes/etc.);
only the "images" list and natural language prompts differ.

Ground-truth answers are derived from binary masks + meta.json color
assignments.

Usage (examples):
  # Generate both qa_dual.jsonl and qa_mask.jsonl (default)
  python qa_gt_generator.py -r /path/to/with_label

  # Only dual-image QA
  python qa_gt_generator.py -r /path/to/with_label --mode dual

  # Only mask-only QA
  python qa_gt_generator.py -r /path/to/with_label --mode mask

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
COORD_EXCLUDE = {"pxl_ito", "com_hole", "gate", "ito"}  # excluded from reverse-window targets


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
        if nx2 < nx1:
            nx1, nx2 = nx2, nx1
        if ny2 < ny1:
            ny1, ny2 = ny2, ny1
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


def images_dual_list(root: str, sample_dir: str) -> List[str]:
    """Dual-image mode: RGB + color_mask if available, otherwise just RGB."""
    rgb_path = os.path.join(sample_dir, "aug_image.png")
    mask_path = os.path.join(sample_dir, "color_mask.png")
    rgb_rel = to_posix(os.path.relpath(rgb_path, root))
    imgs = [rgb_rel]
    if os.path.isfile(mask_path):
        mask_rel = to_posix(os.path.relpath(mask_path, root))
        imgs.append(mask_rel)
    return imgs


def images_mask_list(root: str, sample_dir: str) -> List[str]:
    """Mask-only mode: require color_mask.png."""
    mask_path = os.path.join(sample_dir, "color_mask.png")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Missing color_mask.png for mask-only QA at {sample_dir}")
    mask_rel = to_posix(os.path.relpath(mask_path, root))
    return [mask_rel]


def build_defect_record(sample_name: str, imgs: List[str], w: int, h: int,
                        defect_flag: bool, mode: str) -> Dict:
    if mode == "dual":
        prefix = "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。（仅用于结构参考，不含缺陷）。"
    else:
        prefix = "<image>\n如图所示是一张半导体显示面板的组件掩膜合成图像，其中各组件掩膜以不同颜色区分。（仅用于结构参考，不含缺陷）。"
    return {
        "id": f"{sample_name}_defect_flag",
        "images": imgs,
        "meta": {
            "aug_size": [w, h],
            "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)]
        },
        "conversations": [
            {
                "from": "human",
                "value": f"{prefix}\n问题：原始图像中是否存在缺陷？"
            },
            {"from": "gpt", "value": "是" if defect_flag else "否"}
        ]
    }


def build_count_record(sample_name: str, imgs: List[str], label: str, w: int, h: int,
                       count: int, mode: str) -> Dict:
    if mode == "dual":
        prefix = "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。"
        note = "（请交叉参考两张图以确保一致性）"
    else:
        prefix = "<image>\n如图所示是一张半导体显示面板的组件掩膜合成图像，其中各组件掩膜以不同颜色区分。"
        note = ""
    return {
        "id": f"{sample_name}_{slug(label)}_count",
        "images": imgs,
        "meta": {
            "aug_size": [w, h],
            "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)],
            "component": label
        },
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"\n{prefix}针对组件类型：{label}。\n问题：该类型的组件在掩膜中可见数量是多少？{note}"
                )
            },
            {"from": "gpt", "value": str(count)}
        ]
    }


def build_color_record(sample_name: str, imgs: List[str], label: str, w: int, h: int,
                       color_name: str, mode: str) -> Dict:
    if mode == "dual":
        prefix = "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。"
    else:
        prefix = "<image>\n如图所示是一张半导体显示面板的组件掩膜合成图像，其中各组件掩膜以不同颜色区分。"
    return {
        "id": f"{sample_name}_{slug(label)}_color",
        "images": imgs,
        "meta": {
            "aug_size": [w, h],
            "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)],
            "component": label,
            "color_source": "mask"
        },
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"\n{prefix}针对组件类型：{label}。\n"
                    "问题：若该类型的组件存在，请返回其掩膜颜色的中文名称；否则返回 null。"
                )
            },
            {"from": "gpt", "value": color_name}
        ]
    }


def build_bbox_record(sample_name: str, imgs: List[str], label: str, w: int, h: int,
                      bboxes_scaled: List[List[int]], present: bool, mode: str) -> Dict:
    bbox_list = [{"bbox_2d": bb, "label": label} for bb in bboxes_scaled] if present else None
    bbox_json = json.dumps(bbox_list, ensure_ascii=False, separators=(",", ":")) if bbox_list is not None else "null"

    if mode == "dual":
        prefix = "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。"
        note = "\n注意：请综合参考两张图像进行判断。"
    else:
        prefix = "<image>\n如图所示是一张半导体显示面板的组件掩膜合成图像，其中各组件掩膜以不同颜色区分。"
        note = ""

    return {
        "id": f"{sample_name}_{slug(label)}_bboxes",
        "images": imgs,
        "meta": {
            "aug_size": [w, h],
            "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)],
            "component": label
        },
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"\n{prefix}针对组件类型：{label}。\n问题：若存在，请返回所有该部件在图像像素坐标系中的边界框；"
                    "若不存在则返回 null。" + note
                )
            },
            {"from": "gpt", "value": bbox_json}
        ]
    }


def build_rev_color_record(sample_name: str, imgs: List[str], w: int, h: int,
                           cn: str, comp: str | None,
                           is_positive: bool, mode: str) -> Dict:
    if mode == "dual":
        prefix = "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。"
    else:
        prefix = "<image>\n如图所示是一张半导体显示面板的组件掩膜合成图像，其中各组件掩膜以不同颜色区分。"
    return {
        "id": f"{sample_name}_rev_color_{slug(cn)}",
        "images": imgs,
        "meta": {
            "aug_size": [w, h],
            "aug_size_rounded": [_round_to_28_nearest(w), _round_to_28_nearest(h)],
            "reverse": "color_to_component",
            "color_name": cn,
            "is_positive": bool(is_positive)
        },
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"\n{prefix}指定颜色：{cn}。\n"
                    "问题：该颜色在掩膜中对应的组件类型是什么？如无对应类型请返回 null。"
                )
            },
            {"from": "gpt", "value": comp if (is_positive and comp) else "null"}
        ]
    }


def build_reverse_window_single_record(
    sample_name: str,
    imgs: List[str],
    w: int,
    h: int,
    window: Tuple[int,int,int,int],
    label: str,
    mode: str
) -> Dict:
    x1, y1, x2, y2 = window
    wid = f"{x1}_{y1}_{x2}_{y2}"
    if mode == "dual":
        prefix = "<image>\n<image>\n第一张：原始RGB图像。第二张：组件掩膜图像。"
        note = "\n注意：请综合参考两张图像进行判断。"
    else:
        prefix = "<image>\n如图所示是一张半导体显示面板的组件掩膜合成图像，其中各组件掩膜以不同颜色区分。"
        note = ""

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
                    f"\n{prefix}指定窗口区域：[x1={x1},y1={y1},x2={x2},y2={y2}]。\n"
                    "问题：该窗口内显示的主要组件类型是什么？" + note
                )
            },
            {"from": "gpt", "value": label if label else "null"}
        ]
    }


def process_sample(
    root: str,
    sample_dir: str,
    min_area: int,
    rev_color_neg_max: int,
    rev_coord_per_sample: int,
    rng: random.Random,
    modes: List[str]
) -> Tuple[int, int]:
    """
    Process a single sample folder and generate records for dual and/or mask modes.

    Returns:
        (num_dual_records, num_mask_records)
    """
    rgb, masks = load_rgb_and_masks(sample_dir)
    h, w = rgb.shape[:2]
    rw, rh = _round_to_28_nearest(w), _round_to_28_nearest(h)

    sample_name = os.path.basename(sample_dir)
    meta = load_meta(sample_dir)
    defect_flag = bool(meta.get("defect", False))
    color_assign = meta.get("color_mask", {}).get("assigned_colors", {}) or {}

    # Pre-compute image lists
    imgs_dual = images_dual_list(root, sample_dir) if "dual" in modes else []
    imgs_mask = images_mask_list(root, sample_dir) if "mask" in modes else []

    records_dual: List[Dict] = []
    records_mask: List[Dict] = []

    # Track components that actually appear in this sample (count > 0)
    present_labels: set[str] = set()

    # Precompute resized masks only if needed elsewhere (keeping for parity)
    component_masks_binary_resized: Dict[str, np.ndarray] = {}

    # Accumulate bboxes per-component (scaled to rounded size) for reverse-window sampling
    per_comp_bboxes_scaled: Dict[str, List[List[int]]] = {}

    # Optional defect flag QA (currently not used in original; kept here for completeness)
    # if "dual" in modes:
    #     records_dual.append(build_defect_record(sample_name, imgs_dual, w, h, defect_flag, "dual"))
    # if "mask" in modes:
    #     records_mask.append(build_defect_record(sample_name, imgs_mask, w, h, defect_flag, "mask"))

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
        if present:
            present_labels.add(label)
        bboxes_scaled = _scale_bboxes_to_size(bboxes_orig, w, h, rw, rh) if present else []

        # Save for reverse-window if allowed
        if s not in COORD_EXCLUDE and bboxes_scaled:
            per_comp_bboxes_scaled.setdefault(label, []).extend(bboxes_scaled)

        # COUNT
        if s not in COUNT_EXCLUDE:
            if "dual" in modes:
                records_dual.append(
                    build_count_record(sample_name, imgs_dual, label, w, h, count, "dual")
                )
            if "mask" in modes:
                records_mask.append(
                    build_count_record(sample_name, imgs_mask, label, w, h, count, "mask")
                )

        # COLOR
        if present and label in COLOR_COMPONENTS:
            assign = color_assign.get(label)
            if assign:
                cn = assign.get("color_name_zh")
                if cn:
                    if "dual" in modes:
                        records_dual.append(
                            build_color_record(sample_name, imgs_dual, label, w, h, cn, "dual")
                        )
                    if "mask" in modes:
                        records_mask.append(
                            build_color_record(sample_name, imgs_mask, label, w, h, cn, "mask")
                        )

        # BBOX QA
        if s not in BBOX_EXCLUDE:
            if "dual" in modes:
                records_dual.append(
                    build_bbox_record(sample_name, imgs_dual, label, w, h, bboxes_scaled, present, "dual")
                )
            if "mask" in modes:
                records_mask.append(
                    build_bbox_record(sample_name, imgs_mask, label, w, h, bboxes_scaled, present, "mask")
                )

    # Reverse Color (positive + negatives)
    color_to_component: Dict[str, str] = {}
    for comp, info in color_assign.items():
        cn = info.get("color_name_zh")
        if cn:
            color_to_component[cn] = comp

    for cn, comp in color_to_component.items():
        is_present = comp in present_labels  # derived from counts
        if "dual" in modes:
            records_dual.append(
                build_rev_color_record(sample_name, imgs_dual, w, h, cn, comp, is_present, "dual")
            )
        if "mask" in modes:
            records_mask.append(
                build_rev_color_record(sample_name, imgs_mask, w, h, cn, comp, is_present, "mask")
            )

    unused = [c for c in PALETTE_COLOR_NAMES_ZH if c not in color_to_component]
    rng.shuffle(unused)
    for cn in unused[:max(0, rev_color_neg_max)]:
        if "dual" in modes:
            records_dual.append(
                build_rev_color_record(sample_name, imgs_dual, w, h, cn, None, False, "dual")
            )
        if "mask" in modes:
            records_mask.append(
                build_rev_color_record(sample_name, imgs_mask, w, h, cn, None, False, "mask")
            )

    # Reverse Window (single component): sample from actual bboxes (rounded coords)
    candidates: List[Tuple[str, List[int]]] = []
    for label, bbs in per_comp_bboxes_scaled.items():
        for bb in bbs:
            candidates.append((label, bb))

    if candidates and rev_coord_per_sample > 0:
        rng.shuffle(candidates)
        for label, bb in candidates[:rev_coord_per_sample]:
            window = tuple(bb)
            if "dual" in modes:
                records_dual.append(
                    build_reverse_window_single_record(sample_name, imgs_dual, w, h, window, label, "dual")
                )
            if "mask" in modes:
                records_mask.append(
                    build_reverse_window_single_record(sample_name, imgs_mask, w, h, window, label, "mask")
                )

    # Write outputs
    n_dual = len(records_dual)
    n_mask = len(records_mask)

    if "dual" in modes:
        out_dual = os.path.join(sample_dir, "qa_dual.jsonl")
        with open(out_dual, "w", encoding="utf-8") as f:
            for r in records_dual:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")

    if "mask" in modes:
        out_mask = os.path.join(sample_dir, "qa_mask.jsonl")
        with open(out_mask, "w", encoding="utf-8") as f:
            for r in records_mask:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")

    mode_desc = []
    if "dual" in modes:
        mode_desc.append(f"qa_dual={n_dual}")
    if "mask" in modes:
        mode_desc.append(f"qa_mask={n_mask}")
    mode_desc_str = ", ".join(mode_desc) if mode_desc else "no outputs"

    print(f"{OK} {sample_dir}: {mode_desc_str} (components={len(masks)})")
    return n_dual, n_mask


def main():
    ap = argparse.ArgumentParser(
        description="Generate Domain Dataset VQA (qa_dual.jsonl, qa_mask.jsonl)."
    )
    ap.add_argument("-r", "--root", default=".", help="with_label root directory.")
    ap.add_argument("--only", nargs="*", help="Specific sample folder names to process.")
    ap.add_argument("--min-area", type=int, default=70, help="Minimum area (pixels) for counting instances.")
    ap.add_argument("--aug-only", action="store_true", help="Process only augmented samples.")
    ap.add_argument("--defect-only", action="store_true", help="Process only defect samples.")
    # Reverse QA controls
    ap.add_argument("--rev-color-neg-max", type=int, default=10,
                    help="Max negative unused-color reverse color QAs per sample.")
    ap.add_argument("--rev-coord-per-sample", type=int, default=10,
                    help="Number of reverse window QAs per sample (sampled from true bboxes).")
    ap.add_argument("--seed", type=int, default=67, help="Random seed for reproducibility.")
    ap.add_argument(
        "--mode",
        choices=["dual", "mask", "both"],
        default="both",
        help="Which QA files to generate per sample: dual (qa_dual.jsonl), "
             "mask (qa_mask.jsonl), or both."
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    rng = random.Random(args.seed)

    # Determine modes list
    if args.mode == "dual":
        modes = ["dual"]
    elif args.mode == "mask":
        modes = ["mask"]
    else:
        modes = ["dual", "mask"]

    sample_dirs: List[str] = []
    if not args.defect_only:
        sample_dirs.extend(list_aug_folders(root))
    if not args.aug_only:
        sample_dirs.extend(list_defect_folders(root))

    if args.only:
        wanted = set(args.only)
        sample_dirs = [p for p in sample_dirs if os.path.basename(p) in wanted]

    total_dual = 0
    total_mask = 0
    processed = 0

    for sdir in sample_dirs:
        try:
            n_dual, n_mask = process_sample(
                root,
                sdir,
                args.min_area,
                args.rev_color_neg_max,
                args.rev_coord_per_sample,
                rng,
                modes
            )
            total_dual += n_dual
            total_mask += n_mask
            processed += 1
        except Exception as e:
            print(f"{ERR} {sdir}: {e}")

    msg = f"{TOTAL} Samples processed: {processed}"
    if "dual" in modes:
        msg += f"  qa_dual lines: {total_dual}"
    if "mask" in modes:
        msg += f"  qa_mask lines: {total_mask}"
    print(msg)


if __name__ == "__main__":
    main()
