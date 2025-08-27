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
from colorama import init, Fore, Style

init(autoreset=True)
OK = Fore.GREEN + "[OK]" + Style.RESET_ALL
ERR = Fore.RED + "[ERR]" + Style.RESET_ALL
TOTAL = Fore.CYAN + "[TOTAL]" + Style.RESET_ALL

# Components allowed for COLOR QA (must be present + assigned color)
COLOR_COMPONENTS = [
    "Source", "Drain", "TFT", "VIA_Hole", "Mesh", "Mesh_Hole",
    "Com", "Data", "Gate", "ITO"
]

# Exclusions
COUNT_EXCLUDE = {"pxl_ito", "com_hole", "gate", "ito"}
BBOX_EXCLUDE = {"pxl_ito", "com_hole", "gate", "ito"}

def to_posix(p: str) -> str:
    return p.replace("\\", "/")

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
        "meta": {"aug_size": [w, h]},
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n第一张：原始RGB图像。第二张：彩色组件掩膜图像（仅用于结构参考，不含缺陷）。问题：原始图像中是否存在缺陷？"
            },
            {"from": "gpt", "value": "是" if defect_flag else "否"}
        ]
    }

def build_count_record(sample_name: str, imgs: List[str], label: str, w: int, h: int, count: int) -> Dict:
    return {
        "id": f"{sample_name}_{slug(label)}_count",
        "images": imgs,
        "meta": {"aug_size": [w, h], "component": label},
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n第一张：原始RGB图像。第二张：彩色组件掩膜图像。针对组件类型："
                         f"{label}。\n问题：该类型在掩膜中可见数量是多少？请仅回答一个整数。"
            },
            {"from": "gpt", "value": str(count)}
        ]
    }

def build_color_record(sample_name: str, imgs: List[str], label: str, w: int, h: int, color_name: str) -> Dict:
    return {
        "id": f"{sample_name}_{slug(label)}_color",
        "images": imgs,
        "meta": {"aug_size": [w, h], "component": label, "color_source": "mask"},
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n第一张：原始RGB图像。第二张：彩色组件掩膜图像。针对组件类型："
                         f"{label}。\n问题：若该类型存在，请返回其掩膜颜色的中文名称；否则返回 null。"
            },
            {"from": "gpt", "value": color_name}
        ]
    }

def build_bbox_record(sample_name: str, imgs: List[str], label: str, w: int, h: int, bboxes: List[List[int]], present: bool) -> Dict:
    bbox_json = json.dumps({"bboxes": bboxes}, separators=(",", ":")) if present else "null"
    return {
        "id": f"{sample_name}_{slug(label)}_bboxes",
        "images": imgs,
        "meta": {"aug_size": [w, h], "component": label},
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n第一张：原始RGB图像。第二张：彩色组件掩膜图像。针对组件类型："
                         f"{label}。\n问题：若该类型存在，请严格按格式 {{\"bboxes\":[[x1,y1,x2,y2],...]}} 输出边界框（基于原始图像像素坐标系）；否则返回 null。"
            },
            {"from": "gpt", "value": bbox_json}
        ]
    }

def process_sample(root: str, sample_dir: str, min_area: int) -> int:
    rgb, masks = load_rgb_and_masks(sample_dir)
    h, w = rgb.shape[:2]

    sample_name = os.path.basename(sample_dir)
    meta = load_meta(sample_dir)
    defect_flag = bool(meta.get("defect", False))
    color_assign = meta.get("color_mask", {}).get("assigned_colors", {}) or {}

    has_color_mask = os.path.isfile(os.path.join(sample_dir, "color_mask.png"))
    imgs = images_list(root, sample_dir, include_mask=has_color_mask)

    records: List[Dict] = []
    # Defect record
    records.append(build_defect_record(sample_name, imgs, w, h, defect_flag))

    # Iterate components
    for fname, mask in sorted(masks.items()):
        label = comp_label_from_mask_filename(fname)
        s = slug(label)

        # Compute instances & bboxes
        bboxes, filtered = connected_components_bboxes(mask, min_area)
        count = len(bboxes)
        present = count > 0

        # COUNT QA (unless excluded)
        if s not in COUNT_EXCLUDE:
            records.append(build_count_record(sample_name, imgs, label, w, h, count))

        # COLOR QA only if component in COLOR_COMPONENTS, present, and color assignment exists
        if present and label in COLOR_COMPONENTS:
            assign = color_assign.get(label)
            if assign:
                cn = assign.get("color_name_zh")
                if cn:  # ensure we have a Chinese color name
                    records.append(build_color_record(sample_name, imgs, label, w, h, cn))

        # BBOX QA (unless excluded)
        if s not in BBOX_EXCLUDE:
            records.append(build_bbox_record(sample_name, imgs, label, w, h, bboxes, present))

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
    args = ap.parse_args()

    root = os.path.abspath(args.root)

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
            n = process_sample(root, sdir, args.min_area)
            total_records += n
            processed += 1
        except Exception as e:
            print(f"{ERR} {sdir}: {e}")

    print(f"{TOTAL} Samples processed: {processed}  QA lines: {total_records}")

if __name__ == "__main__":
    main()