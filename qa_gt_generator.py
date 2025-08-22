"""
Generate per-augmentation QA jsonl (qa.jsonl) based on augmented RGB and masks.
Reads aug_image.png and masks/*.png
Answers questions about the components in the augmented image: presence, count, bboxes, color (hex and CSS3 name).
"""

import os
import json
import argparse
from typing import List, Tuple, Dict
import numpy as np
import cv2
import webcolors
from colorama import init, Fore, Back, Style

OK = Fore.GREEN + "[OK]" + Style.RESET_ALL
ERR = Fore.RED + "[ERR]" + Style.RESET_ALL
TOTAL = Back.YELLOW + "[TOTAL]" + Style.RESET_ALL

def to_posix(rel_path: str) -> str:
    return rel_path.replace("\\", "/")

def list_aug_folders(root: str) -> List[str]:
    """
    Finds all augmented sample folders under root:
      <root>/<Class>/aug/<Class_n>
    """
    aug_folders: List[str] = []
    for sample_entry in os.scandir(root):
        if not sample_entry.is_dir() or sample_entry.name.startswith('.'):
            continue
        aug_dir = os.path.join(sample_entry.path, "aug")
        if not os.path.isdir(aug_dir):
            continue
        for aug_entry in os.scandir(aug_dir):
            if aug_entry.is_dir() and not aug_entry.name.startswith('.'):
                aug_folders.append(aug_entry.path)
    return aug_folders

def list_defect_folders(root: str) -> List[str]:
    """
    Finds all defect sample folders under root:
      <root>/defect/<Class>_defect/<Class>_defect_n
    """
    out: List[str] = []
    defect_root = os.path.join(root, "defect")
    if not os.path.isdir(defect_root):
        return out
    for class_def in os.scandir(defect_root):
        if not class_def.is_dir() or class_def.name.startswith('.'):
            continue
        # Expect names like "<Class>_defect"
        for sample_entry in os.scandir(class_def.path):
            if sample_entry.is_dir() and not sample_entry.name.startswith('.'):
                out.append(sample_entry.path)
    return out

def load_aug(aug_dir: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Loads RGB image and all masks in aug_dir/masks.
    """
    img_path = os.path.join(aug_dir, "aug_image.png")
    rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Cannot read aug image: {img_path}")
    masks_dir = os.path.join(aug_dir, "masks")
    masks: Dict[str, np.ndarray] = {}
    if os.path.isdir(masks_dir):
        for f in sorted(os.listdir(masks_dir)):
            if not f.lower().endswith(".png"):
                continue
            p = os.path.join(masks_dir, f)
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            # Binarize to 0/255
            m = (m > 127).astype(np.uint8) * 255
            masks[f] = m
    return rgb, masks

def connected_components_bboxes(mask: np.ndarray, min_area: int = 0) -> Tuple[List[List[int]], np.ndarray]:
    """
    Computes 8-connected components on a binary mask (0/255)
    Returns (bboxes, filtered_mask), where bboxes is a list of [x1,y1,x2,y2] (inclusive)
    for each kept instance, and filtered_mask is a binary mask (0/255) containing only
    the kept instances.
    Set min_area to filter out tiny specks
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    bin01 = (mask > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    min_area = max(int(min_area), 0)
    bboxes: List[List[int]] = []
    kept_labels: List[int] = []
    for lbl in range(1, num):  # skip background 0
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
        x2, y2 = x + w - 1, y + h - 1  # inclusive
        bboxes.append([x1, y1, x2, y2])
        kept_labels.append(lbl)
    # Build filtered mask containing only kept labels
    if len(kept_labels) == 0:
        filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    else:
        filtered_mask = np.isin(labels, kept_labels).astype(np.uint8) * 255
    return bboxes, filtered_mask

def most_frequent_color_hex(rgb_bgr: np.ndarray, mask: np.ndarray, bin_size: int = 16) -> str:
    '''
    Quantize masked RGB pixels into coarse bins (pix // bin_size) and linearize to a single bin_idx.
    Count bins with np.bincount and pick dom = counts.argmax() as the dominant quantized bin.
    Select pixels whose quantized coords match dom (fallback to all masked pixels if none).
    Average selected pixels per channel, round to ints and format as "#RRGGBB".
    Quantization groups similar colors and stabilizes the dominant-color estimate vs raw 24-bit mode.
    '''
    if mask is None:
        return "null"
    m = (mask > 0).astype(np.uint8) * 255
    # Erode to remove boundary bleed; fallback to original if erosion empties
    kernel = np.ones((3, 3), np.uint8)
    m_e = cv2.erode(m, kernel, iterations=1)
    use_m = m_e if np.count_nonzero(m_e) > 0 else m
    if np.count_nonzero(use_m) == 0:
        return "null"

    # Convert to RGB and select masked pixels
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    pix = rgb[use_m > 0]
    if pix.size == 0:
        return "null"

    # Quantize to reduce noise, then find dominant bin
    q = (pix // bin_size).astype(np.int32)  # values in [0, 255/bin_size]
    bases = 256 // bin_size
    bin_idx = (q[:, 0] * bases + q[:, 1]) * bases + q[:, 2]
    counts = np.bincount(bin_idx, minlength=bases * bases * bases)
    dom = int(counts.argmax())

    # Extract pixels in dominant bin and take average in original space
    q0 = dom // (bases * bases)
    q1 = (dom // bases) % bases
    q2 = dom % bases
    sel = (q[:, 0] == q0) & (q[:, 1] == q1) & (q[:, 2] == q2)
    sel_pix = pix[sel]
    if sel_pix.size == 0:
        sel_pix = pix  # fallback

    r, g, b = np.round(sel_pix.mean(axis=0)).astype(int).tolist()
    return f"#{r:02X}{g:02X}{b:02X}"

def closest_css3_color_name(hex_str: str) -> str:
    """
    Return exact CSS3 name if available; otherwise the closest CSS3 name by RGB distance (webcolors only).
    """
    if not hex_str or hex_str == "null":
        return "unknown"
    try:
        return webcolors.hex_to_name(hex_str, spec="css3")
    except Exception:
        pass
    try:
        tr, tg, tb = tuple(webcolors.hex_to_rgb(hex_str))
    except Exception:
        return "unknown"
    # Use CSS3 name set from webcolors; fall back to internal defs if needed
    try:
        from webcolors import CSS3_NAMES_TO_HEX as NAME2HEX 
    except Exception:
        try:
            from webcolors._definitions import CSS3_NAMES_TO_HEX as NAME2HEX 
        except Exception:
            NAME2HEX = {}
    if not NAME2HEX:
        return "unknown"
    best_name, best_d2 = "unknown", None
    for name, hx in NAME2HEX.items():
        r, g, b = tuple(webcolors.hex_to_rgb(hx))
        d2 = (tr - r) ** 2 + (tg - g) ** 2 + (tb - b) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2, best_name = d2, name
    return best_name

def comp_name_from_filename(fname: str) -> str:
    """
    Derive component name (label) from mask filename. E.g., "Com_Hole.png" -> "Com_Hole".
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    return base

def comp_id_slug(label: str) -> str:
    """
    Lowercase component id slug for 'id' construction. E.g., "Com_Hole" -> "com_hole".
    """
    return label.replace(" ", "_").replace("-", "_").lower()

def build_records_for_component(
    root: str,
    aug_dir: str,
    aug_rel_image_path: str,
    label: str,
    img_w: int,
    img_h: int,
    mask: np.ndarray,
    min_area: int
) -> List[Dict]:
    """
    Build three JSONL records (count, color, bboxes) for a component label/mask.
    """
    slug = comp_id_slug(label)
    aug_name = os.path.basename(aug_dir)  # e.g., "27QHD_12"
    # Compute stats (apply min_area filtering)
    bboxes, filtered_mask = connected_components_bboxes(mask, min_area=min_area)
    count = len(bboxes)
    present = (count > 0)
    color_hex = most_frequent_color_hex(cv2.imread(os.path.join(aug_dir, "aug_image.png"), cv2.IMREAD_COLOR), filtered_mask)
    # Prepare shared meta
    meta_base = {"aug_size": [img_w, img_h], "component": label}
    # Records
    records: List[Dict] = []

    # Count (component-level)
    records.append({
        "id": f"{aug_name}_{slug}_count",
        "images": [aug_rel_image_path],
        "meta": meta_base,
        "conversations": [
            # {"from": "human", "value": f"<image>\nFor the component type: {label}.\nQuestion: How many are visible? Reply with an integer only."},
            {"from": "human", "value": f"<image>\n针对部件类型： {label}。\n问题：可见数量有多少？请仅回答一个整数。"},
            {"from": "gpt", "value": str(count)}
        ]
    })

    # Color (component-level)
    if present:
        color_name = closest_css3_color_name(color_hex)
        color_json_str = json.dumps({"hex": color_hex, "name": color_name}, separators=(",", ":"))
    else:
        color_json_str = "null"
    records.append({
        "id": f"{aug_name}_{slug}_color",
        "images": [aug_rel_image_path],
        "meta": {**meta_base, "color_format": "hex+css3_name"},
        "conversations": [
            # {"from": "human", "value": f"<image>\nFor the component type: {label}.\nQuestion: Return the color as a hex string like \"#RRGGBB\" if present, otherwise return null. Reply with a single token."},
            {"from": "human", "value": f"<image>\n针对部件类型： {label}。\n问题：若存在，请以十六进制字符串格式 \"#RRGGBB\" 返回颜色，否则返回 null。请仅回答一个标记。"},
            {"from": "gpt", "value": color_hex if present else "null"}
        ]
    })

    # BBoxes (component-level)
    if present:
        bbox_json_str = json.dumps({"bboxes": bboxes}, separators=(",", ":"))
    else:
        bbox_json_str = "null"
    records.append({
        "id": f"{aug_name}_{slug}_bboxes",
        "images": [aug_rel_image_path],
        "meta": meta_base,
        "conversations": [
            # {"from": "human", "value": f"<image>\nFor the component type: {label}.\nQuestion: Return JSON exactly as {{\"bboxes\":[[x1,y1,x2,y2],...]}} in THIS IMAGE'S pixel coordinates; if absent return null."},
            {"from": "human", "value": f"<image>\n针对部件类型： {label}。\n问题：若存在，请严格按格式 {{\"bboxes\":[[x1,y1,x2,y2],...]}} 返回该图像像素坐标系中的边界框；若不存在则返回 null。"},
            {"from": "gpt", "value": bbox_json_str}
        ]
    })

    return records

def generate_qa_for_aug(root: str, aug_dir: str, min_area: int) -> int:
    """
    Generates qa.jsonl inside aug_dir based on aug_image.png and masks/*.png.
    Also emits a sample-level 'defect present' Q/A using meta.json['defect'] if available.
    Returns number of JSONL lines written.
    """
    rgb_bgr, masks = load_aug(aug_dir)
    h, w = rgb_bgr.shape[:2]

    # Relative image path from root
    aug_image_path = os.path.join(aug_dir, "aug_image.png")
    rel_img = to_posix(os.path.relpath(aug_image_path, root))

    # Read defect flag from meta.json (default: False if missing)
    defect_flag = False
    meta_path = os.path.join(aug_dir, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_obj = json.load(f)
            defect_flag = bool(meta_obj.get("defect", False))
        except Exception:
            defect_flag = False

    all_records: List[Dict] = []

    # Sample-level defect presence record (single per sample)
    aug_name = os.path.basename(aug_dir)
    all_records.append({
        "id": f"{aug_name}_defect_flag",
        "images": [rel_img],
        "meta": {"aug_size": [w, h]},
        "conversations": [
            # {"from": "human", "value": "<image>\nQuestion: Does this image contain a defect? Reply with yes or no only."},
            # {"from": "gpt", "value": "yes" if defect_flag else "no"}
            {"from": "human", "value": "<image>\n问题：该图像是否存在缺陷？请仅回答是或否。"},
            {"from": "gpt", "value": "是" if defect_flag else "否"}
        ]
    })

    # Component-level records (count, color, bboxes)
    for fname, mask in sorted(masks.items()):
        label = comp_name_from_filename(fname)
        recs = build_records_for_component(root, aug_dir, rel_img, label, w, h, mask, min_area)
        all_records.extend(recs)

    # Write qa.jsonl
    out_path = os.path.join(aug_dir, "qa.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

    print(f"{OK} {aug_dir}: wrote qa.jsonl with {len(all_records)} entries for {len(masks)} component(s).")
    return len(all_records)

def main():
    init(autoreset=True)
    ap = argparse.ArgumentParser(description="Generate qa.jsonl for augmented and/or defect sample folders.")
    ap.add_argument("-r", "--root", default=".", help="Root directory (with_label). Default: current dir")
    ap.add_argument("--only", nargs="*", help="Optional list of specific sample folder names to process (e.g., 27QHD_12 or 27QHD_defect_3).")
    ap.add_argument("--min-area", type=int, default=100, help="Minimum area in pixels for a component instance to be counted and included in bboxes/color. Default: 100")
    ap.add_argument("--aug-only", action="store_true", help="Process only augmented samples.")
    ap.add_argument("--defect-only", action="store_true", help="Process only defect samples.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)

    include_aug = True
    include_defect = True
    if args.aug_only:
        include_defect = False
    if args.defect_only:
        include_aug = False

    sample_dirs: List[str] = []
    if include_aug:
        sample_dirs.extend(list_aug_folders(root))
    if include_defect:
        sample_dirs.extend(list_defect_folders(root))

    if args.only:
        names = set(args.only)
        sample_dirs = [d for d in sample_dirs if os.path.basename(d) in names]

    total_entries = 0
    total_samples = 0
    for sample_dir in sample_dirs:
        try:
            n = generate_qa_for_aug(root, sample_dir, args.min_area)
            total_entries += n
            total_samples += 1
        except Exception as e:
            print(f"{ERR} {sample_dir}: {e}")

    print(f"{TOTAL} Wrote {total_entries} JSONL lines across {total_samples} sample folders.")

if __name__ == "__main__":
    main()