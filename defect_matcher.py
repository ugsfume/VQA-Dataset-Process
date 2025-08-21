"""
Batch-import defect images into VQA samples by template matching:

- Walks known defect dataset folders (recursively find *.jpg) under a source root.
- Maps each dataset folder name to a class (e.g., 27QHD, 238FHD).
- For each defect JPG, match against class template with rotations [0,90,180,270] and multi-scale search.
- If score >= threshold: create a defect sample at with_label/defect/<Class>_defect/<Class>_defect_n:
  * aug_image.png = letterboxed defect image (512x512 by default)
  * masks/ = cropped full masks to matched bbox, inverse-rotated to defect orientation, letterboxed to 512x512
  * meta.json = match info, source_defect_image path, defect=true, template paths, etc.
  * (optional) matched.png visualization
- Prints per-dataset and per-class stats and grand totals.

Run from with_label (root):
  PowerShell: python .\src\defect_batch_import.py --save-matched-vis
  Linux:      python src/defect_batch_import.py --save-matched-vis
"""
import os
import re
import json
import argparse
from typing import Tuple, Dict, List

import cv2
import numpy as np
from colorama import init, Fore, Style, Back
import random

# -------------------------------
# Dataset mapping
# -------------------------------
DATASET_TO_CLASS: Dict[str, str] = {
    # 27QHD
    "2701G1D": "27QHD",
    "AA-270QHD-0219": "27QHD",
    "TL270AK2BA01": "27QHD",
    # 238FHD
    "238FHD-TSRTP": "238FHD",
    "AA_238FHD": "238FHD",
    "AA-238FHD-0321": "238FHD",
    "AA_238FHD_1": "238FHD",
    # 270FHD
    "270FHD-TOITP": "270FHD",
    "AA_270FHD": "270FHD",
    "AA-270FHD-0510": "270FHD",
    # TL156
    "AA_156DLS_0424": "TL156",
    "AA-156DLS-0423": "TL156",
    # 238QHD
    "AA_238QHD": "238QHD",
    "TL238AE2BA01": "238QHD",
    "AA-238QHD-0321": "238QHD",
    # 245FHD
    "AA_245FHD_0414": "245FHD",
    "TL245A12BA01": "245FHD",
    # 215FHD
    "TL215A3BBA01": "215FHD",
}

# actual path used for linux deployment
DEFAULT_SOURCE_ROOT = "/mnt/workspace/autorepair_t9/data/最终通用模型/full_train_data"
# windows path just for temporary testing
# DEFAULT_SOURCE_ROOT = "C:\\Users\\tclre\\Downloads\\defect_dataset"

# -----------------------------------------
# Core helpers (ported from defect_match.py)
# -----------------------------------------
def letterbox(img: np.ndarray, target: int = 512) -> np.ndarray:
    """
    Resize with aspect ratio to target x target by padding with black (letterbox).
    Accepts grayscale (H,W) or color (H,W,3).
    """
    if img.ndim == 2:
        h, w = img.shape
        channels = 1
    else:
        h, w = img.shape[:2]
        channels = img.shape[2]
    if h == 0 or w == 0:
        return np.zeros((target, target, channels), dtype=img.dtype) if channels != 1 else np.zeros((target, target), dtype=img.dtype)
    scale = min(target / w, target / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if channels == 1:
        canvas = np.zeros((target, target), dtype=img.dtype)
    else:
        canvas = np.zeros((target, target, channels), dtype=img.dtype)
    x0 = (target - nw) // 2
    y0 = (target - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def rotate_image_90(img: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate by angle in {0,90,180,270} degrees clockwise.
    """
    a = angle % 360
    if a == 0:
        return img.copy()
    # numpy.rot90 rotates CCW; convert to k steps CW
    k = {90: -1, 180: 2, 270: 1}.get(a, None)
    if k is None:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -a, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.rot90(img, k=k)

def multi_scale_match(
    full_bgr: np.ndarray,
    template_bgr: np.ndarray,
    scales: List[float] = None,
    method: int = cv2.TM_CCOEFF_NORMED
) -> Dict:
    """
    Multi-scale template matching in grayscale. Returns best match dict.
    """
    if scales is None:
        scales = [round(s, 3) for s in np.linspace(0.85, 1.15, 13)]
    full_gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    H, W = full_gray.shape
    th0, tw0 = tmpl_gray.shape
    best = {
        "score": -1.0, "scale": None, "x1": 0, "y1": 0, "x2": 0, "y2": 0,
        "w": 0, "h": 0, "method": "TM_CCOEFF_NORMED"
    }
    for s in scales:
        tw = int(round(tw0 * s))
        th = int(round(th0 * s))
        if tw < 8 or th < 8 or tw > W or th > H:
            continue
        tmpl_s = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        res = cv2.matchTemplate(full_gray, tmpl_s, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        score = float(max_val)
        if score > best["score"]:
            tlx, tly = max_loc
            best.update({
                "score": score, "scale": float(s),
                "x1": int(tlx), "y1": int(tly),
                "w": int(tw), "h": int(th),
                "x2": int(tlx + tw - 1), "y2": int(tly + th - 1)
            })
    return best

def draw_match(full_bgr: np.ndarray, match: Dict, out_path: str, rot_deg: int = 0) -> None:
    vis = full_bgr.copy()
    x1, y1, x2, y2 = match["x1"], match["y1"], match["x2"], match["y2"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 225, 0), thickness=3)
    label = f"score={match['score']:.3f} scale={match['scale']:.3f} rot={rot_deg}"
    cv2.putText(vis, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)

def crop_and_normalize_masks(masks_dir: str, out_dir: str, box: Tuple[int, int, int, int], size: int = 512, rot_deg: int = 0) -> int:
    """
    Crop all full-size masks to bbox (x1,y1,x2,y2 inclusive), inverse-rotate by rot_deg to match defect orientation,
    letterbox to size x size, and write to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    x1, y1, x2, y2 = box
    written = 0
    for entry in os.scandir(masks_dir):
        if not entry.is_file() or not entry.name.lower().endswith(".png"):
            continue
        m = cv2.imread(entry.path, cv2.IMREAD_GRAYSCALE)
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
        crop = (crop > 127).astype(np.uint8) * 255
        inv_rot = (-int(rot_deg)) % 360
        if inv_rot != 0:
            crop = rotate_image_90(crop, inv_rot)
        norm = letterbox(crop, target=size)
        out_path = os.path.join(out_dir, entry.name)
        cv2.imwrite(out_path, norm)
        written += 1
    return written

def find_best_match_with_rotations(full_bgr: np.ndarray, template_bgr: np.ndarray, rotations: List[int], scales: List[float]) -> Dict:
    """
    Try matching template rotated by each angle in rotations (degrees clockwise). Returns best match dict + 'rot_deg'.
    """
    best_overall = None
    for rot in rotations:
        tmpl_rot = rotate_image_90(template_bgr, rot)
        match = multi_scale_match(full_bgr, tmpl_rot, scales=scales)
        match["rot_deg"] = int(rot)
        if best_overall is None or match["score"] > best_overall["score"]:
            best_overall = match
    return best_overall

# --------------------
# IO and orchestration
# --------------------
def next_defect_index(class_defect_dir: str, class_name: str) -> int:
    """
    Find next index for subfolders named f"{class_name}_defect_<n>" under class_defect_dir.
    """
    if not os.path.isdir(class_defect_dir):
        return 1
    pat = re.compile(rf"^{re.escape(class_name)}_defect_(\d+)$")
    max_idx = 0
    for e in os.scandir(class_defect_dir):
        if e.is_dir():
            m = pat.match(e.name)
            if m:
                try:
                    max_idx = max(max_idx, int(m.group(1)))
                except ValueError:
                    pass
    return max_idx + 1

def gather_jpgs_recursive(root: str) -> List[str]:
    """
    Recursively collect .jpg/.jpeg files under root.
    """
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            fl = fn.lower()
            if fl.endswith(".jpg") or fl.endswith(".jpeg"):
                out.append(os.path.join(dirpath, fn))
    return out

def ensure_template_assets(root_with_label: str, class_name: str) -> Tuple[str, str]:
    """
    Returns (template_image_path, template_masks_dir). Raises if missing.
    """
    tmpl_img = os.path.join(root_with_label, class_name, f"{class_name}.png")
    tmpl_masks = os.path.join(root_with_label, class_name, "masks")
    if not os.path.isfile(tmpl_img):
        raise FileNotFoundError(f"Template image not found: {tmpl_img}")
    if not os.path.isdir(tmpl_masks):
        raise FileNotFoundError(f"Template masks folder not found: {tmpl_masks}")
    return tmpl_img, tmpl_masks

def process_one_defect(
    class_name: str,
    defect_jpg: str,
    full_png: str,
    full_masks_dir: str,
    out_class_defect_dir: str,
    normalize_size: int,
    rotations: List[int],
    scales: List[float],
    score_threshold: float,
    save_matched_vis: bool,
    random_rotate_sample: bool,
    rng: random.Random,
) -> Tuple[bool, str]:
    """
    Process one defect image; on success, create sample folder with aug_image.png, masks/, meta.json.
    Returns (accepted, out_dir or reason).
    """
    full_bgr = cv2.imread(full_png, cv2.IMREAD_COLOR)
    if full_bgr is None:
        return False, f"Cannot read template: {full_png}"
    defect_bgr = cv2.imread(defect_jpg, cv2.IMREAD_COLOR)
    if defect_bgr is None:
        return False, f"Cannot read defect: {defect_jpg}"

    best = find_best_match_with_rotations(full_bgr, defect_bgr, rotations=rotations, scales=scales)
    if best is None or float(best.get("score", -1)) < score_threshold:
        return False, f"score {best.get('score', -1):.4f} < {score_threshold}"

    # Create output subfolder
    idx = next_defect_index(out_class_defect_dir, class_name)
    out_dir = os.path.join(out_class_defect_dir, f"{class_name}_defect_{idx}")
    os.makedirs(out_dir, exist_ok=False)

    # Save aug_image.png (letterboxed defect image)
    aug_img = letterbox(defect_bgr, target=normalize_size)
    cv2.imwrite(os.path.join(out_dir, "aug_image.png"), aug_img)

    # Save masks (crop from full masks using best bbox, inverse-rotate, letterbox)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    written = crop_and_normalize_masks(
        full_masks_dir,
        os.path.join(out_dir, "masks"),
        (best["x1"], best["y1"], best["x2"], best["y2"]),
        size=normalize_size,
        rot_deg=int(best.get("rot_deg", 0))
    )

    # Optional matched visualization
    if save_matched_vis:
        draw_match(full_bgr, best, os.path.join(out_dir, "matched.png"), rot_deg=int(best.get("rot_deg", 0)))

    # Optional: random rotate the whole sample (aug_image + all masks) by 0/90/180/270 consistently
    random_rot_deg = 0
    if random_rotate_sample:
        random_rot_deg = rng.choice([0, 90, 180, 270])
        if random_rot_deg != 0:
            # Rotate aug_image.png
            ai_path = os.path.join(out_dir, "aug_image.png")
            ai = cv2.imread(ai_path, cv2.IMREAD_COLOR)
            if ai is not None:
                ai_rot = rotate_image_90(ai, random_rot_deg)
                cv2.imwrite(ai_path, ai_rot)
            # Rotate all masks
            mdir = os.path.join(out_dir, "masks")
            for e in os.scandir(mdir):
                if not e.is_file() or not e.name.lower().endswith(".png"):
                    continue
                m = cv2.imread(e.path, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                m_rot = rotate_image_90(m, random_rot_deg)
                cv2.imwrite(e.path, m_rot)

    # Save meta.json
    meta = {
        "defect": True,
        "class": class_name,
        "source_defect_image": os.path.abspath(defect_jpg),
        "template_image": os.path.abspath(full_png),
        "template_masks_dir": os.path.abspath(full_masks_dir),
        "normalize_size": int(normalize_size),
        "score_threshold": float(score_threshold),
        "match": {
            "method": best.get("method"),
            "score": float(best.get("score", 0.0)),
            "scale": float(best.get("scale", 1.0)),
            "rot_deg": int(best.get("rot_deg", 0)),
            "bbox": {
                "x1": int(best["x1"]), "y1": int(best["y1"]),
                "x2": int(best["x2"]), "y2": int(best["y2"]),
                "w": int(best["w"]), "h": int(best["h"])
            }
        }
    }
    if save_matched_vis:
        meta["matched_vis"] = "matched.png"
    if random_rotate_sample:
        meta["random_sample_rotation_deg"] = int(random_rot_deg)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True, out_dir

def build_scales(min_scale: float, max_scale: float, steps: int) -> List[float]:
    steps = max(2, int(steps))
    arr = np.linspace(min_scale, max_scale, steps)
    return [round(float(s), 3) for s in arr]

def main():
    ap = argparse.ArgumentParser(description="Batch import defect JPGs into defect VQA samples via template matching.")
    ap.add_argument("-r", "--root", default=".", help="with_label root directory (default: current dir)")
    ap.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="Root directory containing defect dataset folders.")
    ap.add_argument("--normalize-size", type=int, default=512, help="Output canvas size for aug_image and masks (default: 512).")
    ap.add_argument("--rotations", type=str, default="0,90,180,270", help="Comma list of rotations (deg CW) to try.")
    ap.add_argument("--min-scale", type=float, default=0.85, help="Min scale for multi-scale matching.")
    ap.add_argument("--max-scale", type=float, default=1.15, help="Max scale for multi-scale matching.")
    ap.add_argument("--scale-steps", type=int, default=13, help="Number of scales between min and max (inclusive).")
    ap.add_argument("--score-threshold", type=float, default=0.6, help="Reject matches below this score.")
    ap.add_argument("--save-matched-vis", action="store_true", help="Save matched.png visualization in each output sample.")
    ap.add_argument("--limit", type=int, default=0, help="Max accepted samples per dataset (0 = no cap).")
    ap.add_argument("--shuffle-seed", type=int, default=None, help="Optional seed for shuffling JPGs (for reproducibility).")
    ap.add_argument("--no-random-rotate-sample", dest="random_rotate_sample", action="store_false", default=True, help="Apply a random 0/90/180/270 rotation to aug_image and masks (default: on).")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    source_root = os.path.abspath(args.source_root)
    rotations = [int(x) for x in args.rotations.split(",") if x.strip()]
    scales = build_scales(args.min_scale, args.max_scale, args.scale_steps)
    rng = random.Random(args.shuffle_seed) if args.shuffle_seed is not None else random.Random()

    # Prepare output base: with_label/defect
    defect_root = os.path.join(root, "defect")
    os.makedirs(defect_root, exist_ok=True)

    # Group dataset folders present under source_root by class
    present_sets = {}
    for entry in os.scandir(source_root):
        if entry.is_dir():
            ds_name = entry.name
            if ds_name in DATASET_TO_CLASS:
                present_sets.setdefault(DATASET_TO_CLASS[ds_name], []).append(entry.path)

    if not present_sets:
        print("[INFO] No known defect dataset folders found under source root.")
        return

    # Stats
    total_processed = 0     # jpgs scanned
    total_accepted = 0      # samples created
    per_class_accept: Dict[str, int] = {cls: 0 for cls in present_sets.keys()}

    # Process per class
    for class_name, dataset_dirs in present_sets.items():
        try:
            tmpl_img, tmpl_masks = ensure_template_assets(root, class_name)
        except Exception as e:
            print(f"[SKIP] Class {class_name}: {e}")
            continue

        class_defect_dir = os.path.join(defect_root, f"{class_name}_defect")
        os.makedirs(class_defect_dir, exist_ok=True)

        class_processed = 0
        class_accepted = 0

        print(f"[CLASS] {class_name}: {len(dataset_dirs)} dataset folder(s).")

        for ds_path in dataset_dirs:
            jpgs = gather_jpgs_recursive(ds_path)
            if not jpgs:
                print(f"  [DATASET] {os.path.basename(ds_path)}: 0 jpgs found")
                continue

            # Shuffle order so we can cap accepted samples per dataset randomly
            rng.shuffle(jpgs)

            ds_processed = 0
            ds_accepted = 0

            for jp in jpgs:
                # Per-dataset accepted cap
                if args.limit and ds_accepted >= args.limit:
                    break

                accepted, info = process_one_defect(
                    class_name=class_name,
                    defect_jpg=jp,
                    full_png=tmpl_img,
                    full_masks_dir=tmpl_masks,
                    out_class_defect_dir=class_defect_dir,
                    normalize_size=args.normalize_size,
                    rotations=rotations,
                    scales=scales,
                    score_threshold=args.score_threshold,
                    save_matched_vis=args.save_matched_vis,
                    random_rotate_sample=args.random_rotate_sample,
                    rng=rng,
                )
                total_processed += 1
                class_processed += 1
                ds_processed += 1
                if accepted:
                    total_accepted += 1
                    class_accepted += 1
                    ds_accepted += 1
                    print(f"    {Fore.GREEN}[OK]{Style.RESET_ALL} {class_name} <- {os.path.relpath(jp, ds_path)} -> {os.path.basename(info)}  (score >= {args.score_threshold})")
                else:
                    print(f"    {Fore.RED}[REJ]{Style.RESET_ALL} {class_name} <- {os.path.relpath(jp, ds_path)} ({info})")

            print(f"  {Fore.CYAN}[DATASET]{Style.RESET_ALL} {os.path.basename(ds_path)}: processed {ds_processed}, accepted {ds_accepted}, rejected {ds_processed - ds_accepted}")

        per_class_accept[class_name] = class_accepted
        print(f"{Back.BLUE}[SUMMARY]{Style.RESET_ALL} {class_name}: processed {class_processed}, accepted {class_accepted}, rejected {class_processed - class_accepted}")

    # Totals
    print("\n[PER-CLASS TOTALS]")
    for cls, cnt in per_class_accept.items():
        print(f"  {cls}: {cnt} defect sample(s)")
    print(f"{Back.YELLOW}[TOTAL]{Style.RESET_ALL} Processed JPGs: {total_processed}; Accepted samples: {total_accepted}")

if __name__ == "__main__":
    main()