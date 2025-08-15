""" 
This script generates augmented images of the TFT circuit image sets. Transforming both RGB and mask
Order of operations: Geom(rotation, shear, stretch, flip), Photo(brightness, contrast, HSV, G noise, G blur), Crop
Run from with_label
"""

import os
import re
import json
import math
import random
import argparse
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np

# ============================
# Parameters
# ============================
DEFAULT_PARAMS = {
    # How many augmented sets to create per sample folder
    "num_aug_per_sample": 5,

    # Random seed (None for non-deterministic)
    "seed": None,

    # Geometric transforms (Both RGB and masks)
    "apply_rotate_prob": 0.8,            # Probability to apply rotation
    "max_rotate_deg": 15.0,              # Max absolute rotation angle in degrees

    "apply_shear_prob": 0.5,             # Probability to apply shear
    "max_shear_x": 0.15,                 # Max |kx| for horizontal shear
    "max_shear_y": 0.15,                 # Max |ky| for vertical shear

    "apply_stretch_prob": 0.5,           # Probability to apply anisotropic scaling (stretch)
    "max_stretch_x_delta": 0.15,         # sx in [1-delta, 1+delta]
    "max_stretch_y_delta": 0.15,         # sy in [1-delta, 1+delta]

    "apply_flip_h_prob": 0.5,            # Probability to apply horizontal flip
    "apply_flip_v_prob": 0.2,            # Probability to apply vertical flip

    # Photometric transforms (RGB only)
    "apply_brightness_contrast_prob": 0.7,
    "max_brightness_delta": 40,          # [-max, max]
    "max_contrast_delta": 0.4,           # alpha in [1-max, 1+max]

    "apply_color_jitter_prob": 0.5,
    "max_hue_delta": 10,                 # OpenCV hue is [0,179]; shift in [-max, max]
    "max_saturation_scale_delta": 0.4,   # scale in [1-max, 1+max]
    "max_value_scale_delta": 0.3,        # scale in [1-max, 1+max]

    "apply_gaussian_noise_prob": 0.3,
    "max_noise_std": 12.0,               # Gaussian noise sigma (0..max)

    "apply_blur_prob": 0.25,
    "max_blur_kernel": 5,                # max odd kernel size (use <=1 to disable)

    # Final crop/resize (last step). Width/height chosen independently.
    # If max_* is None, it will default to the original image dimension for that sample.
    "min_crop_width": 70,
    "min_crop_height": 70,
    "max_crop_width": 400,               # None = Defaults to original width
    "max_crop_height": 400,              # None = Defaults to original height
}

# ============================
# Utilities
# ============================

def list_sample_folders(root: str) -> List[str]:
    # Second-level folders under with_label (skip hidden)
    out = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith('.'):
            continue
        out.append(entry.path)
    return out

def find_rgb_png(sample_dir: str) -> Optional[str]:
    # Find a PNG in the sample folder (exclude masks/aug subfolders)
    pngs = [f.path for f in os.scandir(sample_dir)
            if f.is_file() and f.name.lower().endswith(".png")]
    if not pngs:
        return None
    return pngs[0]

def load_masks(mask_dir: str) -> Dict[str, np.ndarray]:
    masks = {}
    if not os.path.isdir(mask_dir):
        return masks
    for f in os.scandir(mask_dir):
        if f.is_file() and f.name.lower().endswith(".png"):
            mask = cv2.imread(f.path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            # Normalize to 0/255
            mask = (mask > 127).astype(np.uint8) * 255
            masks[f.name] = mask
    return masks

def next_aug_start_index(aug_dir: str, base_name: str) -> int:
    # Find max <n> among subfolders named f"{base_name}_{n}"
    if not os.path.isdir(aug_dir):
        return 1
    pat = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    max_idx = 0
    for entry in os.scandir(aug_dir):
        if not entry.is_dir():
            continue
        m = pat.match(entry.name)
        if m:
            try:
                idx = int(m.group(1))
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return max_idx + 1

def odd_kernel(k: int) -> int:
    if k <= 1:
        return 1
    return k if k % 2 == 1 else k - 1

def compose_affine_mats(m2x3_a: np.ndarray, m2x3_b: np.ndarray) -> np.ndarray:
    # Return A ∘ B (apply B then A), both as 2x3
    A = np.vstack([m2x3_a, [0, 0, 1]])
    B = np.vstack([m2x3_b, [0, 0, 1]])
    C = A @ B
    return C[:2, :]

def mat2x3_from_3x3(m3: np.ndarray) -> np.ndarray:
    return m3[:2, :]

def make_center_flip_h(w: int, h: int) -> np.ndarray:
    cx, cy = w / 2.0, h / 2.0
    return np.array([[-1, 0, 2 * cx],
                     [ 0, 1, 0     ]], dtype=np.float32)

def make_center_flip_v(w: int, h: int) -> np.ndarray:
    cx, cy = w / 2.0, h / 2.0
    return np.array([[ 1, 0, 0     ],
                     [ 0,-1, 2 * cy]], dtype=np.float32)

def make_center_rotation(w: int, h: int, angle_deg: float) -> np.ndarray:
    cx, cy = w / 2.0, h / 2.0
    return cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0).astype(np.float32)

def make_center_shear(w: int, h: int, kx: float, ky: float) -> np.ndarray:
    # Centered shear using 3x3 composition: T(cx,cy) * Shear * T(-cx,-cy)
    cx, cy = w / 2.0, h / 2.0
    T1 = np.array([[1, 0,  cx],
                   [0, 1,  cy],
                   [0, 0,   1]], dtype=np.float32)
    Sh = np.array([[1, kx, 0],
                   [ky, 1, 0],
                   [0,  0, 1]], dtype=np.float32)
    T2 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0,   1]], dtype=np.float32)
    M3 = T1 @ Sh @ T2
    return mat2x3_from_3x3(M3.astype(np.float32))

def make_center_stretch(w: int, h: int, sx: float, sy: float) -> np.ndarray:
    # Centered anisotropic scaling
    cx, cy = w / 2.0, h / 2.0
    T1 = np.array([[1, 0,  cx],
                   [0, 1,  cy],
                   [0, 0,   1]], dtype=np.float32)
                   
    Sc = np.array([[sx, 0, 0],
                   [0, sy, 0],
                   [0,  0, 1]], dtype=np.float32)
    T2 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0,   1]], dtype=np.float32)
    M3 = T1 @ Sc @ T2
    return mat2x3_from_3x3(M3.astype(np.float32))

def apply_affine(img: np.ndarray, m2x3: np.ndarray, is_mask: bool) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border_mode = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101
    border_value = 0
    h, w = img.shape[:2]
    warped = cv2.warpAffine(img, m2x3, (w, h), flags=interp, borderMode=border_mode, borderValue=border_value)
    if is_mask:
        warped = (warped > 127).astype(np.uint8) * 255
    return warped

def random_geometric(w: int, h: int, params: Dict, rng: random.Random) -> Tuple[np.ndarray, Dict]:
    """
    Compose geometric transforms in the following order:
    rotation -> shear -> stretch (anisotropic scale) -> flips
    """
    M = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
    meta = {
        "rotation_deg": 0.0,
        "shear_kx": 0.0,
        "shear_ky": 0.0,
        "stretch_sx": 1.0,
        "stretch_sy": 1.0,
        "flip_h": False,
        "flip_v": False
    }

    # Rotation
    if rng.random() < params["apply_rotate_prob"]:
        angle = rng.uniform(-params["max_rotate_deg"], params["max_rotate_deg"])
        R = make_center_rotation(w, h, angle)
        M = compose_affine_mats(R, M)
        meta["rotation_deg"] = angle

    # Shear
    if rng.random() < params["apply_shear_prob"]:
        kx = rng.uniform(-params["max_shear_x"], params["max_shear_x"])
        ky = rng.uniform(-params["max_shear_y"], params["max_shear_y"])
        Sh = make_center_shear(w, h, kx, ky)
        M = compose_affine_mats(Sh, M)
        meta["shear_kx"] = kx
        meta["shear_ky"] = ky

    # Stretch (anisotropic scaling)
    if rng.random() < params["apply_stretch_prob"]:
        sx = 1.0 + rng.uniform(-params["max_stretch_x_delta"], params["max_stretch_x_delta"])
        sy = 1.0 + rng.uniform(-params["max_stretch_y_delta"], params["max_stretch_y_delta"])
        # avoid non-positive scales
        sx = max(sx, 0.05)
        sy = max(sy, 0.05)
        Sc = make_center_stretch(w, h, sx, sy)
        M = compose_affine_mats(Sc, M)
        meta["stretch_sx"] = sx
        meta["stretch_sy"] = sy

    # Flips
    if rng.random() < params["apply_flip_h_prob"]:
        FH = make_center_flip_h(w, h)
        M = compose_affine_mats(FH, M)
        meta["flip_h"] = True

    if rng.random() < params["apply_flip_v_prob"]:
        FV = make_center_flip_v(w, h)
        M = compose_affine_mats(FV, M)
        meta["flip_v"] = True

    return M, meta

def apply_photometric(img_bgr: np.ndarray, params: Dict, rng: random.Random) -> Tuple[np.ndarray, Dict]:
    meta = {}
    out = img_bgr.copy()

    # Brightness/contrast
    if rng.random() < params["apply_brightness_contrast_prob"]:
        beta = rng.uniform(-params["max_brightness_delta"], params["max_brightness_delta"])
        delta = rng.uniform(-params["max_contrast_delta"], params["max_contrast_delta"])
        alpha = 1.0 + delta
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
        meta["brightness_beta"] = beta
        meta["contrast_alpha"] = alpha

    # Color jitter (HSV)
    if rng.random() < params["apply_color_jitter_prob"]:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_, s_, v_ = cv2.split(hsv)
        hue_shift = rng.uniform(-params["max_hue_delta"], params["max_hue_delta"])
        sat_scale = 1.0 + rng.uniform(-params["max_saturation_scale_delta"], params["max_saturation_scale_delta"])
        val_scale = 1.0 + rng.uniform(-params["max_value_scale_delta"], params["max_value_scale_delta"])
        h_ = (h_ + hue_shift) % 180.0
        s_ = np.clip(s_ * sat_scale, 0, 255)
        v_ = np.clip(v_ * val_scale, 0, 255)
        hsv = cv2.merge([h_, s_, v_]).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        meta["hue_shift"] = hue_shift
        meta["saturation_scale"] = sat_scale
        meta["value_scale"] = val_scale

    # Gaussian noise
    if rng.random() < params["apply_gaussian_noise_prob"]:
        std = rng.uniform(0.0, params["max_noise_std"])
        noise_img = np.random.normal(0, std, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise_img, 0, 255).astype(np.uint8)
        meta["gaussian_noise_std"] = std

    # Blur
    if rng.random() < params["apply_blur_prob"] and params["max_blur_kernel"] and params["max_blur_kernel"] > 1:
        kmax = odd_kernel(params["max_blur_kernel"])
        k = int(rng.choice([ks for ks in range(3, kmax + 1, 2)])) if kmax >= 3 else 1
        if k > 1:
            out = cv2.GaussianBlur(out, (k, k), 0)
            meta["gaussian_blur_kernel"] = k

    return out, meta

def choose_target_size(orig_w: int, orig_h: int, params: Dict, rng: random.Random) -> Tuple[int, int]:
    max_w = params["max_crop_width"] if params["max_crop_width"] is not None else orig_w
    max_h = params["max_crop_height"] if params["max_crop_height"] is not None else orig_h
    min_w = max(1, int(params["min_crop_width"]))
    min_h = max(1, int(params["min_crop_height"]))
    max_w = max(min_w, int(max_w))
    max_h = max(min_h, int(max_h))
    tw = rng.randint(min_w, max_w)
    th = rng.randint(min_h, max_h)
    return tw, th

def crop_or_resize_last(img: np.ndarray, masks: Dict[str, np.ndarray], target_w: int, target_h: int,
                        rng: random.Random, geom_M: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Cropping/resizing is done last. If current >= target, random crop; else resize to target.
    Returns augmented img, masks, and metadata including crop info (with original-image coords).
    """
    h, w = img.shape[:2]
    meta = {"final_width": target_w, "final_height": target_h, "crop_applied": False}

    if w >= target_w and h >= target_h:
        x0 = rng.randint(0, w - target_w)
        y0 = rng.randint(0, h - target_h)
        x1, y1 = x0 + target_w, y0 + target_h

        cropped_img = img[y0:y1, x0:x1]
        cropped_masks = {name: m[y0:y1, x0:x1] for name, m in masks.items()}

        # Map crop rectangle back to original image coordinates using inverse of geom_M
        M2 = np.vstack([geom_M, [0, 0, 1]])  # 3x3
        Minv = np.linalg.inv(M2)
        corners = np.array([
            [x0, y0, 1.0],
            [x1, y0, 1.0],
            [x1, y1, 1.0],
            [x0, y1, 1.0],
        ], dtype=np.float32).T  # 3x4
        mapped = Minv @ corners
        mapped = (mapped[:2, :].T).tolist()  # list of [x,y] in original image coords (floats)

        meta["crop_applied"] = True
        meta["crop_box_post_geom"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        meta["crop_quad_in_original"] = mapped  # 4 points in original image coords (tl,tr,br,bl)

        return cropped_img, cropped_masks, meta
    else:
        resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_masks = {name: cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST) for name, m in masks.items()}
        meta["crop_applied"] = False
        meta["crop_box_post_geom"] = None
        meta["crop_quad_in_original"] = None
        return resized_img, resized_masks, meta

def run_one_augmentation(
    rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
    params: Dict,
    rng: random.Random
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Apply geometric (consistent) + photometric (image only) + final crop/resize.
    Returns augmented image, augmented masks, and metadata.
    """
    H, W = rgb.shape[:2]
    metadata = {
        "original_width": W,
        "original_height": H,
        "transforms": {},
        "parameters_used": params.copy(),
    }

    # Geometric
    M_geom, meta_geom = random_geometric(W, H, params, rng)
    metadata["transforms"]["geometric"] = meta_geom

    rgb_geom = apply_affine(rgb, M_geom, is_mask=False)
    masks_geom = {name: apply_affine(m, M_geom, is_mask=True) for name, m in masks.items()}

    # Photometric (RGB only)
    rgb_photo, meta_photo = apply_photometric(rgb_geom, params, rng)
    metadata["transforms"]["photometric"] = meta_photo

    # Final crop/resize
    target_w, target_h = choose_target_size(W, H, params, rng)
    rgb_final, masks_final, meta_crop = crop_or_resize_last(rgb_photo, masks_geom, target_w, target_h, rng, M_geom)
    metadata["transforms"]["final_crop_resize"] = meta_crop

    metadata["new_width"] = int(rgb_final.shape[1])
    metadata["new_height"] = int(rgb_final.shape[0])

    return rgb_final, masks_final, metadata

def save_aug_set(out_dir: str, base_name: str, rgb: np.ndarray, masks: Dict[str, np.ndarray], metadata: Dict):
    os.makedirs(out_dir, exist_ok=True)
    # Save image
    img_path = os.path.join(out_dir, "aug_image.png")
    cv2.imwrite(img_path, rgb)
    # Save masks
    mdir = os.path.join(out_dir, "masks")
    os.makedirs(mdir, exist_ok=True)
    for fname, m in masks.items():
        cv2.imwrite(os.path.join(mdir, fname), m)
    # Save metadata
    meta_path = os.path.join(out_dir, "meta.json")
    metadata["original_image"] = base_name
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def process_sample(sample_dir: str, params: Dict, rng: random.Random) -> int:
    rgb_path = find_rgb_png(sample_dir)
    if not rgb_path:
        print(f"[SKIP] No PNG image found in {sample_dir}")
        return 0

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        print(f"[SKIP] Cannot read image: {rgb_path}")
        return 0

    H, W = rgb.shape[:2]
    mask_dir = os.path.join(sample_dir, "masks")
    masks = load_masks(mask_dir)
    if not masks:
        print(f"[SKIP] No masks found in {mask_dir}")
        return 0

    # Resolve max crop defaults against this sample's original size
    if params["max_crop_width"] is None:
        params["max_crop_width"] = W
    if params["max_crop_height"] is None:
        params["max_crop_height"] = H

    aug_dir = os.path.join(sample_dir, "aug")
    start_idx = next_aug_start_index(aug_dir, os.path.basename(sample_dir))

    generated = 0
    total_to_generate = params["num_aug_per_sample"]
    for i in range(total_to_generate):
        rgb_aug, masks_aug, meta = run_one_augmentation(rgb, masks, params, rng)
        seq_idx = start_idx + i
        sub_name = f"{os.path.basename(sample_dir)}_{seq_idx}"
        out_dir = os.path.join(aug_dir, sub_name)
        save_aug_set(out_dir, os.path.basename(rgb_path), rgb_aug, masks_aug, meta)
        generated += 1
        print(f"[OK] {sample_dir} -> {sub_name} ({generated}/{total_to_generate})")

    print(f"[SUMMARY] {sample_dir}: generated {generated} augmented set(s).")
    return generated
# ...existing code...
def main():
    parser = argparse.ArgumentParser(description="Augment TFT circuit images and masks (run from with_label).")
    parser.add_argument("-r", "--root", default=".", help="Root directory (with_label). Default: current dir")
    parser.add_argument("-n", "--num", type=int, help="Number of augmented sets per sample (overrides default).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    # Optional overrides for key params
    parser.add_argument("--max-rotate", type=float, help="Max absolute rotation degrees.")
    parser.add_argument("--min-w", type=int, help="Min crop width.")
    parser.add_argument("--min-h", type=int, help="Min crop height.")
    parser.add_argument("--max-w", type=int, help="Max crop width (default: original width per sample).")
    parser.add_argument("--max-h", type=int, help="Max crop height (default: original height per sample).")
    # Shear/stretch overrides
    parser.add_argument("--max-shear-x", type=float, help="Max absolute shear kx.")
    parser.add_argument("--max-shear-y", type=float, help="Max absolute shear ky.")
    parser.add_argument("--max-stretch-x", type=float, help="Max stretch delta for sx (sx in [1-d,1+d]).")
    parser.add_argument("--max-stretch-y", type=float, help="Max stretch delta for sy (sy in [1-d,1+d]).")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.num is not None:
        params["num_aug_per_sample"] = int(args.num)
    if args.max_rotate is not None:
        params["max_rotate_deg"] = float(args.max_rotate)
    if args.min_w is not None:
        params["min_crop_width"] = int(args.min_w)
    if args.min_h is not None:
        params["min_crop_height"] = int(args.min_h)
    if args.max_w is not None:
        params["max_crop_width"] = int(args.max_w)
    if args.max_h is not None:
        params["max_crop_height"] = int(args.max_h)
    if args.max_shear_x is not None:
        params["max_shear_x"] = float(args.max_shear_x)
    if args.max_shear_y is not None:
        params["max_shear_y"] = float(args.max_shear_y)
    if args.max_stretch_x is not None:
        params["max_stretch_x_delta"] = float(args.max_stretch_x)
    if args.max_stretch_y is not None:
        params["max_stretch_y_delta"] = float(args.max_stretch_y)

    # Seed
    seed = args.seed if args.seed is not None else params["seed"]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = random.Random(seed)

    root = args.root
    total_aug = 0
    samples_with_output = 0
    for sample_dir in list_sample_folders(root):
        # Reset per-sample max to default None so they can be set to per-image dims inside process_sample
        tmp_params = params.copy()
        tmp_params["max_crop_width"] = params["max_crop_width"]
        tmp_params["max_crop_height"] = params["max_crop_height"]
        gen = process_sample(sample_dir, tmp_params, rng)
        total_aug += gen
        if gen > 0:
            samples_with_output += 1

    print(f"[TOTAL] Generated {total_aug} augmented set(s) across {samples_with_output} sample folder(s).")

if __name__ == "__main__":
    main()