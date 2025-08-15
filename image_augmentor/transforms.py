"""
Geometric and photometric transforms + helpers.
"""

from typing import Dict, Tuple
import numpy as np
import cv2
import random

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
    # Masks use nearest and constant(0); RGB uses bilinear + reflect borders to avoid artifacts.
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
    Order: rot90 (discrete) -> small-angle rotation -> shear -> stretch -> flips.
    Returns 2x3 affine matrix and metadata.
    """
    M = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
    meta = {
        "rot90_deg": 0,               # 0, 90, 180, 270
        "small_rotation_deg": 0.0,    # small angle
        "shear_kx": 0.0,
        "shear_ky": 0.0,
        "stretch_sx": 1.0,
        "stretch_sy": 1.0,
        "flip_h": False,
        "flip_v": False
    }

    # 1) Discrete 90*n rotation
    if rng.random() < params["apply_rot90_prob"]:
        choices = params.get("rot90_choices", [90, 180, 270])
        if choices:
            ang90 = int(rng.choice(choices))
            R90 = make_center_rotation(w, h, float(ang90))
            M = compose_affine_mats(R90, M)
            meta["rot90_deg"] = ang90

    # 2) Small-angle rotation
    if rng.random() < params["apply_small_rotate_prob"]:
        max_deg = float(params["max_small_rotate_deg"])
        angle = rng.uniform(-max_deg, max_deg)
        R = make_center_rotation(w, h, angle)
        M = compose_affine_mats(R, M)
        meta["small_rotation_deg"] = angle

    # 3) Shear
    if rng.random() < params["apply_shear_prob"]:
        kx = rng.uniform(-params["max_shear_x"], params["max_shear_x"])
        ky = rng.uniform(-params["max_shear_y"], params["max_shear_y"])
        Sh = make_center_shear(w, h, kx, ky)
        M = compose_affine_mats(Sh, M)
        meta["shear_kx"] = kx
        meta["shear_ky"] = ky

    # 4) Stretch (anisotropic scaling)
    if rng.random() < params["apply_stretch_prob"]:
        sx = 1.0 + rng.uniform(-params["max_stretch_x_delta"], params["max_stretch_x_delta"])
        sy = 1.0 + rng.uniform(-params["max_stretch_y_delta"], params["max_stretch_y_delta"])
        sx = max(sx, 0.05)
        sy = max(sy, 0.05)
        Sc = make_center_stretch(w, h, sx, sy)
        M = compose_affine_mats(Sc, M)
        meta["stretch_sx"] = sx
        meta["stretch_sy"] = sy

    # 5) Flips
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
    meta: Dict = {}
    out = img_bgr.copy()

    # 6) Brightness/contrast
    if rng.random() < params["apply_brightness_contrast_prob"]:
        beta = rng.uniform(-params["max_brightness_delta"], params["max_brightness_delta"])
        delta = rng.uniform(-params["max_contrast_delta"], params["max_contrast_delta"])
        alpha = 1.0 + delta
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
        meta["brightness_beta"] = beta
        meta["contrast_alpha"] = alpha

    # 7) Color jitter (HSV)
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

    # 8) Gaussian noise
    if rng.random() < params["apply_gaussian_noise_prob"]:
        std = rng.uniform(0.0, params["max_noise_std"])
        noise_img = np.random.normal(0, std, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise_img, 0, 255).astype(np.uint8)
        meta["gaussian_noise_std"] = std

    # 9) Blur
    if rng.random() < params["apply_blur_prob"] and params["max_blur_kernel"] and params["max_blur_kernel"] > 1:
        kmax = odd_kernel(params["max_blur_kernel"])
        choices = [ks for ks in range(3, kmax + 1, 2)]
        if choices:
            k = int(rng.choice(choices))
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
    Also returns crop metadata and crop mapped back to original via inverse of geom_M.
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
        mapped = (mapped[:2, :].T).tolist()  # list of [x,y] in original image coords

        meta["crop_applied"] = True
        meta["crop_box_post_geom"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        meta["crop_quad_in_original"] = mapped
        return cropped_img, cropped_masks, meta
    else:
        resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_masks = {name: cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST) for name, m in masks.items()}
        meta["crop_applied"] = False
        meta["crop_box_post_geom"] = None
        meta["crop_quad_in_original"] = None
        return resized_img, resized_masks, meta