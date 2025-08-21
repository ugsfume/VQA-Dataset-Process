"""
Mask ops: crop full masks to bbox, inverse-rotate, letterbox, and save.
"""
import os
from typing import Tuple
import cv2
import numpy as np

from image_ops import letterbox, rotate_image_90

def crop_and_normalize_masks(masks_dir: str, out_dir: str, box: Tuple[int, int, int, int], size: int = 512, rot_deg: int = 0) -> int:
    """
    Crop masks to bbox (inclusive), inverse-rotate by rot_deg, letterbox to size, and write to out_dir.
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