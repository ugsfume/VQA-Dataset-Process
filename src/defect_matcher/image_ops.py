"""
Image ops: letterbox and 0/90/180/270 rotations.
"""
import cv2
import numpy as np

def letterbox(img: np.ndarray, target: int = 512) -> np.ndarray:
    """
    Keep aspect ratio; fit into target x target with black padding.
    Works for gray (H,W) and color (H,W,3).
    """
    if img is None:
        raise ValueError("letterbox: img is None")
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
    Rotate by 0/90/180/270 degrees clockwise. Uses fast numpy.rot90 where possible.
    """
    a = angle % 360
    if a == 0:
        return img.copy()
    k = {90: -1, 180: 2, 270: 1}.get(a)
    if k is None:
        # Fallback for non-right angles (not used in this pipeline)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -a, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.rot90(img, k=k)