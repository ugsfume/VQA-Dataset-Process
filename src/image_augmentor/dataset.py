"""
Dataset utilities: scan folders, load/save RGB and masks, naming.
"""

from typing import Dict, List, Optional
import os
import re
import cv2
import numpy as np
import json

def list_sample_folders(root: str) -> List[str]:
    # Second-level folders under with_label (skip hidden)
    out = []
    for entry in os.scandir(root):
        if entry.is_dir() and not entry.name.startswith('.'):
            out.append(entry.path)
    return out

def find_rgb_png(sample_dir: str) -> Optional[str]:
    # Find first PNG in the sample folder (exclude masks/aug subfolders)
    for f in os.scandir(sample_dir):
        if f.is_file() and f.name.lower().endswith(".png"):
            return f.path
    return None

def load_masks(mask_dir: str) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
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
        if entry.is_dir():
            m = pat.match(entry.name)
            if m:
                try:
                    idx = int(m.group(1))
                    max_idx = max(max_idx, idx)
                except ValueError:
                    pass
    return max_idx + 1

def save_aug_set(out_dir: str, base_image_name: str, rgb: np.ndarray, masks: Dict[str, np.ndarray], metadata: Dict):
    os.makedirs(out_dir, exist_ok=True)
    # Save RGB
    img_path = os.path.join(out_dir, "aug_image.png")
    cv2.imwrite(img_path, rgb)
    # Save masks
    mdir = os.path.join(out_dir, "masks")
    os.makedirs(mdir, exist_ok=True)
    for fname, m in masks.items():
        cv2.imwrite(os.path.join(mdir, fname), m)
    # Save metadata
    meta_path = os.path.join(out_dir, "meta.json")
    metadata["original_image"] = base_image_name
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)