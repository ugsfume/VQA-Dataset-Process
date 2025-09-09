"""
Per-sample processing and progress reporting.
"""

from typing import Dict
import os

from dataset import (
    list_sample_folders,
    find_rgb_png,
    load_masks,
    next_aug_start_index,
    save_aug_set,
)
from augmentor import Augmentor
from colorama import Fore, Back, Style

OK = Fore.GREEN + "[OK]" + Style.RESET_ALL
SKIP = Fore.RED + "[SKIP]" + Style.RESET_ALL
SUMMARY = Back.BLUE + "[SUMMARY]" + Style.RESET_ALL

def process_sample(sample_dir: str, params: Dict, rng) -> int:
    rgb_path = find_rgb_png(sample_dir)
    if not rgb_path:
        print(f"{SKIP} No PNG image found in {sample_dir}")
        return 0

    import cv2
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        print(f"    {SKIP} Cannot read image: {rgb_path}")
        return 0

    mask_dir = os.path.join(sample_dir, "masks")
    masks = load_masks(mask_dir)
    if not masks:
        print(f"{SKIP} No masks found in {mask_dir}")
        return 0

    # Resolve per-sample max crop against this sample's original size
    H, W = rgb.shape[:2]
    if params["max_crop_width"] is None:
        params["max_crop_width"] = W
    if params["max_crop_height"] is None:
        params["max_crop_height"] = H

    aug_dir = os.path.join(sample_dir, "aug")
    start_idx = next_aug_start_index(aug_dir, os.path.basename(sample_dir))

    augmentor = Augmentor(params, rng)

    generated = 0
    total_to_generate = params["num_aug_per_sample"]
    for i in range(total_to_generate):
        rgb_aug, masks_aug, meta = augmentor.augment_once(rgb, masks)
        seq_idx = start_idx + i
        sub_name = f"{os.path.basename(sample_dir)}_{seq_idx}"
        out_dir = os.path.join(aug_dir, sub_name)
        save_aug_set(out_dir, os.path.basename(rgb_path), rgb_aug, masks_aug, meta)
        generated += 1
        print(f"    {OK} {sample_dir} -> {sub_name} ({generated}/{total_to_generate})")

    print(f"{SUMMARY} {sample_dir}: generated {generated} augmented set(s).")
    return generated