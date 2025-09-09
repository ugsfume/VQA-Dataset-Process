"""
Augmentor class orchestrating the pipeline for a single augmentation.
"""

from typing import Dict, Tuple
import numpy as np
import random

from transforms import (
    apply_affine,
    random_geometric,
    apply_photometric,
    choose_target_size,
    crop_or_resize_last,
)

class Augmentor:
    """
    Runs: Geometric (rotate->shear->stretch->flips) -> Photometric (RGB only) -> Crop/Resize (last).
    """

    def __init__(self, params: Dict, rng: random.Random):
        self.params = params
        self.rng = rng

    def augment_once(self, rgb: np.ndarray, masks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
        H, W = rgb.shape[:2]
        metadata: Dict = {
            "original_width": W,
            "original_height": H,
            "transforms": {},
            "parameters_used": self.params.copy(),
            "defect": False,  
        }

        # Geometric (consistent RGB+masks)
        M_geom, meta_geom = random_geometric(W, H, self.params, self.rng)
        metadata["transforms"]["geometric"] = meta_geom

        rgb_geom = apply_affine(rgb, M_geom, is_mask=False)
        masks_geom = {name: apply_affine(m, M_geom, is_mask=True) for name, m in masks.items()}

        # Photometric (RGB only)
        rgb_photo, meta_photo = apply_photometric(rgb_geom, self.params, self.rng)
        metadata["transforms"]["photometric"] = meta_photo

        # Final crop/resize
        target_w, target_h = choose_target_size(W, H, self.params, self.rng)
        rgb_final, masks_final, meta_crop = crop_or_resize_last(rgb_photo, masks_geom, target_w, target_h, self.rng, M_geom)
        metadata["transforms"]["final_crop_resize"] = meta_crop

        metadata["new_width"] = int(rgb_final.shape[1])
        metadata["new_height"] = int(rgb_final.shape[0])

        return rgb_final, masks_final, metadata