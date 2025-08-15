"""
Parameter defaults and CLI parsing.
"""

import argparse
import random
import numpy as np
from typing import Dict, Tuple

DEFAULT_PARAMS: Dict = {
    # How many augmented sets to create per sample folder
    "num_aug_per_sample": 5,

    # Random seed (None for non-deterministic)
    "seed": None,

    # Geometric transforms (Both RGB and masks)
    # Discrete rotation first (angles in degrees; any values allowed)
    "apply_rot90_prob": 0.7,
    "rot90_choices": [90, 180, 270],

    # Small-angle rotation next
    "apply_small_rotate_prob": 0.0,
    "max_small_rotate_deg": 15.0,

    "apply_shear_prob": 0.0,             # Probability to apply shear
    "max_shear_x": 0.15,                 # Max |kx| for horizontal shear
    "max_shear_y": 0.15,                 # Max |ky| for vertical shear

    "apply_stretch_prob": 0.0,           # Probability to apply anisotropic scaling (stretch)
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
    "min_crop_width": 512,
    "min_crop_height": 512,
    "max_crop_width": 512,              # None = Defaults to original width
    "max_crop_height": 512,             # None = Defaults to original height
}

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Augment TFT circuit images and masks (run from with_label).")
    p.add_argument("-r", "--root", default=".", help="Root directory (with_label). Default: current dir")
    p.add_argument("-n", "--num", type=int, help="Number of augmented sets per sample (overrides default).")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    # Geometric overrides
    p.add_argument("--rot90-prob", type=float, help="Probability of applying a discrete rotation.")
    p.add_argument("--rot90-choices", type=str, help="Comma-separated rotation angles in degrees (e.g. 90,180,270 or any angles).")
    p.add_argument("--small-rotate-prob", type=float, help="Probability of applying small-angle rotation.")
    p.add_argument("--max-small-rotate", type=float, help="Max abs degrees for small-angle rotation.")
    p.add_argument("--max-shear-x", type=float, help="Max absolute shear kx.")
    p.add_argument("--max-shear-y", type=float, help="Max absolute shear ky.")
    p.add_argument("--max-stretch-x", type=float, help="Max stretch delta for sx (sx in [1-d,1+d]).")
    p.add_argument("--max-stretch-y", type=float, help="Max stretch delta for sy (sy in [1-d,1+d]).")
    # Crop size overrides
    p.add_argument("--min-w", type=int, help="Min crop width.")
    p.add_argument("--min-h", type=int, help="Min crop height.")
    p.add_argument("--max-w", type=int, help="Max crop width (default: original width per sample).")
    p.add_argument("--max-h", type=int, help="Max crop height (default: original height per sample).")
    return p

def load_params_from_args(args: argparse.Namespace) -> Dict:
    params = DEFAULT_PARAMS.copy()
    if args.num is not None:
        params["num_aug_per_sample"] = int(args.num)

    # Discrete rotation
    if args.rot90_prob is not None:
        params["apply_rot90_prob"] = float(args.rot90_prob)
    if args.rot90_choices:
        try:
            params["rot90_choices"] = [int(x) for x in args.rot90_choices.split(",") if x.strip()]
        except Exception:
            pass
            
    # Small-angle rotation
    if args.small_rotate_prob is not None:
        params["apply_small_rotate_prob"] = float(args.small_rotate_prob)
    if args.max_small_rotate is not None:
        params["max_small_rotate_deg"] = float(args.max_small_rotate)

    # Shear/stretch
    if args.max_shear_x is not None:
        params["max_shear_x"] = float(args.max_shear_x)
    if args.max_shear_y is not None:
        params["max_shear_y"] = float(args.max_shear_y)
    if args.max_stretch_x is not None:
        params["max_stretch_x_delta"] = float(args.max_stretch_x)
    if args.max_stretch_y is not None:
        params["max_stretch_y_delta"] = float(args.max_stretch_y)

    # Crop
    if args.min_w is not None:
        params["min_crop_width"] = int(args.min_w)
    if args.min_h is not None:
        params["min_crop_height"] = int(args.min_h)
    if args.max_w is not None:
        params["max_crop_width"] = int(args.max_w)
    if args.max_h is not None:
        params["max_crop_height"] = int(args.max_h)
    params["seed"] = args.seed if args.seed is not None else params["seed"]
    return params

def make_rng(seed: int | None) -> random.Random:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    return random.Random(seed)