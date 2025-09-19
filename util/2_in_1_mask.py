#!/usr/bin/env python3
"""
2_in_1_mask.py — Combine two component masks (black/white) by overlaying whites (logical OR).

Default inputs: TFT1.png and TFT2.png in the current directory.
Default output: TFT.png (change with -o/--output).

Example:
  python 2_in_1_mask.py
  python 2_in_1_mask.py --img1 A.png --img2 B.png -o merged.png
  python 2_in_1_mask.py --use-otsu  # robust binarization if inputs aren't strictly 0/255
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path


def read_grayscale(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"[ERR] Could not read image: {path}")
    return img


def to_binary(gray: np.ndarray, threshold: int = 127, use_otsu: bool = False) -> np.ndarray:
    if use_otsu:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = max(0, min(255, int(threshold)))
        _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return bw


def main():
    p = argparse.ArgumentParser(description="Combine two masks by union (overlay white pixels). White=foreground.")
    p.add_argument("--img1", default="TFT1.png", help="Path to first mask image (default: TFT1.png)")
    p.add_argument("--img2", default="TFT2.png", help="Path to second mask image (default: TFT2.png)")
    p.add_argument("-o", "--output", default="TFT.png", help="Output image path (default: TFT.png)")
    p.add_argument("--threshold", type=int, default=127, help="Binarization threshold (0-255). Ignored if --use-otsu.")
    p.add_argument("--use-otsu", action="store_true", help="Use Otsu's method for binarization.")
    args = p.parse_args()

    img1_path = Path(args.img1)
    img2_path = Path(args.img2)
    out_path = Path(args.output)

    g1 = read_grayscale(img1_path)
    g2 = read_grayscale(img2_path)

    if g1.shape != g2.shape:
        sys.exit(f"[ERR] Resolution mismatch: {img1_path} is {g1.shape[::-1]}, {img2_path} is {g2.shape[::-1]}.")

    b1 = to_binary(g1, args.threshold, args.use_otsu)
    b2 = to_binary(g2, args.threshold, args.use_otsu)

    # Union / overlay whites
    merged = cv2.bitwise_or(b1, b2)

    if not cv2.imwrite(str(out_path), merged):
        sys.exit(f"[ERR] Failed to save output image: {out_path}")

    white_pixels = int(np.count_nonzero(merged == 255))
    total_pixels = merged.size
    print(f"[OK] Saved: {out_path}")
    print(f"[INFO] Foreground (white) pixels: {white_pixels}/{total_pixels} ({white_pixels/total_pixels:.2%})")


if __name__ == "__main__":
    main()
