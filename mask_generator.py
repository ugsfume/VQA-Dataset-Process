"""
This script processes the json files in each sample and generates the corresponding mask images. 
Run this from with_label
"""

import os
import json
import argparse
import numpy as np
import cv2

INVALID_FS_CHARS = set('<>:"/\\|?*')

def sanitize_label(label: str) -> str:
    # Keep Unicode, replace filesystem-invalid chars with underscore
    return "".join('_' if ch in INVALID_FS_CHARS else ch for ch in label).strip()

def polygon_to_contour(points, width, height):
    # Points are [[x, y], ...]; convert to int32 contour and clip to bounds
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Polygon points must be of shape (N, 2)")
    arr = np.rint(arr)  # round to nearest int
    arr[:, 0] = np.clip(arr[:, 0], 0, width - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)
    contour = arr.astype(np.int32).reshape((-1, 1, 2))
    return contour

def process_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    height = int(data["imageHeight"])
    width = int(data["imageWidth"])
    shapes = data.get("shapes", [])

    # Collect masks per label
    label_masks = {}

    for shp in shapes:
        if shp.get("shape_type") != "polygon":
            continue  # Only polygons are used to build masks
        pts = shp.get("points", [])
        if not pts:
            continue
        label = shp.get("label", "Unknown")
        safe_label = sanitize_label(label)

        if safe_label not in label_masks:
            # 8-bit single-channel mask
            label_masks[safe_label] = np.zeros((height, width), dtype=np.uint8)

        contour = polygon_to_contour(pts, width, height)
        # Fill polygon with white (255) on black background
        cv2.fillPoly(label_masks[safe_label], [contour], 255)

    # Save masks
    out_dir = os.path.join(os.path.dirname(json_path), "masks")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for lbl, mask in label_masks.items():
        out_path = os.path.join(out_dir, f"{lbl}.png")
        cv2.imwrite(out_path, mask)
        saved += 1

    return saved, len(label_masks)

def find_jsons(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fn in filenames:
            if fn.lower().endswith(".json"):
                yield os.path.join(dirpath, fn)

def main():
    parser = argparse.ArgumentParser(description="Generate per-label masks from polygon annotations in JSON.")
    parser.add_argument("-r", "--root", default=".", help="Root directory (run from with_label). Default: current dir")
    args = parser.parse_args()

    total_json = 0
    total_masks = 0

    for jp in find_jsons(args.root):
        try:
            saved, unique_labels = process_json(jp)
            total_json += 1
            total_masks += saved
            print(f"[OK] {jp} -> {saved} mask(s)")
        except Exception as e:
            print(f"[ERR] {jp}: {e}")

    print(f"Done. Processed {total_json} JSON file(s), wrote {total_masks} mask(s).")

if __name__ == "__main__":
    main()