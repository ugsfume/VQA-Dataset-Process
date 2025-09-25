#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_rule_extractor.py

Walk negative/random/random_* samples and create per-sample repair_rule.json by:
- Building defect-region masks from non-rps contours in output.json
- Intersecting with component masks in masks/*.jpg (thresholded at 128)
- Counting distinct component blobs (connected components) per region
- Mapping detected components (and combos like Data&ITO, Data&Data) to rules via repair_rules_lookup_table.py

Special case:
- For class 'TSCVD', use defect labels 'TSCOK'/'TSCNG' found in output.json and
  apply those rule sets directly (ignore other defect labels; ignore component presence).

Run from dataset root (e.g., gt_datasets_20250915):
  python repair_rule_extractor.py
  python repair_rule_extractor.py --only-missing --verbose
"""

import os
import sys
import json
import argparse
from glob import glob
from typing import Dict, List, Tuple, Set, Any

import cv2
import numpy as np
from colorama import init as colorama_init, Fore, Style

# lookup table (repair_rules.json must be next to this script)
import importlib.util

# ----- CLI colors -----
colorama_init(autoreset=True)
OK   = Fore.GREEN + "[OK]" + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
ERR  = Fore.RED + "[ERR]" + Style.RESET_ALL
INFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL

# ----- Components we care about (must match mask filenames without extension) -----
COMPONENTS_ORDER = [
    "Gate", "Data", "ITO", "Com", "Drain", "Source", "TFT", "Mesh", "Mesh_Hole", "VIA_Hole"
]

# Combos to consider (pairs & special same-type)
COMBO_KEYS = [
    ("Gate", "Data"),
    ("Data", "Com"),
    ("Gate", "Gate"),   # special: needs count >= 2 in region
    ("Gate", "Com"),
    ("Gate", "TFT"),
    ("Gate", "Drain"),
    ("Gate", "Mesh"),
    ("Data", "Drain"),
    ("Data", "Mesh"),
    ("Data", "ITO"),
    ("Gate", "ITO"),
]
SPECIAL_SAME = {"Data", "Gate"}  # for X&X logic (>=2 blobs)

def load_lookup():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import repair_rules_lookup_table as rlt
    # Make sure it finds repair_rules.json in the same folder as the script:
    return rlt.RepairRules(file_dir=os.path.join(here, "repair_rules.json"))

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def is_rps(label: str) -> bool:
    return str(label).startswith("rps_points:")

def to_int_points(pts: List[List[float]]) -> np.ndarray:
    """(N,2) int32"""
    arr = np.array(pts, dtype=np.float32)
    arr = np.rint(arr).astype(np.int32)
    return arr

def mask_from_polygon(points: List[List[float]], h: int, w: int) -> np.ndarray:
    """Fill polygon to a uint8 mask (0/255)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) >= 3:
        cnt = to_int_points(points).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [cnt], 255)
    elif len(points) == 2:
        # Treat a 2-pt "defect" as a thin line; rasterize as 1px line then dilate slightly
        p1, p2 = to_int_points(points)
        cv2.line(mask, tuple(p1), tuple(p2), 255, 1, cv2.LINE_8)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    return mask

def binarize(img: np.ndarray, thresh: int = 128) -> np.ndarray:
    """Return uint8 mask 0/255 from grayscale or RGB."""
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return bw

def load_component_masks(masks_dir: str, h: int, w: int, thresh: int) -> Dict[str, np.ndarray]:
    comp = {}
    for name in COMPONENTS_ORDER:
        p = os.path.join(masks_dir, f"{name}.jpg")
        if not os.path.isfile(p):
            continue
        m = binarize(cv2.imread(p, cv2.IMREAD_UNCHANGED), thresh)
        if m is None:
            continue
        if (m.shape[0], m.shape[1]) != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        comp[name] = m
    return comp

def connected_blob_count(mask01: np.ndarray) -> int:
    """mask01: 0/1 uint8. Returns number of 8-connected components (excluding background)."""
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask01, connectivity=8)
    return max(0, num_labels - 1)

def detect_region_components(defect_mask: np.ndarray, comps: Dict[str, np.ndarray], min_pixels: int = 1) -> Tuple[Set[str], Dict[str, int]]:
    """
    For one defect region, return:
        present: set of component names present (intersection has >min_pixels)
        counts:  number of distinct blobs intersecting region (per component)
    """
    present = set()
    counts: Dict[str, int] = {}
    d01 = (defect_mask > 0).astype(np.uint8)
    for name, cmask in comps.items():
        inter = cv2.bitwise_and(d01, (cmask > 0).astype(np.uint8))
        area = int(inter.sum())
        if area > min_pixels:
            present.add(name)
            counts[name] = connected_blob_count(inter)
    return present, counts

def region_to_composes(present: Set[str], counts: Dict[str, int]) -> Set[str]:
    """Translate detected components in a region to compose keys."""
    out: Set[str] = set()
    # Singles
    for s in present:
        out.add(s)
    # Same-type specials (X&X requires >=2)
    for s in SPECIAL_SAME:
        if s in counts and counts[s] >= 2:
            out.add(f"{s}&{s}")
    # Pairs
    for a, b in COMBO_KEYS:
        if a == b:
            # already handled by SPECIAL_SAME logic
            continue
        if a in present and b in present:
            out.add(f"{a}&{b}")
    return out

def label_component_masks(comps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    For each binary component mask (0/255), compute an int32 label map (0=bg, 1..N blobs).
    """
    label_maps: Dict[str, np.ndarray] = {}
    for name, cmask in comps.items():
        lab_count, lab_map = cv2.connectedComponents((cmask > 0).astype(np.uint8), connectivity=8)
        # lab_map is HxW int32 with values in [0..lab_count-1]
        label_maps[name] = lab_map.astype(np.int32, copy=False)
    return label_maps

def detect_region_components_global(
    defect_mask: np.ndarray,
    comps: Dict[str, np.ndarray],
    comp_labels: Dict[str, np.ndarray],
    min_pixels: int = 1
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Count distinct *global* component blobs that intersect the defect region.
    present: components with >= min_pixels overlap
    counts:  number of distinct global labels overlapping the region
    """
    present: Set[str] = set()
    counts: Dict[str, int] = {}

    if defect_mask.dtype != np.uint8:
        d01 = (defect_mask > 0).astype(np.uint8)
    else:
        d01 = (defect_mask > 0).astype(np.uint8)

    idx = d01 > 0  # boolean index HxW
    for name, lab_map in comp_labels.items():
        if name not in comps:
            continue
        # pixels of this component (global) that lie inside the region:
        labels_inside = lab_map[idx]  # 1D view
        area = int(np.count_nonzero(labels_inside))  # any nonzero label = component pixel
        if area >= min_pixels:
            present.add(name)
            # count unique nonzero global labels overlapping region
            uniq = np.unique(labels_inside)
            uniq = uniq[uniq != 0]
            counts[name] = int(uniq.size)
    return present, counts

def extract_rules_for_sample(sample_dir: str, lookup, thresh: int, min_pixels: int, verbose: bool=False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (rules_list, debug_info)
      rules_list: list of {"damaged_component": key, "rules": [ ... ]}
    """
    out_json = os.path.join(sample_dir, "output.json")
    meta_json = os.path.join(sample_dir, "meta.json")
    masks_dir = os.path.join(sample_dir, "masks")
    img_path  = os.path.join(sample_dir, "original_image.jpg")

    if not (os.path.isfile(out_json) and os.path.isfile(meta_json) and os.path.isdir(masks_dir)):
        return [], {"skipped": True, "reason": "missing files"}

    meta = read_json(meta_json)
    cls = meta.get("class", "").strip() or meta.get("origin", {}).get("class", "").strip()
    if not cls:
        return [], {"skipped": True, "reason": "no class in meta.json"}

    # Image size
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return [], {"skipped": True, "reason": "cannot open original_image.jpg"}
    h, w = img.shape[:2]

    # Load component masks
    comps = load_component_masks(masks_dir, h, w, thresh)
    comp_labels = label_component_masks(comps)
    if not comps:
        return [], {"skipped": True, "reason": "no masks loaded"}

    # Parse contours
    data = read_json(out_json)
    contours = data.get("contours", [])

    # Special handling for TSCVD
    if cls == "TSCVD":
        found = set()
        for item in contours:
            lab = str(item.get("label", ""))
            if lab in ("TSCOK", "TSCNG"):
                found.add(lab)
        rules_out: Dict[str, List[Dict[str, Any]]] = {}
        for key in sorted(found):
            rules = lookup.get_value(key, "") or []
            # Normalize rule items into desired output schema
            norm_rules = []
            for r in rules:
                norm_rules.append({
                    "repair_rule": r.get("repair_rule"),
                    "operations": r.get("operations"),
                    "repair_components": r.get("repair_components"),
                })
            # de-dup by damaged_component key
            rules_out[key] = norm_rules
        # Return flattened list
        result = [{"damaged_component": k, "rules": v} for k, v in rules_out.items()]
        dbg = {"skipped": False, "mode": "TSCVD", "subclasses": sorted(found)}
        return result, dbg

    # General case: build defect-region masks and map to comps
    # Each non-rps contour is one region
    region_composes: List[Set[str]] = []
    used_labels: List[str] = []
    for item in contours:
        lab = str(item.get("label", ""))
        if is_rps(lab):
            continue
        pts = item.get("points", [])
        if not pts:
            continue
        defect = mask_from_polygon(pts, h, w)
        present, counts = detect_region_components_global(defect, comps, comp_labels, min_pixels=min_pixels)
        composes = region_to_composes(present, counts)
        if composes:
            region_composes.append(composes)
            used_labels.append(lab)

    # Union across regions but keep unique keys
    all_keys: Set[str] = set()
    for s in region_composes:
        all_keys.update(s)

    if not all_keys:
        return [], {"skipped": False, "mode": "generic", "note": "no damaged comps detected"}

    # Lookup rules per key for this class
    rules_out: Dict[str, List[Dict[str, Any]]] = {}
    for key in sorted(all_keys):
        rules = lookup.get_value(cls, key) or []
        if not rules:
            continue
        norm_rules = []
        for r in rules:
            norm_rules.append({
                "repair_rule": r.get("repair_rule"),
                "operations": r.get("operations"),
                "repair_components": r.get("repair_components"),
            })
        rules_out[key] = norm_rules

    result = [{"damaged_component": k, "rules": v} for k, v in rules_out.items()]
    dbg = {"skipped": False, "mode": "generic", "regions": len(region_composes), "unique_components": sorted(all_keys)}
    return result, dbg

def main():
    ap = argparse.ArgumentParser(description="Extract repair rules for negative/random samples.")
    ap.add_argument("--root", type=str, default=".", help="Dataset root (run from gt_datasets_20250915).")
    ap.add_argument("--only-missing", action="store_true", help="Skip samples that already have repair_rule.json.")
    ap.add_argument("--mask-thresh", type=int, default=128, help="Threshold for binarizing component masks.")
    ap.add_argument("--min-pixels", type=int, default=1, help="Minimum intersection pixels to consider a component present.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs.")
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.root)
    rnd_root = os.path.join(dataset_root, "negative", "random")
    if not os.path.isdir(rnd_root):
        print(f"{ERR} Not found: {rnd_root}")
        sys.exit(1)

    lookup = load_lookup()

    samples = sorted([p for p in glob(os.path.join(rnd_root, "random_*")) if os.path.isdir(p)])
    if args.verbose:
        print(f"{INFO} Found {len(samples)} samples under {rnd_root}")

    done = 0
    skipped = 0
    for sdir in samples:
        out_path = os.path.join(sdir, "repair_rule.json")
        if args.only_missing and os.path.isfile(out_path):
            skipped += 1
            if args.verbose:
                print(f"{WARN} Skip existing {os.path.relpath(out_path, start=dataset_root)}")
            continue

        rules, dbg = extract_rules_for_sample(
            sdir, lookup,
            thresh=args.mask_thresh,
            min_pixels=args.min_pixels,
            verbose=args.verbose
        )

        if dbg.get("skipped"):
            if args.verbose:
                print(f"{WARN} Skip {os.path.relpath(sdir, start=dataset_root)}: {dbg.get('reason','')}")
            continue

        write_json(rules, out_path)
        done += 1
        if args.verbose:
            mode = dbg.get("mode", "generic")
            extra = ""
            if mode == "generic":
                extra = f" | regions={dbg.get('regions',0)} keys={','.join(dbg.get('unique_components',[]))}"
            elif mode == "TSCVD":
                extra = f" | subclasses={','.join(dbg.get('subclasses',[]))}"
            print(f"{OK} Wrote {os.path.relpath(out_path, start=dataset_root)} [{mode}]{extra}")

    print(f"{OK} Completed. Wrote {done} file(s). Skipped {skipped} pre-existing.")


if __name__ == "__main__":
    main()
