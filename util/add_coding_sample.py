#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_coding_sample.py

Walk a SOURCE directory recursively, find pairs of:
  - result_<k>.json
  - result_<k>.jpg OR result_<k>_w.jpg   (the "_w" suffix is treated the same)

For each found pair, create a new folder in the CURRENT WORKING DIRECTORY
(the output directory where you run this script) named:
  <prefix>_<n>
where:
  - <prefix> defaults to "coding" (override with --prefix)
  - <n> is a running index starting from 1 and CONTINUES after the highest
    existing <prefix>_<n> folder in the output directory.

Inside each created folder:
  - copy the image as "repair_image.jpg"
  - copy the json  as "output.json"
  - create "meta.json" recording absolute paths to the original files

Usage:
  python add_coding_sample.py /path/to/source \
      [--prefix coding] [--dry-run]

Notes:
  - This script COPIES files (does not move).
  - If both result_<k>.jpg and result_<k>_w.jpg exist, the *_w.jpg is preferred.
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

from colorama import init as colorama_init, Fore, Style, Back
colorama_init(autoreset=True)
LBL_WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
LBL_ERR = Fore.RED + "[ERR]" + Style.RESET_ALL
LBL_DO   = Fore.GREEN  + "[DO]"   + Style.RESET_ALL
LBL_INFO = Back.BLUE   + "[INFO]" + Style.RESET_ALL
LBL_DONE = Back.YELLOW + "[DONE]" + Style.RESET_ALL

JSON_RE = re.compile(r"^result_(\d+)\.json$", re.IGNORECASE)

def find_next_start_index(out_dir: Path, prefix: str) -> int:
    """Return next index after the max existing '<prefix>_<n>' directory."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_idx = 0
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.is_dir():
                m = pat.match(p.name)
                if m:
                    try:
                        idx = int(m.group(1))
                        if idx > max_idx:
                            max_idx = idx
                    except ValueError:
                        pass
    return max_idx + 1

def pick_image_for_index(dirpath: Path, idx: str) -> Optional[Path]:
    """
    Prefer 'result_<idx>_w.jpg' if present; otherwise 'result_<idx>.jpg'.
    Return None if neither exists.
    """
    cand_w = dirpath / f"result_{idx}_w.jpg"
    cand_plain = dirpath / f"result_{idx}.jpg"
    if cand_w.exists():
        return cand_w
    if cand_plain.exists():
        return cand_plain
    # also check for uppercase/lowercase variants just in case
    for child in dirpath.iterdir():
        if child.is_file():
            name = child.name.lower()
            if name == f"result_{idx}_w.jpg" or name == f"result_{idx}.jpg":
                return child
    return None

def collect_pairs(source_root: Path) -> Dict[Tuple[Path, str], Tuple[Path, Path]]:
    """
    Return a mapping:
      (dirpath, idx_str) -> (image_path, json_path)
    Only includes entries where both image and json exist.
    """
    pairs = {}
    for dirpath, _, filenames in os.walk(source_root):
        dpath = Path(dirpath)
        # build a quick lookup for jsons
        for fname in filenames:
            m = JSON_RE.match(fname)
            if not m:
                continue
            idx = m.group(1)
            json_path = dpath / fname
            img_path = pick_image_for_index(dpath, idx)
            if img_path is None:
                # No matching image; skip
                continue
            pairs[(dpath, idx)] = (img_path, json_path)
    return pairs

def write_meta(out_dir: Path, image_src: Path, json_src: Path) -> None:
    meta = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_image": str(image_src.resolve()),
        "source_json": str(json_src.resolve()),
        "source_dir": str(image_src.parent.resolve()),
        "source_stem": image_src.stem,  # e.g. "result_15_w"
        "script": "add_coding_sample.py",
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Collect 'result_k' pairs into numbered sample folders."
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to the SOURCE directory to scan recursively (read-only).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="coding",
        help="Folder name prefix for created samples (default: 'coding').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying/creating files.",
    )
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    output_root = Path.cwd().resolve()
    prefix = args.prefix

    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit(f"{LBL_ERR} Source directory does not exist or is not a directory: {source_root}")

    print(f"{LBL_INFO} Source : {source_root}")
    print(f"{LBL_INFO} Output : {output_root}  (current working directory)")
    print(f"{LBL_INFO} Prefix : {prefix}")

    # Gather all valid pairs
    pairs = collect_pairs(source_root)
    if not pairs:
        print(f"{LBL_WARN} No valid (result_k.jpg/_w.jpg + result_k.json) pairs found.")
        return

    # Determine starting index
    next_idx = find_next_start_index(output_root, prefix)
    created = 0

    for (dpath, idx_str), (img_src, json_src) in sorted(pairs.items()):
        out_dir = output_root / f"{prefix}_{next_idx}"
        print(f"{LBL_DO} Create {out_dir.name}  <-  {img_src.relative_to(source_root)} , {json_src.relative_to(source_root)}")

        if not args.dry_run:
            out_dir.mkdir(parents=False, exist_ok=False)
            shutil.copy2(img_src, out_dir / "repair_image.jpg")
            shutil.copy2(json_src, out_dir / "output.json")
            write_meta(out_dir, img_src, json_src)

        next_idx += 1
        created += 1

    print(f"{LBL_DONE} Processed {len(pairs)} pairs. Created {created} folders.")

if __name__ == "__main__":
    main()
