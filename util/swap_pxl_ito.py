#!/usr/bin/env python3
"""
Swap 'PXL_ITO.png' and 'ITO.png' in the current directory and all subdirectories (recursively).

Usage:
  python swap_pxl_ito.py
  # Optional:
  python swap_pxl_ito.py --dry-run
  python swap_pxl_ito.py --start-dir /path/to/root
"""

import os
import sys
import uuid
import argparse

TARGET_A = "PXL_ITO.png"
TARGET_B = "ITO.png"

def swap_in_dir(dirpath: str, dry_run: bool = False) -> tuple[bool, bool]:
    """Swap/rename files in a single directory. Returns (ok, changed)."""
    a_path = os.path.join(dirpath, TARGET_A)
    b_path = os.path.join(dirpath, TARGET_B)

    a_exists = os.path.isfile(a_path)
    b_exists = os.path.isfile(b_path)

    try:
        if a_exists and b_exists:
            # Both exist: swap atomically via a unique temp file
            tmp_path = os.path.join(dirpath, f".swap_tmp_{uuid.uuid4().hex}.tmp")
            print(f"[SWAP] {dirpath}: {TARGET_A} <-> {TARGET_B}")
            if not dry_run:
                os.rename(a_path, tmp_path)
                os.rename(b_path, a_path)
                os.rename(tmp_path, b_path)
            return True, True

        if a_exists:
            # Only PXL_ITO.png present -> rename to ITO.png
            print(f"[RENAME] {dirpath}: {TARGET_A} -> {TARGET_B}")
            if not dry_run:
                os.rename(a_path, b_path)
            return True, True

        if b_exists:
            # Only ITO.png present -> rename to PXL_ITO.png
            print(f"[RENAME] {dirpath}: {TARGET_B} -> {TARGET_A}")
            if not dry_run:
                os.rename(b_path, a_path)
            return True, True

        # Neither present
        return True, False

    except Exception as e:
        print(f"[ERROR] {dirpath}: {e}", file=sys.stderr)
        return False, False

def main():
    parser = argparse.ArgumentParser(
        description=f"Swap '{TARGET_A}' and '{TARGET_B}' recursively starting at a directory."
    )
    parser.add_argument(
        "--start-dir", default=".", help="Directory to start from (default: current directory)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change without renaming."
    )
    args = parser.parse_args()

    total_dirs = 0
    changed_dirs = 0
    errors = 0

    for dirpath, dirnames, filenames in os.walk(args.start_dir):
        total_dirs += 1
        ok, changed = swap_in_dir(dirpath, args.dry_run)
        if not ok:
            errors += 1
        if changed:
            changed_dirs += 1

    summary_stream = sys.stderr if args.dry_run else sys.stdout
    print(
        f"\nDone. Visited {total_dirs} directories; changed {changed_dirs}.",
        file=summary_stream,
    )
    if errors:
        print(f"Encountered {errors} errors.", file=sys.stderr)

if __name__ == "__main__":
    main()
