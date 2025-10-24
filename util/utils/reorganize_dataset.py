#!/usr/bin/env python3
"""
Reorganize TFT repair dataset into a new directory.

Notes:
- Copies files (does not modify the original dataset).
- Any file starting with "mask_" is copied into "masks/" with the prefix removed.
- Other files remain at the sample folder root.
- Tracks samples with fewer than --min-masks mask files.
- Aggregates set of unique mask types (based on filename after "mask_" and before extension).
- Writes reports into dest_root:
    - samples_with_fewer_than_5_masks.txt
    - unique_mask_types.txt
    - mapping.csv (src -> dest)

"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_PREFIX = "mask_"

def is_mask_file(p: Path) -> bool:
    return p.is_file() and p.name.startswith(MASK_PREFIX)

def mask_target_name(src_name: str) -> str:
    """Strip 'mask_' prefix from filename."""
    if src_name.startswith(MASK_PREFIX):
        return src_name[len(MASK_PREFIX):]
    return src_name

def safe_copy(src: Path, dst: Path, dry_run: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY] copy: {src} -> {dst}")
        return
    shutil.copy2(src, dst)

def write_text(path: Path, text: str, dry_run: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY] write: {path}")
        return
    path.write_text(text, encoding="utf-8")

def collect_categories(root: Path) -> Dict[str, List[Path]]:
    """
    Return mapping category_name -> list of sample directories (direct children).
    Only considers immediate subdirectories inside each category folder.
    """
    categories: Dict[str, List[Path]] = {}
    if not root.is_dir():
        return categories
    for category_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        samples = [s for s in sorted(category_dir.iterdir()) if s.is_dir()]
        if samples:
            categories[category_dir.name] = samples
    return categories

def reorganize(
    src_root: Path,
    dest_root: Path,
    min_masks: int = 5,
    dry_run: bool = False,
) -> Tuple[List[Tuple[Path, Path, int]], Set[str], List[Tuple[Path, int]]]:
    """
    Returns:
      - mapping: list of (src_sample_dir, dest_sample_dir, n_masks)
      - unique_masks: set of mask type names (e.g., "ITO", "Data", ...)
      - few_mask_samples: list of (dest_sample_dir, n_masks) for those with < min_masks
    """
    mapping: List[Tuple[Path, Path, int]] = []
    unique_masks: Set[str] = set()
    few_mask_samples: List[Tuple[Path, int]] = []

    for polarity in ("negative", "positive"):
        pol_src = src_root / polarity
        if not pol_src.is_dir():
            print(f"[WARN] Skipping missing '{polarity}' dir at {pol_src}")
            continue

        # Gather categories under this polarity
        cat_map = collect_categories(pol_src)

        for category, samples in cat_map.items():
            # Destination category root
            cat_dest_root = dest_root / polarity / category
            # Counter starts at 1 for each category
            counter = 1

            for src_sample in samples:
                dest_sample_name = f"{category}_{counter}"
                dest_sample = cat_dest_root / dest_sample_name
                counter += 1

                # Create dest sample dir
                if dry_run:
                    print(f"[DRY] mkdir -p {dest_sample}")
                else:
                    dest_sample.mkdir(parents=True, exist_ok=True)

                # Create hidden file with original sample name
                hidden_name = f".{src_sample.name}"
                hidden_path = dest_sample / hidden_name
                write_text(hidden_path, src_sample.name + "\n", dry_run=dry_run)

                # Create masks folder
                masks_dir = dest_sample / "masks"
                if dry_run:
                    print(f"[DRY] mkdir -p {masks_dir}")
                else:
                    masks_dir.mkdir(parents=True, exist_ok=True)

                # Copy files: mask_* go into masks/ (strip prefix), others to root
                mask_count = 0
                for item in sorted(src_sample.iterdir()):
                    if item.is_file():
                        if is_mask_file(item):
                            # Parse mask type and collect
                            target_name = mask_target_name(item.name)
                            # Collect mask type (filename stem without extension)
                            mask_type = Path(target_name).stem
                            unique_masks.add(mask_type)

                            dst_path = masks_dir / target_name

                            # Resolve potential name collisions by appending counter
                            if dst_path.exists() and not dry_run:
                                base = dst_path.stem
                                ext = dst_path.suffix
                                k = 2
                                while True:
                                    candidate = masks_dir / f"{base}_{k}{ext}"
                                    if not candidate.exists():
                                        dst_path = candidate
                                        break
                                    k += 1

                            safe_copy(item, dst_path, dry_run=dry_run)
                            mask_count += 1
                        else:
                            # Non-mask files stay at sample root
                            dst_path = dest_sample / item.name

                            # Avoid overwriting if duplicate filename somehow
                            if dst_path.exists() and not dry_run:
                                base = dst_path.stem
                                ext = dst_path.suffix
                                k = 2
                                while True:
                                    candidate = dest_sample / f"{base}_{k}{ext}"
                                    if not candidate.exists():
                                        dst_path = candidate
                                        break
                                    k += 1
                            safe_copy(item, dst_path, dry_run=dry_run)
                    # If there are subdirs inside a sample, copy them recursively to sample root
                    elif item.is_dir():
                        # Mirror subdir under dest sample
                        rel = item.name
                        dst_subdir = dest_sample / rel
                        if dry_run:
                            print(f"[DRY] copytree: {item} -> {dst_subdir}")
                        else:
                            if dst_subdir.exists():
                                shutil.rmtree(dst_subdir)
                            shutil.copytree(item, dst_subdir)

                mapping.append((src_sample, dest_sample, mask_count))
                if mask_count < min_masks:
                    few_mask_samples.append((dest_sample, mask_count))

    return mapping, unique_masks, few_mask_samples

def write_reports(
    dest_root: Path,
    mapping: List[Tuple[Path, Path, int]],
    unique_masks: Set[str],
    few_mask_samples: List[Tuple[Path, int]],
    min_masks: int,
    dry_run: bool = False,
) -> None:
    # mapping.csv
    lines = ["src_sample,dest_sample,n_masks"]
    for src_path, dst_path, n_masks in mapping:
        lines.append(f"{src_path},{dst_path},{n_masks}")
    write_text(dest_root / "mapping.csv", "\n".join(lines) + "\n", dry_run=dry_run)

    # unique_mask_types.txt
    masks_sorted = "\n".join(sorted(unique_masks))
    write_text(dest_root / "unique_mask_types.txt", masks_sorted + "\n", dry_run=dry_run)

    # samples_with_fewer_than_X_masks.txt
    few_lines = [f"Samples with fewer than {min_masks} mask files:"]
    for dst_path, n_masks in few_mask_samples:
        few_lines.append(f"{dst_path}  (n_masks={n_masks})")
    write_text(dest_root / "samples_with_fewer_than_{}_masks.txt".format(min_masks),
               "\n".join(few_lines) + "\n", dry_run=dry_run)

def main():
    parser = argparse.ArgumentParser(description="Reorganize TFT dataset into a new directory.")
    parser.add_argument("--src-root", type=Path, default=Path("."), help="Source dataset root (default: .)")
    parser.add_argument("--dest-root", type=Path,
                        default=Path("/mnt/workspace/autorepair_vlm/gt_datasets_20250915"),
                        help="Destination root directory.")
    parser.add_argument("--min-masks", type=int, default=5,
                        help="Minimum number of mask files required; below will be reported (default: 5).")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying files.")
    args = parser.parse_args()

    src_root: Path = args.src_root.resolve()
    dest_root: Path = args.dest_root.resolve()

    print(f"[INFO] Source:      {src_root}")
    print(f"[INFO] Destination: {dest_root}")
    print(f"[INFO] Min masks:   {args.min_masks}")
    print(f"[INFO] Dry run:     {args.dry_run}")

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"[ERROR] Source root does not exist or is not a directory: {src_root}")

    # Create destination root (safe if exists)
    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    mapping, unique_masks, few_mask_samples = reorganize(
        src_root=src_root,
        dest_root=dest_root,
        min_masks=args.min_masks,
        dry_run=args.dry_run,
    )

    write_reports(
        dest_root=dest_root,
        mapping=mapping,
        unique_masks=unique_masks,
        few_mask_samples=few_mask_samples,
        min_masks=args.min_masks,
        dry_run=args.dry_run,
    )

    # Summary to stdout
    print("\n[SUMMARY]")
    print(f"  Total samples copied: {len(mapping)}")
    if unique_masks:
        print(f"  Unique mask types ({len(unique_masks)}): {', '.join(sorted(unique_masks))}")
    else:
        print("  Unique mask types: none found")
    if few_mask_samples:
        print(f"  Samples with fewer than {args.min_masks} masks: {len(few_mask_samples)}")
        for dst, n in few_mask_samples[:10]:
            print(f"    - {dst} (n_masks={n})")
        if len(few_mask_samples) > 10:
            print(f"    ... and {len(few_mask_samples) - 10} more")
    else:
        print(f"  All samples have at least {args.min_masks} masks.")

if __name__ == "__main__":
    main()
