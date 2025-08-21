"""
CLI entry: batch-import defect JPGs into with_label/defect via template matching.
"""
import argparse
from colorama import init

from config import (
    DATASET_TO_CLASS,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_NORMALIZE_SIZE,
    DEFAULT_ROTATIONS,
    DEFAULT_MIN_SCALE,
    DEFAULT_MAX_SCALE,
    DEFAULT_SCALE_STEPS,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_SAVE_MATCHED_VIS,
    DEFAULT_LIMIT_PER_DATASET,
    DEFAULT_RANDOM_ROTATE_SAMPLE,
)
from pipeline import DefectImporter

def main():
    init(autoreset=True)

    ap = argparse.ArgumentParser(description="Batch import defect JPGs into defect VQA samples via template matching.")
    ap.add_argument("-r", "--root", default=".", help="with_label root directory (default: current dir)")
    ap.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="Root directory containing defect dataset folders.")
    ap.add_argument("--normalize-size", type=int, default=DEFAULT_NORMALIZE_SIZE, help="Output canvas size for aug_image and masks (default: 512).")
    ap.add_argument("--rotations", type=str, default=",".join(str(x) for x in DEFAULT_ROTATIONS), help="Comma list of rotations (deg CW) to try.")
    ap.add_argument("--min-scale", type=float, default=DEFAULT_MIN_SCALE, help="Min scale for multi-scale matching.")
    ap.add_argument("--max-scale", type=float, default=DEFAULT_MAX_SCALE, help="Max scale for multi-scale matching.")
    ap.add_argument("--scale-steps", type=int, default=DEFAULT_SCALE_STEPS, help="Number of scales between min and max (inclusive).")
    ap.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD, help="Reject matches below this score.")
    ap.add_argument("--save-matched-vis", action="store_true", default=DEFAULT_SAVE_MATCHED_VIS, help="Save matched.png visualization in each output sample.")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT_PER_DATASET, help="Max accepted samples per dataset (0 = no cap).")
    ap.add_argument("--shuffle-seed", type=int, default=None, help="Optional seed for shuffling JPGs (for reproducibility).")
    ap.add_argument("--no-random-rotate-sample", dest="random_rotate_sample", action="store_false", default=DEFAULT_RANDOM_ROTATE_SAMPLE, help="Disable random 0/90/180/270 rotation of aug_image and masks.")
    args = ap.parse_args()

    rotations = [int(x) for x in args.rotations.split(",") if x.strip()]

    importer = DefectImporter(
        root=args.root,
        source_root=args.source_root,
        rotations=rotations,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        scale_steps=args.scale_steps,
        normalize_size=args.normalize_size,
        score_threshold=args.score_threshold,
        save_matched_vis=args.save_matched_vis,
        limit_per_dataset=args.limit,
        random_rotate_sample=args.random_rotate_sample,
        shuffle_seed=args.shuffle_seed,
    )
    importer.import_all()

if __name__ == "__main__":
    main()