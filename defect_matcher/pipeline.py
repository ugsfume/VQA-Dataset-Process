"""
Pipeline: orchestrates batch import from defect datasets into with_label/defect.
"""
import os
import json
import random
from typing import Dict, List, Tuple

import cv2
from colorama import Fore, Style, Back

from config import DATASET_TO_CLASS
from image_ops import letterbox
from matcher import build_scales, find_best_match_with_rotations, draw_match
from mask_ops import crop_and_normalize_masks
from io_utils import gather_jpgs_recursive, ensure_template_assets, next_defect_index

class DefectImporter:
    """
    Batch processor that scans known datasets, matches JPGs to class templates, and writes defect samples.
    """

    def __init__(
        self,
        root: str,
        source_root: str,
        rotations: List[int],
        min_scale: float,
        max_scale: float,
        scale_steps: int,
        normalize_size: int,
        score_threshold: float,
        save_matched_vis: bool,
        limit_per_dataset: int,
        random_rotate_sample: bool,
        shuffle_seed: int | None,
    ) -> None:
        self.root = os.path.abspath(root)
        self.source_root = os.path.abspath(source_root)
        self.rotations = [int(r) for r in rotations]
        self.scales = build_scales(min_scale, max_scale, scale_steps)
        self.normalize_size = int(normalize_size)
        self.score_threshold = float(score_threshold)
        self.save_matched_vis = bool(save_matched_vis)
        self.limit_per_dataset = max(0, int(limit_per_dataset))
        self.random_rotate_sample = bool(random_rotate_sample)
        self.rng = random.Random(shuffle_seed) if shuffle_seed is not None else random.Random()

        # Output base: with_label/defect
        self.defect_root = os.path.join(self.root, "defect")
        os.makedirs(self.defect_root, exist_ok=True)

        # Stats
        self.total_processed = 0
        self.total_accepted = 0
        self.per_class_accept: Dict[str, int] = {}

    def _present_sets(self) -> Dict[str, List[str]]:
        """
        Filter source_root to dataset folders we know and group them by class.
        """
        present_sets: Dict[str, List[str]] = {}
        for entry in os.scandir(self.source_root):
            if entry.is_dir():
                ds_name = entry.name
                if ds_name in DATASET_TO_CLASS:
                    present_sets.setdefault(DATASET_TO_CLASS[ds_name], []).append(entry.path)
        return present_sets

    def _process_one_defect(
        self,
        class_name: str,
        defect_jpg: str,
        full_png: str,
        full_masks_dir: str,
        out_class_defect_dir: str,
    ) -> Tuple[bool, str]:
        """
        Process a single JPG; on success, create output subfolder with aug_image.png, masks/, and meta.json.
        Returns (accepted, out_dir or reason).
        """
        full_bgr = cv2.imread(full_png, cv2.IMREAD_COLOR)
        if full_bgr is None:
            return False, f"Cannot read template: {full_png}"
        defect_bgr = cv2.imread(defect_jpg, cv2.IMREAD_COLOR)
        if defect_bgr is None:
            return False, f"Cannot read defect: {defect_jpg}"

        best = find_best_match_with_rotations(full_bgr, defect_bgr, rotations=self.rotations, scales=self.scales)
        if best is None or float(best.get("score", -1)) < self.score_threshold:
            return False, f"score {best.get('score', -1):.4f} < {self.score_threshold}"

        # Create output subfolder
        idx = next_defect_index(out_class_defect_dir, class_name)
        out_dir = os.path.join(out_class_defect_dir, f"{class_name}_defect_{idx}")
        os.makedirs(out_dir, exist_ok=False)

        # Save aug_image.png (letterboxed defect image)
        aug_img = letterbox(defect_bgr, target=self.normalize_size)
        cv2.imwrite(os.path.join(out_dir, "aug_image.png"), aug_img)

        # Save masks (crop from full masks using best bbox, inverse-rotate, letterbox)
        os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
        written = crop_and_normalize_masks(
            full_masks_dir,
            os.path.join(out_dir, "masks"),
            (best["x1"], best["y1"], best["x2"], best["y2"]),
            size=self.normalize_size,
            rot_deg=int(best.get("rot_deg", 0))
        )

        # Optional matched visualization
        if self.save_matched_vis:
            draw_match(full_bgr, best, os.path.join(out_dir, "matched.png"), rot_deg=int(best.get("rot_deg", 0)))

        # Optional: random rotate the whole sample (aug_image + all masks) by 0/90/180/270 consistently
        random_rot_deg = 0
        if self.random_rotate_sample:
            random_rot_deg = self.rng.choice([0, 90, 180, 270])
            if random_rot_deg != 0:
                # Rotate aug_image.png
                ai_path = os.path.join(out_dir, "aug_image.png")
                ai = cv2.imread(ai_path, cv2.IMREAD_COLOR)
                if ai is not None:
                    import numpy as np  # local import to avoid circular
                    from image_ops import rotate_image_90 as rot90
                    ai_rot = rot90(ai, random_rot_deg)
                    cv2.imwrite(ai_path, ai_rot)
                # Rotate all masks
                mdir = os.path.join(out_dir, "masks")
                for e in os.scandir(mdir):
                    if not e.is_file() or not e.name.lower().endswith(".png"):
                        continue
                    m = cv2.imread(e.path, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    from image_ops import rotate_image_90 as rot90
                    m_rot = rot90(m, random_rot_deg)
                    cv2.imwrite(e.path, m_rot)

        # Save meta.json
        meta = {
            "defect": True,
            "class": class_name,
            "source_defect_image": os.path.abspath(defect_jpg),
            "template_image": os.path.abspath(full_png),
            "template_masks_dir": os.path.abspath(full_masks_dir),
            "normalize_size": int(self.normalize_size),
            "score_threshold": float(self.score_threshold),
            "match": {
                "method": best.get("method"),
                "score": float(best.get("score", 0.0)),
                "scale": float(best.get("scale", 1.0)),
                "rot_deg": int(best.get("rot_deg", 0)),
                "bbox": {
                    "x1": int(best["x1"]), "y1": int(best["y1"]),
                    "x2": int(best["x2"]), "y2": int(best["y2"]),
                    "w": int(best["w"]), "h": int(best["h"])
                }
            }
        }
        if self.save_matched_vis:
            meta["matched_vis"] = "matched.png"
        if self.random_rotate_sample:
            meta["random_sample_rotation_deg"] = int(random_rot_deg)
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return True, out_dir

    def import_all(self) -> None:
        """
        Run batch import across all present datasets, printing progress and totals.
        """
        present_sets = self._present_sets()
        if not present_sets:
            print("[INFO] No known defect dataset folders found under source root.")
            return

        # Init stats
        self.total_processed = 0
        self.total_accepted = 0
        self.per_class_accept = {cls: 0 for cls in present_sets.keys()}

        # Process per class
        for class_name, dataset_dirs in present_sets.items():
            try:
                tmpl_img, tmpl_masks = ensure_template_assets(self.root, class_name)
            except Exception as e:
                print(f"[SKIP] Class {class_name}: {e}")
                continue

            class_defect_dir = os.path.join(self.defect_root, f"{class_name}_defect")
            os.makedirs(class_defect_dir, exist_ok=True)

            class_processed = 0
            class_accepted = 0

            print(f"[CLASS] {class_name}: {len(dataset_dirs)} dataset folder(s).")

            for ds_path in dataset_dirs:
                jpgs = gather_jpgs_recursive(ds_path)
                if not jpgs:
                    print(f"  [DATASET] {os.path.basename(ds_path)}: 0 jpgs found")
                    continue

                # Shuffle order so we can cap accepted samples per dataset randomly
                self.rng.shuffle(jpgs)

                ds_processed = 0
                ds_accepted = 0

                for jp in jpgs:
                    # Per-dataset accepted cap
                    if self.limit_per_dataset and ds_accepted >= self.limit_per_dataset:
                        break

                    accepted, info = self._process_one_defect(
                        class_name=class_name,
                        defect_jpg=jp,
                        full_png=tmpl_img,
                        full_masks_dir=tmpl_masks,
                        out_class_defect_dir=class_defect_dir,
                    )
                    self.total_processed += 1
                    class_processed += 1
                    ds_processed += 1
                    if accepted:
                        self.total_accepted += 1
                        class_accepted += 1
                        ds_accepted += 1
                        print(f"    {Fore.GREEN}[OK]{Style.RESET_ALL} {class_name} <- {os.path.relpath(jp, ds_path)} -> {os.path.basename(info)}  (score>= {self.score_threshold})")
                    else:
                        print(f"    {Fore.RED}[REJ]{Style.RESET_ALL} {class_name} <- {os.path.relpath(jp, ds_path)} ({info})")

                print(f"  {Fore.CYAN}[DATASET]{Style.RESET_ALL} {os.path.basename(ds_path)}: processed {ds_processed}, accepted {ds_accepted}, rejected {ds_processed - ds_accepted}")

            self.per_class_accept[class_name] = class_accepted
            print(f"{Back.BLUE}[SUMMARY]{Style.RESET_ALL} {class_name}: processed {class_processed}, accepted {class_accepted}, rejected {class_processed - class_accepted}")

        # Totals
        print("\n[PER-CLASS TOTALS]")
        for cls, cnt in self.per_class_accept.items():
            print(f"  {cls}: {cnt} defect sample(s)")
        print(f"{Back.YELLOW}[TOTAL]{Style.RESET_ALL} Processed JPGs: {self.total_processed}; Accepted samples: {self.total_accepted}")