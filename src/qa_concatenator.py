"""
Combine all per-sample qa.jsonl or qa_mask.jsonl files (aug + defect) into one JSONL at the dataset root.
Run from with_label.

Usage examples:
  python qa_concatenator.py -o qa.jsonl
  python qa_concatenator.py --mask -o qa_mask.jsonl
Limit reverse QA ratio (only for --mask):
  python qa_concatenator.py --mask --reverse-max-pct 0.25
"""

import os
import json
import argparse
import random
import math
from typing import List, Dict, Generator, Tuple, Optional, Set

from colorama import init, Fore, Style
init(autoreset=True)
OK = Fore.GREEN + "[OK]" + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
ERR = Fore.RED + "[ERR]" + Style.RESET_ALL
INFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL


def iter_sample_qa_files(
    root: str,
    qa_filename: str,
    include_aug: bool,
    include_defect: bool,
    allowed_defect_dirs: Optional[Set[str]] = None
) -> Generator[str, None, None]:
    if include_aug:
        for class_entry in os.scandir(root):
            if not class_entry.is_dir() or class_entry.name.startswith(".") or class_entry.name == "defect":
                continue
            aug_dir = os.path.join(class_entry.path, "aug")
            if not os.path.isdir(aug_dir):
                continue
            for sample_entry in os.scandir(aug_dir):
                if sample_entry.is_dir():
                    qa_path = os.path.join(sample_entry.path, qa_filename)
                    if os.path.isfile(qa_path):
                        yield qa_path
    if include_defect:
        defect_root = os.path.join(root, "defect")
        if os.path.isdir(defect_root):
            for class_def in os.scandir(defect_root):
                if not class_def.is_dir() or class_def.name.startswith("."):
                    continue
                for sample_entry in os.scandir(class_def.path):
                    if not sample_entry.is_dir():
                        continue
                    if allowed_defect_dirs is not None and sample_entry.path not in allowed_defect_dirs:
                        continue
                    qa_path = os.path.join(sample_entry.path, qa_filename)
                    if os.path.isfile(qa_path):
                        yield qa_path

def list_defect_sample_dirs(root: str, qa_filename: str) -> List[str]:
    out = []
    defect_root = os.path.join(root, "defect")
    if not os.path.isdir(defect_root):
        return out
    for class_def in os.scandir(defect_root):
        if not class_def.is_dir() or class_def.name.startswith("."):
            continue
        for sample_entry in os.scandir(class_def.path):
            if sample_entry.is_dir():
                if os.path.isfile(os.path.join(sample_entry.path, qa_filename)):
                    out.append(sample_entry.path)
    return out

def stream_records(qa_path: str) -> Generator[Dict, None, None]:
    with open(qa_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
                yield obj
            except Exception as e:
                print(f"{WARN} Skip malformed JSON in {qa_path}:{ln}: {e}")

def validate_record(rec: Dict, root: str) -> Tuple[bool, str]:
    if not isinstance(rec, dict):
        return False, "record not a dict"
    if "id" not in rec or not isinstance(rec["id"], str):
        return False, "missing or invalid id"
    if "images" not in rec or not isinstance(rec["images"], list) or not all(isinstance(x, str) for x in rec["images"]):
        return False, "missing or invalid images"
    if "conversations" not in rec or not isinstance(rec["conversations"], list) or len(rec["conversations"]) < 2:
        return False, "missing or invalid conversations"
    for img_path in rec["images"]:
        p = img_path if os.path.isabs(img_path) else os.path.join(root, img_path)
        if not os.path.isfile(p):
            return False, f"image not found: {img_path}"
    return True, "ok"

def is_reverse_record(rec: Dict) -> Tuple[bool, str]:
    rid = rec.get("id", "")
    meta = rec.get("meta", {}) or {}
    rtype = meta.get("reverse")
    if rtype:
        return True, str(rtype)
    if "_rev_color_" in rid:
        return True, "color_to_component"
    if "_rev_window_" in rid:
        return True, "window_to_components"
    return False, ""

def cap_reverse_records(records: List[Dict], pct: float, rng: random.Random) -> List[Dict]:
    """
    Keep reverse QA <= pct of final total.
    Let F = forward count. Max reverse = floor(p * F / (1 - p)).
    """
    if pct <= 0:
        kept = [r for r in records if not is_reverse_record(r)[0]]
        print(f"{INFO} Reverse capped: kept 0 (pct<=0). Forward={len(kept)}")
        return kept
    forward = []
    reverse_by_type: Dict[str, List[Dict]] = {}
    for r in records:
        is_rev, rtype = is_reverse_record(r)
        if not is_rev:
            forward.append(r)
        else:
            reverse_by_type.setdefault(rtype or "unknown", []).append(r)
    F = len(forward)
    R_total = sum(len(v) for v in reverse_by_type.values())
    if F == 0:
        print(f"{WARN} No forward records; dropping reverse to avoid all-reverse dataset.")
        return []
    if pct >= 0.999:
        print(f"{INFO} Reverse cap >=1.0; keeping all reverse ({R_total})")
        return records
    max_reverse = math.floor(pct * F / (1 - pct))
    target = min(R_total, max_reverse)
    if R_total <= target:
        print(f"{INFO} Reverse already within cap (R={R_total} <= target={target}).")
        return records
    # Proportional sampling across types
    selected = []
    # Flatten for weighted random if small difference
    all_reverse = []
    for t, lst in reverse_by_type.items():
        all_reverse.extend((t, rec) for rec in lst)
    rng.shuffle(all_reverse)
    # Simple take-first target
    taken_types_count = {}
    for t, rec in all_reverse:
        if len(selected) >= target:
            break
        selected.append(rec)
        taken_types_count[t] = taken_types_count.get(t, 0) + 1
    kept = forward + selected
    print(f"{INFO} Reverse capped from {R_total} -> {len(selected)} (target={target}) "
          f"Forward={F} FinalTotal={len(kept)} "
          f"TypeBreakdown={{{', '.join(f'{k}:{v}' for k,v in taken_types_count.items())}}}")
    return kept

def merge(
    root: str,
    qa_filename: str,
    include_aug: bool,
    include_defect: bool,
    shuffle: bool,
    limit: int,
    validate: bool,
    strict: bool,
    seed: int | None,
    reverse_max_pct: float,
    allowed_defect_dirs: Optional[Set[str]]
) -> List[Dict]:
    all_records: List[Dict] = []
    seen_ids: dict[str, int] = {}
    defect_record_count = 0  # NEW: count defect-origin records

    qa_files = list(iter_sample_qa_files(root, qa_filename, include_aug, include_defect, allowed_defect_dirs))
    if not qa_files:
        print(f"{WARN} No {qa_filename} files found.")
        return []

    print(f"{INFO} Found {len(qa_files)} {qa_filename} source files.")

    for path in qa_files:
        is_defect_file = "defect" in os.path.normpath(path).split(os.sep)
        count_in_file = 0
        for rec in stream_records(path):
            count_in_file += 1
            rid = rec.get("id", "")
            if not isinstance(rid, str) or not rid:
                rid = f"synthetic_{len(all_records)}"
                rec["id"] = rid
            if rid in seen_ids:
                seen_ids[rid] += 1
                rec["id"] = f"{rid}#dup{seen_ids[rid]}"
            else:
                seen_ids[rid] = 0
            imgs = rec.get("images", [])
            abs_imgs = []
            for img in imgs:
                p = img.replace("\\", "/")
                if not os.path.isabs(p):
                    p = os.path.abspath(os.path.join(root, p))
                abs_imgs.append(p)
            rec["images"] = abs_imgs
            if "meta" in rec:
                del rec["meta"]
            if validate:
                ok, msg = validate_record(rec, root)
                if not ok:
                    if strict:
                        raise ValueError(f"Validation failed ({msg}) for record id={rec.get('id')} in {path}")
                    else:
                        print(f"{WARN} Validation skip id={rec.get('id')} ({msg})")
                        continue
            all_records.append(rec)
            if is_defect_file:
                defect_record_count += 1
        print(f"{OK} {os.path.relpath(path, root)}: loaded {count_in_file} record(s)")

    rng = random.Random(seed)

    if qa_filename == "qa_mask.jsonl" and 0 <= reverse_max_pct < 1.0:
        before_rev = len(all_records)
        all_records = cap_reverse_records(all_records, reverse_max_pct, rng)
        after_rev = len(all_records)
        # reverse reduction already logged inside cap_reverse_records

    if shuffle:
        rng.shuffle(all_records)
        print(f"{INFO} Shuffled {len(all_records)} records (seed={seed}).")

    if limit > 0 and len(all_records) > limit:
        all_records = all_records[:limit]
        print(f"{INFO} Limited to first {limit} records.")

    if all_records:
        pct_defect = defect_record_count / len(all_records) * 100
        print(f"{INFO} Defect QA records kept: {defect_record_count} ({pct_defect:.2f}% of final)")

    print(f"{OK} Aggregated total {len(all_records)} records (meta removed).")
    return all_records

def write_jsonl(records: List[Dict], out_path: str) -> None:
    """Write list of records to JSONL."""
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

def main():
    ap = argparse.ArgumentParser(description="Combine all per-sample qa.jsonl or qa_mask.jsonl files into one JSONL.")
    ap.add_argument("-r", "--root", default=".", help="Dataset root (with_label).")
    ap.add_argument("-o", "--out", default=None, help="Output JSONL filename (placed under root).")
    ap.add_argument("--mask", action="store_true", help="Process qa_mask.jsonl files instead of qa.jsonl.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle merged records.")
    ap.add_argument("--seed", type=int, default=None, help="Shuffle / sampling seed.")
    ap.add_argument("--limit", type=int, default=0, help="Keep only first N records after processing.")
    ap.add_argument("--validate", action="store_true", help="Light validation.")
    ap.add_argument("--strict", action="store_true", help="Raise on validation error.")
    ap.add_argument("--no-aug", dest="include_aug", action="store_false", help="Exclude augmentation samples.")
    ap.add_argument("--no-defect", dest="include_defect", action="store_false", help="Exclude defect samples.")
    ap.add_argument("--reverse-max-pct", type=float, default=1.0,
                    help="(qa_mask only) Max fraction of reverse QA in final set (0-1]. "
                         "e.g., 0.25 means reverse <=25%% of final records. Default 1.0 (no cap).")
    ap.add_argument("--defect-sample-pct", type=float, default=1.0,
                    help="Fraction (0-1] of defect SAMPLE folders to include (sample-level downsampling). "
                         "0 excludes all defect samples. Applied before loading records.")
    args = ap.parse_args()

    if args.reverse_max_pct <= 0:
        rmp = 0.0
    else:
        rmp = min(args.reverse_max_pct, 1.0)

    dsp = max(0.0, min(args.defect_sample_pct, 1.0))

    root = os.path.abspath(args.root)
    include_aug = args.include_aug
    include_defect = args.include_defect
    if not include_aug and not include_defect:
        print(f"{ERR} Both augmentation and defect sources disabled; nothing to do.")
        return

    qa_filename = "qa_mask.jsonl" if args.mask else "qa.jsonl"
    out_path = os.path.join(root, args.out if args.out else qa_filename)

    allowed_defect_dirs: Optional[Set[str]] = None
    if include_defect and dsp < 1.0:
        all_defect_dirs = list_defect_sample_dirs(root, qa_filename)
        if not all_defect_dirs:
            print(f"{WARN} No defect sample dirs found.")
        else:
            rng = random.Random(args.seed)
            rng.shuffle(all_defect_dirs)
            k = int(round(dsp * len(all_defect_dirs)))
            if dsp > 0 and k == 0:
                k = 1
            allowed_defect_dirs = set(all_defect_dirs[:k])
            removed = len(all_defect_dirs) - len(allowed_defect_dirs)
            kept_pct = (len(allowed_defect_dirs) / len(all_defect_dirs) * 100) if all_defect_dirs else 0
            print(f"{INFO} Defect samples capped from {len(all_defect_dirs)} -> {len(allowed_defect_dirs)} "
                  f"(removed {removed}, kept {kept_pct:.2f}%)")

    records = merge(
        root=root,
        qa_filename=qa_filename,
        include_aug=include_aug,
        include_defect=include_defect,
        shuffle=args.shuffle,
        limit=args.limit,
        validate=args.validate,
        strict=args.strict,
        seed=args.seed,
        reverse_max_pct=rmp if args.mask else 1.0,
        allowed_defect_dirs=allowed_defect_dirs
    )

    write_jsonl(records, out_path)
    print(f"{OK} Wrote consolidated JSONL: {out_path}")

if __name__ == "__main__":
    main()