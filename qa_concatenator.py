"""
Combine all per-sample qa.jsonl files (aug + defect) into one qa.jsonl at the dataset root.
run from with_label
"""

import os
import json
import argparse
import random
from typing import List, Dict, Generator, Tuple

from colorama import init, Fore, Style
init(autoreset=True)
OK = Fore.GREEN + "[OK]" + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
ERR = Fore.RED + "[ERR]" + Style.RESET_ALL
INFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL


def iter_aug_sample_qa_files(root: str, include_aug: bool, include_defect: bool) -> Generator[str, None, None]:
    """
    Yield paths to qa.jsonl files in augmentation and/or defect sample directories.
    """
    if include_aug:
        for class_entry in os.scandir(root):
            if not class_entry.is_dir() or class_entry.name.startswith("."):
                continue
            aug_dir = os.path.join(class_entry.path, "aug")
            if not os.path.isdir(aug_dir):
                continue
            for sample_entry in os.scandir(aug_dir):
                if sample_entry.is_dir():
                    qa_path = os.path.join(sample_entry.path, "qa.jsonl")
                    if os.path.isfile(qa_path):
                        yield qa_path
    if include_defect:
        defect_root = os.path.join(root, "defect")
        if os.path.isdir(defect_root):
            for class_def in os.scandir(defect_root):
                if not class_def.is_dir() or class_def.name.startswith("."):
                    continue
                for sample_entry in os.scandir(class_def.path):
                    if sample_entry.is_dir():
                        qa_path = os.path.join(sample_entry.path, "qa.jsonl")
                        if os.path.isfile(qa_path):
                            yield qa_path

def stream_records(qa_path: str) -> Generator[Dict, None, None]:
    """
    Stream JSONL records from a qa.jsonl file, skipping blank/comment lines.
    """
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
    """
    Light validation:
      - id: str
      - images: list[str] (now absolute paths after transformation)
      - conversations: list[ {from, value} ] with >=2 entries
      - image files exist
    """
    if not isinstance(rec, dict):
        return False, "record not a dict"
    if "id" not in rec or not isinstance(rec["id"], str):
        return False, "missing or invalid id"
    if "images" not in rec or not isinstance(rec["images"], list) or not all(isinstance(x, str) for x in rec["images"]):
        return False, "missing or invalid images"
    if "conversations" not in rec or not isinstance(rec["conversations"], list) or len(rec["conversations"]) < 2:
        return False, "missing or invalid conversations"
    for img_path in rec["images"]:
        # Images are now absolute; if not, resolve relative to root.
        p = img_path if os.path.isabs(img_path) else os.path.join(root, img_path)
        if not os.path.isfile(p):
            return False, f"image not found: {img_path}"
    return True, "ok"

def merge(
    root: str,
    include_aug: bool,
    include_defect: bool,
    shuffle: bool,
    limit: int,
    validate: bool,
    strict: bool,
    seed: int | None
) -> List[Dict]:
    """
    Collect and optionally shuffle / limit records.
    Converts every image path to an absolute path rooted at 'root'.
    """
    all_records: List[Dict] = []
    seen_ids: dict[str, int] = {}

    qa_files = list(iter_aug_sample_qa_files(root, include_aug, include_defect))
    if not qa_files:
        print(f"{WARN} No qa.jsonl files found.")
        return []

    print(f"{INFO} Found {len(qa_files)} qa.jsonl source files.")

    for path in qa_files:
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

            # Transform image paths -> absolute
            imgs = rec.get("images", [])
            abs_imgs = []
            for img in imgs:
                # Normalize separators, then join if not absolute
                p = img.replace("\\", "/")
                if not os.path.isabs(p):
                    p = os.path.abspath(os.path.join(root, p))
                abs_imgs.append(p)
            rec["images"] = abs_imgs

            if validate:
                ok, msg = validate_record(rec, root)
                if not ok:
                    if strict:
                        raise ValueError(f"Validation failed ({msg}) for record id={rec.get('id')} in {path}")
                    else:
                        print(f"{WARN} Validation skip id={rec.get('id')} ({msg})")
                        continue
            all_records.append(rec)
        print(f"{OK} {os.path.relpath(path, root)}: loaded {count_in_file} record(s)")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_records)
        print(f"{INFO} Shuffled {len(all_records)} records (seed={seed}).")

    if limit > 0 and len(all_records) > limit:
        all_records = all_records[:limit]
        print(f"{INFO} Limited to first {limit} records.")

    print(f"{OK} Aggregated total {len(all_records)} records (unique ids incl. dup suffixes).")
    return all_records

def write_jsonl(records: List[Dict], out_path: str) -> None:
    """
    Write list of JSON objects to a JSONL file.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

def main():
    ap = argparse.ArgumentParser(description="Combine all per-sample qa.jsonl files into one JSONL.")
    ap.add_argument("-r", "--root", default=".", help="Dataset root (with_label). Default: current dir")
    ap.add_argument("-o", "--out", default="qa.jsonl", help="Output JSONL filename (placed under root). Default: qa.jsonl")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle merged records.")
    ap.add_argument("--seed", type=int, default=None, help="Shuffle seed (optional).")
    ap.add_argument("--limit", type=int, default=0, help="Keep only first N records after optional shuffle.")
    ap.add_argument("--validate", action="store_true", help="Light validation (ids, images exist, conversations).")
    ap.add_argument("--strict", action="store_true", help="Fail (raise) on validation error instead of skipping.")
    ap.add_argument("--no-aug", dest="include_aug", action="store_false", help="Exclude augmentation samples.")
    ap.add_argument("--no-defect", dest="include_defect", action="store_false", help="Exclude defect samples.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    include_aug = args.include_aug
    include_defect = args.include_defect
    if not include_aug and not include_defect:
        print(f"{ERR} Both augmentation and defect sources disabled; nothing to do.")
        return

    records = merge(
        root=root,
        include_aug=include_aug,
        include_defect=include_defect,
        shuffle=args.shuffle,
        limit=args.limit,
        validate=args.validate,
        strict=args.strict,
        seed=args.seed
    )

    out_path = os.path.join(root, args.out)
    write_jsonl(records, out_path)
    print(f"{OK} Wrote consolidated JSONL: {out_path}")

if __name__ == "__main__":
    main()