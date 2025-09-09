"""
Mix two VQA datasets into a single JSONL (mixed_qa.jsonl by default):

1) qa.jsonl (line JSON) produced by [qa_gt_generator.py](http://_vscodecontentref_/0) (fields: id, images, conversations[, meta])
2) A long-form JSON file (array) with entries:
   {
     "messages": [{ "role": "user"/"assistant"/... , "content": "..."}, ...],
     "images": [ "path1", ... ]
   }

This script:
- Loads & (optionally) subsamples qa.jsonl (forward dataset)
- Loads & (optionally) subsamples the long-form JSON
- Converts long-form "messages" to "conversations" using role mapping:
      user -> human
      assistant -> gpt
  (system/other roles dropped)
- Ensures at least one user+assistant pair and final message is from assistant; else skips entry.
- Normalizes schema: drops any "meta" field so all records share: {id, images, conversations}
- Shuffles within each dataset before limiting (for unbiased sampling), then (optionally) shuffles final merged list.
- Writes line-delimited JSON to output.

Usage example:
  python qa_mixer.py \
      --qa-jsonl /mnt/workspace/kennethliu/TFT_circuit_images/with_label/qa.jsonl \
      --complex-json /mnt/workspace/yangsidi/LLaMA-Factory/data/TCL_auto_repair_20250825.json \
      --qa-limit 5000 \
      --complex-limit 1000 \
      --shuffle \
      --seed 42 \
      --out mixed_qa.jsonl

Arguments:
  --qa-jsonl PATH          Path to qa.jsonl (default provided)
  --complex-json PATH      Path to long-form JSON (array)
  --qa-limit N             Keep up to N samples from qa.jsonl (after internal shuffle)
  --complex-limit N        Keep up to N samples from complex JSON (after internal shuffle)
  --seed S                 RNG seed for reproducibility
  --shuffle                Shuffle final merged records
  --make-image-abs         Convert image paths to absolute (root-based if --image-root given)
  --image-root PATH        Root to prepend for relative image paths when --make-image-abs
  --out PATH               Output JSONL (default: mixed_qa.jsonl)
  --min-turns N            Minimum total conversation turns required (default 2)
  --strict                 Raise on malformed lines in qa.jsonl instead of skipping

Notes:
- If qa-limit or complex-limit <=0 they are treated as "no limit".
- IDs for complex JSON entries are auto-assigned: complex_{index}
- Existing IDs in qa.jsonl are preserved (duplicates not de-duplicated).
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any, Iterable, Tuple

def load_qa_jsonl(path: str, strict: bool, drop_meta: bool, seed: int | None, limit: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                if strict:
                    raise ValueError(f"Malformed JSON at {path}:{ln}: {e}")
                continue
            if drop_meta and "meta" in obj:
                del obj["meta"]
            # Basic sanity
            if "conversations" not in obj or "images" not in obj:
                continue
            records.append(obj)
    rng = random.Random(seed)
    rng.shuffle(records)
    if limit > 0 and len(records) > limit:
        records = records[:limit]
    return records

def normalize_messages(messages: List[Dict[str, Any]], min_turns: int) -> List[Dict[str, str]]:
    """
    Convert list of {role, content} to list of {from, value}.
    Drops system/other roles. Merges nothing; keeps order.
    Ensures last is assistant (gpt) and total turns >= min_turns.
    """
    conv = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if not content or not isinstance(content, str):
            continue
        if role == "user":
            conv.append({"from": "human", "value": content})
        elif role == "assistant":
            conv.append({"from": "gpt", "value": content})
        else:
            # skip system/tool etc.
            continue
    if len(conv) < min_turns:
        return []
    if conv[-1]["from"] != "gpt":
        return []
    # Optionally ensure at least one human and one gpt
    if not any(c["from"] == "human" for c in conv) or not any(c["from"] == "gpt" for c in conv):
        return []
    return conv

def load_complex_json(path: str, seed: int | None, limit: int, min_turns: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Complex JSON root must be a list.")
    recs: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        messages = item.get("messages")
        images = item.get("images")
        if not isinstance(messages, list) or not isinstance(images, list) or not images:
            continue
        conv = normalize_messages(messages, min_turns=min_turns)
        if not conv:
            continue
        recs.append({
            "id": f"complex_{idx}",
            "images": images,
            "conversations": conv
        })
    rng = random.Random(seed)
    rng.shuffle(recs)
    if limit > 0 and len(recs) > limit:
        recs = recs[:limit]
    return recs

def make_images_absolute(records: List[Dict[str, Any]], root: str) -> None:
    if not root:
        return
    for r in records:
        new_imgs = []
        for p in r.get("images", []):
            if not os.path.isabs(p):
                new_imgs.append(os.path.abspath(os.path.join(root, p)))
            else:
                new_imgs.append(p)
        r["images"] = new_imgs

def write_jsonl(records: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mix qa.jsonl with complex multi-turn JSON dataset.")
    ap.add_argument("--qa-jsonl", default="/mnt/workspace/kennethliu/TFT_circuit_images/with_label/qa.jsonl",
                    help="Path to qa.jsonl")
    ap.add_argument("--complex-json", default="/mnt/workspace/yangsidi/LLaMA-Factory/data/TCL_auto_repair_20250825.json", help="Path to complex JSON (array format).")
    ap.add_argument("--qa-limit", type=int, default=0, help="Max QA jsonl samples (0 = no limit).")
    ap.add_argument("--complex-limit", type=int, default=0, help="Max complex samples (0 = no limit).")
    ap.add_argument("--min-turns", type=int, default=2, help="Min conversation turns required in complex samples.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle final merged set.")
    ap.add_argument("--drop-meta", action="store_true", help="Drop any meta field from qa.jsonl records.")
    ap.add_argument("--make-image-abs", action="store_true", help="Convert image paths to absolute.")
    ap.add_argument("--image-root", default="", help="Root used when making image paths absolute.")
    ap.add_argument("--out", default="mixed_qa.jsonl", help="Output JSONL path.")
    ap.add_argument("--strict", action="store_true", help="Strict parsing for qa.jsonl.")
    return ap.parse_args()

def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Load datasets
    qa_recs = load_qa_jsonl(
        path=args.qa_jsonl,
        strict=args.strict,
        drop_meta=args.drop_meta,
        seed=args.seed,
        limit=args.qa_limit
    )
    complex_recs = load_complex_json(
        path=args.complex_json,
        seed=args.seed,
        limit=args.complex_limit,
        min_turns=args.min_turns
    )

    print(f"[INFO] Loaded qa.jsonl records: {len(qa_recs)} (limit={args.qa_limit or 'none'})")
    print(f"[INFO] Loaded complex JSON records: {len(complex_recs)} (limit={args.complex_limit or 'none'})")

    all_recs = qa_recs + complex_recs
    if args.shuffle:
        rng.shuffle(all_recs)
        print(f"[INFO] Shuffled combined records: {len(all_recs)}")

    # Make image paths absolute if requested
    if args.make_image_abs:
        make_images_absolute(all_recs, args.image_root or "")

    write_jsonl(all_recs, args.out)
    print(f"[OK] Wrote mixed dataset: {args.out} (total {len(all_recs)} records)")

if __name__ == "__main__":
    main()