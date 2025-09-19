"""
Follow up on eval.py
Parse evaluation JSON (array of OpenAI-style response JSON strings)
Extract:
  - Accuracy percentage (Acc: 1 counts as correct)
  - Average Match value
Write results to result.json (or custom --out).
Usage:
  python ../../../src/eval_repair/eval_compute.py \
    --in ./gpt5_eval_qwen_dual_rehearsal_pred_7b_250.json \
    --out ./gpt5_result_qwen_dual_rehearsal_pred_7b_250.json
"""
import json
import re
import argparse
from typing import List, Tuple

ACC_RE = re.compile(r'Acc:\s*([01])')
MATCH_RE = re.compile(r'Match:\s*([0-9]+(?:\.[0-9]+)?)')


def extract_metrics(text: str) -> Tuple[int | None, float | None]:
    """
    Return (acc_int, match_float) or (None, None) if not found.
    Accepts flexible formatting (periods, newlines, extra spaces).
    """
    acc_m = ACC_RE.search(text)
    match_m = MATCH_RE.search(text)
    acc = int(acc_m.group(1)) if acc_m else None
    match = float(match_m.group(1)) if match_m else None
    return acc, match

def parse_file(path: str) -> List[Tuple[int, float]]:
    """
    Load outer JSON array. Each element is a JSON string of an OpenAI response object.
    Extract metrics from choices[0].message.content.
    Returns list of (acc, match).
    """
    with open(path, "r", encoding="utf-8") as f:
        outer = json.load(f)
    results: List[Tuple[int, float]] = []
    for idx, item in enumerate(outer):
        # Each item is a JSON string
        try:
            obj = json.loads(item)
        except Exception as e:
            print(f"[WARN] Skipping item {idx}: cannot parse inner JSON ({e})")
            continue
        try:
            content = obj["choices"][0]["message"]["content"]
        except Exception:
            print(f"[WARN] Skipping item {idx}: missing content")
            continue
        acc, match = extract_metrics(content)
        if acc is None or match is None:
            print(f"[WARN] Skipping item {idx}: metrics not found")
            continue
        results.append((acc, match))
    return results

def compute_stats(pairs: List[Tuple[int, float]]) -> dict:
    if not pairs:
        return {
            "total_samples": 0,
            "accuracy": 0.0,
            "accuracy_percent": 0.0,
            "average_match": 0.0
        }
    total = len(pairs)
    correct = sum(a for a, _ in pairs)
    avg_match = sum(m for _, m in pairs) / total
    return {
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
        "accuracy_percent": (correct / total) * 100.0,
        "average_match": avg_match
    }

def main():
    ap = argparse.ArgumentParser(description="Compute accuracy and average Match from evaluation JSON.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON file (array of JSON strings).")
    ap.add_argument("--out", dest="out_path", default="result.json", help="Output JSON file.")
    args = ap.parse_args()

    pairs = parse_file(args.in_path)
    stats = compute_stats(pairs)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] Parsed {stats['total_samples']} samples.")
    print(f"     Accuracy: {stats['accuracy_percent']:.2f}%  Avg Match: {stats['average_match']:.4f}")
    print(f"     Saved -> {args.out_path}")

if __name__ == "__main__":
    main()