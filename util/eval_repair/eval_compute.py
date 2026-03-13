"""
Parse evaluation JSON
Extract:
  - Accuracy percentage
  - Average Match value
  - Confusion Matrix
Usage:
  python ../../../src/eval_repair/eval_compute.py \
    --in ./gpt5_eval_grpo_20260220_qwen3vl-8b_dual_white_fixed_repair_20260217_global_step_100_merged.json \
    --out ./gpt5_result_grpo_20260220_qwen3vl-8b_dual_white_fixed_repair_20260217_global_step_100_merged.json
"""
import argparse
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Dict

from colorama import init, Fore, Style

init(autoreset=True)
OK = Fore.GREEN + "[OK]" + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
ERR = Fore.RED + "[ERR]" + Style.RESET_ALL
INFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL

ACC_RE = re.compile(r"\bAcc:\s*([01])\b", re.IGNORECASE)
MATCH_RE = re.compile(r"\bMatch:\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)
CLASS_RE = re.compile(r"\bClass:\s*(TP|TN|FP|FN)\b", re.IGNORECASE)
CLASS_PAREN_RE = re.compile(r"\((TP|TN|FP|FN)\)", re.IGNORECASE)

VALID_CLASSES = ("TP", "TN", "FP", "FN")


@dataclass
class EvalResult:
    acc: int
    match: float
    cls: str  # TP/TN/FP/FN/UNK


def _extract_class(text: str) -> Optional[str]:
    m = CLASS_RE.search(text)
    if m:
        return m.group(1).upper()
    m = CLASS_PAREN_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


def extract_metrics(text: str) -> Optional[EvalResult]:
    """
    Extract (acc, match, cls) from model output text.
    Returns None if Acc/Match missing; Class may be UNK if not found.
    """
    acc_m = ACC_RE.search(text)
    match_m = MATCH_RE.search(text)
    if not acc_m or not match_m:
        return None
    acc = int(acc_m.group(1))
    match = float(match_m.group(1))
    cls = _extract_class(text) or "UNK"
    return EvalResult(acc=acc, match=match, cls=cls)


def parse_file(path: str) -> List[EvalResult]:
    """
    Load outer JSON array. Each element is a JSON string of an OpenAI response object.
    Extract metrics from choices[0].message.content.
    Returns list of EvalResult.
    """
    with open(path, "r", encoding="utf-8") as f:
        outer = json.load(f)

    results: List[EvalResult] = []
    for idx, item in enumerate(outer):
        if not isinstance(item, str):
            print(f"{WARN} Skipping item {idx}: outer element is not a JSON string")
            continue

        # Each item is a JSON string
        try:
            obj = json.loads(item)
        except Exception as e:
            print(f"{WARN} Skipping item {idx}: cannot parse inner JSON ({e})")
            continue

        try:
            content = obj["choices"][0]["message"]["content"]
        except Exception:
            print(f"{WARN} Skipping item {idx}: missing choices[0].message.content")
            continue

        r = extract_metrics(str(content))
        if r is None:
            print(f"{WARN} Skipping item {idx}: Acc/Match not found in content")
            continue

        # Normalize cls
        if r.cls != "UNK" and r.cls not in VALID_CLASSES:
            r.cls = "UNK"

        results.append(r)

    return results


def _safe_div(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def compute_stats(results: List[EvalResult]) -> Dict:
    if not results:
        return {
            "total_samples": 0,
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "accuracy_percent": 0.0,
            "average_match": 0.0,
            "confusion_matrix": {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "UNK": 0},
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "pos_label": "符合",
            "neg_label": "不符合",
        }

    total = len(results)
    correct = sum(r.acc for r in results)
    avg_match = sum(r.match for r in results) / total

    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "UNK": 0}
    for r in results:
        cm[r.cls if r.cls in cm else "UNK"] += 1

    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
        "accuracy_percent": (correct / total) * 100.0,
        "average_match": avg_match,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pos_label": "符合",
        "neg_label": "不符合",
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compute Accuracy/Match + Confusion Matrix + Precision/Recall/F1 from evaluation JSON."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON file (array of JSON strings).")
    ap.add_argument("--out", dest="out_path", default="result.json", help="Output JSON file.")
    args = ap.parse_args()

    results = parse_file(args.in_path)
    stats = compute_stats(results)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    cm = stats["confusion_matrix"]
    print(f"{OK} Parsed {stats['total_samples']} samples.")
    print(f"     Accuracy: {stats['accuracy_percent']:.2f}%  Avg Match: {stats['average_match']:.4f}")
    print(f"     Confusion: TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']} UNK={cm['UNK']}")
    print(f"     Precision: {stats['precision']:.4f}  Recall: {stats['recall']:.4f}  F1: {stats['f1']:.4f}")
    print(f"{OK} Saved -> {args.out_path}")


if __name__ == "__main__":
    main()