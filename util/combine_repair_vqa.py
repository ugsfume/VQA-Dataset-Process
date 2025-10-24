#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine_repair_vqa.py

Build VQA samples from the *new* dataset at:
  /mnt/workspace/autorepair_vlm/gt_datasets_20250915 (recursively)

Each valid sample dir must contain:
  - analysis_gt.json OR remix_analysis_gt.json
    (controlled by --analysis {analysis,remix,both})
  - repair_rule.json (provides Q—rules to inject into the prompt)
  - repair_image.jpg or repair_image.png (provides V)

Then (optionally) append to the existing VQA JSON:
  /mnt/workspace/yangsidi/LLaMA-Factory/data/TCL_auto_repair_20250825.json
  (use --no-old to skip)

Save combined JSON to:
  /mnt/workspace/kennethliu/data/repair_dataset_20251010.json

Optionally split into train/test:
  /mnt/workspace/kennethliu/data/repair_train_dataset_20251010.json
  /mnt/workspace/kennethliu/data/repair_test_dataset_20251010.json

Usage:
  # default: use analysis_gt.json + include old JSON if present
  python combine_repair_vqa.py

  # use remix_analysis_gt.json
  python combine_repair_vqa.py --analysis remix

  # include both analysis types (adds one VQA item per type found)
  python combine_repair_vqa.py --analysis both

  # skip the old JSON
  python combine_repair_vqa.py --no-old

  # with split & verbose
  python combine_repair_vqa.py --analysis both --split --train-ratio 0.8 --seed 42 --verbose
"""
import os
import json
import argparse
import random
from typing import List, Dict, Any, Tuple
from colorama import init as colorama_init, Fore, Style

# ---- Colorized CLI tags ----
colorama_init(autoreset=True)
OK   = Fore.GREEN  + "[OK]"   + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
ERR  = Fore.RED    + "[ERR]"  + Style.RESET_ALL
INFO = Fore.CYAN   + "[INFO]" + Style.RESET_ALL

DEFAULT_NEW_ROOT   = "/mnt/workspace/autorepair_vlm/gt_datasets_20250915"
DEFAULT_OLD_VQA    = "/mnt/workspace/yangsidi/LLaMA-Factory/data/TCL_auto_repair_20250825.json"
DEFAULT_OUT        = "/mnt/workspace/kennethliu/data/repair_dataset_20251017.json"
DEFAULT_TRAIN_OUT  = "/mnt/workspace/kennethliu/data/repair_train_dataset_20251017.json"
DEFAULT_TEST_OUT   = "/mnt/workspace/kennethliu/data/repair_test_dataset_20251017.json"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def combine_repair_rules_text(rules_json: List[Dict[str, Any]]) -> str:
    """
    Turn repair_rule.json content into a single Chinese string like:
    "对于<damaged_component>缺陷，<rule1> <rule2> …；对于<next>缺陷，…"
    Only uses the 'repair_rule' field from each rule entry.
    """
    parts = []
    for item in rules_json:
        comp = item.get("damaged_component", "")
        rules = item.get("rules", [])
        rule_texts = [r.get("repair_rule", "").strip() for r in rules if r.get("repair_rule")]
        if not rule_texts:
            continue
        parts.append(f"对于{comp}缺陷，" + " ".join(rule_texts))
    return "；".join(parts)


def count_image_placeholders(messages: List[Dict[str, str]]) -> int:
    return sum(m.get("content", "").count("<image>") for m in messages)


def find_repair_image(sample_dir: str) -> str:
    # Prefer JPG if both exist
    jpg = os.path.join(sample_dir, "repair_image.jpg")
    png = os.path.join(sample_dir, "repair_image.png")
    if os.path.isfile(jpg):
        return jpg
    if os.path.isfile(png):
        return png
    return ""


def build_vqa_for_sample(sample_dir: str, analysis_choice: str, verbose: bool=False) -> Dict[str, Any]:
    """
    Build one VQA item for a sample folder. Returns {} if sample is incomplete.
    analysis_choice: "analysis" -> analysis_gt.json, "remix" -> remix_analysis_gt.json
    """
    analysis_file = "analysis_gt.json" if analysis_choice == "analysis" else "remix_analysis_gt.json"
    path_analysis = os.path.join(sample_dir, analysis_file)
    path_rules    = os.path.join(sample_dir, "repair_rule.json")
    img_path      = find_repair_image(sample_dir)

    if not os.path.isfile(path_analysis):
        if verbose:
            print(f"{WARN} Missing {analysis_file} → skip: {sample_dir}")
        return {}
    if not os.path.isfile(path_rules):
        if verbose:
            print(f"{WARN} Missing repair_rule.json → skip: {sample_dir}")
        return {}
    if not img_path:
        if verbose:
            print(f"{WARN} Missing repair_image.(jpg|png) → skip: {sample_dir}")
        return {}

    try:
        analysis = load_json(path_analysis)
        rules    = load_json(path_rules)
    except Exception as e:
        if verbose:
            print(f"{WARN} Failed to load json in {sample_dir}: {e}")
        return {}

    # Compose repair rules string
    repair_rules_text = combine_repair_rules_text(rules)

    # User message (inject rules)
    user_content = (
        "<image>如图所示是一张半导体显示面板图像，蓝色曲线内区域为缺陷，"
        "算法根据输入的缺陷位置及对应的修补信息输出了需要修补的路径，"
        "红色直线代表模拟的激光切割的路径，黄绿色直线（较淡较宽）代表模拟的ITO remove的路径，"
        f"两种路径有可能有重合。你需要检查图上的可视化路径是否符合以下修补规则：{repair_rules_text}。"
        "请输出两个内容：1.判断图中可视化切割线是否符合所述修补规则，回答符合或者不符合；"
        "2.如果不符合，请说出不符合的地方。输出格式请参考以下示例(需包含规则确认、图像分析、应用规则、结论、理由)："
    )

    # Assistant message
    result        = analysis.get("result", "").strip()
    analysis_text = analysis.get("analysis", "").strip()
    assistant_content = f"结果: {result}\n\n分析报告:\n{analysis_text}"

    messages = [
        {"content": user_content, "role": "user"},
        {"content": assistant_content, "role": "assistant"},
    ]

    # Image placeholders
    n_img   = count_image_placeholders(messages)
    abs_img = os.path.abspath(img_path)
    images  = [abs_img] * n_img if n_img > 0 else [abs_img]

    return {"messages": messages, "images": images}


def collect_new_vqa(root: str, analysis_choice: str, verbose: bool=False) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    """
    Walk root recursively, building VQAs from valid sample folders.

    Returns:
      vqas: list of generated items
      skipped: count of skipped attempts (missing files, load errors, etc.)
      counts: per-source counts, e.g. {"analysis": 123, "remix": 45}
    """
    vqas: List[Dict[str, Any]] = []
    skipped = 0
    counts = {"analysis": 0, "remix": 0}

    if analysis_choice in ("analysis", "remix"):
        analysis_file = "analysis_gt.json" if analysis_choice == "analysis" else "remix_analysis_gt.json"
        for dirpath, dirnames, filenames in os.walk(root):
            if analysis_file not in filenames:
                continue
            item = build_vqa_for_sample(dirpath, analysis_choice, verbose=verbose)
            if item:
                vqas.append(item)
                counts[analysis_choice] += 1
                if verbose:
                    print(f"{OK} Added VQA from: {dirpath} ({analysis_file})")
            else:
                skipped += 1
        return vqas, skipped, counts

    # analysis_choice == "both"
    for dirpath, dirnames, filenames in os.walk(root):
        has_analysis = "analysis_gt.json" in filenames
        has_remix    = "remix_analysis_gt.json" in filenames
        if not (has_analysis or has_remix):
            continue

        if has_analysis:
            item_a = build_vqa_for_sample(dirpath, "analysis", verbose=verbose)
            if item_a:
                vqas.append(item_a)
                counts["analysis"] += 1
                if verbose:
                    print(f"{OK} Added VQA from: {dirpath} (analysis_gt.json)")
            else:
                skipped += 1

        if has_remix:
            item_r = build_vqa_for_sample(dirpath, "remix", verbose=verbose)
            if item_r:
                vqas.append(item_r)
                counts["remix"] += 1
                if verbose:
                    print(f"{OK} Added VQA from: {dirpath} (remix_analysis_gt.json)")
            else:
                skipped += 1

    return vqas, skipped, counts


def load_existing_vqa(path: str, verbose: bool=False) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        if verbose:
            print(f"{WARN} Existing VQA not found: {path} (will combine new only)")
        return []
    try:
        data = load_json(path)
        if isinstance(data, list):
            return data
        if verbose:
            print(f"{WARN} Existing VQA is not a list: {path} (ignoring)")
        return []
    except Exception as e:
        if verbose:
            print(f"{WARN} Failed to load existing VQA: {e}")
        return []


def maybe_split(items: List[Dict[str, Any]], train_ratio: float, seed: int=None) -> Tuple[List, List]:
    if seed is not None:
        random.seed(seed)
    items_shuffled = items[:]
    random.shuffle(items_shuffled)
    n = len(items_shuffled)
    k = int(round(n * train_ratio))
    return items_shuffled[:k], items_shuffled[k:]


def main():
    ap = argparse.ArgumentParser(description="Combine old + new repair VQA datasets.")
    ap.add_argument("--new-root", type=str, default=DEFAULT_NEW_ROOT,
                    help="Root to recursively search for new samples (default: gt_datasets_20250915).")
    ap.add_argument("--old", type=str, default=DEFAULT_OLD_VQA,
                    help="Existing VQA json to append to (default: TCL_auto_repair_20250825.json).")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT,
                    help="Output combined VQA path.")
    ap.add_argument("--split", action="store_true",
                    help="Also write train/test splits.")
    ap.add_argument("--train-out", type=str, default=DEFAULT_TRAIN_OUT,
                    help="Train split json output path.")
    ap.add_argument("--test-out", type=str, default=DEFAULT_TEST_OUT,
                    help="Test split json output path.")
    ap.add_argument("--train-ratio", type=float, default=0.8,
                    help="Train ratio (default 0.8).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Shuffle seed for split.")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose logging.")

    # Choose analysis source: analysis_gt.json, remix_analysis_gt.json, or both
    ap.add_argument("--analysis", choices=["analysis", "remix", "both"], default="analysis",
                    help="Which analysis JSON(s) to use per sample. "
                         "'analysis' -> analysis_gt.json, 'remix' -> remix_analysis_gt.json, "
                         "'both' -> include one item for each file found (default: analysis).")

    # Simple toggle: include old by default; use --no-old to skip
    ap.add_argument("--no-old", dest="include_old", action="store_false",
                    help="Do NOT include the existing VQA JSON.")
    ap.set_defaults(include_old=True)

    args = ap.parse_args()

    print(f"{INFO} New root: {args.new_root}")
    print(f"{INFO} Analysis source: {args.analysis}")
    print(f"{INFO} Old VQA : {args.old} ({'included' if args.include_old else 'skipped'})")
    print(f"{INFO} Out path: {args.out}")

    # 1) Collect new VQAs
    new_vqa, skipped, counts = collect_new_vqa(args.new_root, args.analysis, verbose=args.verbose)
    print(f"{OK} New VQAs collected: {len(new_vqa)} "
          f"(analysis: {counts.get('analysis',0)}, remix: {counts.get('remix',0)}, skipped: {skipped})")

    # 2) Load existing (conditionally)
    if args.include_old:
        old_vqa = load_existing_vqa(args.old, verbose=args.verbose)
        print(f"{OK} Existing VQAs loaded: {len(old_vqa)}")
    else:
        old_vqa = []
        if args.verbose:
            print(f"{INFO} Skipping load of existing VQA per --no-old")

    # 3) Combine
    combined = old_vqa + new_vqa
    print(f"{OK} Combined total: {len(combined)}")

    # 4) Save combined
    save_json(combined, args.out)
    print(f"{OK} Wrote combined dataset → {args.out}")

    # 5) Optional split
    if args.split:
        train, test = maybe_split(combined, args.train_ratio, seed=args.seed)
        save_json(train, args.train_out)
        save_json(test, args.test_out)
        print(f"{OK} Wrote train ({len(train)}) → {args.train_out}")
        print(f"{OK} Wrote test  ({len(test)})  → {args.test_out}")


if __name__ == "__main__":
    main()
