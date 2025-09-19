#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_gt_analysis.py
- Scan the new dataset root (/mnt/workspace/autorepair_vlm/gt_datasets_20250915)
- For each sample directory that DOES NOT contain analysis_gt.json,
  * read repair_rule.json,
  * inject into the Chinese request prompt template,
  * run inference on repair_image.jpg using the specified Qwen2.5-VL model,
  * save the output to analysis_rehearsal7b.json in that sample directory.

Safe to re-run: skips samples that already have the out file (unless --overwrite).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

TEMPLATE = (
    "如图所示是一张半导体显示面板图像,"
    "蓝色曲线内区域为缺陷，算法根据输入的缺陷位置及对应的修补信息输出了需要修补的路径，红色直线代表模拟的激光切割的路径，"
    "黄绿色直线（较淡较宽）代表模拟的ITO remove的路径，两种路径有可能有重合。"
    "你需要检查图上的可视化路径是否符合以下修补规则：{repair_rule}"
    "请输出两个内容：1.判断图中可视化切割线是否符合所述修补规则，回答符合或者不符合；"
    "2.如果不符合，请说出不符合的地方。"
    "输出格式请参考以下示例(需包含规则确认、图像分析、应用规则、结论、理由)："
)

def build_rules_text(rule_json_path: Path) -> Optional[str]:
    """
    rule_json format:
    [
      {
        "damaged_component":"Drain",
        "rules":[{"repair_rule": "...", "operations":[...], "repair_components":[...]}]
      },
      ...
    ]
    We convert to: "对于Drain缺陷，<rule1>；<rule2>；；对于TFT缺陷，<rule1>；..."
    """
    try:
        data = json.loads(rule_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read/parse {rule_json_path}: {e}")
        return None

    if not isinstance(data, list) or len(data) == 0:
        return None

    chunks: List[str] = []
    for item in data:
        comp = item.get("damaged_component")
        rules = item.get("rules", [])
        if not comp or not isinstance(rules, list) or len(rules) == 0:
            continue
        rule_texts = []
        for r in rules:
            t = r.get("repair_rule")
            if t:
                # Trim trailing punctuation to avoid doubling "。"
                rule_texts.append(str(t).strip())
        if rule_texts:
            # join rules with Chinese semicolon
            joined = "；".join(rule_texts)
            chunks.append(f"对于{comp}缺陷，{joined}")
    return "；".join(chunks) if chunks else None

def make_messages(image_path: Path, prompt_text: str) -> List[Dict[str, Any]]:
    """
    Qwen2.5-VL chat message format.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

def infer_one(
    model, processor, messages: List[Dict[str, Any]],
    max_new_tokens: int = 512, temperature: float = 0.0, top_p: float = 0.9, device: str = "cuda"
) -> str:
    """
    Run a single inference and return the decoded text.
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Qwen2.5-VL automatically picks up image paths from messages; we still pass the text string.
    inputs = processor(text=[text], images=None, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    result = processor.batch_decode(out, skip_special_tokens=True)[0]
    # Remove the chat template prefix if present (model may echo prompt).
    return result.strip()

def find_samples(root: Path) -> List[Path]:
    """
    Return a list of sample directories under root/positive/*/* and root/negative/*/*.
    A "sample directory" is any leaf directory at depth >= 4 (polarity/category/sample)
    that contains a 'repair_rule.json' (to ensure it's a valid sample).
    """
    samples: List[Path] = []
    for polarity in ("negative", "positive"):
        pdir = root / polarity
        if not pdir.is_dir():
            continue
        for category_dir in sorted(d for d in pdir.iterdir() if d.is_dir()):
            for sample_dir in sorted(d for d in category_dir.iterdir() if d.is_dir()):
                if (sample_dir / "repair_rule.json").is_file():
                    samples.append(sample_dir)
    return samples

def main():
    ap = argparse.ArgumentParser(description="Run Qwen2.5-VL on samples missing analysis_gt.json and save outputs.")
    ap.add_argument("--root", type=str, default="/mnt/workspace/autorepair_vlm/gt_datasets_20250915",
                    help="Dataset root (reorganized).")
    ap.add_argument("--model", type=str, default="/mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_2/stage2/checkpoint-155",
                    help="Path to model checkpoint.")
    ap.add_argument("--out-name", type=str, default="analysis_rehearsal7b.json",
                    help="Name of output JSON to write per sample.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing out file if present.")
    ap.add_argument("--dry-run", action="store_true", help="Do not run model; just print planned actions.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"[ERROR] Root not found: {root}")

    # Collect candidates: missing analysis_gt.json only
    all_samples = find_samples(root)
    targets: List[Path] = []
    for s in all_samples:
        if not (s / "analysis_gt.json").is_file():
            targets.append(s)

    if not targets:
        print("[INFO] No samples missing analysis_gt.json. Nothing to do.")
        return

    print(f"[INFO] Found {len(targets)} sample(s) missing analysis_gt.json.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dry_run:
        print("[DRY] Skipping model load.")
        model = processor = None
    else:
        print(f"[INFO] Loading model on {device}: {args.model}")
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    processed = 0
    skipped = 0
    for sample in targets:
        out_path = sample / args.out_name
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {out_path}")
            skipped += 1
            continue

        # Required inputs
        repair_img = sample / "repair_image.jpg"
        rule_json = sample / "repair_rule.json"

        if not repair_img.is_file():
            print(f"[WARN] Missing repair_image.jpg in {sample}, skipping.")
            skipped += 1
            continue
        if not rule_json.is_file():
            print(f"[WARN] Missing repair_rule.json in {sample}, skipping.")
            skipped += 1
            continue

        # Build prompt
        rules_text = build_rules_text(rule_json)
        if not rules_text:
            print(f"[WARN] Could not build rules from {rule_json}, skipping.")
            skipped += 1
            continue
        prompt = TEMPLATE.format(repair_rule=rules_text)

        print(f"[RUN ] {sample}")
        if args.dry_run:
            print(f"       Would read: {repair_img.name}, {rule_json.name}")
            print(f"       Would write: {out_path.name}")
            processed += 1
            continue

        # Messages and inference
        messages = make_messages(repair_img, prompt)
        try:
            response_text = infer_one(
                model, processor, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
        except Exception as e:
            print(f"[ERR ] Inference failed for {sample}: {e}")
            skipped += 1
            continue

        # Save JSON output
        payload = {
            "model": str(args.model),
            "prompt": prompt,
            "response": response_text,
        }
        try:
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK  ] Wrote {out_path}")
            processed += 1
        except Exception as e:
            print(f"[ERR ] Failed to write {out_path}: {e}")
            skipped += 1

    print(f"\n[SUMMARY] processed={processed}, skipped={skipped}, total={len(targets)}")

if __name__ == "__main__":
    main()
