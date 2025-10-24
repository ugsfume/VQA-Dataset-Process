#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_gt_analysis.py
- Scan the new dataset root (/mnt/workspace/autorepair_vlm/gt_datasets_20250915)
- For each sample directory that DOES NOT contain analysis_gt.json,
  * read repair_rule.json,
  * inject into the Chinese request prompt template,
  * run inference on repair_image.jpg using the specified Qwen2.5-VL model,
  * save the output to analysis_rehearsal32b.json in that sample directory.

Safe to re-run: skips samples that already have the out file (unless --overwrite).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from colorama import init as colorama_init, Fore, Style, Back
colorama_init(autoreset=True)
LBL_WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
LBL_ERROR = Fore.RED + "[ERROR]" + Style.RESET_ALL
LBL_OK = Back.GREEN + "[OK]" + Style.RESET_ALL
LBL_INFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL
LBL_SKIP = Fore.MAGENTA + "[SKIP]" + Style.RESET_ALL
LBL_DRY = Fore.BLUE + "[DRY]" + Style.RESET_ALL
LBL_SUMMARY = Back.BLUE + "[SUMMARY]" + Style.RESET_ALL
LBL_RUN = Fore.GREEN + "[RUN]" + Style.RESET_ALL

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
    We convert to:
      "对于Drain缺陷，<rule1>；对于Drain缺陷，<rule2>；对于TFT缺陷，<rule1>；..."
    """
    try:
        data = json.loads(rule_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"{LBL_WARN} Failed to read/parse {rule_json_path}: {e}")
        return None

    if not isinstance(data, list) or not data:
        return None

    clauses: List[str] = []
    for item in data:
        comp = (item.get("damaged_component") or "").strip()
        rules = item.get("rules", [])
        if not comp or not isinstance(rules, list) or not rules:
            continue

        for r in rules:
            t = r.get("repair_rule")
            if t:
                clauses.append(f"对于{comp}缺陷，{str(t).strip()}")

    return "；".join(clauses) if clauses else None


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
    model,
    processor,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    """
    Run a single inference and return the decoded text.
    Ensures all inputs are placed on the same device as the model.
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Build vision inputs properly from messages (paths -> tensors)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    )

    # Align inputs to the model device to avoid device mismatch warnings
    model_device = next(model.parameters()).device
    inputs = {k: (v.to(model_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    # EOS resolution (use processor tokenizer if available)
    eos_id = getattr(processor.tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)

    do_sample = temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_id,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    result = processor.batch_decode(out, skip_special_tokens=True)[0]
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
    ap.add_argument("--model", type=str, default="/mnt/workspace/kennethliu/ckpt/qwen2_5vl-32b_expanded_repair/stage2/checkpoint-130",
                    help="Path to model checkpoint.")
    ap.add_argument("--out-name", type=str, default="analysis_rehearsal32b.json",
                    help="Name of output JSON to write per sample.")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing out file if present.")
    ap.add_argument("--dry-run", action="store_true", help="Do not run model; just print planned actions.")
    ap.add_argument("--slow-processor", action="store_true",
                    help="Force slow image processor (reproduce legacy behavior).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"{LBL_ERROR} Root not found: {root}")

    # Collect candidates: missing analysis_gt.json only
    all_samples = find_samples(root)
    targets: List[Path] = [s for s in all_samples if not (s / "analysis_gt.json").is_file()]

    if not targets:
        print(f"{LBL_INFO} No samples missing analysis_gt.json. Nothing to do.")
        return

    print(f"{LBL_INFO} Found {len(targets)} sample(s) missing analysis_gt.json.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dry_run:
        print(f"{LBL_DRY} Skipping model load.")
        model = processor = None
    else:
        print(f"{LBL_INFO} Loading model on {device}: {args.model}")
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_fast=not args.slow_processor,  # explicit choice; avoids future default warning
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    processed = 0
    skipped = 0
    for sample in targets:
        out_path = sample / args.out_name
        if out_path.exists() and not args.overwrite:
            print(f"{LBL_SKIP} Exists: {out_path}")
            skipped += 1
            continue

        # Required inputs
        repair_img = sample / "repair_image.jpg"
        rule_json = sample / "repair_rule.json"

        if not repair_img.is_file():
            print(f"{LBL_WARN} Missing repair_image.jpg in {sample}, skipping.")
            skipped += 1
            continue
        if not rule_json.is_file():
            print(f"{LBL_WARN} Missing repair_rule.json in {sample}, skipping.")
            skipped += 1
            continue

        # Build prompt
        rules_text = build_rules_text(rule_json)
        if not rules_text:
            print(f"{LBL_WARN} Could not build rules from {rule_json}, skipping.")
            skipped += 1
            continue
        prompt = TEMPLATE.format(repair_rule=rules_text)

        print(f"{LBL_RUN} {sample}")
        if args.dry_run:
            print(f"       Would read: {repair_img.name}, {rule_json.name}")
            print(f"       Would write: {out_path.name}")
            processed += 1
            continue

        # Messages and inference
        messages = make_messages(repair_img, prompt)
        try:
            response_text = infer_one(
                model,
                processor,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
        except Exception as e:
            print(f"{LBL_ERROR} Inference failed for {sample}: {e}")
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
            print(f"{LBL_OK} Wrote {out_path}")
            processed += 1
        except Exception as e:
            print(f"{LBL_ERROR} Failed to write {out_path}: {e}")
            skipped += 1

    print(f"\n{LBL_SUMMARY} processed={processed}, skipped={skipped}, total={len(targets)}")

if __name__ == "__main__":
    main()
