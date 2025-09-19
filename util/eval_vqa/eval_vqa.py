#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a trained Qwen2.5-VL model on new-format VQA test sets.

Two test modes:
  - SINGLE (1 image per sample): uses concat_qa.jsonl
      * Metrics: component COUNT and BBOX
  - DUAL (2 images per sample: RGB + color mask): uses concat_qa_mask.jsonl
      * Metrics: component COUNT, BBOX, COLOR (Chinese name exact match)

You can run SINGLE, DUAL, or BOTH. Each mode can either:
  - run inference from a model, or
  - load an existing predictions JSONL (id, images, question, gt, pred).

Outputs (per-mode if run):
  - predictions JSONL: predictions_single.jsonl / predictions_dual.jsonl
  - metrics JSON:     metrics_single.json  / metrics_dual.json
  - a top-level summary.json with quick pointers.

Usage examples:

# Run both test sets with inference
python /mnt/workspace/kennethliu/src/eval_vqa/eval_vqa.py \
  --model /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_domain_judge/vit_freeze/checkpoint-270 \
  --single-json /mnt/workspace/kennethliu/TFT_circuit_images/test_set/concat_qa.jsonl \
  --dual-json /mnt/workspace/kennethliu/TFT_circuit_images/test_set/concat_qa_mask.jsonl \
  --out-dir eval_results

# Only evaluate dual with existing predictions (skip inference)
python eval_vqa.py \
  --mode dual \
  --dual-json /path/to/concat_qa_mask.jsonl \
  --pred-dual-json /path/to/predictions_dual.jsonl \
  --out-dir eval_results
"""

import os
import re
import json
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    HAS_HF = True
except Exception:
    HAS_HF = False

from PIL import Image

# -----------------------
# Parsing / normalization
# -----------------------

INT_RE = re.compile(r"[-+]?\d+")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("//"):
                continue
            data.append(json.loads(line))
    return data

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def resolve_images(img_list: List[str], images_root: Optional[str]) -> List[str]:
    out = []
    for p in img_list:
        if os.path.isabs(p):
            out.append(p)
        else:
            ap = os.path.abspath(os.path.join(images_root, p)) if images_root else os.path.abspath(p)
            out.append(ap)
    return out

def extract_first_int(text: str) -> Optional[int]:
    if not text:
        return None
    m = INT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None

def normalize_color_name_ch(s: str) -> Optional[str]:
    """
    Normalize the color (Chinese name) for exact-match check.
    Rules:
      - strip whitespace
      - drop trailing punctuation (comma/period/quotes)
      - treat "null" (case-insensitive) as None
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    low = s.lower()
    if low == "null":
        return None
    # Remove simple surrounding quotes/backticks/fences if any
    s = s.strip().strip("`").strip()
    # Remove common trailing punctuation
    s = re.sub(r"[，。,．．。!\?；;：:\]\)\}]+$", "", s)
    return s

def parse_bbox_json(val: str) -> List[List[int]] | None:
    """
    Accepts any of:
      1) {"bboxes": [[x1,y1,x2,y2], ...]}
      2) [[x1,y1,x2,y2], ...]
      3) [{"bbox_2d": [x1,y1,x2,y2], "label": "..."} , ...]
    Returns a cleaned list of [x1,y1,x2,y2] (ints).
    """
    if val is None:
        return None
    s = val.strip()
    if s.lower() == "null":
        return None

    def try_load(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    obj = try_load(s)
    if obj is None:
        m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
        if m:
            obj = try_load(m.group(0))
        if obj is None:
            return []

    if isinstance(obj, dict):
        arr = obj.get("bboxes", obj.get("boxes", []))
    elif isinstance(obj, list):
        arr = obj
    else:
        return []

    clean: List[List[int]] = []
    for item in arr:
        if isinstance(item, dict) and "bbox_2d" in item:
            coords = item["bbox_2d"]
        else:
            coords = item

        if (isinstance(coords, (list, tuple)) and len(coords) == 4):
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in coords]
                clean.append([x1, y1, x2, y2])
            except Exception:
                continue
    return clean

# -------------
# BBox metrics
# -------------

def bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1 + 1)*(iy2 - iy1 + 1)
    a_area = (ax2 - ax1 + 1)*(ay2 - ay1 + 1)
    b_area = (bx2 - bx1 + 1)*(by2 - by1 + 1)
    return inter / (a_area + b_area - inter + 1e-9)

def greedy_match(gt: List[List[int]], pred: List[List[int]]) -> List[Tuple[int,int,float]]:
    matches = []
    used_gt = set()
    used_pr = set()
    pairs = []
    for i,g in enumerate(gt):
        for j,p in enumerate(pred):
            iou = bbox_iou(g,p)
            pairs.append((iou,i,j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    for iou, gi, pj in pairs:
        if gi in used_gt or pj in used_pr:
            continue
        used_gt.add(gi)
        used_pr.add(pj)
        matches.append((gi,pj,iou))
    return matches

# ----------------------
# Inference / utilities
# ----------------------

ROLE_SPLIT_PATTERN = re.compile(r'(?:^|\n)assistant\s*\n', re.IGNORECASE)

def extract_final_answer(gen_text: str) -> str:
    """
    From a generated block that may echo roles, return assistant's final answer only.
    Also strips code fences and surrounding whitespace.
    """
    if not gen_text:
        return gen_text
    lower = gen_text.lower()
    marker = "\nassistant\n"
    idx = lower.rfind(marker)
    if idx != -1:
        answer = gen_text[idx + len(marker):]
    else:
        m = ROLE_SPLIT_PATTERN.split(gen_text)
        answer = m[-1] if m else gen_text
    answer = re.sub(r"^```[a-zA-Z]*\s*", "", answer.strip())
    answer = re.sub(r"\s*```$", "", answer.strip())
    if answer.lower().startswith("assistant"):
        answer = answer.split("\n", 1)[1] if "\n" in answer else ""
    return answer.strip()

def build_user_message(rec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Convert a test record in your JSONL format into a chat 'messages' structure for Qwen2.5-VL.
    The record already contains a 'human' content with <image> placeholders.
    We'll map:
      images -> [{"type":"image", "image": PIL.Image}, ...]
      question -> {"type": "text", "text": human_text_without_placeholders}
    """
    conv = rec["conversations"]
    human_msg = conv[0]["value"]
    question_text = human_msg.replace("<image>", "").strip()
    return [{"role": "user", "content": [{"type": "text", "text": question_text}]}], question_text

def run_inference(
    records: List[Dict[str,Any]],
    model_path: str,
    images_root: Optional[str],
    device: str,
    generation_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not HAS_HF:
        raise RuntimeError("transformers not available for inference.")
    print(f"[INFO] Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    preds = []
    for idx, rec in enumerate(records):
        human_msg = rec["conversations"][0]["value"]
        image_paths = resolve_images(rec["images"], images_root)
        pil_images = []
        try:
            for ip in image_paths:
                pil_images.append(Image.open(ip).convert("RGB"))
        except Exception as e:
            gen_text = f"[ERROR] cannot open image(s): {e}"
            cleaned = gen_text
            preds.append({
                "id": rec["id"],
                "images": rec["images"],
                "question": human_msg,
                "gt": rec["conversations"][1]["value"],
                "pred_raw": gen_text,
                "pred": cleaned
            })
            continue

        # Qwen chat template: we pass images separately via processor()
        content = []
        for im in pil_images:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": human_msg.replace("<image>", "").strip()})
        messages = [{"role": "user", "content": content}]

        try:
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = processor(
                text=[text_prompt],
                images=pil_images,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)
            gen_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            cleaned = extract_final_answer(gen_text)
        except Exception as e:
            gen_text = f"[ERROR] {e}"
            cleaned = gen_text

        preds.append({
            "id": rec["id"],
            "images": rec["images"],
            "question": human_msg,
            "gt": rec["conversations"][1]["value"],
            "pred_raw": gen_text,
            "pred": cleaned
        })

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Inference {idx+1}/{len(records)}")

    return preds

# ---------------
# Eval per-mode
# ---------------

IOU_THRESHOLDS = [0.3, 0.5, 0.7]

def safe_mean(vals: List[float]) -> float:
    vals = [v for v in vals if isinstance(v,(int,float)) and not math.isnan(v)]
    return sum(vals)/len(vals) if vals else 0.0

def evaluate_records(pred_records: List[Dict[str, Any]], include_color: bool) -> Dict[str, Any]:
    """
    include_color: False for SINGLE, True for DUAL
    """
    count_total = count_correct = 0
    count_abs_err, count_sq_err, count_bias_vals = [], [], []

    color_total = color_exact = 0  # exact match (Chinese names)
    # bbox micro stats per threshold
    bbox_counts = {t: {"TP":0,"FP":0,"FN":0,"IoUs":[]} for t in IOU_THRESHOLDS}
    exact_set_match = 0
    bbox_samples = 0

    for rec in pred_records:
        rid = rec["id"]
        gt = rec["gt"]
        pred = rec["pred"]

        if rid.endswith("_count"):
            count_total += 1
            gt_int = extract_first_int(gt)
            pred_int = extract_first_int(pred)
            if gt_int is not None and pred_int is not None:
                if gt_int == pred_int:
                    count_correct += 1
                err = pred_int - gt_int
                count_abs_err.append(abs(err))
                count_sq_err.append(err*err)
                count_bias_vals.append(err)
            else:
                count_abs_err.append(float("nan"))
                count_sq_err.append(float("nan"))
                count_bias_vals.append(0)

        elif rid.endswith("_bboxes"):
            bbox_samples += 1
            gt_boxes = parse_bbox_json(gt) or []
            pred_boxes = parse_bbox_json(pred) or []
            matches = greedy_match(gt_boxes, pred_boxes)
            for thr in IOU_THRESHOLDS:
                TP = sum(1 for _,_,iou in matches if iou >= thr)
                FP = len(pred_boxes) - TP
                FN = len(gt_boxes) - TP
                bbox_counts[thr]["TP"] += TP
                bbox_counts[thr]["FP"] += FP
                bbox_counts[thr]["FN"] += FN
                bbox_counts[thr]["IoUs"].extend([iou for _,_,iou in matches if iou >= thr])
            if len(gt_boxes)==len(pred_boxes):
                ok = True
                if len(gt_boxes) > 0:
                    ok = sum(1 for _,_,iou in matches if iou >= 0.5) == len(gt_boxes)
                if ok:
                    exact_set_match += 1

        elif include_color and rid.endswith("_color"):
            color_total += 1
            gt_c = normalize_color_name_ch(gt)
            pr_c = normalize_color_name_ch(pred)
            if gt_c == pr_c:
                color_exact += 1

        else:
            # ignore other record types, if any
            pass

    count_mae = safe_mean(count_abs_err)
    count_rmse = math.sqrt(safe_mean(count_sq_err))
    count_bias = safe_mean(count_bias_vals)

    bbox_metrics = {}
    for thr, st in bbox_counts.items():
        TP, FP, FN = st["TP"], st["FP"], st["FN"]
        prec = TP / (TP + FP + 1e-9)
        rec = TP / (TP + FN + 1e-9)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        mean_iou = safe_mean(st["IoUs"])
        bbox_metrics[f"@{thr}"] = {
            "TP": TP, "FP": FP, "FN": FN,
            "precision": prec, "recall": rec,
            "f1": f1, "mean_IoU": mean_iou
        }

    metrics = {
        "totals": {"records": len(pred_records)},
        "count": {
            "total": count_total,
            "accuracy": count_correct / count_total if count_total else 0.0,
            "mae": count_mae,
            "rmse": count_rmse,
            "bias_mean_pred_minus_gt": count_bias
        },
        "bboxes": {
            "samples": bbox_samples,
            "micro": bbox_metrics,
            "exact_set_match_rate": exact_set_match / bbox_samples if bbox_samples else 0.0
        }
    }
    if include_color:
        metrics["color"] = {
            "total": color_total,
            "exact_accuracy": (color_exact / color_total) if color_total else 0.0
        }

    return metrics

# ---------------
# Orchestration
# ---------------

def save_jsonl(recs: List[Dict[str,Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

def main():
    ap = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL model on new-format VQA test sets.")
    ap.add_argument("--model", help="HF path to model checkpoint. Required unless providing pred JSON(s).")
    ap.add_argument("--images-root", default=None, help="Optional root to prepend to relative image paths.")
    ap.add_argument("--out-dir", default="eval_results", help="Directory to write outputs.")
    ap.add_argument("--mode", choices=["single", "dual", "both"], default="both")

    # test sets
    ap.add_argument("--single-json", help="Path to concat_qa.jsonl (single image test set).")
    ap.add_argument("--dual-json", help="Path to concat_qa_mask.jsonl (dual image test set).")

    # use predictions instead of running inference
    ap.add_argument("--pred-single-json", help="Predictions JSONL for single test (id, gt, pred...).")
    ap.add_argument("--pred-dual-json", help="Predictions JSONL for dual test (id, gt, pred...).")

    # generation controls
    ap.add_argument("--gen-max-new-tokens", type=int, default=512)
    ap.add_argument("--gen-temperature", type=float, default=0.0)
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap per test split for a quick run.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    summary: Dict[str, Any] = {}
    gen_kwargs = {
        "max_new_tokens": args.gen_max_new_tokens,
        "temperature": args.gen_temperature,
        "do_sample": args.gen_temperature > 0,
    }

    def _maybe_cap(recs: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        return recs[:args.max_samples] if args.max_samples > 0 else recs

    # --------
    # SINGLE
    # --------
    if args.mode in ("single", "both"):
        if not args.single_json:
            raise ValueError("Please provide --single-json for single mode.")
        single_records = _maybe_cap(load_jsonl(args.single_json))
        if args.pred_single_json:
            single_preds = _maybe_cap(load_jsonl(args.pred_single_json))
        else:
            if not args.model:
                raise ValueError("Must provide --model when not supplying --pred-single-json.")
            single_preds = run_inference(
                records=single_records,
                model_path=args.model,
                images_root=args.images_root,
                device=args.device,
                generation_kwargs=gen_kwargs
            )
            spath = os.path.join(args.out_dir, "predictions_single.jsonl")
            save_jsonl(single_preds, spath)
            summary["predictions_single"] = spath
            print(f"[INFO] Saved single predictions -> {spath}")

        single_metrics = evaluate_records(single_preds, include_color=False)
        smpath = os.path.join(args.out_dir, "metrics_single.json")
        with open(smpath, "w", encoding="utf-8") as f:
            json.dump(single_metrics, f, ensure_ascii=False, indent=2)
        summary["metrics_single"] = smpath
        print(f"[OK] Single metrics written: {smpath}")

    # ------
    # DUAL
    # ------
    if args.mode in ("dual", "both"):
        if not args.dual_json:
            raise ValueError("Please provide --dual-json for dual mode.")
        dual_records = _maybe_cap(load_jsonl(args.dual_json))
        if args.pred_dual_json:
            dual_preds = _maybe_cap(load_jsonl(args.pred_dual_json))
        else:
            if not args.model:
                raise ValueError("Must provide --model when not supplying --pred-dual-json.")
            dual_preds = run_inference(
                records=dual_records,
                model_path=args.model,
                images_root=args.images_root,
                device=args.device,
                generation_kwargs=gen_kwargs
            )
            dpath = os.path.join(args.out_dir, "predictions_dual.jsonl")
            save_jsonl(dual_preds, dpath)
            summary["predictions_dual"] = dpath
            print(f"[INFO] Saved dual predictions -> {dpath}")

        dual_metrics = evaluate_records(dual_preds, include_color=True)
        dmpath = os.path.join(args.out_dir, "metrics_dual.json")
        with open(dmpath, "w", encoding="utf-8") as f:
            json.dump(dual_metrics, f, ensure_ascii=False, indent=2)
        summary["metrics_dual"] = dmpath
        print(f"[OK] Dual metrics written: {dmpath}")

    # summary file
    sumpath = os.path.join(args.out_dir, "summary.json")
    with open(sumpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Summary written: {sumpath}")


if __name__ == "__main__":
    main()
