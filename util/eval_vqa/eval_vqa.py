"""
Benchmark test for vertical domain SFT model

Evaluate a trained Qwen2.5-VL model on a test VQA JSONL (concat_qa.jsonl) of the same
format as training qa.jsonl (sharegpt style). Computes metrics for:
  - Defect yes/no (accuracy)
  - Count (accuracy, MAE, RMSE, bias)
  - Color (hex exact, ΔE <=2/3/5, mean ΔE)
  - BBoxes (precision/recall/F1 @ IoU 0.5 primary; also @0.3/@0.7; mean IoU; exact set match rate)

Two modes:
  1) Run inference (default): load model + processor, generate answers.
  2) Evaluate an existing prediction JSONL (--pred-json) with same records + added 'pred' field or parallel JSONL.

Outputs:
  - metrics JSON: eval_metrics.json
  - predictions JSONL: predictions.jsonl (if inference run)
  - optional per-sample error breakdown (CSV) for debugging.

Usage example:
  python ../../../src/eval_vqa/eval_vqa.py \
    --model /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain/vit_llm_freeze/checkpoint-1400 \
    --test-json /mnt/workspace/kennethliu/TFT_circuit_images/test_set/concat_qa.jsonl \
    --images-root /mnt/workspace/kennethliu/TFT_circuit_images/test_set \
    --out-dir eval_results

If images paths in JSONL are already absolute you can omit --images-root.
"""
import os
import re
import json
import math
import argparse
import statistics
from typing import List, Dict, Any, Tuple

import torch

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    HAS_HF = True
except Exception:
    HAS_HF = False
from PIL import Image

YES_SET = {"yes", "y", "是", "對", "对", "true", "1"}
NO_SET  = {"no", "n", "否", "不", "false", "0", "無", "无"}

IOU_THRESHOLDS = [0.3, 0.5, 0.7]

HEX_RE = re.compile(r"#?[0-9a-fA-F]{6}")
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

def resolve_images(img_list: List[str], images_root: str | None) -> List[str]:
    out = []
    for p in img_list:
        if os.path.isabs(p):
            out.append(p)
        else:
            if images_root:
                ap = os.path.abspath(os.path.join(images_root, p))
            else:
                ap = os.path.abspath(p)
            out.append(ap)
    return out

def normalize_yes_no(ans: str) -> str | None:
    if ans is None:
        return None
    s = ans.strip().lower()
    # take first word/punctuation-split token
    tok = re.split(r"[\s,.;:!?，。]", s)[0]
    if tok in YES_SET:
        return "yes"
    if tok in NO_SET:
        return "no"
    return None

def extract_first_int(text: str) -> int | None:
    if not text:
        return None
    m = INT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except:
        return None

def normalize_hex(s: str) -> str | None:
    if s is None:
        return None
    s = s.strip()
    if s.lower() == "null":
        return None
    m = HEX_RE.search(s)
    if not m:  # maybe model printed something like '#ABCDEF,' with punctuation
        return None
    hx = m.group(0).upper()
    if not hx.startswith("#"):
        hx = "#" + hx
    return hx

def hex_to_lab(hex_color: str) -> Tuple[float,float,float]:
    """Rough sRGB->Lab conversion (D65)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2],16)/255.0
    g = int(hex_color[2:4],16)/255.0
    b = int(hex_color[4:6],16)/255.0
    def inv_gamma(c):
        return math.pow((c+0.055)/1.055,2.4) if c>0.04045 else c/12.92
    rl, gl, bl = inv_gamma(r), inv_gamma(g), inv_gamma(b)
    # sRGB -> XYZ (D65)
    X = rl*0.4124564 + gl*0.3575761 + bl*0.1804375
    Y = rl*0.2126729 + gl*0.7151522 + bl*0.0721750
    Z = rl*0.0193339 + gl*0.1191920 + bl*0.9503041
    # Normalize by reference white D65
    X /= 0.95047
    Y /= 1.00000
    Z /= 1.08883
    def f(t):
        return math.pow(t,1/3) if t>0.008856 else (7.787*t + 16/116)
    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b2 = 200*(fy - fz)
    return L,a,b2

def deltaE2000(lab1: Tuple[float,float,float], lab2: Tuple[float,float,float]) -> float:
    # Simplified implementation, adequate for evaluation (not hyper-optimized).
    L1,a1,b1 = lab1
    L2,a2,b2 = lab2
    avg_L = (L1+L2)/2
    C1 = math.sqrt(a1*a1 + b1*b1)
    C2 = math.sqrt(a2*a2 + b2*b2)
    avg_C = (C1+C2)/2
    G = 0.5 * (1 - math.sqrt((avg_C**7)/(avg_C**7 + 25**7)))
    a1p = (1+G)*a1
    a2p = (1+G)*a2
    C1p = math.sqrt(a1p*a1p + b1*b1)
    C2p = math.sqrt(a2p*a2p + b2*b2)
    avg_Cp = (C1p + C2p)/2
    def hp(a,b):
        if a==0 and b==0:
            return 0
        h = math.degrees(math.atan2(b,a))
        return h+360 if h<0 else h
    h1p = hp(a1p,b1)
    h2p = hp(a2p,b2)
    if abs(h1p - h2p) > 180:
        avg_hp = (h1p + h2p + 360)/2
    else:
        avg_hp = (h1p + h2p)/2
    T = 1 - 0.17*math.cos(math.radians(avg_hp - 30)) + \
        0.24*math.cos(math.radians(2*avg_hp)) + \
        0.32*math.cos(math.radians(3*avg_hp + 6)) - \
        0.20*math.cos(math.radians(4*avg_hp - 63))
    dhp = h2p - h1p
    if abs(dhp) > 180:
        dhp = dhp - 360 if dhp > 0 else dhp + 360
    dHp = 2*math.sqrt(C1p*C2p) * math.sin(math.radians(dhp/2))
    dLp = L2 - L1
    dCp = C2p - C1p
    Sl = 1 + (0.015*(avg_L - 50)**2)/math.sqrt(20 + (avg_L - 50)**2)
    Sc = 1 + 0.045*avg_Cp
    Sh = 1 + 0.015*avg_Cp*T
    delta_ro = 30*math.exp(-((avg_hp - 275)/25)**2)
    Rc = 2*math.sqrt((avg_Cp**7)/(avg_Cp**7 + 25**7))
    Rt = -Rc*math.sin(math.radians(2*delta_ro))
    return math.sqrt(
        (dLp/Sl)**2 +
        (dCp/Sc)**2 +
        (dHp/Sh)**2 +
        Rt*(dCp/Sc)*(dHp/Sh)
    )

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

    # If the model wrapped JSON in prose/code fences, try to keep it lenient:
    # First, try a direct json.loads; if it fails, try extracting the first JSON-ish block.
    def try_load(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    obj = try_load(s)
    if obj is None:
        # Try to extract a JSON substring (either {...} or [...])
        m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
        if m:
            obj = try_load(m.group(0))
        if obj is None:
            return []

    # Normalize to a list of candidates
    if isinstance(obj, dict):
        arr = obj.get("bboxes", obj.get("boxes", []))
    elif isinstance(obj, list):
        arr = obj
    else:
        return []

    clean: List[List[int]] = []
    for item in arr:
        # Case 3: {"bbox_2d": [...], "label": "..."}
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

ROLE_SPLIT_PATTERN = re.compile(r'(?:^|\n)assistant\s*\n', re.IGNORECASE)

def extract_final_answer(gen_text: str) -> str:
    """
    From a generated block that may echo:
      system\n...\nuser\n...\nassistant\nANSWER
    return ANSWER only. Also strips code fences and surrounding whitespace.
    """
    if not gen_text:
        return gen_text
    # Take substring after the LAST 'assistant' role marker if present
    lower = gen_text.lower()
    marker = "\nassistant\n"
    idx = lower.rfind(marker)
    if idx != -1:
        answer = gen_text[idx + len(marker):]
    else:
        # Fallback: if pattern not found, try splitting on first double newline after 'assistant'
        m = ROLE_SPLIT_PATTERN.split(gen_text)
        answer = m[-1] if m else gen_text
    # Remove fenced code blocks markers
    answer = re.sub(r"^```[a-zA-Z]*\s*", "", answer.strip())
    answer = re.sub(r"\s*```$", "", answer.strip())
    # If still starts with 'assistant' (edge case), strip first line
    if answer.lower().startswith("assistant"):
        answer = answer.split("\n", 1)[1] if "\n" in answer else ""
    return answer.strip()

def run_inference(
    records: List[Dict[str,Any]],
    model_path: str,
    images_root: str | None,
    device: str,
    generation_kwargs: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    For each record produce 'pred' text (model answer only) using Qwen2.5-VL chat template.
    """
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
        conv = rec["conversations"]
        human_msg = conv[0]["value"]
        question_text = human_msg.replace("<image>", "").strip()
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
                "gt": conv[1]["value"],
                "pred_raw": gen_text,
                "pred": cleaned
            })
            continue

        content = []
        for im in pil_images:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": question_text})
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
            "gt": conv[1]["value"],
            "pred_raw": gen_text,   # keep raw for debugging
            "pred": cleaned         # cleaned used for metrics
        })

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Inference {idx+1}/{len(records)}")

    return preds

def evaluate(pred_records: List[Dict[str,Any]]) -> Dict[str, Any]:
    # Containers
    defect_total = defect_correct = 0
    count_total = count_correct = 0
    count_abs_err = []
    count_sq_err = []
    count_bias_vals = []

    color_total = color_exact = 0
    deltaE_vals = []
    deltaE_acc_2 = deltaE_acc_3 = deltaE_acc_5 = 0
    color_present_pairs = 0

    # BBoxes micro stats per threshold
    bbox_counts = {t: {"TP":0,"FP":0,"FN":0,"IoUs":[]} for t in IOU_THRESHOLDS}
    exact_set_match = 0
    bbox_samples = 0

    for rec in pred_records:
        rid = rec["id"]
        gt = rec["gt"]
        pred = rec["pred"]

        # Identify type
        if rid.endswith("_defect_flag"):
            defect_total += 1
            gt_norm = normalize_yes_no(gt)
            pred_norm = normalize_yes_no(pred)
            if pred_norm == gt_norm:
                defect_correct += 1

        elif rid.endswith("_count"):
            count_total += 1
            gt_int = extract_first_int(gt)
            pred_int = extract_first_int(pred)
            if gt_int is not None and pred_int is not None:
                if gt_int == pred_int:
                    count_correct += 1
                err = abs(pred_int - gt_int)
                count_abs_err.append(err)
                count_sq_err.append(err*err)
                count_bias_vals.append(pred_int - gt_int)
            else:
                # treat as maximal error if parse fails
                count_abs_err.append(float("nan"))
                count_sq_err.append(float("nan"))
                count_bias_vals.append(0)

        elif rid.endswith("_color"):
            color_total += 1
            gt_hex = normalize_hex(gt)
            pred_hex = normalize_hex(pred)
            if gt_hex is None and pred_hex is None:
                color_exact += 1
            elif gt_hex is not None and pred_hex is not None:
                if gt_hex == pred_hex:
                    color_exact += 1
                # deltaE metrics only if both present
                lab_gt = hex_to_lab(gt_hex)
                lab_pr = hex_to_lab(pred_hex)
                dE = deltaE2000(lab_gt, lab_pr)
                deltaE_vals.append(dE)
                color_present_pairs += 1
                if dE <= 2: deltaE_acc_2 += 1
                if dE <= 3: deltaE_acc_3 += 1
                if dE <= 5: deltaE_acc_5 += 1

        elif rid.endswith("_bboxes"):
            bbox_samples += 1
            # parse JSON
            gt_boxes = parse_bbox_json(gt) or []
            pred_boxes = parse_bbox_json(pred) or []
            # primary matching for IoU lists (use greedy)
            matches = greedy_match(gt_boxes, pred_boxes)
            # store IoUs for each threshold
            for thr in IOU_THRESHOLDS:
                TP = sum(1 for _,_,iou in matches if iou >= thr)
                FP = len(pred_boxes) - TP
                FN = len(gt_boxes) - TP
                bbox_counts[thr]["TP"] += TP
                bbox_counts[thr]["FP"] += FP
                bbox_counts[thr]["FN"] += FN
                bbox_counts[thr]["IoUs"].extend([iou for _,_,iou in matches if iou >= thr])
            # exact-set match @0.5 (same cardinality and all matched IoUs >=0.5)
            if len(gt_boxes)==len(pred_boxes):
                ok = True
                if len(gt_boxes)>0:
                    # need each gt matched with IoU>=0.5
                    gt_matched = 0
                    for _,_,iou in matches:
                        if iou >= 0.5:
                            gt_matched += 1
                    ok = (gt_matched == len(gt_boxes))
                if ok:
                    exact_set_match += 1

    # Aggregate metrics
    def safe_mean(vals: List[float]) -> float:
        vals = [v for v in vals if isinstance(v,(int,float)) and not math.isnan(v)]
        return sum(vals)/len(vals) if vals else 0.0

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
        "totals": {
            "records": len(pred_records)
        },
        "defect": {
            "total": defect_total,
            "accuracy": defect_correct / defect_total if defect_total else 0.0
        },
        "count": {
            "total": count_total,
            "accuracy": count_correct / count_total if count_total else 0.0,
            "mae": count_mae,
            "rmse": count_rmse,
            "bias_mean_pred_minus_gt": count_bias
        },
        "color": {
            "total": color_total,
            "exact_accuracy": color_exact / color_total if color_total else 0.0,
            "present_pairs_for_DeltaE": color_present_pairs,
            "mean_deltaE": safe_mean(deltaE_vals),
            "deltaE_accuracy": {
                "<=2": deltaE_acc_2 / color_present_pairs if color_present_pairs else 0.0,
                "<=3": deltaE_acc_3 / color_present_pairs if color_present_pairs else 0.0,
                "<=5": deltaE_acc_5 / color_present_pairs if color_present_pairs else 0.0,
            }
        },
        "bboxes": {
            "samples": bbox_samples,
            "micro": bbox_metrics,
            "exact_set_match_rate": exact_set_match / bbox_samples if bbox_samples else 0.0
        }
    }
    return metrics

def save_jsonl(recs: List[Dict[str,Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

def main():
    ap = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL domain model on test VQA JSONL.")
    ap.add_argument("--model", help="Path to model checkpoint (HF format). If omitted and --pred-json given, only evaluation runs.")
    ap.add_argument("--test-json", required=True, help="Path to test concat_qa.jsonl ground truth.")
    ap.add_argument("--images-root", default=None, help="Optional root to prepend to relative image paths.")
    ap.add_argument("--out-dir", default="eval_results", help="Directory to write outputs.")
    ap.add_argument("--pred-json", default=None, help="If provided, JSONL with predictions (id, gt, pred). Skips inference.")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap on number of QA records for quick test.")
    ap.add_argument("--gen-max-new-tokens", type=int, default=32)
    ap.add_argument("--gen-temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    records = load_jsonl(args.test_json)
    if args.max_samples > 0:
        records = records[:args.max_samples]

    prediction_records = None
    if args.pred_json:
        prediction_records = load_jsonl(args.pred_json)
    else:
        if not args.model:
            raise ValueError("Must provide --model when not supplying --pred-json.")
        gen_kwargs = {
            "max_new_tokens": args.gen_max_new_tokens,
            "temperature": args.gen_temperature,
            "do_sample": args.gen_temperature > 0
        }
        prediction_records = run_inference(
            records=records,
            model_path=args.model,
            images_root=args.images_root,
            device=args.device,
            generation_kwargs=gen_kwargs
        )
        save_jsonl(prediction_records, os.path.join(args.out_dir, "predictions.jsonl"))
        print(f"[INFO] Saved predictions to {os.path.join(args.out_dir,'predictions.jsonl')}")

    metrics = evaluate(prediction_records)
    metrics_path = os.path.join(args.out_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metrics written: {metrics_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
