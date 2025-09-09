"""
Render eval_metrics.json into a formatted table (HTML) using great_tables.

Usage (run where eval_metrics.json lives):
  python ../../../../src/eval_vqa/table_generator.py --metrics eval_metrics.json --out eval_metrics_table.html \
    --model-name "Qwen2.5-VL 7B Mask Domain SFT Projector (ckpt-1400)"

Main columns:
  - Category
  - Samples
  - Accuracy (%)
  - Additional metrics (compact)
Percent values are converted from fractions. Non‑percentage metrics (MAE, RMSE, bias, mean ΔE) kept numeric.

Requires: pip install great-tables pandas
"""
import json
import argparse
import pandas as pd
from great_tables import GT, html, loc

def pct(v, digits=1):
    if v is None:
        return ""
    return f"{100*v:.{digits}f}%"

def fmt_num(v, digits=3):
    if v is None:
        return ""
    return f"{v:.{digits}f}"

def build_rows(m):
    rows = []

    # Defect
    defect = m.get("defect", {})
    rows.append({
        "Category": "Defect",
        "Samples": defect.get("total", 0),
        "Accuracy (%)": pct(defect.get("accuracy")),
        "Details": ""
    })

    # Count
    count = m.get("count", {})
    rows.append({
        "Category": "Count",
        "Samples": count.get("total", 0),
        "Accuracy (%)": pct(count.get("accuracy")),
        "Details": f"MAE {fmt_num(count.get('mae'))}; RMSE {fmt_num(count.get('rmse'))}; Bias {fmt_num(count.get('bias_mean_pred_minus_gt'))}"
    })

    # Color
    color = m.get("color", {})
    dE = color.get("deltaE_accuracy", {})
    rows.append({
        "Category": "Color",
        "Samples": color.get("total", 0),
        "Accuracy (%)": pct(color.get("exact_accuracy")),
        "Details": (
            f"ΔE mean {fmt_num(color.get('mean_deltaE'))}; "
            f"ΔE≤2 {pct(dE.get('<=2'))}; ΔE≤3 {pct(dE.get('<=3'))}; ΔE≤5 {pct(dE.get('<=5'))}"
        )
    })

    # BBoxes (use IoU@0.5 primary)
    bboxes = m.get("bboxes", {})
    micro = bboxes.get("micro", {}).get("@0.5", {})
    rows.append({
        "Category": "BBoxes",
        "Samples": bboxes.get("samples", 0),
        "Accuracy (%)": pct(micro.get("f1")),  # Present F1@0.5 as main (coverage+quality)
        "Details": (
            f"P {pct(micro.get('precision'))} R {pct(micro.get('recall'))} F1 {pct(micro.get('f1'))}; "
            f"Exact-set {pct(bboxes.get('exact_set_match_rate'))}; "
            f"mIoU {fmt_num(micro.get('mean_IoU'))}"
        )
    })

    return rows

def add_footnotes(gt_obj: GT):
    # Add concise footnotes
    footnotes = {
        "MAE": "Mean absolute error (|pred-gt|).",
        "RMSE": "Root mean squared error (penalizes larger errors).",
        "Bias": "Mean(pred - gt); negative = undercount.",
        "ΔE": "Perceptual color distance (CIEDE2000). Lower is closer.",
        "ΔE≤k": "Percent with ΔE <= threshold.",
        "P/R/F1": "Precision/Recall/F1 at IoU 0.5.",
        "Exact-set": "All GT boxes predicted (no extras).",
        "mIoU": "Mean IoU of matched boxes (IoU≥0.5)."
    }
    note_text = " | ".join(f"{k}: {v}" for k,v in footnotes.items())
    gt_obj = gt_obj.tab_source_note(note_text)
    return gt_obj

def main():
    ap = argparse.ArgumentParser(description="Create HTML table from eval_metrics.json.")
    ap.add_argument("--metrics", default="eval_metrics.json", help="Path to eval_metrics.json.")
    ap.add_argument("--out", default="eval_metrics_table.html", help="Output HTML file.")
    ap.add_argument("--title", default="Evaluation Metrics Summary", help="Table title.")
    ap.add_argument("--model-name", default="", help="Optional model name to display atop the table.")
    args = ap.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    rows = build_rows(metrics)
    df = pd.DataFrame(rows)

    gt = (
        GT(df)
        .tab_header(
            title=args.title,
            subtitle=" | ".join(
                p for p in [
                    f"Total QA pairs: {metrics.get('totals', {}).get('records', '')}",
                    f"Model: {args.model_name}" if args.model_name else ""
                ] if p
            )
        )
        .cols_label(
            Category="Category",
            Samples="Samples",
            **{"Accuracy (%)":"Accuracy (%)"},
            Details="Details"
        )
        .cols_align(columns=["Samples","Accuracy (%)"], align="center")
        .cols_align(columns=["Details"], align="left")
    )

    gt = add_footnotes(gt)

    # Generate raw HTML safely (great_tables API differences handled)
    try:
        # Preferred method
        html_str = gt.as_raw_html()
    except AttributeError:
        try:
            # Fallback: representation used in notebooks
            html_str = gt._repr_html_()
        except Exception:
            try:
                # If html() helper returns an object with to_html()
                html_obj = html(gt)
                html_str = html_obj.to_html() if hasattr(html_obj, "to_html") else str(html_obj)
            except Exception:
                html_str = str(gt)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"[OK] Wrote table -> {args.out}")

if __name__ == "__main__":
    main()