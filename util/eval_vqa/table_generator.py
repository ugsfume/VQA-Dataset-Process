#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render new eval metrics into a formatted HTML table using great_tables.

Supports three modes:
  - single : metrics_single.json (Count, BBoxes)
  - dual   : metrics_dual.json   (Count, Color-exact, BBoxes)
  - both   : show single and dual side-by-side

Typical usage (run in the folder with metrics_* or pass explicit paths):

  # Auto-detect files in a directory (summary.json created by the new evaluator)
  python table_generator.py --mode both --in-dir eval_results_new \
    --out eval_table.html --title "Qwen2.5-VL SFT Evaluation" \
    --model-name "Qwen2.5-VL-32B Mask Domain SFT (ckpt-750)"

  # Or pass metrics explicitly
  python /mnt/workspace/kennethliu/src/eval_vqa/table_generator.py --mode both \
    --metrics-single ./metrics_single.json \
    --metrics-dual ./metrics_dual.json \
    --title " Qwen2.5-VL 7B 3-stage Dual-image Domain SFT (Stage 2) " \
    --out eval_table.html

Requires: pip install great-tables pandas
"""

import os
import json
import argparse
import pandas as pd
from typing import Any, Dict, Optional
from great_tables import GT, loc

# -----------------
# Formatting utils
# -----------------

def pct(v: Optional[float], digits=1) -> str:
    if v is None:
        return ""
    try:
        return f"{100*float(v):.{digits}f}%"
    except Exception:
        return ""

def fmt_num(v: Optional[float], digits=3) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return ""

# -----------------
# Metrics helpers
# -----------------

def _extract_bbox_primary(m: Dict[str, Any]) -> Dict[str, Any]:
    """Return micro metrics at IoU@0.5 (primary)."""
    bb = (m or {}).get("bboxes", {})
    micro = (bb.get("micro", {}) or {}).get("@0.5", {}) or {}
    return {
        "samples": bb.get("samples", 0),
        "precision": micro.get("precision"),
        "recall": micro.get("recall"),
        "f1": micro.get("f1"),
        "miou": micro.get("mean_IoU"),
        "exact_set": bb.get("exact_set_match_rate"),
    }

def _rows_from_metrics(m: Dict[str, Any], include_color: bool) -> Dict[str, Dict[str, str]]:
    """
    Build per-category row cells for one metric file.
    Returns a dict keyed by Category -> {"Samples","Accuracy","Details"} with strings formatted.
    """
    rows: Dict[str, Dict[str, str]] = {}

    # Count
    c = (m or {}).get("count", {}) or {}
    rows["Count"] = {
        "Samples": str(c.get("total", 0)),
        "Accuracy": pct(c.get("accuracy")),
        "Details": f"MAE {fmt_num(c.get('mae'))}; RMSE {fmt_num(c.get('rmse'))}; Bias {fmt_num(c.get('bias_mean_pred_minus_gt'))}",
    }

    # Color (only for dual)
    if include_color:
        col = (m or {}).get("color", {}) or {}
        rows["Color"] = {
            "Samples": str(col.get("total", 0)),
            "Accuracy": pct(col.get("exact_accuracy")),
            "Details": "Exact name match.",
        }

    # BBoxes
    bp = _extract_bbox_primary(m)
    rows["BBoxes"] = {
        "Samples": str(bp.get("samples", 0)),
        "Accuracy": pct(bp.get("f1")),  # present F1@0.5 as the headline
        "Details": (
            f"P {pct(bp.get('precision'))} · R {pct(bp.get('recall'))} · F1 {pct(bp.get('f1'))}; "
            f"Exact-set {pct(bp.get('exact_set'))}; "
            f"mIoU {fmt_num(bp.get('miou'))}"
        )
    }

    return rows

# -----------------
# Table builders
# -----------------

def _build_single_table(metrics_single: Dict[str, Any], title: str, subtitle: str) -> GT:
    rows = _rows_from_metrics(metrics_single, include_color=False)
    df = pd.DataFrame(
        [{
            "Category": k,
            "Samples": v["Samples"],
            "Accuracy (%)": v["Accuracy"],
            "Details": v["Details"],
        } for k, v in rows.items()]
    )
    gt = (
        GT(df)
        .tab_header(title=title, subtitle=subtitle)
        .cols_align(columns=["Samples","Accuracy (%)"], align="center")
        .cols_align(columns=["Details"], align="left")
    )
    return _add_footnotes(gt, include_color=False)

def _build_dual_table(metrics_dual: Dict[str, Any], title: str, subtitle: str) -> GT:
    rows = _rows_from_metrics(metrics_dual, include_color=True)
    # Ensure consistent category order
    order = ["Count", "Color", "BBoxes"]
    data = []
    for cat in order:
        if cat in rows:
            r = rows[cat]
            data.append({"Category": cat, "Samples": r["Samples"], "Accuracy (%)": r["Accuracy"], "Details": r["Details"]})
    df = pd.DataFrame(data)
    gt = (
        GT(df)
        .tab_header(title=title, subtitle=subtitle)
        .cols_align(columns=["Samples","Accuracy (%)"], align="center")
        .cols_align(columns=["Details"], align="left")
    )
    return _add_footnotes(gt, include_color=True)

def _build_both_table(metrics_single: Dict[str, Any], metrics_dual: Dict[str, Any], title: str, subtitle: str) -> GT:
    rows_s = _rows_from_metrics(metrics_single, include_color=False)
    rows_d = _rows_from_metrics(metrics_dual, include_color=True)

    # Unified category list & order
    cats = ["Count", "Color", "BBoxes"]
    data = []
    for cat in cats:
        s = rows_s.get(cat, {"Samples":"-","Accuracy":"-","Details":"-"})
        d = rows_d.get(cat, {"Samples":"-","Accuracy":"-","Details":"-"})
        data.append({
            "Category": cat,
            "S Samples": s["Samples"],
            "S Accuracy (%)": s["Accuracy"],
            "S Details": s["Details"],
            "D Samples": d["Samples"],
            "D Accuracy (%)": d["Accuracy"],
            "D Details": d["Details"],
        })
    df = pd.DataFrame(data)

    gt = (
        GT(df)
        .tab_header(title=title, subtitle=subtitle)
        # column spanners use tab_spanner (not cols_spanner)
        .tab_spanner(label="Single image", columns=["S Samples", "S Accuracy (%)", "S Details"])
        .tab_spanner(label="Dual image",   columns=["D Samples", "D Accuracy (%)", "D Details"])
        .cols_align(columns=["S Samples","S Accuracy (%)","D Samples","D Accuracy (%)"], align="center")
        .cols_align(columns=["S Details","D Details"], align="left")
        .tab_stubhead(label="Category")
        .cols_label(
            **{
                "S Samples": "Samples",
                "S Accuracy (%)": "Accuracy (%)",
                "S Details": "Details",
                "D Samples": "Samples",
                "D Accuracy (%)": "Accuracy (%)",
                "D Details": "Details",
            }
        )
    )

    return _add_footnotes(gt, include_color=True)


# -----------------
# Footnotes
# -----------------

def _add_footnotes(gt_obj: GT, include_color: bool) -> GT:
    """
    Add concise footnotes explaining abbreviations/metrics.
    """
    parts = [
        "Single: Augmented images only",
        "Dual: Augmented images + color masks",
        "MAE: Mean Absolute Error (|pred−gt|).",
        "RMSE: Root Mean Squared Error.",
        "Bias: Mean(pred − gt); negative = undercount.",
        "P/R/F1: Precision/Recall/F1 at IoU 0.5.",
        "Exact-set: All GT boxes predicted with no extras (IoU≥0.5).",
        "mIoU: Mean IoU of matched boxes (IoU≥0.5).",
    ]
    note_text = " | ".join(parts)
    return gt_obj.tab_source_note(note_text)

# -----------------
# IO helpers
# -----------------

def _load_metrics_from_dir(in_dir: str, mode: str,
                           metrics_single_path: Optional[str],
                           metrics_dual_path: Optional[str]) -> tuple[Optional[Dict[str,Any]], Optional[Dict[str,Any]], str, str]:
    """
    If summary.json exists, use it to locate metrics_single/dual. Else, fall back to
    provided explicit paths. Returns (ms, md, subtitle_single, subtitle_dual).
    """
    ms = md = None
    subtitle_s = subtitle_d = ""

    summary_path = os.path.join(in_dir, "summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            if mode in ("single", "both") and "metrics_single" in s and os.path.isfile(s["metrics_single"]):
                with open(s["metrics_single"], "r", encoding="utf-8") as f:
                    ms = json.load(f)
                subtitle_s = f"Total QA pairs: {ms.get('totals',{}).get('records','')}"
            if mode in ("dual", "both") and "metrics_dual" in s and os.path.isfile(s["metrics_dual"]):
                with open(s["metrics_dual"], "r", encoding="utf-8") as f:
                    md = json.load(f)
                subtitle_d = f"Total QA pairs: {md.get('totals',{}).get('records','')}"
        except Exception:
            pass

    # Explicit paths override or fill gaps
    if mode in ("single", "both") and metrics_single_path and os.path.isfile(metrics_single_path):
        with open(metrics_single_path, "r", encoding="utf-8") as f:
            ms = json.load(f)
        subtitle_s = f"Total QA pairs: {ms.get('totals',{}).get('records','')}"
    if mode in ("dual", "both") and metrics_dual_path and os.path.isfile(metrics_dual_path):
        with open(metrics_dual_path, "r", encoding="utf-8") as f:
            md = json.load(f)
        subtitle_d = f"Total QA pairs: {md.get('totals',{}).get('records','')}"
    return ms, md, subtitle_s, subtitle_d

# -----------------
# Main
# -----------------

def main():
    ap = argparse.ArgumentParser(description="Create HTML table from new eval metrics.")
    ap.add_argument("--mode", choices=["single", "dual", "both"], default="both")
    ap.add_argument("--in-dir", default=".", help="Directory containing metrics_*.json or summary.json (from evaluator).")
    ap.add_argument("--metrics-single", help="Path to metrics_single.json (overrides in-dir).")
    ap.add_argument("--metrics-dual", help="Path to metrics_dual.json (overrides in-dir).")
    ap.add_argument("--out", default="eval_metrics_table.html", help="Output HTML file.")
    ap.add_argument("--title", default="Evaluation Metrics Summary", help="Table title.")
    ap.add_argument("--model-name", default="", help="Optional model name to display atop the table.")
    args = ap.parse_args()

    ms, md, subtitle_s, subtitle_d = _load_metrics_from_dir(
        in_dir=args.in_dir,
        mode=args.mode,
        metrics_single_path=args.metrics_single,
        metrics_dual_path=args.metrics_dual
    )

    # Compose subtitle line(s)
    subs = []
    if args.model_name:
        subs.append(f"Model: {args.model_name}")
    # show per-mode totals
    if args.mode in ("single", "both") and subtitle_s:
        subs.append(f"[Single] {subtitle_s}")
    if args.mode in ("dual", "both") and subtitle_d:
        subs.append(f"[Dual] {subtitle_d}")
    subtitle = " | ".join(subs)

    # Build table per mode
    if args.mode == "single":
        if not ms:
            raise FileNotFoundError("metrics_single.json not found (use --metrics-single or provide summary.json in --in-dir).")
        gt = _build_single_table(ms, args.title, subtitle)

    elif args.mode == "dual":
        if not md:
            raise FileNotFoundError("metrics_dual.json not found (use --metrics-dual or provide summary.json in --in-dir).")
        gt = _build_dual_table(md, args.title, subtitle)

    else:  # both
        if not ms or not md:
            raise FileNotFoundError("Need both metrics_single.json and metrics_dual.json for mode=both.")
        gt = _build_both_table(ms, md, args.title, subtitle)

    # Render HTML (handle API differences safely)
    try:
        html_str = gt.as_raw_html()
    except AttributeError:
        try:
            html_str = gt._repr_html_()
        except Exception:
            from great_tables import html
            try:
                html_obj = html(gt)
                html_str = html_obj.to_html() if hasattr(html_obj, "to_html") else str(html_obj)
            except Exception:
                html_str = str(gt)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"[OK] Wrote table -> {args.out}")

if __name__ == "__main__":
    main()
