#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gt_logic_corrector.py

Finds sample folders containing `.bad_gt_analysis`, reads the "analysis" field from
analysis_gt.json (or gt_analysis.json fallback), sends it to the API with model
"qwen3-235b-a22b-instruct-2507" using your correction prompt, and saves the model
response to `.gt_correction` beside analysis_gt.json.

Formatting preservation:
- Default: write **JSON-escaped** text (so `\n` stays literally `\\n`, quotes/backslashes escaped),
  matching the style used inside JSON strings.
- `--raw`: write the model output **verbatim** (actual newlines, no escaping).
- No `.strip()` anywhere; leading/trailing whitespace is preserved.

Usage:
  cd gt_datasets_20250915
  python gt_logic_corrector.py
  # or verbatim output:
  python gt_logic_corrector.py --raw
  # optional:
  TURING_API_KEY="sk-override..." python gt_logic_corrector.py --overwrite --sleep 0.2
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

import requests
from colorama import init as colorama_init, Fore, Style

# ---- CLI colors ----
colorama_init(autoreset=True)
OK   = Fore.GREEN  + "[OK]"   + Style.RESET_ALL
ERR  = Fore.RED    + "[ERR]"  + Style.RESET_ALL
INFO = Fore.CYAN   + "[INFO]" + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL

API_BASE_DEFAULT = "https://test-turing.cn.llm.tcljd.com/api/v1"
MODEL_DEFAULT    = "qwen3-235b-a22b-instruct-2507"

# Embedded API key (overridden by TURING_API_KEY if set)
API_KEY_DEFAULT  = "sk-ZISnDqR3Yi5FHVScO5hZCWLO0xx4luKRgT29r9X4sbh"

PROMPT_TEMPLATE = (
    "以下文本的推理逻辑有前文后理不连贯的情况出现，例如应用规则中认为样本符合所有规则，"
    "但结论和理由中却显示因样本违反其中一条规则而导致不符合整体规则，是明显的矛盾。\n\n"
    "文本中的结论和理由内容才是正确的推理答案。请你先检查文本中是否有足够的资讯让你去修正文本的推理逻辑，"
    "若是足够的话，请你以结论和理由内容为正确逻辑去修改文本(请不要对文本的整体格式做任何改动，例如\\n)，"
    "好让前文后理的推理连贯并没矛盾。以下为文本内容: \"{text}\"\n"
    "请你在你的回应中严格跟随这格式:\"solvable: yes/no; (如果是solvable的话，把修正后的完整文本放在这)\""
)

def make_headers(api_key: str) -> dict:
    key = api_key.strip()
    if not key.lower().startswith("bearer "):
        key = f"Bearer {key}"
    return {"Authorization": key, "Content-Type": "application/json"}

def call_qwen(api_base: str, api_key: str, model: str, prompt: str,
              timeout: int = 90) -> Tuple[bool, str]:
    """Call the OpenAI-compatible /chat/completions endpoint. Returns (ok, content_or_error)."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.3  # stable corrections
    }
    try:
        r = requests.post(url, headers=make_headers(api_key), json=payload, timeout=timeout)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:500]}"
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content is None:
            return False, f"Empty content in response: {str(data)[:500]}"
        # IMPORTANT: do NOT strip — preserve leading/trailing whitespace/newlines
        return True, content
    except Exception as e:
        return False, f"Request error: {e}"

def read_analysis_text(sample_dir: Path) -> Optional[str]:
    """Read 'analysis' field from analysis_gt.json or gt_analysis.json."""
    for name in ("analysis_gt.json", "gt_analysis.json"):
        p = sample_dir / name
        if p.is_file():
            try:
                with p.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                val = obj.get("analysis")
                if isinstance(val, str) and val != "":
                    return val
                else:
                    print(f"{WARN} 'analysis' missing/empty in {p}")
                    return None
            except Exception as e:
                print(f"{ERR} Failed to read {p}: {e}")
                return None
    print(f"{WARN} No analysis JSON found in {sample_dir} (looked for analysis_gt.json / gt_analysis.json)")
    return None

def to_json_escaped(text: str) -> str:
    """
    Convert text to a JSON-escaped string body WITHOUT surrounding quotes.
    This preserves control characters like newline as literal '\\n'.
    """
    dumped = json.dumps(text, ensure_ascii=False)
    if dumped.startswith('"') and dumped.endswith('"'):
        return dumped[1:-1]
    return dumped  # fallback; shouldn't happen

def main():
    ap = argparse.ArgumentParser(description="Run Qwen correction on samples flagged by .bad_gt_analysis.")
    ap.add_argument("--root", default=".", help="Root directory to scan (default: current).")
    ap.add_argument("--api-base", default=API_BASE_DEFAULT, help="API base URL.")
    ap.add_argument("--model", default=MODEL_DEFAULT, help="Model name.")
    ap.add_argument("--timeout", type=int, default=90, help="HTTP timeout per request (sec).")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between calls.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .gt_correction files.")
    ap.add_argument("--dry-run", action="store_true", help="List targets but do not call the API.")
    ap.add_argument("--raw", action="store_true",
                    help="Write model output verbatim (no escaping). Default writes JSON-escaped text.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"{INFO} Starting in: {root}")

    api_key = os.environ.get("TURING_API_KEY", API_KEY_DEFAULT).strip()
    if not api_key and not args.dry_run:
        print(f"{ERR} Missing API key (TURING_API_KEY not set and default empty).")
        return

    total = 0
    corrected = 0
    skipped = 0
    failed = 0

    for dirpath, _, filenames in os.walk(root):
        sample_dir = Path(dirpath)
        if ".bad_gt_analysis" not in filenames:
            continue  # only handle gt-analysis-bad samples

        out_path = sample_dir / ".gt_correction"
        if out_path.exists() and not args.overwrite:
            print(f"{INFO} Skip (exists): {out_path}")
            skipped += 1
            continue

        analysis_text = read_analysis_text(sample_dir)
        if analysis_text is None:
            skipped += 1
            continue

        total += 1
        prompt = PROMPT_TEMPLATE.format(text=analysis_text)

        if args.dry_run:
            print(f"{INFO} (dry-run) Would correct: {sample_dir}")
            continue

        ok, content = call_qwen(args.api_base, api_key, args.model, prompt, timeout=args.timeout)
        if not ok:
            print(f"{ERR} API call failed for {sample_dir}: {content}")
            failed += 1
            continue

        # Preserve formatting:
        # - default: JSON-escaped (so '\n' is literal backslash-n)
        # - --raw: verbatim (actual newlines)
        try:
            with out_path.open("w", encoding="utf-8", newline="\n") as f:
                if args.raw:
                    f.write(content)            # no .strip(), no newline added/removed
                else:
                    f.write(to_json_escaped(content))
            corrected += 1
            print(f"{OK} Wrote: {out_path}")
        except Exception as e:
            print(f"{ERR} Failed to write {out_path}: {e}")
            failed += 1

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\n{INFO} Done.")
    print(f"{INFO} Samples targeted: {total}")
    print(f"{INFO} Corrections written: {corrected}")
    print(f"{INFO} Skipped: {skipped}")
    print(f"{INFO} Failures: {failed}")

if __name__ == "__main__":
    main()
