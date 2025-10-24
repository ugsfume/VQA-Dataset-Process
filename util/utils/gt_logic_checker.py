#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gt_logic_checker.py

Recursively checks logic consistency in the "analysis" field of:
  - analysis_gt.json (or gt_analysis.json)
  - remix_analysis_gt.json (or remix_gt_analysis.json)

Per sample:
  - Sends the analysis text to GPT-5 (test-turing.cn.llm.tcljd.com API).
  - Expects response format: "bad_logic: yes/no; Reason: ..."
  - If bad_logic == yes:
        writes a marker file in the sample folder:
            .bad_gt_analysis           (for analysis_gt.json / gt_analysis.json)
            .bad_remix_gt_analysis     (for remix_analysis_gt.json / remix_gt_analysis.json)
        marker file contains the full model response.

After the walk:
  - Writes a summary list to ./bad_logic_samples.txt in the root (cwd).

Run from the dataset root, e.g.:
    cd /path/to/gt_datasets_20250915
    python gt_logic_checker.py
"""

import os
import re
import json
import time
import argparse
import requests
from typing import Dict, Tuple, Optional, List
from colorama import init as colorama_init, Fore, Back, Style

# ---------- Pretty CLI tags ----------
colorama_init(autoreset=True)
OK   = Fore.GREEN + "[OK]" + Style.RESET_ALL
BAD  = Fore.YELLOW+ "[BAD]" + Style.RESET_ALL
ERR  = Fore.RED   + "[ERR]" + Style.RESET_ALL
INFO = Fore.CYAN  + "[INFO]" + Style.RESET_ALL
WARN = Back.YELLOW+ "[WARN]" + Style.RESET_ALL

API_BASE = "https://test-turing.cn.llm.tcljd.com/api/v1"
MODEL_NAME = "turing/gpt-5"

# Default to your provided key; env var can override.
DEFAULT_KEY = "sk-ZISnDqR3Yi5FHVScO5hZCWLO0xx4luKRgT29r9X4sbh"

PROMPT_TEMPLATE = (
    "I want you to examine the following text and determine if the logic is consistent throughout the text. "
    "For instance, an example showing inconsistent logic would mention that the subject conforms to rules 1-3 in **应用规则**, "
    "but then say that the subject violates the rules in the conclusion and reasons section, therefore contradicting itself. "
    "The main focus is that **应用规则**, **结论**, **理由** must not contradict one another, where if a rule fails in **应用规则**, the overall verdict should show invalid. "
    "Minor flaws, such as slight inconsistency in descriptions can be tolerated."
    "In your response, in English, I want you to strictly follow this format: \"bad_logic: yes/no; Reason: ...\" "
    "The following is the text in question: {analysis_text}"
)

# Consider common filename variants just in case
GT_FILES = [
    ("analysis_gt.json", ".bad_gt_analysis"),
    ("gt_analysis.json", ".bad_gt_analysis"),
]
REMIX_FILES = [
    ("remix_analysis_gt.json", ".bad_remix_gt_analysis"),
    ("remix_gt_analysis.json", ".bad_remix_gt_analysis"),
]


def make_auth_header(key: str) -> str:
    """Return a proper Authorization header value, tolerating keys that already include 'Bearer '."""
    key = key.strip()
    if key.lower().startswith("bearer "):
        return key
    return f"Bearer {key}"


def read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{ERR} Failed to read JSON: {path} ({e})")
        return None


def extract_analysis_field(data: Dict) -> Optional[str]:
    """Return the 'analysis' field if present and non-empty; else None."""
    if not isinstance(data, dict):
        return None
    val = data.get("analysis")
    if isinstance(val, str) and val.strip():
        return val
    return None


def call_gpt5(api_key: str, analysis_text: str, timeout: int = 60) -> Tuple[bool, Optional[str]]:
    """
    Call GPT-5 chat endpoint with the given analysis text.
    Returns (success, content_text). On failure, (False, error_msg).
    """
    url_chat = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": make_auth_header(api_key),
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(analysis_text=analysis_text)}
        ],
        "stream": False,
        # Optionally make the check a bit steadier:
        "temperature": 0.2
    }

    try:
        resp = requests.post(url_chat, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        return False, f"Request error: {e}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text}"

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return True, content
    except Exception as e:
        return False, f"Parse error: {e}; raw: {resp.text[:500]}"


BAD_LOGIC_RE = re.compile(r"bad_logic\s*:\s*(yes|no)", flags=re.IGNORECASE)
REASON_RE = re.compile(r"reason\s*:\s*(.*)", flags=re.IGNORECASE | re.DOTALL)


def parse_verdict(text: str) -> Tuple[Optional[bool], str]:
    """
    Parse GPT5 response expecting "bad_logic: yes/no; Reason: ...".
    Returns (bad_logic_bool or None if unparsable, reason_text).
    """
    # Remove code fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        # drop the first line (``` or ```xxx) and the last ``` if present
        lines = stripped.splitlines()
        # remove first code fence
        if lines:
            lines = lines[1:]
        # remove last code fence if any
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    m_logic = BAD_LOGIC_RE.search(stripped)
    if not m_logic:
        return None, stripped

    logic_val = m_logic.group(1).lower()
    bad_logic = True if logic_val == "yes" else False

    m_reason = REASON_RE.search(stripped)
    reason = m_reason.group(1).strip() if m_reason else ""
    return bad_logic, reason


def write_flag_file(dirpath: str, flag_filename: str, gpt_response: str) -> None:
    out_path = os.path.join(dirpath, flag_filename)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(gpt_response.strip() + "\n")
        print(f"{OK} Flag written: {out_path}")
    except Exception as e:
        print(f"{ERR} Failed to write flag file: {out_path} ({e})")


def save_summary(root_dir: str, flagged: List[Tuple[str, str, str]]) -> str:
    """
    Save a summary list in the root_dir.
    flagged: list of (rel_dir, which_file, reason_or_raw)
    Returns the path to the summary file.
    """
    out_path = os.path.join(root_dir, "bad_logic_samples.txt")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Bad logic samples\n")
            f.write("# Format: <relative_sample_dir> | <file> | <reason_or_raw>\n\n")
            for rel_dir, which_file, reason in flagged:
                reason_one_line = " ".join(reason.split())  # compress whitespace
                f.write(f"{rel_dir} | {which_file} | {reason_one_line}\n")
        print(f"{OK} Summary saved: {out_path}")
    except Exception as e:
        print(f"{ERR} Failed to write summary: {out_path} ({e})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Check logic consistency in analysis_gt.json / remix_analysis_gt.json using GPT-5.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep seconds between API calls.")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout per request (seconds).")
    parser.add_argument("--root", type=str, default=".", help="Root directory to scan (default: current).")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    print(f"{INFO} Scanning from root: {root_dir}")

    # Auth key: prefer env, fallback to provided
    api_key = os.getenv("TURING_API_KEY", DEFAULT_KEY)

    flagged: List[Tuple[str, str, str]] = []
    checked_count = 0
    skipped_no_analysis = 0
    api_failures = 0

    # Walk
    for dirpath, _, filenames in os.walk(root_dir):
        # Prepare relative path for nicer printing
        rel_dir = os.path.relpath(dirpath, root_dir)

        # Build a small helper to process a specific file
        def process_file(json_name: str, flag_name: str) -> None:
            nonlocal checked_count, skipped_no_analysis, api_failures

            json_path = os.path.join(dirpath, json_name)
            if not os.path.isfile(json_path):
                return

            data = read_json(json_path)
            if data is None:
                return

            analysis_text = extract_analysis_field(data)
            if not analysis_text:
                print(f"{WARN} No 'analysis' field in: {os.path.join(rel_dir, json_name)}")
                skipped_no_analysis += 1
                return

            checked_count += 1
            print(f"{INFO} Checking: {os.path.join(rel_dir, json_name)}")

            ok, content = call_gpt5(api_key, analysis_text, timeout=args.timeout)
            if not ok:
                print(f"{ERR} API call failed for {os.path.join(rel_dir, json_name)}: {content}")
                api_failures += 1
                return

            verdict, reason = parse_verdict(content)
            if verdict is None:
                # Unparsable; still record to help debugging, but do not flag as bad by default
                print(f"{WARN} Unrecognized format from GPT-5 for {os.path.join(rel_dir, json_name)}; not flagging.")
                # You could choose to persist the raw content for inspection:
                # write_flag_file(dirpath, flag_name + ".unparsed", content)
                return

            if verdict is True:
                print(f"{BAD} Bad logic detected → {os.path.join(rel_dir, flag_name)}")
                write_flag_file(dirpath, flag_name, content)
                flagged.append((rel_dir, json_name, reason))
            else:
                print(f"{OK} Logic consistent: {os.path.join(rel_dir, json_name)}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        # Process "gt" file(s)
        for jname, flag in GT_FILES:
            if jname in filenames:
                process_file(jname, flag)

        # Process "remix" file(s)
        for jname, flag in REMIX_FILES:
            if jname in filenames:
                process_file(jname, flag)

    # Save summary
    save_summary(root_dir, flagged)

    # Final stats
    print(f"\n{INFO} Done.")
    print(f"{INFO} Files checked: {checked_count}")
    print(f"{INFO} Flagged (bad logic): {len(flagged)}")
    print(f"{INFO} Skipped (missing 'analysis' field): {skipped_no_analysis}")
    print(f"{INFO} API failures: {api_failures}")


if __name__ == "__main__":
    main()
