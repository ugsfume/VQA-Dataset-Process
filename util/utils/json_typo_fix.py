#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
json_typo_fix.py

Recursively scan all .json files from the current working directory,
and replace the following typos everywhere they appear (keys/values):

  "Sourcb"   -> "Source"
  "切cDrain" -> "切断Drain"

Behavior:
  - Creates a .bak backup next to each modified file (disable with --no-backup)
  - Verifies JSON validity after replacement before overwriting
  - Supports dry-run mode

Usage:
  python json_typo_fix.py
  python json_typo_fix.py --dry-run
  python json_typo_fix.py --no-backup
  python json_typo_fix.py --root /path/to/start
  python json_typo_fix.py -v
"""

import os
import sys
import json
import argparse
from typing import Tuple

# ---------- Pretty CLI tags ----------
from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)
OK   = Fore.GREEN   + "[OK]"   + Style.RESET_ALL
ERR  = Fore.RED     + "[ERR]"  + Style.RESET_ALL
INFO = Fore.CYAN    + "[INFO]" + Style.RESET_ALL
WARN = Fore.YELLOW  + "[WARN]" + Style.RESET_ALL
DRY  = Fore.MAGENTA + "[DRY]"  + Style.RESET_ALL
SKIP = Fore.BLUE    + "[SKIP]" + Style.RESET_ALL

TYPO_MAP = {
    "当缺陷同时发生在Gate和ITO组件区域时，需要检查Gate的断口宽度和长度。有以下两种情况，1.当Gate断口宽度大于等于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为a.用激光切割将缺陷两端的Gate组件切断，b.在每个激光切割的路径上覆盖一次ITO remove，c.Turn on T400；2.当Gate断口宽度小于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为用激光切割将Gate与ITO组件中间的缺陷部分切断，并在每个激光切割的路径上覆盖一次ITO remove。": "当缺陷同时发生在Gate和ITO组件区域时，需要检查Gate的断口宽度和长度。有以下两种情况，1.当Gate断口宽度大于等于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为a.用激光切割将缺陷两端的Gate组件切断，b.在每个激光切割的路径上覆盖一次ITO remove，c.Turn on T400；2.当Gate断口宽度小于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为d.用激光切割将Gate与ITO组件中间的缺陷部分切断，e.在每个激光切割的路径上覆盖一次ITO remove。",
    "切cDrain": "切断Drain",
}

def read_text(path: str) -> str:
    # Try UTF-8 first; fall back to UTF-8 with BOM if present
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()

def write_text(path: str, data: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(data)

def replace_typos(text: str) -> Tuple[str, int, int]:
    c1 = text.count("当缺陷同时发生在Gate和ITO组件区域时，需要检查Gate的断口宽度和长度。有以下两种情况，1.当Gate断口宽度大于等于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为a.用激光切割将缺陷两端的Gate组件切断，b.在每个激光切割的路径上覆盖一次ITO remove，c.Turn on T400；2.当Gate断口宽度小于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为用激光切割将Gate与ITO组件中间的缺陷部分切断，并在每个激光切割的路径上覆盖一次ITO remove。")
    c2 = text.count("切cDrain")
    new_text = text.replace("当缺陷同时发生在Gate和ITO组件区域时，需要检查Gate的断口宽度和长度。有以下两种情况，1.当Gate断口宽度大于等于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为a.用激光切割将缺陷两端的Gate组件切断，b.在每个激光切割的路径上覆盖一次ITO remove，c.Turn on T400；2.当Gate断口宽度小于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为用激光切割将Gate与ITO组件中间的缺陷部分切断，并在每个激光切割的路径上覆盖一次ITO remove。", "当缺陷同时发生在Gate和ITO组件区域时，需要检查Gate的断口宽度和长度。有以下两种情况，1.当Gate断口宽度大于等于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为a.用激光切割将缺陷两端的Gate组件切断，b.在每个激光切割的路径上覆盖一次ITO remove，c.Turn on T400；2.当Gate断口宽度小于完好Gate组件最窄处宽度的2/3，同时长度小于300um时，需做修补，具体手法为d.用激光切割将Gate与ITO组件中间的缺陷部分切断，e.在每个激光切割的路径上覆盖一次ITO remove。").replace("切cDrain", "切断Drain")
    return new_text, c1, c2

def process_json_file(path: str, dry_run: bool, make_backup: bool, verbose: bool) -> Tuple[bool, int, int]:
    original = read_text(path)
    replaced, c1, c2 = replace_typos(original)

    if (c1 + c2) == 0:
        if verbose:
            print(f"{SKIP} {path} (no matches)")
        return False, 0, 0

    # Validate JSON after replacement to avoid breaking files
    try:
        json.loads(replaced)
    except json.JSONDecodeError as e:
        print(f"{ERR} JSON became invalid after replacement in: {path}\n      {e}")
        return False, 0, 0

    if dry_run:
        print(f"{DRY} {path}: Sourcb→Source:{c1}, 切cDrain→切断Drain:{c2}")
        return False, c1, c2

    # Write backup if requested
    if make_backup:
        backup_path = path + ".bak"
        try:
            if not os.path.exists(backup_path):
                write_text(backup_path, original)
        except Exception as e:
            print(f"{ERR} Could not write backup for {path}: {e}")
            return False, 0, 0

    # Overwrite with fixed content
    try:
        write_text(path, replaced)
    except Exception as e:
        print(f"{ERR} Could not write updated file {path}: {e}")
        return False, 0, 0

    print(f"{OK}  {path}: Sourcb→Source:{c1}, 切cDrain→切断Drain:{c2}")
    return True, c1, c2

def iter_json_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".json"):
                yield os.path.join(dirpath, name)

def main() -> int:
    parser = argparse.ArgumentParser(description="Fix specific typos in JSON files recursively.")
    parser.add_argument("--root", default=".", help="Root directory to start scanning (default: current directory).")
    parser.add_argument("--dry-run", action="store_true", help="Show planned changes without modifying files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backups.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output (show skipped files).")
    args = parser.parse_args()

    print(f"{INFO} Scanning root: {os.path.abspath(args.root)}")
    total_files = 0
    changed_files = 0
    total_c1 = 0
    total_c2 = 0

    try:
        for path in iter_json_files(args.root):
            total_files += 1
            changed, c1, c2 = process_json_file(
                path,
                dry_run=args.dry_run,
                make_backup=not args.no_backup,
                verbose=args.verbose
            )
            if changed:
                changed_files += 1
                total_c1 += c1
                total_c2 += c2
            else:
                if args.dry_run:
                    total_c1 += c1
                    total_c2 += c2

        print("\n" + INFO + " Summary")
        print(f"  Files scanned:                 {total_files}")
        print(f"  Files modified:                {changed_files}{' (dry-run: 0 changes written)' if args.dry_run else ''}")
        print(f"  Replacements Sourcb→Source:    {total_c1}")
        print(f"  Replacements 切cDrain→切断Drain: {total_c2}")
        print(OK if (changed_files or args.dry_run) else WARN, " Done.")
        return 0
    except Exception as e:
        print(f"{ERR} {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
