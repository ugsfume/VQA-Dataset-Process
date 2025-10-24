#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
find_tscok_repair_rules.py
Scan from the directory this script is executed in (recursively) and list all
`repair_rule.json` files that include a `damaged_component` == "TSCOK".

Usage:
  python find_tscok_repair_rules.py
  python find_tscok_repair_rules.py --root /path/to/search
  python find_tscok_repair_rules.py --absolute
  python find_tscok_repair_rules.py --verbose
"""

import os
import json
import argparse
from typing import Any

from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)
OK   = Fore.GREEN + "[OK]" + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
ERR  = Fore.RED + "[ERR]" + Style.RESET_ALL
INFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def contains_tscok_strict(obj: Any) -> bool:
    """
    Strict check: returns True if any dict has damaged_component == "TSCOK".
    """
    if isinstance(obj, dict):
        if str(obj.get("damaged_component", "")).strip() == "TSCOK":
            return True
        # Recurse into values
        for v in obj.values():
            if contains_tscok_strict(v):
                return True
        return False
    if isinstance(obj, list):
        return any(contains_tscok_strict(v) for v in obj)
    return False

def main():
    ap = argparse.ArgumentParser(description="Find all repair_rule.json that include a TSCOK entry.")
    ap.add_argument("--root", type=str, default=".", help="Root folder to scan (default: current dir).")
    ap.add_argument("--absolute", action="store_true", help="Print absolute paths (default: relative).")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if args.verbose:
        print(f"{INFO} Scanning from: {root}")

    hits = []
    checked = 0
    errors = 0

    for dirpath, dirnames, filenames in os.walk(root):
        if "repair_rule.json" not in filenames:
            continue
        path = os.path.join(dirpath, "repair_rule.json")
        try:
            data = load_json(path)
            checked += 1
            if contains_tscok_strict(data):
                hits.append(path)
                rel = path if args.absolute else os.path.relpath(path, root)
                print(f"{OK} {rel}")
            elif args.verbose:
                rel = path if args.absolute else os.path.relpath(path, root)
                print(f"{WARN} No TSCOK in {rel}")
        except Exception as e:
            errors += 1
            rel = path if args.absolute else os.path.relpath(path, root)
            print(f"{ERR} Failed to parse {rel}: {e}")

    print()
    print(f"{OK} Done. Checked: {checked}, Found TSCOK: {len(hits)}, Errors: {errors}")

if __name__ == "__main__":
    main()
