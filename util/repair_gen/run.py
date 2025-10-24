#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py — pipeline entry for:
  1) random_negative_augmentor.py
  2) visualize_random_overlays.py
  3) repair_rule_extractor.py

Run from dataset root (gt_datasets_20250915):
  python repair_gen/run.py
"""

import os
import sys
import shlex
import subprocess
from colorama import init as colorama_init, Fore, Style

# Pretty logs
colorama_init(autoreset=True)
OK   = Fore.GREEN + "[OK]" + Style.RESET_ALL
WARN = Fore.YELLOW + "[WARN]" + Style.RESET_ALL
ERR  = Fore.RED + "[ERR]" + Style.RESET_ALL
INFO = Fore.CYAN + "[INFO]" + Style.RESET_ALL

# Load params
try:
    from .params import Params  # if run as a module
except Exception:
    # fallback if executed as a script without package context
    from params import Params

def script_path(name: str) -> str:
    """Resolve a script filename inside this folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, name)

def run_cmd(cmd_list, cwd=None):
    """Run a subprocess with nice printing and error propagation."""
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd_list)
    print(f"{INFO} $ {cmd_str}")
    subprocess.run(cmd_list, check=True, cwd=cwd)

def preflight():
    """Basic checks for required companion files."""
    here = os.path.dirname(os.path.abspath(__file__))
    need = [
        "random_negative_augmentor.py",
        "visualize_random_overlays.py",
        "repair_rule_extractor.py",
        "repair_rules_lookup_table.py",
        "repair_rules.json",
    ]
    missing = [n for n in need if not os.path.isfile(os.path.join(here, n))]
    if missing:
        print(f"{ERR} Missing in repair_gen/: {', '.join(missing)}")
        sys.exit(1)

def main():
    preflight()

    py = sys.executable
    ds_root = os.path.abspath(Params.dataset_root)
    pipeline_root = os.path.dirname(os.path.abspath(__file__))

    print(f"{INFO} Dataset root: {ds_root}")
    print(f"{INFO} Pipeline dir: {pipeline_root}")

    # 1) AUGMENT
    if Params.steps.get("augment", True):
        a = Params.augment
        aug_script = script_path("random_negative_augmentor.py")
        include = ",".join(a["include"])
        ops = ",".join(a["ops"])

        cmd = [
            py, aug_script,
            "--root", ds_root,
            "--n", str(a["n"]),
            "--include", include,
            "--ops", ops,
            "--shift-std", str(a["shift_std"]),
            "--scale-std", str(a["scale_std"]),
            "--p-delete", str(a["p_delete"]),
            "--img-w", str(a["img_w"]),
            "--img-h", str(a["img_h"]),
        ]
        if a.get("seed") is not None:
            cmd += ["--seed", str(a["seed"])]
        run_cmd(cmd)

    else:
        print(f"{WARN} Skipping augmentation stage")

    # 2) VISUALIZE
    if Params.steps.get("visualize", True):
        v = Params.visualize
        viz_script = script_path("visualize_random_overlays.py")
        cmd = [
            py, viz_script,
            "--root", ds_root,
        ]
        if v.get("defect_open", False):
            cmd.append("--defect-open")
        if v.get("verbose", False):
            cmd.append("--verbose")
        run_cmd(cmd)
    else:
        print(f"{WARN} Skipping visualization stage")

    # 3) EXTRACT RULES
    if Params.steps.get("extract_rules", True):
        e = Params.extract_rules
        ext_script = script_path("repair_rule_extractor.py")
        cmd = [
            py, ext_script,
            "--root", ds_root,
            "--mask-thresh", str(e["mask_thresh"]),
            "--min-pixels", str(e["min_pixels"]),
        ]
        if e.get("only_missing", False):
            cmd.append("--only-missing")
        if e.get("verbose", False):
            cmd.append("--verbose")
        run_cmd(cmd)
    else:
        print(f"{WARN} Skipping rule extraction stage")

    print(f"{OK} Pipeline completed.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"{ERR} Stage failed with exit code {e.returncode}")
        sys.exit(e.returncode)
