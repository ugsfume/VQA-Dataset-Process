#!/usr/bin/env bash
# Lists subdirectories (recursive) that do NOT contain analysis_gt.json directly inside.
# Usage:
#   ./list_missing_analysis_gt.sh            # scan from current directory
#   ./list_missing_analysis_gt.sh /path/dir  # scan from a given root

set -euo pipefail

ROOT="${1:-.}"

if [[ ! -d "$ROOT" ]]; then
  echo "Error: '$ROOT' is not a directory" >&2
  exit 1
fi

# -mindepth 1: skip the ROOT itself; only list subdirectories
# For each directory, print it if $dir/analysis_gt.json does NOT exist
find "$ROOT" -mindepth 1 -type d -print0 \
| while IFS= read -r -d '' dir; do
    if [[ ! -e "$dir/analysis_gt.json" ]]; then
      printf '%s\n' "$dir"
    fi
  done