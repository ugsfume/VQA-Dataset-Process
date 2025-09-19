#!/usr/bin/env bash
# repair_bench_queue.sh
# Queue and run the "repair/rule-check" benchmark across many models.
# Supports per-job output filenames, streams progress, logs per run,
# and uses per-file .done/.failed markers.

###############################################################################
# EDIT THESE DEFAULTS
###############################################################################
PYTHON="${PYTHON:-python}"
# If eval.py is elsewhere, set an absolute path here.
EVAL_SCRIPT="${EVAL_SCRIPT:-eval.py}"

# Optional: override the dataset used by eval.py (leave blank to use eval.py default)
JSON_PATH="${JSON_PATH:-}"

# Toggle streaming of eval output to terminal (1 = show, 0 = only log to file)
STREAM_TO_TTY="${STREAM_TO_TTY:-1}"

# If you want to pin a GPU: export CUDA_VISIBLE_DEVICES=0 before running, or set here.
# export CUDA_VISIBLE_DEVICES=0

###############################################################################
# QUEUE
# One line per job:
#   "model_path|save_dir|save_name.json"
# If save_name is omitted, we'll auto-name: pred_<parent_of_model>_<basename_of_model>.json
# NOTE: save_name must be a filename only (no slashes).
###############################################################################
read -r -d '' QUEUE <<'EOF'
/mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage1a/checkpoint-3495|/mnt/workspace/kennethliu/eval/dual_rehearsal_3/stage1a|qwen_dual_rehearsal_pred_7b_3495.json
/mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage1b/checkpoint-750|/mnt/workspace/kennethliu/eval/dual_rehearsal_3/stage1b|qwen_dual_rehearsal_pred_7b_750.json
/mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage1b/checkpoint-2000|/mnt/workspace/kennethliu/eval/dual_rehearsal_3/stage1b|qwen_dual_rehearsal_pred_7b_2000.json
/mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage2/checkpoint-250|/mnt/workspace/kennethliu/eval/dual_rehearsal_3/stage2|qwen_dual_rehearsal_pred_7b_250.json
EOF

# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_domain_bbox_judge/vit_llm_freeze_plus_vit2/checkpoint-100|/mnt/workspace/kennethliu/eval/domain_bbox_judge/vit_llm_freeze_plus_vit2|qwen_domain_bbox_judge_pred_7b_100.json
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_domain_filtered/vit_llm_freeze/checkpoint-2057|/mnt/workspace/kennethliu/eval/domain_filtered/vit_llm_freeze|qwen_domain_filtered_pred_7b_2057.json
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal/stage1a/checkpoint-2968|/mnt/workspace/kennethliu/eval/dual_rehearsal/stage1a|qwen_dual_rehearsal_pred_7b_2968.json
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain/llm_freeze/checkpoint-2000|/mnt/workspace/kennethliu/eval/mask_domain/llm_freeze|qwen_mask_domain_pred_7b_2000.json
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain/vit_freeze/resume/checkpoint-1000|/mnt/workspace/kennethliu/eval/mask_domain/vit_freeze|qwen_mask_domain_pred_7b_1400.json
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain/vit_freeze_4096tok/checkpoint-1400|/mnt/workspace/kennethliu/eval/mask_domain/vit_freeze_4096tok|qwen_mask_domain_pred_7b_1400.json
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_mixed/vit_freeze/checkpoint-750|/mnt/workspace/kennethliu/eval/mask_domain_mixed/vit_freeze|qwen_mask_domain_mixed_pred_7b_750.json
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-32b_mask_domain_mixed/vit_llm_freeze/checkpoint-750|/mnt/workspace/kennethliu/eval/32b_mask_domain_mixed/vit_llm_freeze|qwen_mask_domain_mixed_pred_32b_750.json
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-32b_mask_domain_mixed_judge/vit_llm_freeze/checkpoint-50|/mnt/workspace/kennethliu/eval/32b_mask_domain_mixed_judge/vit_llm_freeze|qwen_mask_domain_mixed_judge_pred_32b_50.json


###############################################################################
# BEHAVIOR / MARKERS
###############################################################################
DONE_SUFFIX=".done"
FAIL_SUFFIX=".failed"

###############################################################################
# Helpers
###############################################################################
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { printf "[%s] %s\n" "$(ts)" "$*"; }

have_gnu_time() {
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v true >/dev/null 2>&1
    return $?
  fi
  return 1
}

default_save_name() {
  local model_path="$1"
  local base="$(basename "$model_path")"
  local parent="$(basename "$(dirname "$model_path")")"
  echo "pred_${parent}_${base}.json"
}

sanitize() {
  # Make a string safe for filenames in logs; not used to write results.
  local s="$1"
  s="${s//[^A-Za-z0-9._-]/_}"
  echo "$s"
}

run_job() {
  local model_path="$1"
  local save_dir="$2"
  local save_name="$3"

  mkdir -p "$save_dir"

  # Auto-name if not provided
  if [[ -z "$save_name" ]]; then
    save_name="$(default_save_name "$model_path")"
  fi

  # Enforce filename-only for save_name
  if [[ "$save_name" == */* ]]; then
    log "ERROR: save_name must be a filename only (no slashes): $save_name"
    return 1
  fi

  local out_path="$save_dir/$save_name"
  local done_marker="${out_path}${DONE_SUFFIX}"
  local fail_marker="${out_path}${FAIL_SUFFIX}"

  # Skip logic: if output JSON exists or per-file .done marker exists
  if [[ -f "$out_path" || -f "$done_marker" ]]; then
    log "SKIP (exists/done): $model_path -> $out_path"
    return 0
  fi

  # Clear per-file fail marker if retrying
  [[ -f "$fail_marker" ]] && rm -f "$fail_marker"

  local save_name_safe
  save_name_safe="$(sanitize "$save_name")"
  local log_file="$save_dir/run_$(date +%Y%m%d_%H%M%S)__${save_name_safe}.log"

  log "START:   $model_path"
  log "SAVE TO: $out_path"
  log "Logging: $log_file"

  # Record environment & inputs
  {
    echo "=== START $(ts) ==="
    echo "PYTHON: $PYTHON"
    echo "EVAL_SCRIPT: $EVAL_SCRIPT"
    echo "MODEL: $model_path"
    echo "SAVE_DIR: $save_dir"
    echo "SAVE_NAME: $save_name"
    [[ -n "$JSON_PATH" ]] && echo "JSON_PATH: $JSON_PATH"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo
  } | tee -a "$log_file"

  local status
  local -a cmd=( env PYTHONUNBUFFERED=1 "$PYTHON" -u "$EVAL_SCRIPT"
                 --model "$model_path"
                 --save-dir "$save_dir"
                 --save-name "$save_name" )

  # Optionally pass JSON_PATH if set
  if [[ -n "$JSON_PATH" ]]; then
    cmd+=( --json-path "$JSON_PATH" )
  fi

  if [[ "$STREAM_TO_TTY" == "1" ]]; then
    set -o pipefail
    if have_gnu_time; then
      /usr/bin/time -v "${cmd[@]}" 2>&1 | tee -a "$log_file"
      status=${PIPESTATUS[0]}
    else
      "${cmd[@]}" 2>&1 | tee -a "$log_file"
      status=${PIPESTATUS[0]}
    fi
    set +o pipefail
  else
    if have_gnu_time; then
      { /usr/bin/time -v "${cmd[@]}"; } >>"$log_file" 2>&1
      status=$?
    else
      { "${cmd[@]}"; } >>"$log_file" 2>&1
      status=$?
    fi
  fi

  if [[ $status -eq 0 ]]; then
    touch "$done_marker"
    log "DONE: $model_path -> $out_path"
    echo "=== DONE $(ts) ===" >>"$log_file"
    return 0
  else
    echo "=== FAILED $(ts) (exit $status) ===" >>"$log_file"
    touch "$fail_marker"
    log "FAIL (exit $status): $model_path -> $out_path (see $log_file)"
    return $status
  fi
}

###############################################################################
# Main
###############################################################################
# Safety check
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Error: EVAL_SCRIPT not found at $EVAL_SCRIPT" >&2
  exit 1
fi

# Parse queue lines
IFS=$'\n' read -r -d '' -a LINES < <(printf "%s" "$QUEUE"$'\0')

declare -i total=0 ok=0 fail=0 skipped=0
declare -a FAIL_LIST=()
declare -a SKIP_LIST=()

for line in "${LINES[@]}"; do
  # Trim whitespace
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" || "$line" == \#* ]] && continue

  total+=1
  IFS='|' read -r model_path save_dir save_name <<<"$line"

  # Pre-check skip (compute out_path & markers)
  if [[ -z "$save_name" ]]; then
    save_name="$(default_save_name "$model_path")"
  fi
  if [[ "$save_name" == */* ]]; then
    log "ERROR (pre): save_name must be a filename only: $save_name"
    FAIL_LIST+=("$model_path|$save_dir|$save_name")
    fail+=1
    continue
  fi

  out_path="$save_dir/$save_name"
  done_marker="${out_path}${DONE_SUFFIX}"

  if [[ -f "$out_path" || -f "$done_marker" ]]; then
    log "SKIP (pre-check): $model_path -> $out_path"
    SKIP_LIST+=("$model_path|$save_dir|$save_name")
    skipped+=1
    continue
  fi

  if run_job "$model_path" "$save_dir" "$save_name"; then
    ok+=1
  else
    fail+=1
    FAIL_LIST+=("$model_path|$save_dir|$save_name")
  fi
done

echo
log "SUMMARY: total=$total ok=$ok fail=$fail skipped=$skipped"
if (( fail > 0 )); then
  echo "Failed runs:"
  for item in "${FAIL_LIST[@]}"; do
    echo "  - $item"
  done
fi
if (( skipped > 0 )); then
  echo "Skipped runs:"
  for item in "${SKIP_LIST[@]}"; do
    echo "  - $item"
  done
fi
