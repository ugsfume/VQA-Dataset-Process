#!/usr/bin/env bash
# domain_bench_queue.sh
# Queue and run multiple VLM benchmark jobs sequentially with logging & summary.
# Shows live progress in the terminal while also writing logs to file.

###############################################################################
# EDIT THESE DEFAULTS (paths from your examples)
###############################################################################
PYTHON="${PYTHON:-python}"
EVAL_SCRIPT="/mnt/workspace/kennethliu/src/eval_vqa/eval_vqa.py"
SINGLE_JSON="/mnt/workspace/kennethliu/TFT_circuit_images/test_set_2/concat_qa.jsonl"
DUAL_JSON="/mnt/workspace/kennethliu/TFT_circuit_images/test_set_2/concat_qa_mask.jsonl"

# If you want to pin a GPU: export CUDA_VISIBLE_DEVICES=0 before running, or set here.
# export CUDA_VISIBLE_DEVICES=0

# Toggle streaming of eval output to terminal (1 = show, 0 = only log to file)
STREAM_TO_TTY="${STREAM_TO_TTY:-1}"

###############################################################################
# QUEUE: One "model_path|out_dir" per line. No spaces around the pipe.
###############################################################################
read -r -d '' QUEUE <<'EOF'
/mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage1b/checkpoint-2000|/mnt/workspace/kennethliu/eval/dual_rehearsal_3/stage1b/eval_results_2000
EOF

# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage1a/checkpoint-3495|/mnt/workspace/kennethliu/eval/dual_rehearsal_3/stage1a/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage1b/checkpoint-750|/mnt/workspace/kennethliu/eval/dual_rehearsal_3/stage1b/eval_results_750
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage2/checkpoint-250|/mnt/workspace/kennethliu/eval/dual_rehearsal_3/stage2/eval_results

# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_domain_bbox_judge/vit_llm_freeze_plus_vit2/checkpoint-100|/mnt/workspace/kennethliu/eval/domain_bbox_judge/vit_llm_freeze_plus_vit2/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_domain_filtered/vit_llm_freeze/checkpoint-2057|/mnt/workspace/kennethliu/eval/domain_filtered/vit_llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_domain_judge/full/sft/checkpoint-270|/mnt/workspace/kennethliu/eval/domain_judge/full/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_domain_judge/vit_freeze/checkpoint-270|/mnt/workspace/kennethliu/eval/domain_judge/vit_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_domain_judge/vit_llm_freeze/checkpoint-540|/mnt/workspace/kennethliu/eval/domain_judge/vit_llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_domain_judge_filtered/vit_llm_freeze/checkpoint-540|/mnt/workspace/kennethliu/eval/domain_judge_filtered/vit_llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal/stage1a/checkpoint-2968|/mnt/workspace/kennethliu/eval/dual_rehearsal/stage1a/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain/llm_freeze/checkpoint-2000|/mnt/workspace/kennethliu/eval/mask_domain/llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain/vit_freeze/resume/checkpoint-1000|/mnt/workspace/kennethliu/eval/mask_domain/vit_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain/vit_freeze_4096tok/checkpoint-1400|/mnt/workspace/kennethliu/eval/mask_domain/vit_freeze_4096tok/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain/vit_llm_freeze/checkpoint-1400|/mnt/workspace/kennethliu/eval/mask_domain/vit_llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_judge/llm_freeze/checkpoint-200|/mnt/workspace/kennethliu/eval/mask_domain_judge/llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_judge/vit_freeze/checkpoint-100|/mnt/workspace/kennethliu/eval/mask_domain_judge/vit_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_judge/vit_freeze_4096/repair_proj_unfreeze/checkpoint-270|/mnt/workspace/kennethliu/eval/mask_domain_judge/vit_freeze_4096/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_judge/vit_freeze_4096tok/checkpoint-100|/mnt/workspace/kennethliu/eval/mask_domain_judge/vit_freeze_4096tok/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_judge/vit_llm_freeze/checkpoint-270|/mnt/workspace/kennethliu/eval/mask_domain_judge/vit_llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_mixed/vit_freeze/checkpoint-750|/mnt/workspace/kennethliu/eval/mask_domain_mixed/vit_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_mixed/vit_llm_freeze/checkpoint-1250|/mnt/workspace/kennethliu/eval/mask_domain_mixed/vit_llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_mixed_judge/vit_freeze/checkpoint-100|/mnt/workspace/kennethliu/eval/mask_domain_mixed_judge/vit_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_mixed_judge/vit_freeze2/checkpoint-270|/mnt/workspace/kennethliu/eval/mask_domain_mixed_judge/vit_freeze2/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_mask_domain_mixed_judge/vit_llm_freeze/checkpoint-270|/mnt/workspace/kennethliu/eval/mask_domain_mixed_judge/vit_llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-32b_mask_domain_mixed/vit_llm_freeze/checkpoint-750|/mnt/workspace/kennethliu/eval/32b_mask_domain_mixed/vit_llm_freeze/eval_results
# /mnt/workspace/kennethliu/ckpt/qwen2_5vl-32b_mask_domain_mixed_judge/vit_llm_freeze/checkpoint-50|/mnt/workspace/kennethliu/eval/32b_mask_domain_mixed_judge/vit_llm_freeze/eval_results


###############################################################################
# BEHAVIOR
###############################################################################
DONE_MARKER=".done"
FAIL_MARKER=".failed"
# If your eval script already writes a metrics file you prefer to check, set it:
# METRICS_SENTINEL="metrics_single.json"
METRICS_SENTINEL=""

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

run_job() {
  local model_path="$1"
  local out_dir="$2"

  mkdir -p "$out_dir"

  # Skip logic
  if [[ -n "$METRICS_SENTINEL" && -f "$out_dir/$METRICS_SENTINEL" ]]; then
    log "SKIP (metrics present): $model_path -> $out_dir"
    return 0
  fi
  if [[ -f "$out_dir/$DONE_MARKER" ]]; then
    log "SKIP (done marker): $model_path -> $out_dir"
    return 0
  fi

  # Clear fail marker if retrying
  [[ -f "$out_dir/$FAIL_MARKER" ]] && rm -f "$out_dir/$FAIL_MARKER"

  local log_file="$out_dir/run_$(date +%Y%m%d_%H%M%S).log"
  log "START: $model_path -> $out_dir"
  log "Logging to: $log_file"

  # Record environment & inputs
  {
    echo "=== START $(ts) ==="
    echo "PYTHON: $PYTHON"
    echo "EVAL_SCRIPT: $EVAL_SCRIPT"
    echo "MODEL: $model_path"
    echo "SINGLE_JSON: $SINGLE_JSON"
    echo "DUAL_JSON:   $DUAL_JSON"
    echo "OUT_DIR: $out_dir"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo
  } | tee -a "$log_file"

  local status

  if [[ "$STREAM_TO_TTY" == "1" ]]; then
    # Stream to terminal and log; capture true exit status of the left command.
    set -o pipefail
    if have_gnu_time; then
      # Force unbuffered Python output for timely progress
      /usr/bin/time -v env PYTHONUNBUFFERED=1 "$PYTHON" -u "$EVAL_SCRIPT" \
        --model "$model_path" \
        --single-json "$SINGLE_JSON" \
        --dual-json "$DUAL_JSON" \
        --out-dir "$out_dir" 2>&1 | tee -a "$log_file"
      status=${PIPESTATUS[0]}
    else
      env PYTHONUNBUFFERED=1 "$PYTHON" -u "$EVAL_SCRIPT" \
        --model "$model_path" \
        --single-json "$SINGLE_JSON" \
        --dual-json "$DUAL_JSON" \
        --out-dir "$out_dir" 2>&1 | tee -a "$log_file"
      status=${PIPESTATUS[0]}
    fi
    set +o pipefail
  else
    # Quiet mode: only write to log file
    if have_gnu_time; then
      { /usr/bin/time -v env PYTHONUNBUFFERED=1 "$PYTHON" -u "$EVAL_SCRIPT" \
          --model "$model_path" \
          --single-json "$SINGLE_JSON" \
          --dual-json "$DUAL_JSON" \
          --out-dir "$out_dir"; } >>"$log_file" 2>&1
      status=$?
    else
      { env PYTHONUNBUFFERED=1 "$PYTHON" -u "$EVAL_SCRIPT" \
          --model "$model_path" \
          --single-json "$SINGLE_JSON" \
          --dual-json "$DUAL_JSON" \
          --out-dir "$out_dir"; } >>"$log_file" 2>&1
      status=$?
    fi
  fi

  if [[ $status -eq 0 ]]; then
    touch "$out_dir/$DONE_MARKER"
    log "DONE: $model_path -> $out_dir"
    echo "=== DONE $(ts) ===" >>"$log_file"
    return 0
  else
    echo "=== FAILED $(ts) (exit $status) ===" >>"$log_file"
    touch "$out_dir/$FAIL_MARKER"
    log "FAIL (exit $status): $model_path -> $out_dir (see $log_file)"
    return $status
  fi
}

###############################################################################
# Main
###############################################################################
# Safety checks
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Error: EVAL_SCRIPT not found at $EVAL_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SINGLE_JSON" ]]; then
  echo "Error: SINGLE_JSON not found at $SINGLE_JSON" >&2
  exit 1
fi
if [[ ! -f "$DUAL_JSON" ]]; then
  echo "Error: DUAL_JSON not found at $DUAL_JSON" >&2
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
  IFS='|' read -r model_path out_dir <<<"$line"

  # Pre-check for skip
  if [[ -n "$METRICS_SENTINEL" && -f "$out_dir/$METRICS_SENTINEL" ]] || [[ -f "$out_dir/$DONE_MARKER" ]]; then
    log "SKIP (pre-check): $model_path -> $out_dir"
    SKIP_LIST+=("$model_path|$out_dir")
    skipped+=1
    continue
  fi

  if run_job "$model_path" "$out_dir"; then
    ok+=1
  else
    fail+=1
    FAIL_LIST+=("$model_path|$out_dir")
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
