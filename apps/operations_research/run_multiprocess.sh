#!/bin/bash
# Runner for run_exp_with_kb_full_multiprocess.py
# Usage examples:
#   ./apps/operations_research/run_multiprocess.sh
#   ./apps/operations_research/run_multiprocess.sh -d BWOR -m o3 -p 100
#   ./apps/operations_research/run_multiprocess.sh --dataset industryor --model o4-mini --use_iterative_refinement
set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
DATASET="industryor"        # BWOR | industryor | complexlp
MODEL="o4-mini"
NUM_PROC=""                 # empty => number of CPU cores
EXTRA=()                    # passthrough flags (e.g. --use_iterative_refinement)

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dataset)        DATASET="$2"; shift 2 ;;
    -m|--model)          MODEL="$2"; shift 2 ;;
    -p|--num_processes)  NUM_PROC="$2"; shift 2 ;;
    *)                   EXTRA+=("$1"); shift ;;   # pass anything else straight through
  esac
done

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(python -m apps.operations_research.run_exp_with_kb_full_multiprocess
     --dataset "$DATASET"
     --model_id "$MODEL")

[[ -n "$NUM_PROC" ]] && CMD+=(--num_processes "$NUM_PROC")
CMD+=("${EXTRA[@]}")

echo "Running: ${CMD[*]}"
"${CMD[@]}"
