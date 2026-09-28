#!/usr/bin/env bash
# Use identical calibration IDs for every score/allocation comparison.
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 MODEL CALIBRATION.npy OUTPUT_DIR [SPARSITY] [extra prune options...]" >&2
  exit 2
fi
model_id=$1
calibration_file=$2
run_root=$3
target_sparsity=${4:-0.5}
if [[ $# -ge 4 ]]; then shift 4; else shift 3; fi
for setting in wanda-row wanda-matrix qk-separate qk-shared; do
  case "$setting" in
    wanda-row) method=wanda; budget=row ;;
    wanda-matrix) method=wanda; budget=separate ;;
    qk-separate) method=qk-wanda; budget=separate ;;
    qk-shared) method=qk-wanda; budget=shared ;;
  esac
  qk-wanda prune --model "$model_id" --calibration-tokens "$calibration_file" \
    --output "$run_root/$setting" --sparsity "$target_sparsity" \
    --method "$method" --budget "$budget" --rounding ceil --eval "$@"
done
