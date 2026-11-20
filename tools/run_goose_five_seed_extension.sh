#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=python
export PATH=/path/to/miniconda3/envs/sparseocc/bin:$PATH
BASE=work_dir/GOOSETemporalOccMono

validate_run() {
  local gpu="$1"
  local config="$2"
  local run_name="$3"
  local seed="$4"
  local run_dir="$BASE/$run_name"
  local output="$run_dir/val_best.json"
  local checkpoint
  local checkpoint_count

  checkpoint_count=$(find "$run_dir" -maxdepth 1 -type f \
    -name 'best_O-SSC-mIoU_epoch_*.pth' -print | wc -l)
  [[ "$checkpoint_count" -eq 1 ]]
  checkpoint=$(find "$run_dir" -maxdepth 1 -type f \
    -name 'best_O-SSC-mIoU_epoch_*.pth' -print)
  [[ -n "$checkpoint" && -f "$checkpoint" ]]
  if [[ -f "$output" ]]; then
    echo "[$(date --iso-8601=seconds)] Existing validation: $output"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" val.py \
    --config "$config" \
    --weights "$checkpoint" \
    --world_size 1 \
    --override "seed=$seed" \
    --out-json "$output"
}

train_run() {
  local gpu="$1"
  local config="$2"
  local run_name="$3"
  local seed="$4"
  local run_dir="$BASE/$run_name"

  if [[ -f "$run_dir/epoch_24.pth" ]]; then
    echo "[$(date --iso-8601=seconds)] Existing complete run: $run_name"
  else
    if [[ -e "$run_dir" ]]; then
      echo "Refusing to overwrite incomplete run directory: $run_dir" >&2
      return 1
    fi
    echo "[$(date --iso-8601=seconds)] START gpu=$gpu seed=$seed $run_name"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" train.py \
      --config "$config" \
      --run_name "$run_name" \
      --world_size 1 \
      --override "seed=$seed"
  fi
  validate_run "$gpu" "$config" "$run_name" "$seed"
  echo "[$(date --iso-8601=seconds)] COMPLETE gpu=$gpu seed=$seed $run_name"
}

run_seed() {
  local gpu="$1"
  local seed="$2"
  train_run "$gpu" configs/goose_temporal_common10_3h_binary_conf.py \
    "goose_temporal_common10_3h_binary_conf_seed${seed}" "$seed"
  train_run "$gpu" configs/goose_temporal_occlusion_common10_3h.py \
    "goose_temporal_occlusion_common10_3h_seed${seed}" "$seed"
}

pids=()
run_seed 0 3 &
pids+=("$!")
run_seed 1 4 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one five-seed extension branch failed" >&2
  exit "$status"
fi

"$PYTHON" tools/summarize_goose_experiments.py
"$PYTHON" tools/decide_goose_claim_gate.py
"$PYTHON" tools/plot_goose_paper_figures.py
"$PYTHON" tools/freeze_goose_final_results.py
echo "[$(date --iso-8601=seconds)] FIVE-SEED EXTENSION COMPLETE"
