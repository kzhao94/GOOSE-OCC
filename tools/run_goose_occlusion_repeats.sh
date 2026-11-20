#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=python
PATH=/path/to/miniconda3/envs/sparseocc/bin:$PATH
export PATH
BASE=work_dir/GOOSETemporalOccMono

wait_for_run_complete() {
  local work_dir="$1"
  local run_name
  run_name=$(basename "$work_dir")
  while [[ ! -f "$work_dir/epoch_24.pth" ]] \
      || pgrep -af '[/]python train.py' \
        | grep -F -- "--run_name $run_name" >/dev/null; do
    sleep 60
  done
}

wait_for_run_complete \
  "$BASE/goose_temporal_occlusion_common10_3h_weight100_seed0"

for candidate in \
  goose_temporal_occlusion_common10_3h_weight025_seed0 \
  goose_temporal_occlusion_common10_3h_seed0; do
  while [[ ! -f "$BASE/$candidate/val_best.json" ]]; do
    sleep 60
  done
done

validate_if_needed() {
  local config="$1"
  local work_dir="$2"
  local output="$work_dir/val_best.json"
  [[ -f "$output" ]] && return
  local checkpoint
  checkpoint=$(find "$work_dir" -maxdepth 1 -type f \
    -name 'best_O-SSC-mIoU_epoch_*.pth' -print | sort -V | tail -n 1)
  [[ -n "$checkpoint" && -f "$checkpoint" ]]
  local seed
  seed=$($PYTHON - "$work_dir/resolved_config.json" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1]))['seed']))
PY
)
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    val.py --config "$config" --weights "$checkpoint" --world_size 2 \
    --override "seed=$seed" --out-json "$output"
}

validate_if_needed configs/goose_temporal_occlusion_common10_3h_weight100.py \
  "$BASE/goose_temporal_occlusion_common10_3h_weight100_seed0"

selection=$($PYTHON - <<'PY'
import json
from pathlib import Path
specs = [
    ('configs/goose_temporal_occlusion_common10_3h_weight025.py',
     'goose_temporal_occlusion_common10_3h_weight025_seed0', 0.25),
    ('configs/goose_temporal_occlusion_common10_3h.py',
     'goose_temporal_occlusion_common10_3h_seed0', 0.5),
    ('configs/goose_temporal_occlusion_common10_3h_weight100.py',
     'goose_temporal_occlusion_common10_3h_weight100_seed0', 1.0),
]
rows = []
for config, name, weight in specs:
    path = Path('work_dir/GOOSETemporalOccMono') / name / 'val_best.json'
    if path.is_file():
        payload = json.loads(path.read_text())
        metrics = payload.get('metrics', payload)
        rows.append((
            float(metrics['O-SSC-mIoU']), abs(weight - 0.5), weight,
            config, name))
if not rows:
    raise SystemExit('no complete occlusion validation results')
rows.sort(key=lambda item: (-item[0], item[1], item[2]))
print(rows[0][3], rows[0][4], '{:.4f}'.format(rows[0][0]))
PY
)
read -r SELECTED_CONFIG SELECTED_STEM SELECTED_SCORE <<< "$selection"
echo "selected $SELECTED_CONFIG $SELECTED_STEM O-SSC-mIoU=$SELECTED_SCORE"

train_pids=()
for seed in 1 2; do
  run_name="${SELECTED_STEM%_seed0}_seed${seed}"
  work_dir="$BASE/$run_name"
  if [[ ! -f "$work_dir/epoch_24.pth" ]]; then
    gpu=$((seed - 1))
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" train.py \
      --config "$SELECTED_CONFIG" --run_name "$run_name" --world_size 1 \
      --override "seed=$seed" &
    train_pids+=("$!")
  fi
done
for pid in "${train_pids[@]}"; do
  wait "$pid"
done
for seed in 1 2; do
  run_name="${SELECTED_STEM%_seed0}_seed${seed}"
  validate_if_needed "$SELECTED_CONFIG" "$BASE/$run_name"
done
FINAL_DIR="$BASE/$SELECTED_STEM"
FINAL_CHECKPOINT=$(find "$FINAL_DIR" -maxdepth 1 -type f \
  -name 'best_O-SSC-mIoU_epoch_*.pth' -print | sort -V | tail -n 1)
QUAL_DIR="paper/cviu_sparseocc_manuscript/figures/generated/goose_predictions"
if [[ -n "$FINAL_CHECKPOINT" && ! -f "$QUAL_DIR/prediction_0000.npz" ]]; then
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    val.py --config "$SELECTED_CONFIG" --weights "$FINAL_CHECKPOINT" \
    --world_size 2 --dump-predictions "$QUAL_DIR" --dump-count 4 \
    --out-json "$FINAL_DIR/val_qualitative.json"
  $PYTHON tools/plot_goose_qualitative_predictions.py \
    --prediction-dir "$QUAL_DIR" \
    --baseline-prediction-dir \
      paper/cviu_sparseocc_manuscript/figures/generated/goose_predictions_baseline
fi
$PYTHON tools/summarize_goose_experiments.py
$PYTHON tools/plot_goose_paper_figures.py
