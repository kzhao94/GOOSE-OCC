#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=python
export PATH=/path/to/miniconda3/envs/sparseocc/bin:$PATH
GPU=${WEAK_PRED_GPU:-1}
OUTPUT_ROOT=work_dir/formal_predictions

runs=(
  GOOSEOccMono/goose_mono_common10_seed0
  GOOSEOccMono/goose_mono_common10_seed1
  GOOSEOccMono/goose_mono_common10_seed2
  GOOSETemporalOccMono/goose_temporal_common10_3h_binary_conf_seed0
  GOOSETemporalOccMono/goose_temporal_common10_3h_binary_conf_seed1
  GOOSETemporalOccMono/goose_temporal_common10_3h_binary_conf_seed2
  GOOSETemporalOccMono/goose_temporal_common10_3h_binary_conf_seed3
  GOOSETemporalOccMono/goose_temporal_common10_3h_binary_conf_seed4
  GOOSETemporalOccMono/goose_temporal_occlusion_common10_3h_seed0
  GOOSETemporalOccMono/goose_temporal_occlusion_common10_3h_seed1
  GOOSETemporalOccMono/goose_temporal_occlusion_common10_3h_seed2
  GOOSETemporalOccMono/goose_temporal_occlusion_common10_3h_seed3
  GOOSETemporalOccMono/goose_temporal_occlusion_common10_3h_seed4
)

for relative in "${runs[@]}"; do
  run_dir="work_dir/$relative"
  run_name="$(basename "$run_dir")"
  output_dir="$OUTPUT_ROOT/$run_name"
  audit_json="$output_dir/prediction_audit.json"
  checkpoint_count=$(find "$run_dir" -maxdepth 1 -type f \
    -name 'best_O-SSC-mIoU_epoch_*.pth' -print | wc -l)
  [[ -f "$run_dir/epoch_24.pth" && -f "$run_dir/val_best.json" ]]
  [[ "$checkpoint_count" -eq 1 ]]
  checkpoint=$(find "$run_dir" -maxdepth 1 -type f \
    -name 'best_O-SSC-mIoU_epoch_*.pth' -print)

  readarray -t metadata < <("$PYTHON" - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
result = json.loads((run_dir / 'val_best.json').read_text())
resolved = json.loads((run_dir / 'resolved_config.json').read_text())
print(result['config'])
seed = int(resolved.get('seed', result.get('seed', 0)))
if seed != int(result['seed']):
    raise ValueError('resolved and canonical validation seeds differ')
print(seed)
PY
  )
  config=${metadata[0]}
  seed=${metadata[1]}

  if [[ -f "$audit_json" ]]; then
    count=$(find "$output_dir" -maxdepth 1 -type f \
      -name 'prediction_*.npz' | wc -l)
    if [[ "$count" -eq 469 ]]; then
      echo "[$(date --iso-8601=seconds)] Existing formal predictions: $run_name"
      continue
    fi
  fi

  echo "[$(date --iso-8601=seconds)] DUMP gpu=$GPU run=$run_name"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" val.py \
    --config "$config" \
    --weights "$checkpoint" \
    --world_size 1 \
    --override "seed=$seed" \
    --dump-predictions "$output_dir" \
    --dump-count 469 \
    --out-json "$audit_json"
done

"$PYTHON" tools/audit_goose_formal_predictions.py \
  --weak-root "$OUTPUT_ROOT"
echo "[$(date --iso-8601=seconds)] FORMAL WEAK PREDICTIONS COMPLETE"
