#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate sparseocc

PRIMARY_SERVICE="visibilityocc-goose-experiments.service"

echo "[$(date --iso-8601=seconds)] Waiting for all seed-0 candidates."
while ! python tools/select_goose_final_temporal.py --shell \
        > /tmp/visibilityocc_goose_final_selection 2>/dev/null; do
    sleep 60
done

echo "[$(date --iso-8601=seconds)] Waiting for the seed-0 queue to stop."
while [[ "$(systemctl --user is-active "$PRIMARY_SERVICE" 2>/dev/null || true)" \
        == "active" ]]; do
    sleep 60
done

run_val() {
    local config="$1"
    local weights="$2"
    local output="$3"
    local seed="$4"
    if [[ -f "$output" ]]; then
        echo "[$(date --iso-8601=seconds)] Skipping existing ${output}."
        return
    fi
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 val.py \
        --config "$config" \
        --weights "$weights" \
        --world_size 2 \
        --override "seed=${seed}" \
        --out-json "$output"
    python tools/summarize_goose_experiments.py
}

run_train() {
    local config="$1"
    local run_name="$2"
    local dataset_type="$3"
    local seed="$4"
    local work_dir="work_dir/${dataset_type}/${run_name}"

    if [[ -f "${work_dir}/epoch_24.pth" ]]; then
        echo "[$(date --iso-8601=seconds)] Skipping completed ${run_name}."
    else
        if [[ -e "$work_dir" ]]; then
            echo "Incomplete work directory exists: ${work_dir}" >&2
            exit 1
        fi
        CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
            train.py \
            --config "$config" \
            --run_name "$run_name" \
            --world_size 2 \
            --override "seed=${seed}"
    fi

    local best_checkpoint
    best_checkpoint=$(find "$work_dir" -maxdepth 1 -type f \
        -name 'best_O-SSC-mIoU_epoch_*.pth' -print | sort -V | tail -n 1)
    if [[ -z "$best_checkpoint" || ! -f "$best_checkpoint" ]]; then
        echo "Missing best O-SSC-mIoU checkpoint in ${work_dir}" >&2
        exit 1
    fi
    run_val "$config" "$best_checkpoint" "${work_dir}/val_best.json" "$seed"
}

IFS=$'\t' read -r selected_temporal_config selected_temporal_stem < <(
    python tools/select_goose_final_temporal.py --shell
)
echo "[$(date --iso-8601=seconds)] Selected ${selected_temporal_config}."

for seed in 1 2; do
    run_train configs/goose_mono_common10.py \
        "goose_mono_common10_seed${seed}" GOOSEOccMono "$seed"
    run_train "$selected_temporal_config" \
        "${selected_temporal_stem}_seed${seed}" GOOSETemporalOccMono "$seed"
done

python tools/summarize_goose_experiments.py
echo "[$(date --iso-8601=seconds)] GOOSE final seed repeats completed."
