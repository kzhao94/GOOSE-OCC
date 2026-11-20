#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate sparseocc

gpu_processes() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | sed '/^[[:space:]]*$/d'
}

if [[ "${GOOSE_SHARE_GPUS:-0}" == "1" ]]; then
    echo "[$(date --iso-8601=seconds)] Sharing GPUs with existing jobs."
else
    echo "[$(date --iso-8601=seconds)] Waiting for both GPUs to become idle."
    while [[ -n "$(gpu_processes)" ]]; do
        sleep 60
    done
    echo "[$(date --iso-8601=seconds)] GPUs are idle; starting GOOSE experiments."
fi

run_val() {
    local config="$1"
    local weights="$2"
    local output="$3"
    local seed="${4:-0}"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 val.py \
        --config "$config" \
        --weights "$weights" \
        --world_size 2 \
        --override "seed=${seed}" \
        --out-json "$output"
    python tools/summarize_goose_experiments.py
}

run_smoke() {
    local config="$1"
    CUDA_VISIBLE_DEVICES=0 python tools/smoke_test_goose.py \
        --config "$config" \
        --train-iters 1 \
        --val-samples 1 \
        --batch-size 1 \
        --load-pretrain
}

run_train() {
    local config="$1"
    local run_name="$2"
    local dataset_type="$3"
    local seed="${4:-0}"
    local work_dir="work_dir/${dataset_type}/${run_name}"
    if [[ -f "${work_dir}/epoch_24.pth" ]]; then
        echo "[$(date --iso-8601=seconds)] Skipping completed ${run_name}."
    else
        if [[ -e "$work_dir" ]]; then
            echo "Incomplete work directory exists: ${work_dir}" >&2
            exit 1
        fi
        local train_args=(
            --config "$config"
            --run_name "$run_name"
            --world_size 2
        )
        if [[ "$seed" != "0" ]]; then
            train_args+=(--override "seed=${seed}")
        fi
        CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
            train.py "${train_args[@]}"
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

# Fail early on data, loss, pose rendering, or per-GPU memory regressions.
run_smoke configs/goose_mono_fair_3h.py
run_smoke configs/goose_temporal_fair_3h.py

# Diagnostic re-evaluation of already trained, budget-mismatched checkpoints.
run_val \
    configs/goose_mono_matched_3h.py \
    work_dir/GOOSEOccMono/goose_single_frame_v14_2gpu/epoch_22.pth \
    work_dir/GOOSEOccMono/goose_single_frame_v14_2gpu/val_matched_3h_epoch22.json
run_val \
    configs/goose_temporal.py \
    work_dir/GOOSETemporalOccMono/goose_temporal_3h_2gpu/epoch_24.pth \
    work_dir/GOOSETemporalOccMono/goose_temporal_3h_2gpu/val_epoch24.json

# Fair main comparison.
run_train configs/goose_mono_fair_3h.py goose_mono_fair_3h_seed0 GOOSEOccMono
run_train configs/goose_temporal_fair_3h.py goose_temporal_fair_3h_seed0 GOOSETemporalOccMono

# History-count study on one common target population.
run_train configs/goose_mono_common10.py goose_mono_common10_seed0 GOOSEOccMono
run_train configs/goose_temporal_common10_1h.py goose_temporal_common10_1h_seed0 GOOSETemporalOccMono
run_train configs/goose_temporal_common10_3h.py goose_temporal_common10_3h_seed0 GOOSETemporalOccMono
run_smoke configs/goose_temporal_common10_5h.py
run_train configs/goose_temporal_common10_5h.py goose_temporal_common10_5h_seed0 GOOSETemporalOccMono
run_smoke configs/goose_temporal_common10_10h.py
run_train configs/goose_temporal_common10_10h.py goose_temporal_common10_10h_seed0 GOOSETemporalOccMono

# Focused supervision ablations around the default 3-history configuration.
run_train configs/goose_temporal_common10_3h_binary_conf.py goose_temporal_common10_3h_binary_conf_seed0 GOOSETemporalOccMono
run_train configs/goose_temporal_common10_3h_weight025.py goose_temporal_common10_3h_weight025_seed0 GOOSETemporalOccMono
run_train configs/goose_temporal_common10_3h_weight100.py goose_temporal_common10_3h_weight100_seed0 GOOSETemporalOccMono

# Three-seed confirmation on the common target population. The selector uses
# only the predeclared seed-0 candidates and deterministic tie-break rules.
IFS=$'\t' read -r selected_temporal_config selected_temporal_stem < <(
    python tools/select_goose_final_temporal.py --shell
)
echo "[$(date --iso-8601=seconds)] Selected ${selected_temporal_config} for seed repeats."
for seed in 1 2; do
    run_train configs/goose_mono_common10.py \
        "goose_mono_common10_seed${seed}" GOOSEOccMono "$seed"
    run_train "$selected_temporal_config" \
        "${selected_temporal_stem}_seed${seed}" GOOSETemporalOccMono "$seed"
done

echo "[$(date --iso-8601=seconds)] GOOSE fair experiment queue completed."
