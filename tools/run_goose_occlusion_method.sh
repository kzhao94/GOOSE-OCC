#!/usr/bin/env bash
set -euo pipefail

# Seed-0 method smoke plus training/validation. No source timestamps or
# observation masks are consumed by this configuration.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="configs/goose_temporal_occlusion_common10_3h.py"
CUDA_VISIBLE_DEVICES=0 python tools/smoke_test_goose.py \
  --config "$CONFIG" --train-iters 1 --val-samples 1 --batch-size 1 \
  --load-pretrain
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --config "$CONFIG" --run_name goose_temporal_occlusion_common10_3h_seed0 \
  --world_size 2
