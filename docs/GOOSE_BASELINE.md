# GOOSE-OCC v1.4 Baselines

VisibilityOcc provides separate single-frame and temporal GOOSE-OCC configs:

```text
configs/goose_mono.py
configs/goose_temporal.py
```

Publication comparisons and history-count ablations use the matched-sample
configs documented in `docs/GOOSE_EXPERIMENT_PLAN.md`; the two configs above
remain the backward-compatible baseline entry points.

The default dataset root is:

```text
/path/to/GOOSE-OCC
```

Set `GOOSE_OCC_ROOT` to move the dataset. All paths stored in the v1.4 infos
are relative to this root and are resolved by the GOOSE datasets.

The occupancy grid is `[200, 128, 32]` at 0.25 m resolution over
`[0, -16, -4, 50, 16, 4]`. Model output has eight channels: free ID 0 and
occupied semantic IDs 1 through 7. Semantic ID 255 remains ignored.

## Supervision

Single-frame training uses one `CAM_WINDSHIELD_VIS` target image, the official
target 2D semantic label, and the released target Metric3D depth. Metric3D
uint16 depth is decoded with `depth_meters = png / 256`. LiDAR is not used as
training depth supervision.

The temporal config defaults to the target plus the three nearest real history
frames. It joins temporal and base infos by sequence, frame index, and target
sensor timestamp. History frames use SAM3 pseudo semantics,
confidence-weighted cross entropy, and Metric3D depth. The target-coordinate
Gaussian field is projected into every history camera with:

```text
T_history_camera_target_lidar =
    inverse(T_global_camera_history) @ T_global_lidar_target
```

Change `num_history` near the top of `configs/goose_temporal.py` for the
1/3/5/10-history ablation. Samples without enough real histories are filtered;
no image or label is duplicated for padding. With three histories, train/val
contain 2,812/482 targets.

For a command-line override, update the dataset and transformer together. A
five-history run uses:

```bash
python train.py --config configs/goose_temporal.py --run_name goose_temporal_5h \
  --override \
  data.train.num_history=5 data.val.num_history=5 data.test.num_history=5 \
  model.pts_bbox_head.transformer.num_frames=6
```

## Evaluation

Validation reports historical full-grid SC/SSC metrics and optional geometric
FOV metrics. Full-grid metrics retain the old convention that unstored sparse
OCC voxels are free.

GOOSE-OCC v1.4 observation files provide the valid domains for V/O metrics:

```text
V-valid = visible_known_free_mask | visible_occupied_mask
O-valid = occluded_known_free_mask | occluded_observed_mask
```

The evaluator reports V-SC-IoU, V-SSC-mIoU, O-SC-IoU, and O-SSC-mIoU overall
and in 0-10 m, 10-25 m, and 25-50 m horizontal radial bands. `fov_mask.npy` is
geometric and is not used as visibility or occlusion ground truth.

## Commands

Single-frame smoke test:

```bash
conda run -n sparseocc python tools/smoke_test_goose.py \
  --config configs/goose_mono.py \
  --train-iters 1 \
  --val-samples 1
```

Temporal smoke test:

```bash
conda run -n sparseocc python tools/smoke_test_goose.py \
  --config configs/goose_temporal.py \
  --train-iters 1 \
  --val-samples 1
```

Metric contract test:

```bash
conda run -n sparseocc python tools/test_goose_visibility_metrics.py
```

Train:

```bash
conda activate sparseocc
python train.py --config configs/goose_mono.py --run_name goose_single_frame_v14
python train.py --config configs/goose_temporal.py --run_name goose_temporal_3h
```

Two-GPU training uses one process per GPU and keeps the same configured global
batch/gradient-accumulation protocol:

```bash
conda activate sparseocc
export PATH="$CONDA_PREFIX/bin:$PATH"
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --config configs/goose_mono.py \
  --run_name goose_single_frame_v14_2gpu --world_size 2
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train.py --config configs/goose_temporal.py \
  --run_name goose_temporal_3h_2gpu --world_size 2
```

Validate:

```bash
python val.py \
  --config configs/goose_mono.py \
  --weights work_dir/GOOSEOccMono/goose_single_frame_v14/epoch_24.pth \
  --out-json work_dir/GOOSEOccMono/goose_single_frame_v14/val_metrics.json
```

Two-GPU validation:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  val.py --config configs/goose_mono.py \
  --weights work_dir/GOOSEOccMono/goose_single_frame_v14_2gpu/epoch_24.pth \
  --world_size 2 \
  --out-json work_dir/GOOSEOccMono/goose_single_frame_v14_2gpu/val_metrics.json
```

Use the same command with `configs/goose_temporal.py` and its matching
checkpoint for temporal validation. For prediction dumps, add
`--dump-predictions <directory> --dump-count 4`; dumps use shape
`[200,128,32]` and axis order `x,y,z`.

## Weak-Supervision Boundary

Training pipelines collect only RGB, 2D semantic supervision, confidence,
Metric3D depth, and camera geometry. They do not expose `occ_path` or
`observation_path`. OCC GT and LiDAR-derived observation masks are opened only
by validation pipelines and `dataset.evaluate()`.

The existing four-epoch single-frame checkpoint still loads strictly because
the model parameter schema is unchanged. Continuing it now would change the
depth protocol from sparse LiDAR projection to Metric3D, so use a new run name
rather than presenting it as the same training run.

The bounded smoke-test metrics use an effectively untrained model and are not
publication baseline results.
