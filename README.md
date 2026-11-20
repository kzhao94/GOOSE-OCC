# GOOSE-OCC

This repository provides the official source release for the GOOSE-OCC
dataset, benchmark, and monocular semantic occupancy experiments. The
implementation, internally named VisibilityOcc, contains the single-frame
baseline, causal temporal baseline, pose-guided temporal occlusion supervision,
and visible/occluded evaluation used in the accompanying manuscript.

GOOSE-OCC data are distributed separately under CC BY-SA 4.0. No dataset,
checkpoint, prediction, pretrained weight, or observation mask is bundled in
this repository.

## Paper and dataset

- Manuscript: *GOOSE-OCC: A Route-Split Dataset and Visibility-Aware Evaluation
  Protocol for Monocular Off-Road Occupancy Prediction*.
- Dataset: *GOOSE-OCC: Semantic Occupancy and Observation-State Annotations
  for Monocular Off-Road Perception*, content release v1.4.0.
- Public archival record: [Science Data Bank](https://doi.org/10.57760/sciencedb.43855),
  published on 23 July 2026. The record contains 29 files totalling 80.94 GB;
  its archive version is V1, while the dataset content release is v1.4.0.
- Data licence: CC BY-SA 4.0. Code licence: Apache-2.0.

## Research boundary

- Weak training uses target official 2D semantics, history SAM3 pseudo
  semantics/confidence, Metric3D pseudo-depth, and camera geometry.
- OCC labels, LiDAR, observation masks, fabricated sweeps, and occupancy-fusion
  timestamps do not enter weak training losses.
- Observation masks are used only for validation, model selection, and
  evaluation.
- WildOcc and ORAD-3D native metrics are not ranked against GOOSE-OCC metrics,
  and no GOOSE V/O masks are inferred for those datasets.

## Environment

Create the core environment, then install the OpenMMLab binaries compatible
with PyTorch 2.0 and CUDA 11.8 as described in
`docs/GOOSE_REPRODUCIBILITY_ENVIRONMENT.md`:

```bash
conda env create -f environment/goose_sparseocc_core.yml
conda activate sparseocc
pip install openmim==0.3.9
mim install mmcv-full==1.6.0
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST=8.6
(cd models/csrc && python setup.py build_ext --inplace)
pip install -v ./models/voxelizer
```

Set the independently downloaded dataset path and enable scoped imports:

```bash
export GOOSE_OCC_ROOT=/path/to/GOOSE-OCC
export VISIBILITYOCC_GOOSE_ONLY=1
```

The ResNet-50 nuImages initialization referenced by the configs is a separate
third-party download and must be placed at the configured `pretrain/` path or
replaced through a config override.

## Core commands

Single-frame training:

```bash
python train.py --config configs/goose_mono.py --run_name goose_single_frame
```

Causal target-plus-three-history training:

```bash
python train.py --config configs/goose_temporal_common10_3h.py   --run_name goose_temporal_3h
```

Single-GPU validation avoids duplicating a sample in the 469-target validation
population:

```bash
CUDA_VISIBLE_DEVICES=0 python val.py   --config configs/goose_temporal_common10_3h.py   --weights /path/to/checkpoint.pth
```

Run the data-dependent smoke test after GOOSE-OCC is mounted:

```bash
python tools/smoke_test_goose.py --config configs/goose_mono.py
```

See `docs/GOOSE_BASELINE.md`, `docs/GOOSE_OCCLUSION_SUPERVISION.md`, and
`docs/GOOSE_REPRODUCIBILITY_CHECKLIST.md` for the data contract and audits.

## Repository scope

This is a source-only reproducibility repository. Large files remain outside
Git:

- GOOSE-OCC dataset and observation masks;
- pretrained weights and model checkpoints;
- formal validation predictions;
- experiment work directories and logs;
- manuscript source and submission packages.

Model and prediction records will be linked after their archival release is
complete.

## Citation

Please cite the [GOOSE-OCC archival dataset record](https://doi.org/10.57760/sciencedb.43855)
and the original GOOSE paper. The dataset citation is:

> Kai Zhao and Youchun Xu. 2026. *GOOSE-OCC: Semantic Occupancy and
> Observation-State Annotations for Monocular Off-Road Perception*. Version 1.
> Science Data Bank. https://doi.org/10.57760/sciencedb.43855.

The accompanying manuscript citation will be added after publication.
Machine-readable software citation metadata are provided in `CITATION.cff`.
