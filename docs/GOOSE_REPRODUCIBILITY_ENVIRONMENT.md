# GOOSE-OCC Reproducibility Environment

This document records the environment used by the formal VisibilityOcc weakly
supervised experiments and the subsequent isolated-prefix release rebuild.
The release environment was created from the core YAML, its CUDA extensions
were compiled from the reviewer source package, and one-iteration mono and
target-plus-three-history train/validation smoke tests passed. These reduced
query smoke runs verify execution paths, not manuscript performance claims.

## Audited runtime

| Component | Version |
| --- | --- |
| OS kernel | Linux 6.8.0-124-generic, x86_64 |
| Python | 3.8.20 |
| PyTorch | 2.0.0 |
| torchvision | 0.15.0 |
| PyTorch CUDA runtime | 11.8 |
| CUDA toolkit / nvcc | 11.8 / V11.8.89 |
| cuDNN | 8.7.0 |
| NVIDIA driver | 595.71.05 |
| GPU | 2 x NVIDIA GeForce RTX 3090, 24 GB |
| GCC used by Python build | 11.2.0 |
| System GCC | Ubuntu GCC 10.5.0 |
| mmcv-full | 1.6.0 |
| mmdet | 2.28.2 |
| mmdet3d | 1.0.0rc6 |
| mmsegmentation | 0.30.0 |
| NumPy | 1.23.5 |
| SciPy | 1.10.1 |
| OpenCV | 4.12.0 |
| Pillow | 10.4.0 |
| scikit-learn | 1.3.2 |
| matplotlib | 3.5.3 |
| segmentation-models-pytorch | 0.3.3 |
| Weights & Biases | 0.22.3 |
| spconv-cu118 | 2.3.6 |
| cumm-cu118 | 0.4.11 |
| gsplat | 1.5.3 |
| ninja | 1.13.0 |

The core environment specification is
`environment/goose_sparseocc_core.yml`. `mmcv-full==1.6.0` is intentionally
not listed as a normal PyPI dependency in that file because its CUDA/PyTorch
binary must be installed from a compatible OpenMMLab wheel index or built
against the active PyTorch/CUDA environment. The formal environment used
`mmcv-full 1.6.0` at runtime.

An isolated-prefix rebuild found that `segmentation-models-pytorch` and
`wandb` were imported by the training stack but absent from an earlier core
YAML. Both are now pinned, together with the build-compatible setuptools
version, so model import and the command-line entry points do not depend on
packages inherited from another environment. `openmim==0.3.9` remains an
installation helper for the platform-specific MMCV wheel rather than a model
runtime dependency.

## Runtime requirements

The CUDA extensions under `models/csrc` and `models/voxelizer` must be
available to the active environment. The environment-local `bin` directory
must be on `PATH`; otherwise PyTorch extension discovery can fail to find
`ninja` even when it is installed:

```bash
conda activate sparseocc
export PATH="$CONDA_PREFIX/bin:$PATH"
export GOOSE_OCC_ROOT=/path/to/GOOSE-OCC
pip install openmim==0.3.9
mim install mmcv-full==1.6.0
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST=8.6
(cd models/csrc && python setup.py build_ext --inplace)
pip install -v ./models/voxelizer
```

The local path above is only the experimental-machine mount point. The public
release instructions must replace it with a user-selected dataset path and a
persistent GOOSE-OCC repository identifier.

## Formal execution contracts

- Weakly supervised training uses one process per run. The two GPUs may run
  different seeds concurrently, but a seed is never split across GPUs.
- Canonical validation uses one GPU and exactly 469 common targets. Distributed
  validation is not used because padding an odd-sized validation split would
  duplicate a sample.
- Every temporal-control and selected-method run uses 2,726 train targets, 24
  epochs, 32,712 micro-iterations and 8,178 optimizer updates.
- The formal method weight is frozen at `lambda_occ=0.5`; seed 3/4 do not
  trigger reselection or retuning.
- Target and history Metric3D pseudo-depth enter the weak training loss.
  Target official semantics and history SAM3 pseudo-semantics/confidence enter
  the semantic loss.
- OCC GT, LiDAR, observation masks and fusion source timestamps do not enter
  weak training. Observation masks are used for validation/evaluation and
  validation-based checkpoint/model selection only.

## Verification commands

Run the protocol suite from the repository root:

```bash
export PATH=/path/to/miniconda3/envs/sparseocc/bin:$PATH
for test in \
  tools/test_goose_experiment_protocol.py \
  tools/test_goose_temporal_geometry.py \
  tools/test_goose_temporal_loss.py \
  tools/test_goose_temporal_occlusion.py \
  tools/test_goose_visibility_metrics.py \
  tools/test_goose_visibility_reference.py \
  tools/test_goose_experiment_summary.py \
  tools/test_goose_manuscript_citations.py \
  tools/test_goose_label_contract.py \
  tools/test_goose_training_no_gt_leakage.py; do
  python "$test"
done
```

Generate the machine-readable result tables and figures only after all formal
seed and upper-bound records exist:

```bash
python \
  tools/summarize_goose_experiments.py
python \
  tools/plot_goose_paper_figures.py
```

The summarizer requires complete seeds 0--4 for the temporal control and the
selected pose-guided method. It will not emit paired five-seed conclusions
from a partial group. The strict SurroundOcc result is included only when it
declares 469/469 samples and the prediction contract
`uint8 [200,128,32]`, axes `x,y,z`, IDs `0..7`.

## Remaining archival checks

- Retain `goose_sparseocc_clean_conda_explicit.txt`,
  `goose_sparseocc_clean_pip_lock.txt`, and the hash-validated
  `goose_clean_release_smoke.json` beside the portable core YAML.
- Record hashes for the final source commit, configs, resolved configs,
  checkpoints and prediction files.
- Replace machine-specific absolute paths in public commands while retaining
  path variables and dataset-relative infos contracts.
- Test the dataset, code and model reviewer links outside the owner account.
