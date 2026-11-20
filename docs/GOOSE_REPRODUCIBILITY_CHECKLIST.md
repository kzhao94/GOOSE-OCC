# GOOSE-OCC Reproducibility Checklist

Requirement-to-evidence completion status is tracked in
`docs/GOOSE_GOAL_AUDIT.md`.

## Frozen inputs

- [x] Record the audited training stack and core environment specification in
  `docs/GOOSE_REPRODUCIBILITY_ENVIRONMENT.md` and
  `environment/goose_sparseocc_core.yml`.
- [ ] Set `GOOSE_OCC_ROOT=/path/to/GOOSE-OCC`.
- [ ] Activate the `sparseocc` environment and prepend its `bin` directory to
  `PATH` so the CUDA-extension build can find `ninja`:
  `conda activate sparseocc && export PATH="$CONDA_PREFIX/bin:$PATH"`.
- [ ] Verify `infos/goose_infos_{train,val}.pkl` and
  `temporal/temporal_infos_{train,val}.pkl` are present.
- [ ] Verify release `observation/observation_protocol.json` and
  `observation/observation_validation.json`.
- [ ] After all training jobs stop, verify the frozen release payload with
  `bash tools/verify_goose_dataset_checksums.sh`. It runs
  `sha256sum -c checksums.sha256` and writes a machine-readable JSON plus the
  full log. The result freeze manifest hashes the checksum list and release
  metadata but deliberately does not claim that this full 88 GB payload audit
  has already run.
- [ ] Do not regenerate or modify the frozen dataset during training.

## Protocol checks

- [x] `tools/audit_goose_formal_predictions.py` enforces 469 predictions per
  formal run, `[200,128,32]`, `uint8`, axes `x,y,z`, IDs `0..7`, canonical
  weak sample indices/tokens, and per-file SHA-256 records. Final execution is
  pending creation of all 13 weak and one upper-bound prediction sets.
- [x] `tools/test_goose_temporal_geometry.py` checks real causal histories,
  pose direction, relative paths, and train/val filtering.
- [x] `tools/test_goose_visibility_reference.py` matches SurroundOcc on all
  490 validation targets.
- [x] `tools/test_goose_temporal_occlusion.py` checks pose-guided mask logic,
  margin, ignore/confidence filtering, and gradients.
- [x] `tools/test_goose_temporal_loss.py` checks target/history semantic and
  depth gradients.
- [x] Re-run `tools/test_goose_experiment_protocol.py` after the method config
  additions; common-target counts and update budgets remain identical.
- [x] `tools/test_goose_training_no_gt_leakage.py` executes a complete train
  sample pipeline for mono, temporal, and all three method weights, then
  recursively audits collected tensors and metadata for evaluation-only
  fields.
- [x] `tools/test_goose_experiment_summary.py` checks complete positive-weight
  selection, the seed-0 tie-break, identical-seed deltas, sample standard
  deviation, and preference for canonical `val_best.json` records.
- [x] `tools/test_goose_manuscript_citations.py` checks that every manuscript
  citation key exists exactly once in the independent bibliography and that
  corrected SparseOcc attribution plus validation/model-selection boundaries
  remain present.
- [x] `tools/test_goose_label_contract.py` checks eight output classes,
  `ignore=255`, confidence-zero masking, and dense OCC IDs `0..7`.
- [x] `tools/test_render_goose_manuscript_results.py` checks that final paper
  tables reject incomplete five-seed groups, mismatched claim gates, and a
  missing 469-target fully supervised upper bound.

Run the current protocol suite with the environment-local extension tools on
`PATH`:

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
  tools/test_goose_training_no_gt_leakage.py \
  tools/test_render_goose_manuscript_results.py; do
  python "$test"
done
```

The 2026-07-20 protocol audit passed the original ten scripts; the manuscript
table renderer contract was subsequently added and passes independently. The
combined final audit now runs eleven scripts. It verifies common-target
train/validation counts `2726/469`, `32712` micro-iterations and `8178`
optimizer updates for every method weight; reference equality covered all 490
released validation targets. The summary regression test additionally fixes
seed-0 weight selection, identical-seed pairing, sample-standard-deviation,
and `val_best.json` deduplication behavior.

After every formal run, prediction set and archival identifier is complete,
execute the fail-closed final sequence with:

```bash
bash tools/run_goose_final_submission_audit.sh
```

This command reruns the eleven scripts, final summaries/gate/figures, the
machine-generated manuscript table renderer, prediction audit, full dataset
checksums, Source Data packaging, hash freeze, repository diff check and
submission-readiness gate. It exits nonzero on any pending contract, including
unresolved availability placeholders.

## Training boundary

- [x] Target official 2D semantics and Metric3D pseudo depth are used.
- [x] History SAM3 semantics, confidence, and Metric3D pseudo depth are used.
- [x] History depth loss remains enabled.
- [x] OCC GT, observation masks, LiDAR rays, and `source_stamps_ns` are absent
  from weak-supervision loss.
- [x] Unknown-as-free limitation is documented for standard baseline metrics.

## Experiments

- [x] Completed mono/1/3/5/10-history fair baseline and three-seed summary.
- [x] Complete occlusion weight `0.25/0.5/1.0` seed-0 ablation.
- [x] Select weight by predeclared O-SSC-mIoU rule (`lambda_occ=0.5`).
- [x] Complete selected method seeds 1/2 and report mean +/- sample std.
- [x] Re-run `tools/summarize_goose_experiments.py` and archive resolved
  configs, checkpoints, logs, and validation JSONs.
- [ ] Complete frozen temporal-control and selected-method seeds 3/4, then
  regenerate five-seed statistics without retuning `lambda_occ=0.5`.
- [ ] Train and evaluate the strict `2726/469` fully supervised SurroundOcc
  upper bound; label it separately from all weakly supervised methods.

The five-seed extension is launched by
`tools/run_goose_five_seed_extension.sh`: GPU 0 runs seed 3 and GPU 1 runs seed
4, with the temporal control followed by the frozen pose-guided method on each
card. Canonical common-target validation uses one GPU per run with the run's
resolved seed and exactly 469 targets. This avoids DDP padding of the odd-sized
validation set and preserves seed-dependent lazy Gaussian initialization.

For the formal run, `tools/wait_goose_and_run_parallel_upper_bound.sh` starts
`tools/run_surroundocc_common10_train_only.sh` on GPU 0 as soon as the seed-3
weak branch has completed and been validated, while seed 4 continues on GPU 1.
After both branches finish, `tools/run_surroundocc_common10_upper_bound.sh`
performs the strict evaluation and the 13 weak plus one upper-bound prediction
audit. The upper bound remains explicitly OCC-GT supervised and is not part of
weak-method selection.

## Figures and manuscript

- [x] Dataset-construction and train/evaluation-boundary figure generated from
  generator/release validation JSONs with editable SVG/PDF and source JSON.
- [x] Observation-domain panel generated from a released validation mask.
- [x] Pose-guided method figure generated with a real training pair and the
  exact candidate-geometry function; OCC/observation products are not read.
- [x] Baseline distance and three-seed panels generated from CSV artifacts.
- [x] Regenerate method weight, distance, and multi-seed panels after method
  validation.
- [x] Generate final qualitative target/prediction/disagreement panel.
- [x] GOOSE-specific manuscript draft includes Introduction, Related Work,
  Dataset/Observation Protocol, Method, Experiments, Limitations, and
  Conclusion sections.
- [x] Replace pending method placeholders only after the final audit.
- [ ] Regenerate five-seed panels, upper-bound table entries, and manuscript
  claims after all pending formal results are frozen.
- [ ] Build `paper/goose_occ_manuscript/source_data_v1` with
  `tools/build_goose_source_data_package.py`; it must reject partial seeds and
  require the formal 469-target upper bound, claim gate, 14-run prediction
  audit, all figure-source metadata, and the final machine-generated tables
  from `tools/render_goose_manuscript_results.py`.
