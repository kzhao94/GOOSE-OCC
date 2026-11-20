# Pose-Guided Temporal Occlusion Supervision

## Training contract

The method uses the same target population, image inputs, optimizer schedule,
and target/history supervision as
`configs/goose_temporal_common10_3h_binary_conf.py`. It adds one semantic term
and does not alter the Gaussian representation or renderer.

For history camera `s`, a valid SAM3 pixel with Metric3D z-depth `d_s` is
back-projected with its resized intrinsic matrix `K_s`. The released pose gives
`E_s`, the transform from target-LiDAR coordinates to source-camera
coordinates. The corresponding target-camera point is

```text
p_t = E_t inverse(E_s) d_s K_s^-1 [u_s, v_s, 1]^T .
```

It is projected with `K_t` and compared with target Metric3D depth `d_t` at the
continuous projected coordinate. A source pixel enters the auxiliary domain
when all of the following hold:

- source depth is in the configured depth range;
- SAM3 label is an occupied class `1..7`, is not `255`, and confidence is
  nonzero;
- the transformed point has positive target-camera depth and projects inside
  the target image;
- target Metric3D depth is valid;
- transformed source depth is behind target depth by more than
  `max(0.5 m, 0.05 d_t)`.

The Gaussian field is represented in target-LiDAR coordinates and rendered in
the history camera. Cross-entropy on the selected history pixels therefore
transfers source-visible semantic evidence to candidate 3D locations placed
behind the target pseudo-depth surface. The selected pixels retain the
configured semantic weights. In the selected binary-confidence baseline and
all method-weight configs, nonzero SAM3 confidence supplies a unit
valid-evidence weight, while zero confidence is ignored. The 0--255 magnitude
is not used as a calibrated probability. The ordinary target/history semantic
and Metric3D depth losses remain enabled.

## Weak-supervision boundary

Training consumes only RGB, official target 2D semantics, SAM3 history
semantics/confidence, Metric3D target/history depth, camera intrinsics, and
camera poses. OCC GT, `observation_path`, visible/occluded masks, LiDAR rays,
and OCC-generation `source_stamps_ns` are absent from the training pipeline and
loss interface. Released observation masks remain validation-only.

The method configuration is
`configs/goose_temporal_occlusion_common10_3h.py`. The predeclared seed-0
positive-weight ablation compared `0.25`, `0.5`, and `1.0` by
`O-SSC-mIoU` and selected `0.5` (10.15, versus 9.97 and 9.84). The frozen
zero-weight temporal control scored 10.66 and was not eligible for selecting
a positive method weight. The selected `0.5` configuration is the one used
for the matched three-seed method audit.

The final selected-method mean is `10.24 +/- 0.11` O-SSC-mIoU, compared with
`10.05 +/- 0.89` for the temporal control. Identical-seed deltas are
`-0.51`, `+1.34`, and `-0.25` points (`+0.19 +/- 1.00`), so the evidence does
not support a stable improvement claim. Canonical validation uses each run's
resolved seed and one GPU over exactly 469 common-target validation samples.
