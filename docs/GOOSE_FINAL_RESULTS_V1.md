# GOOSE final_results_v1

This is a hash-addressed evidence manifest. It references checkpoints
in place and does not duplicate large model files.

- Snapshot UTC: `2026-07-21T12:54:42.706547+00:00`
- Git commit: `af4d9df83bfd1b82c2b095fca793ff7ba49d5e9f`
- Complete experiment records: `23`
- Pending five-seed extensions: `0`
- Supporting artifacts: `245`
- Formal prediction audit: `complete`

| run | role | seed | status | official | O-SSC-mIoU |
| --- | --- | ---: | --- | --- | ---: |
| goose_mono_common10_seed0 | main_weakly_supervised_mono | 0 | complete | True | 9.99 |
| goose_mono_common10_seed1 | main_weakly_supervised_mono | 1 | complete | True | 9.05 |
| goose_mono_common10_seed2 | main_weakly_supervised_mono | 2 | complete | True | 10.40 |
| goose_mono_fair_3h_seed0 | excluded_non_common_or_diagnostic | 0 | complete | False | 9.12 |
| goose_temporal_common10_10h_seed0 | common_target_ablation | 0 | complete | True | 10.35 |
| goose_temporal_common10_1h_seed0 | common_target_ablation | 0 | complete | True | 8.99 |
| goose_temporal_common10_3h_binary_conf_seed0 | main_temporal_control | 0 | complete | True | 10.66 |
| goose_temporal_common10_3h_binary_conf_seed1 | main_temporal_control | 1 | complete | True | 9.02 |
| goose_temporal_common10_3h_binary_conf_seed2 | main_temporal_control | 2 | complete | True | 10.46 |
| goose_temporal_common10_3h_binary_conf_seed3 | main_temporal_control | 3 | complete | True | 10.62 |
| goose_temporal_common10_3h_binary_conf_seed4 | main_temporal_control | 4 | complete | True | 9.99 |
| goose_temporal_common10_3h_seed0 | common_target_ablation | 0 | complete | True | 10.30 |
| goose_temporal_common10_3h_weight025_seed0 | common_target_ablation | 0 | complete | True | 9.42 |
| goose_temporal_common10_3h_weight100_seed0 | common_target_ablation | 0 | complete | True | 10.58 |
| goose_temporal_common10_5h_seed0 | common_target_ablation | 0 | complete | True | 9.62 |
| goose_temporal_fair_3h_seed0 | excluded_non_common_or_diagnostic | 0 | complete | False | 10.05 |
| goose_temporal_occlusion_common10_3h_seed0 | main_pose_guided_method | 0 | complete | True | 10.15 |
| goose_temporal_occlusion_common10_3h_seed1 | main_pose_guided_method | 1 | complete | True | 10.36 |
| goose_temporal_occlusion_common10_3h_seed2 | main_pose_guided_method | 2 | complete | True | 10.21 |
| goose_temporal_occlusion_common10_3h_seed3 | main_pose_guided_method | 3 | complete | True | 10.20 |
| goose_temporal_occlusion_common10_3h_seed4 | main_pose_guided_method | 4 | complete | True | 10.56 |
| goose_temporal_occlusion_common10_3h_weight025_seed0 | pose_guided_weight_ablation | 0 | complete | True | 9.97 |
| goose_temporal_occlusion_common10_3h_weight100_seed0 | pose_guided_weight_ablation | 0 | complete | True | 9.84 |

## External baselines

| run | role | train/val | epochs | supervision | O-SSC-mIoU | status |
| --- | --- | ---: | ---: | --- | ---: | --- |
| surroundocc_goose_common10 | fully_supervised_occ_upper_bound | 2726/469 | 6 | OCC_GT | 11.48 | complete |

Machine-readable hashes are stored in
`work_dir/final_results_v1/manifest.json`.
