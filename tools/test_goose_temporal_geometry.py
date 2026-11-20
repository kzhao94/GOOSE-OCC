#!/usr/bin/env python3
"""Audit GOOSE temporal joins, causality, and target-to-history poses."""

import argparse
import sys
from pathlib import Path

import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import loaders  # noqa: E402,F401 - registry side effects


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--configs', nargs='+',
        default=[
            'configs/goose_temporal_fair_3h.py',
            'configs/goose_temporal_common10_10h.py',
        ],
    )
    return parser.parse_args()


def padded_intrinsic(intrinsic):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    return matrix


def audit_dataset(dataset):
    max_pose_error = 0.0
    max_projection_error = 0.0
    min_lag_ms = float('inf')
    max_lag_ms = 0.0
    for index, base_info in enumerate(dataset.data_infos):
        runtime = dataset.get_data_info(index)
        temporal = dataset.temporal_by_key[dataset._base_key(base_info)]
        histories = temporal['histories'][:dataset.num_history]
        target_image = dataset.resolve_path(
            base_info['cams']['CAM_WINDSHIELD_VIS']['data_path'])
        target_semantic = dataset.resolve_path(base_info['seg_label_path'])
        target_lidar2cam = np.asarray(
            base_info['cams']['CAM_WINDSHIELD_VIS']['lidar2cam'],
            dtype=np.float64).reshape(4, 4)
        target_global_lidar = np.asarray(
            base_info['T_global_lidar'], dtype=np.float64).reshape(4, 4)
        intrinsic = np.asarray(
            temporal['camera_K'], dtype=np.float64).reshape(3, 3)
        intrinsic_4x4 = padded_intrinsic(intrinsic)

        assert runtime['img_filename'][0] == target_image
        assert np.allclose(runtime['ego2cam'][0], target_lidar2cam)
        assert len(runtime['img_filename']) == dataset.num_history + 1
        assert runtime['history_ranks'][0] == 0
        assert 'observation_path' in runtime if dataset.test_mode else (
            'observation_path' not in runtime)

        previous_lag_ms = -1.0
        for frame_index, history in enumerate(histories, start=1):
            expected_pose = (
                np.linalg.inv(np.asarray(
                    history['T_global_camera'], dtype=np.float64).reshape(4, 4))
                @ target_global_lidar
            )
            actual_pose = np.asarray(
                runtime['ego2cam'][frame_index], dtype=np.float64)
            pose_error = float(np.abs(actual_pose - expected_pose).max())
            projection_error = float(np.abs(
                np.asarray(runtime['lidar2img'][frame_index], dtype=np.float64)
                - intrinsic_4x4 @ expected_pose
            ).max())
            max_pose_error = max(max_pose_error, pose_error)
            max_projection_error = max(
                max_projection_error, projection_error)

            history_image = dataset.resolve_path(history['image_path'])
            history_semantic = dataset.resolve_path(
                history['pseudo_label_path'])
            assert history_image != target_image
            assert history_semantic != target_semantic
            assert runtime['img_filename'][frame_index] == history_image
            assert runtime['history_ranks'][frame_index] == history['rank']

            lag_ms = (
                int(temporal['target_camera_stamp_ns'])
                - int(history['camera_stamp_ns'])
            ) / 1e6
            assert lag_ms > previous_lag_ms
            previous_lag_ms = lag_ms
            min_lag_ms = min(min_lag_ms, lag_ms)
            max_lag_ms = max(max_lag_ms, lag_ms)

    assert max_pose_error < 1e-5, max_pose_error
    # Loader matrices are float32; multiplying camera intrinsics by the
    # longest-history translations can accumulate several 1e-3 absolute error.
    assert max_projection_error < 1e-2, max_projection_error
    return max_pose_error, max_projection_error, min_lag_ms, max_lag_ms


def main():
    args = parse_args()
    for config_path in args.configs:
        cfg = Config.fromfile(config_path)
        for split in ('train', 'val'):
            dataset = build_dataset(cfg.data[split])
            result = audit_dataset(dataset)
            print(
                '{} {}: samples={}, pose_error={:.3e}, '
                'projection_error={:.3e}, lag_ms={:.3f}-{:.3f}'.format(
                    config_path, split, len(dataset), *result))


if __name__ == '__main__':
    main()
