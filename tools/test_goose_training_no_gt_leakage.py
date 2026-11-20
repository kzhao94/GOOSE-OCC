#!/usr/bin/env python3
"""Check that GOOSE weak-supervision samples do not carry evaluation GT."""

import sys
from collections.abc import Mapping
from pathlib import Path

from mmcv import Config
from mmdet3d.datasets import build_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loaders  # noqa: F401,E402


CONFIGS = (
    'configs/goose_mono_common10.py',
    'configs/goose_temporal_common10_3h.py',
    'configs/goose_temporal_occlusion_common10_3h_weight025.py',
    'configs/goose_temporal_occlusion_common10_3h.py',
    'configs/goose_temporal_occlusion_common10_3h_weight100.py',
)
FORBIDDEN = {
    'occ_path',
    'observation_path',
    'fov_mask_path',
    'source_stamps_ns',
    'source_distances_m',
    'sweeps',
    'voxel_semantics',
}


def forbidden_paths(value, prefix='sample'):
    if hasattr(value, 'data') and value.__class__.__name__ == 'DataContainer':
        return forbidden_paths(value.data, prefix + '.data')
    if isinstance(value, Mapping):
        paths = []
        for key, child in value.items():
            child_path = '{}.{}'.format(prefix, key)
            if key in FORBIDDEN:
                paths.append(child_path)
            paths.extend(forbidden_paths(child, child_path))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, child in enumerate(value):
            paths.extend(forbidden_paths(
                child, '{}[{}]'.format(prefix, index)))
        return paths
    return []


def main():
    for config_path in CONFIGS:
        cfg = Config.fromfile(config_path)
        dataset = build_dataset(cfg.data.train)
        # ``get_data_info`` is intentionally used here instead of executing
        # the image/depth pipeline: it is deterministic and avoids allocating
        # a full Gaussian model while training jobs are running.
        info = dataset.get_data_info(0)
        leaked = sorted(FORBIDDEN.intersection(info.keys()))
        assert not leaked, '{} leaked {}'.format(config_path, leaked)
        pipeline_types = [transform['type'] for transform in
                          cfg.data.train.pipeline]
        assert 'LoadGOOSE2DSupervision' in pipeline_types
        assert 'LoadGOOSEOccGT' not in pipeline_types
        sample = dataset[0]
        assert sample is not None, '{} produced no train sample'.format(
            config_path)
        leaked_paths = forbidden_paths(sample)
        assert not leaked_paths, '{} leaked {}'.format(
            config_path, leaked_paths)
        for required in ('img', 'gt_semantic', 'gt_semantic_weight',
                         'gt_depth', 'img_metas'):
            assert required in sample, '{} missing {}'.format(
                config_path, required)
        if 'Temporal' in cfg.data.train.type:
            expected_views = 1 + int(cfg.data.train.num_history)
            assert len(info['img_filename']) == expected_views
            assert info['frame_is_history'] == [False] + [True] * int(
                cfg.data.train.num_history)
        print('{}: no evaluation GT leakage'.format(config_path))


if __name__ == '__main__':
    main()
