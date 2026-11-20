#!/usr/bin/env python3
"""Check GOOSE class-count, ignore-label, and sparse OCC label contracts."""

import sys
from pathlib import Path

import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loaders  # noqa: E402,F401 - registry side effects
from loaders.goose_occ_mono import load_goose_occ_semantics  # noqa: E402
from val import canonical_prediction_array  # noqa: E402


def main():
    cfg = Config.fromfile(str(ROOT / 'configs/goose_temporal_occlusion_common10_3h.py'))
    dataset = build_dataset(cfg.data.train)
    sample = dataset[0]
    for semantic, weight in zip(sample['gt_semantic'],
                                sample['gt_semantic_weight']):
        assert np.all(((semantic >= 1) & (semantic <= 7)) | (semantic == 255))
        assert np.all(semantic[weight == 0] == 255)
        assert np.all((weight >= 0) & (weight <= 1))

    head = cfg.model.pts_bbox_head
    assert len(head.class_names) == 8
    assert head.class_names[0] == 'free'
    assert cfg.model.pts_bbox_head.loss_cfgs.class_weights[0] == 0.0
    assert cfg.model.pts_bbox_head.loss_cfgs.num_classes == 8
    assert cfg.model.pts_bbox_head.loss_cfgs.get('ignore_index', 255) == 255

    info = dataset.data_infos[0]
    dense = load_goose_occ_semantics(
        dataset.resolve_path(info['occ_path']), dataset.occ_size)
    assert dense.shape == (200, 128, 32)
    assert np.all((dense >= 0) & (dense <= 7))
    old_dump = dense[None]
    assert canonical_prediction_array(
        old_dump, dataset.occ_size).shape == dataset.occ_size
    assert canonical_prediction_array(
        dense, dataset.occ_size).shape == dataset.occ_size
    try:
        canonical_prediction_array(
            dense.transpose(1, 0, 2), dataset.occ_size)
    except ValueError:
        pass
    else:
        raise AssertionError('transposed prediction shape was accepted')
    print('GOOSE label contract passed: 8 classes, ignore=255, OCC IDs 0..7')


if __name__ == '__main__':
    main()
