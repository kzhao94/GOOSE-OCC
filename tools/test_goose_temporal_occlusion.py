#!/usr/bin/env python3
"""Unit tests for pose-guided temporal occlusion supervision."""

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_geometry(geometry):
    batch, height, width = 1, 3, 4
    source_depth = torch.full((batch, height, width), 4.0)
    target_depth = torch.full((batch, height, width), 2.0)
    semantic = torch.ones(batch, height, width, dtype=torch.long)
    confidence = torch.ones(batch, height, width)
    intrinsic = torch.eye(3).unsqueeze(0)
    pose = torch.eye(4).unsqueeze(0)

    mask = geometry.source_visible_to_target_occluded(
        source_depth, semantic, confidence, target_depth,
        intrinsic, intrinsic, pose,
        occlusion_abs_margin=0.5, occlusion_rel_margin=0.0)
    assert mask.all(), mask

    not_behind = geometry.source_visible_to_target_occluded(
        torch.full_like(source_depth, 2.3), semantic, confidence, target_depth,
        intrinsic, intrinsic, pose,
        occlusion_abs_margin=0.5, occlusion_rel_margin=0.0)
    assert not not_behind.any()

    semantic[:, 0, 0] = 255
    confidence[:, 0, 1] = 0
    filtered = geometry.source_visible_to_target_occluded(
        source_depth, semantic, confidence, target_depth,
        intrinsic, intrinsic, pose,
        occlusion_abs_margin=0.5, occlusion_rel_margin=0.0)
    assert not filtered[0, 0, 0]
    assert not filtered[0, 0, 1]
    assert filtered.sum() == height * width - 2

    shifted = pose.clone()
    shifted[:, 0, 3] = (width + 1) * 4
    outside = geometry.source_visible_to_target_occluded(
        source_depth, torch.ones_like(semantic), torch.ones_like(confidence),
        target_depth, intrinsic, intrinsic, shifted,
        occlusion_abs_margin=0.5, occlusion_rel_margin=0.0)
    assert not outside.any()


def test_head_loss(head_module, loss_module):
    criterion = loss_module.MultiLoss(
        num_classes=8,
        use_uncertainty_weights=False,
        use_sem_lovasz=False,
        use_depth_si=False,
        class_weights=[0.0] + [1.0] * 7,
        depth_min=0.1,
        depth_max=50.0,
    )
    head = head_module.SparseGaussianHead.__new__(
        head_module.SparseGaussianHead)
    nn.Module.__init__(head)
    head.criterion = criterion
    head.history_loss_weight = 0.75
    head.history_semantic_ce_only = True
    head.occlusion_loss_weight = 0.5
    head.occlusion_abs_margin = 0.5
    head.occlusion_rel_margin = 0.0
    head.debug_vis = False
    head.debug_vis_interval = 0
    head._iter_counter = 0

    batch, frames, classes, height, width = 1, 2, 8, 2, 2
    rendered = torch.zeros(
        batch, frames, classes + 1, height, width,
        dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        rendered[:, :, :classes] = torch.linspace(
            -0.4, 0.4, classes).view(1, 1, classes, 1, 1)
        rendered[:, :, classes] = 1.0
    semantic = torch.ones(batch, frames, height, width, dtype=torch.long)
    depth = torch.empty(batch, frames, height, width)
    depth[:, 0] = 2.0
    depth[:, 1] = 4.0
    confidence = torch.ones(batch, frames, height, width)
    identity3 = np.eye(3, dtype=np.float32)
    identity4 = np.eye(4, dtype=np.float32)
    img_metas = [dict(
        cam2img=[identity3, identity3],
        ego2cam=[identity4, identity4],
    )]

    original_rasterizer = head_module.rasterize_gaussians
    head_module.rasterize_gaussians = lambda gaussian_pred, metas: rendered
    try:
        losses = {}
        head.loss_single(
            None, losses,
            [semantic[:, index] for index in range(frames)],
            [depth[:, index] for index in range(frames)],
            img_metas,
            [confidence[:, index] for index in range(frames)],
        )
    finally:
        head_module.rasterize_gaussians = original_rasterizer

    assert losses['semantic_raw_occlusion'] > 0
    assert losses['occlusion_pixels'] == height * width
    expected = (
        losses['semantic_raw_current']
        + 0.75 * losses['semantic_raw_history']
        + 0.5 * losses['semantic_raw_occlusion'])
    assert torch.allclose(losses['loss_sem'], expected)
    (losses['loss_sem'] + losses['loss_depth']).backward()
    assert rendered.grad is not None
    assert rendered.grad[:, 1, :classes].abs().sum() > 0


def test_training_contract():
    config = (REPO_ROOT / 'configs' /
              'goose_temporal_occlusion_common10_3h.py').read_text()
    assert 'occlusion_loss_weight' in config
    forbidden = ('observation_path', 'occ_path', 'source_stamps_ns')
    for token in forbidden:
        assert token not in config


def main():
    geometry = importlib.import_module('models.temporal_occlusion')
    head_module = importlib.import_module('models.sparse_gaussian_head')
    loss_module = importlib.import_module('models.loss')
    test_geometry(geometry)
    test_head_loss(head_module, loss_module)
    test_training_contract()
    print('pose-guided temporal occlusion tests passed')


if __name__ == '__main__':
    main()
