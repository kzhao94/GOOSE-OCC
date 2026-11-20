#!/usr/bin/env python3
"""Verify target/history semantic and depth supervision by backpropagation."""

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    head_module = importlib.import_module('models.sparse_gaussian_head')
    loss_module = importlib.import_module('models.loss')

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

    original_rasterizer = head_module.rasterize_gaussians
    head_module.rasterize_gaussians = lambda gaussian_pred, img_metas: rendered
    try:
        semantic = torch.tensor([[[[1, 1], [1, 255]],
                                  [[2, 2], [255, 255]]]])
        semantic_weight = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]],
                                         [[1.0, 0.5], [0.0, 0.0]]]])
        depth = torch.tensor([[[[2.0, 2.0], [2.0, 9.0]],
                               [[4.0, 4.0], [9.0, 9.0]]]])
        losses = {}
        head.loss_single(
            None,
            losses,
            [semantic[:, frame] for frame in range(frames)],
            [depth[:, frame] for frame in range(frames)],
            [{}],
            [semantic_weight[:, frame] for frame in range(frames)],
        )
    finally:
        head_module.rasterize_gaussians = original_rasterizer

    assert torch.allclose(
        losses['loss_sem'],
        losses['semantic_raw_current']
        + 0.75 * losses['semantic_raw_history'])
    assert torch.allclose(losses['depth_raw_current'], torch.tensor(1.0))
    assert torch.allclose(losses['depth_raw_history'], torch.tensor(3.0))
    assert torch.allclose(losses['loss_depth'], torch.tensor(3.25))
    assert losses['semantic_raw_history'] > 0
    assert losses['history_confidence_mean'] > 0

    (losses['loss_sem'] + losses['loss_depth']).backward()
    gradient = rendered.grad
    assert gradient is not None
    assert gradient[0, 0, :, 0, 0].abs().sum() > 0
    assert gradient[0, 1, :, 0, 0].abs().sum() > 0
    assert gradient[0, 1, :classes, 0, 1].abs().sum() > 0
    assert gradient[0, 0, :, 1, 1].abs().sum() == 0
    assert gradient[0, 1, :, 1, 0].abs().sum() == 0
    assert gradient[0, 1, :, 1, 1].abs().sum() == 0
    print('target/history semantic and depth gradients passed')
    print('depth current/history/final:',
          float(losses['depth_raw_current']),
          float(losses['depth_raw_history']),
          float(losses['loss_depth']))


if __name__ == '__main__':
    main()
