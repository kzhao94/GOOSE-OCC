#!/usr/bin/env python3
"""Run a bounded GOOSE-OCC dataset/train/validation smoke test."""

import argparse
import copy
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer, load_checkpoint
from mmcv.utils.logging import logger_initialized
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import loaders  # noqa: F401 - registry side effects
import models  # noqa: F401 - registry side effects
from hooks import safe_single_gpu_test
from loaders.builder import build_dataloader
from loaders.goose_occ_mono import load_goose_occ_semantics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/goose_mono.py')
    parser.add_argument('--train-iters', type=int, default=1)
    parser.add_argument('--val-samples', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-queries', type=int, default=None,
                        help='Optional reduced query count for a faster smoke test.')
    parser.add_argument('--load-pretrain', action='store_true')
    return parser.parse_args()


def quiet_openmmlab_logging():
    for name in ('root', 'mmcv', 'mmdet3d'):
        logger_initialized[name] = logging.Logger(name, logging.WARNING)


def describe_dataset(train_dataset, val_dataset,
                     history_confidence_mode='weighted'):
    train_info = train_dataset.data_infos[0]
    val_info = val_dataset.data_infos[0]
    train_sample = train_dataset[0]
    val_sample = val_dataset[0]
    train_meta = train_sample['img_metas'].data
    dense = load_goose_occ_semantics(
        val_dataset.resolve_path(val_info['occ_path']), val_dataset.occ_size)
    sparse = np.load(
        val_dataset.resolve_path(val_info['occ_path']), allow_pickle=False)

    print('train/val:', len(train_dataset), len(val_dataset))
    print('camera:', list(train_info['cams']))
    print('image:', tuple(train_sample['img'].data.shape))
    print('lidar2img/cam2img/ego2cam:',
          np.asarray(train_meta['lidar2img'][0]).shape,
          np.asarray(train_meta['cam2img'][0]).shape,
          np.asarray(train_meta['ego2cam'][0]).shape)
    print('sweeps:', train_info.get('sweeps', []))
    print('sparse OCC:', sparse.shape, sparse.dtype,
          'coord min/max:', sparse[:, :3].min(0), sparse[:, :3].max(0),
          'semantic IDs:', np.unique(sparse[:, 3]))
    print('dense OCC:', dense.shape, dense.dtype,
          'semantic IDs:', np.unique(dense))
    depths = train_sample['gt_depth']
    semantics = train_sample['gt_semantic']
    weights = train_sample['gt_semantic_weight']
    print('supervision frames:', len(depths),
          'positive depth pixels:', [int((depth > 0).sum()) for depth in depths])
    print('semantic weight means:', [float(weight.mean()) for weight in weights])
    assert all((depth > 0).any() for depth in depths)
    for semantic, weight in zip(semantics, weights):
        assert np.all((weight >= 0) & (weight <= 1))
        assert np.all(semantic[weight == 0] == 255)
    if len(weights) > 1:
        if history_confidence_mode == 'weighted':
            assert any(
                np.any((weight > 0) & (weight < 1))
                for weight in weights[1:])
        elif history_confidence_mode == 'binary':
            assert all(
                np.all((weight == 0) | (weight == 1))
                for weight in weights[1:])
        else:
            raise ValueError('Unexpected confidence mode: {}'.format(
                history_confidence_mode))
    train_runtime = train_dataset.get_data_info(0)
    val_runtime = val_dataset.get_data_info(0)
    print('train carries observation_path:', 'observation_path' in train_runtime)
    print('val carries observation_path:', 'observation_path' in val_runtime)
    assert 'observation_path' not in train_runtime
    assert 'observation_path' in val_runtime


def main():
    args = parse_args()
    if args.train_iters < 0 or args.val_samples < 1 or args.batch_size < 1:
        raise ValueError(
            'train-iters must be >= 0; val-samples and batch-size must be >= 1')
    quiet_openmmlab_logging()
    cfg = Config.fromfile(args.config)
    if args.num_queries is not None:
        cfg.model.pts_bbox_head.transformer.num_queries = args.num_queries

    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)
    history_confidence_mode = 'weighted'
    for transform in cfg.data.train.pipeline:
        if transform.get('type') == 'LoadGOOSE2DSupervision':
            history_confidence_mode = transform.get(
                'confidence_mode', 'weighted')
    describe_dataset(
        train_dataset, val_dataset, history_confidence_mode)

    train_loader = build_dataloader(
        train_dataset, samples_per_gpu=args.batch_size, workers_per_gpu=0,
        num_gpus=1, dist=False, shuffle=False, seed=0)
    model = build_model(cfg.model)
    model.init_weights()
    model = MMDataParallel(model.cuda(), device_ids=[0])

    if args.load_pretrain:
        load_checkpoint(
            model, cfg.load_from, map_location='cpu',
            revise_keys=cfg.get('revise_keys', []))

    optimizer = build_optimizer(model, cfg.optimizer)
    model.train()
    train_iter = iter(train_loader)
    for step in range(args.train_iters):
        data = next(train_iter)
        optimizer.zero_grad(set_to_none=True)
        losses = model(return_loss=True, **data)
        loss, log_vars = model.module._parse_losses(losses)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss at step {}'.format(step))
        loss.backward()
        optimizer.step()
        print('train step {}: loss={:.6f}'.format(step + 1, log_vars['loss']))
        diagnostics = {
            key: value for key, value in log_vars.items()
            if key in (
                'depth_raw_current', 'depth_raw_history',
                'semantic_raw_current', 'semantic_raw_history',
                'history_confidence_mean',
                'semantic_raw_occlusion', 'occlusion_pixels',
            )
        }
        print('train diagnostics:', diagnostics)
        assert diagnostics['depth_raw_current'] > 0
        if len(data['gt_depth']) > 1:
            assert diagnostics['depth_raw_history'] > 0

    smoke_val = copy.copy(val_dataset)
    count = min(args.val_samples, len(val_dataset))
    smoke_val.data_infos = val_dataset.data_infos[:count]
    smoke_val.flag = val_dataset.flag[:count]
    val_loader = build_dataloader(
        smoke_val, samples_per_gpu=1, workers_per_gpu=0,
        num_gpus=1, dist=False, shuffle=False, seed=0)
    results = safe_single_gpu_test(model, val_loader)
    print('validation metrics:', smoke_val.evaluate(results))
    print('max CUDA memory: {:.1f} MB'.format(
        torch.cuda.max_memory_allocated() / 1024 ** 2))


if __name__ == '__main__':
    main()
