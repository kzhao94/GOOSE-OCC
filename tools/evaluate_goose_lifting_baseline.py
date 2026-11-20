#!/usr/bin/env python3
"""Evaluate RGB semantic prediction + Metric3D geometric lifting on GOOSE."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from goose_lifting_baseline import (
    common_target_infos, confusion_result, create_segmenter, densify_occ,
    evaluation_record, lift_to_grid, load_config, prediction_record,
    read_image_tensor, scaled_intrinsic, sha256, update_confusion)


CLASS_NAMES = (
    'free', 'artificial_structures', 'artificial_ground', 'natural_ground',
    'obstacle', 'vehicle', 'vegetation', 'human')


def load_visibility_module():
    path = Path(__file__).resolve().parents[1] / 'loaders' / (
        'goose_visibility_metrics.py')
    spec = importlib.util.spec_from_file_location(
        'goose_lifting_visibility_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=(
        'configs/goose_geometric_lifting_common10.py'))
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-targets', type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(args.output_dir or config['output_dir'])
    checkpoint_path = Path(args.checkpoint or (
        run_dir / 'epoch_{}.pth'.format(config['epochs'])))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = create_segmenter(config['num_classes'], False)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device).eval()

    infos = common_target_infos(
        config['val_infos'], config['val_temporal_infos'],
        config['min_history'])
    if len(infos) != 469:
        raise ValueError('Expected 469 common val targets, got {}'.format(
            len(infos)))
    if args.max_targets is not None:
        infos = infos[:args.max_targets]

    visibility_module = load_visibility_module()
    visibility = visibility_module.GooseVisibilityMetrics(CLASS_NAMES)
    full_confusion = np.zeros((8, 8), dtype=np.int64)
    predictions_dir = run_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    inference_seconds = 0.0
    for index, info in enumerate(infos):
        prediction_input = prediction_record(info, config['dataset_root'])
        image = read_image_tensor(
            prediction_input['image_path'], config['image_size'])
        original = cv2.imread(
            prediction_input['image_path'], cv2.IMREAD_COLOR)
        if original is None:
            raise IOError('Failed to read inference image')
        started = time.perf_counter()
        with torch.no_grad():
            logits = model(image[None].to(args.device))['out']
            semantic = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        depth_raw = cv2.imread(
            prediction_input['depth_path'], cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise IOError('Failed to read Metric3D depth')
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        depth = depth_raw.astype(np.float32) / float(config['depth_scale'])
        height, width = config['image_size']
        depth = cv2.resize(
            depth, (width, height), interpolation=cv2.INTER_LINEAR)
        intrinsic = scaled_intrinsic(
            prediction_input['intrinsic'], original.shape[:2],
            config['image_size'])
        prediction = lift_to_grid(
            semantic, depth, intrinsic, prediction_input['lidar2cam'],
            config['pc_range'], config['occ_size'], config['voxel_size'])
        if prediction.shape != tuple(config['occ_size']):
            raise ValueError('Prediction shape contract failed')
        inference_seconds += time.perf_counter() - started

        output_path = predictions_dir / 'prediction_{:04d}.npz'.format(index)
        np.savez_compressed(output_path, prediction=prediction)

        # Evaluation-only products are opened after prediction construction.
        evaluation = evaluation_record(info, config['dataset_root'])
        target = densify_occ(evaluation['occ_path'], config['occ_size'])
        masks = visibility_module.load_observation(
            evaluation['observation_path'])
        update_confusion(full_confusion, prediction, target)
        visibility.update(prediction, target, masks)
        manifest.append({
            'index': index,
            'token': prediction_input['token'],
            'sequence': prediction_input['sequence'],
            'frame_index': prediction_input['frame_index'],
            'prediction_path': str(output_path),
            'prediction_sha256': sha256(output_path),
            'occupied_voxels': int((prediction > 0).sum()),
        })
        if (index + 1) % 50 == 0 or index + 1 == len(infos):
            print('evaluated {}/{}'.format(index + 1, len(infos)), flush=True)

    result = {
        'experiment': 'goose_geometric_lifting_common10',
        'role': 'architecture-distinct image-semantic + depth lifting baseline',
        'supervision': (
            'train official 2D semantics; val RGB semantic inference + '
            'Metric3D depth; no OCC/LiDAR/mask training'),
        'num_samples': len(infos),
        'prediction_contract': {
            'shape': list(config['occ_size']),
            'axis_order': 'x,y,z',
            'dtype': 'uint8',
            'class_ids': '0..7',
        },
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256(checkpoint_path),
        'inference_seconds': inference_seconds,
        'fps': len(infos) / inference_seconds,
        'full_grid': confusion_result(full_confusion, CLASS_NAMES),
        'visibility': visibility.result(),
        'prediction_manifest': manifest,
    }
    result_path = run_dir / 'goose_visibility_val.json'
    result_path.write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'result': str(result_path),
        'samples': len(infos),
        'SC-IoU': result['full_grid']['SC-IoU'],
        'SSC-mIoU': result['full_grid']['SSC-mIoU'],
        'V-SC-IoU': result['visibility']['V-SC-IoU'],
        'O-SC-IoU': result['visibility']['O-SC-IoU'],
    }, indent=2))


if __name__ == '__main__':
    main()
