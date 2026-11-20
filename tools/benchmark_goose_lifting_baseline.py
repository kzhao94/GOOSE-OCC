#!/usr/bin/env python3
"""Benchmark the geometric lifting baseline under the shared GPU protocol."""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from goose_lifting_baseline import (
    common_target_infos, create_segmenter, lift_to_grid, load_config,
    prediction_record, read_image_tensor, scaled_intrinsic)


class LiftingInputDataset(Dataset):
    def __init__(self, infos, config):
        self.records = [
            prediction_record(info, config['dataset_root']) for info in infos]
        self.config = config

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_image_tensor(record['image_path'], self.config['image_size'])
        original = cv2.imread(record['image_path'], cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(record['depth_path'], cv2.IMREAD_UNCHANGED)
        if original is None or depth_raw is None:
            raise IOError('Failed to load lifting input')
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        height, width = self.config['image_size']
        depth = cv2.resize(
            depth_raw.astype(np.float32) / float(self.config['depth_scale']),
            (width, height), interpolation=cv2.INTER_LINEAR)
        intrinsic = scaled_intrinsic(
            record['intrinsic'], original.shape[:2], self.config['image_size'])
        return {
            'image': image,
            'depth': torch.from_numpy(depth),
            'intrinsic': torch.from_numpy(intrinsic),
            'lidar2cam': torch.from_numpy(record['lidar2cam']),
        }


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=(
        'configs/goose_geometric_lifting_common10.py'))
    parser.add_argument('--checkpoint', default=(
        'work_dir/goose_geometric_lifting_common10/epoch_12.pth'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--output', default=(
        'work_dir/efficiency/goose_geometric_lifting_common10.json'))
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = create_segmenter(config['num_classes'], False)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device).eval()
    infos = common_target_infos(
        config['val_infos'], config['val_temporal_infos'],
        config['min_history'])[:args.samples]
    loader = DataLoader(
        LiftingInputDataset(infos, config), batch_size=1, shuffle=False,
        num_workers=int(config['workers']), pin_memory=True)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    forward_times = []
    data_times = []
    end_to_end_times = []
    baseline_allocated = None
    iterator = iter(loader)
    with torch.no_grad():
        for index in range(len(infos)):
            all_started = time.perf_counter()
            batch = next(iterator)
            data_elapsed = time.perf_counter() - all_started

            torch.cuda.synchronize()
            forward_started = time.perf_counter()
            logits = model(batch['image'].to(args.device))['out']
            semantic = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            prediction = lift_to_grid(
                semantic, batch['depth'][0].numpy(),
                batch['intrinsic'][0].numpy(),
                batch['lidar2cam'][0].numpy(),
                config['pc_range'], config['occ_size'], config['voxel_size'])
            torch.cuda.synchronize()
            forward_elapsed = time.perf_counter() - forward_started
            if prediction.shape != tuple(config['occ_size']):
                raise ValueError('Prediction contract failed')
            if index >= args.num_warmup:
                if baseline_allocated is None:
                    baseline_allocated = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                forward_times.append(forward_elapsed)
                data_times.append(data_elapsed)
                end_to_end_times.append(data_elapsed + forward_elapsed)

    measured = len(forward_times)
    payload = {
        'config': args.config,
        'weights': args.checkpoint,
        'num_warmup': args.num_warmup,
        'samples': args.samples,
        'measured_samples': measured,
        'parameters': parameters,
        'trainable_parameters': trainable,
        'checkpoint_size_bytes': Path(args.checkpoint).stat().st_size,
        'gpu_name': torch.cuda.get_device_name(0),
        'mean_time_sec': statistics.mean(forward_times),
        'forward_std_sec': statistics.stdev(forward_times),
        'forward_p50_sec': percentile(forward_times, 0.5),
        'forward_p95_sec': percentile(forward_times, 0.95),
        'fps': measured / sum(forward_times),
        'data_loading_mean_sec': statistics.mean(data_times),
        'end_to_end_mean_sec': statistics.mean(end_to_end_times),
        'end_to_end_p50_sec': percentile(end_to_end_times, 0.5),
        'end_to_end_p95_sec': percentile(end_to_end_times, 0.95),
        'end_to_end_fps': measured / sum(end_to_end_times),
        'persistent_allocated_mb': baseline_allocated / 1024 ** 2,
        'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
        'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024 ** 2,
        'evaluation_gt_stripped': True,
        'times_sec': forward_times,
        'data_times_sec': data_times,
        'end_to_end_times_sec': end_to_end_times,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'forward_fps': payload['fps'],
        'end_to_end_fps': payload['end_to_end_fps'],
        'peak_allocated_mb': payload['peak_allocated_mb'],
    }, indent=2))


if __name__ == '__main__':
    main()
