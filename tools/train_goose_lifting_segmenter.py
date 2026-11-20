#!/usr/bin/env python3
"""Train the train-label-only 2D segmenter for geometric lifting."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from goose_lifting_baseline import (
    Goose2DSemanticTrainDataset, common_target_infos, create_segmenter,
    load_config, taxonomy_lut)


def jsonable(config):
    return {key: value for key, value in config.items()
            if isinstance(value, (str, int, float, bool, list, tuple))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=(
        'configs/goose_geometric_lifting_common10.py'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output_dir or config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)

    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    infos = common_target_infos(
        config['train_infos'], config['train_temporal_infos'],
        config['min_history'])
    if len(infos) != 2726:
        raise ValueError('Expected 2726 common train targets, got {}'.format(
            len(infos)))
    dataset = Goose2DSemanticTrainDataset(
        infos, config['dataset_root'], taxonomy_lut(config['taxonomy_path']),
        config['image_size'], augment=True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=int(config['batch_size']), shuffle=True,
        num_workers=int(config['workers']), pin_memory=True,
        drop_last=True, generator=generator)

    pretrained = bool(config['pretrained_backbone']) and not args.no_pretrained
    model = create_segmenter(config['num_classes'], pretrained).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config['epochs']))
    class_weights = torch.tensor(
        [0.0] + [1.0] * (int(config['num_classes']) - 1),
        device=args.device)
    scaler = GradScaler(enabled=args.device.startswith('cuda'))

    resolved = jsonable(config)
    resolved.update({
        'config_path': str(Path(args.config).resolve()),
        'device': args.device,
        'common_train_targets': len(infos),
        'pretrained_backbone_used': pretrained,
        'model_selection': 'none; fixed final epoch',
    })
    (output / 'resolved_config.json').write_text(
        json.dumps(resolved, indent=2) + '\n', encoding='utf-8')
    log_path = output / 'train_log.jsonl'
    global_step = 0
    for epoch in range(1, int(config['epochs']) + 1):
        model.train()
        loss_sum = 0.0
        valid_pixels = 0
        started = time.perf_counter()
        for images, labels in loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.device.startswith('cuda')):
                logits = model(images)['out']
                loss = F.cross_entropy(
                    logits, labels, weight=class_weights,
                    ignore_index=int(config['ignore_id']))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite 2D segmentation loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = int((labels != int(config['ignore_id'])).sum().item())
            loss_sum += float(loss.item()) * count
            valid_pixels += count
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        scheduler.step()
        record = {
            'epoch': epoch,
            'global_step': global_step,
            'mean_pixel_weighted_loss': loss_sum / max(valid_pixels, 1),
            'valid_pixels': valid_pixels,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'elapsed_seconds': time.perf_counter() - started,
        }
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'resolved_config': resolved,
            'last_record': record,
        }
        torch.save(state, output / 'latest.pth')
        print(json.dumps(record), flush=True)
        if args.max_steps is not None and global_step >= args.max_steps:
            break
    if args.max_steps is None:
        torch.save(state, output / 'epoch_{}.pth'.format(config['epochs']))


if __name__ == '__main__':
    main()
