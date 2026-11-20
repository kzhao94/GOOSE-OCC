#!/usr/bin/env python3
"""Plot one released GOOSE-OCC observation record as top-down mask panels."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 8,
    'axes.spines.right': False,
    'axes.spines.top': False,
})


def load_infos(path):
    # Released pickles can reference NumPy 2.x module names while this
    # environment may still run NumPy 1.x.
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    payload = pickle.load(path.open('rb'))
    return payload.get('infos', payload)


def unpack(payload, name, shape):
    bits = np.unpackbits(
        np.asarray(payload[name], dtype=np.uint8), bitorder='little',
        count=int(np.prod(shape)))
    return bits.reshape(shape).astype(bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root', default=os.environ.get(
            'GOOSE_OCC_ROOT', '/path/to/GOOSE-OCC'))
    parser.add_argument('--split', default='val', choices=('train', 'val'))
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument(
        '--output-dir',
        default='paper/goose_occ_manuscript/figures/generated')
    args = parser.parse_args()
    root = Path(args.data_root)
    infos = load_infos(root / 'infos' / f'goose_infos_{args.split}.pkl')
    info = infos[args.index]
    path = root / info['observation_path']
    with np.load(path, allow_pickle=False) as payload:
        shape = tuple(int(value) for value in payload['shape'])
        names = (
            'visible_known_free_mask', 'visible_occupied_mask',
            'occluded_known_free_mask', 'occluded_observed_mask')
        masks = [unpack(payload, name, shape) for name in names]
        observation_format = str(payload['format'])
    masks += [masks[0] | masks[1], masks[2] | masks[3]]
    assert not np.any(masks[0] & masks[1])
    assert not np.any(masks[2] & masks[3])
    titles = (
        'Visible known-free', 'Visible occupied', 'Occluded known-free',
        'Occluded observed', 'V-valid', 'O-valid')
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), constrained_layout=True)
    for ax, mask, title in zip(axes.flat, masks, titles):
        image = mask.any(axis=2).T
        ax.imshow(image, origin='lower', cmap='Greys', interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('x voxel')
        ax.set_ylabel('y voxel')
        ax.set_aspect('equal')
    fig.suptitle('GOOSE-OCC v1.4 observation domains', y=1.02)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f'fig_goose_observation_masks_{args.split}_{args.index:04d}'
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
    plt.close(fig)
    source = {
        'figure': stem.name,
        'split': args.split,
        'index': args.index,
        'sample_token': str(info['token']),
        'observation_format': observation_format,
        'shape': list(shape),
        'mask_voxel_counts': {
            title: int(mask.sum())
            for title, mask in zip(titles, masks)
        },
        'projection': 'top-down any over z',
        'training_use': False,
    }
    stem.with_name(stem.name + '_source.json').write_text(
        json.dumps(source, indent=2, sort_keys=True), encoding='utf-8')
    print(stem)


if __name__ == '__main__':
    main()
