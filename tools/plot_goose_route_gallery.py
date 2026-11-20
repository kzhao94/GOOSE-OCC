#!/usr/bin/env python3
"""Create a deterministic multi-route GOOSE-OCC image gallery."""

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_ROOT = Path('/path/to/GOOSE-OCC')
DEFAULT_OUTPUT = Path(
    'paper/goose_occ_manuscript/figures/generated/fig_goose_route_gallery')

ROUTE_ORDER = (
    ('train', '2022-07-22_flight'),
    ('train', '2022-09-14_garching_uebungsplatz'),
    ('train', '2022-10-12_solalinden_waldwege'),
    ('train', '2023-02-23_campus_roads'),
    ('train', '2023-05-15_neubiberg_rain'),
    ('train', '2023-05-24_neubiberg_cloudy'),
    ('val', '2022-07-27_hoehenkirchner_forst'),
    ('val', '2022-08-30_siegertsbrunn_feldwege'),
)


def load_infos(root, split):
    # The frozen infos were serialized with NumPy 2.x module names, while the
    # training environment uses NumPy 1.x. This alias is read-only and affects
    # only pickle module resolution.
    import sys
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    path = root / 'infos' / 'goose_infos_{}.pkl'.format(split)
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    return payload['infos'] if isinstance(payload, dict) else payload


def select_route_midpoints(rows, route_order=ROUTE_ORDER):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['_split'], row['sequence'])].append(row)
    selected = []
    for split, sequence in route_order:
        candidates = sorted(
            grouped[(split, sequence)],
            key=lambda row: (row.get('frame_index', 0), row['token']))
        if not candidates:
            raise ValueError('Missing route {} {}'.format(split, sequence))
        selected.append(candidates[len(candidates) // 2])
    return selected


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def route_label(sequence):
    parts = sequence.split('_', 1)
    return parts[1].replace('_', ' ') if len(parts) == 2 else sequence


def render(root, output):
    rows = []
    for split in ('train', 'val'):
        for row in load_infos(root, split):
            item = dict(row)
            item['_split'] = split
            rows.append(item)
    selected = select_route_midpoints(rows)

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'font.size': 7,
    })
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 2.45))
    source_rows = []
    for panel_index, (ax, row) in enumerate(zip(axes.flat, selected)):
        relative = row['cams']['CAM_WINDSHIELD_VIS']['data_path']
        image_path = root / relative
        with Image.open(image_path) as raw:
            rgb = raw.convert('RGB')
            ax.imshow(rgb)
            width, height = rgb.size
        ax.set_axis_off()
        panel = chr(ord('a') + panel_index)
        ax.text(0.012, 0.965, panel, transform=ax.transAxes,
                ha='left', va='top', fontsize=9, fontweight='bold',
                color='white',
                bbox={'boxstyle': 'square,pad=0.18', 'facecolor': 'black',
                      'edgecolor': 'none', 'alpha': 0.68})
        ax.set_title('{} | {}'.format(
            row['_split'], route_label(row['sequence'])),
            loc='left', fontsize=7, pad=2.0)
        source_rows.append({
            'panel': panel,
            'split': row['_split'],
            'sequence': row['sequence'],
            'frame_index': int(row['frame_index']),
            'token': row['token'],
            'image_path': relative,
            'image_sha256': sha256(image_path),
            'width': width,
            'height': height,
        })
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.93,
                        wspace=0.035, hspace=0.19)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
    plt.close(fig)

    source = {
        'figure': output.name,
        'selection': 'middle indexed target from each predeclared route',
        'route_order': [list(item) for item in ROUTE_ORDER],
        'image_processing': {
            'crop': 'none',
            'resizing': 'matplotlib display/export only',
            'brightness_contrast_gamma': 'none',
            'local_edits': 'none',
        },
        'panels': source_rows,
    }
    output.with_name(output.name + '_source').with_suffix('.json').write_text(
        json.dumps(source, indent=2) + '\n', encoding='utf-8')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = render(args.dataset_root, args.output)
    print('Rendered {} route panels'.format(len(source['panels'])))


if __name__ == '__main__':
    main()
