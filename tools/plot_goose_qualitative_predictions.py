#!/usr/bin/env python3
"""Create top-down qualitative panels from val.py prediction dumps."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
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

CLASS_NAMES = (
    'artificial structures', 'artificial ground', 'natural ground',
    'obstacle', 'vehicle', 'vegetation', 'human')
CLASS_COLORS = (
    '#4C78A8', '#B8B8B8', '#D6A84B', '#E45756', '#8F6BB3', '#59A14F',
    '#E377C2')
SEMANTIC_CMAP = ListedColormap(CLASS_COLORS)
SEMANTIC_NORM = BoundaryNorm(np.arange(0.5, 8.5), SEMANTIC_CMAP.N)


def load_infos(path):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    payload = pickle.load(path.open('rb'))
    return payload.get('infos', payload)


def load_gt(root, info):
    sparse = np.load(root / info['occ_path'], allow_pickle=False).astype(np.int64)
    dense = np.zeros((200, 128, 32), dtype=np.uint8)
    dense[sparse[:, 0], sparse[:, 1], sparse[:, 2]] = sparse[:, 3]
    return dense


def load_predictions(prediction_dir, count):
    paths = sorted(Path(prediction_dir).glob('prediction_*.npz'))[:count]
    if not paths:
        raise FileNotFoundError(
            'No prediction_*.npz in {}'.format(prediction_dir))
    predictions = []
    for path in paths:
        with np.load(path) as payload:
            if 'sample_token' not in payload:
                raise ValueError(
                    '{} is an obsolete dump without sample_token'.format(path))
            if 'axis_order' in payload and str(payload['axis_order']) != 'x,y,z':
                raise ValueError(
                    '{} has unexpected axis order {}'.format(
                        path, str(payload['axis_order'])))
            prediction = payload['occ_pred']
            if prediction.ndim == 4 and prediction.shape[0] == 1:
                prediction = prediction[0]
            if prediction.shape != (200, 128, 32):
                raise ValueError(
                    '{} has prediction shape {}, expected [200,128,32]'.format(
                        path, prediction.shape))
            if prediction.size and (
                    prediction.min() < 0 or prediction.max() > 7):
                raise ValueError(
                    '{} has prediction IDs outside 0..7: [{},{}]'.format(
                        path, prediction.min(), prediction.max()))
            predictions.append((str(payload['sample_token']), prediction))
    return predictions


def topdown_semantic(volume):
    occupied = (volume >= 1) & (volume <= 7)
    has_occupied = occupied.any(axis=2)
    highest_from_top = occupied[..., ::-1].argmax(axis=2)
    highest_z = volume.shape[2] - 1 - highest_from_top
    bev = np.take_along_axis(volume, highest_z[..., None], axis=2)[..., 0]
    return np.ma.masked_where(~has_occupied, bev).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/path/to/GOOSE-OCC')
    parser.add_argument('--infos', default=None)
    parser.add_argument('--prediction-dir', required=True)
    parser.add_argument('--baseline-prediction-dir', default=None)
    parser.add_argument(
        '--output-dir',
        default='paper/goose_occ_manuscript/figures/generated')
    parser.add_argument('--count', type=int, default=4)
    args = parser.parse_args()
    root = Path(args.data_root)
    infos_path = Path(args.infos) if args.infos else root / 'infos' / 'goose_infos_val.pkl'
    infos = load_infos(infos_path)
    infos_by_token = {str(info['token']): info for info in infos}
    pred_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(pred_dir, args.count)
    baseline_by_token = None
    if args.baseline_prediction_dir:
        baseline_by_token = dict(load_predictions(
            args.baseline_prediction_dir, args.count))
        missing = [token for token, _ in predictions
                   if token not in baseline_by_token]
        if missing:
            raise ValueError(
                'Baseline predictions do not contain method tokens: {}'.format(
                    ', '.join(missing)))
    columns = 4 if baseline_by_token is not None else 3
    width = 7.2 if columns == 4 else 7.0
    fig, axes = plt.subplots(len(predictions), columns,
                             figsize=(width, 1.75 * len(predictions)),
                             squeeze=False, constrained_layout=True)
    for row, (token, prediction) in enumerate(predictions):
        if token not in infos_by_token:
            raise KeyError('prediction token is absent from infos: {}'.format(
                token))
        target = load_gt(root, infos_by_token[token])
        if baseline_by_token is None:
            panels = (target, prediction, prediction != target)
            titles = ('OCC target', 'Prediction', 'Voxel disagreement')
        else:
            baseline = baseline_by_token[token]
            if baseline.shape != prediction.shape:
                raise ValueError(
                    'Prediction shape mismatch for token {}: {} versus {}'.format(
                        token, baseline.shape, prediction.shape))
            panels = (target, baseline, prediction, baseline != prediction)
            titles = ('OCC target', 'Temporal baseline',
                      'Pose-guided method', 'Prediction disagreement')
        for column, (ax, panel, title) in enumerate(zip(
                axes[row], panels, titles)):
            semantic_panel = column < (3 if baseline_by_token is not None else 2)
            if semantic_panel:
                image = topdown_semantic(panel)
                ax.imshow(image, origin='lower', cmap=SEMANTIC_CMAP,
                          norm=SEMANTIC_NORM, interpolation='nearest')
                ax.set_facecolor('#F4F4F4')
            else:
                image = panel.any(axis=2).T
                ax.imshow(image, origin='lower', cmap='Greys', vmin=0, vmax=1,
                          interpolation='nearest')
            ax.set_title(title if row == 0 else '')
            ax.set_xlabel('x voxel')
            ax.set_ylabel('y voxel')
            ax.set_aspect('equal')
    stem = output_dir / 'fig_goose_qualitative_predictions'
    handles = [Patch(facecolor=color, edgecolor='none', label=name)
               for color, name in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=4, frameon=False, fontsize=7)
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
    source = {
        'sample_tokens': [token for token, _ in predictions],
        'mode': 'baseline_method_comparison'
        if baseline_by_token is not None else 'single_prediction',
        'prediction_dir': pred_dir.name,
        'baseline_prediction_dir': (
            Path(args.baseline_prediction_dir).name
            if args.baseline_prediction_dir else None),
        'bev_projection': 'semantic class of highest occupied voxel',
        'change_panel': 'binary voxel difference, not error or occlusion',
        'class_colors': dict(zip(CLASS_NAMES, CLASS_COLORS)),
        'observation_masks_read': False,
    }
    stem.with_name(stem.name + '_source').with_suffix('.json').write_text(
        json.dumps(source, indent=2) + '\n')
    plt.close(fig)
    print(stem)


if __name__ == '__main__':
    main()
