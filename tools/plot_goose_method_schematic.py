#!/usr/bin/env python3
"""Render the pose-guided occlusion-supervision method figure."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
from mmcv import Config
from mmdet3d.datasets import build_dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loaders  # noqa: E402,F401
from models.temporal_occlusion import (  # noqa: E402
    source_visible_to_target_occluded,
)


mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 7,
    'axes.linewidth': 0.8,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

COLORS = {
    'neutral': '#4D4D4D',
    'history': '#3775BA',
    'target': '#B64342',
    'candidate': '#C43C88',
    'depth': '#2A9D8F',
    'field': '#DCE8F3',
    'eval': '#E6E6E6',
    'background': '#F7F7F7',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default='configs/goose_temporal_occlusion_common10_3h.py')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--history-index', type=int, default=1)
    parser.add_argument(
        '--output',
        default=('paper/goose_occ_manuscript/figures/generated/'
                 'fig_goose_pose_guided_method'))
    return parser.parse_args()


def candidate_mask(sample, history_index, abs_margin, rel_margin, depth_max):
    meta = sample['img_metas'].data
    depths = sample['gt_depth']
    semantics = sample['gt_semantic']
    weights = sample['gt_semantic_weight']
    target_pose = torch.as_tensor(meta['ego2cam'][0])[None].float()
    source_pose = torch.as_tensor(meta['ego2cam'][history_index])[None].float()
    source_to_target = target_pose @ torch.linalg.inv(source_pose)
    return source_visible_to_target_occluded(
        torch.from_numpy(depths[history_index])[None],
        torch.from_numpy(semantics[history_index].astype(np.int64))[None],
        torch.from_numpy(weights[history_index])[None],
        torch.from_numpy(depths[0])[None],
        torch.as_tensor(meta['cam2img'][history_index])[None].float(),
        torch.as_tensor(meta['cam2img'][0])[None].float(),
        source_to_target,
        depth_max=depth_max,
        occlusion_abs_margin=abs_margin,
        occlusion_rel_margin=rel_margin,
    )[0].cpu().numpy()


def image_rgb(sample, index):
    image = sample['img'].data[index].permute(1, 2, 0).cpu().numpy()
    return image[:, :, ::-1]


def show_image(ax, image, title, color, overlay=None):
    ax.imshow(image)
    if overlay is not None:
        rgba = np.zeros((*overlay.shape, 4), dtype=np.float32)
        rgba[overlay, :3] = mpl.colors.to_rgb(COLORS['candidate'])
        rgba[overlay, 3] = 0.52
        ax.imshow(rgba)
    ax.set_title(title, color=color, fontsize=8, fontweight='bold', pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.2)


def draw_geometry(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.add_patch(Rectangle(
        (0.1, 0.2), 9.8, 5.55, facecolor=COLORS['background'],
        edgecolor='#BDBDBD', linewidth=0.8))

    target_camera = (1.0, 1.0)
    source_camera = (1.0, 4.8)
    blocker = (5.1, 1.7)
    candidate = (7.9, 3.25)
    ax.add_patch(Circle(target_camera, 0.18, color=COLORS['target']))
    ax.add_patch(Circle(source_camera, 0.18, color=COLORS['history']))
    ax.text(0.55, 0.7, 'target camera', color=COLORS['target'],
            ha='left', fontweight='bold')
    ax.text(0.55, 5.2, 'history camera', color=COLORS['history'],
            ha='left', fontweight='bold')

    rng = np.random.default_rng(7)
    centers = rng.uniform((3.0, 0.7), (8.9, 5.15), size=(24, 2))
    palette = ['#B8D5E5', '#A8C9A5', '#E6C99B', '#C9B7D8']
    for index, center in enumerate(centers):
        ax.add_patch(Ellipse(
            center, rng.uniform(0.28, 0.65), rng.uniform(0.15, 0.38),
            angle=rng.uniform(-35, 35), facecolor=palette[index % 4],
            edgecolor='white', linewidth=0.5, alpha=0.85))

    ax.add_patch(Ellipse(
        blocker, 0.65, 0.35, angle=25, facecolor=COLORS['depth'],
        edgecolor='white', linewidth=0.8, zorder=5))
    ax.add_patch(Ellipse(
        candidate, 0.75, 0.42, angle=-15,
        facecolor=COLORS['candidate'], edgecolor='white', linewidth=0.8,
        zorder=6))

    ax.add_patch(FancyArrowPatch(
        source_camera, candidate, arrowstyle='-|>', mutation_scale=10,
        color=COLORS['history'], linewidth=1.6, zorder=7))
    ax.add_patch(FancyArrowPatch(
        target_camera, blocker, arrowstyle='-|>', mutation_scale=10,
        color=COLORS['target'], linewidth=1.6, zorder=7))
    ax.plot(
        [blocker[0], candidate[0]], [blocker[1], candidate[1]],
        color=COLORS['target'], linestyle=(0, (3, 2)), linewidth=1.2,
        zorder=7)
    ax.text(5.05, 1.13, 'target pseudo-depth surface',
            color=COLORS['depth'], ha='center')
    ax.text(8.1, 3.75, 'source-visible candidate',
            color=COLORS['candidate'], ha='center', fontweight='bold')
    ax.text(6.25, 2.05, 'behind pseudo-depth', color=COLORS['target'],
            ha='center', rotation=25)
    ax.text(5.0, 5.45, 'target-coordinate sparse Gaussian field',
            ha='center', color=COLORS['neutral'], fontweight='bold')
    ax.text(
        5.0, 0.43,
        r'$z_{t\leftarrow s} > d_t + \max(0.5\,\mathrm{m},\,0.05d_t)$',
        ha='center', color=COLORS['neutral'])


def box(ax, xy, width, height, text, facecolor, edgecolor, textcolor=None,
        fontsize=5.8):
    patch = Rectangle(
        xy, width, height, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.0)
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2, text,
        ha='center', va='center', fontsize=fontsize,
        color=textcolor or COLORS['neutral'])
    return patch


def arrow(ax, start, end, color=COLORS['neutral'], dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=9, linewidth=1.0,
        linestyle='--' if dashed else '-', color=color))


def draw_loss_boundary(ax):
    ax.set_xlim(0, 15.8)
    ax.set_ylim(0, 4.2)
    ax.axis('off')
    box(ax, (0.2, 2.7), 2.5, 1.0,
        'target RGB\nofficial 2D semantics\nMetric3D depth',
        '#F7DFDC', COLORS['target'])
    box(ax, (0.2, 1.15), 2.5, 1.0,
        'history RGB\nSAM3 + confidence\nMetric3D depth',
        '#DCE9F5', COLORS['history'])
    box(ax, (3.55, 1.65), 2.55, 1.35,
        'target-coordinate\nsparse Gaussian field',
        COLORS['field'], '#6F91B2')
    box(ax, (7.0, 2.55), 2.35, 0.9,
        'target semantic\n+ depth loss', '#F7DFDC', COLORS['target'])
    box(ax, (7.0, 1.35), 2.35, 0.9,
        'history semantic\n+ depth loss', '#DCE9F5', COLORS['history'])
    box(ax, (7.0, 0.15), 2.35, 0.9,
        'pose-guided\ncandidate\nsemantic loss', '#F4DCEA',
        COLORS['candidate'])
    box(ax, (10.25, 1.65), 2.35, 1.25,
        r'$L=L_{target}+\lambda_hL_{history}$'
        '\n' r'$+\lambda_{occ}L_{candidate}$',
        '#F0F0F0', COLORS['neutral'], fontsize=5.1)

    arrow(ax, (2.7, 3.2), (3.55, 2.65), COLORS['target'])
    arrow(ax, (2.7, 1.65), (3.55, 2.0), COLORS['history'])
    arrow(ax, (6.1, 2.5), (7.0, 3.0), COLORS['target'])
    arrow(ax, (6.1, 2.25), (7.0, 1.8), COLORS['history'])
    arrow(ax, (2.7, 1.35), (7.0, 0.6), COLORS['candidate'])
    for y in (3.0, 1.8, 0.6):
        arrow(ax, (9.35, y), (10.25, 2.25), COLORS['neutral'])

    box(ax, (13.2, 1.65), 2.4, 1.25,
        'OCC GT +\nobservation masks\nevaluation only',
        COLORS['eval'], '#8C8C8C', fontsize=5.1)
    ax.plot([12.9, 12.9], [0.02, 3.75], color='#8C8C8C',
            linestyle=(0, (2, 2)), linewidth=0.8)
    ax.text(12.98, 3.72, 'training boundary', color='#666666',
            ha='left', va='top', fontsize=6.4)


def panel_label(ax, label):
    ax.text(-0.04, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight='bold', ha='left', va='bottom')


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output.with_suffix('.tiff'), dpi=600, bbox_inches='tight')


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError('sample index is outside the training dataset')
    sample = dataset[args.sample_index]
    frame_count = sample['img'].data.shape[0]
    if not 1 <= args.history_index < frame_count:
        raise IndexError('history index must identify a real history frame')

    head = cfg.model.pts_bbox_head
    abs_margin = float(head.get('occlusion_abs_margin', 0.5))
    rel_margin = float(head.get('occlusion_rel_margin', 0.05))
    depth_max = float(head.loss_cfgs.get('depth_max', 50.0))
    mask = candidate_mask(
        sample, args.history_index, abs_margin, rel_margin, depth_max)
    valid_history = (
        (sample['gt_semantic'][args.history_index] != 255)
        & (sample['gt_semantic_weight'][args.history_index] > 0))
    if not mask.any():
        raise RuntimeError('selected sample has no pose-guided candidates')

    fig = plt.figure(figsize=(7.2, 5.5), facecolor='white')
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.45, 1.0], width_ratios=[1.0, 1.25],
        hspace=0.22, wspace=0.16)
    image_grid = grid[0, 0].subgridspec(2, 1, hspace=0.2)
    ax_history = fig.add_subplot(image_grid[0, 0])
    ax_target = fig.add_subplot(image_grid[1, 0])
    show_image(
        ax_history, image_rgb(sample, args.history_index),
        'history: source-visible candidates', COLORS['history'], mask)
    show_image(
        ax_target, image_rgb(sample, 0),
        'target: current observation', COLORS['target'])
    panel_label(ax_history, 'a')

    ax_geometry = fig.add_subplot(grid[0, 1])
    draw_geometry(ax_geometry)
    panel_label(ax_geometry, 'b')

    ax_loss = fig.add_subplot(grid[1, :])
    draw_loss_boundary(ax_loss)
    panel_label(ax_loss, 'c')

    output = Path(args.output)
    save_figure(fig, output)
    plt.close(fig)

    meta = sample['img_metas'].data
    provenance = {
        'figure': output.name,
        'config': args.config,
        'sample_index': args.sample_index,
        'sample_token': str(meta['sample_idx']),
        'sequence': str(meta['sequence']),
        'history_index': args.history_index,
        'history_rank': int(meta['history_ranks'][args.history_index]),
        'render_size': list(meta['render_size']),
        'valid_history_pixels': int(valid_history.sum()),
        'candidate_pixels': int(mask.sum()),
        'candidate_fraction_of_valid_history': float(
            mask.sum() / max(valid_history.sum(), 1)),
        'occlusion_abs_margin_m': abs_margin,
        'occlusion_rel_margin': rel_margin,
        'training_inputs': [
            'RGB', 'official target semantics', 'SAM3 history semantics',
            'SAM3 confidence', 'Metric3D target/history depth',
            'camera intrinsics', 'camera poses',
        ],
        'evaluation_only': ['OCC GT', 'observation masks'],
    }
    output.with_name(output.name + '_source.json').write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
