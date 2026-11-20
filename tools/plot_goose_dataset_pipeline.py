#!/usr/bin/env python3
"""Render a release-grounded GOOSE-OCC construction and protocol figure."""

import argparse
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 7,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

COLORS = {
    'raw': '#D9D9D9',
    'target': '#B64342',
    'history': '#3775BA',
    'occ': '#2A9D8F',
    'observation': '#9A4D8E',
    'audit': '#F0C36E',
    'text': '#3F3F3F',
    'line': '#777777',
    'background': '#F8F8F8',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset-root',
        default=os.environ.get(
            'GOOSE_OCC_ROOT', '/path/to/GOOSE-OCC'))
    parser.add_argument(
        '--generator-root',
        default=('/path/to/SurroundOcc/tools/'
                 'generate_occupancy_goose'))
    parser.add_argument(
        '--output',
        default=('paper/goose_occ_manuscript/figures/generated/'
                 'fig_goose_dataset_pipeline'))
    return parser.parse_args()


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def draw_box(ax, xy, width, height, title, body, color, fontsize=6.0):
    ax.add_patch(Rectangle(
        xy, width, height, facecolor=mpl.colors.to_rgba(color, 0.16),
        edgecolor=color, linewidth=1.1))
    ax.text(
        xy[0] + width / 2, xy[1] + height * 0.68, title,
        ha='center', va='center', fontweight='bold', color=color,
        fontsize=fontsize + 0.4)
    ax.text(
        xy[0] + width / 2, xy[1] + height * 0.34, body,
        ha='center', va='center', color=COLORS['text'], fontsize=fontsize,
        linespacing=1.25)


def arrow(ax, start, end, color=COLORS['line']):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=9, linewidth=1.0,
        color=color))


def panel_label(ax, label):
    ax.text(-0.04, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight='bold', ha='left', va='bottom')


def draw_funnel(ax, audit, match):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    stages = [
        ('official semantic frames', audit['semantic_frames'], '#777777'),
        ('complete modalities', audit['complete_targets'], '#8C8C8C'),
        ('readable bag archive',
         audit['complete_targets_matched_to_readable_archive'], '#6F91B2'),
        ('pose-valid OCC targets', match['usable_targets'], COLORS['occ']),
    ]
    x_values = (0.15, 3.15, 6.15, 9.15)
    for index, ((label, value, color), x) in enumerate(zip(stages, x_values)):
        draw_box(ax, (x, 1.05), 2.35, 1.65, label,
                 '{:,} frames'.format(value), color, fontsize=5.8)
        if index < len(stages) - 1:
            arrow(ax, (x + 2.35, 1.88), (x_values[index + 1], 1.88))
    ax.text(
        6.0, 3.45,
        'RGB + 2D/3D semantics + LiDAR timestamp + calibration + pose',
        ha='center', color=COLORS['text'], fontsize=6.4)
    ax.text(
        10.32, 0.55,
        '3,293 exact + 74 validated nearest joins',
        ha='center', color=COLORS['occ'], fontsize=5.7)
    panel_label(ax, 'a')


def draw_release_contract(ax, release, analysis):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    split = release['split_counts']
    draw_box(
        ax, (0.2, 1.0), 3.25, 2.1, 'route-level split',
        'train {:,}  |  val {:,}\nno sequence crosses splits'.format(
            split['train'], split['val']), COLORS['target'])
    draw_box(
        ax, (4.35, 1.0), 3.25, 2.1, 'single-camera task',
        'CAM_WINDSHIELD_VIS\ntarget RGB + causal history', COLORS['history'])
    grid = release['observation']['protocol']['occ_size']
    pc_range = release['observation']['protocol']['pc_range']
    visible_fraction = analysis['ranges']['50x32x8_final'][
        'fraction_of_visible_points_in_range']
    draw_box(
        ax, (8.5, 1.0), 3.25, 2.1, 'frozen OCC grid',
        '{} x {} x {} at 0.25 m\n[0,50) x [-16,16) x [-4,4) m\n'
        'captures {:.2f}% front-visible points'.format(
            grid[0], grid[1], grid[2], 100 * visible_fraction), COLORS['occ'],
        fontsize=5.6)
    panel_label(ax, 'b')


def draw_training_products(ax, release):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    temporal = release['temporal']
    pseudo = temporal['pseudo_labels']
    metric = release['metric3d_depth']
    draw_box(
        ax, (0.2, 2.5), 3.2, 1.65, 'target supervision',
        '{:,} official 2D labels\n{:,} Metric3D\npseudo-depth maps'.format(
            release['targets'], metric['validated_files']['target']),
        COLORS['target'], fontsize=5.6)
    draw_box(
        ax, (0.2, 0.45), 3.2, 1.65, 'history supervision',
        '{:,} causal relations\n{:,} unique history frames\nRGB + SAM3 + depth'.format(
            temporal['history_relations'], temporal['unique_history_images']),
        COLORS['history'], fontsize=5.6)
    draw_box(
        ax, (4.45, 1.45), 3.3, 1.95, 'weak-supervision inputs',
        'RGB + semantics + confidence\nMetric3D depth + intrinsics + pose',
        '#6F91B2')
    draw_box(
        ax, (8.75, 1.45), 3.0, 1.95, 'offline pseudo-label audit',
        '353 official overlaps\ncoverage {:.2f}%  |  accuracy {:.2f}%\n'
        '7-class mIoU {:.2f}%'.format(
            100 * pseudo['official_overlap_audit']['coverage'],
            100 * pseudo['official_overlap_audit'][
                'covered_pixel_accuracy'],
            100 * pseudo['official_overlap_audit']['penalized_miou']),
        COLORS['audit'], fontsize=5.6)
    arrow(ax, (3.4, 3.3), (4.45, 2.75), COLORS['target'])
    arrow(ax, (3.4, 1.25), (4.45, 2.05), COLORS['history'])
    arrow(ax, (7.75, 2.42), (8.75, 2.42), COLORS['audit'])
    ax.text(
        6.1, 4.55,
        'VisibilityOcc training branch: image-derived supervision only',
        ha='center', color=COLORS['history'], fontweight='bold', fontsize=6.5)
    panel_label(ax, 'c')


def draw_evaluation_products(ax, release, validation, observation):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    modalities = release['modalities']
    draw_box(
        ax, (0.2, 2.5), 3.2, 1.65, 'semantic OCC GT',
        '{:,} sparse int32 labels\n7 occupied classes + free'.format(
            modalities['occupancy']['files']), COLORS['occ'])
    draw_box(
        ax, (0.2, 0.45), 3.2, 1.65, 'observation state',
        '{:,} compressed mask records\nray free + visible/occluded\noccupied'.format(
            modalities['observation_masks']['files']),
        COLORS['observation'], fontsize=5.7)
    draw_box(
        ax, (4.45, 1.45), 3.3, 1.95, 'evaluation-only domains',
        'V-valid = visible free | occupied\n'
        'O-valid = occluded free | observed\n0-10 / 10-25 / 25-50 m',
        COLORS['observation'], fontsize=5.6)
    draw_box(
        ax, (8.75, 1.45), 3.0, 1.95, 'release integrity',
        'OCC missing 0  |  mask missing 0\n'
        'pose missing 0  |  errors 0\nSHA-256 release manifest',
        COLORS['audit'], fontsize=5.6)
    arrow(ax, (3.4, 3.3), (4.45, 2.75), COLORS['occ'])
    arrow(ax, (3.4, 1.25), (4.45, 2.05), COLORS['observation'])
    arrow(ax, (7.75, 2.42), (8.75, 2.42), COLORS['audit'])
    ax.text(
        6.1, 4.55,
        'LiDAR-derived products: validation and evaluation only',
        ha='center', color=COLORS['observation'], fontweight='bold',
        fontsize=6.5)
    assert validation['errors'] == []
    assert observation['errors'] == []
    panel_label(ax, 'd')


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output.with_suffix('.tiff'), dpi=600, bbox_inches='tight')


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    generator_root = Path(args.generator_root)
    audit = load_json(generator_root / 'artifacts/audit/summary.json')
    match = load_json(
        generator_root / 'artifacts/index/target_match_summary.json')
    analysis = load_json(generator_root / 'artifacts/analysis/summary.json')
    release = load_json(dataset_root / 'release.json')
    validation = load_json(dataset_root / 'validation.json')
    observation = load_json(
        dataset_root / 'observation/observation_validation.json')

    assert release['version'] == '1.4.0'
    assert release['targets'] == match['usable_targets']
    assert release['targets'] == validation['manifest_rows']
    assert release['targets'] == observation['targets']

    fig = plt.figure(figsize=(7.2, 8.2), facecolor='white')
    grid = fig.add_gridspec(4, 1, hspace=0.18)
    draw_funnel(fig.add_subplot(grid[0, 0]), audit, match)
    draw_release_contract(fig.add_subplot(grid[1, 0]), release, analysis)
    draw_training_products(fig.add_subplot(grid[2, 0]), release)
    draw_evaluation_products(
        fig.add_subplot(grid[3, 0]), release, validation, observation)
    output = Path(args.output)
    save_figure(fig, output)
    plt.close(fig)

    source = {
        'figure': output.name,
        'release_name': release['name'],
        'release_version': release['version'],
        'targets': release['targets'],
        'split_counts': release['split_counts'],
        'history_relations': release['temporal']['history_relations'],
        'unique_history_images': release['temporal']['unique_history_images'],
        'observation_targets': observation['targets'],
        'release_valid': release['validation']['valid'],
        'source_records': [
            'generator audit summary', 'generator target-match summary',
            'generator range analysis', 'release manifest',
            'release validation', 'observation validation',
        ],
    }
    output.with_name(output.name + '_source.json').write_text(
        json.dumps(source, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(source, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
