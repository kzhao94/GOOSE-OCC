#!/usr/bin/env python3
"""Render the frozen GOOSE-OCC observation-state protocol as a diagram."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_PROTOCOL = Path(
    '/path/to/GOOSE-OCC/observation/'
    'observation_protocol.json')
DEFAULT_OUTPUT = Path(
    'paper/goose_occ_manuscript/figures/generated/'
    'fig_goose_observation_protocol')

COLORS = {
    'free': '#D9EAD3',
    'occupied': '#F4CCCC',
    'unknown': '#E7E6E6',
    'visible': '#D9EAF7',
    'occluded': '#EADCF2',
    'neutral': '#F7F7F7',
    'edge': '#3F3F3F',
    'accent': '#C55A11',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def box(ax, xy, width, height, text, facecolor, fontsize=7,
        edgecolor=COLORS['edge'], linewidth=0.8):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle='round,pad=0.012,rounding_size=0.012',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha='center', va='center', fontsize=fontsize, linespacing=1.15)
    return patch


def arrow(ax, start, end, color=COLORS['edge'], style='-|>'):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=8,
        linewidth=0.8, color=color, shrinkA=2, shrinkB=2))


def render(protocol_path, output):
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol['format'] != 'goose_occ_observation_v1':
        raise ValueError('Unexpected observation protocol format')

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'font.size': 7,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

    left, right = axes
    left.text(0.0, 1.02, 'a', fontsize=9, fontweight='bold', va='top')
    left.text(0.06, 1.02, 'Observation evidence partitions the grid',
              fontsize=8, fontweight='bold', va='top')
    box(left, (0.06, 0.75), 0.88, 0.12,
        'Target LiDAR + recorded OCC-construction sources\n'
        '(non-target vehicle/human rays removed)',
        COLORS['neutral'], fontsize=7)
    box(left, (0.04, 0.43), 0.27, 0.17,
        'known free\nray voxels\nbefore endpoint',
        COLORS['free'], fontsize=6.3)
    box(left, (0.365, 0.43), 0.27, 0.17,
        'observed occupied\nsemantic endpoint\nIDs 1--7',
        COLORS['occupied'], fontsize=6.3)
    box(left, (0.69, 0.43), 0.27, 0.17,
        'unknown\nno ray or endpoint\nevidence',
        COLORS['unknown'], fontsize=6.3)
    arrow(left, (0.5, 0.75), (0.175, 0.60))
    arrow(left, (0.5, 0.75), (0.5, 0.60))
    arrow(left, (0.5, 0.75), (0.825, 0.60))
    box(left, (0.08, 0.15), 0.84, 0.13,
        'Endpoint evidence overrides traversed-free evidence\n'
        'unknown = not (known free or observed occupied)',
        '#FFF2CC', fontsize=7)
    left.text(0.5, 0.055,
              'The state partition is frozen before model evaluation.',
              ha='center', fontsize=6.5, color='#555555')

    right.text(0.0, 1.02, 'b', fontsize=9, fontweight='bold', va='top')
    right.text(0.06, 1.02, 'Support and camera line of sight define V/O',
               fontsize=8, fontweight='bold', va='top')
    box(right, (0.04, 0.76), 0.42, 0.12,
        'Target-LiDAR support\n+ in frustum + unobstructed LOS',
        COLORS['visible'], fontsize=7)
    box(right, (0.54, 0.76), 0.42, 0.12,
        'Non-target evidence\n+ in frustum + blocked LOS',
        COLORS['occluded'], fontsize=7)
    box(right, (0.04, 0.47), 0.19, 0.13,
        'visible\nknown free', COLORS['free'], fontsize=6.6)
    box(right, (0.27, 0.47), 0.19, 0.13,
        'visible\noccupied', COLORS['occupied'], fontsize=6.6)
    box(right, (0.54, 0.47), 0.19, 0.13,
        'occluded\nknown free', COLORS['free'], fontsize=6.6)
    box(right, (0.77, 0.47), 0.19, 0.13,
        'occluded\nobserved', COLORS['occupied'], fontsize=6.6)
    arrow(right, (0.25, 0.76), (0.135, 0.60), color='#4E79A7')
    arrow(right, (0.25, 0.76), (0.365, 0.60), color='#4E79A7')
    arrow(right, (0.75, 0.76), (0.635, 0.60), color='#8064A2')
    arrow(right, (0.75, 0.76), (0.865, 0.60), color='#8064A2')
    box(right, (0.06, 0.24), 0.40, 0.11,
        'V-valid = visible free\n| visible occupied',
        COLORS['visible'], fontsize=6.3)
    box(right, (0.54, 0.24), 0.40, 0.11,
        'O-valid = occluded free\n| occluded observed',
        COLORS['occluded'], fontsize=6.3)
    arrow(right, (0.25, 0.47), (0.25, 0.35), color='#4E79A7')
    arrow(right, (0.75, 0.47), (0.75, 0.35), color='#8064A2')
    box(right, (0.12, 0.05), 0.76, 0.10,
        'Exclude 1.5 m camera clearance and unknown voxels\n'
        'Masks remain evaluation/model-selection only',
        '#FCE4D6', fontsize=6.3, edgecolor=COLORS['accent'])

    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.04, top=0.91,
                        wspace=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
    plt.close(fig)

    source = {
        'figure': output.name,
        'protocol_format': protocol['format'],
        'protocol_sha256': sha256(protocol_path),
        'camera_clearance_radius_m': protocol['camera_clearance_radius_m'],
        'definitions': {
            key: protocol[key] for key in (
                'known_free_mask', 'observed_occupied_mask',
                'visible_occupied_mask', 'occluded_observed_mask',
                'unknown_mask', 'visible_valid', 'occluded_valid')
        },
        'claim_boundary': (
            'The diagram summarizes the frozen release protocol; masks are '
            'not training supervision.'),
    }
    output.with_name(output.name + '_source').with_suffix('.json').write_text(
        json.dumps(source, indent=2) + '\n', encoding='utf-8')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.protocol, args.output)
    print('Rendered GOOSE observation protocol figure')


if __name__ == '__main__':
    main()
