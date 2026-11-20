#!/usr/bin/env python3
"""Generate evidence-led GOOSE-OCC paper figures from validation CSVs.

Figure contract
---------------
Core conclusion: pose-guided supervision tests an occluded-region hypothesis;
the figure reports its domain-specific effect and variability without assuming
an improvement.
Archetype: quantitative grid with a primary O-SSC comparison and supporting
distance/seed panels.  Backend: Python/matplotlib.  Source data are the
machine-generated validation CSVs; missing method records are skipped rather
than fabricated.  Error bars are seed sample standard deviations.
"""

import argparse
import csv
import json
import math
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
    'axes.linewidth': 0.8,
    'legend.frameon': False,
})

COLORS = {
    'mono': '#4D4D4D',
    'temporal': '#3775BA',
    'occ025': '#9A4D8E',
    'occ050': '#0F4D92',
    'occ100': '#B64342',
}
METRIC_COLORS = {
    'O-SSC-mIoU': '#9A4D8E',
    'V-SSC-mIoU': '#2A9D8F',
    'SC_IoU': '#6F91B2',
}
OCCLUSION_SEED0_NAMES = (
    'goose_temporal_occlusion_common10_3h_weight025_seed0',
    'goose_temporal_occlusion_common10_3h_seed0',
    'goose_temporal_occlusion_common10_3h_weight100_seed0',
)


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value) if value not in ('', None) else ''
            except (TypeError, ValueError):
                pass
    return rows


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def row_by_name(rows):
    return {row.get('experiment'): row for row in rows}


def finite(row, key):
    value = row.get(key, '') if row else ''
    return isinstance(value, (int, float)) and math.isfinite(value)


def occlusion_weight(experiment):
    if 'weight025' in experiment:
        return 0.25
    if 'weight100' in experiment:
        return 1.0
    return 0.5


def selected_occlusion_seed0(rows):
    lookup = row_by_name(rows)
    candidates = [lookup.get(name) for name in OCCLUSION_SEED0_NAMES]
    if any(row is None or not finite(row, 'O-SSC-mIoU')
           for row in candidates):
        return None
    return max(candidates, key=lambda row: (
        row['O-SSC-mIoU'],
        -abs(occlusion_weight(row['experiment']) - 0.5),
        -occlusion_weight(row['experiment']),
    ))['experiment']


def selected_occlusion_stem(rows):
    selected = selected_occlusion_seed0(rows)
    return None if selected is None else selected[:-len('_seed0')]


def plot_weight_ablation(rows, output):
    names = [
        ('goose_temporal_common10_3h_binary_conf_seed0',
         r'$\lambda_{occ}=0$', 'temporal'),
        ('goose_temporal_occlusion_common10_3h_weight025_seed0',
         r'$\lambda_{occ}=0.25$', 'occ025'),
        ('goose_temporal_occlusion_common10_3h_seed0',
         r'$\lambda_{occ}=0.50$', 'occ050'),
        ('goose_temporal_occlusion_common10_3h_weight100_seed0',
         r'$\lambda_{occ}=1.00$', 'occ100'),
    ]
    lookup = row_by_name(rows)
    data = [(label, lookup.get(name), COLORS[color])
            for name, label, color in names if name in lookup]
    if not data:
        return False
    labels = [label for label, _, _ in data]
    metrics = ['O-SSC-mIoU', 'V-SSC-mIoU', 'SC_IoU']
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for index, metric in enumerate(metrics):
        values = [row.get(metric, np.nan) for _, row, _ in data]
        offset = (index - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric,
                      color=METRIC_COLORS[metric], alpha=0.92,
                      edgecolor='white', linewidth=0.4)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x, labels)
    ax.set_ylabel('IoU / mIoU (%)')
    finite_values = [
        row.get(metric) for _, row, _ in data for metric in metrics
        if finite(row, metric)
    ]
    if finite_values:
        ax.set_ylim(0, max(finite_values) * 1.16)
    ax.set_title('Occlusion-supervision weight ablation (seed 0)', pad=36)
    handles, legend_labels = ax.get_legend_handles_labels()
    legend_labels = [
        'SC-IoU' if label == 'SC_IoU' else label
        for label in legend_labels
    ]
    ax.legend(
        handles, legend_labels, ncol=3, fontsize=7, loc='lower center',
        bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.5)
    save(fig, output / 'fig_goose_occlusion_weight_ablation')
    return True


def plot_distance(rows, multiseed_rows, output):
    lookup = {
        row.get('variant') or row.get('run_stem'): row
        for row in multiseed_rows
    }
    method_stem = selected_occlusion_stem(rows)
    candidates = [
        ('mono', 'Mono', COLORS['mono']),
        ('goose_temporal_common10_3h_binary_conf', 'Temporal',
         COLORS['temporal']),
    ]
    if method_stem:
        candidates.append((method_stem, 'Pose-guided', COLORS['occ050']))
    bins = ('0-10m', '10-25m', '25-50m')
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8), sharex=True)
    plotted = False
    for stem, label, color in candidates:
        row = lookup.get(stem)
        if row is None:
            continue
        label = '{} (n={})'.format(label, int(row['num_seeds']))
        for ax, metric, title in zip(
                axes, ('V-SSC-mIoU', 'O-SSC-mIoU'), ('Visible', 'Occluded')):
            means = [row.get(f'{metric}_{item}_mean', np.nan) for item in bins]
            stds = [row.get(f'{metric}_{item}_std', np.nan) for item in bins]
            if not all(np.isfinite(means + stds)):
                continue
            ax.errorbar(
                bins, means, yerr=stds, marker='o', linewidth=1.8,
                markersize=4, capsize=2, color=color, label=label)
            plotted = True
            ax.set_title(title)
            ax.set_ylabel('SSC-mIoU (%)')
            ax.grid(axis='y', color='#D9D9D9', linewidth=0.5)
    if not plotted:
        plt.close(fig)
        return False
    axes[0].legend(fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, output / 'fig_goose_distance_profiles')
    return True


def plot_multiseed(rows, summary_rows, output):
    method_stem = selected_occlusion_stem(summary_rows)
    candidates = [
        ('mono', 'Mono', COLORS['mono']),
        ('goose_temporal_common10_3h_binary_conf', 'Temporal',
         COLORS['temporal']),
    ]
    if method_stem:
        candidates.append((method_stem, 'Pose-guided', COLORS['occ050']))
    lookup = {
        row.get('variant') or row.get('run_stem'): row for row in rows
    }
    data = [
        ('{} (n={})'.format(label, int(lookup[stem]['num_seeds'])),
         lookup[stem], color)
        for stem, label, color in candidates if stem in lookup
    ]
    if not data:
        return False
    metrics = ('SC_IoU_mean', 'SSC_mIoU_mean', 'V-SSC-mIoU_mean',
               'O-SSC-mIoU_mean')
    labels = ('SC', 'SSC', 'V-SSC', 'O-SSC')
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for index, (label, row, color) in enumerate(data):
        means = [row.get(metric, np.nan) for metric in metrics]
        stds = [row.get(metric.replace('_mean', '_std'), np.nan)
                for metric in metrics]
        offset = (index - (len(data) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2,
               label=label, color=color, alpha=0.9,
               edgecolor='white', linewidth=0.4)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Metric (%)')
    ax.set_title('Multi-seed mean +/- sample standard deviation')
    ax.legend(fontsize=7)
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.5)
    fig.tight_layout()
    save(fig, output / 'fig_goose_multiseed_summary')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default='work_dir/goose_experiment_summary.csv')
    parser.add_argument('--multiseed', default='work_dir/goose_experiment_multiseed_summary.csv')
    parser.add_argument(
        '--output-dir',
        default='paper/goose_occ_manuscript/figures/generated')
    args = parser.parse_args()
    output = Path(args.output_dir)
    rows = read_csv(Path(args.summary))
    multiseed = read_csv(Path(args.multiseed))
    generated = []
    if plot_weight_ablation(rows, output):
        generated.append('weight_ablation')
    if plot_distance(rows, multiseed, output):
        generated.append('distance_profiles')
    if plot_multiseed(multiseed, rows, output):
        generated.append('multiseed_summary')
    source = {
        'summary_csv': str(Path(args.summary)),
        'summary_rows': len(rows),
        'multiseed_csv': str(Path(args.multiseed)),
        'multiseed_rows': len(multiseed),
        'selected_method_seed0': selected_occlusion_seed0(rows),
        'selected_method_stem': selected_occlusion_stem(rows),
        'selection_metric': 'seed-0 O-SSC-mIoU among positive weights',
        'generated_panels': generated,
        'error_bars': (
            'sample standard deviation; n is recorded per variant in the '
            'multi-seed CSV and figure legend'),
        'distance_profiles': (
            'multi-seed mean +/- sample standard deviation from the '
            'multi-seed CSV'),
        'missing_records_policy': 'skip; never synthesize values',
    }
    (output / 'fig_goose_quantitative_source.json').write_text(
        json.dumps(source, indent=2) + '\n', encoding='utf-8')
    print('generated:', ', '.join(generated) if generated else 'none')


if __name__ == '__main__':
    main()
