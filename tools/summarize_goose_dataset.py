#!/usr/bin/env python3
"""Generate paper-facing statistics from the frozen GOOSE-OCC release.

The script is read-only with respect to the dataset.  It treats release
manifests and validation JSON files as the contract, and scans the sparse OCC
arrays only for statistics that are not present in those manifests: per-frame
class presence and horizontal-range class counts.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path('/path/to/GOOSE-OCC')
DEFAULT_OUTPUT = Path('paper/goose_occ_manuscript/generated')
DEFAULT_FIGURES = Path('paper/goose_occ_manuscript/figures/generated')
RANGE_BINS = ((0.0, 10.0), (10.0, 25.0), (25.0, 50.0))
QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)


mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 8,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'legend.frameon': False,
})


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        '{:g}'.format(q): float(np.quantile(array, q)) for q in QUANTILES
    }


def source_record(root, relative):
    path = root / relative
    return {
        'path': str(path),
        'relative_to_dataset_root': str(relative),
        'bytes': path.stat().st_size,
        'sha256': sha256(path),
    }


def scan_occupancy(root, rows, metadata):
    class_ids = {
        int(value): name for name, value
        in metadata['stored_class_ids'].items() if int(value) != 0
    }
    class_voxels = Counter()
    class_frames = Counter()
    distance_counts = {
        class_id: Counter() for class_id in sorted(class_ids)
    }
    outside_50 = Counter()
    occupied_per_frame = []
    pc_range = np.asarray(metadata['pc_range'], dtype=np.float64)
    voxel_size = float(metadata['voxel_size'])

    for row in rows:
        array = np.load(root / row['occ_path'], allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError('Invalid sparse OCC shape for {}'.format(
                row['occ_path']))
        if array.dtype != np.int32:
            raise ValueError('Invalid sparse OCC dtype for {}'.format(
                row['occ_path']))
        semantics = array[:, 3]
        if not set(np.unique(semantics)).issubset(class_ids):
            raise ValueError('Unexpected semantic ID in {}'.format(
                row['occ_path']))
        occupied_per_frame.append(int(array.shape[0]))
        x = pc_range[0] + (array[:, 0].astype(np.float64) + 0.5) * voxel_size
        y = pc_range[1] + (array[:, 1].astype(np.float64) + 0.5) * voxel_size
        radius = np.sqrt(x * x + y * y)
        for class_id in class_ids:
            mask = semantics == class_id
            count = int(mask.sum())
            if not count:
                continue
            class_voxels[class_id] += count
            class_frames[class_id] += 1
            class_radius = radius[mask]
            for lower, upper in RANGE_BINS:
                key = '{}-{}m'.format(int(lower), int(upper))
                distance_counts[class_id][key] += int(
                    ((class_radius >= lower) & (class_radius < upper)).sum())
            outside_50[class_id] += int((class_radius >= 50.0).sum())

    return {
        'class_ids': class_ids,
        'class_voxels': class_voxels,
        'class_frames': class_frames,
        'distance_counts': distance_counts,
        'outside_50': outside_50,
        'occupied_per_frame': occupied_per_frame,
    }


def collect(root):
    root = Path(root)
    metadata = read_json(root / 'metadata.json')
    validation = read_json(root / 'validation.json')
    observation_validation = read_json(
        root / 'observation/observation_validation.json')
    temporal_validation = read_json(root / 'temporal/temporal_validation.json')
    pseudo_validation = read_json(
        root / 'temporal/pseudo_label_validation.json')
    pseudo_audit = read_json(root / 'temporal/pseudo_label_audit.json')
    metric_validation = read_json(root / 'metric3d/metric3d_validation.json')
    manifest = read_csv(root / 'manifest.csv')
    observation_rows = read_csv(
        root / 'observation/observation_manifest.csv')
    temporal_targets = read_csv(root / 'temporal/temporal_targets.csv')
    temporal_rows = read_csv(root / 'temporal/temporal_manifest.csv')
    pseudo_rows = read_csv(root / 'temporal/pseudo_label_manifest.csv')

    expected_targets = int(metadata['target_count'])
    if len(manifest) != expected_targets:
        raise ValueError('Manifest target count does not match metadata')
    if len(observation_rows) != expected_targets:
        raise ValueError('Observation manifest is incomplete')
    if len(temporal_targets) != expected_targets:
        raise ValueError('Temporal target manifest is incomplete')

    sequences = defaultdict(Counter)
    for row in manifest:
        sequences[row['split']][row['sequence']] += 1
    train_sequences = set(sequences['train'])
    val_sequences = set(sequences['val'])
    overlap = sorted(train_sequences & val_sequences)
    if overlap:
        raise ValueError('Sequences cross train/val: {}'.format(overlap))

    occ = scan_occupancy(root, manifest, metadata)
    scanned_total = sum(occ['class_voxels'].values())
    release_total = sum(metadata['occupied_voxel_class_counts'].values())
    if scanned_total != release_total:
        raise ValueError(
            'Sparse OCC scan disagrees with release class totals: {} != {}'.format(
                scanned_total, release_total))

    class_statistics = []
    for class_id, name in sorted(occ['class_ids'].items()):
        voxels = int(occ['class_voxels'][class_id])
        release_value = int(metadata['occupied_voxel_class_counts'][name])
        if voxels != release_value:
            raise ValueError('Class total mismatch for {}'.format(name))
        class_statistics.append({
            'class_id': class_id,
            'class_name': name,
            'occupied_voxels': voxels,
            'occupied_fraction': voxels / scanned_total,
            'frames_present': int(occ['class_frames'][class_id]),
            'frame_fraction': occ['class_frames'][class_id] / expected_targets,
            'range_counts': dict(occ['distance_counts'][class_id]),
            'range_outside_50m': int(occ['outside_50'][class_id]),
        })

    mask_names = (
        'known_free_mask', 'observed_occupied_mask', 'unknown_mask',
        'visible_known_free_mask', 'visible_occupied_mask',
        'occluded_known_free_mask', 'occluded_observed_mask',
        'target_free_mask', 'target_occupied_support_mask',
        'non_target_free_mask', 'non_target_occupied_support_mask',
        'camera_clearance_mask',
    )
    mask_totals = Counter()
    mask_per_frame = {name: [] for name in mask_names}
    empty_visible = 0
    empty_occluded = 0
    for row in observation_rows:
        for name in mask_names:
            value = int(row[name])
            mask_totals[name] += value
            mask_per_frame[name].append(value)
        if int(row['visible_known_free_mask']) + int(
                row['visible_occupied_mask']) == 0:
            empty_visible += 1
        if int(row['occluded_known_free_mask']) + int(
                row['occluded_observed_mask']) == 0:
            empty_occluded += 1

    occ_size = metadata['occ_size']
    voxels_per_grid = int(np.prod(occ_size))
    all_grid_voxels = expected_targets * voxels_per_grid
    state_sum = (
        mask_totals['known_free_mask']
        + mask_totals['observed_occupied_mask']
        + mask_totals['unknown_mask'])
    if state_sum != all_grid_voxels:
        raise ValueError('Observation states do not partition the full grid')
    for name, expected in observation_validation['mask_voxel_totals'].items():
        if name in mask_totals and int(expected) != mask_totals[name]:
            raise ValueError('Observation total mismatch for {}'.format(name))

    observation_statistics = []
    for name in mask_names:
        observation_statistics.append({
            'mask': name,
            'voxels': int(mask_totals[name]),
            'fraction_of_all_grid_voxels': mask_totals[name] / all_grid_voxels,
            'per_frame_quantiles': quantiles(mask_per_frame[name]),
            'empty_frames': int(sum(value == 0 for value in mask_per_frame[name])),
        })

    history_counts = Counter(
        int(row['history_count']) for row in temporal_targets)
    common10_by_split = Counter()
    for row in temporal_targets:
        if int(row['history_count']) == 10:
            common10_by_split[row['split']] += 1
    lag_values = [float(row['actual_lag_ms']) for row in temporal_rows]
    selection_errors = [
        abs(float(row['selection_error_ms'])) for row in temporal_rows]
    valid_fractions = [float(row['valid_fraction']) for row in pseudo_rows]
    mean_confidences = [float(row['mean_confidence']) for row in pseudo_rows]

    sequence_rows = []
    for split in ('train', 'val'):
        for sequence, count in sorted(sequences[split].items()):
            sequence_rows.append({
                'split': split,
                'sequence': sequence,
                'targets': count,
            })

    source_paths = (
        'metadata.json', 'validation.json', 'manifest.csv',
        'observation/observation_manifest.csv',
        'observation/observation_protocol.json',
        'observation/observation_validation.json',
        'temporal/temporal_targets.csv',
        'temporal/temporal_manifest.csv',
        'temporal/temporal_validation.json',
        'temporal/pseudo_label_manifest.csv',
        'temporal/pseudo_label_validation.json',
        'temporal/pseudo_label_audit.json',
        'metric3d/metric3d_validation.json',
    )
    return {
        'schema_version': 1,
        'dataset': 'GOOSE-OCC',
        'release_version': (root / 'VERSION').read_text().strip(),
        'dataset_root': str(root),
        'grid': {
            'pc_range_m': metadata['pc_range'],
            'voxel_size_m': metadata['voxel_size'],
            'occ_size': occ_size,
            'voxels_per_grid': voxels_per_grid,
            'all_target_grid_voxels': all_grid_voxels,
        },
        'split': {
            'targets': dict(Counter(row['split'] for row in manifest)),
            'sequence_counts': {
                split: len(sequences[split]) for split in ('train', 'val')
            },
            'route_overlap': overlap,
            'sequences': sequence_rows,
            'common_ten_history_targets': dict(common10_by_split),
        },
        'occupancy': {
            'occupied_voxels': scanned_total,
            'per_frame_quantiles': quantiles(occ['occupied_per_frame']),
            'classes': class_statistics,
        },
        'observation': {
            'masks': observation_statistics,
            'visible_valid_voxels': int(
                mask_totals['visible_known_free_mask']
                + mask_totals['visible_occupied_mask']),
            'occluded_valid_voxels': int(
                mask_totals['occluded_known_free_mask']
                + mask_totals['occluded_observed_mask']),
            'empty_visible_valid_frames': empty_visible,
            'empty_occluded_valid_frames': empty_occluded,
            'full_grid_partition_valid': True,
        },
        'temporal': {
            'targets': len(temporal_targets),
            'history_relations': len(temporal_rows),
            'unique_history_images': temporal_validation['unique_history_images'],
            'history_count_distribution': {
                str(key): value for key, value in sorted(history_counts.items())
            },
            'actual_lag_ms_quantiles': quantiles(lag_values),
            'absolute_selection_error_ms_quantiles': quantiles(selection_errors),
            'pose_status': temporal_validation['pose_status'],
            'anchor_status': temporal_validation['anchor_status'],
            'excluded_history_relations': temporal_validation[
                'excluded_history_relations'],
        },
        'pseudo_labels': {
            'unique_images': pseudo_validation['manifest_rows'],
            'valid_pixel_fraction': pseudo_validation['valid_pixel_fraction'],
            'mean_valid_confidence_uint8': pseudo_validation[
                'mean_valid_confidence'],
            'per_image_valid_fraction_quantiles': quantiles(valid_fractions),
            'per_image_mean_confidence_quantiles': quantiles(mean_confidences),
            'official_overlap_audit_frames': pseudo_audit['audited_overlap'],
            'official_overlap_coverage': pseudo_audit['overall']['coverage'],
            'official_overlap_covered_pixel_accuracy': pseudo_audit[
                'overall']['covered_pixel_accuracy'],
            'official_overlap_penalized_miou': pseudo_audit[
                'overall']['penalized_miou'],
        },
        'metric3d': {
            'validated_target_files': metric_validation[
                'validated_files']['target'],
            'validated_history_files': metric_validation[
                'validated_files']['history'],
            'positive_pixel_fraction': metric_validation[
                'positive_pixel_fraction'],
            'mean_positive_depth_m': metric_validation[
                'mean_positive_depth_m'],
            'target_lidar_scale_audit': metric_validation[
                'target_lidar_scale_audit'],
        },
        'quality': {
            'release_validation_valid': validation['valid'],
            'observation_validation_valid': observation_validation['valid'],
            'temporal_validation_valid': temporal_validation['valid'],
            'pseudo_label_validation_valid': pseudo_validation['valid'],
            'metric3d_validation_valid': metric_validation['valid'],
            'missing_occ_files': len(validation.get('errors', [])),
            'route_split_disjoint': not overlap,
        },
        'sources': [source_record(root, relative) for relative in source_paths],
    }


def write_csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Cannot write empty CSV {}'.format(path))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(payload):
    lines = [
        '# GOOSE-OCC v{} dataset statistics'.format(
            payload['release_version']), '',
        '> Machine-generated from the frozen release manifests and sparse OCC '
        'arrays. Observation masks are evaluation-only.', '',
        '## Release and split', '',
        '| split | targets | sequences | common 10-history targets |',
        '| --- | ---: | ---: | ---: |',
    ]
    for split in ('train', 'val'):
        lines.append('| {} | {} | {} | {} |'.format(
            split,
            payload['split']['targets'][split],
            payload['split']['sequence_counts'][split],
            payload['split']['common_ten_history_targets'][split]))
    lines.extend(['', '## Occupied semantic distribution', '',
                  '| ID | class | voxels | occupied share | frames |',
                  '| ---: | --- | ---: | ---: | ---: |'])
    for row in payload['occupancy']['classes']:
        lines.append('| {class_id} | {class_name} | {occupied_voxels} | '
                     '{occupied_fraction:.4%} | {frames_present} |'.format(
                         **row))
    lines.extend(['', '## Observation-state totals', '',
                  '| mask | voxels | full-grid share | empty frames |',
                  '| --- | ---: | ---: | ---: |'])
    for row in payload['observation']['masks']:
        lines.append('| {mask} | {voxels} | '
                     '{fraction_of_all_grid_voxels:.4%} | '
                     '{empty_frames} |'.format(**row))
    lines.extend([
        '', '## Temporal and pseudo-label release', '',
        '- History relations: {:,}'.format(
            payload['temporal']['history_relations']),
        '- Unique history images: {:,}'.format(
            payload['temporal']['unique_history_images']),
        '- Targets with ten histories: {:,}'.format(sum(
            payload['split']['common_ten_history_targets'].values())),
        '- SAM3 valid-pixel fraction: {:.2%}'.format(
            payload['pseudo_labels']['valid_pixel_fraction']),
        '- SAM3 official-overlap covered-pixel accuracy: {:.2%}'.format(
            payload['pseudo_labels'][
                'official_overlap_covered_pixel_accuracy']),
        '- Metric3D target/history files: {:,}/{:,}'.format(
            payload['metric3d']['validated_target_files'],
            payload['metric3d']['validated_history_files']),
        '', '## Contract checks', '',
        '- Route split disjoint: {}'.format(
            payload['quality']['route_split_disjoint']),
        '- Full observation-state partition: {}'.format(
            payload['observation']['full_grid_partition_valid']),
        '- All release validators passed: {}'.format(all(
            value for key, value in payload['quality'].items()
            if key.endswith('_valid'))),
    ])
    return '\n'.join(lines) + '\n'


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def plot_statistics(payload, output):
    classes = payload['occupancy']['classes']
    names = [row['class_name'].replace('_', ' ') for row in classes]
    values = [row['occupied_voxels'] for row in classes]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(names, values, color='#4C78A8')
    ax.set_yscale('log')
    ax.set_ylabel('Occupied voxels (log scale)')
    ax.tick_params(axis='x', rotation=28)
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output / 'fig_goose_class_distribution')

    masks = {row['mask']: row for row in payload['observation']['masks']}
    labels = ('Known free', 'Observed occupied', 'Unknown',
              'Visible valid', 'Occluded valid')
    totals = (
        masks['known_free_mask']['voxels'],
        masks['observed_occupied_mask']['voxels'],
        masks['unknown_mask']['voxels'],
        payload['observation']['visible_valid_voxels'],
        payload['observation']['occluded_valid_voxels'],
    )
    denominator = payload['grid']['all_target_grid_voxels']
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.bar(labels, np.asarray(totals) / denominator * 100.0,
                  color=('#59A14F', '#E15759', '#BAB0AC', '#4E79A7', '#B07AA1'))
    ax.set_ylabel('Share of all target-grid voxels (%)')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.5)
    for bar, value in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                '{:.1f}%'.format(value / denominator * 100.0),
                ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    save_figure(fig, output / 'fig_goose_observation_statistics')

    distribution = payload['temporal']['history_count_distribution']
    x = np.arange(11)
    y = [distribution.get(str(value), 0) for value in x]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    axes[0].bar(x, y, color='#4C78A8')
    axes[0].set_xlabel('Valid causal histories per target')
    axes[0].set_ylabel('Targets')
    axes[0].set_xticks(x)
    lag_quantiles = payload['temporal']['actual_lag_ms_quantiles']
    q_labels = ('p01', 'p05', 'p50', 'p95', 'p99')
    q_values = [lag_quantiles[key] / 1000.0
                for key in ('0.01', '0.05', '0.5', '0.95', '0.99')]
    axes[1].plot(q_labels, q_values, marker='o', color='#F28E2B')
    axes[1].set_ylabel('Actual causal lag (s)')
    axes[1].set_xlabel('Relation quantile')
    for ax in axes:
        ax.grid(axis='y', color='#D9D9D9', linewidth=0.5)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, output / 'fig_goose_temporal_statistics')


def write_outputs(payload, output, figures):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'GOOSE_DATASET_STATISTICS.json').write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    (output / 'GOOSE_DATASET_STATISTICS.md').write_text(
        render_markdown(payload), encoding='utf-8')
    write_csv_file(
        output / 'GOOSE_SEQUENCE_STATISTICS.csv',
        payload['split']['sequences'])
    class_rows = []
    for row in payload['occupancy']['classes']:
        flat = {key: value for key, value in row.items()
                if key != 'range_counts'}
        for key, value in row['range_counts'].items():
            flat['voxels_{}'.format(key)] = value
        class_rows.append(flat)
    write_csv_file(output / 'GOOSE_CLASS_STATISTICS.csv', class_rows)
    observation_rows = []
    for row in payload['observation']['masks']:
        flat = {key: value for key, value in row.items()
                if key != 'per_frame_quantiles'}
        for key, value in row['per_frame_quantiles'].items():
            flat['per_frame_q{}'.format(key)] = value
        observation_rows.append(flat)
    write_csv_file(
        output / 'GOOSE_OBSERVATION_STATISTICS.csv', observation_rows)
    plot_statistics(payload, figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--figure-dir', type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    payload = collect(args.dataset_root)
    write_outputs(payload, args.output_dir, args.figure_dir)
    print(json.dumps({
        'targets': sum(payload['split']['targets'].values()),
        'sequences': sum(payload['split']['sequence_counts'].values()),
        'occupied_voxels': payload['occupancy']['occupied_voxels'],
        'history_relations': payload['temporal']['history_relations'],
        'output': str(args.output_dir),
    }, indent=2))


if __name__ == '__main__':
    main()
