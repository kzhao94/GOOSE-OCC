#!/usr/bin/env python3
"""Render LaTeX macros and a compact table from GOOSE machine statistics."""

import argparse
import json
from pathlib import Path


def comma(value):
    return '{:,}'.format(int(value))


def percent(value):
    return '{:.2f}'.format(float(value) * 100.0)


def mask_map(payload):
    return {row['mask']: row for row in payload['observation']['masks']}


def render_macros(payload):
    masks = mask_map(payload)
    total_grid = payload['grid']['all_target_grid_voxels']
    values = {
        'GooseTargets': sum(payload['split']['targets'].values()),
        'GooseTrainTargets': payload['split']['targets']['train'],
        'GooseValTargets': payload['split']['targets']['val'],
        'GooseSequences': sum(payload['split']['sequence_counts'].values()),
        'GooseTrainSequences': payload['split']['sequence_counts']['train'],
        'GooseValSequences': payload['split']['sequence_counts']['val'],
        'GooseCommonTrain':
            payload['split']['common_ten_history_targets']['train'],
        'GooseCommonVal':
            payload['split']['common_ten_history_targets']['val'],
        'GooseOccupiedVoxels': payload['occupancy']['occupied_voxels'],
        'GooseHistoryRelations': payload['temporal']['history_relations'],
        'GooseUniqueHistoryImages': payload['temporal']['unique_history_images'],
        'GooseTenHistoryTargets': sum(
            payload['split']['common_ten_history_targets'].values()),
        'GooseMetricTargetFiles': payload['metric3d']['validated_target_files'],
        'GooseMetricHistoryFiles': payload['metric3d']['validated_history_files'],
    }
    percentages = {
        'GooseKnownFreePct': masks['known_free_mask'][
            'fraction_of_all_grid_voxels'],
        'GooseObservedOccupiedPct': masks['observed_occupied_mask'][
            'fraction_of_all_grid_voxels'],
        'GooseUnknownPct': masks['unknown_mask'][
            'fraction_of_all_grid_voxels'],
        'GooseVisibleValidPct':
            payload['observation']['visible_valid_voxels'] / total_grid,
        'GooseOccludedValidPct':
            payload['observation']['occluded_valid_voxels'] / total_grid,
        'GooseSamValidPct': payload['pseudo_labels']['valid_pixel_fraction'],
        'GooseSamOverlapAccuracyPct': payload['pseudo_labels'][
            'official_overlap_covered_pixel_accuracy'],
    }
    lines = ['% Machine-generated; do not edit by hand.']
    for name, value in values.items():
        lines.append(r'\newcommand{{\{}}}{{{}}}'.format(name, comma(value)))
    for name, value in percentages.items():
        lines.append(r'\newcommand{{\{}}}{{{}}}'.format(
            name, percent(value)))
    return '\n'.join(lines) + '\n'


def render_table(payload):
    masks = mask_map(payload)
    total_grid = payload['grid']['all_target_grid_voxels']
    rows = (
        ('Known free', masks['known_free_mask']['voxels'],
         masks['known_free_mask']['fraction_of_all_grid_voxels']),
        ('Observed occupied', masks['observed_occupied_mask']['voxels'],
         masks['observed_occupied_mask']['fraction_of_all_grid_voxels']),
        ('Unknown', masks['unknown_mask']['voxels'],
         masks['unknown_mask']['fraction_of_all_grid_voxels']),
        ('V-valid', payload['observation']['visible_valid_voxels'],
         payload['observation']['visible_valid_voxels'] / total_grid),
        ('O-valid', payload['observation']['occluded_valid_voxels'],
         payload['observation']['occluded_valid_voxels'] / total_grid),
    )
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{GOOSE-OCC observation-state coverage over all '
        r'\GooseTargets{} targets. V-valid and O-valid are evaluation domains '
        r'and overlap the state partition rather than forming additional states.}',
        r'\label{tab:goose_observation_statistics}',
        r'\small',
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r'Domain & Voxels & Grid share (\%) \\',
        r'\midrule',
    ]
    for label, voxels, share in rows:
        lines.append('{} & {} & {:.2f} \\\\'.format(
            label, comma(voxels), share * 100.0))
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(lines)


def render(statistics_path, macros_path, table_path):
    payload = json.loads(Path(statistics_path).read_text(encoding='utf-8'))
    if not payload['quality']['route_split_disjoint']:
        raise ValueError('Route-disjoint split check failed')
    if not payload['observation']['full_grid_partition_valid']:
        raise ValueError('Observation-state partition check failed')
    Path(macros_path).write_text(render_macros(payload), encoding='utf-8')
    Path(table_path).write_text(render_table(payload), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics', default=(
        'paper/goose_occ_manuscript/generated/GOOSE_DATASET_STATISTICS.json'))
    parser.add_argument('--macros-output', default=(
        'paper/goose_occ_manuscript/ras_submission/'
        'generated_dataset_statistics_macros.tex'))
    parser.add_argument('--table-output', default=(
        'paper/goose_occ_manuscript/ras_submission/'
        'generated_dataset_statistics_table.tex'))
    args = parser.parse_args()
    render(args.statistics, args.macros_output, args.table_output)
    print('Rendered GOOSE dataset statistics LaTeX artifacts')


if __name__ == '__main__':
    main()
