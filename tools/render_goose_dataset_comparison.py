#!/usr/bin/env python3
"""Render evidence-backed dataset comparison artifacts for the RAS paper."""

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FIELDS = (
    'dataset', 'citation_key', 'category', 'publication', 'year', 'environment',
    'terrain_weather', 'scenes_or_sequences', 'frames', 'split',
    'route_level_split', 'camera_count_coverage', 'lidar_and_other_sensors',
    'physical_volume_m', 'grid_shape', 'voxel_resolution_m',
    'semantic_classes', 'occupancy_construction', 'temporal_information',
    'causal_camera_histories', 'free_space_definition',
    'unknown_space_definition', 'visible_occluded_state',
    'valid_free_negatives', 'native_metrics', 'supported_model_inputs',
    'published_baseline_supervision', 'public_status', 'licence',
    'persistent_identifier', 'evidence_status', 'sources',
)

DIRECT_DATASETS = (
    'GOOSE-OCC v1.4.0',
    'WildOcc',
    'ORAD-3D',
)

CATEGORY_TITLES = {
    'A:': 'Direct off-road semantic occupancy',
    'B:': 'Adjacent unstructured-scene and source datasets',
    'C:': 'Urban occupancy protocol references',
    'D:': 'Traversability rather than semantic occupancy',
}


def load_and_validate(source_path, goose_stats_path):
    payload = json.loads(Path(source_path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported dataset comparison schema')
    policy = payload.get('comparison_policy', {})
    if policy.get('native_metrics_directly_ranked') is not False:
        raise ValueError('Native cross-dataset metrics must not be ranked')
    datasets = payload.get('datasets', [])
    if not datasets:
        raise ValueError('No dataset evidence records')
    names = set()
    for record in datasets:
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError('{} missing {}'.format(
                record.get('dataset', '<unnamed>'), ', '.join(missing)))
        if record['dataset'] in names:
            raise ValueError('Duplicate dataset {}'.format(record['dataset']))
        names.add(record['dataset'])
        if not record['sources']:
            raise ValueError('{} has no evidence sources'.format(
                record['dataset']))
        if (record['dataset'] != 'GOOSE-OCC v1.4.0' and
                record['visible_occluded_state'].lower().startswith(
                    'released ')):
            raise ValueError(
                'Only GOOSE-OCC may claim a released GOOSE-style V/O state')

    goose = next((record for record in datasets
                  if record['dataset'] == 'GOOSE-OCC v1.4.0'), None)
    if goose is None:
        raise ValueError('GOOSE-OCC record is required')
    stats = json.loads(Path(goose_stats_path).read_text(encoding='utf-8'))
    target_count = sum(stats['split']['targets'].values())
    sequence_count = sum(stats['split']['sequence_counts'].values())
    if goose['frames'] != target_count:
        raise ValueError('GOOSE frame count disagrees with machine statistics')
    if goose['scenes_or_sequences'] != sequence_count:
        raise ValueError('GOOSE sequence count disagrees with machine statistics')
    return payload


def scalar(value):
    if value is None:
        return 'not reported'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    return str(value)


def flatten(record):
    row = {field: scalar(record[field]) for field in REQUIRED_FIELDS
           if field != 'sources'}
    row['sources'] = ' | '.join(
        '{}: {}'.format(source['type'], source['locator'])
        for source in record['sources'])
    return row


def render_markdown(payload):
    lines = [
        '# Evidence-backed occupancy dataset comparison', '',
        '> Native IoU/mIoU values are not ranked across datasets because '
        'taxonomies, grids, free/unknown conventions, modalities, and '
        'supervision differ.', '',
        '| Dataset | Category | Frames | Camera | Volume / voxel | Classes | '
        'Temporal contract | Free / unknown | V/O state |',
        '| --- | --- | ---: | --- | --- | --- | --- | --- | --- |',
    ]
    for record in payload['datasets']:
        volume = '{}; {}'.format(
            record['physical_volume_m'], record['voxel_resolution_m'])
        free_unknown = '{} / {}'.format(
            record['free_space_definition'], record['unknown_space_definition'])
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            record['dataset'], record['category'], scalar(record['frames']),
            record['camera_count_coverage'], volume,
            record['semantic_classes'], record['causal_camera_histories'],
            free_unknown, record['visible_occluded_state']))
    lines.extend(['', '## Comparison boundary', '',
                  payload['comparison_policy']['reason'], '',
                  payload['comparison_policy']['claim_boundary'], ''])
    return '\n'.join(lines)


def tex_escape(value):
    replacements = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    }
    return ''.join(replacements.get(char, char) for char in scalar(value))


def tex_cite(record):
    key = record['citation_key']
    if not key:
        return tex_escape(record['dataset'])
    return '{}~\\citep{{{}}}'.format(tex_escape(record['dataset']), key)


def short_split(record):
    direct = {
        'GOOSE-OCC v1.4.0': '3,367; route-disjoint 2,877 / 490',
        'WildOcc': '10,000; within-scene 7,500 / 1,250 / 1,250',
        'ORAD-3D': '28,531; 17,250 / 3,467 / 7,814',
    }
    if record['dataset'] in direct:
        return direct[record['dataset']]
    return '{}; {}'.format(record['frames'], record['split'])


def short_grid(record):
    resolution = record['voxel_resolution_m'].replace(' isotropic', '')
    if resolution[:1].isdigit():
        resolution += ' m'
    return '{}; {}'.format(record['grid_shape'], resolution)


def short_construction(record):
    direct = {
        'GOOSE-OCC v1.4.0':
            'pose-aligned semantic LiDAR endpoints with auditable source provenance',
        'WildOcc':
            '400-frame aggregation, Poisson reconstruction, and KNN semantics',
        'ORAD-3D':
            'released sparse semantic occupancy; absent voxels are interpreted by the adapter',
    }
    return direct.get(record['dataset'], record['occupancy_construction'])


def short_state(record):
    if record['dataset'] == 'GOOSE-OCC v1.4.0':
        return ('ray-supported known free and explicit unknown; released '
                'target-visible/target-occluded valid domains')
    if record['dataset'] == 'WildOcc':
        return ('full-grid empty/non-empty prediction; no separately released '
                'unknown or V/O state')
    if record['dataset'] == 'ORAD-3D':
        return ('absent sparse voxels treated as free; raw occupied label 0 '
                'is evaluator-ambiguous; no V/O state')
    return '{}; {}'.format(
        record['unknown_space_definition'], record['visible_occluded_state'])


def render_latex(payload):
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Direct off-road semantic occupancy datasets. Native scores '
        r'are not ranked across releases because label, grid, free/unknown, '
        r'modality, and supervision contracts differ.}',
        r'\label{tab:offroad_datasets}',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{3.5pt}',
        r'\begin{tabular}{p{2.4cm}p{3.2cm}p{2.2cm}p{3.6cm}p{4.4cm}}',
        r'\toprule',
        r'Dataset & Scale and split & Grid / voxel & Label construction & '
        r'Observation and evaluation contract \\',
        r'\midrule',
    ]
    for record in payload['datasets']:
        if record['dataset'] not in DIRECT_DATASETS:
            continue
        lines.append('{} & {} & {} & {} & {}'.format(
            tex_cite(record), tex_escape(short_split(record)),
            tex_escape(short_grid(record)),
            tex_escape(short_construction(record)),
            tex_escape(short_state(record))) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])
    return '\n'.join(lines)


def render_supplement_latex(payload):
    lines = [
        r'\begin{table*}[p]',
        r'\centering',
        r'\caption{Broader dataset context. Categories separate direct off-road '
        r'semantic occupancy datasets from source/unstructured-scene datasets, '
        r'urban protocol references, and traversability datasets. Missing fields '
        r'are not inferred from neighboring releases.}',
        r'\label{tab:dataset_context}',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{3.0pt}',
        r'\begin{tabular}{p{2.6cm}p{2.3cm}p{2.6cm}p{2.4cm}p{4.8cm}}',
        r'\toprule',
        r'Dataset & Environment & Scale & Grid / voxel & Protocol distinction \\',
        r'\midrule',
    ]
    active_prefix = None
    for record in payload['datasets']:
        prefix = record['category'][:2]
        if prefix != active_prefix:
            if active_prefix is not None:
                lines.append(r'\addlinespace')
            lines.append(r'\multicolumn{5}{l}{\textit{' +
                         tex_escape(CATEGORY_TITLES[prefix]) + r'}} \\')
            active_prefix = prefix
        distinction = '{}; {}'.format(
            record['free_space_definition'], record['visible_occluded_state'])
        lines.append('{} & {} & {} & {} & {}'.format(
            tex_cite(record), tex_escape(record['environment']),
            tex_escape(short_split(record)), tex_escape(short_grid(record)),
            tex_escape(distinction)) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])
    return '\n'.join(lines)


def render(source_path, goose_stats_path, csv_path, markdown_path, latex_path,
           supplement_latex_path=None):
    payload = load_and_validate(source_path, goose_stats_path)
    rows = [flatten(record) for record in payload['datasets']]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(markdown_path).write_text(render_markdown(payload), encoding='utf-8')
    Path(latex_path).write_text(render_latex(payload), encoding='utf-8')
    if supplement_latex_path is not None:
        Path(supplement_latex_path).write_text(
            render_supplement_latex(payload), encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', default=(
        'paper/goose_occ_manuscript/dataset_comparison_sources.json'))
    parser.add_argument('--goose-statistics', default=(
        'paper/goose_occ_manuscript/generated/GOOSE_DATASET_STATISTICS.json'))
    parser.add_argument('--csv-output', default=(
        'paper/goose_occ_manuscript/dataset_comparison.csv'))
    parser.add_argument('--markdown-output', default=(
        'paper/goose_occ_manuscript/generated/DATASET_COMPARISON.md'))
    parser.add_argument('--latex-output', default=(
        'paper/goose_occ_manuscript/ras_submission/generated_dataset_comparison.tex'))
    parser.add_argument('--supplement-latex-output', default=(
        'paper/goose_occ_manuscript/ras_submission/'
        'generated_dataset_context.tex'))
    args = parser.parse_args()
    payload = render(
        args.sources, args.goose_statistics, args.csv_output,
        args.markdown_output, args.latex_output,
        args.supplement_latex_output)
    print('Rendered {} dataset comparison records'.format(
        len(payload['datasets'])))


if __name__ == '__main__':
    main()
