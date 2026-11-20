#!/usr/bin/env python3
"""Render fail-closed GOOSE-OCC manuscript tables from final CSV/JSON data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


MONO_STEM = 'goose_mono_common10'
CONTROL_STEM = 'goose_temporal_common10_3h_binary_conf'
METHOD_STEM = 'goose_temporal_occlusion_common10_3h'
CORE_METRICS = (
    'SC_IoU',
    'SSC_mIoU',
    'V-SC-IoU',
    'V-SSC-mIoU',
    'O-SC-IoU',
    'O-SSC-mIoU',
)
VISIBILITY_METRICS = (
    'V-SC-IoU',
    'V-SSC-mIoU',
    'O-SC-IoU',
    'O-SSC-mIoU',
)
DISTANCE_BINS = ('0-10m', '10-25m', '25-50m')
CLASS_METRICS = (
    ('free', 'IoU_free'),
    ('structures', 'IoU_artificial_structures'),
    ('artificial ground', 'IoU_artificial_ground'),
    ('natural ground', 'IoU_natural_ground'),
    ('obstacle', 'IoU_obstacle'),
    ('vehicle', 'IoU_vehicle'),
    ('vegetation', 'IoU_vegetation'),
    ('human', 'IoU_human'),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--multiseed-csv',
        default='work_dir/goose_experiment_multiseed_summary.csv')
    parser.add_argument(
        '--paired-csv',
        default='work_dir/goose_experiment_paired_deltas.csv')
    parser.add_argument(
        '--external-csv',
        default='work_dir/goose_external_baseline_summary.csv')
    parser.add_argument(
        '--claim-gate', default='work_dir/goose_claim_gate.json')
    parser.add_argument(
        '--output',
        default=('paper/goose_occ_manuscript/generated/'
                 'FINAL_RESULTS_TABLES.md'))
    parser.add_argument(
        '--metadata-output',
        default=('paper/goose_occ_manuscript/generated/'
                 'FINAL_RESULTS_TABLES_source.json'))
    return parser.parse_args()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def require_one(rows, key, value, source):
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise ValueError(
            'Expected one {}={} row in {}, found {}'.format(
                key, value, source, len(matches)))
    return matches[0]


def require_int(row, key, expected, context):
    actual = int(row.get(key, -1))
    if actual != expected:
        raise ValueError(
            '{} requires {}={}, found {}'.format(
                context, key, expected, actual))


def number(row, key, context):
    value = row.get(key, '')
    if value == '':
        raise ValueError('Missing {} for {}'.format(key, context))
    return float(value)


def mean_std(row, metric, context):
    return '{:.2f} +/- {:.2f}'.format(
        number(row, '{}_mean'.format(metric), context),
        number(row, '{}_std'.format(metric), context))


def scalar(row, metric, context):
    return '{:.2f}'.format(number(row, metric, context))


def append_table(lines, headers, rows):
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def render(multiseed_path, paired_path, external_path, gate_path):
    multiseed = read_csv(multiseed_path)
    paired = read_csv(paired_path)
    external = read_csv(external_path)
    gate = json.loads(Path(gate_path).read_text(encoding='utf-8'))

    mono = require_one(multiseed, 'run_stem', MONO_STEM, multiseed_path)
    control = require_one(
        multiseed, 'run_stem', CONTROL_STEM, multiseed_path)
    method = require_one(
        multiseed, 'run_stem', METHOD_STEM, multiseed_path)
    require_int(mono, 'num_seeds', 3, 'mono aggregate')
    require_int(control, 'num_seeds', 5, 'temporal control aggregate')
    require_int(method, 'num_seeds', 5, 'pose-guided aggregate')

    paired_by_metric = {}
    for metric in CORE_METRICS:
        row = require_one(paired, 'metric', metric, paired_path)
        require_int(row, 'num_paired_seeds', 5, metric)
        if row.get('baseline_stem') != CONTROL_STEM:
            raise ValueError('Unexpected paired baseline for {}'.format(metric))
        if row.get('method_stem') != METHOD_STEM:
            raise ValueError('Unexpected paired method for {}'.format(metric))
        for seed in range(5):
            number(row, 'seed{}_delta'.format(seed), metric)
        paired_by_metric[metric] = row

    if gate.get('num_paired_seeds') != 5 or gate.get('metric') != 'O-SSC-mIoU':
        raise ValueError('Claim gate is not a complete five-seed O-SSC gate')
    gate_deltas = gate.get('seed_deltas', {})
    paired_o_ssc = paired_by_metric['O-SSC-mIoU']
    for seed in range(5):
        gate_value = gate_deltas.get(str(seed), gate_deltas.get(seed))
        csv_value = number(
            paired_o_ssc, 'seed{}_delta'.format(seed), 'O-SSC-mIoU')
        if gate_value is None or abs(float(gate_value) - csv_value) > 1e-9:
            raise ValueError('Claim gate and paired CSV disagree at seed {}'.format(seed))

    if len(external) != 1:
        raise ValueError(
            'Expected one fully supervised external baseline, found {}'.format(
                len(external)))
    upper = external[0]
    require_int(upper, 'num_samples', 469, 'SurroundOcc upper bound')
    if upper.get('experiment_role') != 'fully supervised OCC upper bound':
        raise ValueError('External baseline supervision role is not explicit')

    lines = [
        '# Final machine-generated GOOSE-OCC results',
        '',
        '> Generated only from the frozen CSV/JSON artifacts listed below. ',
        '> Weak methods use image-space supervision; SurroundOcc uses OCC GT ',
        '> and is reported only as a fully supervised upper bound.',
        '',
        '## Main comparison',
        '',
    ]
    main_rows = []
    variants = (
        ('mono', 'image-space weak', mono, True),
        ('temporal control', 'image-space weak', control, True),
        ('pose-guided', 'image-space weak', method, True),
        ('SurroundOcc', 'fully supervised OCC GT', upper, False),
    )
    for label, supervision, row, aggregate in variants:
        context = '{} main comparison'.format(label)
        values = [
            mean_std(row, metric, context) if aggregate
            else scalar(row, metric, context)
            for metric in CORE_METRICS
        ]
        main_rows.append([label, supervision] + values)
    append_table(
        lines,
        ('variant', 'training supervision') + CORE_METRICS,
        main_rows)

    lines.extend([
        '',
        '## Identical-seed pose-guided minus temporal-control deltas',
        '',
        'These five paired seeds are a variability audit, not a significance '
        'test.',
        '',
    ])
    paired_rows = []
    for metric in CORE_METRICS:
        row = paired_by_metric[metric]
        paired_rows.append([
            metric,
            *['{:+.2f}'.format(number(
                row, 'seed{}_delta'.format(seed), metric))
              for seed in range(5)],
            '{:+.2f} +/- {:.2f}'.format(
                number(row, 'delta_mean', metric),
                number(row, 'delta_std', metric)),
        ])
    append_table(
        lines,
        ('metric', 'seed 0', 'seed 1', 'seed 2', 'seed 3', 'seed 4',
         'mean +/- sample std'),
        paired_rows)

    lines.extend(['', '## Visibility metrics by horizontal range', ''])
    distance_rows = []
    for label, row in (('temporal control', control), ('pose-guided', method)):
        for distance_bin in DISTANCE_BINS:
            distance_rows.append([
                label,
                distance_bin,
                *[mean_std(
                    row, '{}_{}'.format(metric, distance_bin),
                    '{} {}'.format(label, distance_bin))
                  for metric in VISIBILITY_METRICS],
            ])
    append_table(
        lines,
        ('variant', 'range') + VISIBILITY_METRICS,
        distance_rows)

    lines.extend(['', '## Per-class IoU', ''])
    class_rows = []
    for label, row, aggregate in (
            ('mono', mono, True),
            ('temporal control', control, True),
            ('pose-guided', method, True),
            ('SurroundOcc', upper, False)):
        class_rows.append([
            label,
            *[(mean_std(row, metric, label) if aggregate
               else scalar(row, metric, label))
              for _, metric in CLASS_METRICS],
        ])
    append_table(
        lines,
        ('variant',) + tuple(label for label, _ in CLASS_METRICS),
        class_rows)

    lines.extend([
        '',
        '## Predeclared claim gate',
        '',
        '- Narrative: `{}`.'.format(gate['narrative']),
        '- Interpretation: {}.'.format(gate['interpretation'].rstrip('.')),
        '- Significance test performed: `{}`.'.format(
            str(bool(gate.get('significance_test_performed'))).lower()),
        '',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    inputs = {
        'multiseed_csv': Path(args.multiseed_csv),
        'paired_csv': Path(args.paired_csv),
        'external_csv': Path(args.external_csv),
        'claim_gate': Path(args.claim_gate),
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError('{} not found: {}'.format(label, path))

    rendered = render(
        inputs['multiseed_csv'], inputs['paired_csv'],
        inputs['external_csv'], inputs['claim_gate'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + '\n', encoding='utf-8')

    metadata = {
        'schema_version': 1,
        'output': str(output),
        'sources': {
            label: {'path': str(path), 'sha256': sha256(path)}
            for label, path in inputs.items()
        },
        'contracts': {
            'mono_seeds': 3,
            'temporal_control_seeds': 5,
            'pose_guided_seeds': 5,
            'paired_seeds': 5,
            'surroundocc_targets': 469,
        },
    }
    metadata_output = Path(args.metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('Rendered {} and {}'.format(output, metadata_output))


if __name__ == '__main__':
    main()
