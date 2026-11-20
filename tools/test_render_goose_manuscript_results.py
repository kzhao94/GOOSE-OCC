#!/usr/bin/env python3
"""Contract tests for final GOOSE-OCC manuscript table rendering."""

import csv
import json
import tempfile
from pathlib import Path

from render_goose_manuscript_results import (
    CLASS_METRICS,
    CONTROL_STEM,
    CORE_METRICS,
    DISTANCE_BINS,
    METHOD_STEM,
    MONO_STEM,
    VISIBILITY_METRICS,
    render,
)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(stem, seeds):
    row = {'run_stem': stem, 'num_seeds': seeds}
    metrics = list(CORE_METRICS)
    metrics.extend(
        '{}_{}'.format(metric, distance_bin)
        for distance_bin in DISTANCE_BINS
        for metric in VISIBILITY_METRICS)
    metrics.extend(metric for _, metric in CLASS_METRICS)
    for index, metric in enumerate(metrics):
        row['{}_mean'.format(metric)] = 1.0 + index
        row['{}_std'.format(metric)] = 0.1
    return row


def paired(metric):
    row = {
        'baseline_stem': CONTROL_STEM,
        'method_stem': METHOD_STEM,
        'metric': metric,
        'num_paired_seeds': 5,
        'delta_mean': 0.1,
        'delta_std': 0.2,
    }
    for seed in range(5):
        row['seed{}_delta'.format(seed)] = 0.1
    return row


def external():
    row = {
        'experiment_role': 'fully supervised OCC upper bound',
        'num_samples': 469,
    }
    for metric in CORE_METRICS:
        row[metric] = 10.0
    for _, metric in CLASS_METRICS:
        row[metric] = 10.0
    return row


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        multiseed = root / 'multiseed.csv'
        paired_path = root / 'paired.csv'
        external_path = root / 'external.csv'
        gate_path = root / 'gate.json'
        write_csv(multiseed, [
            aggregate(MONO_STEM, 3),
            aggregate(CONTROL_STEM, 5),
            aggregate(METHOD_STEM, 5),
        ])
        write_csv(paired_path, [paired(metric) for metric in CORE_METRICS])
        write_csv(external_path, [external()])
        gate_path.write_text(json.dumps({
            'metric': 'O-SSC-mIoU',
            'num_paired_seeds': 5,
            'seed_deltas': {str(seed): 0.1 for seed in range(5)},
            'narrative': 'method-led stable candidate',
            'interpretation': 'Synthetic contract test.',
            'significance_test_performed': False,
        }), encoding='utf-8')

        output = render(multiseed, paired_path, external_path, gate_path)
        assert 'fully supervised OCC GT' in output
        assert 'mean +/- sample std' in output
        assert 'method-led stable candidate' in output

        rows = read_rows(multiseed)
        rows[1]['num_seeds'] = 4
        write_csv(multiseed, rows)
        try:
            render(multiseed, paired_path, external_path, gate_path)
        except ValueError as error:
            assert 'num_seeds=5' in str(error)
        else:
            raise AssertionError('Incomplete temporal control did not fail')

    print('GOOSE manuscript-result renderer tests passed')


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


if __name__ == '__main__':
    main()
