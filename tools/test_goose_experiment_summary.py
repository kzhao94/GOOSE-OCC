#!/usr/bin/env python3
"""Regression tests for GOOSE result selection and statistical summaries."""

import json
import statistics
import tempfile
from pathlib import Path

from summarize_goose_experiments import (
    AGGREGATE_METRICS,
    build_multiseed_records,
    build_paired_delta_records,
    load_records,
    load_surroundocc_upper_bound,
)
from plot_goose_paper_figures import selected_occlusion_seed0


def record(experiment, o_ssc, offset=0.0):
    result = {'experiment': experiment, 'O-SSC-mIoU': o_ssc}
    for index, metric in enumerate(AGGREGATE_METRICS):
        result[metric] = offset + index
    result['O-SSC-mIoU'] = o_ssc
    return result


def test_seed0_selection_and_paired_deltas():
    records = []
    weights = (
        ('goose_temporal_occlusion_common10_3h_weight025', 9.0),
        ('goose_temporal_occlusion_common10_3h', 10.0),
        ('goose_temporal_occlusion_common10_3h_weight100', 10.0),
    )
    for stem, score in weights:
        records.append(record(stem + '_seed0', score, score))

    baseline_stem = 'goose_temporal_common10_3h_binary_conf'
    selected_stem = 'goose_temporal_occlusion_common10_3h'
    for seed in (0, 1, 2, 3, 4):
        if seed:
            records.append(record(
                '{}_seed{}'.format(selected_stem, seed), 10.0 + seed,
                10.0 + seed))
        records.append(record(
            '{}_seed{}'.format(baseline_stem, seed), 8.0 + seed,
            8.0 + seed))

    paired = build_paired_delta_records(records)
    assert paired
    assert {row['method_stem'] for row in paired} == {selected_stem}
    sc_row = next(row for row in paired if row['metric'] == 'SC_IoU')
    assert tuple(
        sc_row['seed{}_delta'.format(seed)] for seed in range(5)
    ) == (2.0, 2.0, 2.0, 2.0, 2.0)
    assert sc_row['delta_mean'] == 2.0
    assert sc_row['delta_std'] == 0.0

    aggregates = build_multiseed_records(records)
    selected = next(
        row for row in aggregates if row['run_stem'] == selected_stem)
    assert selected['num_seeds'] == 5
    expected = statistics.stdev((10.0, 11.0, 12.0, 13.0, 14.0))
    assert selected['SC_IoU_std'] == expected


def test_incomplete_five_seed_group_is_not_reported():
    records = []
    for stem, score in (
            ('goose_temporal_occlusion_common10_3h_weight025', 9.0),
            ('goose_temporal_occlusion_common10_3h', 10.0),
            ('goose_temporal_occlusion_common10_3h_weight100', 9.5)):
        records.append(record(stem + '_seed0', score, score))
    for seed in (0, 1, 2, 3):
        records.append(record(
            'goose_temporal_common10_3h_binary_conf_seed{}'.format(seed),
            8.0 + seed, 8.0 + seed))
        if seed:
            records.append(record(
                'goose_temporal_occlusion_common10_3h_seed{}'.format(seed),
                9.0 + seed, 9.0 + seed))
    aggregates = build_multiseed_records(records)
    assert not any(record['run_stem'] in {
        'goose_temporal_common10_3h_binary_conf',
        'goose_temporal_occlusion_common10_3h',
    } for record in aggregates)
    assert build_paired_delta_records(records) == []


def test_incomplete_weights_do_not_select():
    records = [
        record('goose_temporal_occlusion_common10_3h_weight025_seed0', 9.0),
        record('goose_temporal_occlusion_common10_3h_seed0', 10.0),
    ]
    assert build_paired_delta_records(records) == []
    assert selected_occlusion_seed0(records) is None


def test_plot_selection_matches_predeclared_tie_break():
    records = [
        record('goose_temporal_occlusion_common10_3h_weight025_seed0', 9.0),
        record('goose_temporal_occlusion_common10_3h_seed0', 10.0),
        record('goose_temporal_occlusion_common10_3h_weight100_seed0', 10.0),
    ]
    assert selected_occlusion_seed0(records) == (
        'goose_temporal_occlusion_common10_3h_seed0')


def test_val_best_is_preferred():
    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir) / 'run'
        run_dir.mkdir()
        payload = {'metrics': {'O-SSC-mIoU': 1.0}}
        (run_dir / 'val_qualitative.json').write_text(json.dumps(payload))
        payload['metrics']['O-SSC-mIoU'] = 2.0
        (run_dir / 'val_best.json').write_text(json.dumps(payload))
        records = load_records(temp_dir)
        assert len(records) == 1
        assert records[0]['O-SSC-mIoU'] == 2.0
        assert records[0]['result_json'].endswith('val_best.json')


def test_surroundocc_requires_formal_contract():
    class_names = (
        'free', 'artificial_structures', 'artificial_ground',
        'natural_ground', 'obstacle', 'vehicle', 'vegetation', 'human',
    )
    domain = {'SC-IoU': 0.3, 'SSC-mIoU': 0.15}
    payload = {
        'num_samples': 469,
        'dataset_samples': 469,
        'prediction_contract': {
            'shape': [200, 128, 32],
            'axis_order': 'x,y,z',
            'class_ids': '0..7',
        },
        'standard_unknown_as_free': {
            'SC-IoU': 0.2,
            'SSC-mIoU': 0.1,
            'per_class': {
                name: {'iou': 0.01 * class_id}
                for class_id, name in enumerate(class_names)
            },
        },
        'visibility': {
            'V-SC-IoU': 0.3,
            'V-SSC-mIoU': 0.15,
            'O-SC-IoU': 0.4,
            'O-SSC-mIoU': 0.2,
            'by_horizontal_range': {
                name: {'visible': domain, 'occluded': domain}
                for name in ('0-10m', '10-25m', '25-50m')
            },
        },
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / 'result.json'
        path.write_text(json.dumps(payload))
        records = load_surroundocc_upper_bound(path)
        assert len(records) == 1
        assert records[0]['SC_IoU'] == 20.0
        assert records[0]['O-SSC-mIoU'] == 20.0
        assert records[0]['O-SC-IoU_25-50m'] == 30.0
        payload['num_samples'] = 468
        path.write_text(json.dumps(payload))
        try:
            load_surroundocc_upper_bound(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Non-469 external result was accepted')


if __name__ == '__main__':
    test_seed0_selection_and_paired_deltas()
    test_incomplete_five_seed_group_is_not_reported()
    test_incomplete_weights_do_not_select()
    test_plot_selection_matches_predeclared_tie_break()
    test_val_best_is_preferred()
    test_surroundocc_requires_formal_contract()
    print('GOOSE experiment summary tests passed')
