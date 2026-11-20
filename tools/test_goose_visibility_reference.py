#!/usr/bin/env python3
"""Differentially test VisibilityOcc V/O metrics against SurroundOcc."""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path(
    '/path/to/SurroundOcc/tools/generate_occupancy_goose')
DEFAULT_DATA_ROOT = Path('/path/to/GOOSE-OCC')
EVALUATION_MASKS = (
    'visible_known_free_mask',
    'visible_occupied_mask',
    'occluded_known_free_mask',
    'occluded_observed_mask',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--max-targets', type=int, default=None)
    return parser.parse_args()


def load_local_metrics():
    path = REPO_ROOT / 'loaders' / 'goose_visibility_metrics.py'
    spec = importlib.util.spec_from_file_location(
        'visibilityocc_goose_visibility_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference_metrics():
    sys.path.insert(0, str(REFERENCE_ROOT))
    import goose_observation
    import goose_visibility_metrics
    return goose_observation, goose_visibility_metrics


def observation_path(root, occ_path):
    path = Path(occ_path)
    return root / path.parent.parent / 'observation' / (path.stem + '.npz')


def compare_domain(local, reference, context):
    assert local['SC-IoU'] == reference['SC-IoU'], context
    assert local['SSC-mIoU'] == reference['SSC-mIoU'], context
    assert local['valid_voxels'] == reference['valid_voxels'], context
    assert local['frames_with_valid_voxels'] == (
        reference['frames_with_valid_voxels']), context
    assert np.array_equal(
        np.asarray(local['confusion_matrix']),
        np.asarray(reference['confusion_matrix'])), context
    assert local['gt_occupied_voxels'] == (
        reference['gt_occupied_voxels']), context
    assert local['gt_free_voxels'] == reference['gt_free_voxels'], context
    assert local['per_class'] == reference['per_class'], context


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    local_module = load_local_metrics()
    reference_observation, reference_module = load_reference_metrics()

    with (data_root / 'manifest.csv').open(newline='') as handle:
        rows = [
            row for row in csv.DictReader(handle) if row['split'] == 'val'
        ]
    if args.max_targets is not None:
        rows = rows[:args.max_targets]

    local_metric = local_module.GooseVisibilityMetrics()
    reference_metric = reference_module.GooseVisibilityMetrics()
    for index, row in enumerate(rows, start=1):
        sparse = np.load(data_root / row['occ_path'], allow_pickle=False)
        target = reference_observation.densify_occupancy(sparse)
        path = observation_path(data_root, row['occ_path'])
        reference_masks = reference_observation.load_observation(path)
        local_masks = local_module.load_observation(path)
        for name in EVALUATION_MASKS:
            assert np.array_equal(local_masks[name], reference_masks[name]), (
                row['occ_path'], name)
        local_metric.update(target, target, local_masks)
        reference_metric.update(target, target, reference_masks)
        if index % 100 == 0 or index == len(rows):
            print('checked {}/{}'.format(index, len(rows)), flush=True)

    local_result = local_metric.result()
    reference_result = reference_metric.result()
    assert local_result['frames'] == reference_result['frames'] == len(rows)
    for key in ('V-SC-IoU', 'V-SSC-mIoU', 'O-SC-IoU', 'O-SSC-mIoU'):
        assert local_result[key] == reference_result[key] == 1.0, key
    for domain in ('visible', 'occluded'):
        compare_domain(
            local_result['overall'][domain],
            reference_result['overall'][domain],
            'overall/{}'.format(domain))
    for range_name, local_domains in (
            local_result['by_horizontal_range'].items()):
        for domain in ('visible', 'occluded'):
            compare_domain(
                local_domains[domain],
                reference_result['by_horizontal_range'][range_name][domain],
                '{}/{}'.format(range_name, domain))
    print('VisibilityOcc matches SurroundOcc on {} val targets'.format(
        len(rows)))


if __name__ == '__main__':
    main()
