#!/usr/bin/env python3
"""Synthetic contract tests for GOOSE visible/occluded metrics."""

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / 'loaders' / 'goose_visibility_metrics.py'
SPEC = importlib.util.spec_from_file_location(
    'visibilityocc_goose_visibility_metrics_test', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
GooseVisibilityMetrics = MODULE.GooseVisibilityMetrics
OCC_SIZE = MODULE.OCC_SIZE


def masks():
    result = {
        'visible_known_free_mask': np.zeros(OCC_SIZE, dtype=bool),
        'visible_occupied_mask': np.zeros(OCC_SIZE, dtype=bool),
        'occluded_known_free_mask': np.zeros(OCC_SIZE, dtype=bool),
        'occluded_observed_mask': np.zeros(OCC_SIZE, dtype=bool),
    }
    result['visible_known_free_mask'][1, 1, 1] = True
    result['visible_occupied_mask'][2, 1, 1] = True
    result['occluded_known_free_mask'][3, 1, 1] = True
    result['occluded_observed_mask'][4, 1, 1] = True
    return result


def main():
    target = np.zeros(OCC_SIZE, dtype=np.uint8)
    target[2, 1, 1] = 3
    target[4, 1, 1] = 6

    perfect = GooseVisibilityMetrics()
    perfect.update(target.copy(), target, masks())
    perfect_result = perfect.result()
    for key in ('V-SC-IoU', 'V-SSC-mIoU', 'O-SC-IoU', 'O-SSC-mIoU'):
        assert perfect_result[key] == 1.0, (key, perfect_result[key])

    false_positive = target.copy()
    false_positive[1, 1, 1] = 3
    penalized = GooseVisibilityMetrics()
    penalized.update(false_positive, target, masks())
    penalized_result = penalized.result()
    assert penalized_result['V-SC-IoU'] < 1.0
    assert penalized_result['V-SSC-mIoU'] < 1.0

    occluded_false_positive = target.copy()
    occluded_false_positive[3, 1, 1] = 6
    occluded_penalized = GooseVisibilityMetrics()
    occluded_penalized.update(
        occluded_false_positive, target, masks())
    assert occluded_penalized.result()['O-SC-IoU'] < 1.0

    empty_masks = {name: np.zeros(OCC_SIZE, dtype=bool) for name in masks()}
    empty = GooseVisibilityMetrics()
    empty.update(target, target, empty_masks)
    empty_result = empty.result()
    assert empty_result['V-SC-IoU'] is None
    assert empty_result['O-SC-IoU'] is None
    assert empty_result['overall']['visible']['valid_voxels'] == 0
    assert empty_result['overall']['occluded']['valid_voxels'] == 0

    invalid_prediction = target.copy()
    invalid_prediction[2, 1, 1] = 8
    try:
        GooseVisibilityMetrics().update(
            invalid_prediction, target, masks())
    except ValueError as error:
        assert 'outside [0, 7]' in str(error)
    else:
        raise AssertionError('Invalid prediction ID was accepted')

    invalid_target = target.copy()
    invalid_target[2, 1, 1] = 255
    try:
        GooseVisibilityMetrics().update(
            target, invalid_target, masks())
    except ValueError as error:
        assert 'outside [0, 7]' in str(error)
    else:
        raise AssertionError('Invalid target ID was accepted')

    try:
        GooseVisibilityMetrics().update(
            target.transpose(1, 0, 2), target, masks())
    except ValueError as error:
        assert 'shape' in str(error)
    else:
        raise AssertionError('Axis-permuted prediction was accepted')
    print('perfect prediction metrics: 1.0')
    print('visible free false-positive SC-IoU:',
          penalized_result['V-SC-IoU'])
    print('GOOSE visibility metric tests passed')


if __name__ == '__main__':
    main()
