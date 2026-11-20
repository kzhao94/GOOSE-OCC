#!/usr/bin/env python3
"""Unit tests for robot-centric GOOSE-OCC safety diagnostics."""

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'goose_robot_safety', ROOT / 'tools/goose_robot_safety.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def empty_masks():
    return {
        name: np.zeros(MODULE.OCC_SHAPE, dtype=bool)
        for name in (
            'visible_known_free_mask', 'visible_occupied_mask',
            'occluded_known_free_mask', 'occluded_observed_mask')
    }


def test_perfect_prediction():
    target = np.zeros(MODULE.OCC_SHAPE, dtype=np.uint8)
    target[10, 64, 16] = 4
    target[20, 64, 16] = 2
    masks = empty_masks()
    masks['visible_occupied_mask'][10, 64, 16] = True
    masks['occluded_observed_mask'][20, 64, 16] = True
    masks['visible_known_free_mask'][11, 64, 16] = True
    prediction = target.copy()
    metric = MODULE.RobotSafetyMetrics()
    metric.update(prediction, target, masks)
    result = metric.result()
    assert result['domains']['visible']['occupied_recall'] == 1.0
    assert result['domains']['occluded']['occupied_recall'] == 1.0
    assert result['domains']['visible'][
        'false_occupied_known_free_rate'] == 0.0
    assert result['ego_corridor']['hazard_recall'] == 1.0


def test_false_blockage_and_miss():
    target = np.zeros(MODULE.OCC_SHAPE, dtype=np.uint8)
    target[10, 64, 16] = 4
    masks = empty_masks()
    masks['occluded_observed_mask'][10, 64, 16] = True
    masks['visible_known_free_mask'][11, 64, 16] = True
    prediction = np.zeros_like(target)
    prediction[11, 64, 16] = 3
    metric = MODULE.RobotSafetyMetrics()
    metric.update(prediction, target, masks)
    result = metric.result()
    assert result['domains']['occluded']['occupied_miss_rate'] == 1.0
    assert result['ego_corridor']['hazard_miss_rate'] == 1.0
    assert result['ego_corridor'][
        'false_occupied_known_free_rate'] == 1.0


def test_unknown_invariance():
    target = np.zeros(MODULE.OCC_SHAPE, dtype=np.uint8)
    masks = empty_masks()
    masks['visible_known_free_mask'][10, 64, 16] = True
    first = np.zeros_like(target)
    second = first.copy()
    second[100, 10, 10] = 7
    metric_a = MODULE.RobotSafetyMetrics()
    metric_b = MODULE.RobotSafetyMetrics()
    metric_a.update(first, target, masks)
    metric_b.update(second, target, masks)
    assert metric_a.result() == metric_b.result()


def test_empty_domain_and_invalid_contracts():
    target = np.zeros(MODULE.OCC_SHAPE, dtype=np.uint8)
    prediction = np.zeros_like(target)
    metric = MODULE.RobotSafetyMetrics()
    metric.update(prediction, target, empty_masks())
    result = metric.result()
    assert result['domains']['visible']['occupied_recall'] is None
    assert result['domains']['visible'][
        'false_occupied_known_free_rate'] is None

    try:
        MODULE.validate_prediction(np.zeros((128, 200, 32), dtype=np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError('Axis-permuted prediction was accepted')

    invalid = prediction.copy()
    invalid[0, 0, 0] = 8
    try:
        MODULE.validate_prediction(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid class ID was accepted')


def test_ground_is_not_corridor_hazard():
    target = np.zeros(MODULE.OCC_SHAPE, dtype=np.uint8)
    target[10, 64, 16] = 2
    masks = empty_masks()
    masks['visible_occupied_mask'][10, 64, 16] = True
    metric = MODULE.RobotSafetyMetrics()
    metric.update(np.zeros_like(target), target, masks)
    result = metric.result()
    assert result['domains']['visible']['occupied_miss_rate'] == 1.0
    assert result['ego_corridor']['hazard_miss_rate'] is None


def main():
    test_perfect_prediction()
    test_false_blockage_and_miss()
    test_unknown_invariance()
    test_empty_domain_and_invalid_contracts()
    test_ground_is_not_corridor_hazard()
    print('GOOSE robot safety tests passed')


if __name__ == '__main__':
    main()
