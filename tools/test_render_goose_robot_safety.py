#!/usr/bin/env python3
"""Tests for the GOOSE robot-safety manuscript renderer."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'render_goose_robot_safety',
    ROOT / 'tools/render_goose_robot_safety.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_run(value):
    leaf = {
        'occupied_recall': value,
        'false_occupied_known_free_rate': value,
        'hazard_miss_rate': value,
    }
    return {
        'domains': {'visible': dict(leaf), 'occluded': dict(leaf)},
        'by_horizontal_range': {
            bucket: {'visible': dict(leaf), 'occluded': dict(leaf)}
            for bucket in ('0-10m', '10-25m')
        },
        'ego_corridor': dict(leaf),
    }


def main():
    runs = {}
    value = 0.1
    for _, names in MODULE.GROUPS:
        for name in names:
            runs[name] = fake_run(value)
            value += 0.01
    records = MODULE.aggregate({'runs': runs})
    assert len(records) == 5
    assert records[0]['metrics']['V occupied recall']['n'] == 3
    assert records[1]['metrics']['V occupied recall']['n'] == 5
    main_tex = MODULE.render_main(records)
    range_tex = MODULE.render_range(records)
    assert 'Corridor hazard miss' in main_tex
    assert 'not navigation-success' in main_tex
    assert '0--10 V' in range_tex
    assert all(
        line.endswith(r'\\')
        for line in main_tex.splitlines()
        if line.startswith(tuple(record['label'] for record in records))
    )
    print('GOOSE robot safety renderer tests passed')


if __name__ == '__main__':
    main()
