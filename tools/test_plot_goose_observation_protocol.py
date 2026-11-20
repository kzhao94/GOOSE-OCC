#!/usr/bin/env python3
"""Minimal source-contract test for the observation protocol plotter."""

import json
from pathlib import Path


def main():
    path = Path(
        '/path/to/GOOSE-OCC/observation/'
        'observation_protocol.json')
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['format'] == 'goose_occ_observation_v1'
    assert payload['camera_clearance_radius_m'] == 1.5
    assert 'occupied endpoints override free evidence' in (
        payload['known_free_mask'].lower())
    assert 'non-target lidar source' in (
        payload['occluded_observed_mask'].lower())
    assert payload['visible_valid'].startswith(
        'visible_known_free_mask union visible_occupied_mask')
    assert payload['occluded_valid'].startswith(
        'occluded_known_free_mask union occluded_observed_mask')
    print('GOOSE observation protocol figure source contract passed')


if __name__ == '__main__':
    main()
