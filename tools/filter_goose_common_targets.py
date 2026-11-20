#!/usr/bin/env python3
"""Create base GOOSE infos restricted to targets with real camera histories."""

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path('/path/to/GOOSE-OCC')
DEFAULT_OUTPUT_DIR = ROOT / 'work_dir/final_results_v1'


def install_numpy_pickle_aliases():
    """Allow NumPy <2 environments to read infos written by NumPy 2."""
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--min-history', type=int, default=10)
    return parser.parse_args()


def target_key(record, temporal=False):
    stamp_key = 'target_sensor_stamp_ns' if temporal else 'sensor_stamp_ns'
    return (
        record['sequence'],
        int(record['frame_index']),
        int(record[stamp_key]),
    )


def load_pickle(path):
    with path.open('rb') as handle:
        return pickle.load(handle)


def eligible_temporal_keys(records, min_history):
    keys = []
    for record in records:
        if len(record.get('histories', ())) >= min_history:
            keys.append(target_key(record, temporal=True))
    if len(keys) != len(set(keys)):
        raise ValueError('Temporal infos contain duplicate eligible target keys')
    return set(keys)


def filter_split(data_root, output_dir, split, min_history):
    base_path = data_root / 'infos/goose_infos_{}.pkl'.format(split)
    temporal_path = data_root / 'temporal/temporal_infos_{}.pkl'.format(split)
    base = load_pickle(base_path)
    temporal = load_pickle(temporal_path)

    if not isinstance(base, dict) or not isinstance(base.get('infos'), list):
        raise TypeError('{} must contain a dict with an infos list'.format(base_path))
    if not isinstance(temporal, list):
        raise TypeError('{} must contain a list'.format(temporal_path))

    eligible = eligible_temporal_keys(temporal, min_history)
    base_by_key = {target_key(record): record for record in base['infos']}
    if len(base_by_key) != len(base['infos']):
        raise ValueError('Base {} infos contain duplicate target keys'.format(split))

    missing = eligible.difference(base_by_key)
    if missing:
        raise ValueError(
            '{} eligible temporal targets are absent from base {} infos'.format(
                len(missing), split))

    filtered = [
        record for record in base['infos'] if target_key(record) in eligible
    ]
    if len(filtered) != len(eligible):
        raise RuntimeError('Filtered target count does not match eligible key count')

    payload = copy.copy(base)
    payload['infos'] = filtered
    payload['metadata'] = copy.deepcopy(base.get('metadata', {}))
    payload['metadata']['common_target_filter'] = {
        'criterion': 'real_camera_history_count >= {}'.format(min_history),
        'min_history': min_history,
        'source_base_infos': str(base_path.relative_to(data_root)),
        'source_temporal_infos': str(temporal_path.relative_to(data_root)),
        'source_count': len(base['infos']),
        'filtered_count': len(filtered),
    }

    output_path = output_dir / 'surroundocc_common10_infos_{}.pkl'.format(split)
    with output_path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=4)
    return output_path, len(base['infos']), len(filtered)


def main():
    args = parse_args()
    if args.min_history < 1:
        raise ValueError('--min-history must be positive')
    install_numpy_pickle_aliases()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        path, source_count, filtered_count = filter_split(
            args.data_root.resolve(), args.output_dir.resolve(), split,
            args.min_history)
        print('{}: {} -> {} ({})'.format(
            split, source_count, filtered_count, path))


if __name__ == '__main__':
    main()
