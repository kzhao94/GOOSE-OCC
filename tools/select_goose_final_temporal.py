#!/usr/bin/env python3
"""Select the predeclared GOOSE temporal configuration for seed repeats."""

import argparse
import json
from pathlib import Path


CANDIDATES = (
    dict(
        experiment='goose_temporal_common10_1h_seed0',
        config='configs/goose_temporal_common10_1h.py',
        num_history=1,
        confidence='weighted',
        history_weight=0.75,
    ),
    dict(
        experiment='goose_temporal_common10_3h_seed0',
        config='configs/goose_temporal_common10_3h.py',
        num_history=3,
        confidence='weighted',
        history_weight=0.75,
    ),
    dict(
        experiment='goose_temporal_common10_5h_seed0',
        config='configs/goose_temporal_common10_5h.py',
        num_history=5,
        confidence='weighted',
        history_weight=0.75,
    ),
    dict(
        experiment='goose_temporal_common10_10h_seed0',
        config='configs/goose_temporal_common10_10h.py',
        num_history=10,
        confidence='weighted',
        history_weight=0.75,
    ),
    dict(
        experiment='goose_temporal_common10_3h_binary_conf_seed0',
        config='configs/goose_temporal_common10_3h_binary_conf.py',
        num_history=3,
        confidence='binary',
        history_weight=0.75,
    ),
    dict(
        experiment='goose_temporal_common10_3h_weight025_seed0',
        config='configs/goose_temporal_common10_3h_weight025.py',
        num_history=3,
        confidence='weighted',
        history_weight=0.25,
    ),
    dict(
        experiment='goose_temporal_common10_3h_weight100_seed0',
        config='configs/goose_temporal_common10_3h_weight100.py',
        num_history=3,
        confidence='weighted',
        history_weight=1.0,
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', default='work_dir/GOOSETemporalOccMono')
    parser.add_argument(
        '--shell', action='store_true',
        help='Print tab-separated config and seedless run name.')
    return parser.parse_args()


def load_candidates(work_dir):
    loaded = []
    for candidate in CANDIDATES:
        result_path = (
            work_dir / candidate['experiment'] / 'val_best.json')
        if not result_path.is_file():
            raise FileNotFoundError(
                'Missing candidate result: {}'.format(result_path))
        with result_path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        score = payload.get('metrics', {}).get('O-SSC-mIoU')
        if not isinstance(score, (float, int)):
            raise ValueError(
                'Missing numeric O-SSC-mIoU in {}'.format(result_path))
        item = dict(candidate)
        item['score'] = float(score)
        item['result_json'] = str(result_path)
        loaded.append(item)
    return loaded


def selection_key(candidate):
    return (
        -candidate['score'],
        candidate['num_history'],
        candidate['confidence'] != 'weighted',
        abs(candidate['history_weight'] - 0.75),
        candidate['experiment'],
    )


def main():
    args = parse_args()
    candidates = load_candidates(Path(args.work_dir))
    selected = min(candidates, key=selection_key)
    selected['run_stem'] = selected['experiment'][:-len('_seed0')]
    if args.shell:
        print('{}\t{}'.format(selected['config'], selected['run_stem']))
    else:
        print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
