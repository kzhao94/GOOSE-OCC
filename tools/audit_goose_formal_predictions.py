#!/usr/bin/env python3
"""Audit all formal GOOSE prediction dumps and their voxel contract."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OCC_SHAPE = (200, 128, 32)
EXPECTED_WEAK_RUNS = (
    'goose_mono_common10_seed0',
    'goose_mono_common10_seed1',
    'goose_mono_common10_seed2',
    'goose_temporal_common10_3h_binary_conf_seed0',
    'goose_temporal_common10_3h_binary_conf_seed1',
    'goose_temporal_common10_3h_binary_conf_seed2',
    'goose_temporal_common10_3h_binary_conf_seed3',
    'goose_temporal_common10_3h_binary_conf_seed4',
    'goose_temporal_occlusion_common10_3h_seed0',
    'goose_temporal_occlusion_common10_3h_seed1',
    'goose_temporal_occlusion_common10_3h_seed2',
    'goose_temporal_occlusion_common10_3h_seed3',
    'goose_temporal_occlusion_common10_3h_seed4',
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_prediction(array, path):
    if array.shape != OCC_SHAPE:
        raise ValueError('{} shape {} != {}'.format(path, array.shape, OCC_SHAPE))
    if array.dtype != np.uint8:
        raise ValueError('{} dtype {} != uint8'.format(path, array.dtype))
    if array.size and (int(array.min()) < 0 or int(array.max()) > 7):
        raise ValueError('{} contains IDs outside 0..7'.format(path))


def audit_weak(root):
    records = []
    for run in EXPECTED_WEAK_RUNS:
        run_dir = root / run
        audit_path = run_dir / 'prediction_audit.json'
        if not audit_path.is_file():
            raise ValueError('Missing weak prediction audit: {}'.format(audit_path))
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        contract = audit.get('prediction_contract') or {}
        if audit.get('num_samples') != 469 \
                or audit.get('dumped_predictions') != 469 \
                or audit.get('world_size') != 1:
            raise ValueError('{} does not declare 469 single-GPU dumps'.format(run))
        if contract != {
                'shape': list(OCC_SHAPE), 'axis_order': 'x,y,z',
                'dtype': 'uint8', 'class_ids': '0..7'}:
            raise ValueError('{} has invalid prediction contract'.format(run))

        paths = sorted(run_dir.glob('prediction_*.npz'))
        if len(paths) != 469:
            raise ValueError('{} contains {} predictions, expected 469'.format(
                run, len(paths)))
        tokens = set()
        file_records = []
        for index, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as payload:
                required = {
                    'occ_pred', 'axis_order', 'sample_index', 'sample_token',
                    'sequence', 'frame_index'}
                if not required.issubset(payload.files):
                    raise ValueError('{} missing metadata fields'.format(path))
                check_prediction(payload['occ_pred'], path)
                if str(payload['axis_order'].item()) != 'x,y,z':
                    raise ValueError('{} has invalid axis order'.format(path))
                if int(payload['sample_index'].item()) != index:
                    raise ValueError('{} has non-canonical sample index'.format(path))
                token = str(payload['sample_token'].item())
                if not token or token in tokens:
                    raise ValueError('{} has empty or duplicate token'.format(path))
                tokens.add(token)
                file_records.append({
                    'file': path.name,
                    'sample_token': token,
                    'size_bytes': path.stat().st_size,
                    'sha256': sha256(path),
                })
        records.append({
            'run': run,
            'predictions': len(paths),
            'files': file_records,
        })
    return records


def audit_surround(root, result_path):
    result = json.loads(result_path.read_text(encoding='utf-8'))
    contract = result.get('prediction_contract', {})
    if result.get('num_samples') != 469 or result.get('dataset_samples') != 469:
        raise ValueError('SurroundOcc result is not 469/469')
    if contract != {
            'shape': list(OCC_SHAPE), 'axis_order': 'x,y,z',
            'dtype': 'uint8', 'class_ids': '0..7'}:
        raise ValueError('SurroundOcc result has invalid prediction contract')
    paths = sorted(root.glob('**/*.npy'))
    if len(paths) != 469:
        raise ValueError('SurroundOcc contains {} predictions, expected 469'.format(
            len(paths)))
    file_records = []
    for path in paths:
        check_prediction(np.load(path, allow_pickle=False), path)
        file_records.append({
            'file': str(path.relative_to(root)),
            'size_bytes': path.stat().st_size,
            'sha256': sha256(path),
        })
    return {
        'run': 'surroundocc_goose_common10',
        'predictions': len(paths),
        'files': file_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weak-root', type=Path, required=True)
    parser.add_argument('--surround-root', type=Path, default=None)
    parser.add_argument('--surround-result', type=Path, default=None)
    parser.add_argument(
        '--output', type=Path,
        default=Path('work_dir/goose_prediction_audit.json'))
    args = parser.parse_args()
    records = audit_weak(args.weak_root)
    if (args.surround_root is None) != (args.surround_result is None):
        raise ValueError('Provide both SurroundOcc paths or neither')
    if args.surround_root is not None:
        records.append(audit_surround(
            args.surround_root, args.surround_result))
    output = {
        'prediction_contract': {
            'shape': list(OCC_SHAPE), 'axis_order': 'x,y,z',
            'dtype': 'uint8', 'class_ids': '0..7'},
        'records': records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
