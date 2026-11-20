#!/usr/bin/env python3
"""Audit the formal geometric-lifting checkpoint, predictions, and boundary."""

import argparse
import json
from pathlib import Path

import numpy as np

from goose_lifting_baseline import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default=(
        'work_dir/goose_geometric_lifting_common10'))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    result_path = run_dir / 'goose_visibility_val.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result['num_samples'] != 469:
        raise ValueError('Expected 469 formal predictions')
    if result['prediction_contract'] != {
            'shape': [200, 128, 32], 'axis_order': 'x,y,z',
            'dtype': 'uint8', 'class_ids': '0..7'}:
        raise ValueError('Unexpected prediction contract')
    checkpoint = Path(result['checkpoint'])
    if sha256(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    resolved = json.loads(
        (run_dir / 'resolved_config.json').read_text(encoding='utf-8'))
    expected_training = ['train_rgb', 'train_official_2d_semantics']
    if list(resolved['training_inputs']) != expected_training:
        raise ValueError('Unexpected training inputs')
    forbidden = set(resolved['prohibited_training_inputs'])
    if forbidden != {
            'occupancy_gt', 'lidar', 'observation_masks', 'sweeps',
            'occupancy_fusion_timestamps'}:
        raise ValueError('Prohibited-input contract changed')

    manifests = result['prediction_manifest']
    if len(manifests) != 469:
        raise ValueError('Prediction manifest length mismatch')
    tokens = set()
    total_occupied = 0
    for row in manifests:
        path = Path(row['prediction_path'])
        if sha256(path) != row['prediction_sha256']:
            raise ValueError('Prediction hash mismatch: {}'.format(path))
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {'prediction'}:
                raise ValueError('Unexpected prediction NPZ keys')
            prediction = payload['prediction']
        if prediction.shape != (200, 128, 32):
            raise ValueError('Prediction shape mismatch')
        if prediction.dtype != np.uint8:
            raise ValueError('Prediction dtype mismatch')
        if prediction.size and int(prediction.max()) > 7:
            raise ValueError('Prediction class ID mismatch')
        if row['token'] in tokens:
            raise ValueError('Duplicate prediction token')
        tokens.add(row['token'])
        occupied = int((prediction > 0).sum())
        if occupied != int(row['occupied_voxels']):
            raise ValueError('Occupied count mismatch')
        total_occupied += occupied

    audit = {
        'status': 'pass',
        'predictions': len(manifests),
        'unique_tokens': len(tokens),
        'checkpoint_sha256': result['checkpoint_sha256'],
        'result_sha256': sha256(result_path),
        'total_predicted_occupied_voxels': total_occupied,
        'training_inputs': expected_training,
        'prohibited_training_inputs': sorted(forbidden),
        'prediction_inputs': list(resolved['prediction_inputs']),
    }
    output = run_dir / 'audit.json'
    output.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
