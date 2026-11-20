#!/usr/bin/env python3
"""Contract tests for the architecture-distinct GOOSE lifting baseline."""

import numpy as np

from goose_lifting_baseline import (
    common_target_infos, lift_to_grid, prediction_record, train_record)


def main():
    train_infos = common_target_infos(
        '/path/to/GOOSE-OCC/infos/goose_infos_train.pkl',
        '/path/to/GOOSE-OCC/temporal/'
        'temporal_infos_train.pkl', 10)
    val_infos = common_target_infos(
        '/path/to/GOOSE-OCC/infos/goose_infos_val.pkl',
        '/path/to/GOOSE-OCC/temporal/'
        'temporal_infos_val.pkl', 10)
    assert len(train_infos) == 2726
    assert len(val_infos) == 469
    train = train_record(
        train_infos[0], '/path/to/GOOSE-OCC')
    assert set(train) == {'token', 'image_path', 'semantic_path'}
    assert all(term not in ' '.join(train).lower() for term in (
        'occ', 'lidar', 'observation', 'sweep', 'source_stamp'))
    prediction_input = prediction_record(
        val_infos[0], '/path/to/GOOSE-OCC')
    assert 'semantic_path' not in prediction_input
    assert all(term not in ' '.join(prediction_input).lower() for term in (
        'occ_path', 'observation', 'sweep', 'source_stamp'))

    semantic = np.zeros((3, 3), dtype=np.uint8)
    depth = np.zeros((3, 3), dtype=np.float32)
    semantic[1, 1] = 4
    depth[1, 1] = 1.5
    intrinsic = np.asarray([[1.0, 0.0, 1.0],
                            [0.0, 1.0, 1.0],
                            [0.0, 0.0, 1.0]])
    prediction = lift_to_grid(
        semantic, depth, intrinsic, np.eye(4),
        (-1.0, -1.0, 0.0, 2.0, 2.0, 3.0), (3, 3, 3), 1.0)
    assert prediction.shape == (3, 3, 3)
    assert prediction[1, 1, 1] == 4
    assert np.count_nonzero(prediction) == 1
    print('GOOSE lifting baseline contracts passed: 2726/469, no test labels')


if __name__ == '__main__':
    main()
