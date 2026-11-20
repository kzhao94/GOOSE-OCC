#!/usr/bin/env python3
"""Contract tests for the frozen GOOSE-OCC dataset statistics renderer."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from summarize_goose_dataset import collect, render_markdown, write_outputs


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_release(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'VERSION').write_text('1.4.0\n', encoding='utf-8')
    occ_size = [4, 4, 2]
    metadata = {
        'target_count': 2,
        'occ_size': occ_size,
        'pc_range': [0, -2, -1, 4, 2, 1],
        'voxel_size': 1.0,
        'stored_class_ids': {
            'free': 0,
            'artificial_structures': 1,
            'artificial_ground': 2,
        },
        'occupied_voxel_class_counts': {
            'artificial_structures': 2,
            'artificial_ground': 2,
        },
    }
    write_json(root / 'metadata.json', metadata)
    write_json(root / 'validation.json', {'valid': True, 'errors': []})

    occ_paths = []
    arrays = (
        np.asarray([[0, 0, 0, 1], [1, 1, 0, 2]], dtype=np.int32),
        np.asarray([[2, 2, 1, 1], [3, 3, 1, 2]], dtype=np.int32),
    )
    for index, array in enumerate(arrays):
        path = root / 'samples' / 'occupancy' / '{}.npy'.format(index)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        occ_paths.append(str(path.relative_to(root)))

    manifest_fields = ('split', 'sequence', 'occ_path')
    write_csv(root / 'manifest.csv', manifest_fields, [
        {'split': 'train', 'sequence': 'route_train', 'occ_path': occ_paths[0]},
        {'split': 'val', 'sequence': 'route_val', 'occ_path': occ_paths[1]},
    ])

    mask_fields = (
        'split', 'sequence', 'known_free_mask', 'observed_occupied_mask',
        'visible_occupied_mask', 'occluded_observed_mask', 'unknown_mask',
        'visible_known_free_mask', 'occluded_known_free_mask',
        'non_target_occupied_support_mask', 'non_target_free_mask',
        'target_occupied_support_mask', 'target_free_mask',
        'camera_clearance_mask')
    observation_rows = [
        {
            'split': 'train', 'sequence': 'route_train',
            'known_free_mask': 10, 'observed_occupied_mask': 2,
            'visible_occupied_mask': 1, 'occluded_observed_mask': 0,
            'unknown_mask': 20, 'visible_known_free_mask': 5,
            'occluded_known_free_mask': 0,
            'non_target_occupied_support_mask': 0,
            'non_target_free_mask': 0,
            'target_occupied_support_mask': 2, 'target_free_mask': 10,
            'camera_clearance_mask': 1,
        },
        {
            'split': 'val', 'sequence': 'route_val',
            'known_free_mask': 5, 'observed_occupied_mask': 2,
            'visible_occupied_mask': 1, 'occluded_observed_mask': 1,
            'unknown_mask': 25, 'visible_known_free_mask': 2,
            'occluded_known_free_mask': 2,
            'non_target_occupied_support_mask': 1,
            'non_target_free_mask': 2,
            'target_occupied_support_mask': 1, 'target_free_mask': 3,
            'camera_clearance_mask': 1,
        },
    ]
    write_csv(
        root / 'observation/observation_manifest.csv',
        mask_fields, observation_rows)
    observation_totals = {
        name: sum(int(row[name]) for row in observation_rows)
        for name in mask_fields if name.endswith('_mask')
    }
    write_json(root / 'observation/observation_protocol.json', {'version': 1})
    write_json(root / 'observation/observation_validation.json', {
        'valid': True,
        'mask_voxel_totals': observation_totals,
    })

    target_fields = ('split', 'sequence', 'history_count')
    write_csv(root / 'temporal/temporal_targets.csv', target_fields, [
        {'split': 'train', 'sequence': 'route_train', 'history_count': 10},
        {'split': 'val', 'sequence': 'route_val', 'history_count': 10},
    ])
    temporal_fields = ('actual_lag_ms', 'selection_error_ms')
    write_csv(root / 'temporal/temporal_manifest.csv', temporal_fields, [
        {'actual_lag_ms': 200, 'selection_error_ms': -1},
        {'actual_lag_ms': 400, 'selection_error_ms': 2},
    ])
    write_json(root / 'temporal/temporal_validation.json', {
        'valid': True,
        'unique_history_images': 2,
        'pose_status': {'interpolated': 2},
        'anchor_status': {'ok': 2},
        'excluded_history_relations': 0,
    })
    pseudo_fields = ('valid_fraction', 'mean_confidence')
    write_csv(root / 'temporal/pseudo_label_manifest.csv', pseudo_fields, [
        {'valid_fraction': 0.8, 'mean_confidence': 240},
        {'valid_fraction': 0.9, 'mean_confidence': 245},
    ])
    write_json(root / 'temporal/pseudo_label_validation.json', {
        'valid': True,
        'manifest_rows': 2,
        'valid_pixel_fraction': 0.85,
        'mean_valid_confidence': 242.5,
    })
    write_json(root / 'temporal/pseudo_label_audit.json', {
        'audited_overlap': 2,
        'overall': {
            'coverage': 0.9,
            'covered_pixel_accuracy': 0.95,
            'penalized_miou': 0.7,
        },
    })
    write_json(root / 'metric3d/metric3d_validation.json', {
        'valid': True,
        'validated_files': {'target': 2, 'history': 2},
        'positive_pixel_fraction': 1.0,
        'mean_positive_depth_m': 10.0,
        'target_lidar_scale_audit': {'frames': 2},
    })


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'release'
        output = Path(directory) / 'generated'
        figures = Path(directory) / 'figures'
        build_release(root)
        payload = collect(root)
        assert payload['split']['targets'] == {'train': 1, 'val': 1}
        assert payload['split']['route_overlap'] == []
        assert payload['occupancy']['occupied_voxels'] == 4
        assert payload['observation']['full_grid_partition_valid'] is True
        assert payload['temporal']['history_relations'] == 2
        assert payload['pseudo_labels']['valid_pixel_fraction'] == 0.85
        markdown = render_markdown(payload)
        assert 'Observation-state totals' in markdown
        assert 'Route split disjoint: True' in markdown
        write_outputs(payload, output, figures)
        assert (output / 'GOOSE_DATASET_STATISTICS.json').is_file()
        assert (output / 'GOOSE_CLASS_STATISTICS.csv').is_file()
        assert (figures / 'fig_goose_temporal_statistics.pdf').is_file()
    print('GOOSE dataset statistics tests passed')


if __name__ == '__main__':
    main()
