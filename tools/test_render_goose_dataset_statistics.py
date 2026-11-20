#!/usr/bin/env python3
"""Tests for the GOOSE dataset-statistics LaTeX renderer."""

import json
import tempfile
from pathlib import Path

from render_goose_dataset_statistics import render


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        statistics = root / 'statistics.json'
        macros = root / 'macros.tex'
        table = root / 'table.tex'
        statistics.write_text(json.dumps({
            'split': {
                'targets': {'train': 2, 'val': 1},
                'sequence_counts': {'train': 1, 'val': 1},
                'common_ten_history_targets': {'train': 1, 'val': 1},
            },
            'grid': {'all_target_grid_voxels': 100},
            'occupancy': {'occupied_voxels': 10},
            'observation': {
                'masks': [
                    {'mask': 'known_free_mask', 'voxels': 40,
                     'fraction_of_all_grid_voxels': 0.4},
                    {'mask': 'observed_occupied_mask', 'voxels': 10,
                     'fraction_of_all_grid_voxels': 0.1},
                    {'mask': 'unknown_mask', 'voxels': 50,
                     'fraction_of_all_grid_voxels': 0.5},
                ],
                'visible_valid_voxels': 20,
                'occluded_valid_voxels': 15,
                'full_grid_partition_valid': True,
            },
            'temporal': {'history_relations': 4, 'unique_history_images': 3},
            'pseudo_labels': {
                'valid_pixel_fraction': 0.8,
                'official_overlap_covered_pixel_accuracy': 0.9,
            },
            'metric3d': {
                'validated_target_files': 3,
                'validated_history_files': 3,
            },
            'quality': {'route_split_disjoint': True},
        }), encoding='utf-8')
        render(statistics, macros, table)
        assert r'\newcommand{\GooseTargets}{3}' in macros.read_text()
        assert r'\newcommand{\GooseUnknownPct}{50.00}' in macros.read_text()
        assert r'\label{tab:goose_observation_statistics}' in table.read_text()
        assert 'V-valid & 20 & 20.00' in table.read_text()
    print('GOOSE dataset statistics renderer tests passed')


if __name__ == '__main__':
    main()
