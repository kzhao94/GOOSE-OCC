#!/usr/bin/env python3
"""Contract test for the RAS LaTeX result-table renderer."""

import json
import tempfile
from pathlib import Path

from render_goose_latex_results import render_latex
from test_render_goose_manuscript_results import (
    CONTROL_STEM,
    CORE_METRICS,
    METHOD_STEM,
    MONO_STEM,
    aggregate,
    external,
    paired,
    write_csv,
)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        multiseed = root / 'multiseed.csv'
        paired_path = root / 'paired.csv'
        external_path = root / 'external.csv'
        gate_path = root / 'gate.json'
        lifting_path = root / 'lifting.json'
        write_csv(multiseed, [
            aggregate(MONO_STEM, 3), aggregate(CONTROL_STEM, 5),
            aggregate(METHOD_STEM, 5)])
        write_csv(paired_path, [paired(metric) for metric in CORE_METRICS])
        write_csv(external_path, [external()])
        gate_path.write_text(json.dumps({
            'metric': 'O-SSC-mIoU', 'num_paired_seeds': 5,
            'seed_deltas': {str(seed): 0.1 for seed in range(5)},
            'narrative': 'protocol-led mixed-result study',
            'interpretation': 'Synthetic contract test.',
            'significance_test_performed': False,
        }), encoding='utf-8')
        per_class = {
            metric.replace('IoU_', ''): {'iou': 0.1}
            for _, metric in __import__(
                'render_goose_manuscript_results').CLASS_METRICS}
        by_range = {
            distance_bin: {
                domain: {'SC-IoU': 0.1, 'SSC-mIoU': 0.1}
                for domain in ('visible', 'occluded')}
            for distance_bin in __import__(
                'render_goose_manuscript_results').DISTANCE_BINS}
        lifting_path.write_text(json.dumps({
            'num_samples': 469,
            'supervision': 'train 2D; no OCC/LiDAR/mask training',
            'prediction_contract': {
                'shape': [200, 128, 32], 'axis_order': 'x,y,z',
                'class_ids': '0..7'},
            'full_grid': {
                'SC-IoU': 0.1, 'SSC-mIoU': 0.1,
                'per_class': per_class},
            'visibility': {
                'V-SC-IoU': 0.1, 'V-SSC-mIoU': 0.1,
                'O-SC-IoU': 0.1, 'O-SSC-mIoU': 0.1,
                'by_horizontal_range': by_range},
        }), encoding='utf-8')
        output = render_latex(
            multiseed, paired_path, external_path, gate_path, lifting_path)
        assert output.count(r'\begin{table*}') == 4
        assert r'\label{tab:main_results}' in output
        assert 'Fully supervised OCC GT' in output
        assert 'Geometric lifting' in output
        assert 'No significance test was performed' in output
    print('GOOSE LaTeX result renderer tests passed')


if __name__ == '__main__':
    main()
