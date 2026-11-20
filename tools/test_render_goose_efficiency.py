#!/usr/bin/env python3
"""Contract test for the GOOSE efficiency renderer."""

import json
import tempfile
from pathlib import Path

from render_goose_efficiency import VARIANTS, render


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for _, _, filename in VARIANTS:
            (root / filename).write_text(json.dumps({
                'gpu_name': 'NVIDIA GeForce RTX 3090',
                'num_warmup': 10,
                'measured_samples': 50,
                'evaluation_gt_stripped': True,
                'parameters': 1000000,
                'checkpoint_size_bytes': 2000000,
                'peak_allocated_mb': 300.0,
                'mean_time_sec': 0.01,
                'fps': 100.0,
                'end_to_end_mean_sec': 0.02,
                'end_to_end_fps': 50.0,
            }), encoding='utf-8')
        csv_path = root / 'output.csv'
        latex = root / 'output.tex'
        source = root / 'source.json'
        records = render(root, csv_path, latex, source)
        assert len(records) == 4
        assert r'\label{tab:efficiency}' in latex.read_text()
        assert 'Geometric lifting' in csv_path.read_text()
        assert len(json.loads(source.read_text())['input_sha256']) == 4
    print('GOOSE efficiency renderer tests passed')


if __name__ == '__main__':
    main()
