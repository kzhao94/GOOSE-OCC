#!/usr/bin/env python3
"""Contract tests for the evidence-backed dataset comparison renderer."""

import json
import tempfile
from pathlib import Path

from render_goose_dataset_comparison import REQUIRED_FIELDS, render


def record(name, frames, sequences, vo_state):
    row = {field: 'reported' for field in REQUIRED_FIELDS}
    row.update({
        'dataset': name,
        'category': 'A: direct off-road semantic occupancy',
        'year': 2026,
        'frames': frames,
        'scenes_or_sequences': sequences,
        'visible_occluded_state': vo_state,
        'sources': [{'type': 'paper', 'locator': 'source'}],
    })
    return row


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'sources.json'
        statistics = root / 'statistics.json'
        csv_path = root / 'comparison.csv'
        markdown = root / 'comparison.md'
        latex = root / 'comparison.tex'
        supplement = root / 'context.tex'
        payload = {
            'schema_version': 1,
            'comparison_policy': {
                'native_metrics_directly_ranked': False,
                'reason': 'Different contracts.',
                'claim_boundary': 'Protocol comparison only.',
            },
            'datasets': [
                record('GOOSE-OCC v1.4.0', 2, 2, 'released V/O masks'),
                record('WildOcc', 10, 5, 'not released/reported'),
            ],
        }
        source.write_text(json.dumps(payload), encoding='utf-8')
        statistics.write_text(json.dumps({
            'split': {
                'targets': {'train': 1, 'val': 1},
                'sequence_counts': {'train': 1, 'val': 1},
            },
        }), encoding='utf-8')
        rendered = render(
            source, statistics, csv_path, markdown, latex, supplement)
        assert len(rendered['datasets']) == 2
        assert 'Native IoU/mIoU values are not ranked' in markdown.read_text()
        latex_text = latex.read_text()
        assert r'\label{tab:offroad_datasets}' in latex_text
        assert latex_text.count(r' \\') >= 3
        supplement_text = supplement.read_text()
        assert r'\label{tab:dataset_context}' in supplement_text
        assert 'Direct off-road semantic occupancy' in supplement_text
        assert csv_path.read_text().count('\n') == 3

        payload['comparison_policy']['native_metrics_directly_ranked'] = True
        source.write_text(json.dumps(payload), encoding='utf-8')
        try:
            render(source, statistics, csv_path, markdown, latex)
        except ValueError as error:
            assert 'must not be ranked' in str(error)
        else:
            raise AssertionError('Unsafe cross-dataset ranking did not fail')
    print('GOOSE dataset comparison renderer tests passed')


if __name__ == '__main__':
    main()
