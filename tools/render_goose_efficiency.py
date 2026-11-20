#!/usr/bin/env python3
"""Render the unified GOOSE inference-efficiency table."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


VARIANTS = (
    ('Geometric lifting', '1', 'goose_geometric_lifting_common10.json'),
    ('Mono', '1', 'goose_mono_common10.json'),
    ('Temporal control', 'target + 3', 'goose_temporal_common10_3h.json'),
    ('Pose-guided', 'target + 3', 'goose_pose_guided_common10_3h.json'),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def render(input_dir, csv_output, latex_output, source_output):
    records = []
    hashes = {}
    for label, frames, filename in VARIANTS:
        path = Path(input_dir) / filename
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload['gpu_name'] != 'NVIDIA GeForce RTX 3090':
            raise ValueError('Unexpected GPU for {}'.format(label))
        if payload['num_warmup'] != 10 or payload['measured_samples'] != 50:
            raise ValueError('Timing population mismatch for {}'.format(label))
        if payload.get('evaluation_gt_stripped') is not True:
            raise ValueError('Evaluation-only GT was not stripped')
        records.append({
            'variant': label,
            'input_frames': frames,
            'parameters_m': payload['parameters'] / 1e6,
            'checkpoint_mb': payload['checkpoint_size_bytes'] / 1e6,
            'peak_memory_mb': payload['peak_allocated_mb'],
            'forward_ms': payload['mean_time_sec'] * 1000.0,
            'forward_fps': payload['fps'],
            'end_to_end_ms': payload['end_to_end_mean_sec'] * 1000.0,
            'end_to_end_fps': payload['end_to_end_fps'],
        })
        hashes[filename] = sha256(path)

    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\caption{Inference efficiency on one RTX 3090 at batch size 1 and '
        r'$256\times704$ image resolution. Ten samples are used for warm-up '
        r'and 50 for measurement. Evaluation-only occupancy and observation '
        r'files are removed from the input pipeline. Geometric lifting timing '
        r'includes 2D semantic inference and depth unprojection.}',
        r'\label{tab:efficiency}',
        r'\scriptsize',
        r'\begin{tabular}{lcccccccc}',
        r'\toprule',
        r'Variant & Frames & Params (M) & Checkpoint (MB) & Peak memory (MB) & '
        r'Forward (ms) & FPS & End-to-end (ms) & FPS \\',
        r'\midrule',
    ]
    for row in records:
        lines.append(
            '{variant} & {input_frames} & {parameters_m:.2f} & '
            '{checkpoint_mb:.1f} & {peak_memory_mb:.0f} & '
            '{forward_ms:.2f} & {forward_fps:.2f} & '
            '{end_to_end_ms:.2f} & {end_to_end_fps:.2f} \\\\'.format(**row))
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])
    Path(latex_output).write_text('\n'.join(lines), encoding='utf-8')
    source = {
        'protocol': {
            'gpu': 'NVIDIA GeForce RTX 3090',
            'batch_size': 1,
            'image_size': [256, 704],
            'warmup_samples': 10,
            'measured_samples': 50,
            'evaluation_gt_stripped': True,
        },
        'input_sha256': hashes,
        'records': records,
    }
    Path(source_output).write_text(
        json.dumps(source, indent=2) + '\n', encoding='utf-8')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default='work_dir/efficiency')
    parser.add_argument('--csv-output', default=(
        'paper/goose_occ_manuscript/generated/GOOSE_EFFICIENCY.csv'))
    parser.add_argument('--latex-output', default=(
        'paper/goose_occ_manuscript/ras_submission/generated_efficiency.tex'))
    parser.add_argument('--source-output', default=(
        'paper/goose_occ_manuscript/generated/GOOSE_EFFICIENCY_source.json'))
    args = parser.parse_args()
    records = render(
        args.input_dir, args.csv_output, args.latex_output, args.source_output)
    print('Rendered {} GOOSE efficiency records'.format(len(records)))


if __name__ == '__main__':
    main()
