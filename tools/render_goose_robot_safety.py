#!/usr/bin/env python3
"""Render frozen GOOSE robot-safety diagnostics for the RAS manuscript."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'work_dir/goose_robot_safety.json'
DEFAULT_OUTPUT_DIR = ROOT / 'paper/goose_occ_manuscript/ras_submission'
DEFAULT_SOURCE = (
    ROOT / 'work_dir/goose_robot_safety_summary.json')
GROUPS = (
    (
        'Mono',
        tuple('goose_mono_common10_seed{}'.format(seed) for seed in range(3)),
    ),
    (
        'Temporal control',
        tuple('goose_temporal_common10_3h_binary_conf_seed{}'.format(seed)
              for seed in range(5)),
    ),
    (
        'Pose-guided',
        tuple('goose_temporal_occlusion_common10_3h_seed{}'.format(seed)
              for seed in range(5)),
    ),
    ('Geometric lifting', ('goose_geometric_lifting_common10',)),
    ('SurroundOcc (OCC-GT)', ('surroundocc_goose_common10',)),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def nested(record, path):
    for key in path:
        record = record[key]
    return record


def summarize(values):
    if any(value is None for value in values):
        raise ValueError('Published safety metric has an empty denominator')
    percentages = [100.0 * float(value) for value in values]
    return {
        'n': len(percentages),
        'mean_percent': round(statistics.mean(percentages), 12),
        'sample_std_percent': (
            round(statistics.stdev(percentages), 12)
            if len(percentages) > 1 else None),
        'values_percent': [round(value, 12) for value in percentages],
    }


def format_summary(summary):
    if summary['sample_std_percent'] is None:
        return '{:.2f}'.format(summary['mean_percent'])
    return '${:.2f}\\pm{:.2f}$'.format(
        summary['mean_percent'], summary['sample_std_percent'])


MAIN_METRICS = (
    ('V occupied recall', ('domains', 'visible', 'occupied_recall')),
    ('O occupied recall', ('domains', 'occluded', 'occupied_recall')),
    ('V false occupied', (
        'domains', 'visible', 'false_occupied_known_free_rate')),
    ('O false occupied', (
        'domains', 'occluded', 'false_occupied_known_free_rate')),
    ('Corridor hazard miss', ('ego_corridor', 'hazard_miss_rate')),
    ('Corridor false blockage', (
        'ego_corridor', 'false_occupied_known_free_rate')),
)
RANGE_METRICS = tuple(
    (
        '{} {} recall'.format(bucket, short),
        ('by_horizontal_range', bucket, domain, 'occupied_recall'),
    )
    for bucket in ('0-10m', '10-25m')
    for short, domain in (('V', 'visible'), ('O', 'occluded'))
)


def aggregate(payload):
    runs = payload['runs']
    output = []
    for label, names in GROUPS:
        missing = [name for name in names if name not in runs]
        if missing:
            raise ValueError('Missing safety runs: {}'.format(missing))
        record = {'label': label, 'runs': list(names), 'metrics': {}}
        for metric_name, path in MAIN_METRICS + RANGE_METRICS:
            record['metrics'][metric_name] = summarize(
                [nested(runs[name], path) for name in names])
        output.append(record)
    return output


def render_main(records):
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        (r'\caption{Robot-centric voxel diagnostics on 469 common validation '
         r'targets (\%). Occupied recall uses observation-supported occupied '
         r'voxels; false occupied rate uses only observation-supported known '
         r'free voxels. The ego corridor is $x\in[0,25)$\,m, $|y|\leq1.5$\,m, '
         r'and $z\in[-2,2)$\,m. Corridor hazards exclude artificial and natural '
         r'ground. These are perception diagnostics, not navigation-success '
         r'measures.}'),
        r'\label{tab:robot_safety}',
        r'\scriptsize',
        r'\begin{tabular}{lrrrrrr}',
        r'\toprule',
        (r'Variant & V occ. recall $\uparrow$ & O occ. recall $\uparrow$ & '
         r'V false occ. $\downarrow$ & O false occ. $\downarrow$ & '
         r'Corridor hazard miss $\downarrow$ & False blockage $\downarrow$ \\'),
        r'\midrule',
    ]
    for record in records:
        values = [format_summary(record['metrics'][name])
                  for name, _ in MAIN_METRICS]
        lines.append(r'{} & {} \\'.format(
            record['label'], ' & '.join(values)))
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        (r'\par\vspace{2pt}\raggedright Values are mean $\pm$ sample standard '
         r'deviation for multi-seed weak variants and deterministic values for '
         r'the two fixed controls. SurroundOcc uses occupancy-ground-truth '
         r'supervision and is not supervision matched.'),
        r'\end{table*}',
        '',
    ])
    return '\n'.join(lines)


def render_range(records):
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        (r'\caption{Observation-supported occupied recall by horizontal range '
         r'(\%). V and O denote visible and occluded domains.}'),
        r'\label{tab:robot_safety_range}',
        r'\scriptsize',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Variant & 0--10 V & 0--10 O & 10--25 V & 10--25 O \\',
        r'\midrule',
    ]
    for record in records:
        values = [format_summary(record['metrics'][name])
                  for name, _ in RANGE_METRICS]
        lines.append(r'{} & {} \\'.format(
            record['label'], ' & '.join(values)))
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table*}',
        '',
    ])
    return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--source-output', type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding='utf-8'))
    if payload.get('protocol') != 'goose_occ_robot_safety_v1':
        raise ValueError('Unexpected robot-safety protocol')
    if payload.get('evaluation_only') is not True:
        raise ValueError('Safety analysis must be evaluation-only')
    records = aggregate(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'generated_robot_safety.tex').write_text(
        render_main(records), encoding='utf-8')
    (args.output_dir / 'generated_robot_safety_range.tex').write_text(
        render_range(records), encoding='utf-8')
    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_output.write_text(json.dumps({
        'source': str(args.input),
        'source_sha256': sha256(args.input),
        'protocol': payload['protocol'],
        'population': payload['population'],
        'definitions': next(iter(payload['runs'].values()))['definitions'],
        'groups': records,
    }, indent=2) + '\n', encoding='utf-8')
    print('Rendered {} robot-safety groups'.format(len(records)))


if __name__ == '__main__':
    main()
