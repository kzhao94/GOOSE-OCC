#!/usr/bin/env python3
"""Collect GOOSE validation JSON files into CSV and Markdown tables."""

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


CORE_METRICS = (
    'SC_IoU',
    'SSC_mIoU',
    'V-SC-IoU',
    'V-SSC-mIoU',
    'O-SC-IoU',
    'O-SSC-mIoU',
    'O-SSC-mIoU_0-10m',
    'O-SSC-mIoU_10-25m',
    'O-SSC-mIoU_25-50m',
)

DISTANCE_BINS = ('0-10m', '10-25m', '25-50m')
VISIBILITY_METRICS = (
    'V-SC-IoU',
    'V-SSC-mIoU',
    'O-SC-IoU',
    'O-SSC-mIoU',
)
CLASS_METRICS = (
    ('free', 'IoU_free'),
    ('structures', 'IoU_artificial_structures'),
    ('artificial ground', 'IoU_artificial_ground'),
    ('natural ground', 'IoU_natural_ground'),
    ('obstacle', 'IoU_obstacle'),
    ('vehicle', 'IoU_vehicle'),
    ('vegetation', 'IoU_vegetation'),
    ('human', 'IoU_human'),
)

FAIR_COMPARISON = (
    ('mono', 'goose_mono_fair_3h_seed0'),
    ('target + 3 history', 'goose_temporal_fair_3h_seed0'),
)
HISTORY_ABLATION = (
    ('mono', 'goose_mono_common10_seed0'),
    ('1 history', 'goose_temporal_common10_1h_seed0'),
    ('3 history', 'goose_temporal_common10_3h_seed0'),
    ('5 history', 'goose_temporal_common10_5h_seed0'),
    ('10 history', 'goose_temporal_common10_10h_seed0'),
)
SUPERVISION_ABLATION = (
    ('weighted conf., weight=0.75', 'goose_temporal_common10_3h_seed0'),
    ('binary conf., weight=0.75',
     'goose_temporal_common10_3h_binary_conf_seed0'),
    ('weighted conf., weight=0.25',
     'goose_temporal_common10_3h_weight025_seed0'),
    ('weighted conf., weight=1.00',
     'goose_temporal_common10_3h_weight100_seed0'),
)
OCCLUSION_METHOD_ABLATION = (
    ('occlusion weight=0.25',
     'goose_temporal_occlusion_common10_3h_weight025_seed0'),
    ('occlusion weight=0.50',
     'goose_temporal_occlusion_common10_3h_seed0'),
    ('occlusion weight=1.00',
     'goose_temporal_occlusion_common10_3h_weight100_seed0'),
)


def occlusion_weight(experiment):
    if 'weight025' in experiment:
        return 0.25
    if 'weight100' in experiment:
        return 1.0
    return 0.5


def occlusion_selection_key(record):
    weight = occlusion_weight(record['experiment'])
    return (
        record['O-SSC-mIoU'],
        -abs(weight - 0.5),
        -weight,
    )
FINAL_TEMPORAL_CANDIDATES = (
    ('1 history, weighted conf., weight=0.75',
     'goose_temporal_common10_1h_seed0'),
    ('3 history, weighted conf., weight=0.75',
     'goose_temporal_common10_3h_seed0'),
    ('5 history, weighted conf., weight=0.75',
     'goose_temporal_common10_5h_seed0'),
    ('10 history, weighted conf., weight=0.75',
     'goose_temporal_common10_10h_seed0'),
    ('3 history, binary conf., weight=0.75',
     'goose_temporal_common10_3h_binary_conf_seed0'),
    ('3 history, weighted conf., weight=0.25',
     'goose_temporal_common10_3h_weight025_seed0'),
    ('3 history, weighted conf., weight=1.00',
     'goose_temporal_common10_3h_weight100_seed0'),
)

PROTOCOL_METADATA = (
    'experiment_role',
    'dataset_type',
    'seed',
    'num_history',
    'min_history',
    'train_samples',
    'physical_global_batch',
    'gradient_accumulation',
    'effective_global_batch',
    'epochs',
    'micro_iterations',
    'optimizer_updates',
    'confidence_mode',
    'history_loss_weight',
    'occlusion_loss_weight',
    'max_train_memory_mb_per_gpu',
    'training_wall_time_sec',
)

AGGREGATE_METRICS = tuple(dict.fromkeys(
    ('SC_IoU', 'SSC_mIoU')
    + VISIBILITY_METRICS
    + tuple(
        '{}_{}'.format(metric, distance_bin)
        for distance_bin in DISTANCE_BINS
        for metric in VISIBILITY_METRICS
    )
    + tuple(metric for _, metric in CLASS_METRICS)
))
EFFICIENCY_FIELDS = (
    'elapsed_sec',
    'max_train_memory_mb_per_gpu',
    'training_wall_time_sec',
)

DIAGNOSTIC_EXPERIMENTS = {
    'goose_single_frame_v14_2gpu',
    'goose_temporal_3h_2gpu',
}
THREE_SEEDS = (0, 1, 2)
FIVE_SEEDS = (0, 1, 2, 3, 4)
TEMPORAL_CONTROL_STEM = 'goose_temporal_common10_3h_binary_conf'


def experiment_role(experiment):
    if experiment in DIAGNOSTIC_EXPERIMENTS:
        return 'diagnostic: budget-mismatched'
    canonical = re.sub(r'_seed[1-4]$', '_seed0', experiment)
    if canonical in {name for _, name in FAIR_COMPARISON}:
        return 'fair main comparison'
    if canonical in {name for _, name in HISTORY_ABLATION}:
        return 'common-target history ablation'
    if canonical in {name for _, name in SUPERVISION_ABLATION}:
        return 'supervision ablation'
    if canonical in {name for _, name in OCCLUSION_METHOD_ABLATION}:
        return 'pose-guided occlusion ablation'
    return 'unclassified'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', default='work_dir')
    parser.add_argument(
        '--csv-output', default='work_dir/goose_experiment_summary.csv')
    parser.add_argument(
        '--markdown-output', default='docs/GOOSE_RESULTS.md')
    parser.add_argument(
        '--multiseed-csv-output',
        default='work_dir/goose_experiment_multiseed_summary.csv')
    parser.add_argument(
        '--paired-delta-csv-output',
        default='work_dir/goose_experiment_paired_deltas.csv')
    parser.add_argument(
        '--surroundocc-result',
        default=(
            '/path/to/SurroundOcc/work_dirs/'
            'surroundocc_goose_common10/goose_visibility_val.json'))
    parser.add_argument(
        '--external-baseline-csv-output',
        default='work_dir/goose_external_baseline_summary.csv')
    return parser.parse_args()


def load_protocol_metadata(run_dir):
    config_path = run_dir / 'resolved_config.json'
    if not config_path.is_file():
        return {}
    with config_path.open('r', encoding='utf-8') as handle:
        config = json.load(handle)

    train_config = config.get('data', {}).get('train', {})
    head_config = config.get('model', {}).get('pts_bbox_head', {})
    optimizer_config = config.get('optimizer_config', {})
    physical_batch = config.get('batch_size', '')
    accumulation = optimizer_config.get('cumulative_iters', 1)
    epochs = config.get('total_epochs', '')

    confidence_mode = 'weighted'
    for transform in train_config.get('pipeline', []):
        if transform.get('type') == 'LoadGOOSE2DSupervision':
            confidence_mode = transform.get('confidence_mode', 'weighted')
            break

    train_samples = ''
    seed = ''
    max_train_memory = ''
    training_wall_time = ''
    log_path = run_dir / 'train.log'
    if log_path.is_file():
        log_text = log_path.read_text(encoding='utf-8', errors='replace')
        sample_matches = re.findall(
            r'\[GOOSE(?:Temporal)?OccMono\] train: (\d+)', log_text)
        if sample_matches:
            train_samples = int(sample_matches[-1])
        elif isinstance(physical_batch, int):
            iteration_matches = re.findall(
                r'Epoch \[\d+\]\[\d+/(\d+)\]', log_text)
            if iteration_matches:
                train_samples = int(iteration_matches[-1]) * physical_batch
        seed_matches = re.findall(r'Setting random seed: (\d+)', log_text)
        if seed_matches:
            seed = int(seed_matches[-1])
        memory_matches = re.findall(r'memory: (\d+)', log_text)
        if memory_matches:
            max_train_memory = max(int(value) for value in memory_matches)
        start_matches = re.findall(
            r'^\[([^]]+)\]\[INFO\] - Start running,',
            log_text, flags=re.MULTILINE)
        finish_matches = re.findall(
            r'^\[([^]]+)\]\[INFO\] - Epoch\(val\) \[24\]',
            log_text, flags=re.MULTILINE)
        if start_matches and finish_matches:
            timestamp_format = '%Y-%m-%d %H:%M:%S,%f'
            started = datetime.strptime(start_matches[-1], timestamp_format)
            finished = datetime.strptime(finish_matches[-1], timestamp_format)
            training_wall_time = (finished - started).total_seconds()

    effective_batch = ''
    micro_iterations = ''
    optimizer_updates = ''
    if isinstance(physical_batch, int) and isinstance(accumulation, int):
        effective_batch = physical_batch * accumulation
        if isinstance(train_samples, int) and isinstance(epochs, int):
            iterations_per_epoch = (
                train_samples + physical_batch - 1) // physical_batch
            micro_iterations = iterations_per_epoch * epochs
            optimizer_updates = (
                micro_iterations + accumulation - 1) // accumulation

    return {
        'dataset_type': config.get('dataset_type', ''),
        'seed': seed,
        'num_history': train_config.get('num_history', 0),
        'min_history': train_config.get('min_history', 0),
        'train_samples': train_samples,
        'physical_global_batch': physical_batch,
        'gradient_accumulation': accumulation,
        'effective_global_batch': effective_batch,
        'epochs': epochs,
        'micro_iterations': micro_iterations,
        'optimizer_updates': optimizer_updates,
        'confidence_mode': confidence_mode,
        'history_loss_weight': head_config.get('history_loss_weight', ''),
        'occlusion_loss_weight': head_config.get('occlusion_loss_weight', 0.0),
        'max_train_memory_mb_per_gpu': max_train_memory,
        'training_wall_time_sec': training_wall_time,
    }


def load_records(work_dir):
    records = []
    paths_by_run = {}
    for path in sorted(Path(work_dir).glob('**/val*.json')):
        current = paths_by_run.get(path.parent)
        if current is None or path.name == 'val_best.json':
            paths_by_run[path.parent] = path
    for path in sorted(paths_by_run.values()):
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        metrics = payload.get('metrics')
        if not isinstance(metrics, dict):
            continue
        record = {
            'experiment': path.parent.name,
            'experiment_role': experiment_role(path.parent.name),
            'result_json': str(path),
            'config': payload.get('config', ''),
            'weights': payload.get('weights', ''),
            'num_samples': payload.get('num_samples', ''),
            'elapsed_sec': payload.get('elapsed_sec', ''),
        }
        record.update(load_protocol_metadata(path.parent))
        record.update(metrics)
        records.append(record)
    return records


def as_percent(value):
    return 100.0 * value if isinstance(value, (float, int)) else ''


def load_surroundocc_upper_bound(path):
    path = Path(path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('num_samples') != 469 or payload.get(
            'dataset_samples') != 469:
        raise ValueError(
            'Formal SurroundOcc result must evaluate exactly 469 targets')
    contract = payload.get('prediction_contract', {})
    if contract.get('shape') != [200, 128, 32] \
            or contract.get('axis_order') != 'x,y,z' \
            or contract.get('class_ids') != '0..7':
        raise ValueError('Invalid SurroundOcc prediction contract')

    standard = payload.get('standard_unknown_as_free', {})
    visibility = payload.get('visibility', {})
    record = {
        'experiment': 'surroundocc_goose_common10',
        'experiment_role': 'fully supervised OCC upper bound',
        'supervision': payload.get('supervision', ''),
        'result_json': str(path),
        'config': payload.get('config', ''),
        'weights': payload.get('checkpoint', ''),
        'num_samples': payload.get('num_samples', ''),
        'SC_IoU': as_percent(standard.get('SC-IoU')),
        'SSC_mIoU': as_percent(standard.get('SSC-mIoU')),
    }
    for metric in VISIBILITY_METRICS:
        record[metric] = as_percent(visibility.get(metric))
    for distance_bin in DISTANCE_BINS:
        domains = visibility.get(
            'by_horizontal_range', {}).get(distance_bin, {})
        for domain_name, prefix in (
                ('visible', 'V'), ('occluded', 'O')):
            domain = domains.get(domain_name, {})
            for metric in ('SC-IoU', 'SSC-mIoU'):
                record['{}-{}_{}'.format(
                    prefix, metric, distance_bin)] = as_percent(
                        domain.get(metric))
    per_class = standard.get('per_class', {})
    source_names = {
        'IoU_free': 'free',
        'IoU_artificial_structures': 'artificial_structures',
        'IoU_artificial_ground': 'artificial_ground',
        'IoU_natural_ground': 'natural_ground',
        'IoU_obstacle': 'obstacle',
        'IoU_vehicle': 'vehicle',
        'IoU_vegetation': 'vegetation',
        'IoU_human': 'human',
    }
    for metric_name, class_name in source_names.items():
        record[metric_name] = as_percent(
            per_class.get(class_name, {}).get('iou'))
    return [record]


def write_external_baseline_csv(records, output_path):
    metadata = (
        'experiment', 'experiment_role', 'supervision', 'result_json',
        'config', 'weights', 'num_samples',
    )
    metrics = CORE_METRICS + tuple(
        '{}_{}'.format(metric, distance_bin)
        for distance_bin in DISTANCE_BINS
        for metric in VISIBILITY_METRICS
    ) + tuple(metric for _, metric in CLASS_METRICS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=metadata + metrics, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def write_csv(records, output_path):
    metadata = [
        'experiment', 'result_json', 'config', 'weights',
        'num_samples', 'elapsed_sec',
    ] + list(PROTOCOL_METADATA)
    metric_names = sorted({
        key for record in records for key in record if key not in metadata
    })
    fieldnames = metadata + metric_names
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def build_multiseed_records(records):
    records_by_name = {record['experiment']: record for record in records}
    candidates = complete_group(OCCLUSION_METHOD_ABLATION, records_by_name)
    selected_method_stem = None
    if candidates:
        _, selected = max(
            candidates, key=lambda item: occlusion_selection_key(item[1]))
        selected_method_stem = selected['experiment'][:-len('_seed0')]
    stems = set()
    for experiment in records_by_name:
        match = re.match(
            r'^(goose_(?:mono|temporal(?:_occlusion)?)_common10.*)_seed[0-4]$',
                         experiment)
        if match:
            stems.add(match.group(1))

    aggregates = []
    for stem in sorted(stems):
        seeds = (FIVE_SEEDS if stem in {
            TEMPORAL_CONTROL_STEM, selected_method_stem
        } else THREE_SEEDS)
        seed_records = [
            records_by_name.get('{}_seed{}'.format(stem, seed))
            for seed in seeds
        ]
        if any(record is None for record in seed_records):
            continue
        aggregate = {
            'variant': 'mono' if stem == 'goose_mono_common10' else stem,
            'run_stem': stem,
            'num_seeds': len(seeds),
            'experiments': ','.join(
                record['experiment'] for record in seed_records),
        }
        complete = True
        for metric in AGGREGATE_METRICS:
            values = [record.get(metric) for record in seed_records]
            if not all(isinstance(value, (float, int)) for value in values):
                complete = False
                break
            aggregate['{}_mean'.format(metric)] = statistics.mean(values)
            aggregate['{}_std'.format(metric)] = statistics.stdev(values)
        for field in EFFICIENCY_FIELDS:
            values = [record.get(field) for record in seed_records]
            if all(isinstance(value, (float, int)) for value in values):
                aggregate['{}_mean'.format(field)] = statistics.mean(values)
                aggregate['{}_std'.format(field)] = statistics.stdev(values)
        if complete:
            aggregates.append(aggregate)
    return aggregates


def write_multiseed_csv(records, output_path):
    metadata = ['variant', 'run_stem', 'num_seeds', 'experiments']
    metric_fields = [
        field for metric in AGGREGATE_METRICS
        for field in ('{}_mean'.format(metric), '{}_std'.format(metric))
    ]
    metric_fields.extend(
        field for metric in EFFICIENCY_FIELDS
        for field in ('{}_mean'.format(metric), '{}_std'.format(metric)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata + metric_fields)
        writer.writeheader()
        writer.writerows(records)


def build_paired_delta_records(records):
    record_by_name = {record['experiment']: record for record in records}
    candidates = complete_group(OCCLUSION_METHOD_ABLATION, record_by_name)
    if not candidates:
        return []
    _, selected = max(
        candidates, key=lambda item: occlusion_selection_key(item[1]))
    selected_stem = selected['experiment'][:-len('_seed0')]
    baseline_stem = TEMPORAL_CONTROL_STEM
    pairs = []
    for seed in FIVE_SEEDS:
        baseline = record_by_name.get('{}_seed{}'.format(
            baseline_stem, seed))
        method = record_by_name.get('{}_seed{}'.format(
            selected_stem, seed))
        if baseline is None or method is None:
            return []
        pairs.append((baseline, method))

    output = []
    for metric in AGGREGATE_METRICS:
        deltas = [
            method.get(metric) - baseline.get(metric)
            for baseline, method in pairs
            if isinstance(method.get(metric), (float, int))
            and isinstance(baseline.get(metric), (float, int))
        ]
        if len(deltas) != len(FIVE_SEEDS):
            continue
        output.append({
            'baseline_stem': baseline_stem,
            'method_stem': selected_stem,
            'selection_rule': 'seed-0 O-SSC-mIoU',
            'metric': metric,
            'num_paired_seeds': len(FIVE_SEEDS),
            **{
                'seed{}_delta'.format(seed): delta
                for seed, delta in zip(FIVE_SEEDS, deltas)
            },
            'delta_mean': statistics.mean(deltas),
            'delta_std': statistics.stdev(deltas),
        })
    return output


def write_paired_delta_csv(records, output_path):
    fieldnames = (
        'baseline_stem', 'method_stem', 'selection_rule', 'metric',
        'num_paired_seeds',
    ) + tuple(
        'seed{}_delta'.format(seed) for seed in FIVE_SEEDS
    ) + (
        'delta_mean', 'delta_std',
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def display_value(value):
    if isinstance(value, (float, int)):
        return '{:.2f}'.format(value)
    return str(value) if value != '' else '-'


def display_mean_std(record, metric):
    mean = record.get('{}_mean'.format(metric))
    std = record.get('{}_std'.format(metric))
    if not isinstance(mean, (float, int)) or not isinstance(std, (float, int)):
        return '-'
    return '{:.2f} +/- {:.2f}'.format(mean, std)


def append_table(lines, columns, records, metric_keys=None):
    lines.extend([
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ])
    keys = metric_keys if metric_keys is not None else columns
    for record in records:
        lines.append('| ' + ' | '.join(
            display_value(record.get(key, '')) for key in keys
        ) + ' |')


def append_experiment_group(lines, title, definitions, record_by_name):
    group_records = []
    for label, experiment in definitions:
        source = record_by_name.get(experiment)
        if source is None:
            continue
        record = dict(source)
        record['variant'] = label
        group_records.append(record)
    if not group_records:
        return
    columns = (
        'variant', 'SC_IoU', 'SSC_mIoU', 'V-SC-IoU', 'V-SSC-mIoU',
        'O-SC-IoU', 'O-SSC-mIoU', 'O-SSC-mIoU_0-10m',
        'O-SSC-mIoU_10-25m', 'O-SSC-mIoU_25-50m',
    )
    lines.extend(['', '## {}'.format(title), ''])
    append_table(lines, columns, group_records)


def complete_group(definitions, record_by_name):
    records = []
    for label, experiment in definitions:
        record = record_by_name.get(experiment)
        if record is None:
            return []
        records.append((label, record))
    return records


def append_derived_comparisons(lines, record_by_name):
    conclusions = []
    fair_records = complete_group(FAIR_COMPARISON, record_by_name)
    if fair_records:
        mono = fair_records[0][1]
        temporal = fair_records[1][1]
        delta_metrics = (
            'SC_IoU', 'SSC_mIoU', 'V-SC-IoU', 'V-SSC-mIoU',
            'O-SC-IoU', 'O-SSC-mIoU', 'O-SSC-mIoU_0-10m',
            'O-SSC-mIoU_10-25m', 'O-SSC-mIoU_25-50m',
        )
        deltas = []
        for metric in delta_metrics:
            if isinstance(mono.get(metric), (float, int)) and isinstance(
                    temporal.get(metric), (float, int)):
                deltas.append('{} {:+.2f}'.format(
                    metric, temporal[metric] - mono[metric]))
        conclusions.append(
            '**Seed-0 fair target+3-history minus mono:** '
            + ', '.join(deltas) + '.')

    history_records = complete_group(HISTORY_ABLATION, record_by_name)
    if history_records:
        best_value = max(
            record['O-SSC-mIoU'] for _, record in history_records)
        best_labels = [
            label for label, record in history_records
            if record['O-SSC-mIoU'] == best_value
        ]
        conclusions.append(
            '**Seed-0 history-count selection by O-SSC-mIoU:** {} '
            '({:.2f}).'.format(' / '.join(best_labels), best_value))

    supervision_records = complete_group(
        SUPERVISION_ABLATION, record_by_name)
    if supervision_records:
        best_value = max(
            record['O-SSC-mIoU'] for _, record in supervision_records)
        best_labels = [
            label for label, record in supervision_records
            if record['O-SSC-mIoU'] == best_value
        ]
        conclusions.append(
            '**Seed-0 supervision-ablation selection by O-SSC-mIoU:** {} '
            '({:.2f}).'.format(' / '.join(best_labels), best_value))

    occlusion_records = complete_group(
        OCCLUSION_METHOD_ABLATION, record_by_name)
    if occlusion_records:
        best_label, best_record = max(
            occlusion_records,
            key=lambda item: occlusion_selection_key(item[1]))
        best_value = best_record['O-SSC-mIoU']
        conclusions.append(
            '**Predeclared pose-guided weight selection by seed-0 '
            'O-SSC-mIoU:** {} ({:.2f}).'.format(
                best_label, best_value))

    final_candidates = complete_group(
        FINAL_TEMPORAL_CANDIDATES, record_by_name)
    if final_candidates:
        best_value = max(
            record['O-SSC-mIoU'] for _, record in final_candidates)
        best_labels = [
            label for label, record in final_candidates
            if record['O-SSC-mIoU'] == best_value
        ]
        conclusions.append(
            '**Predeclared temporal configuration repeated at non-selection '
            'seeds:** {} '
            '({:.2f} seed-0 O-SSC-mIoU).'.format(
                ' / '.join(best_labels), best_value))

    if conclusions:
        lines.extend(['', '## Automatically derived comparisons', ''])
        lines.append(
            'These seed-0 comparisons select the protocol. Multi-seed rows '
            'below are emitted only when every seed required by that variant '
            'is available.')
        lines.append('')
        lines.extend('- ' + conclusion for conclusion in conclusions)


def write_markdown(records, multiseed_records, paired_deltas, output_path,
                   external_records=()):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ('experiment', 'experiment_role', 'num_samples') + CORE_METRICS
    lines = [
        '# GOOSE-OCC Experiment Results',
        '',
        'Primary checkpoint-selection metric: `O-SSC-mIoU`.',
        '',
        '## Overall results',
        '',
    ]
    append_table(lines, columns, records)

    if external_records:
        lines.extend([
            '',
            '## Fully supervised OCC upper bound',
            '',
            'This result uses OCC GT during training and is not part of the '
            'weak-supervision method selection or multi-seed aggregates.',
            '',
        ])
        append_table(lines, columns, external_records)

    record_by_name = {record['experiment']: record for record in records}
    append_experiment_group(
        lines, 'Fair matched mono/temporal comparison', FAIR_COMPARISON,
        record_by_name)
    append_experiment_group(
        lines, 'Common-target history-count ablation', HISTORY_ABLATION,
        record_by_name)
    append_experiment_group(
        lines, 'Three-history supervision ablation', SUPERVISION_ABLATION,
        record_by_name)
    append_experiment_group(
        lines, 'Pose-guided occlusion supervision ablation',
        OCCLUSION_METHOD_ABLATION, record_by_name)
    append_derived_comparisons(lines, record_by_name)

    if multiseed_records:
        aggregate_columns = ('variant', 'num_seeds') + CORE_METRICS
        lines.extend([
            '',
            '## Multi-seed results (mean +/- sample standard deviation)',
            '',
            '| ' + ' | '.join(aggregate_columns) + ' |',
            '| ' + ' | '.join(['---'] * len(aggregate_columns)) + ' |',
        ])
        for record in multiseed_records:
            lines.append('| ' + ' | '.join(
                [record['variant'], str(record['num_seeds'])]
                + [display_mean_std(record, metric) for metric in CORE_METRICS]
            ) + ' |')

        lines.extend(['', '### Multi-seed per-class IoU', ''])
        class_columns = ('variant', 'num_seeds') + tuple(
            display_name for display_name, _ in CLASS_METRICS)
        lines.extend([
            '| ' + ' | '.join(class_columns) + ' |',
            '| ' + ' | '.join(['---'] * len(class_columns)) + ' |',
        ])
        for record in multiseed_records:
            lines.append('| ' + ' | '.join(
                [record['variant'], str(record['num_seeds'])]
                + [display_mean_std(record, metric)
                   for _, metric in CLASS_METRICS]
            ) + ' |')

        lines.extend(['', '### Multi-seed efficiency', ''])
        efficiency_columns = ('variant', 'num_seeds') + EFFICIENCY_FIELDS
        lines.extend([
            '| ' + ' | '.join(efficiency_columns) + ' |',
            '| ' + ' | '.join(['---'] * len(efficiency_columns)) + ' |',
        ])
        for record in multiseed_records:
            lines.append('| ' + ' | '.join(
                [record['variant'], str(record['num_seeds'])]
                + [display_mean_std(record, field)
                   for field in EFFICIENCY_FIELDS]
            ) + ' |')

    paired_by_metric = {
        record['metric']: record for record in paired_deltas
    }
    paired_core = [
        paired_by_metric[metric] for metric in CORE_METRICS
        if metric in paired_by_metric
    ]
    if paired_core:
        lines.extend([
            '',
            '## Paired-seed method minus temporal baseline',
            '',
            'The method weight is fixed by seed-0 `O-SSC-mIoU` before this '
            'comparison. Deltas pair identical seeds. With only five seeds, '
            'the table remains a variability audit and is not used to claim '
            'statistical significance.',
            '',
            '| metric | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | '
            'mean +/- sample std |',
            '| --- | --- | --- | --- | --- | --- | --- |',
        ])
        for record in paired_core:
            lines.append(
                '| {metric} | {seed0_delta:.2f} | {seed1_delta:.2f} | '
                '{seed2_delta:.2f} | {seed3_delta:.2f} | '
                '{seed4_delta:.2f} | {delta_mean:.2f} +/- '
                '{delta_std:.2f} |'.format(**record))

    protocol_records = [
        record for record in records if record.get('train_samples', '') != ''
    ]
    if protocol_records:
        protocol_columns = ('experiment',) + PROTOCOL_METADATA
        lines.extend(['', '## Resolved training protocol', ''])
        append_table(lines, protocol_columns, protocol_records)

    distance_columns = ('experiment', 'range') + VISIBILITY_METRICS
    distance_records = []
    for record in records:
        for distance_bin in DISTANCE_BINS:
            distance_record = {
                'experiment': record['experiment'],
                'range': distance_bin,
            }
            for metric in VISIBILITY_METRICS:
                distance_record[metric] = record.get(
                    '{}_{}'.format(metric, distance_bin), '')
            distance_records.append(distance_record)
    lines.extend(['', '## Visibility metrics by distance', ''])
    append_table(lines, distance_columns, distance_records)

    class_columns = ('experiment',) + tuple(
        display_name for display_name, _ in CLASS_METRICS)
    class_keys = ('experiment',) + tuple(
        metric_name for _, metric_name in CLASS_METRICS)
    lines.extend(['', '## Per-class IoU', ''])
    append_table(lines, class_columns, records, class_keys)

    runtime_records = []
    for record in records:
        elapsed = record.get('elapsed_sec', '')
        samples = record.get('num_samples', '')
        samples_per_sec = ''
        if isinstance(elapsed, (float, int)) and elapsed > 0 \
                and isinstance(samples, (float, int)):
            samples_per_sec = samples / elapsed
        runtime_records.append({
            'experiment': record['experiment'],
            'num_samples': samples,
            'elapsed_sec': elapsed,
            'samples_per_sec': samples_per_sec,
        })
    lines.extend(['', '## Validation runtime', ''])
    append_table(
        lines,
        ('experiment', 'num_samples', 'elapsed_sec', 'samples_per_sec'),
        runtime_records,
    )

    lines.extend([
        '',
        'The CSV contains every standard, FOV, per-class, V/O, and '
        'distance-binned metric emitted by validation, together with the '
        'resolved config and checkpoint paths.',
        '',
    ])
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    args = parse_args()
    records = load_records(args.work_dir)
    if not records:
        print('No validation JSON files found under {}'.format(args.work_dir))
        return
    csv_path = Path(args.csv_output)
    markdown_path = Path(args.markdown_output)
    multiseed_csv_path = Path(args.multiseed_csv_output)
    paired_delta_csv_path = Path(args.paired_delta_csv_output)
    external_csv_path = Path(args.external_baseline_csv_output)
    multiseed_records = build_multiseed_records(records)
    paired_deltas = build_paired_delta_records(records)
    external_records = load_surroundocc_upper_bound(args.surroundocc_result)
    write_csv(records, csv_path)
    write_multiseed_csv(multiseed_records, multiseed_csv_path)
    write_paired_delta_csv(paired_deltas, paired_delta_csv_path)
    write_external_baseline_csv(external_records, external_csv_path)
    write_markdown(
        records, multiseed_records, paired_deltas, markdown_path,
        external_records)
    print('Summarized {} results ({} complete multi-seed groups, {} paired '
          'deltas, {} external baselines) to {}, {}, {}, {}, and {}'.format(
              len(records), len(multiseed_records), len(paired_deltas),
              len(external_records), csv_path, multiseed_csv_path,
              paired_delta_csv_path, external_csv_path, markdown_path))


if __name__ == '__main__':
    main()
