#!/usr/bin/env python3
"""Robot-centric diagnostics on frozen GOOSE-OCC predictions.

Observation masks and occupancy labels are opened only by this evaluation
script. They are never exposed to the weak-training pipeline or losses.
"""

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/path/to/GOOSE-OCC')
OCC_SHAPE = (200, 128, 32)
PC_RANGE = (0.0, -16.0, -4.0, 50.0, 16.0, 4.0)
VOXEL_SIZE = 0.25
HAZARD_CLASS_IDS = (1, 4, 5, 6, 7)
RANGE_BINS = ((0.0, 10.0), (10.0, 25.0))
CORRIDOR = {
    'x_m': (0.0, 25.0),
    'abs_y_max_m': 1.5,
    'z_m': (-2.0, 2.0),
}
WEAK_RUNS = (
    'goose_mono_common10_seed0',
    'goose_mono_common10_seed1',
    'goose_mono_common10_seed2',
    'goose_temporal_common10_3h_binary_conf_seed0',
    'goose_temporal_common10_3h_binary_conf_seed1',
    'goose_temporal_common10_3h_binary_conf_seed2',
    'goose_temporal_common10_3h_binary_conf_seed3',
    'goose_temporal_common10_3h_binary_conf_seed4',
    'goose_temporal_occlusion_common10_3h_seed0',
    'goose_temporal_occlusion_common10_3h_seed1',
    'goose_temporal_occlusion_common10_3h_seed2',
    'goose_temporal_occlusion_common10_3h_seed3',
    'goose_temporal_occlusion_common10_3h_seed4',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_visibility_module():
    path = ROOT / 'loaders/goose_visibility_metrics.py'
    spec = importlib.util.spec_from_file_location(
        'goose_robot_safety_visibility', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pickle(path):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def common_val_infos(data_root=DATA_ROOT, min_history=10):
    base = load_pickle(Path(data_root) / 'infos/goose_infos_val.pkl')
    temporal = load_pickle(
        Path(data_root) / 'temporal/temporal_infos_val.pkl')
    eligible = {
        (
            item['sequence'], int(item['frame_index']),
            int(item['target_sensor_stamp_ns']),
        )
        for item in temporal if len(item.get('histories', ())) >= min_history
    }
    infos = [
        item for item in base['infos']
        if (
            item['sequence'], int(item['frame_index']),
            int(item['sensor_stamp_ns']),
        ) in eligible
    ]
    if len(infos) != len(eligible):
        raise ValueError('Base/temporal common-target alignment failed')
    return infos


def densify_occ(path):
    sparse = np.load(path, allow_pickle=False)
    if sparse.ndim != 2 or sparse.shape[1] != 4:
        raise ValueError('Expected sparse OCC array (N, 4): {}'.format(path))
    coords = sparse[:, :3].astype(np.int64, copy=False)
    labels = sparse[:, 3].astype(np.int64, copy=False)
    if coords.size and (
            np.any(coords < 0)
            or np.any(coords >= np.asarray(OCC_SHAPE)[None])):
        raise ValueError('OCC coordinates outside {}'.format(OCC_SHAPE))
    if np.any(~(((labels >= 1) & (labels <= 7)) | (labels == 255))):
        raise ValueError('OCC labels outside 1..7/255')
    dense = np.zeros(OCC_SHAPE, dtype=np.uint8)
    if len(coords):
        dense[coords[:, 0], coords[:, 1], coords[:, 2]] = labels
    return dense


def coordinate_grids():
    x = PC_RANGE[0] + (np.arange(OCC_SHAPE[0]) + 0.5) * VOXEL_SIZE
    y = PC_RANGE[1] + (np.arange(OCC_SHAPE[1]) + 0.5) * VOXEL_SIZE
    z = PC_RANGE[2] + (np.arange(OCC_SHAPE[2]) + 0.5) * VOXEL_SIZE
    radial = np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2)
    corridor = (
        (x[:, None, None] >= CORRIDOR['x_m'][0])
        & (x[:, None, None] < CORRIDOR['x_m'][1])
        & (np.abs(y[None, :, None]) <= CORRIDOR['abs_y_max_m'])
        & (z[None, None, :] >= CORRIDOR['z_m'][0])
        & (z[None, None, :] < CORRIDOR['z_m'][1])
    )
    return radial, corridor


def validate_prediction(prediction):
    prediction = np.asarray(prediction)
    if prediction.shape != OCC_SHAPE:
        raise ValueError(
            'Prediction shape {} != {}'.format(prediction.shape, OCC_SHAPE))
    if prediction.dtype != np.uint8:
        raise ValueError('Prediction dtype {} != uint8'.format(
            prediction.dtype))
    if prediction.size and (
            int(prediction.min()) < 0 or int(prediction.max()) > 7):
        raise ValueError('Prediction contains IDs outside 0..7')


class SafetyAccumulator:
    """Aggregate binary occupancy and corridor-risk counts."""

    def __init__(self):
        self.frames = 0
        self.frames_with_occupied = 0
        self.frames_with_known_free = 0
        self.gt_occupied = 0
        self.detected_occupied = 0
        self.known_free = 0
        self.false_occupied = 0
        self.gt_hazard = 0
        self.detected_hazard = 0

    def update(self, prediction, target, occupied_mask, free_mask):
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        occupied_mask = np.asarray(occupied_mask, dtype=bool)
        free_mask = np.asarray(free_mask, dtype=bool)
        validate_prediction(prediction)
        if target.shape != OCC_SHAPE:
            raise ValueError('Target shape {} != {}'.format(
                target.shape, OCC_SHAPE))
        if occupied_mask.shape != OCC_SHAPE or free_mask.shape != OCC_SHAPE:
            raise ValueError('Safety masks must match occupancy shape')
        if np.any(occupied_mask & free_mask):
            raise ValueError('Occupied and known-free masks overlap')
        if np.any(target[free_mask] != 0):
            raise ValueError('Known-free mask contains non-free target labels')
        occupied_labels = target[occupied_mask]
        if np.any((occupied_labels < 1) | (occupied_labels > 7)):
            raise ValueError(
                'Observed-occupied mask contains non-occupied target labels')

        pred_occupied = prediction > 0
        hazard = occupied_mask & np.isin(target, HAZARD_CLASS_IDS)
        self.frames += 1
        self.frames_with_occupied += int(np.any(occupied_mask))
        self.frames_with_known_free += int(np.any(free_mask))
        self.gt_occupied += int(occupied_mask.sum())
        self.detected_occupied += int((pred_occupied & occupied_mask).sum())
        self.known_free += int(free_mask.sum())
        self.false_occupied += int((pred_occupied & free_mask).sum())
        self.gt_hazard += int(hazard.sum())
        self.detected_hazard += int((pred_occupied & hazard).sum())

    @staticmethod
    def _ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    def result(self):
        occupied_recall = self._ratio(
            self.detected_occupied, self.gt_occupied)
        hazard_recall = self._ratio(self.detected_hazard, self.gt_hazard)
        return {
            'occupied_recall': occupied_recall,
            'occupied_miss_rate': (
                None if occupied_recall is None else 1.0 - occupied_recall),
            'false_occupied_known_free_rate': self._ratio(
                self.false_occupied, self.known_free),
            'hazard_recall': hazard_recall,
            'hazard_miss_rate': (
                None if hazard_recall is None else 1.0 - hazard_recall),
            'counts': {
                'gt_occupied_voxels': self.gt_occupied,
                'detected_occupied_voxels': self.detected_occupied,
                'known_free_voxels': self.known_free,
                'false_occupied_voxels': self.false_occupied,
                'gt_hazard_voxels': self.gt_hazard,
                'detected_hazard_voxels': self.detected_hazard,
                'frames': self.frames,
                'frames_with_occupied': self.frames_with_occupied,
                'frames_with_known_free': self.frames_with_known_free,
            },
        }


class RobotSafetyMetrics:
    def __init__(self):
        self.radial, self.corridor = coordinate_grids()
        self.domains = {
            'visible': SafetyAccumulator(),
            'occluded': SafetyAccumulator(),
        }
        self.by_range = {
            '{}-{}m'.format(int(lower), int(upper)): {
                'visible': SafetyAccumulator(),
                'occluded': SafetyAccumulator(),
            }
            for lower, upper in RANGE_BINS
        }
        self.ego_corridor = SafetyAccumulator()
        self.frames = 0

    def update(self, prediction, target, masks):
        pairs = {
            'visible': (
                masks['visible_occupied_mask'],
                masks['visible_known_free_mask'],
            ),
            'occluded': (
                masks['occluded_observed_mask'],
                masks['occluded_known_free_mask'],
            ),
        }
        for domain, (occupied, free) in pairs.items():
            self.domains[domain].update(prediction, target, occupied, free)
            for lower, upper in RANGE_BINS:
                radial = (self.radial >= lower) & (self.radial < upper)
                key = '{}-{}m'.format(int(lower), int(upper))
                self.by_range[key][domain].update(
                    prediction, target, occupied & radial, free & radial)

        occupied = pairs['visible'][0] | pairs['occluded'][0]
        free = pairs['visible'][1] | pairs['occluded'][1]
        self.ego_corridor.update(
            prediction, target, occupied & self.corridor,
            free & self.corridor)
        self.frames += 1

    def result(self):
        return {
            'frames': self.frames,
            'domains': {
                key: value.result() for key, value in self.domains.items()
            },
            'by_horizontal_range': {
                key: {
                    domain: accumulator.result()
                    for domain, accumulator in values.items()
                }
                for key, values in self.by_range.items()
            },
            'ego_corridor': self.ego_corridor.result(),
            'definitions': {
                'occupied': 'GOOSE class ID 1..7',
                'hazard_class_ids': list(HAZARD_CLASS_IDS),
                'hazard_classes': [
                    'artificial_structures', 'obstacle', 'vehicle',
                    'vegetation', 'human',
                ],
                'hazard_note': (
                    'Artificial and natural ground (IDs 2 and 3) remain '
                    'occupied for occupied recall but are excluded from the '
                    'corridor hazard subset.'),
                'known_free_negative': (
                    'Visible-known-free or occluded-known-free evidence only; '
                    'unknown voxels are excluded.'),
                'horizontal_range': (
                    'Horizontal radial distance from target LiDAR origin to '
                    'voxel center.'),
                'ego_corridor': CORRIDOR,
                'interpretation_boundary': (
                    'Voxel-level diagnostic only; not a planner or navigation '
                    'success metric.'),
            },
        }


def load_weak_prediction(path, expected_index, expected_token):
    with np.load(path, allow_pickle=False) as payload:
        if str(payload['axis_order'].item()) != 'x,y,z':
            raise ValueError('{} has invalid axis order'.format(path))
        if int(payload['sample_index'].item()) != expected_index:
            raise ValueError('{} has invalid sample index'.format(path))
        if str(payload['sample_token'].item()) != expected_token:
            raise ValueError('{} token does not match infos'.format(path))
        prediction = payload['occ_pred']
    validate_prediction(prediction)
    return prediction


def load_lifting_prediction(path):
    with np.load(path, allow_pickle=False) as payload:
        prediction = payload['prediction']
    validate_prediction(prediction)
    return prediction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument(
        '--weak-root', type=Path,
        default=ROOT / 'work_dir/formal_predictions')
    parser.add_argument(
        '--lifting-root', type=Path,
        default=ROOT / 'work_dir/goose_geometric_lifting_common10/predictions')
    parser.add_argument(
        '--surround-root', type=Path,
        default=Path('/path/to/SurroundOcc/work_dirs/'
                     'surroundocc_goose_common10/predictions'))
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'work_dir/goose_robot_safety.json')
    return parser.parse_args()


def main():
    args = parse_args()
    infos = common_val_infos(args.data_root)
    if len(infos) != 469:
        raise ValueError('Expected 469 common validation targets')
    visibility = load_visibility_module()
    run_names = list(WEAK_RUNS) + [
        'goose_geometric_lifting_common10',
        'surroundocc_goose_common10',
    ]
    metrics = {name: RobotSafetyMetrics() for name in run_names}

    for index, info in enumerate(infos):
        target = densify_occ(args.data_root / info['occ_path'])
        masks = visibility.load_observation(
            args.data_root / info['observation_path'])
        token = info['token']
        for run in WEAK_RUNS:
            path = args.weak_root / run / 'prediction_{:04d}.npz'.format(index)
            prediction = load_weak_prediction(path, index, token)
            metrics[run].update(prediction, target, masks)

        lifting_path = args.lifting_root / 'prediction_{:04d}.npz'.format(index)
        metrics['goose_geometric_lifting_common10'].update(
            load_lifting_prediction(lifting_path), target, masks)

        surround_path = args.surround_root / info['occ_path']
        surround_prediction = np.load(surround_path, allow_pickle=False)
        validate_prediction(surround_prediction)
        metrics['surroundocc_goose_common10'].update(
            surround_prediction, target, masks)
        if (index + 1) % 50 == 0 or index + 1 == len(infos):
            print('safety evaluation {}/{}'.format(index + 1, len(infos)),
                  flush=True)

    result = {
        'protocol': 'goose_occ_robot_safety_v1',
        'evaluation_only': True,
        'training_use_of_occ_or_observation': False,
        'population': {
            'split': 'val',
            'filter': 'at least 10 real causal camera histories',
            'targets': len(infos),
        },
        'prediction_contract': {
            'shape': list(OCC_SHAPE),
            'axis_order': 'x,y,z',
            'dtype': 'uint8',
            'class_ids': '0..7',
        },
        'runs': {name: accumulator.result()
                 for name, accumulator in metrics.items()},
        'source_hashes': {
            'base_infos_val': sha256(
                args.data_root / 'infos/goose_infos_val.pkl'),
            'temporal_infos_val': sha256(
                args.data_root / 'temporal/temporal_infos_val.pkl'),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n',
                           encoding='utf-8')
    print('Wrote {}'.format(args.output))


if __name__ == '__main__':
    main()
