"""GOOSE-OCC v1.4 observation-mask loading and V/O metrics.

The definitions mirror SurroundOcc's ``goose_observation.py`` and
``goose_visibility_metrics.py``. Observation masks are evaluation-only data.
"""

import numpy as np


PC_RANGE = np.asarray(
    [0.0, -16.0, -4.0, 50.0, 16.0, 4.0], dtype=np.float64)
VOXEL_SIZE = 0.25
OCC_SIZE = (200, 128, 32)
OBSERVATION_FORMAT = 'goose_occ_observation_v1'
CLASS_NAMES = (
    'free',
    'artificial_structures',
    'artificial_ground',
    'natural_ground',
    'obstacle',
    'vehicle',
    'vegetation',
    'human',
)
RANGE_BINS = ((0.0, 10.0), (10.0, 25.0), (25.0, 50.0))
EVALUATION_MASKS = (
    'visible_known_free_mask',
    'visible_occupied_mask',
    'occluded_known_free_mask',
    'occluded_observed_mask',
)


def unpack_mask(packed, shape=OCC_SIZE):
    flat = np.unpackbits(
        np.asarray(packed, dtype=np.uint8),
        bitorder='little',
        count=int(np.prod(shape)),
    )
    return flat.reshape(shape).astype(bool, copy=False)


def load_observation(path, names=EVALUATION_MASKS):
    with np.load(path, allow_pickle=False) as payload:
        if str(payload['format'].item()) != OBSERVATION_FORMAT:
            raise ValueError('Unsupported observation format: {}'.format(path))
        shape = tuple(int(value) for value in payload['shape'])
        if shape != OCC_SIZE:
            raise ValueError('Unexpected observation shape: {}'.format(shape))
        return {name: unpack_mask(payload[name], shape) for name in names}


def valid_domains(masks):
    visible = (
        masks['visible_known_free_mask'] | masks['visible_occupied_mask'])
    occluded = (
        masks['occluded_known_free_mask'] | masks['occluded_observed_mask'])
    return {'visible': visible, 'occluded': occluded}


def horizontal_range_grid():
    x = PC_RANGE[0] + (np.arange(OCC_SIZE[0]) + 0.5) * VOXEL_SIZE
    y = PC_RANGE[1] + (np.arange(OCC_SIZE[1]) + 0.5) * VOXEL_SIZE
    return np.sqrt(x[:, None, None] ** 2 + y[None, :, None] ** 2)


def range_name(lower, upper):
    return '{}-{}m'.format(int(lower), int(upper))


class DomainAccumulator:
    def __init__(self, class_names=CLASS_NAMES):
        self.class_names = tuple(class_names)
        class_count = len(self.class_names)
        self.confusion = np.zeros(
            (class_count, class_count), dtype=np.int64)
        self.valid_voxels = 0
        self.frames_with_valid_voxels = 0

    def update(self, prediction, target, valid):
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        valid = np.asarray(valid, dtype=bool)
        if prediction.shape != OCC_SIZE or target.shape != OCC_SIZE:
            raise ValueError(
                'Prediction and target must have shape {}'.format(OCC_SIZE))
        if valid.shape != OCC_SIZE:
            raise ValueError('Valid mask must have shape {}'.format(OCC_SIZE))
        if not np.any(valid):
            return

        pred = prediction[valid].astype(np.int64, copy=False)
        truth = target[valid].astype(np.int64, copy=False)
        class_count = len(self.class_names)
        if np.any((pred < 0) | (pred >= class_count)):
            raise ValueError('Prediction contains labels outside [0, 7]')
        if np.any((truth < 0) | (truth >= class_count)):
            raise ValueError('Target contains labels outside [0, 7]')
        encoded = truth * class_count + pred
        self.confusion += np.bincount(
            encoded, minlength=class_count * class_count,
        ).reshape(class_count, class_count)
        self.valid_voxels += int(valid.sum())
        self.frames_with_valid_voxels += 1

    def result(self):
        matrix = self.confusion
        occupied_tp = int(matrix[1:, 1:].sum())
        false_occupied = int(matrix[0, 1:].sum())
        missed_occupied = int(matrix[1:, 0].sum())
        occupied_union = occupied_tp + false_occupied + missed_occupied
        sc_iou = occupied_tp / occupied_union if occupied_union else None

        semantic_ious = []
        per_class = {}
        for class_id, class_name in enumerate(self.class_names):
            tp = int(matrix[class_id, class_id])
            fp = int(matrix[:, class_id].sum() - tp)
            fn = int(matrix[class_id, :].sum() - tp)
            union = tp + fp + fn
            iou = tp / union if union else None
            per_class[class_name] = {
                'iou': iou,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'union': union,
            }
            if class_id > 0 and iou is not None:
                semantic_ious.append(tp / union)
        return {
            'SC-IoU': sc_iou,
            'SSC-mIoU': (
                sum(semantic_ious) / len(semantic_ious)
                if semantic_ious else None
            ),
            'per_class': per_class,
            'valid_voxels': self.valid_voxels,
            'gt_occupied_voxels': int(matrix[1:, :].sum()),
            'gt_free_voxels': int(matrix[0, :].sum()),
            'frames_with_valid_voxels': self.frames_with_valid_voxels,
            'confusion_matrix': matrix.tolist(),
        }


class GooseVisibilityMetrics:
    def __init__(self, class_names=CLASS_NAMES, range_bins=RANGE_BINS):
        self.class_names = tuple(class_names)
        self.range_bins = tuple(range_bins)
        self.range_grid = horizontal_range_grid()
        self.overall = {
            'visible': DomainAccumulator(self.class_names),
            'occluded': DomainAccumulator(self.class_names),
        }
        self.by_range = {
            range_name(lower, upper): {
                'visible': DomainAccumulator(self.class_names),
                'occluded': DomainAccumulator(self.class_names),
            }
            for lower, upper in self.range_bins
        }
        self.frames = 0

    def update(self, prediction, target, masks):
        domains = valid_domains(masks)
        for domain_name, valid in domains.items():
            self.overall[domain_name].update(prediction, target, valid)
        for lower, upper in self.range_bins:
            radial = (self.range_grid >= lower) & (self.range_grid < upper)
            bucket = self.by_range[range_name(lower, upper)]
            for domain_name, valid in domains.items():
                bucket[domain_name].update(
                    prediction, target, valid & radial)
        self.frames += 1

    def result(self):
        visible = self.overall['visible'].result()
        occluded = self.overall['occluded'].result()
        return {
            'frames': self.frames,
            'V-SC-IoU': visible['SC-IoU'],
            'V-SSC-mIoU': visible['SSC-mIoU'],
            'O-SC-IoU': occluded['SC-IoU'],
            'O-SSC-mIoU': occluded['SSC-mIoU'],
            'overall': {'visible': visible, 'occluded': occluded},
            'by_horizontal_range': {
                name: {
                    domain: accumulator.result()
                    for domain, accumulator in domains.items()
                }
                for name, domains in self.by_range.items()
            },
            'range_definition': (
                'Horizontal radial distance from target LiDAR origin to '
                'voxel center.'),
        }


def flatten_visibility_results(result):
    output = {}
    for key in ('V-SC-IoU', 'V-SSC-mIoU', 'O-SC-IoU', 'O-SSC-mIoU'):
        value = result[key]
        output[key] = float('nan') if value is None else round(value * 100, 2)
    for range_key, domains in result['by_horizontal_range'].items():
        for short_name, domain_name in (('V', 'visible'), ('O', 'occluded')):
            for metric_name in ('SC-IoU', 'SSC-mIoU'):
                value = domains[domain_name][metric_name]
                key = '{}-{}_{}'.format(short_name, metric_name, range_key)
                output[key] = (
                    float('nan') if value is None else round(value * 100, 2))
    return output
