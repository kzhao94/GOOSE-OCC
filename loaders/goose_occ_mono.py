import os
import pickle
import sys

import numpy as np
import torch
from mmdet.datasets import DATASETS
from mmdet3d.datasets.pipelines import Compose
from torch.utils.data import Dataset
from tqdm import tqdm

from .goose_visibility_metrics import (
    GooseVisibilityMetrics,
    flatten_visibility_results,
    load_observation,
)


OCC_CLASS_NAMES = [
    'free',
    'artificial_structures',
    'artificial_ground',
    'natural_ground',
    'obstacle',
    'vehicle',
    'vegetation',
    'human',
]
NUM_CLASSES = len(OCC_CLASS_NAMES)
OCCUPIED_CLASS_IDS = tuple(range(1, NUM_CLASSES))


def load_goose_pickle(path):
    """Load v1.4 infos under both NumPy 1.x and NumPy 2.x runtimes."""
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def load_goose_occ_semantics(occ_path, occ_size, num_classes=NUM_CLASSES):
    """Densify a GOOSE-OCC sparse ``[x, y, z, class]`` label."""
    occ_size = tuple(int(v) for v in occ_size)
    sparse = np.load(occ_path, allow_pickle=False)
    if sparse.ndim != 2 or sparse.shape[1] != 4:
        raise ValueError('Expected sparse OCC shape (N, 4), got {}: {}'.format(
            sparse.shape, occ_path))

    sparse = sparse.astype(np.int64, copy=False)
    coords = sparse[:, :3]
    labels = sparse[:, 3]
    if coords.size:
        lower_ok = (coords >= 0).all()
        upper_ok = (coords < np.asarray(occ_size)[None]).all()
        if not lower_ok or not upper_ok:
            raise ValueError('Out-of-range voxel coordinate in {}'.format(occ_path))
    valid_labels = ((labels >= 1) & (labels < num_classes)) | (labels == 255)
    if not valid_labels.all():
        raise ValueError('Invalid occupied class ID in {}'.format(occ_path))

    dense = np.zeros(occ_size, dtype=np.int64)
    if len(sparse):
        dense[coords[:, 0], coords[:, 1], coords[:, 2]] = labels
    return dense


class _SSCAccumulator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add(self, pred, gt, mask):
        valid_domain = mask & (gt != 255)
        pred = pred[valid_domain].astype(np.int64, copy=False)
        gt = gt[valid_domain].astype(np.int64, copy=False)
        pred_occupied = pred > 0
        gt_occupied = gt > 0
        self.completion_tp += int((pred_occupied & gt_occupied).sum())
        self.completion_fp += int((pred_occupied & ~gt_occupied).sum())
        self.completion_fn += int((~pred_occupied & gt_occupied).sum())

        valid = (
            (gt >= 0) & (gt < self.num_classes)
            & (pred >= 0) & (pred < self.num_classes)
        )
        self.hist += np.bincount(
            self.num_classes * gt[valid] + pred[valid],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def results(self, prefix=''):
        eps = 1e-8
        precision = self.completion_tp / (
            self.completion_tp + self.completion_fp + eps)
        recall = self.completion_tp / (
            self.completion_tp + self.completion_fn + eps)
        sc_iou = self.completion_tp / (
            self.completion_tp + self.completion_fp + self.completion_fn + eps)

        tp = np.diag(self.hist).astype(np.float64)
        fp = self.hist.sum(axis=0) - tp
        fn = self.hist.sum(axis=1) - tp
        denom = tp + fp + fn
        class_iou = np.divide(
            tp, denom,
            out=np.full(self.num_classes, np.nan, dtype=np.float64),
            where=denom > 0,
        )
        occupied_iou = class_iou[list(OCCUPIED_CLASS_IDS)]
        ssc_miou = np.nanmean(occupied_iou)

        output = {
            prefix + 'SC_Precision': precision,
            prefix + 'SC_Recall': recall,
            prefix + 'SC_IoU': sc_iou,
            prefix + 'SSC_mIoU': ssc_miou,
        }
        for class_id, class_name in enumerate(OCC_CLASS_NAMES):
            output[prefix + 'IoU_' + class_name] = class_iou[class_id]
        return {
            key: round(float(value) * 100.0, 2)
            for key, value in output.items()
        }


@DATASETS.register_module()
class GOOSEOccMono(Dataset):
    """Single-camera, single-frame GOOSE-OCC baseline dataset.

    Training consumes official 2D semantics and Metric3D pseudo depth.
    Evaluation consumes the existing sparse multi-frame-fused OCC labels. The
    current label protocol treats every unstored voxel as free.
    """

    def __init__(self, data_root, ann_file, occ_size, pc_range, pipeline,
                 split='train', test_mode=False, evaluate_fov=True,
                 evaluate_visibility=True, classes=None, modality=None,
                 temporal_filter_ann_file=None, min_history=0,
                 **kwargs):
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.ann_file = os.path.abspath(os.path.expanduser(ann_file))
        self.occ_size = tuple(int(v) for v in occ_size)
        self.pc_range = tuple(float(v) for v in pc_range)
        self.split = split
        self.test_mode = test_mode
        self.evaluate_fov = evaluate_fov
        self.evaluate_visibility = evaluate_visibility
        self.temporal_filter_ann_file = temporal_filter_ann_file
        self.min_history = int(min_history)
        self.pipeline = Compose(pipeline)

        payload = load_goose_pickle(self.ann_file)
        self.metadata = payload.get('metadata', {}) if isinstance(payload, dict) else {}
        self.data_infos = payload['infos'] if isinstance(payload, dict) else payload
        if self.temporal_filter_ann_file is not None:
            temporal_path = os.path.abspath(os.path.expanduser(
                self.temporal_filter_ann_file))
            temporal_infos = load_goose_pickle(temporal_path)
            eligible = {
                (
                    info['sequence'], int(info['frame_index']),
                    int(info['target_sensor_stamp_ns']),
                )
                for info in temporal_infos
                if len(info['histories']) >= self.min_history
            }
            self.data_infos = [
                info for info in self.data_infos
                if (
                    info['sequence'], int(info['frame_index']),
                    int(info['sensor_stamp_ns']),
                ) in eligible
            ]
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
        self._validate_metadata()

        print('[GOOSEOccMono] {}: {} samples from {}'.format(
            self.split, len(self.data_infos), self.ann_file))

    def _validate_metadata(self):
        meta_size = tuple(self.metadata.get('occ_size', self.occ_size))
        meta_range = tuple(self.metadata.get('pc_range', self.pc_range))
        if meta_size != self.occ_size:
            raise ValueError('occ_size mismatch: config={} infos={}'.format(
                self.occ_size, meta_size))
        if not np.allclose(meta_range, self.pc_range):
            raise ValueError('pc_range mismatch: config={} infos={}'.format(
                self.pc_range, meta_range))
        stored_classes = self.metadata.get('stored_classes')
        if stored_classes and list(stored_classes) != OCC_CLASS_NAMES:
            raise ValueError('Unexpected GOOSE taxonomy: {}'.format(stored_classes))

    @staticmethod
    def _pre_pipeline(results):
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def resolve_path(self, path):
        if path is None or os.path.isabs(path):
            return path
        return os.path.join(self.data_root, path)

    def get_data_info(self, index):
        info = self.data_infos[index]
        if len(info['cams']) != 1 or 'CAM_WINDSHIELD_VIS' not in info['cams']:
            raise ValueError('GOOSE baseline expects one CAM_WINDSHIELD_VIS camera')
        cam = info['cams']['CAM_WINDSHIELD_VIS']

        cam2img = np.asarray(cam['cam_intrinsic'], dtype=np.float32)
        lidar2cam = np.asarray(cam['lidar2cam'], dtype=np.float32)
        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:3, :3] = cam2img
        lidar2img = viewpad @ lidar2cam
        ego2lidar = np.eye(4, dtype=np.float32)

        input_dict = dict(
            sample_idx=info['token'],
            token=info['token'],
            sequence=info['sequence'],
            frame_id=int(info['frame_index']),
            timestamp=float(info['timestamp']) / 1e6,
            img_timestamp=[float(info['timestamp']) / 1e6],
            img_filename=[self.resolve_path(cam['data_path'])],
            filename=[self.resolve_path(cam['data_path'])],
            seg_label_path=self.resolve_path(info['seg_label_path']),
            seg_label_paths=[self.resolve_path(info['seg_label_path'])],
            semantic_is_raw=[True],
            semantic_confidence_paths=[None],
            depth_metric3d_path=self.resolve_path(
                info['depth_metric3d_path']),
            depth_metric3d_paths=[self.resolve_path(
                info['depth_metric3d_path'])],
            occ_size=np.asarray(self.occ_size, dtype=np.int64),
            pc_range=np.asarray(self.pc_range, dtype=np.float32),
            lidar2img=[lidar2img],
            ego2lidar=[ego2lidar],
            lidar2ego=[ego2lidar.copy()],
            cam2img=[cam2img],
            ego2cam=[lidar2cam],
            lidar2cam=[lidar2cam],
            ego2global=np.asarray(info['T_global_lidar'], dtype=np.float32),
            taxonomy_path=os.path.join(
                self.data_root, 'infos', 'goose_occ_taxonomy.json'),
        )
        if self.test_mode:
            input_dict.update(
                occ_path=self.resolve_path(info['occ_path']),
                fov_mask_path=self.resolve_path(info['fov_mask_path']),
                observation_path=self.resolve_path(info['observation_path']),
            )
        self._pre_pipeline(input_dict)
        return input_dict

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        return self.pipeline(self.get_data_info(index))

    @staticmethod
    def _prediction_array(result):
        pred = result['occ_pred']
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred)
        while pred.ndim > 3 and pred.shape[0] == 1:
            pred = pred[0]
        return pred.astype(np.int64, copy=False)

    def evaluate(self, occ_results, runner=None, show_dir=None, **kwargs):
        if len(occ_results) != len(self.data_infos):
            raise ValueError('Prediction count {} != dataset size {}'.format(
                len(occ_results), len(self.data_infos)))

        print('\nStarting GOOSE-OCC baseline evaluation...')
        full_metrics = _SSCAccumulator(NUM_CLASSES)
        fov_metrics = _SSCAccumulator(NUM_CLASSES) if self.evaluate_fov else None
        visibility_metrics = (
            GooseVisibilityMetrics(OCC_CLASS_NAMES)
            if self.evaluate_visibility else None
        )
        fov_cache = {}
        full_mask = np.ones(self.occ_size, dtype=bool)

        for index, result in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            gt = load_goose_occ_semantics(
                self.resolve_path(info['occ_path']), self.occ_size, NUM_CLASSES)
            pred = self._prediction_array(result)
            if pred.shape != self.occ_size:
                raise ValueError('Prediction shape {} != {} for {}'.format(
                    pred.shape, self.occ_size, info['token']))
            if np.any((pred < 0) | (pred >= NUM_CLASSES)):
                raise ValueError('Prediction contains invalid class IDs: {}'.format(
                    info['token']))

            full_metrics.add(pred, gt, full_mask)
            if fov_metrics is not None:
                fov_path = self.resolve_path(info['fov_mask_path'])
                if fov_path not in fov_cache:
                    mask = np.load(fov_path, allow_pickle=False).astype(bool)
                    if mask.shape != self.occ_size:
                        raise ValueError('FOV mask shape {} != {}'.format(
                            mask.shape, self.occ_size))
                    fov_cache[fov_path] = mask
                fov_metrics.add(pred, gt, fov_cache[fov_path])
            if visibility_metrics is not None:
                masks = load_observation(
                    self.resolve_path(info['observation_path']))
                visibility_metrics.update(pred, gt, masks)

        metrics = full_metrics.results()
        if fov_metrics is not None:
            metrics.update(fov_metrics.results(prefix='FOV_'))
        if visibility_metrics is not None:
            metrics.update(flatten_visibility_results(
                visibility_metrics.result()))
        return metrics
