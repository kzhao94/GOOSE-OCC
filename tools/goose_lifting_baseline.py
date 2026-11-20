#!/usr/bin/env python3
"""Shared data and geometry helpers for the GOOSE lifting baseline."""

import hashlib
import json
import pickle
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_config(path):
    payload = runpy.run_path(str(path))
    return {key: value for key, value in payload.items()
            if not key.startswith('__') and key != 'os'}


def load_pickle(path):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def common_target_infos(info_path, temporal_path, min_history=10):
    payload = load_pickle(info_path)
    infos = payload['infos'] if isinstance(payload, dict) else payload
    temporal = load_pickle(temporal_path)
    eligible = {
        (row['sequence'], int(row['frame_index']),
         int(row['target_sensor_stamp_ns']))
        for row in temporal if len(row['histories']) >= int(min_history)
    }
    selected = [
        row for row in infos
        if (row['sequence'], int(row['frame_index']),
            int(row['sensor_stamp_ns'])) in eligible
    ]
    selected.sort(key=lambda row: (
        row['sequence'], int(row['frame_index']), int(row['sensor_stamp_ns'])))
    return selected


def resolve(root, relative):
    path = Path(relative)
    return path if path.is_absolute() else Path(root) / path


def taxonomy_lut(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    lut = np.full(256, 255, dtype=np.uint8)
    for raw_id, stored_id in payload['raw_goose_2d_to_stored_id'].items():
        lut[int(raw_id)] = int(stored_id)
    return lut


def train_record(info, root):
    camera = info['cams']['CAM_WINDSHIELD_VIS']
    return {
        'token': info['token'],
        'image_path': str(resolve(root, camera['data_path'])),
        'semantic_path': str(resolve(root, info['seg_label_path'])),
    }


def prediction_record(info, root):
    camera = info['cams']['CAM_WINDSHIELD_VIS']
    return {
        'token': info['token'],
        'sequence': info['sequence'],
        'frame_index': int(info['frame_index']),
        'image_path': str(resolve(root, camera['data_path'])),
        'depth_path': str(resolve(root, info['depth_metric3d_path'])),
        'intrinsic': np.asarray(camera['cam_intrinsic'], dtype=np.float64),
        'lidar2cam': np.asarray(camera['lidar2cam'], dtype=np.float64),
    }


def evaluation_record(info, root):
    return {
        'token': info['token'],
        'occ_path': str(resolve(root, info['occ_path'])),
        'observation_path': str(resolve(root, info['observation_path'])),
    }


def read_image_tensor(path, image_size):
    height, width = image_size
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError('Failed to read image {}'.format(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    array = rgb.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1)).float()


class Goose2DSemanticTrainDataset(Dataset):
    """Sanitized train-only RGB/2D-label dataset with no OCC fields."""

    def __init__(self, infos, root, lut, image_size, augment=True):
        self.records = [train_record(info, root) for info in infos]
        self.lut = np.asarray(lut, dtype=np.uint8)
        self.image_size = tuple(image_size)
        self.augment = bool(augment)
        allowed = {'token', 'image_path', 'semantic_path'}
        if any(set(record) != allowed for record in self.records):
            raise ValueError('Unexpected training record fields')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_image_tensor(record['image_path'], self.image_size)
        raw = cv2.imread(record['semantic_path'], cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError('Failed to read label {}'.format(
                record['semantic_path']))
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        if raw.max() >= len(self.lut):
            raise ValueError('Raw semantic ID exceeds LUT')
        label = self.lut[raw.astype(np.int64)]
        height, width = self.image_size
        label = cv2.resize(
            label, (width, height), interpolation=cv2.INTER_NEAREST)
        label = torch.from_numpy(label.astype(np.int64))
        if self.augment and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(2,))
            label = torch.flip(label, dims=(1,))
        return image, label


def create_segmenter(num_classes, pretrained_backbone):
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large
    backbone = (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1
        if pretrained_backbone else None)
    return lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=backbone,
        num_classes=int(num_classes))


def scaled_intrinsic(intrinsic, original_shape, image_size):
    original_h, original_w = original_shape
    height, width = image_size
    scaled = np.asarray(intrinsic, dtype=np.float64).copy()
    scaled[0, :] *= width / float(original_w)
    scaled[1, :] *= height / float(original_h)
    return scaled


def lift_to_grid(semantic, depth, intrinsic, lidar2cam, pc_range,
                 occ_size, voxel_size):
    """Lift predicted semantic/depth endpoints into a target-LiDAR grid."""
    semantic = np.asarray(semantic)
    depth = np.asarray(depth, dtype=np.float32)
    if semantic.shape != depth.shape:
        raise ValueError('Semantic/depth shape mismatch')
    valid = (
        (semantic >= 1) & (semantic <= 7) & np.isfinite(depth)
        & (depth > 0.1) & (depth < float(pc_range[3])))
    rows, cols = np.nonzero(valid)
    prediction = np.zeros(tuple(occ_size), dtype=np.uint8)
    if not len(rows):
        return prediction
    z = depth[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (rows.astype(np.float64) - intrinsic[1, 2]) / intrinsic[1, 1] * z
    camera_points = np.stack((x, y, z, np.ones_like(z)), axis=1)
    lidar_points = camera_points.dot(np.linalg.inv(lidar2cam).T)[:, :3]
    minimum = np.asarray(pc_range[:3], dtype=np.float64)
    coordinates = np.floor(
        (lidar_points - minimum[None]) / float(voxel_size)).astype(np.int64)
    inside = (
        (coordinates >= 0).all(axis=1)
        & (coordinates < np.asarray(occ_size)[None]).all(axis=1))
    coordinates = coordinates[inside]
    labels = semantic[rows, cols][inside].astype(np.uint8)
    depths = z[inside]
    if not len(coordinates):
        return prediction
    linear = (
        (coordinates[:, 0] * int(occ_size[1]) + coordinates[:, 1])
        * int(occ_size[2]) + coordinates[:, 2])
    order = np.lexsort((depths, linear))
    linear_sorted = linear[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = linear_sorted[1:] != linear_sorted[:-1]
    chosen = order[first]
    coords = coordinates[chosen]
    prediction[coords[:, 0], coords[:, 1], coords[:, 2]] = labels[chosen]
    return prediction


def densify_occ(path, occ_size):
    sparse = np.load(path, allow_pickle=False).astype(np.int64, copy=False)
    if sparse.ndim != 2 or sparse.shape[1] != 4:
        raise ValueError('Invalid sparse OCC shape')
    dense = np.zeros(tuple(occ_size), dtype=np.uint8)
    if len(sparse):
        coordinates = sparse[:, :3]
        labels = sparse[:, 3]
        valid = (
            (coordinates >= 0).all(axis=1)
            & (coordinates < np.asarray(occ_size)[None]).all(axis=1)
            & (labels >= 1) & (labels <= 7))
        if not np.all(valid):
            raise ValueError('Invalid sparse OCC coordinate or label')
        dense[tuple(coordinates.T)] = labels.astype(np.uint8)
    return dense


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def confusion_result(matrix, class_names):
    matrix = np.asarray(matrix, dtype=np.int64)
    occupied_tp = int(matrix[1:, 1:].sum())
    false_occupied = int(matrix[0, 1:].sum())
    missed_occupied = int(matrix[1:, 0].sum())
    union = occupied_tp + false_occupied + missed_occupied
    per_class = {}
    occupied_ious = []
    for class_id, name in enumerate(class_names):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        class_union = tp + fp + fn
        iou = tp / class_union if class_union else None
        per_class[name] = {'iou': iou, 'tp': tp, 'fp': fp, 'fn': fn,
                           'union': class_union}
        if class_id > 0 and iou is not None:
            occupied_ious.append(iou)
    return {
        'SC-IoU': occupied_tp / union if union else None,
        'SSC-mIoU': (sum(occupied_ious) / len(occupied_ious)
                     if occupied_ious else None),
        'per_class': per_class,
        'confusion_matrix': matrix.tolist(),
    }


def update_confusion(matrix, prediction, target, num_classes=8):
    prediction = np.asarray(prediction).reshape(-1).astype(np.int64)
    target = np.asarray(target).reshape(-1).astype(np.int64)
    if np.any((prediction < 0) | (prediction >= num_classes)):
        raise ValueError('Prediction ID outside evaluation taxonomy')
    encoded = target * num_classes + prediction
    matrix += np.bincount(
        encoded, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes)
