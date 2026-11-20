import json

import cv2
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES

from loaders.goose_occ_mono import NUM_CLASSES, load_goose_occ_semantics


@PIPELINES.register_module()
class LoadGOOSEImageAndCalib:
    """Load target/history images and resize their projection matrices."""

    def __init__(self, to_float32=False, color_type='color',
                 render_size=(256, 704)):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.render_size = tuple(render_size)

    def __call__(self, results):
        render_h, render_w = self.render_size
        images = []
        lidar2img = []
        cam2img = []
        image_shapes = []
        original_shapes = []
        scale_factors = []
        for index, filename in enumerate(results['img_filename']):
            image = mmcv.imread(filename, self.color_type)
            if image is None:
                raise IOError('Failed to read image: {}'.format(filename))
            original_h, original_w = image.shape[:2]
            image = cv2.resize(
                image, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
            if self.to_float32:
                image = image.astype(np.float32)

            scale_x = render_w / float(original_w)
            scale_y = render_h / float(original_h)
            resize_4x4 = np.eye(4, dtype=np.float32)
            resize_4x4[0, 0] = scale_x
            resize_4x4[1, 1] = scale_y
            resize_3x3 = np.eye(3, dtype=np.float32)
            resize_3x3[0, 0] = scale_x
            resize_3x3[1, 1] = scale_y

            images.append(image)
            lidar2img.append(resize_4x4 @ results['lidar2img'][index])
            cam2img.append(resize_3x3 @ results['cam2img'][index])
            image_shapes.append(image.shape)
            original_shapes.append((original_h, original_w, image.shape[2]))
            scale_factors.append(np.asarray(
                [scale_x, scale_y, scale_x, scale_y], dtype=np.float32))

        results['lidar2img'] = lidar2img
        results['cam2img'] = cam2img
        results['img'] = images
        results['img_shape'] = image_shapes
        results['ori_shape'] = image_shapes
        results['pad_shape'] = image_shapes
        results['scale_factor'] = scale_factors
        results['img_norm_cfg'] = dict(
            mean=np.zeros(3, dtype=np.float32),
            std=np.ones(3, dtype=np.float32),
            to_rgb=False,
        )
        results['render_size'] = self.render_size
        results['original_image_shape'] = original_shapes
        return results


@PIPELINES.register_module()
class LoadGOOSE2DSupervision:
    """Load official/pseudo semantics, confidence, and Metric3D depth."""

    def __init__(self, taxonomy_key='raw_goose_2d_to_stored_id',
                 depth_scale=256.0, confidence_mode='weighted'):
        self.taxonomy_key = taxonomy_key
        self.depth_scale = float(depth_scale)
        if confidence_mode not in ('weighted', 'binary'):
            raise ValueError(
                'confidence_mode must be "weighted" or "binary"')
        self.confidence_mode = confidence_mode
        self._lut_cache = {}

    def _load_lut(self, taxonomy_path):
        if taxonomy_path not in self._lut_cache:
            with open(taxonomy_path, 'r') as handle:
                taxonomy = json.load(handle)
            mapping = taxonomy[self.taxonomy_key]
            lut = np.full(256, 255, dtype=np.uint8)
            for raw_id, stored_id in mapping.items():
                lut[int(raw_id)] = int(stored_id)
            self._lut_cache[taxonomy_path] = lut
        return self._lut_cache[taxonomy_path]

    def __call__(self, results):
        render_h, render_w = results['render_size']
        lut = self._load_lut(results['taxonomy_path'])
        semantic_paths = results.get(
            'seg_label_paths', [results['seg_label_path']])
        semantic_is_raw = results.get(
            'semantic_is_raw', [True] * len(semantic_paths))
        confidence_paths = results.get(
            'semantic_confidence_paths', [None] * len(semantic_paths))
        depth_paths = results.get(
            'depth_metric3d_paths', [results['depth_metric3d_path']])
        lengths = {
            len(semantic_paths), len(semantic_is_raw),
            len(confidence_paths), len(depth_paths),
        }
        if len(lengths) != 1:
            raise ValueError('GOOSE supervision lists have inconsistent lengths')

        semantics = []
        semantic_weights = []
        depths = []
        for semantic_path, is_raw, confidence_path, depth_path in zip(
                semantic_paths, semantic_is_raw, confidence_paths, depth_paths):
            label = cv2.imread(semantic_path, cv2.IMREAD_UNCHANGED)
            if label is None:
                raise IOError('Failed to read semantic label: {}'.format(
                    semantic_path))
            if label.ndim == 3:
                label = label[:, :, 0]
            if is_raw:
                if label.max() >= len(lut):
                    raise ValueError('GOOSE semantic ID exceeds LUT range')
                label = lut[label.astype(np.int64)]
            else:
                valid_ids = ((label >= 1) & (label < NUM_CLASSES)) | (label == 255)
                if not np.all(valid_ids):
                    raise ValueError(
                        'Invalid temporal semantic ID: {}'.format(semantic_path))

            if confidence_path is None:
                confidence = np.ones(label.shape, dtype=np.float32)
            else:
                confidence_raw = cv2.imread(
                    confidence_path, cv2.IMREAD_UNCHANGED)
                if confidence_raw is None:
                    raise IOError('Failed to read confidence: {}'.format(
                        confidence_path))
                if confidence_raw.ndim == 3:
                    confidence_raw = confidence_raw[:, :, 0]
                if confidence_raw.shape != label.shape:
                    raise ValueError('Confidence/semantic shape mismatch')
                label = label.copy()
                label[confidence_raw == 0] = 255
                if self.confidence_mode == 'weighted':
                    confidence = confidence_raw.astype(np.float32) / 255.0
                else:
                    confidence = (confidence_raw > 0).astype(np.float32)

            label = cv2.resize(
                label, (render_w, render_h), interpolation=cv2.INTER_NEAREST)
            confidence_interpolation = (
                cv2.INTER_LINEAR
                if self.confidence_mode == 'weighted'
                else cv2.INTER_NEAREST)
            confidence = cv2.resize(
                confidence, (render_w, render_h),
                interpolation=confidence_interpolation)
            label[confidence <= 0] = 255
            confidence[label == 255] = 0.0

            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise IOError('Failed to read Metric3D depth: {}'.format(
                    depth_path))
            if depth_raw.ndim == 3:
                depth_raw = depth_raw[:, :, 0]
            depth = depth_raw.astype(np.float32) / self.depth_scale
            depth = cv2.resize(
                depth, (render_w, render_h), interpolation=cv2.INTER_LINEAR)

            semantics.append(label.astype(np.uint8))
            semantic_weights.append(confidence.astype(np.float32))
            depths.append(depth.astype(np.float32))

        results['gt_semantic'] = semantics
        results['gt_semantic_weight'] = semantic_weights
        results['gt_depth'] = depths
        return results


@PIPELINES.register_module()
class LoadGOOSEOccGT:
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes

    def __call__(self, results):
        results['voxel_semantics'] = load_goose_occ_semantics(
            results['occ_path'], results['occ_size'], self.num_classes)
        return results
