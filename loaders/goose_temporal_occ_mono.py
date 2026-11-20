import os

import numpy as np
from mmdet.datasets import DATASETS

from .goose_occ_mono import GOOSEOccMono, load_goose_pickle


@DATASETS.register_module()
class GOOSETemporalOccMono(GOOSEOccMono):
    """GOOSE target plus causal history frames in target-LiDAR coordinates."""

    def __init__(self, temporal_ann_file, num_history=3, min_history=None,
                 **kwargs):
        self.temporal_ann_file = os.path.abspath(
            os.path.expanduser(temporal_ann_file))
        self.num_history = int(num_history)
        if self.num_history < 1:
            raise ValueError('num_history must be at least 1')
        self.temporal_min_history = (
            self.num_history if min_history is None else int(min_history))
        if self.temporal_min_history < self.num_history:
            raise ValueError(
                'min_history ({}) must be >= num_history ({})'.format(
                    self.temporal_min_history, self.num_history))

        temporal_infos = load_goose_pickle(self.temporal_ann_file)
        self.temporal_by_key = {
            self._temporal_key(info): info for info in temporal_infos
        }
        super().__init__(**kwargs)

        original_count = len(self.data_infos)
        self.data_infos = [
            info for info in self.data_infos
            if self._base_key(info) in self.temporal_by_key
            and len(self.temporal_by_key[self._base_key(info)]['histories'])
            >= self.temporal_min_history
        ]
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
        print(
            '[GOOSETemporalOccMono] {}: {} of {} targets have at least {} '
            'real history frames; using {}'.format(
                self.split, len(self.data_infos), original_count,
                self.temporal_min_history, self.num_history))

    @staticmethod
    def _base_key(info):
        return (
            info['sequence'], int(info['frame_index']),
            int(info['sensor_stamp_ns']),
        )

    @staticmethod
    def _temporal_key(info):
        return (
            info['sequence'], int(info['frame_index']),
            int(info['target_sensor_stamp_ns']),
        )

    @staticmethod
    def _target_lidar_to_camera(t_global_camera, t_global_lidar):
        t_global_camera = np.asarray(
            t_global_camera, dtype=np.float64).reshape(4, 4)
        t_global_lidar = np.asarray(
            t_global_lidar, dtype=np.float64).reshape(4, 4)
        return (np.linalg.inv(t_global_camera) @ t_global_lidar).astype(
            np.float32)

    @staticmethod
    def _lidar2img(cam2img, lidar2cam):
        viewpad = np.eye(4, dtype=np.float32)
        viewpad[:3, :3] = cam2img
        return viewpad @ lidar2cam

    def get_data_info(self, index):
        results = super().get_data_info(index)
        base_info = self.data_infos[index]
        temporal = self.temporal_by_key[self._base_key(base_info)]
        histories = temporal['histories'][:self.num_history]

        cam2img = np.asarray(temporal['camera_K'], dtype=np.float32).reshape(3, 3)
        t_global_lidar = np.asarray(
            base_info['T_global_lidar'], dtype=np.float64).reshape(4, 4)
        target_lidar2cam = np.asarray(
            results['ego2cam'][0], dtype=np.float32)

        image_paths = [results['img_filename'][0]]
        semantic_paths = [results['seg_label_path']]
        confidence_paths = [None]
        depth_paths = [self.resolve_path(
            temporal['target_depth_metric3d_path'])]
        timestamps = [float(temporal['target_camera_stamp_ns']) / 1e9]
        ego2cam = [target_lidar2cam]
        for history in histories:
            image_paths.append(self.resolve_path(history['image_path']))
            semantic_paths.append(self.resolve_path(
                history['pseudo_label_path']))
            confidence_paths.append(self.resolve_path(
                history['confidence_path']))
            depth_paths.append(self.resolve_path(
                history['depth_metric3d_path']))
            timestamps.append(float(history['camera_stamp_ns']) / 1e9)
            ego2cam.append(self._target_lidar_to_camera(
                history['T_global_camera'], t_global_lidar))

        frame_count = 1 + self.num_history
        identities = [np.eye(4, dtype=np.float32) for _ in range(frame_count)]
        results.update(
            img_filename=image_paths,
            filename=image_paths,
            img_timestamp=timestamps,
            seg_label_paths=semantic_paths,
            semantic_is_raw=[True] + [False] * self.num_history,
            semantic_confidence_paths=confidence_paths,
            depth_metric3d_paths=depth_paths,
            depth_metric3d_path=depth_paths[0],
            cam2img=[cam2img.copy() for _ in range(frame_count)],
            ego2cam=ego2cam,
            lidar2cam=ego2cam,
            lidar2img=[
                self._lidar2img(cam2img, transform)
                for transform in ego2cam
            ],
            ego2lidar=identities,
            lidar2ego=[matrix.copy() for matrix in identities],
            frame_is_history=[False] + [True] * self.num_history,
            history_ranks=[0] + [history['rank'] for history in histories],
        )
        if self.test_mode:
            results['observation_path'] = self.resolve_path(
                temporal['target_observation_path'])
        return results
