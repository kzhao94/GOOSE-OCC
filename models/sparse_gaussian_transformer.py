import copy
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from .sparse_gaussian_decoder import SparseGaussianDecoder


@TRANSFORMER.register_module()
class SparseGaussianTransformer(BaseModule):
    def __init__(self, embed_dims=None, num_layers=None, num_queries=None,
                 num_frames=None, num_points=None, num_groups=None,
                 num_levels=None, num_classes=None, pc_range=None,
                 occ_size=None, voxel_size = None, scale_range=None):
        super().__init__()
        self.num_frames = num_frames

        # 使用高斯解码器替代体素解码器
        self.gaussian_decoder = SparseGaussianDecoder(
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            voxel_size = voxel_size,
            scale_range=scale_range,
            num_queries = num_queries,
        )

    @torch.no_grad()
    def init_weights(self):
        self.gaussian_decoder.init_weights()

    def forward(self, mlvl_feats, img_metas):
        # 预处理多尺度特征
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape
            # N, T, G, C = 6, TN // 6, 4, GC // 4
            # ⚠️ 原代码硬编码 N=6，单目前视需要改为
            T = self.num_frames  # 帧数
            N = TN // T
            G = 4
            C = GC // G
            feat = feat.reshape(B, T, N, G, C, H, W)
            feat = feat.permute(0, 1, 3, 2, 5, 6, 4)
            feat = feat.reshape(B * T * G, N, H, W, C)
            mlvl_feats[lvl] = feat.contiguous()

        # 准备变换矩阵
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(feat.device)

        ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
        ego2lidar = torch.from_numpy(ego2lidar).to(feat.device)

        img_metas = copy.deepcopy(img_metas)
        img_metas[0]['lidar2img'] = torch.matmul(lidar2img, ego2lidar)

        # 高斯解码
        gaussian_preds = self.gaussian_decoder(mlvl_feats, img_metas=img_metas)

        return gaussian_preds
