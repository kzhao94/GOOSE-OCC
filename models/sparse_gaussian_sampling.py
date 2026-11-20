import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from .gaussian_utils import decode_gaussian, quaternion_to_matrix
from .utils import DUMP
from .checkpoint import checkpoint as cp
from .csrc.wrapper import msmv_sampling


class SparseGaussianSampling(BaseModule):
    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8,
                 num_levels=4, pc_range=[], init_cfg=None):
        """
        基于高斯参数的稀疏采样模块

        Args:
            embed_dims: 特征维度
            num_frames: 帧数
            num_groups: 组数
            num_points: 每组采样点数
            num_levels: 特征层级数
            pc_range: 点云范围
            init_cfg: 初始化配置
        """
        # 关键：继承BaseModule，正确传递init_cfg
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        # 采样偏移预测（预测在归一化椭球体内的偏移）
        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)

        # 多尺度特征权重
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        """初始化权重"""
        # 初始化采样偏移
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias, -0.5, 0.5)

        # 初始化尺度权重
        nn.init.zeros_(self.scale_weights.weight)
        nn.init.uniform_(self.scale_weights.bias, -0.1, 0.1)

    def inner_forward(self, query_means, query_feat, mlvl_feats, img_metas):
        """
        前向传播核心逻辑

        Args:
            query_gaussian: [B, Q, 11] 高斯查询参数
            query_feat: [B, Q, C] 查询特征
            mlvl_feats: 多尺度特征
            img_metas: 图像元数据

        Returns:
            sampled_feats: [B, Q, G, FP, C] 采样得到的特征
        """
        B, Q = query_means.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # 1. 预测采样偏移
        sampling_offset = self.sampling_offset(query_feat)  # [B, Q, GP*3]
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)

        # 2. 生成采样点
        sampling_points = query_means.unsqueeze(2) + sampling_offset

        # 3. 扩展到多帧
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # 5. 多尺度特征权重
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # 6. 执行4D采样
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['lidar2img'],
            image_h,
            image_w
        )

        return sampled_feats

    def forward(self, query_means, query_feat, mlvl_feats, img_metas):
        if self.training and query_feat.requires_grad:
            # 训练时使用梯度检查点节省显存
            return cp(self.inner_forward, query_means, query_feat, mlvl_feats, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_means, query_feat, mlvl_feats, img_metas)


def sampling_4d(sample_points, mlvl_feats, scale_weights, lidar2img, image_h, image_w, eps=1e-5):
    B, Q, T, G, P, _ = sample_points.shape  # [B, Q, T, G, P, 4]
    # 动态获取视角数 N
    N = mlvl_feats[0].shape[1]

    sample_points = sample_points.reshape(B, Q, T, G * P, 3)

    if DUMP.enabled:
        torch.save(sample_points,
                   '{}/sample_points_3d_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

    # get the projection matrix
    lidar2img = lidar2img[:, :(T * N), None, None, :, :]  # [B, TN, 1, 1, 4, 4]
    lidar2img = lidar2img.expand(B, T * N, Q, G * P, 4, 4)
    lidar2img = lidar2img.reshape(B, T, N, Q, G * P, 4, 4)

    # expand the points
    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)  # [B, Q, GP, 4]
    sample_points = sample_points[:, :, None, ..., None]  # [B, Q, T, GP, 4]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 3)  # [B, T, N, Q, GP, 4, 1]

    # project 3d sampling points to image
    sample_points_cam = torch.matmul(lidar2img, sample_points).squeeze(-1)  # [B, T, N, Q, GP, 4]

    # homo coord -> pixel coord
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero  # [B, T, N, Q, GP, 2]

    # normalize
    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    # check if out of image
    valid_mask = ((homo > eps) \
                  & (sample_points_cam[..., 1:2] > 0.0)
                  & (sample_points_cam[..., 1:2] < 1.0)
                  & (sample_points_cam[..., 0:1] > 0.0)
                  & (sample_points_cam[..., 0:1] < 1.0)
                  ).squeeze(-1).float()  # [B, T, N, Q, GP]

    if DUMP.enabled:
        torch.save(torch.cat([sample_points_cam, homo_nonzero], dim=-1),
                   '{}/sample_points_cam_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
        torch.save(valid_mask,
                   '{}/sample_points_cam_valid_mask_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)  # [B, T, Q, GP, N]
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)  # [B, T, Q, GP, N, 2]

    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)
    i_view = torch.argmax(valid_mask, dim=-1)[..., None]  # [B, T, Q, GP, 1]

    sample_points_cam = sample_points_cam[i_batch, i_time, i_query, i_point, i_view, :]  # [B, Q, GP, 1, 2]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]  # [B, Q, GP, 1]

    # sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / 5], dim=-1)
    sample_points_cam = torch.cat([sample_points_cam, i_view[..., None].float() / max(N - 1, 1)], dim=-1)
    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)  # [B, T, G, Q, P, 1, 3]
    sample_points_cam = sample_points_cam.reshape(B * T * G, Q, P, 3)

    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B * G * T, Q, P, -1)

    final = msmv_sampling(mlvl_feats, sample_points_cam, scale_weights)
    C = final.shape[2]  # [BTG, Q, C, P]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4)
    final = final.flatten(3, 4)  # [B, Q, G, FP, C]

    return final