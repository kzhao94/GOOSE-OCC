import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from .sparse_gaussian_sampling import SparseGaussianSampling
from .sparsebev_transformer import AdaptiveMixing
from .checkpoint import checkpoint as cp
import numpy as np
import math
import os


class SparseGaussianDecoder(BaseModule):
    def __init__(self, embed_dims=None, num_layers=None, num_frames=None, num_points=None, num_groups=None,
                 num_levels=None, num_classes=None, pc_range=None,
                 interval_xy=8, interval_z=8, voxel_size=0.4, num_queries=None, scale_range=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.save_counter = 0
        self.scale_range = scale_range
        self.N = num_queries  # 固定点云数量

        # ==========================================
        # 1. Scale 范围与 voxel_size 绑定
        # ==========================================
        if scale_range is None:
            # 自动推算：最小缩放为体素的1/4，最大缩放为体素的1.5倍
            self.scale_range = [voxel_size * 0.25, voxel_size * 1.5]
        else:
            self.scale_range = scale_range

        # ==========================================
        # 2. 注册占位 buffer 和可学习初始化残差
        # ==========================================
        self.register_buffer('init_means', torch.zeros(1, num_queries, 3))
        self.init_means_delta = nn.Parameter(torch.zeros(1, num_queries, 3))
        self.register_buffer('anchor_delta_limit', torch.tensor([2.0, 1.0, 0.5]).view(1, 1, 3))
        self.anchor_reg_weight = 1e-3
        self._frustum_initialized = False

        # ==========================================
        # 2. Decoder 层 (只预测 xyz 的增量)
        # ==========================================
        self.decoder_layers = nn.ModuleList()
        self.delta_xyz_heads = nn.ModuleList()

        for i in range(num_layers):
            self.decoder_layers.append(SparseGaussianDecoderLayer(
                embed_dims=embed_dims,
                num_frames=num_frames,
                num_points=num_points,
                num_groups=num_groups,
                num_levels=num_levels,
                pc_range=pc_range,
                self_attn_type='distance' if i in [0, 1] else 'none'
            ))
            self.delta_xyz_heads.append(nn.Linear(embed_dims, 3))

        # ==========================================
        # 3. 最终的属性预测头 (仿照 GaussianFlowOcc，移出循环)
        # ==========================================
        self.seg_pred_heads = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dims * 2, num_classes)
        )

        self.opacity_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dims * 2, 1),
            nn.Sigmoid()  # 直接预测 0~1
        )

        self.scale_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dims * 2, 3),
            nn.Sigmoid()  # 预测 0~1，然后 rescale
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dims * 2, 4)  # 预测四元数，外面做归一化
        )

    @torch.no_grad()
    def init_weights(self):
        for i in range(len(self.decoder_layers)):
            self.decoder_layers[i].init_weights()
            # 核心：xyz 增量头初始化为 0，保证初始时点不乱跑
            nn.init.zeros_(self.delta_xyz_heads[i].weight)
            nn.init.zeros_(self.delta_xyz_heads[i].bias)

    @torch.no_grad()
    def init_frustum_geometry(self, img_metas):
        """
        Corrected non-uniform frustum initialization for monocular OCC.
        """
        device = next(self.parameters()).device

        ego2cam = torch.tensor(img_metas['ego2cam'][0], device=device).float()
        cam2img = torch.tensor(img_metas['cam2img'][0], device=device).float()

        if cam2img.shape[0] == 3:
            cam2img_4x4 = torch.eye(4, device=device)
            cam2img_4x4[:3, :3] = cam2img
            cam2img = cam2img_4x4

        if 'lidar2ego' in img_metas:
            lidar2ego = torch.tensor(img_metas['lidar2ego'][0], device=device).float()
        else:
            lidar2ego = torch.eye(4, device=device)

        render_size = img_metas['render_size']
        H, W = render_size[0], render_size[1]
        lidar2cam = ego2cam @ lidar2ego
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        fx = cam2img[0, 0]
        tan_half_h = W / (2 * fx)

        visible_means = None
        for sample_mult in [3, 6, 10]:
            n_sample = self.N * sample_mult
            pts_unit = halton_sequence(n_sample, dim=3).to(device)

            means = torch.empty(n_sample, 3, device=device)
            near = max(float(x_min), 1.0)
            means[:, 0] = near + (float(x_max) - near) * torch.pow(pts_unit[:, 0], 1.5)

            y_range_at_x = means[:, 0] * tan_half_h
            means[:, 1] = (pts_unit[:, 1] - 0.5) * 2.0 * y_range_at_x
            means[:, 1].clamp_(min=float(y_min), max=float(y_max))

            u_z = pts_unit[:, 2]
            z = torch.empty(n_sample, device=device)
            ground_mask = u_z < 0.70
            object_mask = (u_z >= 0.70) & (u_z < 0.95)
            upper_mask = u_z >= 0.95

            z[ground_mask] = -1.5 + (u_z[ground_mask] / 0.70) * 2.0
            z[object_mask] = 0.5 + ((u_z[object_mask] - 0.70) / 0.25) * 1.5
            z[upper_mask] = 2.0 + ((u_z[upper_mask] - 0.95) / 0.05) * 2.4
            means[:, 2] = z.clamp(min=float(z_min), max=float(z_max))

            ones = torch.ones(n_sample, 1, device=device)
            pts_lidar_homo = torch.cat([means, ones], dim=-1)
            pts_cam_homo = (lidar2cam @ pts_lidar_homo.T).T
            front_mask = pts_cam_homo[:, 2] > 0.1

            pts_img_homo = (cam2img @ pts_cam_homo.T).T
            valid_depth = pts_img_homo[:, 2] > 0.1
            px = torch.where(valid_depth, pts_img_homo[:, 0] / pts_img_homo[:, 2], torch.zeros(n_sample, device=device))
            py = torch.where(valid_depth, pts_img_homo[:, 1] / pts_img_homo[:, 2], torch.zeros(n_sample, device=device))
            in_image_mask = (px > 0) & (px < W) & (py > 0) & (py < H)

            means = means[front_mask & in_image_mask]
            visible_means = means
            if means.shape[0] >= self.N:
                break

        if visible_means is None or visible_means.shape[0] == 0:
            visible_means = torch.empty(0, 3, device=device)
            means = visible_means
        elif visible_means.shape[0] >= self.N:
            choice = torch.randperm(visible_means.shape[0], device=device)[:self.N]
            means = visible_means[choice]
        else:
            means = visible_means

        if means.shape[0] < self.N:
            pad_len = self.N - means.shape[0]
            pad_means = torch.randn(pad_len, 3, device=device) * 1.0 + torch.tensor([10.0, 0.0, 0.0], device=device)
            means = torch.cat([means, pad_means], dim=0)

        self.init_means.copy_(means.unsqueeze(0))
        self._frustum_initialized = True

        print(f"Corrected non-uniform frustum initialized: {self.N} points, visible candidates: {visible_means.shape[0]}, scale range: {self.scale_range}")

    """均匀采样"""
    # @torch.no_grad()
    # def init_frustum_geometry(self, img_metas):
    #     """
    #     3D空间均匀采样 + 正向投影视锥裁剪
    #     """
    #     # 安全获取 device，不再依赖已删除的 template_u
    #     device = next(self.parameters()).device

    #     # 1. 获取无增强的相机参数
    #     ego2cam = torch.tensor(img_metas['ego2cam'][0], device=device).float()
    #     cam2img = torch.tensor(img_metas['cam2img'][0], device=device).float()

    #     if cam2img.shape[0] == 3:
    #         cam2img_4x4 = torch.eye(4, device=device)
    #         cam2img_4x4[:3, :3] = cam2img
    #         cam2img = cam2img_4x4

    #     if 'lidar2ego' in img_metas:
    #         lidar2ego = torch.tensor(img_metas['lidar2ego'][0], device=device).float()
    #     else:
    #         lidar2ego = torch.eye(4, device=device)

    #     render_size = img_metas['render_size']
    #     H, W = render_size[0], render_size[1]

    #     lidar2cam = ego2cam @ lidar2ego

    #     # 2. 在 pc_range 内用 Halton 均匀采样 (多采5倍用于过滤)
    #     x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
    #     N_sample = self.N * 5
    #     pts_unit = halton_sequence(N_sample, dim=3).to(device)

    #     means = torch.zeros(N_sample, 3, device=device)
    #     means[:, 0] = x_min + pts_unit[:, 0] * (x_max - x_min)
    #     means[:, 1] = y_min + pts_unit[:, 1] * (y_max - y_min)
    #     means[:, 2] = z_min + pts_unit[:, 2] * (z_max - z_min)

    #     # 3. 正向投影：LiDAR → Cam → Img，检查是否在视锥内
    #     ones = torch.ones(N_sample, 1, device=device)
    #     pts_lidar_homo = torch.cat([means, ones], dim=-1)

    #     pts_cam_homo = (lidar2cam @ pts_lidar_homo.T).T
    #     front_mask = pts_cam_homo[:, 2] > 0.1

    #     pts_img_homo = (cam2img @ pts_cam_homo.T).T
    #     valid_depth = pts_img_homo[:, 2] > 0.1

    #     pts_img_x = torch.where(valid_depth, pts_img_homo[:, 0] / pts_img_homo[:, 2],
    #                             torch.zeros(N_sample, device=device))
    #     pts_img_y = torch.where(valid_depth, pts_img_homo[:, 1] / pts_img_homo[:, 2],
    #                             torch.zeros(N_sample, device=device))

    #     in_image_mask = (pts_img_x > 0) & (pts_img_x < W) & (pts_img_y > 0) & (pts_img_y < H)
    #     frustum_mask = front_mask & in_image_mask
    #     means = means[frustum_mask]

    #     # 4. 截取或补齐到 N 个点
    #     if means.shape[0] >= self.N:
    #         means = means[:self.N]
    #     else:
    #         pad_len = self.N - means.shape[0]
    #         pad_means = torch.randn(pad_len, 3, device=device) * 2.0 + torch.tensor([10.0, 0.0, 1.5], device=device)
    #         means = torch.cat([means, pad_means], dim=0)

    #     # 使用 copy_ 覆盖占位 buffer 的数据，而不是重新 register_buffer
    #     self.init_means.copy_(means.unsqueeze(0))
    #     self._frustum_initialized = True

    #     print(f"✅ Frustum Initialized! {self.N} points in 3D space. Scale range: {self.scale_range}")


    def forward(self, mlvl_feats, img_metas):
        B = len(img_metas)
        device = mlvl_feats[0].device

        if not self._frustum_initialized:
            self.init_frustum_geometry(img_metas[0])

        base_means = self.init_means.expand(B, -1, -1).to(device)
        anchor_delta = torch.tanh(self.init_means_delta).to(device) * self.anchor_delta_limit.to(device)
        query_means = base_means + anchor_delta
        query_feat = torch.zeros([B, self.N, self.embed_dims], device=device)

        # Decoder 循环
        for i, layer in enumerate(self.decoder_layers):
            query_feat = layer(query_means, query_feat, mlvl_feats, img_metas)
            delta_xyz = self.delta_xyz_heads[i](query_feat)
            query_means = query_means + delta_xyz

        # 循环外独立预测属性
        opacities = self.opacity_head(query_feat)

        scales_norm = self.scale_head(query_feat)
        scale_min, scale_max = self.scale_range
        scales = scale_min + scales_norm * (scale_max - scale_min)

        quats = F.normalize(self.rotation_head(query_feat), dim=-1)

        final_gaussian = torch.cat([query_means, scales, quats, opacities], dim=-1)
        seg_pred = self.seg_pred_heads(query_feat) if hasattr(self, 'seg_pred_heads') else None
        gaussian_preds = [(final_gaussian, seg_pred, query_feat)]

        if self.training and self.save_counter % 200 == 0:
            pts_to_save = query_means[0].detach().cpu().numpy()
            save_path = f"./vis_init_means/iter_{self.save_counter:06d}.npy"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, pts_to_save)

        self.save_counter += 1

        return gaussian_preds

    def anchor_regularization_loss(self):
        anchor_delta = torch.tanh(self.init_means_delta) * self.anchor_delta_limit.to(self.init_means_delta.device)
        return anchor_delta.pow(2).mean() * self.anchor_reg_weight

    def anchor_delta_stats(self):
        anchor_delta = torch.tanh(self.init_means_delta) * self.anchor_delta_limit.to(self.init_means_delta.device)
        return {
            'anchor_delta_mean': anchor_delta.abs().mean().detach(),
            'anchor_delta_max': anchor_delta.abs().max().detach(),
        }



class SparseGaussianDecoderLayer(BaseModule):
    def __init__(self, embed_dims=None, num_frames=None, num_points=None,
                 num_groups=None, num_levels=None, pc_range=None, self_attn_type='none', use_local_attention=True):
        super().__init__()

        # 位置编码
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )
        # self.position_encoder = RandomFourierEmbedding(embed_dims = embed_dims)

        # 自注意力选择
        if self_attn_type == 'distance':
            print("Initializing Layer with Distance-Biased Self Attention (O(N^2))")
            self.self_attn = SparseGaussianSelfAttention(
                embed_dims, num_heads=8, dropout=0.1,
                pc_range=pc_range, scale_adaptive=True
            )
            self.norm1 = nn.LayerNorm(embed_dims)
        elif self_attn_type == 'induced':
            print("Initializing Layer with Induced Self Attention (O(NM))")
            self.self_attn = InducedGaussianAttention(
                in_channels=embed_dims,
                num_heads=8,
                dropout=0.1,
                n_inducing_points=256,
            )
            self.norm1 = nn.LayerNorm(embed_dims)
        else:
            self.self_attn = None

        # 高斯采样
        self.sampling = SparseGaussianSampling(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range
        )

        # 自适应混合
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=num_groups,
            out_points=num_points * num_frames * num_groups
        )

        # 前馈网络
        self.ffn = FFN(embed_dims, feedforward_channels=embed_dims * 2, ffn_drop=0.1)

        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    @torch.no_grad()
    def init_weights(self):
        if self.self_attn is not None:
            if hasattr(self.self_attn, 'init_weights'):
                self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        self.ffn.init_weights()

    def forward(self, query_means, query_feat, mlvl_feats, img_metas):
        """
        Args:
            query_means: [B, N, 3]
            query_feat: [B, N, C]
        """
        query_pos = self.position_encoder(query_means)
        query_feat = query_feat + query_pos

        if self.self_attn is not None:
            if isinstance(self.self_attn, SparseGaussianSelfAttention):
                query_feat = self.norm1(self.self_attn(query_means, query_feat))
            else:
                query_feat = self.norm1(self.self_attn(query_feat))

        sampled_feat = self.sampling(query_means, query_feat, mlvl_feats, img_metas)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))
        return query_feat



class SparseGaussianSelfAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], scale_adaptive=True):
        super().__init__()
        self.pc_range = pc_range

        # 标准 MHA
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)

        # 自适应距离缩放因子
        if scale_adaptive:
            self.gen_tau = nn.Linear(embed_dims, num_heads)
        else:
            self.gen_tau = None

    @torch.no_grad()
    def init_weights(self):
        if self.gen_tau is not None:
            # 初始化策略与 SparseBEV 保持一致
            nn.init.zeros_(self.gen_tau.weight)
            nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def calc_gaussian_dists(self, query_means):
        """
        计算高斯中心点之间的负欧氏距离
        Args:
            query_gaussian: [B, Q, 11]
        Returns:
            dist: [B, Q, Q] (负距离，用于注意力掩码)
        """
        # 计算成对欧氏距离: [B, Q, Q]

        # torch.cdist 计算的是 ||x - y||_2，值为正
        dist_pos = torch.cdist(query_means, query_means, p=2)

        # 返回负距离：距离越远，值越小（负得越多），softmax 后权重越小
        return -dist_pos

    def inner_forward(self, query_means, query_feat, pre_attn_mask=None):
        """
        Args:
            query_gaussian: [B, Q, 11] (替代原 query_bbox)
            query_feat: [B, Q, C]
            pre_attn_mask: 可选的预先计算 mask (一般不再使用，因移除了 active_mask)
        """
        if self.gen_tau is not None:
            # 1. 计算负距离矩阵
            dist = self.calc_gaussian_dists(query_means)  # [B, Q, Q]

            # 2. 预测 Tau
            tau = self.gen_tau(query_feat)  # [B, Q, 8]

            tau = tau.permute(0, 2, 1)  # [B, 8, Q]
            # 3. 计算注意力掩码
            # dist: [B, Q, Q] -> [B, 1, Q, Q]
            # tau:  [B, 8, Q] -> [B, 8, Q, 1]
            # 结果: [B, 8, Q, Q]
            attn_mask = dist[:, None, :, :] * tau[..., None]

            if pre_attn_mask is not None:
                attn_mask[:, :, pre_attn_mask] = float('-inf')

            # 展平以适配 MultiheadAttention
            attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        else:
            attn_mask = None

        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_means, query_feat, pre_attn_mask=None):
        # 保持训练时的梯度检查点逻辑
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_means, query_feat, pre_attn_mask, use_reentrant=False)
        else:
            return self.inner_forward(query_means, query_feat, pre_attn_mask)


class InducedGaussianAttention(nn.Module):
    """
    基于 Induced Attention 的高效全局注意力机制
    复杂度从 O(N^2) 降低到 O(N*M)，M 为诱导点数量
    """
    def __init__(self, in_channels=128, num_heads=8, dropout=0.1, n_inducing_points=256, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.n_inducing_points = n_inducing_points

        # 1. 诱导点：可学习的全局上下文 "瓶颈"
        self.inducing_points = nn.Parameter(torch.randn(n_inducing_points, in_channels))

        # 2. 第一步：Input -> Inducing (信息聚合)
        # 输入：高斯特征 (作为 Key/Value)
        # 查询：诱导点 (作为 Query)
        self.induced_attention = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)
        self.induced_ff = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),  # 建议用 inplace 节省显存
            nn.Linear(in_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.induced_ln1 = nn.LayerNorm(in_channels)
        self.induced_ln2 = nn.LayerNorm(in_channels)

        # 3. 第二步：Inducing -> Output (信息分发)
        # 输入：更新后的诱导点 (作为 Key/Value)
        # 查询：高斯特征 (作为 Query)
        self.output_attention = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout, batch_first=True)
        self.output_ff = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.output_ln1 = nn.LayerNorm(in_channels)
        self.output_ln2 = nn.LayerNorm(in_channels)

    def forward(self, query_feat):
        """
        Args:
            query_feat: [B, N, C] (输入的高斯特征，已经加上了位置编码)
        Returns:
            output: [B, N, C] (更新后的特征)
        """
        B, N, C = query_feat.shape

        # 扩展诱导点以匹配 Batch 维度
        inducing_points = self.inducing_points.unsqueeze(0).expand(B, -1, -1)  # [B, M, C]

        if self.pre_norm:
            # === Stage 1: Aggregate (Input -> Inducing) ===
            # Query: Inducing points; Key/Value: Input features
            q_inducing = self.induced_ln1(inducing_points)
            # k, v 来自 query_feat
            z = self.induced_attention(q_inducing, query_feat, query_feat)[0] + inducing_points
            z = self.induced_ln2(self.induced_ff(z) + z)

            # === Stage 2: Broadcast (Inducing -> Output) ===
            # Query: Input features; Key/Value: Updated Inducing points
            q_output = self.output_ln1(query_feat)
            # k, v 来自 z (更新后的诱导点)
            out = self.output_attention(q_output, z, z)[0] + query_feat
            out = self.output_ln2(self.output_ff(out) + out)

        else:
            # Post-norm implementation (less common in modern Transformers)
            z = self.induced_ln1(self.induced_attention(inducing_points, query_feat, query_feat)[0] + inducing_points)
            z = self.induced_ln2(self.induced_ff(z) + z)
            out = self.output_ln1(self.output_attention(query_feat, z, z)[0] + query_feat)
            out = self.output_ln2(self.output_ff(out) + out)

        return out


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, pc_range=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi
        self.pc_range = pc_range
        self.x_range = pc_range[3] - pc_range[0]
        self.y_range = pc_range[4] - pc_range[1]
        self.z_range = pc_range[5] - pc_range[2]

    def forward(self, xyz):
        # 1. 动态归一化到 [0, 1]
        xyz_norm = xyz.clone()
        xyz_norm[..., 0] = (xyz_norm[..., 0] - self.pc_range[0]) / self.x_range
        xyz_norm[..., 1] = (xyz_norm[..., 1] - self.pc_range[1]) / self.y_range
        xyz_norm[..., 2] = (xyz_norm[..., 2] - self.pc_range[2]) / self.z_range

        # 2. 缩放到 [0, 2pi]
        x_embed = xyz_norm[..., 0] * self.scale
        y_embed = xyz_norm[..., 1] * self.scale
        z_embed = xyz_norm[..., 2] * self.scale

        # 3. 生成频率项
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 4. 计算正弦和余弦并拼接 (修复了 torch.stack 的括号错误)
        # 先计算 embedding，再提取奇偶项，代码更易读
        emb_x = x_embed[..., None] / dim_t
        emb_y = y_embed[..., None] / dim_t
        emb_z = z_embed[..., None] / dim_t

        # 注意这里：用列表 [tensor1, tensor2] 包裹起来传给 stack
        pos_x = torch.stack([emb_x[..., 0::2].sin(), emb_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([emb_y[..., 0::2].sin(), emb_y[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_z = torch.stack([emb_z[..., 0::2].sin(), emb_z[..., 1::2].cos()], dim=-1).flatten(-2)

        return torch.cat((pos_x, pos_y, pos_z), dim=-1)  # [B, N, 384]


class RandomFourierEmbedding(nn.Module):
    def __init__(self, embed_dims=256, sigma=10.0):
        super().__init__()
        # RFF 的标准操作：生成一个固定的随机投影矩阵 B
        # sigma 控制长度尺度。sigma=10 意味着网络对 10 米内的位移最敏感
        # out_dim 必须是偶数，因为一半给 sin，一半给 cos
        out_dim = embed_dims // 2
        self.register_buffer('B', torch.randn(3, out_dim) * sigma)

    def forward(self, xyz):
        # xyz: [B, N, 3]
        # 投影: [B, N, 3] @ [3, out_dim] -> [B, N, out_dim]
        x_proj = xyz @ self.B
        # 拼接 sin 和 cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [B, N, embed_dims]


def halton_sequence(n, dim=3):
    """
    Halton 低差异准蒙特卡洛序列生成器

    与 torch.rand 的区别：
    - torch.rand: 纯随机，可能出现大片空白或过度聚集
    - Halton:     基于素数倒数分解的确定性序列，保证极度均匀

    原理：以不同素数为基底，将整数索引 i 拆分为素数次幂的权重和
    例：base=2 时，i=5 → 5 = 1×4 + 0×2 + 1×1 → 反向 → 0.101₂ = 0.625

    三个轴分别用不同的素数 (2, 3, 5)，彻底打破轴线对齐

    Args:
        n:   生成点数
        dim: 维度 (X=素数2, Y=素数3, Z=素数5)

    Returns:
        seq: [n, dim] 的张量，值域 [0, 1]
    """
    primes = [2, 3, 5]
    seq = torch.zeros(n, dim)
    for i in range(n):
        for d in range(dim):
            x = i
            f = 1.0 / primes[d]
            while x > 0:
                seq[i, d] += f * (x % primes[d])
                x = int(x / primes[d])
                f /= primes[d]
    return seq


# def generate_halton_warped_field(N_target, pc_range):
#     """
#     纯净版初始化策略：
#     1. XY 平面：Halton 序列绝对均匀分布 (不施加任何人为偏见，让网络自己学)
#     2. Z 轴：   分段线性逆 CDF (注入地-车-天的强物理先验)
#     """
#     x_min, y_min, z_min, x_max, y_max, z_max = pc_range
#     x_len = x_max - x_min
#     y_len = y_max - y_min
#     z_len = z_max - z_min
#     volume = x_len * y_len * z_len
#
#     spacing = (volume / N_target) ** (1.0 / 3.0)
#
#     # ================================================================
#     # Step 1: 在单位立方体 [0, 1]³ 中生成极度均匀的点集
#     # ================================================================
#     pts_unit = halton_sequence(N_target, dim=3)
#
#     means = torch.zeros(N_target, 3)
#
#     # ================================================================
#     # Step 2: XY 平面直接拉伸 (不搞极坐标，保持最纯粹的均匀)
#     # ================================================================
#     means[..., 0] = x_min + pts_unit[..., 0] * x_len
#     means[..., 1] = y_min + pts_unit[..., 1] * y_len
#
#     # ================================================================
#     # Step 3: Z 轴 — 分段线性逆 CDF (保留物理先验)
#     # ================================================================
#     u = pts_unit[..., 2]  # [N] 均匀分布 [0, 1]
#
#     # 直接定义区间的左端点，避免切片操作引发 non-contiguous 警告
#     # 30% 给地面 [-1.0, 0.0]，55% 给车体 [0.0, 2.5]，15% 给空中 [2.5, 5.4]
#     cdf_starts = torch.tensor([0.00, 0.30, 0.85])
#     cdf_ends = torch.tensor([0.30, 0.85, 1.00])
#     z_starts = torch.tensor([z_min, 0.0, 2.5])
#     z_ends = torch.tensor([0.0, 2.5, z_max])
#
#     # 找到每个 u 落在哪个区间
#     idx = torch.searchsorted(cdf_ends.contiguous(), u.contiguous(), right=True) # [N], 值域 [0, 3]
#     idx = idx.clamp(0, 2)  # 限制在有效区间索引 [0, 1, 2] 内
#
#     # 提取对应区间的起止点
#     u_s = cdf_starts[idx]
#     u_e = cdf_ends[idx]
#     z_s = z_starts[idx]
#     z_e = z_ends[idx]
#
#     # 区间内线性插值
#     denom = u_e - u_s
#     t = torch.where(denom > 1e-6, (u - u_s) / denom, torch.zeros_like(u))
#
#     means[..., 2] = z_s + t * (z_e - z_s)
#
#     # ================================================================
#     # Step 4: 尺度初始化 (保持 0.5 倍 spacing)
#     # ================================================================
#     scales = torch.full((N_target, 3), spacing)
#
#     return means, scales, N_target


def generate_halton_warped_field(N_target, pc_range, front_view_only=True):
    """
    单目前视适配版初始化：
    1. 保留原有的 Halton 均匀生长 + Z轴逆CDF物理先验
    2. 新增前视视锥过滤，剔除侧面和后方拿不到特征的死区高斯
    """
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    x_len = x_max - x_min
    y_len = y_max - y_min
    z_len = z_max - z_min
    volume = x_len * y_len * z_len
    spacing = (volume / N_target) ** (1.0 / 3.0)

    # ==========================================
    # 1. 生成候选点 (为了过滤后数量足够，多生成3倍)
    # ==========================================
    N_sample = N_target * 3 if front_view_only else N_target
    pts_unit = halton_sequence(N_sample, dim=3)
    means = torch.zeros(N_sample, 3)

    # ==========================================
    # 2. XY 平面拉伸
    # ==========================================
    means[..., 0] = x_min + pts_unit[..., 0] * x_len
    means[..., 1] = y_min + pts_unit[..., 1] * y_len

    # ==========================================
    # 3. Z 轴分段线性逆 CDF (保留地-车-天先验)
    # ==========================================
    u = pts_unit[..., 2]
    cdf_starts = torch.tensor([0.00, 0.30, 0.85])
    cdf_ends = torch.tensor([0.30, 0.85, 1.00])
    z_starts = torch.tensor([z_min, 0.0, 2.5])
    z_ends = torch.tensor([0.0, 2.5, z_max])

    idx = torch.searchsorted(cdf_ends.contiguous(), u.contiguous(), right=True)
    idx = idx.clamp(0, 2)

    u_s = cdf_starts[idx]
    u_e = cdf_ends[idx]
    z_s = z_starts[idx]
    z_e = z_ends[idx]

    denom = u_e - u_s
    t = torch.where(denom > 1e-6, (u - u_s) / denom, torch.zeros_like(u))
    means[..., 2] = z_s + t * (z_e - z_s)

    # ==========================================
    # 4. 前视视锥过滤 (核心修改)
    # ==========================================
    if front_view_only:
        # NuScenes 前视相机(CAM_FRONT)在 LiDAR 坐标系下的近似位置
        # x 朝前, y 朝左, z 朝上
        cam_x, cam_y, cam_z = 1.7, 0.0, 1.5

        # 前视相机水平FOV约70度(半角35度, tan≈0.70)，垂直FOV约50度(半角25度, tan≈0.46)
        # 为了防止边缘目标丢失，留出一定余量
        max_tan_h = 0.85  # 水平方向余量
        max_tan_v = 0.65  # 垂直方向余量

        dx = means[..., 0] - cam_x
        dy = means[..., 1] - cam_y
        dz = means[..., 2] - cam_z

        # 条件1: 必须在相机前方 (x > cam_x)
        front_mask = (dx > 0)
        # 条件2: 在水平视场角内
        h_mask = (torch.abs(dy) < dx * max_tan_h)
        # 条件3: 在垂直视场角内
        v_mask = (torch.abs(dz) < dx * max_tan_v)

        frustum_mask = front_mask & h_mask & v_mask
        means = means[frustum_mask]

        # 如果过滤后点数不足，则重复现有的点补齐（保证张量形状稳定）
        if means.shape[0] < N_target:
            repeat_idx = torch.randint(0, means.shape[0], (N_target - means.shape[0],))
            means = torch.cat([means, means[repeat_idx]], dim=0)
        elif means.shape[0] > N_target:
            # 随机打散后截取，避免边缘聚集
            perm = torch.randperm(means.shape[0])[:N_target]
            means = means[perm]

    N_actual = means.shape[0]
    scales = torch.full((N_actual, 3), spacing)

    return means, scales, N_actual