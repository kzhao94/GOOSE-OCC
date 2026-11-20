import torch
import torch.nn as nn
from .gaussian_utils import apply_to_items, generate_grid, unbatched_forward
from .gsplat_backend import preload_cached_gsplat

preload_cached_gsplat()
from gsplat import quat_scale_to_covar_preci


class GaussianVoxelizer(nn.Module):
    def __init__(self,
                 vol_range,
                 voxel_size,
                 filter_gaussians=False,
                 opacity_thresh=0,
                 covariance_thresh=0,
                 mode="dynamic",
                 max_neighborhood=12,
                 chunk_size=60000):
        super().__init__()
        if mode not in ("dynamic", "fixed", "adaptive"):
            raise ValueError(
                f"Unsupported voxelizer mode: {mode}. "
                "Expected one of: dynamic, fixed, adaptive.")
        self.voxel_size = voxel_size
        self.mode = mode
        self.max_neighborhood = max_neighborhood
        self.chunk_size = chunk_size
        vol_range = torch.tensor(vol_range, dtype=torch.float32)
        self.register_buffer('vol_range', vol_range)

        self.grid_shape = ((vol_range[3:] - vol_range[:3]) /
                           voxel_size).int().tolist()
        grid_coords = generate_grid(self.grid_shape, offset=0.5)
        grid_coords = grid_coords * voxel_size + vol_range[:3]
        self.register_buffer('grid_coords', grid_coords)

        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh
        self.covariance_thresh = covariance_thresh

    @unbatched_forward
    def forward(self,
                means3d=None,  # 3D位置
                scales=None,
                rotations=None,
                opacities=None,
                features=None,  # 特征概率
                ):
        # # ==========================================
        # # 0. 终极优化：先过滤，再算协方差！(省去大量无效的 CUDA 计算)
        # # ==========================================
        # if self.filter_gaussians:
        #     # 阈值1：淘汰 opacity 接近 0 的废节点 (Decoder 饿死的节点)
        #     # sigmoid(-5) ≈ 0.006，设为 0.005 可以精准拦截
        #     mask = opacities.squeeze(1) > self.opacity_thresh
        #
        #     # # 阈值2：淘汰飞出场景边界的节点
        #     # for i in range(3):
        #     #     mask &= (means3d[:, i] >= self.vol_range[i]) & (
        #     #             means3d[:, i] <= self.vol_range[i + 3])
        #
        #     # 阈值3：淘汰极细高斯 (如果最短轴的 6σ 都不够一个体素大小，直接淘汰)
        #     if self.covariance_thresh > 0:
        #         # 因为没有算 covariances，我们直接用 scales 近似判断 (scales 是 1σ)
        #         # 6σ = scales.min * 6
        #         min_6sigma = scales.min(dim=1)[0] * 6.0
        #         mask &= (min_6sigma > self.covariance_thresh)
        #
        #     # 执行过滤
        #     means3d = means3d[mask]
        #     scales = scales[mask]
        #     rotations = rotations[mask]
        #     opacities = opacities[mask]
        #     features = features[mask]

        # 使用 gsplat 的 CUDA 算子高效计算协方差和精度矩阵
        covariances, precisions = quat_scale_to_covar_preci(
            quats=rotations,
            scales=scales,
            compute_covar=True,
            compute_preci=True,
            triu=False
        )

        gaussians = dict(
            means3d=means3d,
            opacities=opacities,
            covariances=covariances,
            precisions=precisions,  # 传 precisions 而非 covariances（避免求逆）
            features=features)

        # ===========================================
        # ---------------方法1：GaussTR---------------
        # ===========================================
        # return splat_into_3d(
        #     self.grid_coords,
        #     **gaussians,
        #     vol_range=self.vol_range,
        #     voxel_size=self.voxel_size)

        # ====================================================================
        # ---------------方法2：参考GaussianFlowOCC实现的向量化版本---------------
        # ====================================================================
        # return splat_into_3d_vectorized(
        #     self.grid_coords,
        #     **gaussians,
        #     vol_range=self.vol_range,
        #     voxel_size=self.voxel_size,
        #     max_neighborhood=6,  # 固定邻域大小（3σ 覆盖）
        #     chunk_size=90000  # 分批处理大小（根据 GPU 显存调整）
        # )

        # ================================================================
        # ---------------方法3：CUDA版本（同时包含方法1和方法2）---------------
        # ================================================================
        from g2v import voxelizer
        grid_density, grid_feats = voxelizer(
            gaussians['means3d'],  # (N, 3)
            gaussians['opacities'],  # (N, 1)
            gaussians['precisions'],  # (N, 3, 3)
            gaussians['features'],  # (N, C)
            self.vol_range,  # (6,)
            self.voxel_size,
            self.grid_shape,
            mode=self.mode,  # "dynamic" | "fixed" | "adaptive"
            covariances=gaussians['covariances'],  # needed for dynamic and adaptive mode
            max_neighborhood=self.max_neighborhood,  # used in fixed and adaptive mode
            chunk_size=self.chunk_size
        )
        # kitti
        # dynamic 101.6268 秒    miou 2.21
        # fixed 74.1947 秒       miou 2.20  max_neighborhood=12
        # fixed 48.3116 秒       miou 1.83  max_neighborhood=6
        # adaptive 72.3002 秒    miou 2.20  max_neighborhood=12

        return grid_density, grid_feats


def splat_into_3d(grid_coords,
                  means3d,
                  opacities,
                  covariances,
                  vol_range,
                  voxel_size,
                  features=None,
                  eps=1e-6):
    """
        原始版本：逐个高斯进行体素化（串行实现，仅用于对比或小规模场景）

        该函数对每个高斯单独处理：
          1. 计算其 3σ 影响范围（动态 bounds）
          2. 在该范围内计算马氏距离
          3. 累加密度和特征到体素网格

        缺点：Python for 循环导致无法并行，高斯数量大时极慢。
        """
    # 初始化体素网格：密度图 (W, H, D, 1)
    grid_density = torch.zeros((*grid_coords.shape[:-1], 1),
                               device=grid_coords.device)
    # 初始化特征图 (W, H, D, C)
    if features is not None:
        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                 device=grid_coords.device)

    # 逐个处理每个高斯（串行，瓶颈所在）
    for g in range(means3d.size(0)):
        # 从协方差矩阵提取标准差（sigma = sqrt(diag(Σ))）
        sigma = torch.sqrt(torch.diag(covariances[g]))
        # 计算 3σ 范围：[-3σ, +3σ]
        factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma)
        bounds = means3d[g, None] + factor * sigma[None]
        # 检查高斯是否在体素空间范围内，若完全在外则跳过
        if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                ((bounds < vol_range[None, 3:]).max(0).values.min())):
            continue
        # 将 bounds 限制在体素空间内
        bounds = bounds.clamp(vol_range[:3], vol_range[3:])
        # 转换为体素索引（整数）
        bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist()
        # 构建切片对象，用于索引体素网格
        slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)])

        # 计算体素中心与高斯均值的差值
        diff = grid_coords[slices] - means3d[g]
        # 计算马氏距离：diff^T @ Σ⁻¹ @ diff
        maha_dist = (diff.unsqueeze(-2) @ covariances[g].inverse()
                     @ diff.unsqueeze(-1)).squeeze(-1)
        # 计算密度贡献：opacity * exp(-0.5 * maha_dist)
        density = opacities[g] * torch.exp(-0.5 * maha_dist)
        grid_density[slices] += density
        if features is not None:
            # 累加加权特征
            grid_feats[slices] += density * features[g]

    if features is None:
        return grid_density
    # 特征归一化：加权平均（避免高 opacity 主导）
    grid_feats /= grid_density.clamp(eps)
    # print("grid_feats min/max:", grid_feats.min().item(), grid_feats.max().item())

    return grid_density, grid_feats


def splat_into_3d_vectorized(
    grid_coords,
    means3d,
    opacities,
    precisions,
    vol_range,
    voxel_size,
    features=None,
    eps=1e-6,
    max_neighborhood=3,   # 定义每个高斯在体素网格中最多影响多少个体素
                          # 基于 3σ 原则：高斯分布在 ±3σ 范围内覆盖 99.7% 的概率质量
                          # - max_neighborhood=1 → 3×3×3 = 27 个体素
                          # - max_neighborhood=2 → 5×5×5 = 125 个体素
                          # - max_neighborhood=3 → 7×7×7 = 343 个体素（默认）
    chunk_size=30000,     # 每批最多处理 chunk_size 个高斯
                          # 作用：避免一次性分配过大中间张量导致 GPU 显存溢出（OOM）
                          # 注意：不会丢弃数据，只是分批处理
):
    """
        优化版本：向量化体素化（固定邻域，高效并行）

        核心思想：
          - 借鉴 gaussians_to_occupancy 的固定邻域策略
          - 所有高斯并行处理，无 Python for 循环
          - 预计算 precision matrix (Σ⁻¹)，避免重复求逆
          - 使用 chunking 控制显存峰值
        """
    device = grid_coords.device
    W, H, D = grid_coords.shape[:3]  # 体素网格尺寸
    N = means3d.size(0)  # 高斯总数

    # 将体素网格展平为 (W*H*D, 3)，便于后续高级索引
    voxel_coords_flat = grid_coords.view(-1, 3)  # (W*H*D, 3)
    # 初始化密度图（展平形式）
    grid_density = torch.zeros((W * H * D, 1), device=device)
    if features is not None:
        # 初始化特征图（展平形式）
        grid_feats = torch.zeros((W * H * D, features.size(-1)), device=device)

    # 预计算邻域偏移量：例如 max_neighborhood=2 → offsets = [-2,-1,0,1,2]
    nb = torch.arange(-max_neighborhood, max_neighborhood + 1, device=device)
    # 生成 3D 网格偏移 (K, 3)，K = (2*max_neighborhood+1)^3
    neighborhood = torch.stack(torch.meshgrid(nb, nb, nb, indexing='ij'), dim=-1).reshape(-1, 3)
    K = neighborhood.shape[0]

    # 分批处理高斯，防止显存溢出
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        means_chunk = means3d[start:end]  # (C, 3)

        # 计算每个高斯中心所在的体素索引（整数坐标）
        base_vox = ((means_chunk - vol_range[:3]) / voxel_size).long()
        # 扩展到邻域：每个高斯生成 K 个体素候选 (C, K, 3)
        neigh_vox = base_vox[:, None, :] + neighborhood[None, :, :]  # (C, K, 3)

        # 掩码1：过滤超出体素网格边界的体素
        valid_mask = (
            (neigh_vox >= 0).all(dim=-1) &
            (neigh_vox < torch.tensor([W, H, D], device=device)).all(dim=-1)
        )  # (C, K)

        # 若当前批次无有效体素，跳过
        if not valid_mask.any():
            continue

        # 展平所有有效 (高斯, 体素) 对
        flat_neigh = neigh_vox[valid_mask]  # (V, 3)
        # 生成对应的高斯 ID（全局索引）
        gauss_ids = torch.arange(start, end, device=device).view(-1, 1).expand(-1, K)
        flat_gauss_ids = gauss_ids[valid_mask]  # (V,)

        # 将 3D 体素索引转换为 1D 索引（用于展平网格）
        flat_idx = flat_neigh[:, 0] * H * D + flat_neigh[:, 1] * D + flat_neigh[:, 2]
        # 获取体素世界坐标
        voxel_coords = voxel_coords_flat[flat_idx]  # (V, 3)
        # 获取对应高斯的均值
        means_sel = means3d[flat_gauss_ids]  # (V, 3)
        # 计算差值
        diff = voxel_coords - means_sel  # (V, 3)

        # 使用预计算的 precision matrix (Σ⁻¹) 计算马氏距离
        prec_sel = precisions[flat_gauss_ids]  # (V, 3, 3)
        # 等价于: diff @ prec_sel，但更高效
        diff_prec = torch.bmm(diff.unsqueeze(1), prec_sel).squeeze(1)  # (V, 3)
        maha = torch.sum(diff_prec * diff, dim=1)  # (V,)

        # 计算密度贡献
        density = opacities[flat_gauss_ids].squeeze(1) * torch.exp(-0.5 * maha)

        # 使用 index_add_ 高效累加到密度图
        grid_density.index_add_(0, flat_idx, density.unsqueeze(1))
        if features is not None:
            # 累加加权特征
            feat_contrib = density.unsqueeze(1) * features[flat_gauss_ids]
            grid_feats.index_add_(0, flat_idx, feat_contrib)

    # 重塑回原始网格形状
    grid_density = grid_density.view(W, H, D, 1)
    if features is None:
        return grid_density
    grid_feats = grid_feats.view(W, H, D, -1)
    # 特征归一化：加权平均
    grid_feats /= grid_density.clamp(eps)
    return grid_density, grid_feats
