import torch
from g2v import _C


def voxelizer(
        means3d,
        opacities,
        precisions,
        features,
        vol_range,
        voxel_size,
        grid_shape,
        mode="adaptive",  # "dynamic" | "fixed" | "adaptive"
        covariances=None,
        max_neighborhood=12,
        chunk_size=10000
):
    """
    高斯体素化器

    Args:
        mode: 
            - "dynamic": 高精度模式，动态计算3σ边界
            - "fixed": 高速模式，固定邻域半径
            - "adaptive": 平衡模式，自适应邻域半径（推荐）
    """
    if not isinstance(vol_range, torch.Tensor):
        vol_range = torch.tensor(vol_range, device=means3d.device, dtype=means3d.dtype)

    N = means3d.size(0)
    W, H, D = grid_shape
    C = features.size(1)

    density_out = torch.zeros((W, H, D, 1), device=means3d.device, dtype=torch.float32)
    feats_out = torch.zeros((W, H, D, C), device=means3d.device, dtype=torch.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        if mode == "dynamic":
            density_chunk, feats_chunk = _C.forward_dynamic(
                means3d[start:end].contiguous(),
                opacities[start:end].contiguous(),
                precisions[start:end].view(-1, 9).contiguous(),
                features[start:end].contiguous(),
                covariances[start:end].view(-1, 9).contiguous(),
                vol_range.contiguous(),
                voxel_size,
                grid_shape
            )
        elif mode == "fixed":
            density_chunk, feats_chunk = _C.forward_fixed(
                means3d[start:end].contiguous(),
                opacities[start:end].contiguous(),
                precisions[start:end].view(-1, 9).contiguous(),
                features[start:end].contiguous(),
                vol_range.contiguous(),
                voxel_size,
                grid_shape,
                max_neighborhood
            )
        elif mode == "adaptive":
            density_chunk, feats_chunk = _C.forward_adaptive(
                means3d[start:end].contiguous(),
                opacities[start:end].contiguous(),
                precisions[start:end].view(-1, 9).contiguous(),
                features[start:end].contiguous(),
                covariances[start:end].view(-1, 9).contiguous(),
                vol_range.contiguous(),
                voxel_size,
                grid_shape,
                max_neighborhood
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from: dynamic, fixed, adaptive")

        density_out += density_chunk
        feats_out += feats_chunk

    # 归一化
    feats_out = feats_out / density_out.clamp(1e-6)
    return density_out, feats_out
