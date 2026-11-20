import torch
import numpy as np
import torch.nn.functional as F

def decode_gaussian(query_gaussian):
    """
    解码高斯参数 (用于最终渲染或可视化)
    """
    xyz = query_gaussian[..., :3]
    scales = query_gaussian[..., 3:6]
    quats = query_gaussian[..., 6:10]
    opacities = query_gaussian[..., 10:11]

    return xyz, scales, quats, opacities


def quaternion_to_matrix(quaternions):
    """
    将四元数转换为旋转矩阵
    Args:
        quaternions: [B, N, 4] (w,x,y,z)
    Returns:
        [B, N, 3, 3]
    """
    w, x, y, z = torch.unbind(quaternions, dim=-1)
    # 计算旋转矩阵元素
    R = torch.stack([
        1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
        2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
        2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y
    ], dim=-1)

    # 不管输入是 [N, 4] 还是 [B, N, 4]，都能正确变成 [..., 3, 3]
    return R.reshape(*quaternions.shape[:-1], 3, 3)


# https://github.com/MCG-NJU/SparseOcc/blob/main/utils.py
def batch_indexing(batched_data, batched_indices, layout='channel_last'):
    """
    复用自 SparseOcc，支持 channel_last (B, N, C) 格式
    """

    def batch_indexing_channel_last(batched_data, batched_indices):
        assert batched_data.shape[0] == batched_indices.shape[0]
        batch_size = batched_data.shape[0]
        view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
        expand_shape = [batch_size] + list(batched_indices.shape)[1:]
        indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
        indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)
        if len(batched_data.shape) == 2:
            return batched_data[indices_of_batch, batched_indices.to(torch.long)]
        else:
            return batched_data[indices_of_batch, batched_indices.to(torch.long), :]

    if layout == 'channel_last':
        return batch_indexing_channel_last(batched_data, batched_indices)
    else:
        raise NotImplementedError("Only channel_last layout is supported in this context.")


# https://github.com/hustvl/GaussTR/blob/main/gausstr/models/utils.py
def apply_to_items(func, iterable):
    if isinstance(iterable, list):
        return [func(i) for i in iterable]
    elif isinstance(iterable, dict):
        return {k: func(v) for k, v in iterable.items()}


# https://github.com/hustvl/GaussTR/blob/main/gausstr/models/utils.py
def unbatched_forward(func):
    def wrapper(*args, **kwargs):
        bs = None
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, torch.Tensor):
                if bs is None:
                    bs = arg.size(0)
                else:
                    assert bs == arg.size(0)

        outputs = []
        for i in range(bs):
            output = func(
                *[
                    arg[i] if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ], **{
                    k: v[i] if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                })
            outputs.append(output)

        if isinstance(outputs[0], tuple):
            return tuple([
                torch.stack([out[i] for out in outputs])
                for i in range(len(outputs[0]))
            ])
        else:
            return torch.stack(outputs)

    return wrapper


# https://github.com/hustvl/GaussTR/blob/main/gausstr/models/utils.py
def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [*grid_shape, len(grid_shape)]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= val
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(*shape_).expand(*grid_shape)
        grid.append(g)
    return torch.stack(grid, dim=-1)