"""Pose-guided source-visible to target-occluded supervision geometry.

The geometry in this module only consumes pseudo image supervision and camera
poses.  OCC labels and observation-state masks deliberately do not appear in
the interface so that the auxiliary loss cannot leak evaluation GT.
"""

import torch
import torch.nn.functional as F


def source_visible_to_target_occluded(
        source_depth,
        source_semantic,
        source_weight,
        target_depth,
        source_intrinsic,
        target_intrinsic,
        source_to_target_camera,
        depth_min=0.1,
        depth_max=50.0,
        occlusion_abs_margin=0.5,
        occlusion_rel_margin=0.05):
    """Return a source-pixel mask whose surface is behind target depth.

    ``source_to_target_camera`` maps source-camera coordinates into
    target-camera coordinates.  Metric3D stores camera z depth, so pixels are
    back-projected as ``[u*d, v*d, d]``.  The returned mask has shape
    ``[B,H,W]`` and is detached from all pseudo-label tensors.
    """
    if source_depth.ndim != 3 or target_depth.ndim != 3:
        raise ValueError('depth tensors must have shape [B,H,W]')
    if source_depth.shape != source_semantic.shape:
        raise ValueError('source depth/semantic shape mismatch')
    if source_weight.shape != source_depth.shape:
        raise ValueError('source weight/depth shape mismatch')
    batch, height, width = source_depth.shape
    if target_depth.shape != (batch, height, width):
        raise ValueError('target/source image shapes must match')

    with torch.no_grad():
        device = source_depth.device
        dtype = source_depth.dtype
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij')
        pixels = torch.stack((xx, yy, torch.ones_like(xx)), dim=0)
        pixels = pixels.reshape(3, -1).unsqueeze(0).expand(batch, -1, -1)

        source_depth = source_depth.float()
        target_depth = target_depth.float()
        source_weight = source_weight.float()
        source_points = torch.bmm(
            torch.linalg.inv(source_intrinsic.float()), pixels)
        source_points = source_points * source_depth.reshape(batch, 1, -1)
        source_points_h = torch.cat(
            (source_points, torch.ones(batch, 1, height * width,
                                       device=device)), dim=1)
        target_points = torch.bmm(
            source_to_target_camera.float(), source_points_h)[:, :3]
        target_z = target_points[:, 2]

        projected = torch.bmm(target_intrinsic.float(), target_points)
        projected_xy = projected[:, :2] / projected[:, 2:3].clamp_min(1e-6)
        u = projected_xy[:, 0]
        v = projected_xy[:, 1]
        inside = (
            (target_z > depth_min) &
            (u >= 0) & (u <= width - 1) &
            (v >= 0) & (v <= height - 1))

        # Sample target pseudo depth at the projected location.  Border values
        # are zero and are rejected by target_valid below.
        grid = torch.stack((
            2 * u / max(width - 1, 1) - 1,
            2 * v / max(height - 1, 1) - 1), dim=-1)
        sampled_target = F.grid_sample(
            target_depth[:, None], grid.reshape(batch, height, width, 2),
            mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_target = sampled_target[:, 0].reshape(batch, -1)
        target_valid = (
            (sampled_target > depth_min) &
            (sampled_target < depth_max))

        source_valid = (
            (source_depth.reshape(batch, -1) > depth_min) &
            (source_depth.reshape(batch, -1) < depth_max) &
            (source_semantic.reshape(batch, -1) != 255) &
            (source_semantic.reshape(batch, -1) > 0) &
            (source_weight.reshape(batch, -1) > 0))
        tolerance = torch.maximum(
            torch.full_like(sampled_target, float(occlusion_abs_margin)),
            sampled_target.abs() * float(occlusion_rel_margin))
        mask = source_valid & inside & target_valid & (
            target_z > sampled_target + tolerance)
        return mask.reshape(batch, height, width)
