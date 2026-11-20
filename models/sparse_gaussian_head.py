import torch
import torch.nn as nn
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.builder import build_loss
from mmdet.models.utils import build_transformer
import torch.nn.functional as F
from .gsplat_backend import preload_cached_gsplat

preload_cached_gsplat()
from gsplat import rasterization
from .gaussian_utils import decode_gaussian
from .loss import MultiLoss
from .temporal_occlusion import source_visible_to_target_occluded
import os


@HEADS.register_module()
class SparseGaussianHead(nn.Module):
    def __init__(self,
                 transformer=None,
                 class_names=None,
                 embed_dims=None,
                 occ_size=None,
                 pc_range=None,
                 loss_cfgs=None,
                 panoptic=False,
                 history_loss_weight=0.25,
                 history_semantic_ce_only=False,
                 occlusion_loss_weight=0.0,
                 occlusion_abs_margin=0.5,
                 occlusion_rel_margin=0.05,
                 debug_vis=False,  # 训练可视化默认关闭
                 debug_vis_interval=1000,
                 debug_vis_dir='./vis_result',
                 **kwargs):
        super(SparseGaussianHead, self).__init__()
        self._iter_counter = 0  # 总调用次数
        self._last_vis_epoch = -1
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.pc_range = pc_range
        self.occ_size = occ_size
        self.embed_dims = embed_dims
        self.score_threshold = 0.3
        self.overlap_threshold = 0.8
        self.panoptic = panoptic
        self.history_loss_weight = history_loss_weight
        self.history_semantic_ce_only = history_semantic_ce_only
        self.occlusion_loss_weight = float(occlusion_loss_weight)
        self.occlusion_abs_margin = float(occlusion_abs_margin)
        self.occlusion_rel_margin = float(occlusion_rel_margin)
        self.debug_vis = debug_vis
        self.debug_vis_interval = debug_vis_interval
        self.debug_vis_dir = debug_vis_dir

        self.transformer = build_transformer(transformer)
        self.criterion = build_loss(loss_cfgs)

    def init_weights(self):
        self.transformer.init_weights()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        # gaussian_preds 和 depth_logits 都从 transformer 返回
        gaussian_preds = self.transformer(mlvl_feats, img_metas=img_metas)
        return {
            'gaussian_preds': gaussian_preds,
        }

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_semantic, gt_depth, outs, img_metas,
             gt_semantic_weight=None):
        loss_dict = {}

        self.loss_single(
            outs['gaussian_preds'][-1], loss_dict, gt_semantic, gt_depth,
            img_metas, gt_semantic_weight)
        gaussian_decoder = self.transformer.gaussian_decoder
        loss_dict['loss_anchor_reg'] = gaussian_decoder.anchor_regularization_loss()
        loss_dict.update(gaussian_decoder.anchor_delta_stats())
        means, scales, _, opacities = decode_gaussian(outs['gaussian_preds'][-1][0])
        loss_dict.update({
            'gaussian_scale_mean': scales.mean().detach(),
            'gaussian_scale_max': scales.max().detach(),
            'gaussian_opacity_mean': opacities.mean().detach(),
        })

        return loss_dict

    def loss_single(self, gaussian_pred, loss_dict, gt_semantic, gt_depth,
                    img_metas, gt_semantic_weight=None):
        # Unpack
        rendered = rasterize_gaussians(gaussian_pred, img_metas)
        rendered_depth = rendered[:, :, -1, :, :]  # [B, V, C, H, W]
        rendered_seg = rendered[:, :, :-1, :, :]  # [B, V, H, W]

        gt_semantic = torch.stack(gt_semantic, dim=1)
        gt_depth = torch.stack(gt_depth, dim=1)
        if gt_semantic_weight is not None:
            gt_semantic_weight = torch.stack(
                gt_semantic_weight, dim=1).to(rendered_seg.dtype)

        B, V, C, H, W = rendered_seg.shape
        assert gt_semantic.shape == (B, V, H, W)
        assert gt_depth.shape == (B, V, H, W)
        if gt_semantic_weight is not None:
            assert gt_semantic_weight.shape == (B, V, H, W)

        if self.debug_vis and self.debug_vis_interval > 0 and self._iter_counter % self.debug_vis_interval == 0:
            from vis import visualize_batch
            save_path = os.path.join(self.debug_vis_dir, f'vis_result_iter_{self._iter_counter:06d}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            visualize_batch(rendered_seg, gt_semantic, rendered_depth, gt_depth, batch_idx=0, save_path=save_path)
        self._iter_counter += 1

        current_sem, current_depth = self.criterion.compute_raw_losses(
            rendered_seg[:, 0].float(),
            gt_semantic[:, 0].long(),
            rendered_depth[:, 0],
            gt_depth[:, 0],
            None if gt_semantic_weight is None else gt_semantic_weight[:, 0],
        )
        loss_dict['semantic_raw_current'] = current_sem.detach()
        loss_dict['depth_raw_current'] = current_depth.detach()

        if V > 1 and self.history_loss_weight > 0:
            hist_sem, hist_depth = self.criterion.compute_raw_losses(
                rendered_seg[:, 1:].reshape(B * (V - 1), C, H, W).float(),
                gt_semantic[:, 1:].reshape(B * (V - 1), H, W).long(),
                rendered_depth[:, 1:].reshape(B * (V - 1), H, W),
                gt_depth[:, 1:].reshape(B * (V - 1), H, W),
                None if gt_semantic_weight is None else
                gt_semantic_weight[:, 1:].reshape(B * (V - 1), H, W),
                include_lovasz=not self.history_semantic_ce_only,
            )
            loss_sem_raw = current_sem + self.history_loss_weight * hist_sem
            loss_depth_raw = current_depth + self.history_loss_weight * hist_depth
            loss_dict['semantic_raw_history'] = hist_sem.detach()
            loss_dict['depth_raw_history'] = hist_depth.detach()
            if gt_semantic_weight is not None:
                loss_dict['history_confidence_mean'] = (
                    gt_semantic_weight[:, 1:].mean().detach())
        else:
            loss_sem_raw = current_sem
            loss_depth_raw = current_depth

        occlusion_loss_weight = getattr(self, 'occlusion_loss_weight', 0.0)
        if V > 1 and occlusion_loss_weight > 0:
            occlusion_losses = []
            occlusion_pixels = []
            target_depth = gt_depth[:, 0]
            target_intrinsic = self._meta_matrices(
                img_metas, 'cam2img', 0, rendered_seg.device)
            target_pose = self._meta_matrices(
                img_metas, 'ego2cam', 0, rendered_seg.device)
            for frame in range(1, V):
                source_intrinsic = self._meta_matrices(
                    img_metas, 'cam2img', frame, rendered_seg.device)
                source_pose = self._meta_matrices(
                    img_metas, 'ego2cam', frame, rendered_seg.device)
                source_to_target = target_pose @ torch.linalg.inv(source_pose)
                source_semantic_weight = (
                    torch.ones_like(gt_depth[:, frame])
                    if gt_semantic_weight is None else
                    gt_semantic_weight[:, frame])
                mask = source_visible_to_target_occluded(
                    gt_depth[:, frame], gt_semantic[:, frame],
                    source_semantic_weight,
                    target_depth, source_intrinsic, target_intrinsic,
                    source_to_target,
                    depth_max=self.criterion.depth_max,
                    occlusion_abs_margin=getattr(
                        self, 'occlusion_abs_margin', 0.5),
                    occlusion_rel_margin=getattr(
                        self, 'occlusion_rel_margin', 0.05))
                source_target = gt_semantic[:, frame].long().clone()
                source_target[~mask] = self.criterion.ignore_index
                source_weight = mask.to(rendered_seg.dtype)
                if gt_semantic_weight is not None:
                    source_weight = source_weight * gt_semantic_weight[:, frame]
                occlusion_losses.append(self._semantic_ce(
                    rendered_seg[:, frame], source_target, source_weight))
                occlusion_pixels.append(mask.float().sum())
            if occlusion_losses:
                occlusion_sem = torch.stack(occlusion_losses).mean()
                loss_sem_raw = loss_sem_raw + occlusion_loss_weight * occlusion_sem
                loss_dict['semantic_raw_occlusion'] = occlusion_sem.detach()
                loss_dict['occlusion_pixels'] = torch.stack(occlusion_pixels).mean().detach()

        if self.criterion.use_uncertainty:
            precision_sem = torch.exp(-self.criterion.log_var_sem)
            precision_depth = torch.exp(-self.criterion.log_var_depth)
            loss_dict['learned_weight_sem'] = precision_sem.detach()
            loss_dict['learned_weight_depth'] = precision_depth.detach()
            loss_dict['loss_sem'] = precision_sem * loss_sem_raw + 0.5 * self.criterion.log_var_sem
            loss_dict['loss_depth'] = precision_depth * loss_depth_raw + 0.5 * self.criterion.log_var_depth
        else:
            loss_dict['loss_sem'] = loss_sem_raw
            loss_dict['loss_depth'] = loss_depth_raw

    @staticmethod
    def _meta_matrices(img_metas, key, view_index, device):
        return torch.stack([
            torch.as_tensor(meta[key][view_index], dtype=torch.float32,
                            device=device)
            for meta in img_metas
        ])

    def _semantic_ce(self, prediction, target, semantic_weight):
        valid = (target != self.criterion.ignore_index) & (semantic_weight > 0)
        if not valid.any():
            return prediction.sum() * 0.0
        class_weights = self.criterion.class_weights
        if class_weights is not None:
            class_weights = class_weights.to(prediction.device)
        loss_map = F.cross_entropy(
            prediction, target, weight=class_weights,
            ignore_index=self.criterion.ignore_index, reduction='none')
        weights = semantic_weight.to(prediction.dtype)
        return (loss_map[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-6)


def rasterize_gaussians(gaussian_pred, img_metas):
    gaussians = gaussian_pred[0]
    means, scales, quats, opacities = decode_gaussian(gaussians)
    colors = gaussian_pred[1]
    device = gaussians.device

    # 构造 viewmats: [B, V, 4, 4]
    viewmats = torch.stack([
        torch.stack([torch.from_numpy(arr).to(device, dtype=torch.float32) for arr in meta['ego2cam']], dim=0)
        for meta in img_metas
    ], dim=0)

    # 构造 Ks: [B, V, 3, 3]
    Ks = torch.stack([
        torch.stack([torch.from_numpy(arr).to(device, dtype=torch.float32) for arr in meta['cam2img']], dim=0)
        for meta in img_metas
    ], dim=0)

    height, width = img_metas[0]['render_size']

    rendered_image = rasterization(
        means,  # 3D高斯位置
        quats,  # 高斯旋转
        scales,  # 高斯尺度
        opacities.squeeze(-1),  # 高斯不透明度
        colors,  # 高斯颜色
        viewmats,  # 世界到相机的视图矩阵
        Ks,  # 相机内参
        width,  # 输出图像宽度
        height,  # 输出图像高度
        # near_plane=0.1,  # 近平面
        # far_plane=100,  # 远平面
        render_mode='RGB+ED',  # 渲染模式（RGB+深度）
        # channel_chunk=32  # 通道分块处理（内存优化）
    )[0]  # 取第一个输出
    # 将维度从 [B, N, H, W, C] 转为 [B, N, C, H, W]
    rendered_image = rendered_image.permute(0, 1, 4, 2, 3)
    return rendered_image
