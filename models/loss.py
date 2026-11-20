import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import LovaszLoss
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class MultiLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 use_uncertainty_weights=True,
                 # 语义分割配置：标准配置
                 use_sem_ce=True, sem_ce_weight=1.0,
                 use_sem_lovasz=True, sem_lovasz_weight=0.5,
                 # 深度估计配置
                 use_depth_l1=True, depth_l1_weight=1.0,  # 主力 Loss
                 use_depth_si=True, depth_si_weight=0.2,  # 辅助正则化，防止过大
                 # 其他参数
                 ignore_index=255,
                 class_weights=None,
                 lambda_si=0.5,
                 depth_min=0.1,
                 depth_max=85.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.lambda_si = lambda_si
        self.depth_min = depth_min
        self.depth_max = depth_max

        # 开关
        self.use_sem_ce = use_sem_ce
        self.use_sem_lovasz = use_sem_lovasz
        self.use_depth_l1 = use_depth_l1
        self.use_depth_si = use_depth_si
        self.use_uncertainty = use_uncertainty_weights

        # 固定权重
        self.sem_ce_weight = sem_ce_weight
        self.sem_lovasz_weight = sem_lovasz_weight
        self.depth_l1_weight = depth_l1_weight
        self.depth_si_weight = depth_si_weight

        if class_weights is not None:
            self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)

        if self.use_uncertainty:
            self.log_var_sem = nn.Parameter(torch.zeros(1))
            self.log_var_depth = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('fixed_weight_sem', torch.tensor(1.0))
            self.register_buffer('fixed_weight_depth', torch.tensor(0.25))

        # 2. 定义具体的 Loss 函数
        if self.use_sem_lovasz:
            self.lovasz_loss_fn = LovaszLoss(
                mode='multiclass',
                ignore_index=ignore_index,
                from_logits=True
            )

    def compute_raw_losses(self, seg_pred, seg_target, depth_pred,
                           depth_target, semantic_weight=None,
                           include_lovasz=True):
        device = seg_pred.device
        loss_sem_raw = 0.0

        has_semantic_weight = semantic_weight is not None
        if has_semantic_weight:
            semantic_weight = semantic_weight.to(
                device=device, dtype=seg_pred.dtype)
            if semantic_weight.shape != seg_target.shape:
                raise ValueError('semantic_weight and seg_target shape mismatch')
            if torch.any((semantic_weight < 0) | (semantic_weight > 1)):
                raise ValueError('semantic_weight must be in [0, 1]')
            semantic_valid = (
                (seg_target != self.ignore_index) & (semantic_weight > 0))

        # 1.1 Cross Entropy
        if self.use_sem_ce:
            weight = self.class_weights.to(device) if self.class_weights is not None else None
            if has_semantic_weight:
                loss_ce_map = F.cross_entropy(
                    seg_pred, seg_target,
                    weight=weight,
                    ignore_index=self.ignore_index,
                    reduction='none',
                )
                if semantic_valid.any():
                    loss_ce = (
                        loss_ce_map[semantic_valid]
                        * semantic_weight[semantic_valid]
                    ).sum() / semantic_weight[semantic_valid].sum().clamp_min(1e-6)
                else:
                    loss_ce = seg_pred.sum() * 0.0
            else:
                loss_ce = F.cross_entropy(
                    seg_pred, seg_target,
                    weight=weight,
                    ignore_index=self.ignore_index,
                    reduction='mean',
                )
            loss_sem_raw += loss_ce * self.sem_ce_weight

        # 1.2 Lovasz-Softmax
        if self.use_sem_lovasz and include_lovasz:
            loss_lovasz = self.lovasz_loss_fn(seg_pred, seg_target)
            loss_sem_raw += loss_lovasz * self.sem_lovasz_weight

        # ==================== 2. Depth Losses ====================
        loss_depth_raw = 0.0
        depth_valid_mask = (depth_target > self.depth_min) & (depth_target < self.depth_max) & (seg_target != 0) & (seg_target != self.ignore_index)

        if depth_valid_mask.sum() == 0:
            loss_depth_raw = torch.tensor(0.0, device=device)
        else:
            pred_valid = depth_pred[depth_valid_mask]
            target_valid = depth_target[depth_valid_mask]

            # 2.1 L1 Loss
            if self.use_depth_l1:
                loss_l1 = F.l1_loss(
                    pred_valid,
                    target_valid,
                    reduction='mean'
                )
                loss_depth_raw += loss_l1 * self.depth_l1_weight

            # 2.2 Scale-Invariant Loss (SiLog)
            if self.use_depth_si:
                log_pred = torch.log(torch.clamp(pred_valid, min=1e-6))
                log_gt = torch.log(torch.clamp(target_valid, min=1e-6))
                diff = log_pred - log_gt
                silog = torch.sqrt((diff ** 2).mean() - self.lambda_si * (diff.mean() ** 2) + 1e-6)

                loss_depth_raw += silog * self.depth_si_weight

        return loss_sem_raw, loss_depth_raw

    def forward(self, seg_pred, seg_target, depth_pred, depth_target,
                semantic_weight=None):
        loss_dict = {}
        loss_sem_raw, loss_depth_raw = self.compute_raw_losses(
            seg_pred, seg_target, depth_pred, depth_target, semantic_weight)

        if self.use_uncertainty:
            precision_sem = torch.exp(-self.log_var_sem)
            precision_depth = torch.exp(-self.log_var_depth)
            loss_dict['learned_weight_sem'] = precision_sem.detach()
            loss_dict['learned_weight_depth'] = precision_depth.detach()
            loss_dict['loss_sem'] = precision_sem * loss_sem_raw + 0.5 * self.log_var_sem
            loss_dict['loss_depth'] = precision_depth * loss_depth_raw + 0.5 * self.log_var_depth
        else:
            loss_dict['loss_sem'] = loss_sem_raw
            loss_dict['loss_depth'] = loss_depth_raw

        return loss_dict
