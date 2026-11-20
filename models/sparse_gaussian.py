import torch
import queue
import numpy as np
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import pad_multiple, GpuPhotoMetricDistortion
from .gaussian_voxelizer import GaussianVoxelizer
from .gaussian_utils import decode_gaussian

@DETECTORS.register_module()
class SparseGaussian(MVXTwoStageDetector):
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 data_aug=None,
                 use_mask_camera=False,
                 pc_range = None,
                 voxel_size = None,
                 free_class_idx=17,  # ✅ 新增参数，默认17兼容nuScenes
                 density_thresh=4e-2,  # ✅ 顺带把阈值也参数化
                 voxelizer_cfg=None,
                 **kwargs):

        super(SparseGaussian, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        self.use_mask_camera = use_mask_camera
        self.fp16_enabled = False
        self.data_aug = data_aug
        self.color_aug = GpuPhotoMetricDistortion()

        self.memory = {}
        self.queue = queue.Queue()

        self.vol_range = pc_range
        self.voxel_size = voxel_size

        self.free_class_idx = free_class_idx
        self.density_thresh = density_thresh

        voxelizer_cfg = voxelizer_cfg or {}
        self.voxelizer_cfg = voxelizer_cfg
        self.voxelizer = GaussianVoxelizer(
            self.vol_range,
            self.voxel_size,
            **voxelizer_cfg).cuda()

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None):
        """Extract features from images and points."""
        if len(img.shape) == 6:
            img = img.flatten(1, 2)  # [B, TN, C, H, W]

        B, N, C, H, W = img.size()  # [8, 4, 3, 256, 704]
        img = img.view(B * N, C, H, W)
        img = img.float()

        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])
                H, W = img.shape[-2:]

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward_pts_train(self, mlvl_feats, gt_semantic, gt_depth, img,
                          img_metas, gt_semantic_weight=None):
        """
        voxel_semantics: [bs, 200, 200, 16], value in range [0, num_cls - 1]
        voxel_instances: [bs, 200, 200, 16], value in range [0, num_obj - 1]
        instance_class_ids: [[bs0_num_obj], [bs1_num_obj], ...], value in range [0, num_cls - 1]
        """
        outs = self.pts_bbox_head(mlvl_feats, img_metas)
        loss_inputs = [
            gt_semantic, gt_depth, outs, img_metas, gt_semantic_weight]
        return self.pts_bbox_head.loss(*loss_inputs)

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @force_fp32(apply_to=('img'))
    def forward_train(self, img_metas=None, img=None, gt_semantic=None,
                      gt_depth=None, gt_semantic_weight=None, **kwargs):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        return self.forward_pts_train(
            img_feats, gt_semantic, gt_depth, img, img_metas,
            gt_semantic_weight)

    def forward_test(self, img_metas, img=None, **kwargs):
        outputs = self.simple_test(img_metas, img)

        gaussian_preds = outputs['gaussian_preds'][-1]
        gaussians = gaussian_preds[0]

        means, scales, quats, opacities = decode_gaussian(gaussians)
        features = gaussian_preds[1]

        density, grid_feats = self.voxelizer(
            means3d=means,  # 3D位置
            scales=scales,
            rotations=quats,
            opacities=opacities,  # 不透明度
            features=features.softmax(-1),  # 特征概率
        )

        preds = grid_feats.argmax(-1)

        # ✅ 用参数控制 free 类别
        occ_preds = torch.where(density.squeeze(-1) > self.density_thresh, preds, self.free_class_idx)

        return [{'occ_pred': occ_preds}]


    def simple_test_pts(self, img_feats, img, img_metas, rescale=False):
        outs = self.pts_bbox_head(img_feats, img_metas)
        # outs = self.pts_bbox_head.merge_occ_pred(outs)
        return outs

    def simple_test(self, img_metas, img=None, rescale=False):
        world_size = get_dist_info()[1]
        if world_size == 1:  # online
            return self.simple_test_online(img_metas, img, rescale)
        else:  # offline
            return self.simple_test_offline(img_metas, img, rescale)

    def simple_test_offline(self, img_metas, img=None, rescale=False):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        return self.simple_test_pts(img_feats, img, img_metas, rescale=rescale)

    def simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape

        # ==========================================
        # 1. 明确视角数和帧数 (单目核心修改)
        # ==========================================
        N_views = 1  # 单目前视相机，视角死死定为 1
        T = N // N_views  # T 就是时序帧数

        # 保持和原版一样的 reshape 逻辑: [B, T, N_views, C, H, W]
        img = img.reshape(B, T, N_views, C, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = T  # 帧数以实际传入的图像张数为准

        # ==========================================
        # 2. 对齐 Meta 长度，杜绝越界
        # ==========================================
        img_shape = (H, W, C)
        total_images = num_frames * N_views
        # 这三个键强制覆盖，保证长度 = N，后面切片绝不会越界
        img_metas[0]['img_shape'] = [img_shape for _ in range(total_images)]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(total_images)]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(total_images)]

        img_feats_list, img_metas_list = [], []

        # ==========================================
        # 3. 逐帧提取特征 (完全对齐原版逻辑)
        # ==========================================
        for i in range(num_frames):
            # 单目：每次只取当前帧的 1 个索引
            img_indices = list(np.arange(i * N_views, (i + 1) * N_views))

            img_metas_curr = [{}]
            for k, v in img_metas[0].items():
                if isinstance(v, list):
                    # 正常的 list 属性 (如 lidar2img, filename)，按索引切片
                    # 因为上面保证了长度对齐，这里绝对不会越界
                    img_metas_curr[0][k] = [v[idx] for idx in img_indices]
                else:
                    # ✅ 致命修复：非 list 属性 (如 render_size, token, occ_size)
                    # 必须原样复制下去，否则后续初始化会 KeyError！
                    img_metas_curr[0][k] = v

            curr_filename = img_filenames[img_indices[0]]

            if curr_filename in self.memory:
                # found in memory
                img_feats_curr = self.memory[curr_filename]
            else:
                # extract feature and put into memory
                # img[:, i] 形状为 [B, N_views, C, H, W]
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                self.memory[curr_filename] = img_feats_curr
                self.queue.put(curr_filename)
                while self.queue.qsize() > 16:  # avoid OOM
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        # ==========================================
        # 4. 重组特征和 Meta (原版逻辑，一字不改)
        # ==========================================
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # run detector
        return self.simple_test_pts(img_feats, img, img_metas, rescale=rescale)
