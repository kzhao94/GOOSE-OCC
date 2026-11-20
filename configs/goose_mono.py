import os


dataset_type = 'GOOSEOccMono'
dataset_root = os.environ.get(
    'GOOSE_OCC_ROOT', '/path/to/GOOSE-OCC')

point_cloud_range = [0.0, -16.0, -4.0, 50.0, 16.0, 4.0]
occ_size = [200, 128, 32]
voxel_size = 0.25

occ_class_names = [
    'free',
    'artificial_structures',
    'artificial_ground',
    'natural_ground',
    'obstacle',
    'vehicle',
    'vegetation',
    'human',
]

# The official 2D labels contain occupied semantics, not a 2D free class.
pseudo_2d_class_weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True,
)

_dim_ = 256
_num_points_ = 4
_num_groups_ = 4
_num_layers_ = 3
_num_frames_ = 1
_num_queries_ = 2000

model = dict(
    type='SparseGaussian',
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    free_class_idx=0,
    density_thresh=4e-2,
    voxelizer_cfg=dict(
        mode='dynamic',
        max_neighborhood=12,
        chunk_size=60000,
    ),
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32),
    ),
    use_mask_camera=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True,
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        num_outs=4,
    ),
    pts_bbox_head=dict(
        type='SparseGaussianHead',
        class_names=occ_class_names,
        embed_dims=_dim_,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        history_loss_weight=0.0,
        debug_vis=False,
        transformer=dict(
            type='SparseGaussianTransformer',
            embed_dims=_dim_,
            num_layers=_num_layers_,
            num_frames=_num_frames_,
            num_points=_num_points_,
            num_groups=_num_groups_,
            num_queries=_num_queries_,
            num_levels=4,
            num_classes=len(occ_class_names),
            pc_range=point_cloud_range,
            occ_size=occ_size,
            voxel_size=voxel_size,
            scale_range=[0.1, 2.0],
        ),
        loss_cfgs=dict(
            type='MultiLoss',
            num_classes=len(occ_class_names),
            class_weights=pseudo_2d_class_weights,
            depth_min=0.1,
            depth_max=point_cloud_range[3],
        ),
    ),
)

render_size = (256, 704)

train_pipeline = [
    dict(type='LoadGOOSEImageAndCalib', render_size=render_size),
    dict(type='LoadGOOSE2DSupervision'),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(
        type='Collect3D',
        keys=['img', 'gt_semantic', 'gt_semantic_weight', 'gt_depth'],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape',
            'lidar2img', 'img_timestamp', 'ego2lidar', 'lidar2ego',
            'cam2img', 'ego2cam', 'render_size', 'sample_idx',
            'sequence', 'frame_id',
        ),
    ),
]

test_pipeline = [
    dict(type='LoadGOOSEImageAndCalib', render_size=render_size),
    dict(type='LoadGOOSEOccGT', num_classes=len(occ_class_names)),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(
        type='Collect3D',
        keys=['img', 'voxel_semantics'],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape',
            'lidar2img', 'img_timestamp', 'ego2lidar', 'lidar2ego',
            'cam2img', 'ego2cam', 'render_size', 'sample_idx',
            'sequence', 'frame_id',
        ),
    ),
]

data = dict(
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=os.path.join(dataset_root, 'infos', 'goose_infos_train.pkl'),
        split='train',
        occ_size=occ_size,
        pc_range=point_cloud_range,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=os.path.join(dataset_root, 'infos', 'goose_infos_val.pkl'),
        split='val',
        occ_size=occ_size,
        pc_range=point_cloud_range,
        pipeline=test_pipeline,
        test_mode=True,
        evaluate_fov=True,
        evaluate_visibility=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=os.path.join(dataset_root, 'infos', 'goose_infos_val.pkl'),
        split='val',
        occ_size=occ_size,
        pc_range=point_cloud_range,
        pipeline=test_pipeline,
        test_mode=True,
        evaluate_fov=True,
        evaluate_visibility=True,
    ),
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    by_epoch=True,
    step=[22, 24],
    gamma=0.2,
)

total_epochs = 24
batch_size = 8
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]
resume_from = None
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', interval=100, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True),
    ],
)
eval_config = dict(interval=1)
debug = False
