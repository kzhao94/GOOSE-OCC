_base_ = './goose_temporal_common10_3h.py'

# Ablation: retain the confidence>0 validity mask but remove confidence
# magnitudes from the history semantic loss.
train_pipeline = [
    dict(type='LoadGOOSEImageAndCalib', render_size=(256, 704)),
    dict(type='LoadGOOSE2DSupervision', confidence_mode='binary'),
    dict(type='DefaultFormatBundle3D', class_names=[
        'free',
        'artificial_structures',
        'artificial_ground',
        'natural_ground',
        'obstacle',
        'vehicle',
        'vegetation',
        'human',
    ]),
    dict(
        type='Collect3D',
        keys=['img', 'gt_semantic', 'gt_semantic_weight', 'gt_depth'],
        meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape',
            'lidar2img', 'img_timestamp', 'ego2lidar', 'lidar2ego',
            'cam2img', 'ego2cam', 'render_size', 'sample_idx',
            'sequence', 'frame_id', 'frame_is_history', 'history_ranks',
        ),
    ),
]

data = dict(train=dict(pipeline=train_pipeline))
