import os


_base_ = './goose_mono.py'

dataset_root = os.environ.get(
    'GOOSE_OCC_ROOT', '/path/to/GOOSE-OCC')

# Fair target+3-history comparison protocol. Both mono and temporal use a
# physical global batch of 2 with four-step accumulation (effective batch 8).
batch_size = 2
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=4,
    grad_clip=dict(max_norm=35, norm_type=2),
)
lr_config = dict(warmup_iters=2000)
eval_config = dict(
    interval=1,
    save_best='O-SSC-mIoU',
    rule='greater',
)

data = dict(
    train=dict(
        temporal_filter_ann_file=os.path.join(
            dataset_root, 'temporal', 'temporal_infos_train.pkl'),
        min_history=3,
    ),
    val=dict(
        temporal_filter_ann_file=os.path.join(
            dataset_root, 'temporal', 'temporal_infos_val.pkl'),
        min_history=3,
    ),
    test=dict(
        temporal_filter_ann_file=os.path.join(
            dataset_root, 'temporal', 'temporal_infos_val.pkl'),
        min_history=3,
    ),
)
