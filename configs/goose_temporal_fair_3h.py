_base_ = './goose_temporal.py'

# Fair single-frame comparison protocol: use the same target subset and an
# effective global batch of 8 without increasing temporal GPU memory.
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
    train=dict(min_history=3),
    val=dict(min_history=3),
    test=dict(min_history=3),
)
