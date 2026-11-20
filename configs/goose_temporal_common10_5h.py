_base_ = './goose_temporal_fair_3h.py'

model = dict(
    pts_bbox_head=dict(
        transformer=dict(num_frames=6),
    ),
)
data = dict(
    train=dict(num_history=5, min_history=10),
    val=dict(num_history=5, min_history=10),
    test=dict(num_history=5, min_history=10),
)
