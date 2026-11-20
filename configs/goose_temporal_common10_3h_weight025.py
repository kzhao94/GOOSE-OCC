_base_ = './goose_temporal_common10_3h.py'

model = dict(
    pts_bbox_head=dict(history_loss_weight=0.25),
)
