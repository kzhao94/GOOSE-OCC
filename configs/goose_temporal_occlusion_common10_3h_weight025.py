_base_ = './goose_temporal_occlusion_common10_3h.py'
model = dict(pts_bbox_head=dict(occlusion_loss_weight=0.25))
