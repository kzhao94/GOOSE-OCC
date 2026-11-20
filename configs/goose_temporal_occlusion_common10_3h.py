_base_ = './goose_temporal_common10_3h_binary_conf.py'

# Proposed method: focus an additional history semantic term on source-visible
# surfaces that pose/depth geometry places behind the target-visible surface.
# OCC GT and observation masks are not collected by the training pipeline.
model = dict(
    pts_bbox_head=dict(
        occlusion_loss_weight=0.5,
        occlusion_abs_margin=0.5,
        occlusion_rel_margin=0.05,
    ),
)
