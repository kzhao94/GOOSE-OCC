import os


_base_ = './goose_mono_fair_3h.py'

dataset_root = os.environ.get(
    'GOOSE_OCC_ROOT', '/path/to/GOOSE-OCC')

# Control baseline for the 1/3/5/10-history ablation. All variants use only
# targets with at least ten real histories.
data = dict(
    train=dict(
        temporal_filter_ann_file=os.path.join(
            dataset_root, 'temporal', 'temporal_infos_train.pkl'),
        min_history=10,
    ),
    val=dict(
        temporal_filter_ann_file=os.path.join(
            dataset_root, 'temporal', 'temporal_infos_val.pkl'),
        min_history=10,
    ),
    test=dict(
        temporal_filter_ann_file=os.path.join(
            dataset_root, 'temporal', 'temporal_infos_val.pkl'),
        min_history=10,
    ),
)
