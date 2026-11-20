import os


_base_ = './goose_mono.py'

dataset_root = os.environ.get(
    'GOOSE_OCC_ROOT', '/path/to/GOOSE-OCC')

# Single-frame model and supervision, restricted to exactly the targets used
# by the default target+3-history temporal experiment.
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
