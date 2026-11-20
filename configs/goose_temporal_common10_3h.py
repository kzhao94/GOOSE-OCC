_base_ = './goose_temporal_fair_3h.py'

data = dict(
    train=dict(num_history=3, min_history=10),
    val=dict(num_history=3, min_history=10),
    test=dict(num_history=3, min_history=10),
)
