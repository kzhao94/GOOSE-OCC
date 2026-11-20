#!/usr/bin/env python3
"""Assert sample and optimization parity across GOOSE experiment configs."""

import sys
from pathlib import Path

from mmcv import Config
from mmdet3d.datasets import build_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import loaders  # noqa: E402,F401 - registry side effects


MAIN_CONFIGS = (
    'configs/goose_mono_fair_3h.py',
    'configs/goose_temporal_fair_3h.py',
)
COMMON_CONFIGS = (
    'configs/goose_mono_common10.py',
    'configs/goose_temporal_common10_1h.py',
    'configs/goose_temporal_common10_3h.py',
    'configs/goose_temporal_common10_5h.py',
    'configs/goose_temporal_common10_10h.py',
    'configs/goose_temporal_common10_3h_binary_conf.py',
    'configs/goose_temporal_common10_3h_weight025.py',
    'configs/goose_temporal_common10_3h_weight100.py',
    'configs/goose_temporal_occlusion_common10_3h_weight025.py',
    'configs/goose_temporal_occlusion_common10_3h.py',
    'configs/goose_temporal_occlusion_common10_3h_weight100.py',
)
OCCLUSION_WEIGHTS = {
    'configs/goose_temporal_occlusion_common10_3h_weight025.py': 0.25,
    'configs/goose_temporal_occlusion_common10_3h.py': 0.5,
    'configs/goose_temporal_occlusion_common10_3h_weight100.py': 1.0,
}


def sample_key(info):
    return (
        info['sequence'], int(info['frame_index']),
        int(info['sensor_stamp_ns']))


def dataset_keys(dataset):
    return tuple(sample_key(info) for info in dataset.data_infos)


def optimization_signature(cfg):
    return (
        cfg.batch_size,
        cfg.optimizer.to_dict(),
        cfg.optimizer_config.type,
        cfg.optimizer_config.cumulative_iters,
        cfg.optimizer_config.grad_clip.to_dict(),
        cfg.lr_config.to_dict(),
        cfg.total_epochs,
        cfg.get('seed', 0),
    )


def audit_occlusion_config(path, cfg):
    if path not in OCCLUSION_WEIGHTS:
        return
    head = cfg.model.pts_bbox_head
    assert head.history_loss_weight == 0.75, path
    assert head.history_semantic_ce_only is True, path
    assert head.occlusion_loss_weight == OCCLUSION_WEIGHTS[path], path
    assert head.occlusion_abs_margin == 0.5, path
    assert head.occlusion_rel_margin == 0.05, path
    assert cfg.data.train.num_history == 3, path
    assert cfg.data.train.min_history == 10, path
    supervision = [
        transform for transform in cfg.data.train.pipeline
        if transform.type == 'LoadGOOSE2DSupervision'
    ]
    assert len(supervision) == 1, path
    assert supervision[0].confidence_mode == 'binary', path


def audit_group(paths, expected_train, expected_val, expected_updates):
    reference_train_keys = None
    reference_val_keys = None
    reference_optimization = None
    for path in paths:
        cfg = Config.fromfile(path)
        train = build_dataset(cfg.data.train)
        val = build_dataset(cfg.data.val)
        assert (len(train), len(val)) == (expected_train, expected_val), path
        train_keys = dataset_keys(train)
        val_keys = dataset_keys(val)
        optimization = optimization_signature(cfg)
        if reference_train_keys is None:
            reference_train_keys = train_keys
            reference_val_keys = val_keys
            reference_optimization = optimization
        else:
            assert train_keys == reference_train_keys, path
            assert val_keys == reference_val_keys, path
            assert optimization == reference_optimization, path

        assert cfg.batch_size == 2
        assert cfg.optimizer_config.cumulative_iters == 4
        assert cfg.lr_config.warmup_iters == 2000
        assert cfg.total_epochs == 24
        audit_occlusion_config(path, cfg)
        assert cfg.batch_size % 2 == 0
        micro_iterations = (len(train) // cfg.batch_size) * cfg.total_epochs
        assert len(train) % cfg.batch_size == 0
        assert micro_iterations % cfg.optimizer_config.cumulative_iters == 0
        optimizer_updates = (
            micro_iterations // cfg.optimizer_config.cumulative_iters)
        assert optimizer_updates == expected_updates, path

        runtime = train.get_data_info(0)
        assert 'observation_path' not in runtime
        assert 'occ_path' not in runtime
        assert 'source_stamps_ns' not in runtime
        assert 'sweeps' not in runtime
        print('{}: train/val={}/{}, micro={}, updates={}'.format(
            path, len(train), len(val), micro_iterations, optimizer_updates))


def audit_validation_seed_contract():
    source = (REPO_ROOT / 'val.py').read_text(encoding='utf-8')
    assert "seed = int(cfgs.get('seed', 0))" in source
    assert 'set_random_seed(seed, deterministic=True)' in source
    assert 'seed=seed,' in source

    for relative_path in (
            'tools/run_goose_fair_experiments.sh',
            'tools/run_goose_final_seed_repeats.sh',
            'tools/run_goose_occlusion_repeats.sh'):
        script = (REPO_ROOT / relative_path).read_text(encoding='utf-8')
        assert '--override "seed=' in script, relative_path


def main():
    audit_validation_seed_contract()
    audit_group(MAIN_CONFIGS, 2812, 482, 8436)
    audit_group(COMMON_CONFIGS, 2726, 469, 8178)
    print('GOOSE experiment protocol parity passed')


if __name__ == '__main__':
    main()
