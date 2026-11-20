import time
import json
import utils
import logging
import argparse
import importlib
import statistics
import os
from pathlib import Path

import torch
import torch.distributed
import torch.backends.cudnn as cudnn
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


def _json_default(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)


def _strip_evaluation_gt(cfgs):
    """Keep the validation image pipeline but remove evaluation-only OCC GT."""
    pipeline = []
    for stage in cfgs.data.val.pipeline:
        stage = dict(stage)
        if stage.get('type') == 'LoadGOOSEOccGT':
            continue
        if stage.get('type') == 'Collect3D':
            stage['keys'] = [
                key for key in stage.get('keys', [])
                if key != 'voxel_semantics']
        pipeline.append(stage)
    cfgs.data.val.pipeline = pipeline


def _percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main():
    parser = argparse.ArgumentParser(description='Time detector inference')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--num_warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--out-json', default=None,
                        help='Optional path to write machine-readable timing results.')
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)
    if str(cfgs.data.val.type).startswith('GOOSE'):
        _strip_evaluation_gt(cfgs)

    # register custom module
    if str(cfgs.data.val.type).startswith('GOOSE'):
        os.environ['VISIBILITYOCC_GOOSE_ONLY'] = '1'
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)
    utils.init_logging(None, cfgs.debug)

    # you need GPUs
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))
    torch.cuda.set_device(0)

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()

    assert torch.cuda.device_count() == 1
    model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.weights)
    load_checkpoint(
        model, args.weights, map_location='cuda', strict=False,
        logger=logging.Logger(__name__, logging.ERROR)
    )
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)
    checkpoint_size_bytes = Path(args.weights).stat().st_size

    print('Timing w/ data loading:')
    pure_inf_time = 0.0
    data_time = 0.0
    measured = 0
    elapsed_values = []
    data_values = []
    end_to_end_values = []
    iterator = iter(val_loader)
    baseline_allocated = None
    with torch.no_grad():
        for i in range(args.samples):
            load_start = time.perf_counter()
            try:
                data = next(iterator)
            except StopIteration:
                break
            loaded = time.perf_counter() - load_start
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= args.num_warmup:
                if baseline_allocated is None:
                    baseline_allocated = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                pure_inf_time += elapsed
                data_time += loaded
                measured += 1
                elapsed_values.append(elapsed)
                data_values.append(loaded)
                end_to_end_values.append(loaded + elapsed)
                if (i + 1) % args.log_interval == 0:
                    fps = measured / pure_inf_time
                    print(f'Done sample [{i + 1:<3}/ {args.samples}], '
                        f'fps: {fps:.1f} sample / s')

    fps = measured / pure_inf_time if pure_inf_time > 0 else 0.0
    end_to_end_time = pure_inf_time + data_time
    end_to_end_fps = measured / end_to_end_time if end_to_end_time else 0.0
    print(f'Overall fps: {fps:.4f} sample / s over {measured} measured samples')
    print(f'End-to-end fps: {end_to_end_fps:.4f} sample / s '
          f'(data loading + model)')

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            config=args.config,
            weights=args.weights,
            override=args.override or {},
            num_warmup=args.num_warmup,
            samples=args.samples,
            measured_samples=measured,
            total_measured_sec=pure_inf_time,
            mean_time_sec=(pure_inf_time / measured) if measured else None,
            fps=fps,
            data_loading_total_sec=data_time,
            data_loading_mean_sec=(data_time / measured) if measured else None,
            end_to_end_total_sec=end_to_end_time,
            end_to_end_mean_sec=(end_to_end_time / measured)
            if measured else None,
            end_to_end_fps=end_to_end_fps,
            forward_p50_sec=_percentile(elapsed_values, 0.50),
            forward_p95_sec=_percentile(elapsed_values, 0.95),
            end_to_end_p50_sec=_percentile(end_to_end_values, 0.50),
            end_to_end_p95_sec=_percentile(end_to_end_values, 0.95),
            forward_std_sec=(statistics.stdev(elapsed_values)
                             if len(elapsed_values) > 1 else 0.0),
            parameters=parameter_count,
            trainable_parameters=trainable_parameter_count,
            checkpoint_size_bytes=checkpoint_size_bytes,
            gpu_name=torch.cuda.get_device_name(0),
            persistent_allocated_mb=(baseline_allocated / 1024 ** 2)
            if baseline_allocated is not None else None,
            peak_allocated_mb=(torch.cuda.max_memory_allocated() / 1024 ** 2),
            peak_reserved_mb=(torch.cuda.max_memory_reserved() / 1024 ** 2),
            evaluation_gt_stripped=True,
            times_sec=elapsed_values,
            data_times_sec=data_values,
            end_to_end_times_sec=end_to_end_values,
        )
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2,
                      default=_json_default)
        logging.info('Wrote timing JSON to %s', out_path)


if __name__ == '__main__':
    main()
