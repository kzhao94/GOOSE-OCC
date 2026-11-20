import os
import json
import time
import utils
import logging
import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from hooks import safe_multi_gpu_test, safe_single_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


def evaluate(dataset, results):
    metrics = dataset.evaluate(results, jsonfile_prefix=None)

    logging.info('--- Evaluation Results ---')
    for k, v in metrics.items():
        logging.info('%s: %.4f' % (k, v))

    return metrics


def _json_default(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)


def canonical_prediction_array(prediction, expected_shape):
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    prediction = np.asarray(prediction)
    while prediction.ndim > 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    expected_shape = tuple(int(value) for value in expected_shape)
    if prediction.shape != expected_shape:
        raise ValueError('Prediction dump shape {} != {}'.format(
            prediction.shape, expected_shape))
    return prediction.astype(np.uint8, copy=False)


def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--out-json', default=None,
                        help='Optional path to write machine-readable metrics.')
    parser.add_argument('--dump-predictions', default=None,
                        help='Optional directory for compressed prediction dumps.')
    parser.add_argument('--dump-count', type=int, default=0,
                        help='Number of validation predictions to dump.')
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if local_rank == 0:
        utils.init_logging(None, cfgs.debug)
    else:
        logging.root.disabled = True

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist.init_process_group('nccl', init_method='env://')

    seed = int(cfgs.get('seed', 0))
    logging.info('Setting random seed: %d', seed)
    set_random_seed(seed, deterministic=True)
    cudnn.benchmark = True

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False,
        seed=seed,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank], broadcast_buffers=False)
    else:
        model = MMDataParallel(model, [0])

    if os.path.isfile(args.weights):
        logging.info('Loading checkpoint from %s' % args.weights)
        load_checkpoint(
            model, args.weights, map_location='cuda', strict=True,
            logger=logging.Logger(__name__, logging.ERROR)
        )
    else:
        raise FileNotFoundError(args.weights)

    if torch.cuda.is_available():
        torch.cuda.synchronize(local_rank)
    start_time = time.time()
    if world_size > 1:
        results = safe_multi_gpu_test(model, val_loader)
    else:
        results = safe_single_gpu_test(model, val_loader)
    if torch.cuda.is_available():
        torch.cuda.synchronize(local_rank)
    elapsed_sec = time.time() - start_time

    if local_rank == 0:
        print(f"\n运行时间: {elapsed_sec:.4f} 秒")
        dumped_predictions = 0
        prediction_contract = None
        if args.dump_predictions is not None and args.dump_count > 0:
            dump_dir = Path(args.dump_predictions)
            dump_dir.mkdir(parents=True, exist_ok=True)
            count = min(args.dump_count, len(results))
            for index in range(count):
                raw_prediction = results[index]['occ_pred']
                expected_shape = tuple(getattr(
                    val_dataset, 'occ_size', raw_prediction.shape[-3:]))
                prediction = canonical_prediction_array(
                    raw_prediction, expected_shape)
                info = val_dataset.data_infos[index]
                np.savez_compressed(
                    dump_dir / 'prediction_{:04d}.npz'.format(index),
                    occ_pred=prediction,
                    axis_order=np.asarray('x,y,z'),
                    sample_index=np.asarray(index, dtype=np.int64),
                    sample_token=np.asarray(str(info['token'])),
                    sequence=np.asarray(str(info['sequence'])),
                    frame_index=np.asarray(
                        int(info['frame_index']), dtype=np.int64),
                )
            logging.info('Wrote %d prediction dumps to %s', count, dump_dir)
            dumped_predictions = count
            prediction_contract = dict(
                shape=list(expected_shape),
                axis_order='x,y,z',
                dtype='uint8',
                class_ids='0..7',
            )
        metrics = evaluate(val_dataset, results)
        if args.out_json is not None:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(
                config=args.config,
                weights=args.weights,
                override=args.override or {},
                seed=seed,
                batch_size=args.batch_size,
                world_size=world_size,
                elapsed_sec=elapsed_sec,
                num_samples=len(val_dataset),
                dumped_predictions=dumped_predictions,
                prediction_contract=prediction_contract,
                metrics=metrics,
            )
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2,
                          default=_json_default)
            logging.info('Wrote metrics JSON to %s', out_path)


if __name__ == '__main__':
    main()
