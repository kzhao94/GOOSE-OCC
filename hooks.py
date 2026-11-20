# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import torch
import mmcv
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu
import os.path as osp


def _to_numpy(obj):
    """递归地把结果里的所有 CUDA Tensor 转成 CPU numpy，并将 occ_pred 转为 uint8"""
    if isinstance(obj, torch.Tensor):
        arr = obj.cpu().numpy()
        if arr.dtype in (np.float32, np.float64, np.int32, np.int64):
            if arr.max() <= 255 and arr.min() >= 0:
                arr = arr.astype(np.uint8)
        return arr
    elif isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_numpy(v) for v in obj)
    else:
        return obj


@torch.no_grad()
def safe_single_gpu_test(model, data_loader):
    """单卡测试，每步强制把结果迁到 CPU numpy，避免显存累积"""
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        result = model(return_loss=False, rescale=True, **data)
        result = _to_numpy(result)
        if isinstance(result, list):
            results.extend(result)
            batch_size = len(result)
        else:
            results.append(result)
            batch_size = 1
        for _ in range(batch_size):
            prog_bar.update()
    return results


@torch.no_grad()
def safe_multi_gpu_test(model, data_loader, tmpdir=None):
    """多卡测试，每步强制把结果迁到 CPU numpy，避免 pickle 反序列化时恢复 CUDA Tensor"""
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        result = model(return_loss=False, rescale=True, **data)
        result = _to_numpy(result)
        if isinstance(result, list):
            results.extend(result)
            batch_size = len(result)
        else:
            results.append(result)
            batch_size = 1
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    return results


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class EvalHook(BaseEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        results = safe_single_gpu_test(runner.model, self.dataloader)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.
class DistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        results = safe_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=None)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
