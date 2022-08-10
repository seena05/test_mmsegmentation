import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm

from clearml import Task, OutputModel


class EvalHook(BaseEvalHook):

    def __init__(self, *args, **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)

        self.task = Task.current_task()
        if self.task:
            self.output_model = OutputModel(task=self.task, framework='pytorch')
        else:
            self.output_model = None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)
            if self.output_model and self.best_ckpt_path and self.file_client.isfile(self.best_ckpt_path):
                step = runner.epoch if self.by_epoch else runner.iter
                self.output_model.update_weights(weights_filename=self.best_ckpt_path, target_filename=f'best_{self.key_indicator}.pth', iteration=step)


class DistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)

        self.task = Task.current_task()
        if self.task:
            self.output_model = OutputModel(task=self.task, framework='pytorch')
        else:
            self.output_model = None

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

        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
                if self.output_model and self.best_ckpt_path and self.file_client.isfile(self.best_ckpt_path):
                    step = runner.epoch if self.by_epoch else runner.iter
                    self.output_model.update_weights(weights_filename=self.best_ckpt_path, target_filename=f'best_{self.key_indicator}.pth', iteration=step)
