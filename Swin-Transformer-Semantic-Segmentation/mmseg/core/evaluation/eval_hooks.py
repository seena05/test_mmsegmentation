import os.path as osp
import warnings
from math import inf

import mmcv
from mmcv.runner import Hook
from torch.utils.data import DataLoader

from mmseg.utils import get_root_logger

from clearml import Task, OutputModel


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = ['mIoU', 'mAP', 'AR']
    less_keys = ['loss']

    def __init__(self,
                 dataloader,
                 interval=1,
                 by_epoch=False,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        assert isinstance(save_best, str) or save_best is None
        self.save_best = save_best
        self.best_ckpt_path = None
        self.eval_kwargs = eval_kwargs

        self.logger = get_root_logger()
        
        if self.save_best is not None:
            self._init_rule(rule, self.save_best)

        self.task = Task.current_task()
        if self.task:
            self.output_model = OutputModel(task=self.task, framework='pytorch')
        else:
            self.output_model = None
        
    def _init_rule(self, rule, key_indicator):
        """Initialize rule, key_indicator, comparison_func, and best score.

        Args:
            rule (str | None): Comparison rule for best score.
            key_indicator (str | None): Key indicator to determine the
                comparison rule.
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, '
                           f'but got {rule}.')

        if rule is None:
            if key_indicator != 'auto':
                if any(key in key_indicator for key in self.greater_keys):
                    rule = 'greater'
                elif any(key in key_indicator for key in self.less_keys):
                    rule = 'less'
                else:
                    raise ValueError(f'Cannot infer the rule for key '
                                     f'{key_indicator}, thus a specific rule '
                                     f'must be specified.')
        self.rule = rule
        self.key_indicator = key_indicator
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def before_run(self, runner):
        if self.save_best is not None:
            if runner.meta is None:
                warnings.warn('runner.meta is None. Creating a empty one.')
                runner.meta = dict()
            runner.meta.setdefault('hook_msgs', dict())

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self.save_best_checkpoint(runner, key_score)
            if self.output_model and self.best_ckpt_path and osp.exists(self.best_ckpt_path):
                step = runner.epoch if self.by_epoch else runner.iter
                self.output_model.update_weights(weights_filename=self.best_ckpt_path, target_filename=f'best_{self.key_indicator}.pth', iteration=step)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self.save_best_checkpoint(runner, key_score)
            if self.output_model and self.best_ckpt_path and osp.exists(self.best_ckpt_path):
                step = runner.epoch if self.by_epoch else runner.iter
                self.output_model.update_weights(weights_filename=self.best_ckpt_path, target_filename=f'best_{self.key_indicator}.pth', iteration=step)

    def save_best_checkpoint(self, runner, key_score):
        best_score = runner.meta['hook_msgs'].get(
            'best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score
            print(runner.meta['hook_msgs'])
            if 'last_ckpt' in runner.meta['hook_msgs']:
                last_ckpt = runner.meta['hook_msgs']['last_ckpt']
                runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
                mmcv.symlink(
                    last_ckpt,
                    osp.join(runner.work_dir, f'best_{self.key_indicator}.pth'))
                self.best_ckpt_path = osp.join(runner.work_dir, f'best_{self.key_indicator}.pth')
                time_stamp = runner.epoch + 1 if self.by_epoch else runner.iter + 1
                self.logger.info(f'Now best checkpoint is epoch_{time_stamp}.pth.'
                                 f'Best {self.key_indicator} is {best_score:0.4f}')

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=False,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        super().__init__(
            dataloader,
            interval=interval,
            gpu_collect=gpu_collect,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            **eval_kwargs)
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.by_epoch = by_epoch
        self.eval_kwargs = eval_kwargs

        self.task = Task.current_task()
        if self.task:
            self.output_model = OutputModel(task=self.task, framework='pytorch')
        else:
            self.output_model = None
        
    def after_train_iter(self, runner):
        """After train epoch hook."""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)
            if self.save_best:
                self.save_best_checkpoint(runner, key_score)
                if self.output_model and self.best_ckpt_path and osp.exists(self.best_ckpt_path):
                    step = runner.epoch if self.by_epoch else runner.iter
                    self.output_model.update_weights(weights_filename=self.best_ckpt_path, target_filename=f'best_{self.key_indicator}.pth', iteration=step)

    def after_train_epoch(self, runner):
        """After train epoch hook."""
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)
            if self.save_best:
                self.save_best_checkpoint(runner, key_score)
                if self.output_model and self.best_ckpt_path and osp.exists(self.best_ckpt_path):
                    step = runner.epoch if self.by_epoch else runner.iter
                    self.output_model.update_weights(weights_filename=self.best_ckpt_path, target_filename=f'best_{self.key_indicator}.pth', iteration=step)
