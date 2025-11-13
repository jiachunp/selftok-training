# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import weakref
import os

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")
# if DEVICE_TYPE == "ascend":
#     import torch_npu

from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torch.profiler import profile, schedule
import torch.distributed as dist

# import torch_npu

__all__ = ["HookBase", "TrainerBase"]


class HookBase:
    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)

        self._hooks.extend(hooks)

    def register_model(self, model):
        model.trainer = weakref.proxy(self)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        # os.makedirs('/mnt/sfs_turbo/pkh/selftok/try160-1125', exist_ok=True)

        # 为每个进程设置唯一的输出目录
        rank = dist.get_rank()
        profiling_dir = "/home/jovyan/cache/pkh/selftok/try160-1125/"
        os.makedirs(profiling_dir, exist_ok=True)

        try:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
                # pro.step()
            # 可选项
            #print(pro.key_averages())
            #pro.export_chrome_trace("/home/ma-user/work/zmz/performance/trace.json")
        except Exception:
            logger.exception("Exception during training:")
            raise
        finally:
            self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError
