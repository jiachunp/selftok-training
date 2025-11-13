# -*- coding: utf-8 -*-

import os
import time
import torch
import psutil
import datetime
import torch.distributed as dist

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print("# Ascend didn't support SummaryWriter, skipped")
from .train_loop import HookBase
from mimogpt.utils import hf_logger


def get_cpu_mem_status():
    mem = psutil.virtual_memory()
    tot_mem = float(mem.total) / (1024**3)
    used_mem = float(mem.used) / (1024**3)
    return "[{:.1f}G/{:.1f}G]".format(used_mem, tot_mem)


class TimerAndLogger(HookBase):
    def __init__(self, cfg, metrics=None):
        self.cfg = cfg
        self.metrics = metrics
        self.is_root = dist.get_rank() == 0

    def before_train(self):
        self._train_time = 0
        self._train_time_log_interval = 0
        self._train_time_epoch = 0

    def after_train(self):
        if self.is_root:
            hf_logger.info(
                "Finish training, total time: {}".format(str(datetime.timedelta(seconds=int(self._train_time))))
            )

    def before_step(self):
        self._start_time = time.perf_counter()

    def after_step(self):
        if self.is_root:
            step_time = time.perf_counter() - self._start_time
            self._train_time += step_time
            self._train_time_log_interval += step_time
            self._train_time_epoch += step_time
            current_epoch = self.trainer.iter // self.cfg.dataloader_len
            current_iter = self.trainer.iter % self.cfg.dataloader_len
            if (current_iter + 1) % self.cfg.common.log_interval == 0:
                if self.cfg.dataloader.train.hybrid:
                    frame_num = (
                        (self.cfg.dataloader.train.batch_size.image + self.cfg.dataloader.train.batch_size.video * 8)
                        * self.cfg.world_size
                        * self.cfg.common.log_interval
                        // 2
                    )
                else:
                    frame_num = (
                        (self.cfg.dataloader.train.batch_size.video * 8)
                        * self.cfg.world_size
                        * self.cfg.common.log_interval
                    )
                FPS = int(frame_num / self._train_time_log_interval)
                metric_value = self.metrics(self.trainer._outs)
                current_ime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                hf_logger.info(
                    "{} training: [{}][{}/{}] iterations, loss: {:.8f}, {}: {:.8f}, lr: {:.8f}, lr_tune: {:.8f}, speed: {:.3f}s/it, scaler: {}, {} FPS".format(
                        current_ime,
                        current_epoch,
                        current_iter + 1,
                        self.cfg.dataloader_len,
                        self.trainer._loss.item(),
                        self.metrics.name,
                        round(metric_value.item() * 100, 2),
                        self.trainer.optimizer.param_groups[0]["lr"],
                        self.trainer.optimizer.param_groups[-1]["lr"],
                        self._train_time_log_interval / self.cfg.common.log_interval,
                        self.trainer._scaler.get_scale(),
                        FPS,
                    )
                )
                self._train_time_log_interval = 0
            if (current_iter + 1) == self.cfg.dataloader_len:
                frame_num = (
                    (self.cfg.dataloader.train.batch_size.image + self.cfg.dataloader.train.batch_size.video * 8)
                    * self.cfg.world_size
                    * self.cfg.common.log_interval
                    // 2
                )
                FPS = int(frame_num / self._train_time_epoch)
                mem_reserved = "{:.1f}G".format(torch.cuda.memory_reserved() / (1024**3))
                hf_logger.info(
                    "training: epoch: {}, mean speed: {:.3f}s/it, mean FPS: {}, gpu_mem: {}, cpu_mem: {}".format(
                        current_epoch,
                        self._train_time_epoch / self.cfg.dataloader_len,
                        FPS,
                        mem_reserved,
                        get_cpu_mem_status(),
                    )
                )
                self._train_time_epoch = 0


class UniversalMeterLogger(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg

    def before_train(self):
        self._train_time = 0

    def after_train(self):
        hf_logger.info("Finish training, total time: {}".format(str(datetime.timedelta(seconds=int(self._train_time)))))

    def before_step(self):
        self._start_time = time.perf_counter()

    def after_step(self):
        step_time = time.perf_counter() - self._start_time
        self._train_time += step_time
        self.trainer.meters.batch_time.update(step_time)
        current_iter = self.trainer.iter % self.cfg.dataloader_len
        current_epoch = self.trainer.iter // self.cfg.dataloader_len
        if self.cfg.common.task == "mimo_vqgan":
            lr_d = self.trainer.optimizer_d.param_groups[0]["lr"]
            lr_g = self.trainer.optimizer_g.param_groups[0]["lr"]
        if (current_iter + 1) % self.cfg.common.log_interval == 0:
            data_cnt = (
                self.cfg.dataloader.train.batch_size
                * self.trainer.dist.world_size
                * self.trainer.gradient_accumulation_steps
            )
            fps = data_cnt / step_time
            meters = self.trainer.meters
            totals = self.trainer.totals
            total_iters = totals.total_iters
            remain_secs = (total_iters - self.trainer.iter) * meters.batch_time.avg
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
            time_str = f"\tRemainingTime {remain_time} ({finish_time})"
            msg = f"Iter: [{current_iter+1}/{totals.iter_per_epoch}]"
            msg += f"\tEpoch: [{current_epoch}/{totals.epochs} ({totals.total_iters})]"
            for k in meters.keys():
                msg += f"\t{k} {meters[k].get_val_str()} ({meters[k].get_avg_str()})"
            if self.cfg.common.task == "mimo_vqgan":
                msg += f"\tlr_d: {lr_d} \tlr_g: {lr_g}"
            msg += time_str
            msg += f"\tFPS {fps: .3f}"
            hf_logger.info(msg)


class VideoMeterLogger(UniversalMeterLogger):
    def after_step(self):
        step_time = time.perf_counter() - self._start_time
        self._train_time += step_time
        self.trainer.meters.batch_time.update(step_time)
        current_iter = self.trainer.iter % self.cfg.dataloader_len
        current_epoch = self.trainer.iter // self.cfg.dataloader_len
        if (current_iter + 1) % self.cfg.common.log_interval == 0:
            data_cnt = (
                self.cfg.dataloader.train.batch_size
                * self.cfg.dataloader.train.video_length
                * self.trainer.dist.world_size
                * self.trainer.gradient_accumulation_steps
            )
            fps = data_cnt / step_time
            meters = self.trainer.meters
            totals = self.trainer.totals
            total_iters = totals.total_iters
            remain_secs = (total_iters - self.trainer.iter) * meters.batch_time.avg
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
            time_str = f"\tRemainingTime {remain_time} ({finish_time})"
            msg = f"Iter: [{current_iter+1}/{totals.iter_per_epoch}]"
            msg += f"\tEpoch: [{current_epoch}/{totals.epochs} ({totals.total_iters})]"
            for k in meters.keys():
                msg += f"\t{k} {meters[k].get_val_str()} ({meters[k].get_avg_str()})"
            msg += time_str
            msg += f"\tFPS {fps: .3f}"
            hf_logger.info(msg)


class SimpleLogger(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_fps = 0
        self.is_root = dist.get_rank() == 0

    def before_train(self):
        self._train_time = 0
        self._train_time_log_interval = 0
        self._train_time_epoch = 0
        self._avg_loss = {}
        self._avg_loss_for_last_metric = {}

    def after_train(self):
        if self.is_root:
            hf_logger.info(
                "Finish training, total time: {}".format(str(datetime.timedelta(seconds=int(self._train_time))))
            )
            CI_Metric = f"[Last metrics]: "
            for k, v in self._avg_loss_for_last_metric.items():
                CI_Metric += f"{k}: {v[1] / v[0]:.3f}, "
            CI_Metric += f"FPS: {self.last_fps:.6f}"
            hf_logger.info(CI_Metric)

    def before_step(self):
        self._start_time = time.perf_counter()

    def after_step(self):
        if self.is_root:
            step_time = time.perf_counter() - self._start_time
            self._train_time += step_time
            self._train_time_log_interval += step_time
            self._train_time_epoch += step_time
            current_epoch = self.trainer.iter // self.cfg.dataloader_len
            current_iter = self.trainer.iter % self.cfg.dataloader_len

            for k, v in self.trainer._outs.items():
                if isinstance(v, torch.Tensor):
                    v = v.mean().item()
                if k not in self._avg_loss.keys():
                    self._avg_loss[k] = [0, 0.0]
                self._avg_loss[k][0] += 1
                self._avg_loss[k][1] += v

            if (current_iter + 1) % self.cfg.common.log_interval == 0:
                data_cnt = (
                    self.cfg.dataloader.train.batch_size * self.trainer.dist.world_size * self.cfg.common.log_interval
                )
                fps = data_cnt / self._train_time_log_interval
                self.last_fps = fps
                logging_info = "training: [{}][{}/{}] iterations, loss: {:.3f}, ".format(
                    current_epoch,
                    current_iter + 1,
                    self.cfg.dataloader_len,
                    self.trainer._loss.item(),
                )

                for k, v in self._avg_loss.items():
                    cur_loss = self.trainer._outs[k]
                    if isinstance(cur_loss, torch.Tensor):
                        cur_loss = cur_loss.mean().item()
                    logging_info += f"{k}: {cur_loss:.3f}({v[1] / v[0]:.3f}), "

                logging_info += "lr: {:.8f}, speed: {:.3f}s/it, FPS: {:.3f}, scaler: {}".format(
                    self.trainer.optimizer.param_groups[0]["lr"],
                    self._train_time_log_interval / self.cfg.common.log_interval,
                    fps,
                    self.trainer._scaler.get_scale(),
                )

                hf_logger.info(logging_info)
                self._train_time_log_interval = 0
                self._avg_loss_for_last_metric = self._avg_loss
                self._avg_loss = {}

            if (current_iter + 1) == self.cfg.dataloader_len:
                mem_reserved = "{:.1f}G".format(torch.cuda.memory_reserved() / (1024**3))
                hf_logger.info(
                    "training: epoch: {}, mean speed: {:.3f}s/it, gpu_mem: {}, cpu_mem: {}".format(
                        current_epoch,
                        self._train_time_epoch / self.cfg.dataloader_len,
                        mem_reserved,
                        get_cpu_mem_status(),
                    )
                )
                self._train_time_epoch = 0


class TensorboardLogger(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_root = dist.get_rank() == 0
        self.log_root_dir = cfg.common.log_path
        if self.is_root:
            event_path = os.path.join(self.log_root_dir, "events")
            os.makedirs(event_path, exist_ok=True)
            self.tb_logger = SummaryWriter(event_path, max_queue=1, flush_secs=10)

    def after_step(self):
        if self.is_root:
            current_iter = self.trainer.iter
            if (current_iter + 1) % self.cfg.common.log_interval == 0:
                name_value_dict = {}
                meters = self.trainer.meters
                for k in meters.keys():
                    self.tb_logger.add_scalar(k, meters[k].avg, current_iter)

    def after_train(self):
        if self.is_root:
            self.tb_logger.close()


class TensorboardPriorLogger(TensorboardLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.eval_interval = cfg.evaluation.validation.eval_interval
        self.test_interval = getattr(cfg.evaluation.test, "test_interval", -1)

    def after_step(self):
        super().after_step()
        if self.is_root:
            current_iter = self.trainer.iter
            if current_iter % self.eval_interval == int(self.eval_interval - 1):
                name_value_dict = {}
                meters = self.trainer.eval_board
                for k in meters.keys():
                    self.tb_logger.add_scalar(k, meters[k], current_iter)
            if current_iter % self.test_interval == int(self.test_interval - 1):
                name_value_dict = {}
                meters = self.trainer.test_board
                for k in meters.keys():
                    self.tb_logger.add_scalar(k, meters[k], current_iter)

    def after_train(self):
        if self.is_root:
            self.tb_logger.close()
