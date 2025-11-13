import pprint
import os
import torch.distributed as dist
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

from mimogpt.engine.utils.train_loop import HookBase
from mimogpt.utils import hf_logger


class Nothing(object):
    def __init__(self, *args, **kwargs):
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def start(self):
        return

    def stop(self):
        return

    def step(self):
        return

    def export_chrome_trace(self, path):
        return


class TorchProfileHook(HookBase):
    def __init__(self, cfg):
        is_root = dist.get_rank() == 0
        if is_root:
            skip_first = cfg.get("profile_skip_first", 0)
            wait = cfg.get("profile_wait", 1)
            warmup = cfg.get("profile_warmup", 1)
            active = cfg.get("profile_active", 1)
            repeat = cfg.get("profile_repeat", 0)

            profile_tb_logger = os.path.join(cfg.common.log_path, "gpu_profile_results")
            os.makedirs(profile_tb_logger, exist_ok=True)

            my_schedule = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first)
            tb_logger_trace_handler = tensorboard_trace_handler(profile_tb_logger)
            self.profiler = profile(
                schedule=my_schedule,
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                on_trace_ready=tb_logger_trace_handler,
            )
            hf_logger.info("Profile enabled in rank 0")
        else:
            self.profiler = Nothing()

    def before_train(self):
        self.profiler.start()
        hf_logger.info(f"Profiller start!!!!!")

    def after_step(self):
        self.profiler.step()

    def after_train(self):
        self.profiler.stop()
        hf_logger.info(f"Profiller stop!!!!!")
