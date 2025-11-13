import os
import torch.distributed as dist

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")
if DEVICE_TYPE == "ascend":
    import torch_npu

from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from mimogpt.engine.utils.profiler_npu.profile_utils import (
    get_profile_fn,
    trace_handler,
    Nothing,
)
from mimogpt.engine.utils.train_loop import HookBase
from mimogpt.utils import hf_logger


class NPUTorchProfileHook(HookBase):
    def __init__(self, cfg):
        train_schedule, profile_fn = get_profile_fn(cfg)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,  # 如需修改profiling级别，可在此调整参数Leval0、Leval1、Leval2
            l2_cache=False,  # 此参数仅在Leval2时设置成True
        )
        profile_tb_logger = os.path.join(cfg.common.log_path, "ascend_profile_results")
        os.makedirs(profile_tb_logger, exist_ok=True)
        tb_logger_trace_handler = tensorboard_trace_handler(profile_tb_logger)

        if cfg.profile and DEVICE_TYPE == "ascend" and dist.get_rank() == 0:
            self.profiler = profile_fn(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                ],
                schedule=train_schedule,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,  # 专家参数默认级别Leval0，可根据需要设置不同Leval
                on_trace_ready=tb_logger_trace_handler,
            )
        else:
            self.profiler = Nothing()

    def before_train(self):
        self.profiler.__enter__()
        hf_logger.info(f"NPU Profiller start!!!!!")

    def after_step(self):
        self.profiler.step()

    def after_train(self):
        self.profiler.__exit__(None, None, None)
        hf_logger.info(f"NPU Profiller stop!!!!!")
