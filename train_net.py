# -*- coding: utf-8 -*-

import os
# os.environ["CUDA_HOME"] = "/opt/conda/envs/selftok_env"
# os.environ["PATH"] = f'{os.environ["CUDA_HOME"]}/bin:' + os.environ["PATH"]
# os.environ["LD_LIBRARY_PATH"] = f'{os.environ["CUDA_HOME"]}/lib64:' + os.environ.get("LD_LIBRARY_PATH", "")
import pprint
import time
import datetime
import numpy as np
import gc

import torch
import deepspeed

import torch.distributed as dist
import re
from mimogpt.utils import hf_logger
from mimogpt.engine import setup_task
from mimogpt.engine.utils import set_seed, parse_args, parse_args_from_yaml
from mimogpt.models.selftok.model_zoo import selftok_ckpts

if __name__ == "__main__":
    gc.disable()
    gc.set_threshold(70,10,1000)

    # parse args directly returns easydict, which is the same as ConfigObject, but easy to use
    args = parse_args()
    
    cfg_yml = parse_args_from_yaml(args.yml_path)
    cfg = args
    cfg.update(cfg_yml)

    set_seed(cfg.common.random_seed)
    # setup rank and local_rank
    if "RANK" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        args.rank = int(os.environ["RANK"])
    if "LOCAL_RANK" in os.environ:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
    is_root = args.rank % 8 == 0

    print("machine nums: {}\n".format(cfg.common.machines))
    print("save model --> is_root: {}\n".format(is_root))
    print(f"[rank{args.rank}] WORLD_SIZE={args.world_size}, LOCAL_RANK={args.local_rank}")

    if hasattr(cfg.common, "use_deepspeed") and cfg.common.use_deepspeed:
        deepspeed.init_distributed(
            dist_backend=args.backend,
            # init_method=args.init_method,
            auto_mpi_discovery=False,
            rank=args.rank,
            world_size=args.world_size,
            timeout=datetime.timedelta(hours=2.0),
        )
    else:
        torch.distributed.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            rank=args.rank,
            world_size=args.world_size,
            timeout=datetime.timedelta(hours=2.0),
        )

    torch.cuda.set_device(args.rank % 8)

    now_time = torch.from_numpy(np.array(int(time.time()))).float().cuda()
    dist.all_reduce(now_time)
 
    log_path = cfg.common.get("log_path", None)

    if log_path is None:
        cfg.common.log_path = os.path.join(log_path, str(now_time.cpu().numpy()))

    hf_logger.setup_logging_file(cfg.common.log_path, cfg.rank)
    hf_logger.info(f"CFG: {pprint.pformat(cfg)}")
    if hasattr(cfg, "tokenizer"):
        hf_logger.info("*** Important: Adopt [{}] Tokenizer ***".format(cfg.tokenizer.get("tokenizer_type", None)))

    # build trainer and register hooks
    Trainer, hook_list = setup_task(cfg=cfg, is_root=is_root)
    Trainer.register_hooks(hook_list)
    Trainer.register_model(Trainer.model.module if hasattr(Trainer.model, "module") else Trainer.model)
    # support float max epoch setting for detailed iter control
    max_iter = int(cfg.optimize.max_epochs * 100000)
    pretrained_model = cfg.model.get("pretrain_model", "")
    if pretrained_model:
        ckpt_path = pretrained_model
        matched = re.search(r"\w+_(\d+).pth", ckpt_path)
        if matched:
            cur_step = int(matched.group(1)) + 1
        else:
            cur_step = 0
    else:
        cur_step =  cfg.common.get("resume_from_steps", 0)
    Trainer.train(cur_step, max_iter)
