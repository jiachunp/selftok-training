# -*- coding: utf-8 -*-

import os
from mimogpt.engine.utils import *
from .trainer_selftok_enc import TrainerSelftokEnc


DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

def setup_task(cfg, is_root):
    if cfg.common.task == 'selftokenc':
        Trainer = TrainerSelftokEnc(cfg)
        hook_list = [
            SelfTokHook(cfg),
            SelfTokSaveHook(cfg, is_root=is_root),
        ]
        if cfg.common.val_interval > 0:
            hook_list.append(EvalSelftokHook(cfg))
        # 
    else:
        raise NotImplementedError

    cfg.profile = cfg.common.use_profile if hasattr(cfg.common, "use_profile") else 0
    if cfg.profile:
        hook_list.insert(0, TorchProfileHook(cfg))

    return Trainer, hook_list
