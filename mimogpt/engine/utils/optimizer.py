# -*- coding: utf-8 -*-
import os
import json
import torch
import inspect
import deepspeed

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")


def get_optimizer(cfg, parameters):
    if cfg.optimize.optimizer == "adam":
        optimize_cls = torch.optim.Adam
    elif cfg.optimize.optimizer == "adamw":
        optimize_cls = torch.optim.AdamW
    elif cfg.optimize.optimizer == "sgd":
        optimize_cls = torch.optim.SGD
    elif cfg.optimize.optimizer == "adamw1":
        from .lion import Lion

        optimize_cls = Lion
    else:
        raise NotImplementedError("Not Implement {} optimizer !!!".format(cfg.optimize.optimizer))

    if hasattr(cfg.common, "use_deepspeed") and cfg.common.use_deepspeed:
        return optimize_cls(
            parameters,
            lr=cfg.optimize.lr,
            weight_decay=cfg.optimize.get("weight_decay", 0.01),
            betas=cfg.optimize.get("betas", (0.9, 0.999)),
            eps=cfg.optimize.get("eps", 1e-08),
        )
        # return deepspeed.ops.adam.DeepSpeedCPUAdam(parameters, cfg.optimize.lr)
    elif cfg.common.use_zero:
        print("USING ZeRO")
        from fairscale import __version__ as __fs_version__
        from fairscale.optim import OSS

        oss_kwargs = {"broadcast_fp16": cfg.common.use_fp16}
        if __fs_version__ >= "0.4.6":
            oss_kwargs["force_broadcast_object"] = True
        return OSS(
            parameters,
            optimize_cls,
            lr=cfg.optimize.lr,
            weight_decay=cfg.optimize.get("weight_decay", 0.01),
            betas=cfg.optimize.get("betas", (0.9, 0.999)),
            eps=cfg.optimize.get("eps", 1e-08),
            **oss_kwargs,
        )
    else:
        return optimize_cls(
            parameters,
            lr=cfg.optimize.lr,
            weight_decay=cfg.optimize.get("weight_decay", 0.01),
            betas=cfg.optimize.get("betas", (0.9, 0.999)),
            eps=cfg.optimize.get("eps", 1e-08),
        )


def build_optimizer(cfg, model):
    from mimogpt.models.modules.llama.model import RMSNorm
    from mimogpt.models.modules.nanoGPT.model import LayerNorm

    norm_types = (
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
        RMSNorm,
        LayerNorm,
    )

    nodecay_names = []
    for name, mod in model.named_modules():
        if hasattr(mod, "bias"):
            nodecay_names.append("{}.bias".format(name))
        if isinstance(mod, norm_types):
            if hasattr(mod, "weight"):
                nodecay_names.append("{}.weight".format(name))
            if hasattr(mod, "scale"):
                nodecay_names.append("{}.scale".format(name))

    model_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay_params = [p for n, p in model_params if n not in nodecay_names]
    nodecay_params = [p for n, p in model_params if n in nodecay_names]
    # decay_params = [p for n, p in model_params if p.dim() >= 2]
    # nodecay_params = [p for n, p in model_params if p.dim() < 2]
    grouped_parameters = [
        {
            "params": decay_params,
            "init_lr": cfg.optimize.lr,
            "lr": cfg.optimize.lr,
            "weight_decay": cfg.optimize.weight_decay,
        },
        {
            "params": nodecay_params,
            "init_lr": cfg.optimize.lr,
            "lr": cfg.optimize.lr,
            "weight_decay": 0.0,
        },
    ]
    num_all_params = sum(p.numel() for n, p in model_params)
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num all trained parameter tensors: {len(model_params)}, with {num_all_params:,} parameters")
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    # exclude = lambda n: "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n or 'LayerNorm' in n
    # include = lambda n: not exclude(n)
    #
    # params_to_learn = [
    #     "frame_position_embeddings",
    #     "temporal_fusion",
    #     "visual_decoder",
    #     "text_decoder",
    # ]
    #
    # model_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # grouped_parameters = [
    #     # params to learn
    #     {
    #         'params': [p for n, p in model_params if any(nd in n for nd in params_to_learn) and exclude(n)],
    #         "init_lr": cfg.optimize.lr,
    #         "lr": cfg.optimize.lr,
    #         "weight_decay": 0.,
    #     },
    #     {
    #         'params': [p for n, p in model_params if any(nd in n for nd in params_to_learn) and include(n)],
    #         "init_lr": cfg.optimize.lr,
    #         "lr": cfg.optimize.lr,
    #         "weight_decay": cfg.optimize.weight_decay,
    #     },
    #
    #     # params to tune
    #     {
    #         'params': [p for n, p in model_params if not any(nd in n for nd in params_to_learn) and exclude(n)],
    #         "init_lr": cfg.optimize.lr * cfg.optimize.tune_lr_scale,
    #         "lr": cfg.optimize.lr * cfg.optimize.tune_lr_scale,
    #         "weight_decay": 0.
    #     },
    #     {
    #         'params': [p for n, p in model_params if not any(nd in n for nd in params_to_learn) and include(n)],
    #         "init_lr": cfg.optimize.lr * cfg.optimize.tune_lr_scale,
    #         "lr": cfg.optimize.lr * cfg.optimize.tune_lr_scale,
    #         "weight_decay": cfg.optimize.weight_decay
    #     }
    # ]

    optimizer = get_optimizer(cfg, grouped_parameters)
    return optimizer


def setup_deepspeed(args, rank):
    deepspeed_config = json.load(open(args.deepspeed_config))
    if "fp16" in deepspeed_config:
        assert "bf16" not in deepspeed_config
        if deepspeed_config["fp16"]["enabled"] is False:
            torch.backends.cuda.matmul.allow_tf32 = args.tf32
            torch.backends.cudnn.allow_tf32 = args.tf32
            if args.tf32:
                print(["rank", rank, "enable tf32 if device support"])
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    elif "bf16" in deepspeed_config:
        assert "fp16" not in deepspeed_config
        if deepspeed_config["bf16"]["enabled"] is False:
            torch.backends.cuda.matmul.allow_tf32 = args.tf32
            torch.backends.cudnn.allow_tf32 = args.tf32
            if args.tf32:
                print(["rank", rank, "enable tf32 if device support"])
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        if args.tf32:
            print(["rank", rank, "enable tf32 if device support"])


def clip_gradient(optimizer, grad_clip=5.0, eps=1.0e-15):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data = torch.where(
                    torch.isnan(param.grad.data), torch.zeros_like(param.grad.data), param.grad.data
                )
                param.grad.data = torch.where(
                    torch.abs(param.grad.data) < eps, torch.zeros_like(param.grad.data), param.grad.data
                )
                param.grad.data.clamp_(-grad_clip, grad_clip)
