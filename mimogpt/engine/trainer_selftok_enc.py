# -*- coding: utf-8 -*-
import os
import sys
import torch
import deepspeed
import functools
import numpy as np
from time import time as ttime
import time
from itertools import repeat
import torch.distributed as dist
from torch.cuda.amp import autocast
from easydict import EasyDict
from collections import OrderedDict
from torch.cuda.amp import GradScaler
from copy import deepcopy
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file

sys.path.append(".")
from mimogpt.utils import hf_logger, AverageMeter
from mimogpt.models import build_backbone
from mimogpt.engine.utils import TrainerBase, print_model_param_num, print_model_params
from mimogpt.engine.utils import clip_gradient, build_optimizer, setup_deepspeed, build_selftok_optimizer
from mimogpt.models.selftok.multires_image_tokenizer import MultiImageTokenizer ### mark
from mimogpt.models.selftok.image_tokenizer import ImageTokenizer
from mimogpt.models.selftok.sd3.sd3_impls import SDVAE, CFGDenoiser, SD3LatentFormat
from mimogpt.models.selftok.model_zoo import selftok_ckpts
from mimogpt.datasets.imagenet_dataset import build_trainloader
import re
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from transformers import AutoModel
from transformers import AutoProcessor
import torch
from PIL import Image, ImageOps

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

log_dict_keys = ["loss", "dm_mse", "vl_loss", "perplexity", "enc_sum_grad",  "n_active", "perplexity_list"]

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
    

def _ddp_mean_tensor(x: torch.Tensor) -> torch.Tensor:
    """All-reduce mean for a scalar tensor living on CUDA."""
    if not dist.is_available() or not dist.is_initialized():
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x

def _to_scalar(x):
    """Detach any tensor and return a Python float; pass through numbers; stringify others."""
    if torch.is_tensor(x):
        return x.detach().float().mean().item()
    if isinstance(x, (float, int)):
        return float(x)
    return str(x)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def load_state(model, state_dict, prefix='',init_method = None):
    model_dict = model.state_dict()  # 当前网络结构
    if prefix == 'model.diffusion_model.':
        excluded_keys = ['context_embedder.bias', 'context_embedder.weight']
        if init_method == 1:
            excluded_keys = ['context_embedder.bias', 'context_embedder.weight', 'final_layer.adaLN_modulation.1.bias', 'final_layer.adaLN_modulation.1.weight', 'final_layer.linear.bias', 'final_layer.linear.weight']
            pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and k.replace(prefix,'') not in excluded_keys and 'context_block' not in k}
        elif init_method == 2:
            pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and k.replace(prefix,'') not in excluded_keys and 'context_block' not in k and 'x_block.attn' not in k}
        else:
            pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and k.replace(prefix,'') not in excluded_keys and 'context_block' not in k}
    elif prefix == 'encoder.':
        excluded_keys = ['query', 'q_norm1', 'q_norm2', 'post_norm', 'q_mlp', 'adaLN', 't_emb', 'final', 'quantizer']
        pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and all(ek not in k for ek in excluded_keys)}
    else:
        pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict}

    dict_t = deepcopy(pretrained_dict)
    for key, weight in dict_t.items():
        if key in model_dict and model_dict[key].shape != dict_t[key].shape:
            pretrained_dict.pop(key)
   
    m, u = model.load_state_dict(pretrained_dict, strict=False)
    if len(m) > 0:
        hf_logger.info(f"model missing keys:{m}")
    if len(u) > 0:
        hf_logger.info(f"mode unexpected keys:{u}")


class TrainerSelftokEnc(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.set_attribute()
        # dataloaders
        rank = dist.get_rank() #0 # machine index
        world_size = dist.get_world_size() #1 # num nodes
        print(f"Rank {rank} with world size {world_size} initiating dataset...")
        self.data_loader = build_trainloader(cfg.data, rank, world_size)
        self._data_loader_iter = iter(self.data_loader)
        hf_logger.info("dataloader length: %d", len(self.data_loader))
        self.cfg.dataloader_len = len(self.data_loader)
        
        # models
        self.set_vae()  # load vae and set to eval
        self.low_res_list = None
        self.decoder_use_low_res_rec = cfg.tokenizer.params.decoder_config.get('low_res', False)
        if hasattr(cfg.model, 'fix_decoder'):
            cfg.tokenizer.params.train_encoder_only = cfg.model.fix_decoder
        self.model = ImageTokenizer(**cfg.tokenizer.params)
        self.model.set_train()                          # set encoder and decoder to train
        # self.model.vl_model.visual.eval()
        # requires_grad(self.model.vl_model, False)
        # requires_grad(self.model.vl_model.model.visual.merger, True)
        requires_grad(self.model.vl_model, True)
        requires_grad(self.model.vl_model.model.visual, False)
        requires_grad(self.model.vl_model.model.language_model.embed_tokens, False)
        requires_grad(self.model.vl_model.abs_pos, False)
        requires_grad(self.model.vl_model.lm_head, False)

        state_dict = None
        self.ema = None
        pretrain_model_path = self.cfg.model.pretrain_model
        if pretrain_model_path != '':
            #state_dict = load_file(pretrain_model_path)
            state_dict = torch.load(pretrain_model_path, map_location="cpu", weights_only=False)
            try:
                hf_logger.info(f"Loading all...")
                #self.model.load_state_dict(state_dict, strict=True)
                self.model.load_state_dict(state_dict['state_dict'], strict=True)
            except:
                import time
                time.sleep(10 * (dist.get_rank() % 8))
                hf_logger.info(f"Loading partial state dict for rank: {dist.get_rank()}...")
                load_state(self.model, state_dict['state_dict'])
        else:
            if hasattr(self.cfg.tokenizer, "pretrained_dit_path") and self.cfg.tokenizer.pretrained_dit_path:
                teacher_path = self.cfg.tokenizer.pretrained_dit_path
                hf_logger.info(f'mmdit init_from {teacher_path}...')
                # state_dict = torch.load(
                #     teacher_path,
                #     map_location='cpu'
                # )
                state_dict = load_file(teacher_path)
                load_state(self.model.model, state_dict, 'model.diffusion_model.', init_method = cfg.tokenizer.params.decoder_config.init_method)

        
        print_model_param_num(cfg.model, self.model)
        print_model_param_num('encoder', self.model.encoder)
        print_model_param_num('decoder', self.model.model)
        print_model_param_num('vl_decoder', self.model.vl_model)
        self.model.cuda()
        if hasattr(self.cfg.common, "use_fsdp") and self.cfg.common.use_fsdp:
            print("USING FSDP from Pytorch...")
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                BackwardPrefetch,
                ShardingStrategy,
                FullStateDictConfig,
                StateDictType,
            )
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                CPUOffload,
                BackwardPrefetch,
            )
            from torch.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy,
                enable_wrap,
                wrap,
            )
            
            from mimogpt.models.selftok.sd3.mmdit import JointBlock
            print("Using size based policy")
            my_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={JointBlock}, )

            bf16 = MixedPrecision(
                ## param precision
                param_dtype=torch.bfloat16,
                # Gradient communication precision.
                reduce_dtype=torch.bfloat16,
                # Buffer precision.
                buffer_dtype=torch.bfloat16,
            )

            fp32 = MixedPrecision(
                ## param precision
                param_dtype=torch.float32,
                # Gradient communication precision.
                reduce_dtype=torch.float32,
                # Buffer precision.
                buffer_dtype=torch.float32,
            )

            self.model = self.model.to(torch.float32)
            
            self.model = FSDP(self.model,
                                  auto_wrap_policy=my_auto_wrap_policy,
                                  mixed_precision=bf16,
                                  device_id=torch.cuda.current_device(),
                                  # sharding_strategy=ShardingStrategy.FULL_SHARD,
                                  sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,
                              	  forward_prefetch=True,
                                  #sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                                  backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                                  use_orig_params=True,
                                  limit_all_gathers=True)

            self.optimizer = build_selftok_optimizer(self.model, self.cfg)
            if state_dict is not None and (not self.resume_exclude_opt) and 'opt' in state_dict:
                if hasattr(self.cfg.common, "use_fsdp") and self.cfg.common.use_fsdp:
                    from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig, StateDictType
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                    save_policy = FullStateDictConfig(rank0_only=False)
                    save_opt_policy = FullOptimStateDictConfig(rank0_only=False)
                    FSDP.set_state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy, save_opt_policy)
                    opt_state = FSDP.optim_state_dict_to_load(
                            self.model, self.optimizer,state_dict['opt']
                    )
                    self.optimizer.load_state_dict(opt_state)
                else:
                    self.optimizer.load_state_dict(state_dict['opt'])
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            self._scaler = ShardedGradScaler(enabled=self.use_fp16)
            dist.barrier()
        else:
            # setup DDP
            self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.rank % 8], find_unused_parameters=True)
            dist.barrier()

            # create optimizer and scaler
            self.optimizer = build_selftok_optimizer(self.model, self.cfg)
            if state_dict is not None and (not self.resume_exclude_opt) and 'opt' in state_dict:
                self.optimizer.load_state_dict(state_dict['opt'])
            if self.use_zero:
                from fairscale.optim.grad_scaler import ShardedGradScaler
                from fairscale.nn.data_parallel import ShardedDataParallel
                self.model = ShardedDataParallel(
                    self.model.module,
                    self.optimizer,
                    reduce_buffer_size=2000000,
                    reduce_fp16=self.use_fp16,
                )
                self._scaler = ShardedGradScaler(enabled=self.use_fp16)
            else:
                if DEVICE_TYPE == "ascend":
                    dynamic = cfg.common.use_dynamic if hasattr(cfg.common, "use_dynamic") else True
                    self._scaler = GradScaler(enabled=self.use_fp16, dynamic=dynamic)
                else: 
                    self._scaler = GradScaler(enabled=self.use_fp16)

        # logging
        self.meters = EasyDict()
        for key in log_dict_keys:
            self.meters[key] = AverageMeter(self.cfg.common.log_interval, fstr="%.5f")
        self.flops1 = None
        self.flops2 = None

    def set_attribute(self):
        self.start_time = ttime()
        self.vae_path = self.cfg.common.vae_path
        self.resume_exclude_opt = self.cfg.common.resume_exclude_opt
        self.pre_encode = self.cfg.common.pre_encode
        self.log_every = self.cfg.common.log_interval
        self.full_tokens = self.cfg.model.full_tokens \
            if hasattr(self.cfg.model, "full_tokens") else False
        if not hasattr(self.cfg.model, "fix_encoder"):
            self.cfg.model.fix_encoder = False
        if not hasattr(self.cfg.model, "fix_decoder"):
            self.cfg.model.fix_decoder = False
        if not hasattr(self.cfg.model, "fix_vlm"):
            self.cfg.model.fix_vlm = False
        self.dist = EasyDict()
        self.dist.rank = dist.get_rank()
        self.dist.world_size = dist.get_world_size()
        self.dist_rank = dist.get_rank()
        self.dist_world_size = dist.get_world_size() 
        self.use_deepspeed = self.cfg.common.use_deepspeed \
            if hasattr(self.cfg.common, "use_deepspeed") else False
        self.use_fp16 = self.cfg.common.use_fp16
        self.use_bf16 = self.cfg.common.use_bf16
        self.use_zero = self.cfg.common.use_zero

    def set_vae(self):
        self.vae = AutoencoderKL.from_pretrained(self.vae_path)
        self.vae.cuda()
        self.vae.eval()
    
    def set_ema_model(self, model):
        ema = deepcopy(model).to(torch.float32)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        update_ema(ema, model, decay=0)
        if not self.cfg.optimize.ema_in_cpu:
            ema = ema.cuda()
        ema.eval()
        return ema

    def get_model(self,):
        model = self.model.module if hasattr(self.model, "module") else self.model
        return model
    
    def run_step(self):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            batch = next(self._data_loader_iter)
        except StopIteration:
            # DataLoader 已经读完，重新初始化迭代器
            self._data_loader_iter = iter(self.data_loader)
            batch = next(self._data_loader_iter)

        x_image = batch["image"].cuda()  # for your VAE branch
        vit_pixel_values = batch["vit_pixel_values"].cuda().reshape(-1, 1176)
        grid_thw = batch["vit_grid_thw"].reshape(-1, 3)
        input_ids = batch["input_ids"].cuda() 
        attention_mask = batch["attention_mask"].cuda()
        labels = batch['labels'].cuda()
        position_ids = batch['pos_ids'].cuda()
        position_ids = position_ids.permute(1, 0, 2)

        # if not self.use_deepspeed:
        #     self.optimizer.zero_grad()
        # get vae latent
        with torch.no_grad():
            latent_vae = self.vae.encode(x_image).latent_dist.sample()
            latent_vae = SD3LatentFormat().process_in(latent_vae)
            # if isinstance(grid_thw, torch.Tensor):
            #     grid_thw = grid_thw.cuda()  # if already a tensor
            # latent_vit = self.model.vl_model.visual(vit_pixel_values, grid_thw=grid_thw).reshape(-1, 324, 2048)       

        full_tokens = self.full_tokens

        device = latent_vae.device

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.bfloat16, cache_enabled=False):
            loss, safe_log = self.model(
                x_vae=latent_vae,
                vit_pixel_values=vit_pixel_values,
                grid_thw=grid_thw,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                full_tokens=full_tokens,
                position_ids=position_ids,
            )

        loss.backward()
        self.optimizer.step()
        
        # with autocast(dtype=torch.bfloat16, cache_enabled=False):
        #     self._loss, log_dict = self.model(x_vae=latent_vae, vit_pixel_values=vit_pixel_values, grid_thw=grid_thw, input_ids=input_ids, \
        #         attention_mask=attention_mask, labels=labels, full_tokens=full_tokens)
            
        # self._loss.backward()
        # self.optimizer.step()

        enc_sum_grad = torch.zeros((), device=device, dtype=torch.float32)
        for p in self.get_model().encoder.parameters():
            if p.grad is not None:
                enc_sum_grad = enc_sum_grad + p.grad.detach().abs().sum()

        enc_sum_grad = _ddp_mean_tensor(enc_sum_grad).item()

        # ---- reduce and update meters (DDP safe, no numpy hops) ----
        # attach new fields as scalars
        safe_log['enc_sum_grad'] = float(enc_sum_grad)

        # DDP reduce each metric as mean and feed meters
        for k, v in safe_log.items():
            # only reduce numeric values
            if isinstance(v, (int, float)):
                t = torch.tensor(v, device=device, dtype=torch.float32)
                t = _ddp_mean_tensor(t)
                # your meters API: choose list vs scalar path
                self.meters[k] = t.item()
            elif isinstance(v, list):
                v = np.asarray(v, dtype=np.float32)
                t = torch.from_numpy(v).to(device)
                t = _ddp_mean_tensor(t)
                self.meters[k] = [float(x) for x in t.detach().cpu().tolist()]
            else:
                # non-numeric (strings) – skip or handle as needed
                pass

        # expose a couple of fields
        batch_mse = float(safe_log.get('dm_mse', 0.0))
        batch_ce = float(safe_log.get("vl_loss", 0.0))

        # ---- pretty print (rank 0 only) ----
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            import datetime
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            number_npus_per_node = 8
            number_nodes = 1
            iteration = self.iter
            total_iterations = 1200000
            consumed_images = self.iter * x_image.shape[0] * number_npus_per_node * number_nodes
            elapsed_time_per_iteration = (t1 - t0) * 1000.0  # ms
            learning_rate = 1e-4
            global_batch_size = x_image.shape[0] * number_npus_per_node * number_nodes
            lm_loss = batch_mse
            grad_norm = enc_sum_grad


            print(
                f"[{current_time}] iteration: {iteration}/{total_iterations} | "
                f"consumed images: {consumed_images} | "
                f"elapsed time per iteration (ms): {elapsed_time_per_iteration:.1f} | "
                f"learning rate: {learning_rate:.7E} | "
                f"global batch size: {global_batch_size} | "
                f"mse loss: {lm_loss:.6E} | "
                f"ce loss: {batch_ce:.6E} | "
                f"grad norm: {grad_norm:.3f} | "
            )

        # prepare logs
        # grads = [param.grad.abs().sum().item()
        #         if param.grad is not None else 0.0
        #         for param in self.get_model().encoder.parameters()
        # ]
        # sum_grads = sum(grads)
        # log_dict['enc_sum_grad'] = sum_grads
        # for k, v in log_dict.items():
        #     reduced_metric = torch.from_numpy(np.array(v)).float().cuda() / self.dist.world_size
        #     if isinstance(v,list):
        #         self.meters[k].reduce_update_list(reduced_metric)
        #     else:
        #         self.meters[k].reduce_update(reduced_metric)
        # self.batch_mse = log_dict['dm_mse']
        # self.n_active = log_dict['n_active']

        # torch.cuda.synchronize()
        # t1 = time.perf_counter()
        # if dist.get_rank() == 0:
        #     import datetime
        #     current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     number_npus_per_node = 8
        #     number_nodes = 1
        #     # number_npus_per_node = torch.cuda.device_count()
        #     # number_nodes = self.dist.world_size // number_npus_per_node
        #     iteration = self.iter
        #     total_iterations = 120000000
        #     consumed_images = self.iter * x_image.shape[0] * number_npus_per_node * number_nodes
        #     elapsed_time_per_iteration = (t1-t0)*1000  # 毫秒
        #     throughput_per_gpu = 107.8  # TFLOP/s/GPU (computing without renderer)
        #     learning_rate = 1E-04
        #     global_batch_size = x_image.shape[0] * number_npus_per_node * number_nodes
        #     lm_loss = self.batch_mse
        #     loss_scale = 1.0
        #     grad_norm = sum_grads
        #     skipped_iterations = 0
        #     nan_iterations = 0
        #     baseline_time = 1500  # 毫秒

        #     print(f"[{current_time}] iteration: {iteration}/{total_iterations} | "
        #         f"consumed images: {consumed_images} | "
        #         f"elapsed time per iteration (ms): {elapsed_time_per_iteration:.1f} | "
        #         f"throughput per GPU (TFLOP/s/GPU): {throughput_per_gpu:.1f} | "
        #         f"learning rate: {learning_rate:.7E} | "
        #         f"global batch size:  {global_batch_size} | "
        #         f"lm loss: {lm_loss:.6E} | "
        #         f"loss scale: {loss_scale:.1f} | "
        #         f"grad norm: {grad_norm:.3f} | "
        #         f"number of skipped iterations:   {skipped_iterations} | "
        #         f"number of nan iterations:   {nan_iterations} | "
        #         )