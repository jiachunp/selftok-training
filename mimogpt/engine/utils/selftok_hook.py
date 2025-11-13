# -*- coding: utf-8 -*-

import os
# import time
from time import time
import torch
import psutil
import datetime
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from collections import OrderedDict
from .selftok.lr_scheduler import MyLRScheduler
from .checkpoint import SaveModel
from .selftok.threshold_scheduler import ThresholdScheduler, ObjectiveScheduler, LowResDroprateScheduler
from .train_loop import HookBase
from mimogpt.utils import hf_logger


def extract_exp_name(output_path, train_url):
    try:
        outputs = train_url.split('/')
        for i, output in enumerate(outputs):
            if 'time' in output and output[0] == '2':
                part2 = output[5:].replace('time_', '')
                part1 = outputs[i-1]
                break
        return f"{part1}_{part2}"
    except:
        return "debug"
    


def build_selftok_optimizer(model, cfg):
    fix_encoder, fix_decoder, fix_vlm = cfg.model.fix_encoder, cfg.model.fix_decoder, cfg.model.fix_vlm
    if hasattr(model, "module"):
        model_base = model.module
    else:
        model_base = model
    
    param_groups = []
    if not fix_encoder:
        if isinstance(model_base.encoder,dict):
            train_encoder_res = cfg.tokenizer.params.train_encoder_res
            param_groups.append({
            'params': model_base.encoder[train_encoder_res].parameters(),
            'lr': cfg.optimize.lr_scheduler.init_lr,
            'name': 'encoder'
            })
        else:
            param_groups.append({
                'params': model_base.encoder.parameters(),
                'lr': cfg.optimize.lr_scheduler.init_lr,
                'name': 'encoder'
            })
    if not fix_decoder:
        token_lr = cfg.optimize.lr_scheduler.token_lr if hasattr(cfg.optimize.lr_scheduler, "token_lr") else None
        if token_lr is None:
            param_groups.append({
                'params': model_base.model.parameters(),
                'lr': cfg.optimize.lr_scheduler.dit_lr,
                'name': 'decoder'
            })
        else:
            param_groups.append({
                'params': model_base.model.get_params_by_filter(select_list=['x_block', 'x_embedder']),
                'lr': cfg.optimize.lr_scheduler.dit_lr,
                'name': 'decoder'
            })
            param_groups.append({
                'params': model_base.model.get_params_by_filter(remove_list=['x_block', 'x_embedder']),
                'lr': token_lr,
                'name': 'token'
            })  # when train_filter does not include x_block and x_embedder, these two param groups equal model.parameters()
    if not fix_vlm:
        lm = model_base.vl_model.model.language_model
        to_train_parameters = [
            p for n, p in lm.named_parameters()
            if ('lm_head' not in n) and ('embed_tokens' not in n)
        ]
        param_groups.append({
            'params': to_train_parameters,
            'lr': cfg.optimize.lr_scheduler.vlm_mlp_lr,
            'name': 'vl_model_language_model'
        })
            
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.99), weight_decay=0)
    return opt


class SelfTokHook(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_attribute()
        self.is_root = (dist.get_rank() == 0)
        self.log_root_dir = os.path.join(cfg.common.output_path, "tb")
        self.save_interval = self.cfg.common.ckpt_interval
        
        self.exp_name = extract_exp_name(cfg.common.output_path, cfg.train_url)
        self.debug = (self.exp_name == 'debug')
        self.tb_copy_dir = os.path.join(cfg.common.log_path, self.exp_name)
        hf_logger.info(f"tb copy dir: {self.tb_copy_dir}")
        hf_logger.info(f"Log root path: {self.log_root_dir}")
        if self.is_root and not self.debug:
            os.makedirs(self.log_root_dir, exist_ok=True)
            self.tb_logger = SummaryWriter(self.log_root_dir, max_queue=1, flush_secs=10)
        else:
            self.tb_logger = None

    @torch.no_grad()
    def update_ema(self, ema_factor=0.9999):
        if hasattr(self.cfg.common, "use_fsdp") and self.cfg.common.use_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            with FSDP.summon_full_params(self.trainer.model):
                with torch.no_grad():
                    if self.trainer.ema is not None:
                        for p, p_ema in zip(self.trainer.model.model.parameters(), self.trainer.ema.parameters()):
                            if self.ema_in_cpu:
                                p1 = p.data.detach().clone().cpu()
                            else:
                                p1 = p.data.detach().clone()
                            p_ema.data.mul_(ema_factor).add_((1 - ema_factor) * p1)
        else:
            with torch.no_grad():
                if hasattr(self.trainer.model, "module"):
                    model_before_ddp = self.trainer.model.module.model
                else:
                    model_before_ddp = self.trainer.model.model
                for p, p_ema in zip(model_before_ddp.parameters(), self.trainer.ema.parameters()):
                    if self.ema_in_cpu:
                        p1 = p.data.detach().clone().cpu()
                    else:
                        p1 = p.data.detach().clone()
                    p_ema.data.mul_(ema_factor).add_((1 - ema_factor) * p1)

    def set_attribute(self):
        self.dead_code_threshold = self.cfg.tokenizer.params.quantizer_config.dead_code_threshold
        self.init_step1 = self.cfg.optimize.lr_scheduler.init_step1
        self.init_step2 = self.cfg.optimize.lr_scheduler.init_step2
        self.max_step = self.cfg.optimize.lr_scheduler.max_step
        self.min_lr1 = self.cfg.optimize.lr_scheduler.min_lr1
        self.min_lr2 = self.cfg.optimize.lr_scheduler.min_lr2 
        self.ema_in_cpu = self.cfg.optimize.ema_in_cpu
        self.gradient_accumulation_steps = self.cfg.optimize.get("gradient_accumulation_steps", 1)

    def set_scheduler(self):
        if hasattr(self.cfg.optimize, "dead_code_scheduler"):
            self.dead_code_scheduler = ThresholdScheduler(
                init_threshold=self.dead_code_threshold,
                final_threshold=self.cfg.optimize.dead_code_scheduler.final_threshold,
                constant_step=self.cfg.optimize.dead_code_scheduler.constant_step,
                end_step=self.cfg.optimize.dead_code_scheduler.end_step
            )
        self.lr_scheduler = MyLRScheduler(
            self.trainer.optimizer, 
            init_step1=self.init_step1,
            init_step2=self.init_step2,
            max_step=self.max_step,
            init_lr=self.cfg.optimize.lr_scheduler.init_lr,
            min_lr1=self.min_lr1,
            min_lr2=self.min_lr2
        )
        if hasattr(self.cfg.optimize, 'objective_scheduler'):
            self.objective_scheduler = ObjectiveScheduler(**self.cfg.optimize.objective_scheduler)
        if hasattr(self.cfg.optimize, 'low_res_drop_rate_scheduler'):
            self.low_res_drop_rate_scheduler = LowResDroprateScheduler(
                **self.cfg.optimize.low_res_drop_rate_scheduler
            )
    
    def before_step(self):
        current_iter = self.trainer.iter
        if hasattr(self.trainer.model, "module"):
            model_base = self.trainer.model.module
        else:
            model_base = self.trainer.model
        if hasattr(self, 'low_res_drop_rate_scheduler'):
            self.low_res_drop_rate_scheduler.step(model_base.model, current_iter)

    def update_scheduler(self):
        current_iter = self.trainer.iter
        self.lr_scheduler.step(current_iter)
        if hasattr(self.trainer.model, "module"):
            model_base = self.trainer.model.module
        else:
            model_base = self.trainer.model
        if hasattr(self, 'objective_scheduler'):
            self.objective_scheduler.step(model_base, current_iter)
        if hasattr(self, "dead_code_scheduler"):    
            self.dead_code_scheduler.step(model_base.encoder, current_iter)

    def ema_update(self):
        if (self.trainer.iter + 1) % self.gradient_accumulation_steps == 0:
            self.update_ema()
    
    def log(self):
        if self.trainer.iter % self.trainer.log_every == 0:
            if self.is_root:
                torch.cuda.synchronize()
                self.trainer.end_time = time()
                steps_per_sec = self.trainer.log_every / (self.trainer.end_time - self.trainer.start_time)
                lrr = self.trainer.optimizer.param_groups[0]["lr"]
                current_iter = self.trainer.iter
                logging_info = f"(step={current_iter:07d}), lr={lrr:.8f}, "
                
                meters = self.trainer.meters
                for k, v in meters.items():
                    avg_metric = meters[k]
                    if k in ['perplexity_list']:
                        metric_str = ",".join([str(int(p)) for p in avg_metric])
                        # avg_metric = avg_metric.mean()
                        logging_info += f"avg_{k}={metric_str}, "
                    elif k in ['n_active']:
                        logging_info += f"avg_{k}={int(avg_metric):04d}, "
                    elif k not in ['loss_small', 'loss_mid', 'loss_large', 'loss_uncon']:
                        logging_info += f"avg_{k}={avg_metric:.4f}, "
                    if not self.debug:
                        self.tb_logger.add_scalar(f"avg_{k}", avg_metric, current_iter)
                        
                ###
                logging_info += f"steps/sec={steps_per_sec:.2f}"
                hf_logger.info(logging_info)

                # if not self.debug:
                #     mox.file.copy_parallel(self.log_root_dir, self.tb_copy_dir)
                self.trainer.start_time = time()

    def after_step(self):
        self.update_scheduler()
        self.log()

    def after_train(self):
        if self.is_root and not self.debug:
            self.tb_logger.close()

    def set_train_state(self):
        fix_encoder, fix_decoder = self.cfg.model.fix_encoder, self.cfg.model.fix_decoder
        if hasattr(self.trainer.model, "module"):
            model_base = self.trainer.model.module
        else:
            model_base = self.trainer.model
        if fix_encoder:
            model_base.encoder.eval()
        if fix_decoder:
            model_base.model.eval()

    def before_train(self):
        self.set_scheduler()
        self.set_train_state()
        

class SelfTokSaveHook(SaveModel):
    def __init__(self, cfg, is_root):
        super().__init__(cfg, is_root)
        self.cfg = cfg
        self.save_interval = self.cfg.common.ckpt_interval

    def save_model(self, checkpoint, save_name):
        if self.is_root:
            local_weights = os.path.join(self.save_path, save_name)
            # for k, v in checkpoint.items():
            #     checkpoint[k] = checkpoint[k].cpu()
            torch.save(checkpoint, local_weights)
            self.upload_and_delete_local_model(local_weights)


    def after_step(self):
        if self.trainer.iter % self.save_interval == (self.save_interval - 1) and self.trainer.iter > 0:
            save_name = "iter_%d.pth" % (self.trainer.iter)
            # ema_state = self.trainer.ema.state_dict()
            # for k, v in ema_state.items():
            #     ema_state[k] = ema_state[k].cpu()
            if hasattr(self.cfg.common, "use_fsdp") and self.cfg.common.use_fsdp:
                from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig, StateDictType
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                save_opt_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
                #with FSDP.state_dict_type(self.trainer.model, StateDictType.FULL_STATE_DICT, save_policy, save_opt_policy):
                FSDP.set_state_dict_type(self.trainer.model, StateDictType.FULL_STATE_DICT, save_policy, save_opt_policy)
                cpu_state = self.trainer.model.state_dict()
                #original_osd = self.trainer.optimizer.state_dict()
                #opt_state =  FSDP.optim_state_dict(self.trainer.model, self.trainer.optimizer, optim_state_dict=original_osd)
                save_data = {
                    'iter': self.trainer.iter,
                    'state_dict': cpu_state,
                    #'ema_state_dict': ema_state,
                    #'opt': opt_state,
                    'cfg': self.cfg
                }
                self.save_model(save_data, save_name)
                hf_logger.info(f"Saved checkpoint to {save_name}")
                torch.cuda.empty_cache()
            else:
                if hasattr(self.trainer.model, "module"):
                    state_dict = self.trainer.model.module.state_dict()
                    # self.save_model(self.trainer.model.module.state_dict(), save_name)
                else:
                    state_dict = self.trainer.model.state_dict()
                for k, v in state_dict.items():
                    state_dict[k] = state_dict[k].cpu()
                save_data = {
                    'iter': self.trainer.iter,
                    'state_dict': state_dict,
                    # 'opt': self.trainer.optimizer.state_dict(),
                    'cfg': self.cfg
                }
                self.save_model(save_data, save_name)
                hf_logger.info(f"Saved checkpoint to {save_name}")
                torch.cuda.empty_cache()
