# -*- coding: utf-8 -*-

import os

import copy
import yaml
import json
import shutil
import torch
import random
import numpy as np
from fairscale.optim import OSS

from .cloud_copy import mox_copy
from .train_loop import HookBase
from mimogpt.utils import hf_logger

import torch.distributed as dist


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_state_dict(pretrained_path):
    if "ViT-B-32.pt" not in pretrained_path and "RN50.pt" not in pretrained_path:
        # for HW pretrained model
        state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        if "module" in state_dict.keys():
            state_dict = state_dict["module"]
        return state_dict
    else:
        # for openai official pretrained model
        model = torch.jit.load(pretrained_path, map_location="cpu")
        state_dict = model.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            del state_dict[key]
        return state_dict


def get_state_dict_resume(pretrained_path):
    state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
    if len(state_dict) < 10:
        print(state_dict.keys())
    return state_dict


def print_model_params(model):
    print("\n--------------- trainable params ---------------")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("--------------- trainable params ---------------\n")


def print_model_param_num(model_info, model):
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(
                "\nmodel_info:\n{}\ntotal_params: {}\ntrainable_params: {}\n".format(
                    model_info, params_total / 1024 / 1024, params_trainable / 1024 / 1024
                )
            )
    else:
        print(
            "\nmodel_info:\n{}\ntotal_params: {}\ntrainable_params: {}\n".format(
                model_info, params_total, params_trainable
            )
        )


def calc_total_flops(model_info, model):
    from thop import profile

    input_v = torch.randn(1, 8, 3, 224, 224).cuda()  # img:[1, 1, 3, 224, 224] / video:[1, 8, 3, 224, 224]
    input_t = torch.ones(1, 80).long().cuda()  # [1, 80]
    flops, params = profile(model, inputs=(input_v, input_t))
    print("model: {}\nflops: {:.2f} Gflops\ntotal_params: {:.2f} M".format(model_info, flops / 1.0e9, params / 1.0e6))


class SaveModel(HookBase):
    def __init__(self, cfg, is_root):
        self.output_path = cfg.common.output_path
        self.is_root = is_root
        self.save_interval = int(cfg.common.save_per_epochs * getattr(cfg, 'dataloader_len', 2000))

        # only for save config before train
        self._cfg = cfg
        self.delete_after_upload = cfg.common.get("delete_after_upload", False)
        self.only_save_lora = cfg.common.get("only_save_lora", False)

    @property
    def save_path(self):
        if self.__dict__.get("_output_path", None) is None:
            output_path = os.path.join(self.output_path, "ckpt")
            os.makedirs(output_path, exist_ok=True)
            self.__dict__["_output_path"] = output_path
        return self.__dict__["_output_path"]

    def before_train(self):
        self.save_config()
        self.save_code()

    def after_step(self):
        if self.trainer.iter % self.save_interval == (self.save_interval - 1):
            if self.only_save_lora:
                save_name = "lora_iter_%d.pth" % (self.trainer.iter)
            else:
                save_name = "iter_%d.pth" % (self.trainer.iter)
            # tmp_cfg = copy.deepcopy(self._cfg.__dict__)
            #
            # self.save_model({
            #     'iter': self.trainer.iter,
            #     'state_dict': self.trainer.model.module.state_dict(),
            #     'opt': self.trainer.optimizer.state_dict(),
            #     'cfg': tmp_cfg
            # }, save_name)
            if hasattr(self._cfg.common, "use_fsdp") and self._cfg.common.use_fsdp:
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

                with FSDP.state_dict_type(self.trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
                    cpu_state = self.trainer.model.state_dict()
                    self.save_model(cpu_state, save_name)
            elif hasattr(self._cfg.common, "use_deepspeed") and self._cfg.common.use_deepspeed:
                self.save_model_deepspeed(save_name)
            else:
                if hasattr(self.trainer.model, "module"):
                    self.save_model(self.trainer.model.module.state_dict(), save_name)
                else:
                    self.save_model(self.trainer.model.state_dict(), save_name)
            torch.cuda.empty_cache()

    def save_config(self):
        if self.is_root:
            hf_logger.save_args(self._cfg)
            run_save_path = os.path.join(self.output_path, "run.yml")
            if not os.path.isfile(run_save_path):
                try:
                    with open(run_save_path, "w") as args_fh:
                        yaml.dump(self._cfg.__dict__, args_fh, sort_keys=False)
                    hf_logger.info("Run configs dump to %s" % run_save_path)
                except:
                    hf_logger.info("fail to dump run config!!")

    def save_code(self):
        if self.is_root:
            try:
                import moxing as mox

                local_code_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
                roma_code_path = os.path.join(self._cfg.train_url, os.path.split(local_code_path)[-1])
                mox_copy(local_code_path, roma_code_path, parallel=True)
                hf_logger.info("backup code success, roma_code_path:{}".format(roma_code_path))
            except:
                pass

    def save_model(self, checkpoint, save_name):
        if self.is_root:
            local_weights = os.path.join(self.save_path, save_name)
            for k, v in checkpoint.items():
                checkpoint[k] = checkpoint[k].cpu()
            torch.save(checkpoint, local_weights)
            self.upload_and_delete_local_model(local_weights)

    def save_model_deepspeed(self, save_name, ema_state=None):
        # deepspeed save pths
        self.trainer.model.cpu().save_checkpoint(self.save_path, f"deepspeed_save_dir_{save_name[:-4]}")
        self.trainer.model.cuda()
        pth_src_path = os.path.join(self.save_path, f"deepspeed_save_dir_{save_name[:-4]}")
        if ema_state is not None:
            if self.is_root:
                local_weights = os.path.join(pth_src_path, "ema.pt")
                torch.save(ema_state, local_weights)
        if hasattr(self._cfg.common, "deepspeed_merge_save") and self._cfg.common.deepspeed_merge_save:
            # create nas path to merge deepspeed pths
            MA_NFS_MOUNT_VOLUMES = json.loads(os.getenv("MA_NFS_MOUNT_VOLUMES"))
            nas_dst_path = MA_NFS_MOUNT_VOLUMES[0]["local_path"]
            nas_dst_path = os.path.join(nas_dst_path, "ckpt_tmp_{}".format(os.getenv("MASTER_ADDR")))
            if dist.get_rank() == 0 and not os.path.exists(nas_dst_path):
                os.mkdir(nas_dst_path)
                os.mkdir(os.path.join(nas_dst_path, f"deepspeed_save_dir_{save_name[:-4]}"))
            dist.barrier()
            # copy local pths to nas
            if dist.get_rank() % 8 == 0:
                os.system("cp {}/* {}/deepspeed_save_dir_{}".format(pth_src_path, nas_dst_path, save_name[:-4]))
            dist.barrier()
            # merge pths to one on nas
            if dist.get_rank() == 0:
                os.system("cp {} {}".format(os.path.join(self.save_path, "latest"), nas_dst_path))
                merged_pth_path = os.path.join(self.save_path, save_name)
                try:
                    os.system("python {}/zero_to_fp32.py {} {}".format(self.save_path, nas_dst_path, merged_pth_path))
                    self.upload_and_delete_local_model(merged_pth_path)
                except Exception as e:
                    print("save and upload deepspeed checkpoint {} failed, error: ".format(merged_pth_path, e))
                shutil.rmtree(nas_dst_path)
            dist.barrier()
            # delete local tmp pths
            if dist.get_rank() % 8 == 0:
                shutil.rmtree(pth_src_path)
        else:
            self.upload_and_delete_local_model(pth_src_path, rank0_only=False, parallel=True)
        dist.barrier()

    def upload_and_delete_local_model(self, local_weights, rank0_only=True, parallel=False):
        device_rank = dist.get_rank()
        allow_current_device_operate_files = (device_rank % 8 == 0 and not rank0_only) or (
            device_rank == 0 and rank0_only
        )
        try:
            import moxing as mox

            roma_weights_fp = os.path.join(self._cfg.train_url, local_weights)
            roma_weights_dirname = os.path.dirname(roma_weights_fp)
            if allow_current_device_operate_files:
                if not mox.file.exists(roma_weights_dirname):
                    mox.file.make_dirs(roma_weights_dirname)
                mox_copy(local_weights, roma_weights_fp, parallel)
                hf_logger.info("save weight success, roma_weights_fp:{}".format(roma_weights_fp))
        except:
            hf_logger.info("save weight success, local_weights_fp:{}".format(local_weights))

        if self.delete_after_upload and allow_current_device_operate_files:
            if os.path.isdir(local_weights):
                shutil.rmtree(local_weights)
            else:
                os.remove(local_weights)
            hf_logger.info(f"{local_weights} removed")


class SavePipeline(SaveModel):
    def save_model(self, checkpoint, save_name):
        # if hasattr(self.trainer.optimizer, "consolidate_state_dict"):
        #     self.trainer.optimizer.consolidate_state_dict(recipient_rank=0)
        if dist.get_rank() == 0:
            if not os.path.exists("/cache/pipe"):
                vae = self.trainer.first_stage_model
                tokenizer = self.trainer.tokenizer
                text_encoder = self.trainer.cond_stage_model
                unet = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
                scheduler = self.trainer.noise_scheduler
                from diffusers import TextToVideoSDPipeline

                t2v_pipe = TextToVideoSDPipeline(
                    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler
                )
                t2v_pipe.save_pretrained("/cache/pipe/")
                import moxing as mox

                roma_weights_dirname = os.path.join(self._cfg.train_url)
                if not mox.file.exists(roma_weights_dirname):
                    mox.file.make_dirs(roma_weights_dirname)
                mox_copy("/cache/pipe/", roma_weights_dirname, parallel=True)
                hf_logger.info("save pipe success, roma_weights_fp:{}".format(roma_weights_dirname))
            else:
                unet = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
                unet.save_pretrained("/cache/pipe/unet", variant="iter" + str(self.trainer.iter))
                # unet.save_pretrained('/cache/pipe/unet')
                # ckpt_name = f'diffusion_pytorch_model.iter{self.trainer.iter}.bin'
                # roma_weights_fp = os.path.join(self._cfg.train_url, 'unet', ckpt_name)
                # mox_copy(f'/cache/pipe/unet/{ckpt_name}', roma_weights_fp)
                # hf_logger.info("save pipe success, roma_weights_fp:{}".format(roma_weights_fp))
                optimizer = None if isinstance(self.trainer.optimizer, OSS) else self.trainer.optimizer.state_dict()
                state = {
                    "iter": self.trainer.iter,
                    "opt": optimizer,
                    "cfg": copy.deepcopy(self._cfg.__dict__),
                }
                torch.save(state, "/cache/pipe/unet/state.pth")
                roma_weights_dirname = os.path.join(self._cfg.train_url, "unet")
                mox_copy("/cache/pipe/unet", roma_weights_dirname, parallel=True)

        dist.barrier()


class LoadStateDict(HookBase):
    def __init__(self, pretrained_path, task, cfg):
        self.task = task
        self.pretrained_path = pretrained_path
        self.use_deepspeed = cfg.common.use_deepspeed
        self.cfg = cfg

    def before_train(self):
        if self.use_deepspeed:
            return
        if self.task == "vlip":
            self.load_vlip()
        elif self.task == "mimo":
            self.load_mimo()
        elif self.task == "mimo_gpt":
            self.load_mimo_gpt()
        elif self.task in ("mimo_interleaved", "mimo_interleaved_enc"):
            self.load_mimo_interleaved()
        elif self.task == "video_ldm":
            self.load_video_ldm()
        elif self.task == "mmdit":
            self.load_mmdit()
        else:
            self.load_mimo()

    def load_mimo_interleaved(self):
        print("I AM IN load_mimo_interleaved!!!!")
        if self.pretrained_path:
            state_dict = get_state_dict(self.pretrained_path)
            try:
                hf_logger.info("pretrained_path is {}".format(self.pretrained_path))
                self.trainer.model.module.load_state_dict(state_dict)
                hf_logger.info("successful loaded state dict from {}".format(self.pretrained_path))
            except:
                hf_logger.info("\nload pretrained_path: strict=False, remove visual branches......\n")

                model_dict = self.trainer.model.module.state_dict()  # 当前网络结构
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}  # 预训练模型中可用的weight

                # load V2
                cur_vocab = model_dict["text_embeddings.weight"].shape[0]
                use_df = self.cfg.tokenizer.get("use_dfvqgan", False)
                if use_df:
                    img_vocab = 8192
                else:
                    img_vocab = 16384
                for key, weight in state_dict.items():
                    if key in model_dict and model_dict[key].shape != state_dict[key].shape:
                        if "text_embeddings" in key:
                            pretrained_dict[key] = torch.cat(
                                (
                                    state_dict[key][:30522],
                                    model_dict[key][30522:cur_vocab, :].to(state_dict[key].device),
                                ),
                                dim=0,
                            )
                            hf_logger.info(
                                "[{}] shape change: pretrained model {} --> new model {}".format(
                                    key, state_dict[key].shape, model_dict[key].shape
                                )
                            )
                        elif "to_logits" in key:
                            pretrained_dict[key] = torch.cat(
                                (
                                    state_dict[key][:30522],
                                    model_dict[key][30522:cur_vocab].to(state_dict[key].device),
                                    state_dict[key][-img_vocab:],
                                ),
                                dim=0,
                            )
                            hf_logger.info(
                                "[{}] shape change: pretrained model {} --> new model {}".format(
                                    key, state_dict[key].shape, model_dict[key].shape
                                )
                            )
                        elif "text_pos_embeddings" in key:
                            pre_len = state_dict[key].shape[0]
                            cur_len = model_dict[key].shape[0]
                            if pre_len < cur_len:
                                pretrained_dict[key] = torch.cat(
                                    (
                                        state_dict[key][:pre_len],
                                        model_dict[key][pre_len:cur_len].to(state_dict[key].device),
                                    ),
                                    dim=0,
                                )
                            else:
                                pretrained_dict[key] = state_dict[key][:pre_len]
                        else:
                            pretrained_dict.pop(key)
                            hf_logger.info(
                                "[{}] popped: pretrained model {} --> new model {}".format(
                                    key, state_dict[key].shape, model_dict[key].shape
                                )
                            )

                # 输出没有加载的参数
                for k in model_dict:
                    if k not in pretrained_dict:
                        hf_logger.info("{} not loaded".format(k))

                model_dict.update(pretrained_dict)
                self.trainer.model.module.load_state_dict(model_dict, strict=False)

    def load_mimo_gpt(self):
        if self.pretrained_path:
            state_dict = get_state_dict(self.pretrained_path)
            try:
                hf_logger.info("pretrained_path is {}".format(self.pretrained_path))
                self.trainer.model.module.load_state_dict(state_dict)
                hf_logger.info("successful loaded state dict from {}".format(self.pretrained_path))
            except:
                hf_logger.info("\nload pretrained_path: strict=False, remove visual branches......\n")

                model_dict = self.trainer.model.module.state_dict()  # 当前网络结构
                pretrained_dict = dict()
                for k, v in state_dict.items():
                    if "decoder" in k and "transformer" not in k:
                        k = k.replace("decoder", "decoder.transformer")
                        if ".c_" in k:
                            v = v.t()
                    if k in model_dict:
                        pretrained_dict[k] = v
                pretrained_keys = list(pretrained_dict.keys())

                for key in pretrained_keys:
                    if model_dict[key].shape != pretrained_dict[key].shape:
                        print(
                            "[{}] shape change: pretrained model {} --> new model {}".format(
                                key, pretrained_dict[key].shape, model_dict[key].shape
                            )
                        )
                        # pretrained_dict.pop(key)
                        if len(model_dict[key].shape) == 2:
                            cur_shape0, cur_shape1 = model_dict[key].shape
                            pre_shape0, pre_shape1 = pretrained_dict[key].shape
                            if cur_shape0 >= pre_shape0 and cur_shape1 >= pre_shape1:
                                tmp = torch.cat(
                                    (
                                        state_dict[key],
                                        model_dict[key][pre_shape0:, :pre_shape1].to(state_dict[key].device),
                                    ),
                                    dim=0,
                                )
                                tmp = torch.cat(
                                    (tmp, model_dict[key][:, pre_shape1:].to(state_dict[key].device)), dim=1
                                )
                                pretrained_dict[key] = tmp
                            else:
                                pretrained_dict.pop(key)
                        elif len(model_dict[key].shape) == 1:
                            cur_shape0 = model_dict[key].shape[0]
                            pre_shape0 = pretrained_dict[key].shape[0]
                            if cur_shape0 >= pre_shape0:
                                tmp = torch.cat(
                                    (state_dict[key], model_dict[key][pre_shape0:].to(state_dict[key].device)), dim=0
                                )
                                pretrained_dict[key] = tmp
                            else:
                                pretrained_dict.pop(key)

                # 输出没有加载的参数
                for k in model_dict:
                    if k not in pretrained_dict:
                        hf_logger.info("{} not loaded".format(k))

                model_dict.update(pretrained_dict)
                self.trainer.model.module.load_state_dict(model_dict, strict=True)

    def load_mimo(self):
        if self.pretrained_path:
            state_dict = get_state_dict(self.pretrained_path)
            try:
                hf_logger.info("pretrained_path is {}".format(self.pretrained_path))
                self.trainer.model.module.load_state_dict(state_dict)
                hf_logger.info("successful loaded state dict from {}".format(self.pretrained_path))
            except:
                hf_logger.info("\nload pretrained_path: strict=False, remove visual branches......\n")

                model_dict = self.trainer.model.module.state_dict()  # 当前网络结构
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}  # 预训练模型中可用的weight

                # # TODO: hard code, load 3B model weights
                # for key, _ in model_dict.items():
                #     if "transformer.layers." in key:
                #         cur_layer_id = key.split('layers')[-1].split('.')[1]  # '41'
                #         # ref_layer_id = int(cur_layer_id) // 2
                #         if int(cur_layer_id) > 23:
                #             ref_layer_id = int(cur_layer_id) - 24
                #             k_ref = key.replace('transformer.layers.{}.'.format(cur_layer_id),
                #                                 'transformer.layers.{}.'.format(ref_layer_id))
                #             if k_ref in state_dict and model_dict[key].shape == state_dict[k_ref].shape:
                #                 pretrained_dict[key] = state_dict[k_ref]
                #                 hf_logger.info('{} weight is load from {}'.format(key, k_ref))

                cur_vocab = model_dict["text_embeddings.weight"].shape[0]
                img_vocab = 16384
                for key, weight in state_dict.items():
                    if key in model_dict and model_dict[key].shape != state_dict[key].shape:
                        if "text_embeddings" in key:
                            pretrained_dict[key] = torch.cat(
                                (
                                    state_dict[key][:30522],
                                    model_dict[key][30522:cur_vocab, :].to(state_dict[key].device),
                                ),
                                dim=0,
                            )
                            hf_logger.info(
                                "[{}] shape change: pretrained model {} --> new model {}".format(
                                    key, state_dict[key].shape, model_dict[key].shape
                                )
                            )
                        elif "to_logits" in key:
                            pretrained_dict[key] = torch.cat(
                                (
                                    state_dict[key][:30522],
                                    model_dict[key][30522:cur_vocab].to(state_dict[key].device),
                                    state_dict[key][-img_vocab:],
                                ),
                                dim=0,
                            )
                            hf_logger.info(
                                "[{}] shape change: pretrained model {} --> new model {}".format(
                                    key, state_dict[key].shape, model_dict[key].shape
                                )
                            )
                        elif "text_pos_embeddings" in key:
                            pre_len = state_dict[key].shape[0]
                            cur_len = model_dict[key].shape[0]
                            if pre_len < cur_len:
                                pretrained_dict[key] = torch.cat(
                                    (
                                        state_dict[key][:pre_len],
                                        model_dict[key][pre_len:cur_len].to(state_dict[key].device),
                                    ),
                                    dim=0,
                                )
                            else:
                                pretrained_dict[key] = state_dict[key][:pre_len]
                        else:
                            pretrained_dict.pop(key)
                            hf_logger.info(
                                "[{}] popped: pretrained model {} --> new model {}".format(
                                    key, state_dict[key].shape, model_dict[key].shape
                                )
                            )

                        # if "condition_to_decoder" in key:
                        #     pretrained_dict[key] = state_dict[key][:, :512]
                        #     hf_logger.info("[{}] shape change: pretrained model {} --> new model {}"
                        #                    .format(key, state_dict[key].shape, model_dict[key].shape))
                        # elif 'image_row_embeddings' in key or 'image_col_embeddings' in key:
                        #     pretrained_dict[key] = torch.cat(
                        #         (state_dict[key][:16, :],
                        #          model_dict[key][16:32, :].to(state_dict[key].device)),
                        #         dim=0)
                        #     hf_logger.info("[{}] shape change: pretrained model {} --> new model {}"
                        #                    .format(key, state_dict[key].shape, model_dict[key].shape))

                # # load dalle weights
                # for key, weight in state_dict.items():
                #     # text_embeddings: 16384 + 128 -> 21128 + 128
                #     if key == 'text_embeddings.weight':
                #         if model_dict[key].shape != state_dict[key].shape:
                #             pretrained_dict.pop(key)
                #             print("[{}] shape change: pretrained model {} --> new model {}"
                #                   .format(key, state_dict[key].shape, model_dict[key].shape))
                #
                #     if key in ['to_logits.1.weight', 'to_logits.1.bias']:
                #         if model_dict[key].shape != state_dict[key].shape:
                #             total_vocab_size = model_dict[key].shape[0]
                #             pretrained_dict[key] = torch.cat(
                #                 (model_dict[key][:total_vocab_size - 8192].to(state_dict[key].device),
                #                  state_dict[key][-8192:]), dim=0)
                #
                # # load GPT2 weights
                # for key, _ in model_dict.items():
                #     if "decoder." in key:
                #         k_ref = key.replace("decoder.", "")
                #         if k_ref in state_dict and model_dict[key].shape == state_dict[k_ref].shape:
                #             pretrained_dict[key] = state_dict[k_ref]
                #
                # for key, weight in state_dict.items():
                #     if key in model_dict and model_dict[key].shape != state_dict[key].shape:
                #         pretrained_dict.pop(key)
                #         hf_logger.info("[{}] shape change: pretrained model {} --> new model {}"
                #                        .format(key, state_dict[key].shape, model_dict[key].shape))

                # 输出没有加载的参数
                for k in model_dict:
                    if k not in pretrained_dict:
                        hf_logger.info("{} not loaded".format(k))

                model_dict.update(pretrained_dict)
                self.trainer.model.module.load_state_dict(model_dict, strict=False)

    def load_vlip(self):
        if self.pretrained_path:
            state_dict = get_state_dict(self.pretrained_path)
            try:
                hf_logger.info("pretrained_path is {}".format(self.pretrained_path))
                self.trainer.model.module.load_state_dict(state_dict)
                hf_logger.info("successful loaded state dict from {}".format(self.pretrained_path))
            except:
                hf_logger.info("\nload pretrained_path: strict=False, remove visual branches......\n")
                state_dict["state_dict"] = state_dict

                model_dict = self.trainer.model.module.state_dict()  # 当前网络结构
                pretrained_dict = {
                    k: v for k, v in state_dict["state_dict"].items() if k in model_dict
                }  # 预训练模型中可用的weight

                for k, _ in model_dict.items():
                    if "frame_fusion" in k:
                        k_ref = k.replace("frame_fusion", "transformer")
                        pretrained_dict[k] = state_dict["state_dict"][k_ref]

                for k, _ in model_dict.items():
                    if "textual." in k:
                        k_ref = k.replace("textual.", "")
                        pretrained_dict[k] = state_dict["state_dict"][k_ref]

                # for CLIP_VIP
                for key, weight in state_dict["state_dict"].items():
                    if "in_proj_weight" in key:
                        q_proj, k_proj, v_proj = weight.chunk(3, dim=0)
                        key_q = key.replace("in_proj_weight", "q_proj.weight")
                        key_k = key.replace("in_proj_weight", "k_proj.weight")
                        key_v = key.replace("in_proj_weight", "v_proj.weight")
                        pretrained_dict[key_q] = q_proj
                        pretrained_dict[key_k] = k_proj
                        pretrained_dict[key_v] = v_proj
                    elif "in_proj_bias" in key:
                        q_proj, k_proj, v_proj = weight.chunk(3, dim=0)
                        key_q = key.replace("in_proj_bias", "q_proj.bias")
                        key_k = key.replace("in_proj_bias", "k_proj.bias")
                        key_v = key.replace("in_proj_bias", "v_proj.bias")
                        pretrained_dict[key_q] = q_proj
                        pretrained_dict[key_k] = k_proj
                        pretrained_dict[key_v] = v_proj
                    elif key == "visual.positional_embedding":
                        pretrained_dict["visual.positional_embedding.weight"] = weight
                    elif key == "visual.class_embedding":
                        pretrained_dict["visual.added_cls"] = weight.expand(3, -1)

                # 输出没有加载的参数
                for k in model_dict:
                    if k not in pretrained_dict:
                        hf_logger.info("{} not loaded".format(k))

                model_dict.update(pretrained_dict)
                self.trainer.model.module.load_state_dict(model_dict, strict=False)

    def load_video_ldm(self):
        if self.pretrained_path:
            state = torch.load(self.pretrained_path, "cpu")
            if hasattr(self.trainer.model, "module"):
                missing_keys, unexpected_keys = self.trainer.model.module.load_state_dict(state, strict=False)
            else:
                missing_keys, unexpected_keys = self.trainer.model.load_state_dict(state, strict=False)
            hf_logger.info(f"missing_keys: {missing_keys}")
            hf_logger.info(f"unexpected_keys: {unexpected_keys}")

    def load_mmdit(self):
        if self.pretrained_path:
            state_dict = get_state_dict(self.pretrained_path)
            try:
                hf_logger.info("pretrained_path is {}".format(self.pretrained_path))
                self.trainer.model.module.load_state_dict(state_dict)
                hf_logger.info("successful loaded state dict from {}".format(self.pretrained_path))
            except:
                hf_logger.info("\nload pretrained_path: strict=False, remove invalid parameter......\n")

                model_dict = self.trainer.model.module.state_dict()  # 当前网络结构
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}  # 预训练模型中可用的weight

                for key, weight in state_dict.items():
                    if key in model_dict and model_dict[key].shape != state_dict[key].shape:
                        if "x_embedder.proj.weight" in key:  # [1280, 4, 2, 2] -> [1280, 16, 2, 2]
                            pretrained_dict[key] = torch.cat(
                                (state_dict[key], model_dict[key][:, 4:].to(state_dict[key].device)), dim=1
                            )
                            hf_logger.info(
                                "[{}] shape change: pretrained model {} --> new model {}".format(
                                    key, state_dict[key].shape, model_dict[key].shape
                                )
                            )
                        else:
                            pretrained_dict.pop(key)
                            hf_logger.info(
                                "[{}] popped: pretrained model {} --> new model {}".format(
                                    key, state_dict[key].shape, model_dict[key].shape
                                )
                            )

                # 输出没有加载的参数
                for k in model_dict:
                    if k not in pretrained_dict:
                        hf_logger.info("{} not loaded".format(k))

                model_dict.update(pretrained_dict)
                self.trainer.model.module.load_state_dict(model_dict, strict=False)

    def load_selftok(self):
        if self.pretrained_path:
            state_dict = get_state_dict(self.pretrained_path)
            try:
                hf_logger.info("pretrained_path is {}".format(self.pretrained_path))
                self.trainer.model.module.load_state_dict(state_dict)
                hf_logger.info("successful loaded state dict from {}".format(self.pretrained_path))
            except:
                hf_logger.info("\nload pretrained_path: strict=False, remove invalid parameter......\n")

                model_dict = self.trainer.model.module.state_dict()  # 当前网络结构
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}  # 预训练模型中可用的weight

                cur_vocab = model_dict["image_embeddings.weight"]
                print('cur_vocab',cur_vocab.shape)
                img_vocab = 16384
                for key, weight in state_dict.items():
                    if "image_embeddings" in key:
                        print(model_dict[key].shape)
                        print('state_dict',state_dict[key].shape)
                        pretrained_dict[key] = torch.cat(
                            (
                                state_dict[key][:16384],
                                model_dict[key][16384:].to(state_dict[key].device),
                            ),
                            dim=0,
                        )
                        hf_logger.info(
                            "[{}] shape change: pretrained model {} --> new model {}".format(
                                key, state_dict[key].shape, model_dict[key].shape
                            )
                        )
                    elif "head" in key:
                        pretrained_dict[key] = torch.cat(
                            (
                                state_dict[key][:16384],
                                model_dict[key][16384:].to(state_dict[key].device),
                            ),
                            dim=0,
                        )
                        hf_logger.info(
                            "[{}] shape change: pretrained model {} --> new model {}".format(
                                key, state_dict[key].shape, model_dict[key].shape
                            )
                        )

                # 输出没有加载的参数
                for k in model_dict:
                    if k not in pretrained_dict:
                        hf_logger.info("{} not loaded".format(k))

                model_dict.update(pretrained_dict)
                self.trainer.model.module.load_state_dict(model_dict, strict=False)


class LoadStateDict_resume(HookBase):
    def __init__(self, pretrained_path):
        self.pretrained_path = pretrained_path

    def before_train(self):
        if self.pretrained_path:
            state_dict = get_state_dict_resume(self.pretrained_path)

            self.trainer.optimizer.load_state_dict(state_dict["opt"])
            hf_logger.info("successfully resume optimizer state dict from {}".format(self.pretrained_path))

            self.trainer.iter = state_dict["iter"]
            hf_logger.info("successfully resume from iter: {}".format(self.trainer.iter))

            self.trainer.model.module.load_state_dict(state_dict["state_dict"])
            hf_logger.info("successfully resume model state dict from {}".format(self.pretrained_path))


class SaveModelandDisc(HookBase):
    def __init__(self, cfg, is_root):
        self.output_path = cfg.common.output_path
        self.is_root = is_root
        self.save_interval = int(cfg.common.save_per_epochs * getattr(cfg, 'dataloader_len', 2000))

        # only for save config before train
        self._cfg = cfg
        self.delete_after_upload = cfg.common.get("delete_after_upload", False)

    @property
    def save_path(self):
        if self.__dict__.get("_output_path", None) is None:
            output_path = os.path.join(self.output_path, "ckpt")
            os.makedirs(output_path, exist_ok=True)
            self.__dict__["_output_path"] = output_path
        return self.__dict__["_output_path"]

    def before_train(self):
        self.save_config()
        self.save_code()

    def after_step(self):
        if self.trainer.iter % self.save_interval == (self.save_interval - 1):
            save_name = "iter_%d.pth" % (self.trainer.iter)
            save_disc_name = "iter_%d_disc.pth" % (self.trainer.iter)

            self.save_model(self.trainer.model.module.state_dict(), save_name)
            self.save_model(self.trainer.loss.module.discriminator.state_dict(), save_disc_name)
            torch.cuda.empty_cache()

    def save_config(self):
        if self.is_root:
            hf_logger.save_args(self._cfg)
            run_save_path = os.path.join(self.output_path, "run.yml")
            if not os.path.isfile(run_save_path):
                try:
                    with open(run_save_path, "w") as args_fh:
                        yaml.dump(self._cfg.__dict__, args_fh, sort_keys=False)
                    hf_logger.info("Run configs dump to %s" % run_save_path)
                except:
                    hf_logger.info("fail to dump run config!!")

    def save_code(self):
        if self.is_root:
            try:
                import moxing as mox

                local_code_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
                roma_code_path = os.path.join(self._cfg.train_url, os.path.split(local_code_path)[-1])
                mox_copy(local_code_path, roma_code_path, parallel=True)
                hf_logger.info("backup code success, roma_code_path:{}".format(roma_code_path))
            except:
                pass

    def save_model(self, checkpoint, save_name):
        if self.is_root:
            local_weights = os.path.join(self.save_path, save_name)
            torch.save(checkpoint, local_weights)
            try:
                import moxing as mox

                roma_weights_fp = os.path.join(self._cfg.train_url, local_weights)
                roma_weights_dirname = os.path.dirname(roma_weights_fp)
                if not mox.file.exists(roma_weights_dirname):
                    mox.file.make_dirs(roma_weights_dirname)
                mox_copy(local_weights, roma_weights_fp)
                hf_logger.info("save weight success, roma_weights_fp:{}".format(roma_weights_fp))
            except:
                hf_logger.info("save weight success, local_weights_fp:{}".format(local_weights))

            if self.delete_after_upload:
                os.remove(local_weights)
                hf_logger.info(f"{local_weights} removed")
