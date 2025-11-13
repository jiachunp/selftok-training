import os
import yaml
from collections import OrderedDict
import random
from .model_zoo import Enc_models, DiT_models
from .models_ours import Encoder
import torch
from mimogpt.utils import hf_logger
from torch import nn
import numpy as np
from copy import deepcopy
from diffusers.models import AutoencoderKL
from mimogpt.models.selftok.diffusion import create_diffusion
from mimogpt.models.selftok.diti_utils import DiTi, DiTi_cont, DiTi_normal
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow
import torch.nn.functional as F
from mimogpt.models.selftok.vlm import Qwen2_5_VLForConditionalGenerationAbsImage
from typing import Any, List, Optional, Tuple, Union

MAX_LATENT_SIZE = 384

class ImageTokenizer(nn.Module):
    def __init__(
        self,
        image_size,
        k,
        encoder_hidden_size,
        pretrained_vl_path,
        enc,
        model,
        encoder_config,
        decoder_config,
        quantizer_config,
        k_m = None,
        k_s = None,
        stages =None,
        k_per_stage = None,
        noise_schedule_config = None,
        gradient_checkpointing = False,
        vl_loss_weight = 0.25,
        in_channels = 16,
        diffusion_type = 'flow',
        t2k = 1.,
        **kwargs,
    ):
        super().__init__()

        # reformat configs
        train_filter = decoder_config['train_filter']
        freeze_filter = decoder_config['freeze_filter']
        decoder_config['train_filter'] = train_filter.split('+') if train_filter != 'all' else None
        decoder_config['freeze_filter'] = freeze_filter.split('+') if freeze_filter != '' else []

        self.k_m = k_m
        self.k_s = k_s
        self.k = k
        self.t2k = t2k

        # create model
        self.diffusion_type = diffusion_type
        assert diffusion_type == 'flow'
        self.diffusion = RectifiedFlow(**noise_schedule_config)
        self.recon_ratio = 1.0      # reconstruction loss ratio (against velocity loss)
        assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        if stages is not None:
            self.diti = DiTi_cont(1000, k, stages, k_per_stage)
        else:
            self.diti = DiTi_normal(1000, self.k, self.k_m, self.k_s)

        latent_size = image_size // 8

        # modify configs
        encoder_config['pos_embed_max_size'] = 2 * latent_size
        encoder_config['diti'] = self.diti
        decoder_config['diti'] = self.diti

        enc_k = self.k

        self.encoder = Enc_models[enc](
            K=enc_k,
            input_size=latent_size,
            encoder_hidden_size=encoder_hidden_size,
            in_channels=in_channels,
            gradient_checkpointing=gradient_checkpointing,
            quantizer_config=quantizer_config,
            **encoder_config
        )

        self.model = DiT_models[model](
            K=self.k,
            input_size=latent_size, 
            encoder_hidden_size=encoder_hidden_size,
            in_channels=in_channels,
            gradient_checkpointing=gradient_checkpointing,
            **decoder_config
            )

        self.model.freeze()  # keep only params matching train_filter
        self.vl_model = Qwen2_5_VLForConditionalGenerationAbsImage.from_pretrained(
                pretrained_vl_path,
                torch_dtype=torch.bfloat16,
                device_map=None,
            )
        self.vl_loss_weight = vl_loss_weight

        self.T = self.diffusion.num_timesteps

        self.context_see_xt = kwargs.get('context_see_xt', True)
        self.context_see_rec = kwargs.get('context_see_rec', False)

    def set_train(self):
        self.model.train()
        self.encoder.train()
        self.vl_model.train()

    def set_eval(self):
        self.model.eval()
        self.encoder.eval()
        self.vl_model.eval()
    
    def forward(
        self,
        x_vae: torch.Tensor = None,
        vit_pixel_values: torch.Tensor = None,
        grid_thw: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_token_splits: Optional[List[int]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        full_tokens: bool = None,
        vl_backprop_through_encoder: bool = True,   # NEW: toggle to save memory
        **kwargs,
    ):
        device = x_vae.device
        dtype  = x_vae.dtype

        # make t on the right device/dtype (avoid stray .cuda() calls)
        t = torch.rand(x_vae.size(0), device=device, dtype=torch.float32)

        # move inputs once
        if isinstance(grid_thw, torch.Tensor) and grid_thw.device != device:
            grid_thw = grid_thw.to(device, non_blocking=True)
        if vit_pixel_values is not None and vit_pixel_values.device != device:
            vit_pixel_values = vit_pixel_values.to(device, non_blocking=True)

        # visual encoder is inference-only here – keep it out of autograd
        with torch.no_grad():
            x_vit = self.vl_model.visual(vit_pixel_values, grid_thw=grid_thw)  # [B, 324, 2048] after view
            x_vit = x_vit.reshape(-1, 324, 2048).detach()

        # set number of tokens
        if self.k_m is None:
            if full_tokens:
                k_batch = self.diti.to_indices(torch.ones_like(t) * 1000.0)
            else:
                t_tmp = (self.t2k * t).clamp(0, 1.0)
                k_batch = self.diti.to_indices(t_tmp * 1000.0)
        else:
            if full_tokens:
                k_batch = self.diti.to_indices(torch.ones_like(t))
            else:
                t_tmp = (self.t2k * t).clamp(0, 1.0)
                k_batch = self.diti.to_indices(t_tmp)

        # shift t according to timestep
        t = self.diffusion.shift_t(t, 1.0)

        # --- Encoder pass (optionally with grads) ---
        if not self.encoder.training:
            with torch.no_grad():
                encoder_hidden_states, to_quantizer_features, ori_hidden_states, attn_mask, quan_loss, enc_log, _ = \
                    self.encoder(x_vae=x_vae, x_vit=x_vit, d=k_batch, kwargs=kwargs)
        else:
            encoder_hidden_states, to_quantizer_features, ori_hidden_states, attn_mask, quan_loss, enc_log, _ = \
                self.encoder(x_vae=x_vae, x_vit=x_vit, d=k_batch, kwargs=kwargs)

        # diffusion (flow) loss
        noise = torch.randn_like(x_vae)
        model_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            mask=attn_mask,
            context_see_xt=self.context_see_xt,
            context_see_rec=self.context_see_rec,
        )
        loss_dict = self.diffusion.training_losses(self.model, x_vae, t, model_kwargs, noise=noise)
        dm_mse = loss_dict["loss"].mean()
        loss = dm_mse + quan_loss

        # VL loss (optionally stop grad through encoder to save memory)
        image_token_splits = [self.k for _ in range(x_vae.size(0))]
        # vl_image_embeds = ori_hidden_states if vl_backprop_through_encoder else ori_hidden_states.detach()
        vl_image_embeds = to_quantizer_features

        out = self.vl_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_embeds=vl_image_embeds,
            position_ids=position_ids,
            image_token_splits=image_token_splits,
            image_grid_thw=None,
            logits_to_keep=0,
        )
        vl_loss = out.loss
        loss = loss + self.vl_loss_weight * vl_loss

        # --- Clean, CPU-only logs (no GPU tensors, no graphs retained) ---
        # convert any tensor values in enc_log to Python floats
        safe_log = {}
        if isinstance(enc_log, dict):
            for k, v in enc_log.items():
                if torch.is_tensor(v):
                    # mean to scalar, then detach to CPU
                    safe_log[k] = v.detach().float().mean().item()
                else:
                    try:
                        safe_log[k] = float(v)
                    except Exception:
                        safe_log[k] = v 

        safe_log["loss"] = float(loss.detach().item())
        safe_log["vl_loss"] = float(vl_loss.detach().item())
        safe_log["dm_mse"] = float(dm_mse.detach().item())

        return loss, safe_log
        
    # def forward(
    #     self, 
    #     x_vae: torch.Tensor = None, 
    #     vit_pixel_values: torch.Tensor = None, 
    #     grid_thw: torch.Tensor = None, 
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     image_token_splits: Optional[List[int]] = None,
    #     full_tokens: bool = None, 
    #     **kwargs
    # ):
        
    #     shift = 1.0
    #     t = (torch.rand(x_vae.shape[0])).cuda()

    #     with torch.no_grad():
    #         if isinstance(grid_thw, torch.Tensor):
    #             grid_thw = grid_thw.cuda()  # if already a tensor
    #         x_vit = self.vl_model.visual(vit_pixel_values, grid_thw=grid_thw).reshape(-1, 324, 2048) 

    #     # set number of tokens
    #     if self.k_m is None:
    #         if full_tokens:
    #             k_batch = self.diti.to_indices(torch.ones_like(t) * 1000.0)
    #         else:
    #             t_tmp = (self.t2k * t).clamp(0, 1.0)
    #             k_batch = self.diti.to_indices(t_tmp * 1000.0)
    #     else:
    #         if full_tokens:
    #             k_batch = self.diti.to_indices(torch.ones_like(t))
    #         else:
    #             t_tmp = (self.t2k * t).clamp(0, 1.0)
    #             k_batch = self.diti.to_indices(t_tmp)
        
    #     # shift t according to timestep
    #     t = self.diffusion.shift_t(t, shift)

    #     # encode to get tokens
    #     if not self.encoder.training:
    #         with torch.no_grad():
    #             encoder_hidden_states, to_quantizer_features, ori_hidden_states, attn_mask, quan_loss, log_dict, _ = self.encoder(x_vae=x_vae, x_vit=x_vit, d=k_batch, kwargs=kwargs)
    #     else:
    #         encoder_hidden_states, to_quantizer_features, ori_hidden_states, attn_mask, quan_loss, log_dict, _ = self.encoder(x_vae=x_vae, x_vit=x_vit, d=k_batch, kwargs=kwargs)

    #     encoder_hidden_states_d = encoder_hidden_states
    #     attn_mask_d = attn_mask

    #     # flow training
    #     noise = torch.randn_like(x_vae)
    #     model_kwargs = dict(
    #         encoder_hidden_states=encoder_hidden_states_d,
    #         mask=attn_mask_d,
    #         context_see_xt = self.context_see_xt,
    #         context_see_rec = self.context_see_rec,
    #     )
    #     loss_dict = self.diffusion.training_losses(
    #         self.model, x_vae, t, model_kwargs, noise=noise,
    #     )
    #     batch_mse = loss_dict["loss"].sum() / loss_dict["loss"].shape[0]
    #     loss = batch_mse + quan_loss

    #     #VL training 
    #     image_token_splits = [self.k for _ in range(x_vae.shape[0])]

    #     out = self.vl_model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #         image_embeds=ori_hidden_states,
    #         image_token_splits=image_token_splits,
    #         image_grid_thw=None,
    #         logits_to_keep=0,
    #     )
    #     vl_loss = out.loss
    #     loss = loss + self.vl_loss_weight * vl_loss
    #     # prepare logs
    #     log_dict["loss"] = loss.item()
    #     log_dict["vl_loss"] = vl_loss.item()
    #     log_dict["dm_mse"] = batch_mse.item()
        
    #     return loss, log_dict
        