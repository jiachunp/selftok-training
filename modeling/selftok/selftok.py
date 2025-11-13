# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from .qwen2_5_VL import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding, SelftokPositionEmbedding

from tqdm import tqdm


class SelftokConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        vl_config=None,
        vit_config=None,
        # vae_config=None, #zfd
        selftok_config=None, #zfd
        # latent_patch_size=2, #zfd
        # max_latent_size=32, #zfd
        selftok_token_len=1024, #zfd
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        # timestep_shift=1.0, #zfd
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.vl_config = vl_config
        self.vit_config = vit_config
        # self.vae_config = vae_config #zfd
        self.selftok_config = selftok_config #zfd
        # self.latent_patch_size = latent_patch_size #zfd
        # self.max_latent_size = max_latent_size #zfd
        self.selftok_token_len = selftok_token_len #zfd
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        # self.timestep_shift = timestep_shift #zfd


class Selftok(PreTrainedModel):
    config_class = SelftokConfig

    # def __init__(self, language_model, vit_model, config: SelftokConfig):
    def __init__(self, vl_model, config: SelftokConfig):
        super().__init__(config)
        self.vl_model = vl_model
        self.hidden_size = config.vl_config.text_config.hidden_size
        self.use_moe = "Mo" in config.vl_config.text_config.layer_module
        self.num_heads = config.vl_config.text_config.num_attention_heads

        if config.visual_gen:
            self.selftok_token_len = config.selftok_token_len #zfd
            self.selftok_pos_embed = SelftokPositionEmbedding(self.selftok_token_len, self.hidden_size) #zfd

        # if config.visual_und:
        #     self.vit_model = vit_model
        #     self.vit_patch_size = config.vit_config.patch_size
        #     self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
        #     self.vit_hidden_size = config.vit_config.hidden_size
        #     self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
        #     self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        # TODO
        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config


    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor, #全局相对位置编码 3D Rope of shape (3, seq_len)
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None, #图像内部绝对位置编码
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        packed_selftok_ids: Optional[torch.Tensor] = None, #zfd #image token ids
        packed_selftok_indexes: Optional[torch.LongTensor] = None, #zfd #image token position index
        packed_selftok_label_ids: Optional[torch.LongTensor] = None, #zfd #gt label
        packed_selftok_position_ids: Optional[torch.LongTensor] = None, #zfd #图像内部绝对位置编码
        selftok_ce_loss_indexes: Optional[torch.BoolTensor] = None, #zfd #selftok loss
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 3-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.vl_model.model.language_model.embed_tokens(packed_text_ids) #TODO
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            
            packed_vit_token_embed = self.vl_model.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        #zfd bgein
        # packed_selftok_ids: Optional[torch.Tensor] = None, #zfd #image token ids
        # packed_selftok_indexes: Optional[torch.LongTensor] = None, #zfd #image token position index
        # packed_selftok_label_ids: Optional[torch.LongTensor] = None, #zfd #gt label
        # packed_selftok_position_ids: Optional[torch.LongTensor] = None, #zfd #图像内部绝对位置编码
        # selftok_ce_loss_indexes: Optional[torch.BoolTensor] = None, #zfd #selftok loss
        if self.config.visual_gen:
            packed_selftok_embedding = self.vl_model.model.language_model.selftok_embed_tokens(packed_selftok_ids) #TODO
            selftok_token_pos_emb = self.selftok_pos_embed(packed_selftok_position_ids)
            packed_selftok_embedding = packed_selftok_embedding + selftok_token_pos_emb
            packed_sequence[packed_selftok_indexes] = packed_selftok_embedding
        #zfd end

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_selftok_indexes,
            )

        last_hidden_state = self.vl_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        # mse = None
        # if self.config.visual_gen:
        #     packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
        #     target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
        #     has_mse = packed_timesteps > 0
        #     mse = (packed_mse_preds - target[has_mse]) ** 2

        #zfd begin
        # packed_selftok_ids: Optional[torch.Tensor] = None, #zfd #image token ids
        # packed_selftok_indexes: Optional[torch.LongTensor] = None, #zfd #image token position index
        # packed_selftok_label_ids: Optional[torch.LongTensor] = None, #zfd #gt label
        # packed_selftok_position_ids: Optional[torch.LongTensor] = None, #zfd #图像内部绝对位置编码
        # selftok_ce_loss_indexes: Optional[torch.BoolTensor] = None, #zfd #selftok loss
        selftok_ce = None
        if self.config.visual_gen:
            packed_ce_selftok_preds = self.vl_model.selftok_lm_head(last_hidden_state[selftok_ce_loss_indexes])
            selftok_ce = F.cross_entropy(packed_ce_selftok_preds, packed_selftok_label_ids, reduction="none")
        #zfd end

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.vl_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(selftok_ce=selftok_ce, ce=ce)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids, final_text=False):
        packed_text_ids = list()
        packed_text_position_ids = [[],[],[]]
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()
        device = next(self.parameters()).device

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            if final_text:
                assistant = tokenizer.encode("assistant\n")
                text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']] + [new_token_ids['bos_token_id']] + assistant
            else:
                text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids[0].extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_position_ids[1].extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_position_ids[2].extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int, device=device),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    
    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.vl_model.model.language_model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.vl_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # image_tensor = transforms(image) # wwj
            image_tensor = image
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope
    
    def prepare_vit_images_qwen(
        self,
        curr_kvlens: List[int],
        curr_rope: List[int],
        images: List,                   
        processor,                      # Qwen 的 AutoProcessor
        new_token_ids,  # 需包含: {'start_of_image','end_of_image','image_token'}
        qwen_model=None,            
    ):
        """
        用 Qwen2.5-VL 的预处理来“估算”每张图的视觉 token 数 (K)，
        构造文本侧占位符，并把 pixel_values/grid_thw 带到下一步使用。
        不再返回 packed_vit_position_ids。
        """
        device = next(self.parameters()).device

        batch = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(device)          # [B, C, H, W] 或 [B, T, C, H, W]
        image_grid_thw = batch.get("image_grid_thw", None)       # [B, 3] (T, H, W)
        if image_grid_thw is None:
            raise ValueError("processor 未返回 image_grid_thw, 请升级 transformers/Qwen2.5-VL 权重。")

        thws = image_grid_thw.tolist()
        per_image_tokens = [int(t*h*w // (self.config.vl_config.vision_config.spatial_merge_size ** 2)) for (t, h, w) in thws]     # 每张图的视觉 token 数 K_i

        packed_vit_token_indexes = []
        vit_token_seqlens        = []
        packed_text_ids, packed_text_indexes = [], []
        packed_seqlens, packed_position_ids, packed_indexes = [], [[], [], []], []
        packed_key_value_indexes = []

        _curr = curr = 0
        newlens, new_rope = [], []

        soi_id  = new_token_ids['start_of_image']
        eoi_id  = new_token_ids['end_of_image']
        bos_id  = new_token_ids['bos_token_id']
        eos_id  = new_token_ids['eos_token_id']
        user_id = 872
        newline = 198

        for K, curr_kvlen, curr_position_id, thw in zip(per_image_tokens, curr_kvlens, curr_rope, thws):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            # [SOI]
            start_ids = [bos_id, newline, user_id, newline, soi_id]
            for ids in start_ids:
                packed_text_ids.append(ids)
                packed_text_indexes.append(_curr)
                packed_indexes.append(curr) 
                curr += 1
                _curr += 1

            packed_vit_token_indexes.extend(range(_curr, _curr + K))
            packed_indexes.extend(range(curr, curr + K))
            curr += K; _curr += K

            # [EOI]
            end_ids = [eoi_id, eos_id]
            for ids in end_ids:
                packed_text_ids.append(ids)
                packed_text_indexes.append(_curr)
                packed_indexes.append(curr); curr += 1; _curr += 1

            image_tokens = torch.tensor([bos_id, newline, user_id, newline, soi_id] + [self.config.vl_config.image_token_id] * K + [eoi_id, eos_id]).to(device)

            t, h, w = thw
            image_grid_thw = torch.tensor([[t, h, w]]) 
            image_rope_ids, mrope_position_deltas = self.vl_model.model.get_rope_index(image_tokens.unsqueeze(0), image_grid_thw)
            image_rope_ids = image_rope_ids.squeeze(1)
            
            # 记录长度/位置
            packed_position_ids[0].extend(curr_position_id + image_rope_ids[0])
            packed_position_ids[1].extend(curr_position_id + image_rope_ids[1])
            packed_position_ids[2].extend(curr_position_id + image_rope_ids[2])
            packed_seqlens.append(K + 7)
            vit_token_seqlens.append(K)

            newlens.append(curr_kvlen + K + 7)
            new_rope.append(packed_position_ids[2][-1] + 1)

        generation_input = {
            "packed_text_ids":          torch.tensor(packed_text_ids,          dtype=torch.long, device=device),
            "packed_text_indexes":      torch.tensor(packed_text_indexes,      dtype=torch.long, device=device),
            "vit_token_seqlens":        torch.tensor(vit_token_seqlens,        dtype=torch.int,  device=device),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device),
            "packed_position_ids":      torch.tensor(packed_position_ids,      dtype=torch.long, device=device),
            "packed_seqlens":           torch.tensor(packed_seqlens,           dtype=torch.int,  device=device),
            "packed_indexes":           torch.tensor(packed_indexes,           dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens":          torch.tensor(curr_kvlens,              dtype=torch.int,  device=device),
        }

        # 额外打包给下一步：像素与 THW
        extra_vision_meta = {
            "pixel_values":     pixel_values,          # [B, ...]
            "image_grid_thw":   image_grid_thw.to(device),
            "per_image_tokens": torch.tensor(per_image_tokens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope, extra_vision_meta

    @torch.no_grad
    def forward_cache_update_vit_qwen(
        self,
        past_key_values: "NaiveCache",
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        # ↓↓↓ Qwen 版不再需要这两个输入 ↓↓↓
        # packed_vit_tokens: torch.Tensor,
        # packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        *,
        pixel_values: torch.Tensor, # from extra_vision_meta["pixel_values"]
        image_grid_thw: torch.Tensor, # from extra_vision_meta["image_grid_thw"]
    ):
        """
        用 Qwen 内置 ViT 直接产出视觉 token 向量并注入。
        """
        device = next(self.parameters()).device
        hidden_size = self.hidden_size  # 你的 vl_model 的隐藏维度

        # 1) 文本嵌入 → 初始化 packed_sequence
        packed_text_embedding = self.vl_model.model.language_model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((int(packed_seqlens.sum().item()), hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        vis_out = self.vl_model.model.get_image_features(pixel_values, image_grid_thw.to(pixel_values.device))
        if isinstance(vis_out, (tuple, list)):
            image_embeds = vis_out[0]                 # [sum_K, Dv]
        elif hasattr(vis_out, "last_hidden_state"):
            image_embeds = vis_out.last_hidden_state  # [sum_K, Dv]
        else:
            image_embeds = vis_out

        if image_embeds.dtype != packed_sequence.dtype:
            image_embeds = image_embeds.to(packed_sequence.dtype)

        # 4) 直接覆盖占位的 [IMG] 位置（不再加 vit_pos_embed）
        packed_sequence[packed_vit_token_indexes] = image_embeds

        # 5) 调用你原来的推理入口，更新 KV
        extra_inputs = {"mode": "und"}
        output = self.vl_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,   # 这里仍是你原来的 1D pos
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            # is_causal=False,
            is_causal=True,
            **extra_inputs,
        )

        return output.past_key_values

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.vl_model.model.language_model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.vl_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        device = next(self.parameters()).device
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = [[],[],[]]

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(198)
            #packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids[0].append(curr_position_id)
            packed_query_position_ids[1].append(curr_position_id)
            packed_query_position_ids[2].append(curr_position_id)
            curr += curr_kvlen 

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long,  device=device),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long,  device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int,  device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long,  device=device),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.vl_model.model.language_model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.vl_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.vl_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)


    def prepare_selftok_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        #packed_query_position_ids = list()
        packed_query_position_ids = [[],[],[]]

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['start_of_image'])
            # packed_query_position_ids.append(curr_position_id)
            packed_query_position_ids[0].append(curr_position_id)
            packed_query_position_ids[1].append(curr_position_id)
            packed_query_position_ids[2].append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input
    

    def top_k_top_p_filtering(
        self,
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits
    
    @torch.no_grad
    def generate_selftok(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        cfg_past_key_values: Optional[NaiveCache] = None,
        cfg_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_packed_query_position_ids: Optional[torch.LongTensor] = None,
        selftok_token_len: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        cfg_scale: float = 1.0, 
        end_token_id: int = None,
        top_k: int = 0, 
        top_p: float = 1.0,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        selftok_pos_embed = self.selftok_pos_embed([i for i in range(1024)])
        while step < selftok_token_len:
            generated_sequence.append(curr_tokens)
            if step == 0:
                packed_selftok_embedding = self.vl_model.model.language_model.embed_tokens(curr_tokens)
            else:
                packed_selftok_embedding = self.vl_model.model.language_model.selftok_embed_tokens(curr_tokens)
                packed_selftok_embedding = packed_selftok_embedding + selftok_pos_embed[step-1].unsqueeze(0)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            if cfg_scale > 1.0:
                cfg_packed_query_indexes = torch.cumsum(cfg_key_values_lens, dim=0) + torch.arange(
                    0, len(cfg_key_values_lens), 
                    device=cfg_key_values_lens.device, 
                    dtype=cfg_key_values_lens.dtype
                )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            if cfg_scale > 1.0:
                cfg_uppacked = list(cfg_packed_key_value_indexes.split(cfg_key_values_lens.tolist(), dim=0))
                for i in range(len(cfg_uppacked)):
                    cfg_uppacked[i] += i
                cfg_packed_key_value_indexes = torch.cat(cfg_uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                if step == 0:
                    extra_inputs = {"mode": "und"}
                else:
                    extra_inputs = {"mode": "gen"}

            output = self.vl_model.forward_inference(
                packed_query_sequence=packed_selftok_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )

            if cfg_scale > 1.0: 
                cfg_output = self.vl_model.forward_inference(
                    packed_query_sequence=packed_selftok_embedding,
                    query_lens=query_lens,
                    packed_query_position_ids=cfg_packed_query_position_ids,
                    packed_query_indexes=cfg_packed_query_indexes,
                    past_key_values=cfg_past_key_values,
                    key_values_lens=cfg_key_values_lens,
                    packed_key_value_indexes=cfg_packed_key_value_indexes,
                    update_past_key_values=True,
                    is_causal=True,
                    **extra_inputs,
                )

                cfg_past_key_values = cfg_output.past_key_values
                cfg_packed_query_sequence = cfg_output.packed_query_sequence
                cfg_pred_logits = self.vl_model.selftok_lm_head(cfg_packed_query_sequence)

            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.vl_model.selftok_lm_head(packed_query_sequence)

            if cfg_scale > 1.0:
                pred_logits = cfg_pred_logits + cfg_scale * (pred_logits - cfg_pred_logits)

            if do_sample:
                pred_logits = pred_logits / temperature
                if top_k > 0 or top_p < 1.0:
                    pred_logits = self.top_k_top_p_filtering(pred_logits, top_k=top_k, top_p=top_p)
                probs = nn.functional.softmax(pred_logits, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            
            if cfg_scale > 1.0: 
                cfg_uppacked = list(cfg_packed_key_value_indexes.split(cfg_key_values_lens.tolist(), dim=0))
                for i in range(len(cfg_uppacked)):
                    cfg_uppacked[i] = torch.cat(
                        [cfg_uppacked[i], torch.tensor([cfg_uppacked[i][-1] + 1], device=cfg_uppacked[i].device)], dim=0
                    )
                cfg_packed_key_value_indexes = torch.cat(cfg_uppacked, dim=0)
                cfg_key_values_lens = cfg_key_values_lens + 1

            step += 1
        generated_sequence.append(curr_tokens)

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)


    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        processor,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.vl_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # system prompt for mme
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=["system\nYou are a helpful assistant."],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)


        # add images
        for image in images:
            generation_input, newlens, new_rope,extra_vision_meta = self.prepare_vit_images_qwen(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                processor=processor,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit_qwen(past_key_values, pixel_values=extra_vision_meta["pixel_values"],image_grid_thw=extra_vision_meta["image_grid_thw"], **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
            final_text=True
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        # output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        # output = self.tokenizer.decode(unpacked_latent[:,0])
        
        output = output.split('<|im_start|>')[-1]
 
        return output