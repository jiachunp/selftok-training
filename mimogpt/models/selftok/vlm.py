# modeling_qwen2_5_vl_absimage.py

from typing import Any, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn

# ⬇️ 按你本地实际路径调整导入（Hugging Face 官方路径如下）
from transformers.utils import TransformersKwargs
from mimogpt.models.selftok.qwen2_5_VL.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

Tensor = torch.Tensor


class Absolute1DPositionalEncoding(nn.Module):
    """
    1D 绝对位置编码（learned 或 sinusoidal）。
    用法：传入每个样本内的图像 token 数（或所有图片的拼接长度列表），返回拼接后的编码并加到 image_embeds 上。
    """

    def __init__(self, hidden_size: int, max_len: int = 8192, learned: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.learned = learned

        if learned:
            self.pe = nn.Embedding(max_len, hidden_size)
            nn.init.normal_(self.pe.weight, std=0.02)
        else:
            # sinusoidal
            pe = torch.zeros(max_len, hidden_size, dtype=torch.float32)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, lengths: List[int], device, dtype) -> Tensor:
        """
        Args:
            lengths: 每张图/每段图像序列的 token 长度列表
        Returns:
            Tensor: [sum(lengths), hidden_size]
        """
        if any(L <= 0 for L in lengths):
            raise ValueError(f"All lengths must be > 0, got {lengths}")

        out = []
        for L in lengths:
            if L > self.max_len:
                raise ValueError(
                    f"Absolute1DPositionalEncoding max_len={self.max_len} < needed length {L}. "
                    f"Increase max_len or缩短序列。"
                )
            positions = torch.arange(L, device=device)
            if self.learned:
                enc = self.pe(positions)
            else:
                enc = self.pe.index_select(0, positions)
            out.append(enc.to(dtype))
        return torch.stack(out, dim=0)


class Qwen2_5_VLForConditionalGenerationAbsImage(Qwen2_5_VLForConditionalGeneration):
    """
    扩展版：允许直接给定 image_embeds，并为其叠加 1D 绝对位置编码（仅作用于图像 token 内部）。
    全局仍沿用 Qwen2.5-VL 的 mRoPE 相对位置编码（需要 image_grid_thw 来计算 3D 位置）。
    """

    def __init__(
        self,
        config,
        abs_pos_max_len: int = 8192,
        abs_pos_learned: bool = True,
    ):
        super().__init__(config)
        self.abs_pos = Absolute1DPositionalEncoding(
            hidden_size=config.text_config.hidden_size,  # 与 LM hidden 对齐
            max_len=abs_pos_max_len,
            learned=abs_pos_learned,
        )

    @staticmethod
    def _split_lengths_from_grid_thw(
        image_grid_thw: torch.LongTensor, spatial_merge_size: int
    ) -> List[int]:
        """
        Qwen2.5-VL 视觉侧的 merger 会把每 s×s 的 patch 合并成 1 个 token。
        每张图 token 数 = (t * h * w) / s^2
        """
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required to compute image token splits.")
        t, h, w = image_grid_thw.unbind(dim=1)  # [N]
        s2 = spatial_merge_size * spatial_merge_size
        lengths = (t * h * w // s2).tolist()
        if any(L <= 0 for L in lengths):
            raise ValueError(f"Invalid lengths from grid_thw={image_grid_thw.tolist()}")
        return lengths

    def _add_abs_pos_into_image_embeds(
        self,
        image_embeds: Tensor,
        image_token_splits: List[int],
        dtype,
        device,
    ) -> Tensor:
        """
        对拼接的 image_embeds 逐段叠加 1D 绝对位置编码（每段对应一张图）。
        """
        abs_pos = self.abs_pos(image_token_splits, device=device, dtype=dtype)  # [sumL, H]
        if abs_pos.shape != image_embeds.shape:
            # 仅检查最后一维相同，第一维必须一致
            if abs_pos.shape[0] != image_embeds.shape[0] or abs_pos.shape[1] != image_embeds.shape[1]:
                raise ValueError(
                    f"abs_pos shape {abs_pos.shape} not match image_embeds {image_embeds.shape}"
                )
        return image_embeds + abs_pos

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # ✅ 新增：直接传入图像嵌入
        image_embeds: Optional[torch.FloatTensor] = None,                 # [sum_image_tokens, H]
        image_token_splits: Optional[List[int]] = None,                    # 每张图 token 数；若不传则按 grid_thw 推导
        # ✅ 保留：用于全局 mRoPE 计算（必须提供，以便 3D 相对位置）
        image_grid_thw: Optional[torch.LongTensor] = None,                 # [num_images, 3]
        # ✅ 视频保持原逻辑（可选）
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: TransformersKwargs,
    ):
        """
        使用方法（伪代码）：
            # 1) 构造 input_ids，包含视觉占位符
            # 2) 准备 image_embeds: [sum_image_tokens, hidden]
            # 3) 提供 image_grid_thw，以便 mRoPE 正确计算
            outputs = model(
                input_ids=input_ids,
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # === 1) 先走父类的前半逻辑：把 inputs_embeds 准备好 ===
        # 但我们会阻止父类再去 get_image_features(pixel_values) —— 因为我们已直接给 image_embeds
        # 所以这里显式把 pixel_values 设为 None
        pixel_values = None  # 不再用视觉编码器提特征

        # 直接调用底层 self.model（Qwen2_5_VLModel）之前，我们需要先把 image_embeds 写进 inputs_embeds
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must pass either input_ids or inputs_embeds.")
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # === 2) 处理 image_embeds：叠加 1D 绝对位置编码，并替换占位符 ===
        if image_embeds is not None:
            # 2.1 计算每张图的 token 长度（若未给出则按 grid_thw 推导）
            if image_token_splits is None:
                if image_grid_thw is None:
                    raise ValueError("When image_token_splits is None, image_grid_thw must be provided.")
                spatial_merge_size = self.model.visual.spatial_merge_size
                image_token_splits = self._split_lengths_from_grid_thw(image_grid_thw, spatial_merge_size)

            # 2.2 叠加 1D 绝对位置编码（仅作用于 image_embeds 内部顺序）
            image_embeds = self._add_abs_pos_into_image_embeds(
                image_embeds=image_embeds,
                image_token_splits=image_token_splits,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

            # 2.3 把占位符位置找出来并替换为 image_embeds
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,  # 仅用于一致性检查
                video_features=None,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))

        # === 3) 调用底层语言模型（保持全局 mRoPE 相对位置编码）===
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=None,  # 不再提图像特征
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # 只计算必要 logits 以节省显存/算力
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    # 生成阶段也要支持 image_embeds（只在 prefill 步替换，占位后续 decode 步不再重复传）
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        # 扩展：接收 abs-image 模式的参数
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_splits: Optional[List[int]] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        # 视频保持原逻辑
        pixel_values_videos=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=None,  # 不跑视觉编码器
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        # 仅在 prefill（cache_position[0]==0）阶段传入 image_embeds；decode 步置空以避免重复替换
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["image_embeds"] = None
            model_inputs["image_token_splits"] = None
        else:
            model_inputs["image_embeds"] = image_embeds
            model_inputs["image_token_splits"] = image_token_splits

        return model_inputs
