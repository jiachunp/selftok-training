# -*- coding: utf-8 -*-
"""
训练脚本：直接使用外部 image_embeds（[sum_image_tokens, H]）进行 Qwen2.5-VL SFT 训练
- 模型类：Qwen2_5_VLForConditionalGenerationAbsImage（基于你上面实现的类）
- 仅对 assistant 回复部分打标签，image 占位符与 prompt 均 label=-100
- 多图：支持 image_token_splits 与 image_grid_thw 的拼接

【数据 JSONL 格式示例】（一行一个样本）
{
  "prompt": "Describe the image.",                # 用户问题
  "response": "A cat sitting on a sofa.",         # 助手回答（监督目标）
  "image_embed_paths": ["/data/tokens/cat_001.npy"],     # 每张图的 token 向量 (L_i, H)，支持 .npy / .pt
  "image_grid_thw": [[1, 32, 32]],                # 每张图的 (t,h,w)（视觉 grid；用于全局 mRoPE；必备）
  "bos": true,                                     # 可选，是否在文本前加 BOS（默认 True）
  "eos": true                                      # 可选，是否在文本末尾加 EOS（默认 True）
}

多图样例：
{
  "prompt": "Compare the two images.",
  "response": "Left is a cat, right is a dog.",
  "image_embed_paths": ["/data/tokens/left.npy", "/data/tokens/right.npy"],
  "image_grid_thw": [[1, 32, 32], [1, 32, 32]]
}
"""

import os, json, math, argparse, random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from mimogpt.models.selftok.vlm import Qwen2_5_VLForConditionalGenerationAbsImage

model_name_or_path = '/data/ckpt/Qwen2.5VL/Qwen2.5-VL-3B-Instruct'
tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
tok.padding_side = "right"
model = Qwen2_5_VLForConditionalGenerationAbsImage.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map=None,
)

# 检查外部 image_embeds 的 hidden size 是否匹配
text_hidden = model.config.text_config.hidden_size

device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)) if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()


####
# 读取关键特殊 token id（占位时要用）
cfg = model.config
IMAGE = cfg.image_token_id
VSTART = cfg.vision_start_token_id
VEND = getattr(cfg, "vision_end_token_id", None)
PAD = tok.pad_token_id or tok.eos_token_id
BOS = tok.bos_token_id
EOS = tok.eos_token_id

# ===== 2) 设定一个小批的“图像 token 数”方案（与 grid_thw 一致）=====
# 假设 spatial_merge_size=2，则 L = t*h*w/s^2
# 例子：样本1 1张图 L=64（t=1,h=16,w=16）；样本2 两张图 L=64 和 L=32（h=16,w=8）
splits_s1 = [64]
splits_s2 = [64]
splits_batch = splits_s1 + splits_s2                    # [64, 64, 32]

# 与上面对齐的 grid_thw（按 batch 顺序拼接）：[[t,h,w], ...]
image_grid_thw = torch.tensor([[1,16,16], [1,16,16]], dtype=torch.long)  # [3,3]

# ===== 3) 造“文本”：prompt/response（随便写，能tokenize就行）=====
prompt_1   = "Describe the image."
response_1 = "A cat sitting on a sofa."
prompt_2   = "Describe the image."
response_2 = "A dog."

def build_ids_and_labels(prompt, response, splits, add_bos=True, add_eos=True):
    ids, labels = [], []
    if add_bos and BOS is not None:
        ids.append(BOS); labels.append(-100)
    for L in splits:
        ids.append(VSTART); labels.append(-100)
        ids.extend([IMAGE] * L); labels.extend([-100] * L)
        if VEND is not None:
            ids.append(VEND); labels.append(-100)
    p_ids = tok.encode(prompt, add_special_tokens=False)
    r_ids = tok.encode(response, add_special_tokens=False)
    ids.extend(p_ids);         labels.extend([-100] * len(p_ids))
    ids.extend(r_ids);         labels.extend(r_ids)
    if add_eos and EOS is not None:
        ids.append(EOS);       labels.append(EOS)
    return ids, labels

ids1, labels1 = build_ids_and_labels(prompt_1, response_1, splits_s1)
ids2, labels2 = build_ids_and_labels(prompt_2, response_2, splits_s2)
print('ids1.shape', len(ids1), ids1)
print('ids2.shape', len(ids2), ids2)
print('labels1.shape', len(labels1), labels1)
print('labels2.shape', len(labels2), labels2)
# 动态 padding
max_len = max(len(ids1), len(ids2))
def pad_to(x, pad_id=PAD):
    return x + [pad_id] * (max_len - len(x))
input_ids = torch.tensor([pad_to(ids1), pad_to(ids2)], dtype=torch.long)
labels    = torch.tensor([pad_to(labels1, -100), pad_to(labels2, -100)], dtype=torch.long)
attention_mask = (input_ids != PAD).long()

# ===== 4) 造“假 image_embeds”：拼接顺序必须与 splits_batch 对齐 =====
H = model.config.text_config.hidden_size
sumL = sum(splits_batch)
# 用正态分布造个随机向量；dtype 与模型一致
image_embeds = torch.randn(sumL, H, dtype=model.dtype)

# ===== 5) 搬到 device，跑一次 forward =====
batch = {
    "input_ids": input_ids.to(device),
    "attention_mask": attention_mask.to(device),
    "labels": labels.to(device),
    "image_embeds": image_embeds.to(device),
    "image_grid_thw": image_grid_thw.to(device),
    "image_token_splits": splits_batch,  # list[int]，不需要 .to(device)
}

####

input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
labels = batch["labels"].to(device)
image_embeds = batch["image_embeds"].to(device)
image_grid_thw = batch["image_grid_thw"].to(device)
image_token_splits = batch["image_token_splits"]  # list[int]

print(input_ids.shape)  # torch.Size([2, 90])
print(attention_mask.shape)  # torch.Size([2, 90])
print(attention_mask[0])
print(attention_mask[1])
print(labels.shape)  # torch.Size([2, 90])
print(labels)
print(image_embeds.shape)
print('image_grid_thw', image_grid_thw.shape, image_grid_thw)
print('image_token_splits', len(image_token_splits), image_token_splits)

with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16), \
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_embeds=image_embeds,
        image_token_splits=image_token_splits,
        image_grid_thw=image_grid_thw,
        logits_to_keep=0,
    )
print(labels)
print(out.loss)
