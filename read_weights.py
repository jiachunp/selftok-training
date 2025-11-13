import torch
from safetensors.torch import load_file

# safetensors 文件路径
# path = "/home/jovyan/zfd/SelfTok-o-main/ckpt_path_imagenet/0003000/model.safetensors"
# path = "/data/zfd/SelfTok-o-main/ckpt_pretrain_blip3o_long_caption/0002500/model.safetensors"
# path = "/data/zfd/SelfTok-o-main/ckpt_pretrain_blip3o_long_caption/0000500/model.safetensors"
# path = "/data/zfd/SelfTok-o-main/ckpt_pretrain_blip3o_long_caption/0002500/ema.safetensors"
# path = "/data/zfd/SelfTok-o-main/ckpt_pretrain_blip3o_long_caption/0000500/ema.safetensors"
path = '/data/zfd/SelfTok-o-main/ckpt_pretrain_FLUX6M/0015500/model.safetensors'
# 加载参数字典（不会执行反序列化的代码，比 torch.load 更安全）
state_dict = load_file(path, device="cpu")

# 打印所有 key（可先浏览一下）
print(list(state_dict.keys())[:50])  # 打印前 50 个 key

# key_list = list(state_dict.keys())
# key_
# print()

# 假设我们要看 'vl_model.model.language_model.layers.0.self_attn.k_proj.weight'
key = "vl_model.model.language_model.layers.0.self_attn.k_proj.weight"

# state_dict['selftok_pos_embed.pos_embed']
# state_dict['vl_model.lm_head.weight']
# state_dict['vl_model.model.language_model.embed_tokens.weight']
# state_dict['vl_model.selftok_lm_head.weight']
# state_dict['vl_model.model.language_model.selftok_embed_tokens.weight']

import pdb
pdb.set_trace()

if key in state_dict:
    tensor = state_dict[key]
    print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
    print(tensor)  # 打印完整内容（可能很大）
    # 或者只看前几行
    print(tensor[:5, :5])
else:
    print(f"Key '{key}' not found in checkpoint")
