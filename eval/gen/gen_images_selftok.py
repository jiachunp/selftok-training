# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import argparse
from safetensors.torch import load_file

import torch
import torch.distributed as dist
import sys 
sys.path.append("./")
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from PIL import Image
from modeling.bagel.qwen2_navit import NaiveCache
import pandas as pd


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def generate_image(prompt, cfg_scale=10.0, num_images=4, do_sample=False, temperature=1.0, max_length=1536, device=None):
    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt] * num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images

    cfg_generation_input, cfg_newlens, cfg_new_rope = gen_model.prepare_prompts(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope, 
        prompts=[""] * num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )

    cfg_generation_input = move_generation_input_to_device(cfg_generation_input, device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            cfg_past_key_values = gen_model.forward_cache_update_text(cfg_past_key_values, **cfg_generation_input)


    generation_input = gen_model.prepare_selftok_start_tokens(newlens, new_rope, new_token_ids)
    generation_input = move_generation_input_to_device(generation_input, device)

    cfg_generation_input = gen_model.prepare_selftok_start_tokens(cfg_newlens, cfg_new_rope, new_token_ids)
    cfg_generation_input = move_generation_input_to_device(cfg_generation_input, device)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        selftok_token_ids = gen_model.generate_selftok(
            past_key_values=past_key_values, 
            packed_key_value_indexes=generation_input["packed_key_value_indexes"],
            key_values_lens=generation_input["key_values_lens"],
            packed_start_tokens=generation_input["packed_start_tokens"],
            packed_query_position_ids=generation_input["packed_query_position_ids"],
            cfg_past_key_values=cfg_past_key_values, 
            cfg_packed_key_value_indexes=cfg_generation_input["packed_key_value_indexes"],
            cfg_key_values_lens=cfg_generation_input["key_values_lens"],
            cfg_packed_query_position_ids=cfg_generation_input["packed_query_position_ids"],
            selftok_token_len=max_length,
            do_sample=do_sample,
            temperature=temperature,
            cfg_scale=cfg_scale,
            end_token_id=new_token_ids['end_of_image'],
        )

    return selftok_token_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--metadata_file", type=str, required=True, help="JSONL file containing lines of metadata for each prompt.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=3)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--bagel-path', type=str, default='/home/jovyan/weijiawu/bagel_selftok/weights/BAGEL-7B-MoT')
    args = parser.parse_args()
    
    seed = 42
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images tokens are saved in {output_dir}")

    llm_config = Qwen2Config.from_pretrained("/home/jovyan/zfd/BAGEL/hf/Qwen2.5-0.5B-Instruct")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_pretrained("/home/jovyan/zfd/BAGEL/hf/siglip-so400m-14-980-flash-attn2-navit")
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join("/home/jovyan/zfd/BAGEL/flux/vae/", "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained("/home/jovyan/zfd/BAGEL/hf/Qwen2.5-0.5B-Instruct")
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    if rank == 0:
        print(msg)
    del model_state_dict

    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    gen_model = model

    cfg_scale = args.cfg_scale

    metadatas = pd.read_parquet(args.metadata_file)
    total_metadatas = len(metadatas)
    
    #prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    prompts_per_gpu = 1
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, 8)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    for idx in range(start, end):
        metadata = metadatas.iloc[idx]
        image_name = metadata.image_id
        prompt = metadata.caption
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start}")

        selftok_token_ids = generate_image(
            prompt=prompt,
            cfg_scale=cfg_scale, 
            num_images=args.batch_size,
            do_sample=True,
            device=device,
        )
        token_ids_np = selftok_token_ids.squeeze(1).cpu().numpy()
        
        save_path = os.path.join(args.output_dir, f"{image_name}.py")
        np.save(save_path, token_ids_np)

    print(f"GPU {rank} has completed all tasks")
    dist.barrier()