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
    BagelConfig, Bagel, Qwen2_5_VLConfig, Qwen2_5_VLForCausalLM,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from PIL import Image
from modeling.bagel.qwen2_navit import NaiveCache
import pandas as pd

from mimogpt.infer.infer_utils import parse_args_from_yaml
from mimogpt.infer.SelftokPipeline import SelftokPipeline
from torchvision.utils import save_image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def generate_text(inputs, do_sample=False, temperature=0.3, max_length=512, device=None):
    past_key_values = NaiveCache(gen_model.config.vl_config.text_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    generation_input, newlens, new_rope = gen_model.prepare_vl_inputs(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        inputs=inputs, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_start_tokens(newlens, new_rope, new_token_ids)
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        unpacked_latent = gen_model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
    output = self.tokenizer.decode(unpacked_latent[:,0])
    output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--metadata_file", type=str, required=True, help="JSONL file containing lines of metadata for each prompt.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=3)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--bagel-path', type=str, default='/home/jovyan/weijiawu/bagel_selftok/weights/BAGEL-7B-MoT')
    parser.add_argument("--yml-path", type=str, default="./configs/renderer/E31.yml") # download from https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium.safetensors?download=true, require huggingface login, you have to change the format to .pt with safetensor_to_pt.py
    parser.add_argument("--pretrained", type=str, default="/home/jovyan/ckpt/E31/E31_renderer.safetensors") 
    parser.add_argument("--sd3_pretrained", type=str, default="/home/jovyan/ckpt/stable_diffusion_3_medium/sd3_medium.safetensors") 
    parser.add_argument("--data_size", type=int, default=512)
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

    vl_config = Qwen2_5_VLConfig.from_pretrained("/home/jovyan/weijiawu/bagel_selftok_qwenvl/weight/Qwen2.5-VL-3B-Instruct")
    vl_config.text_config.qk_norm = False
    vl_config.text_config.tie_word_embeddings = False
    vl_config.text_config.layer_module = "Qwen2_5_VLMoTDecoderLayer"

    config = BagelConfig(
        visual_gen=True,
        visual_und=False,
        vl_config=vl_config,  
    )
    vl_model = Qwen2_5_VLForCausalLM(vl_config)

    model = Bagel(vl_model, config)

    processor = AutoProcessor.from_pretrained("/home/jovyan/weijiawu/bagel_selftok_qwenvl/weight/Qwen2.5-VL-3B-Instruct")
    tokenizer = Qwen2Tokenizer.from_pretrained("/home/jovyan/weijiawu/bagel_selftok_qwenvl/weight/Qwen2.5-VL-3B-Instruct")
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    if rank == 0:
        print(msg)
    del model_state_dict

    model = model.to(device).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    prompt = "What is shown in this image?"

    text = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    output = generate_text(
            inputs = inputs, 
            do_sample=True,
            device=device,
        )

    print(output)