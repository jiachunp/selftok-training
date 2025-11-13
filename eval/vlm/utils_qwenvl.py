# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
import yaml

from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import (
    BagelConfig, 
    Bagel, 
    Qwen2_5_VLConfig, 
    Qwen2_5_VLForCausalLM, 
)
from modeling.qwen2 import Qwen2Tokenizer
from safetensors.torch import load_file

from data.transforms import ImageTransform


def load_model_and_tokenizer(args):
    config = Qwen2_5_VLConfig.from_json_file(os.path.join(args.model_path, "config.json"))
    config.qk_norm = True
    config.tie_word_embeddings = False
    config.layer_module ="Qwen2MoTDecoderLayer"

    config = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=config, 
        vit_config=config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
    )
    vl_model = Qwen2_5_VLForCausalLM(config)
    model = Bagel(vl_model, config)
    #model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")

    if os.path.exists(ema_path):
        model_state_dict = load_file(ema_path, device="cpu")
    else:
        model_state_dict = _collect_sharded_state_dict(args.model_path)

    #model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(msg)
    del model_state_dict
    model = model.cuda().eval()

    return model, tokenizer, new_token_ids


def _collect_sharded_state_dict(model_dir, prefer_glob_if_no_index=True):
    """
    Load a HuggingFace-style sharded safetensors checkpoint into one state_dict.
    Expects model.safetensors.index.json and model-00001-of-000XX.safetensors files.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    state_dict = {}

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_json = json.load(f)
        weight_map = index_json.get("weight_map", {})
        # Unique list of shard filenames, keep stable order
        shard_files = sorted(set(weight_map.values()))

        for shard_name in shard_files:
            shard_path = os.path.join(model_dir, shard_name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Shard missing: {shard_path}")
            # Load all tensors in this shard and merge
            part = load_file(shard_path, device="cpu")
            state_dict.update(part)
    else:
        # No index.json present: fall back to glob by filename pattern
        if not prefer_glob_if_no_index:
            raise FileNotFoundError(f"Missing index file: {index_path}")
        shard_files = sorted(
            fn for fn in os.listdir(model_dir)
            if fn.startswith("model-") and fn.endswith(".safetensors")
        )
        if not shard_files:
            raise FileNotFoundError(
                "No sharded safetensors found (model-*.safetensors)."
            )
        for shard_name in shard_files:
            shard_path = os.path.join(model_dir, shard_name)
            part = load_file(shard_path, device="cpu")
            state_dict.update(part)

    return state_dict

def build_transform():
    with open("./data/configs/example.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    max_image_size = data_config['vlm_sft']['image_transform_args']['max_image_size']
    min_image_size = data_config['vlm_sft']['image_transform_args']['min_image_size']
    image_stride = data_config['vlm_sft']['image_transform_args']['image_stride']
    max_pixels = data_config['vlm_sft']['image_transform_args']['max_pixels']

    image_transform = ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
    )

    return image_transform


def process_conversation(images, conversation):
    images = [pil_img2rgb(image) for image in images]
    return images, conversation
