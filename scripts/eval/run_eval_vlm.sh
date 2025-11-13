# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# Set proxy and API key
export OPENAI_API_KEY=$openai_api_key

export GPUS=8

DATASETS=("mme")
# DATASETS=("mme" "mmbench-dev-en" "mmvet" "mmmu-val" "mathvista-testmini" "mmvp")
# DATASETS=("mmmu-val_cot")

DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

output_path=./results/mme 
model_path=/home/jovyan/weijiawu/bagel_selftok_qwenvl/ckpt_path_imagenet1k/0002000
bash scripts/eval/eval_vlm.sh \
    $output_path \
    --model-path $model_path