export PYTHONPATH=$PYTHONPATH:/data/zfd/SelfTok-o-main
export WANDB_NAME="run_$(date +%Y%m%d_%H%M%S)"
export WANDB_PROJECT="selftok-o-try"
export WANDB_API_KEY="7f022df747bd563e0af2ac4ead78a4262b337dc5"

# pip install /data/zfd/flash_attn-2.8.2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# conda install -y -c conda-forge gcc_linux-64 gxx_linux-64 make cmake
# export CC="$(which x86_64-conda-linux-gnu-cc || which gcc)"
# export CXX="$(which x86_64-conda-linux-gnu-c++ || which g++)"

PROJECT_DIR="/data/zfd/SelfTok-o-main" 
cd "${PROJECT_DIR}" 

pip list
# pip install transformers==4.56.1 -i https://pypi.org/simple


# replace the variables with your own
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_vl.py \
  --dataset_config_file ./data/configs/example_selftok_FLUX.yaml \
  --layer_module Qwen2_5_VLMoTDecoderLayer \
  --vl_path /data/ckpt/Qwen2.5VL/Qwen2.5-VL-3B-Instruct \
  --use_flex True \
  --results_dir ./output_path_Flux6M \
  --checkpoint_dir ./ckpt_path_Flux6M \
  --num_replicate 1 \
  --warmup_steps 1000 \
  --lr 1e-4 \
  --ema 0.9999 \
  --expected_num_tokens 40960 \
  --max_num_tokens 46080 \
  --max_num_tokens_per_sample 40960 \
  --resume_from /data/zfd/SelfTok-o-main/ckpt_pretrain_npu/0078000 \
  --resume_model_only True
