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
  --dataset_config_file ./data/configs/example_selftok_overfitting.yaml \
  --layer_module Qwen2_5_VLMoTDecoderLayer \
  --vl_path /data/ckpt/Qwen2.5VL/Qwen2.5-VL-3B-Instruct \
  --use_flex True \
  --results_dir ./output_path_imagenet_noweight_overfit \
  --checkpoint_dir ./ckpt_path_imagenet_noweight_overfit \
  --num_replicate 1 \
  --num_workers 1 \
  --warmup_steps 200 \
  --lr 1e-4 \
  --ema 0.9999 \
  --expected_num_tokens 20480 \
  --max_num_tokens 23040 \
  --max_num_tokens_per_sample 20480 \
  --save_every 2000 \
  --text_cond_dropout_prob 0.0

  # --expected_num_tokens 10240 \
  # --max_num_tokens 11520 \
  # --max_num_tokens_per_sample 10240


  # --nnodes=${WORLD_SIZE} \
  # --node_rank=${RANK} \
  # --nproc_per_node=${NUM_GPUS_PER_NODE} \
  # --rdzv_backend=c10d \
  # --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \