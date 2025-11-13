export PYTHONPATH=$PYTHONPATH:/data/zfd/SelfTok-o-main-202511111010_long_und
export WANDB_NAME="run_$(date +%Y%m%d_%H%M%S)"
export WANDB_PROJECT="selftok-o-vae_vit_long_und_270k"
export WANDB_API_KEY="7f022df747bd563e0af2ac4ead78a4262b337dc5"

PROJECT_DIR="/data/zfd/SelfTok-o-main-202511111010_long_und" 
cd "${PROJECT_DIR}" 

pip list
pip install -U transformers -i https://pypi.org/simple

echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "NUM_GPUS_PER_NODE: ${NUM_GPUS_PER_NODE}"
echo "MASTER_HOST: ${MASTER_HOST}"
echo "MASTER_ADDR: ${MASTER_ADDR}"

# replace the variables with your own
torchrun \
  --nnodes=${WORLD_SIZE} \
  --node_rank=${RANK} \
  --nproc_per_node=${NUM_GPUS_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/pretrain_unified_vl.py \
  --dataset_config_file ./data/configs/example_selftok_rum.yaml \
  --layer_module Qwen2_5_VLMoTDecoderLayer \
  --vl_path /data/ckpt/Qwen2.5VL/Qwen2.5-VL-3B-Instruct \
  --use_flex True \
  --results_dir ./results/output_path_vae_vit_long_und_270k \
  --checkpoint_dir ./results/ckpt_path_vae_vit_long_und_270k \
  --num_replicate $WORLD_SIZE \
  --num_workers 1 \
  --warmup_steps 2000 \
  --lr 1e-4 \
  --ema 0.9995 \
  --expected_num_tokens 40960 \
  --max_num_tokens 46080 \
  --max_num_tokens_per_sample 40960

  # --expected_num_tokens 10240 \
  # --max_num_tokens 11520 \
  # --max_num_tokens_per_sample 10240


  # --nnodes=${WORLD_SIZE} \
  # --node_rank=${RANK} \
  # --nproc_per_node=${NUM_GPUS_PER_NODE} \
  # --rdzv_backend=c10d \
  # --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \