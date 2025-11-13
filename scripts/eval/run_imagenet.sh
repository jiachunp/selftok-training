set -x

GPUS=8

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=1008 \
    ./eval/gen/gen_images_selftok_rum.py \
    --output_dir ./outputs_imagenet1k_rum_70k_cfg1_train \
    --metadata_file /data/data/Unified_Parquet/ImageNet-Long-Caption/train-00000-of-00207.parquet \
    --batch_size 1 \
    --num_images 1 \
    --cfg_scale 1 \
    --model-path /data/zfd/SelfTok-o-main/ckpt_pretrain_RUM/0070000