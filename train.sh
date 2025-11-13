torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  train_net.py \
  --yml_path /data/code/selftok-tokenizer-training/configs/mimo/selftok/sd3-res512/1024-FSQ.yml \
  --backend nccl \
  --init_method env:// \
  --world_size 8