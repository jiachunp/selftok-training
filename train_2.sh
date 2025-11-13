torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  train_net.py \
  --yml_path ./configs/mimo/selftok/sd3-res512/1536-FSQ.yml \
  --backend nccl \
  --init_method env:// \
  --world_size 8