export DEVICE_TYPE='npu';export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;python -m torch.distributed.run --standalone --nproc_per_node=8 main_eval.py --port 50000 --evaluation_type reconstruction
export DEVICE_TYPE='npu';export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;python -m torch.distributed.run --standalone --nproc_per_node=8 main_eval.py --port 50000 --evaluation_type reconstruction_renderer

export DEVICE_TYPE='npu';export ASCEND_RT_VISIBLE_DEVICES=0;python -m torch.distributed.run --standalone --nproc_per_node=1 main_eval.py --port 50000 --evaluation_type reconstruction

export DEVICE_TYPE='npu';export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7;python -m torch.distributed.run --standalone --nproc_per_node=7 main_eval.py --port 50000 --evaluation_type reconstruction
export DEVICE_TYPE='npu';export ASCEND_RT_VISIBLE_DEVICES=0;python -m torch.distributed.run --standalone --nproc_per_node=1 main_eval.py --port 50001 --evaluation_type insufficient