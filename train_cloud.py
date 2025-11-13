# -*- coding: utf-8 -*-


import os
import pprint
import sys
import time
import socket
import argparse
import multiprocessing

os.environ["MOX_SILENT_MODE"] = "1"
os.environ["MOX_FILE_LARGE_FILE_METHOD"] = "1"  # for moxing download acceleration
import moxing as mox  # must after set MOX_FILE_LARGE_FILE_METHOD=1 !!!
from mimogpt.utils import read_from_yaml


project_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, project_path)


def cmd_exec(cmd, just_print=False):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("\n{}:INFO:{}".format(t, cmd))
    if not just_print:
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # --------------------distributed parameter---------------------
    parser.add_argument("--backend", type=str, default="nccl", help="use for current backend to support distributed")
    parser.add_argument(
        "--init_method", type=str, default="tcp://127.0.0.1:50717", help="init method to support distributed"
    )
    parser.add_argument("--rank", type=int, default=0, help="node rank")
    parser.add_argument("--world_size", type=int, default=1, help="current process number to support distributed")

    # -----------------------cloud parameter-----------------------
    parser.add_argument("--data_url", type=str, default="", help="data url")
    parser.add_argument("--train_url", type=str, default="", help="train url")
    parser.add_argument("--bucket", type=str, default=None, help="bucket name")
    parser.add_argument("--debug", type=int, default=0, help="use debug mode")

    # -----------------------train parameter-----------------------
    parser.add_argument("--args_yml_fn", type=str, default="", help="args_yml_fn")

    # -----------------------environment-----------------------
    parser.add_argument("--requirements", type=str, default="requirements.txt", help="requirements file")

    args, unknown = parser.parse_known_args()
    print("train_cloud.py argparse:")
    print("args:\n{}\n".format(args))
    print("unknown:\n{}\n".format(unknown))

    # write unknow into env
    unknown_args = {}
    for i in unknown:
        i = i[2:].split("=")
        unknown_args[i[0]] = i[1]
    print("unknown_args:\n{}\n".format(unknown_args))

    for env_args in unknown_args:
        os.environ[env_args] = unknown_args[env_args]

    if args.train_url == "":
        exit(0)
    print(">>>>>>> project_path ", project_path)

    ### setup environment
    os.system("export")
    os.system("lscpu")
    os.system("df -h")
    os.system("nvidia-smi topo --matrix")
    os.system("pip install -U pip")
    requirements_path = os.path.join(project_path, args.requirements)
    if os.path.exists(requirements_path):
        print("install requirements:", requirements_path)
        os.system("pip install -r {}".format(requirements_path))
    os.system("pip install youtokentome")
    os.system("pip list")

    # must after pip install!
    import torch
    from mimogpt.engine.utils import parse_args_from_yaml, common_cloud_copy, universal_cloud_copy

    # replace bucket-name with args.bucket
    yml_path = os.path.join(project_path, args.args_yml_fn)
    os.system("sed -i s/bucket-name/{}/g {}".format(args.bucket, yml_path))
    yml = parse_args_from_yaml(yml_path)
    yml_dict = read_from_yaml(yml_path)

    try:
        copy_func = yml_dict.get("cloud_copy").get("copy_func")  # common_cloud_copy / universal_cloud_copy
    except:
        copy_func = None  # universal_cloud_copy

    # copy data from s3 to cache
    if copy_func == "common_cloud_copy":
        common_cloud_copy(cfg=yml, rank=max(args.rank, 0), world_size=max(args.world_size, 1))  # node rand/world_size

    else:
        # Change all complex copy logic into a uniform manner
        cloud_cpy_list = yml_dict.get("cloud_copy", [])
        for download_task in cloud_cpy_list:
            print(pprint.pformat(download_task))
            universal_cloud_copy(rank=args.rank, world_size=args.world_size, **download_task)

    cloud_universal_cpy_list = yml_dict.get("universal_copy", [])
    for download_task in cloud_universal_cpy_list:
        print(pprint.pformat(download_task))
        universal_cloud_copy(rank=args.rank, world_size=args.world_size, **download_task)

    print("Show disk space after dataset copy is complete.")
    os.system("df -h")

    # for multi node synchronization
    s3_rank_id_fn = os.path.join(args.train_url, "rank_{}.txt".format(args.rank))
    if mox.file.exists(s3_rank_id_fn):
        mox.file.remove(s3_rank_id_fn, recursive=False)

    # 1 node
    if (args.rank == -1 and args.world_size == 0) or args.world_size == 1:
        rank = 0
        world_size = 1
        args.world_size = 1
        args.init_method = "tcp://127.0.0.1:50717"
        ip_get = socket.gethostbyname(socket.gethostname())
        mox.file.write(s3_rank_id_fn, "{}".format(ip_get))
    # multi node
    elif args.rank >= 0 and args.world_size > 1 and args.rank < args.world_size:
        rank = args.rank
        world_size = args.world_size

        ip_get = socket.gethostbyname(socket.gethostname())
        mox.file.write(s3_rank_id_fn, "{}".format(ip_get))
        # multi node, wait for other rank transfer data
        while True:
            all_rank_exist = True
            for rank_item in range(world_size):
                rank_fn_item = os.path.join(args.train_url, "rank_{}.txt".format(rank_item))
                if not mox.file.exists(rank_fn_item):
                    all_rank_exist = False
            if all_rank_exist:
                break
            else:
                time.sleep(5)  # delay 5 sec
    else:
        print("wrong rank:{}, world_size:{}".format(args.rank, args.world_size))
        exit(0)

    ngpus_per_node = torch.cuda.device_count()
    nnodes = world_size
    world_size = ngpus_per_node * world_size
    core_per_proc = int(multiprocessing.cpu_count() / 8)

    if hasattr(yml.common, "use_deepspeed") and yml.common.use_deepspeed:
        init_method = args.init_method.split("//")[-1]
        master_ip = init_method.split(":")[0]
        master_port = init_method.split(":")[-1]

        for fake_rank in range(ngpus_per_node):
            real_rank = rank * ngpus_per_node + fake_rank
            cmd = "cd {};export LD_LIBRARY_PATH=/home/ma-user/modelarts/lib:$LD_LIBRARY_PATH;".format(project_path)
            cmd += " OMP_NUM_THREADS=4 MASTER_ADDR={} MASTER_PORT={} LOCAL_RANK={} RANK={} WORLD_SIZE={} python train_net.py".format(
                master_ip, master_port, fake_rank, real_rank, world_size
            )
            cmd += " --distributed-backend={} ".format(args.backend)
            cmd += " --train_url={} ".format(args.train_url)
            cmd += " --deepspeed_config={}".format(yml.common.deepspeed_config)
            cmd += " --yml_path={}".format(args.args_yml_fn)
            cmd += " --world_size={}".format(world_size)

            for it in unknown:
                it = it.replace("--", "")
                key, val = it.split("=")
                cmd += " --{}={}".format(key, val)

            if fake_rank < ngpus_per_node - 1:
                cmd += " &"

            print(cmd)
            os.system(cmd)
    elif yml.common.task == "pangu":
        for fake_rank in range(ngpus_per_node):
            real_rank = rank * ngpus_per_node + fake_rank
            taskset = "taskset -c {}-{}".format(core_per_proc * fake_rank, core_per_proc * (fake_rank + 1) - 1)
            # export LD_LIBRARY_PATH solve the error "OSError: libaccess_sdk.so" when using memarts
            cmd = "cd {};export LD_LIBRARY_PATH=/home/ma-user/modelarts/lib:$LD_LIBRARY_PATH;".format(project_path)
            cmd += "OMP_NUM_THREADS=4 {} python mimogpt/models/pangu/finetune_pangu.py".format(taskset)
            cmd += " --distributed-backend={} --init-method={}".format(args.backend, args.init_method)
            cmd += " --rank={} --world_size={}".format(real_rank, world_size)
            cmd += " --train_url={}".format(args.train_url)
            cmd += " --yml_path={}".format(args.args_yml_fn)

            if fake_rank < ngpus_per_node - 1:
                cmd += " &"
            print(cmd)
            os.system(cmd)
    else:
        for fake_rank in range(ngpus_per_node):
            real_rank = rank * ngpus_per_node + fake_rank
            taskset = "taskset -c {}-{}".format(core_per_proc * fake_rank, core_per_proc * (fake_rank + 1) - 1)
            # export LD_LIBRARY_PATH solve the error "OSError: libaccess_sdk.so" when using memarts
            cmd = "cd {};export LD_LIBRARY_PATH=/home/ma-user/modelarts/lib:$LD_LIBRARY_PATH;".format(project_path)
            cmd += "OMP_NUM_THREADS=4 {} python train_net.py".format(taskset)
            cmd += " --backend={} --init_method={}".format(args.backend, args.init_method)
            cmd += " --rank={} --world_size={}".format(real_rank, world_size)
            cmd += " --train_url={}".format(args.train_url)
            cmd += " --yml_path={}".format(args.args_yml_fn)
            for it in unknown:
                it = it.replace("--", "")
                key, val = it.split("=")
                cmd += " --{}={}".format(key, val)

            if fake_rank < ngpus_per_node - 1:
                cmd += " &"
            print(cmd)
            os.system(cmd)
