import os
import sys
from datetime import datetime
import random
import itertools
import time
import copy
import argparse

exec_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
project_path = os.path.join(exec_path, "..")
sys.path.insert(0, project_path)
thirdparty = os.path.join(exec_path, "../third_party")
sys.path.insert(0, thirdparty)

from modelarts_utils.api import modelarts_job_create_v2, modelarts_job_query_v2
from modelarts_utils.utils import get_pre_version_cfg_v2, get_pool_status
from ldm.common.cfg_parser import parse_yaml


def print_with_time(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("{}:INFO:{}".format(t, msg))


def task_create():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--auto_upload_code",
        type=int,
        default=1,
        help="auto_upload_code",
    )
    parser.add_argument(
        "--modelarts_cfg_path",
        type=str,
        default="tools/modelarts_cfg.yml",
        help="modelarts_cfg_path",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="timestamp, for CI",
    )
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.modelarts_cfg_path):
        print("can not find modelarts_cfg_path, please contact to 各自分队负责人!")
        exit()

    if args.auto_upload_code:
        os.system("python tools/code_upload.py --modelarts_cfg_path={}".format(args.modelarts_cfg_path))
        print("finish code upload")

    modelarts_cfg = parse_yaml(args.modelarts_cfg_path)
    user_id = modelarts_cfg["account"]

    # Read all job cfg from a yaml
    job_cfg = modelarts_cfg["job_cfg"]
    job_name = job_cfg["job_name"]
    app_name, prefix = job_cfg["app_name"], job_cfg["prefix"]
    bucket = job_cfg["bucket"]
    region = job_cfg["region"]
    job_priority = job_cfg["job_priority"]
    priority = job_cfg["priority"]
    # cfg name and special task cfg
    cfgs = job_cfg["cfg_names"]

    job_name = "{}_{}".format(job_name, prefix)

    # default cfg
    task_cfg = {
        "user_id": user_id,
        "code_path": "s3://{}/code/{}/MGM/".format(bucket, user_id),
        "running_script": "train_cloud_eagle.py",
        "running_args": {
            "args_yml_fn": None,
            "bucket": bucket,
            "profile": 0,
        },
        "base_train_url": "/{}/outputs/{}/MGM/{}/".format(bucket, user_id, job_name),
        "GPUs": 8,  # how many GPUs each node
        "num_nodes": 1,
        "repeat_times": 1,
        "order": "cycle",  # order or cycle
        "environments": None,
    }
    running_args_from_cfg = job_cfg["running_args"]
    for k, v in running_args_from_cfg.items():
        if isinstance(v, str) and "{}" in v:
            running_args_from_cfg[k] = v.format(bucket)

    task_cfg["running_args"].update(running_args_from_cfg)

    enum_params = {
        # 'max_epoch': [100],
    }

    enum_params = list(itertools.product(*[[[k, _v] for _v in v] for k, v in enum_params.items()]))
    enum_params = [dict(x) for x in enum_params]

    if args.timestamp is None:
        task_time_id = datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S")
    else:
        task_time_id = args.timestamp

    task_cfgs = list(itertools.product(cfgs, enum_params))
    if task_cfg["order"] == "order":
        task_cfgs = [
            [*x, random.randint(10000000, 99999999)] for x in task_cfgs for _ in range(task_cfg["repeat_times"])
        ]
    else:
        task_cfgs = [
            [*x, random.randint(10000000, 99999999)] for _ in range(task_cfg["repeat_times"]) for x in task_cfgs
        ]

    app_token = modelarts_cfg["app"][app_name]["token"]
    find, pre_version_cfg = get_pre_version_cfg_v2(job_name + "_V0001", app_name, app_token)
    version_strat_idx = 1
    pre_version_uid = ""
    if find:
        pre_version_uid = pre_version_cfg["id"]
        track_id = pre_version_cfg["trackId"]
        ret = modelarts_job_query_v2(app_name, app_token, filter_word=job_name)
        if ret["success"]:
            train_jobs = ret["trainJobs"]
            version_names = [x["name"] for x in train_jobs if x["trackId"] == track_id]
            version_names = [x.split("_")[-1] for x in version_names]
            version_names = [x for x in version_names if len(x) == 5 and x[0] == "V"]
            if len(version_names) > 0:
                version_strat_idx = max([int(x[1:]) for x in version_names]) + 1
        print_with_time("find job name:{}, pre_version_uid={}".format(job_name, pre_version_uid))
    else:
        print_with_time("can not find job name: {}, please check it, now will create new job".format(job_name))

    app_cfgs = {
        "name": app_name,
        "token": app_token,
        "vendor": modelarts_cfg["app"][app_name]["vendor"],
    }
    bucket_id = modelarts_cfg["app"][app_name]["bucket"][bucket]["id"]
    pool_cfgs = modelarts_cfg["app"][app_name]["pool"][region]

    free_device_num = -1
    for idx, ((cfg_name, special_task_cfg), _enum_params, seed) in enumerate(task_cfgs):
        this_task_cfg = copy.deepcopy(task_cfg)
        this_pool_cfgs = copy.deepcopy(pool_cfgs)
        for k, v in special_task_cfg.items():
            if k in this_task_cfg.keys():
                this_task_cfg[k] = v
            else:
                this_task_cfg["running_args"][k] = v
        this_pool_cfgs["flavorCode"] = this_pool_cfgs["flavorCode"][str(this_task_cfg["GPUs"])]

        desc = os.path.splitext(os.path.basename(cfg_name))[0]
        for k, v in _enum_params.items():
            desc += "_{}_{}".format(k, v)
        desc += "_seed_{}".format(seed)

        this_task_cfg["desc"] = desc
        this_task_cfg["train_url"] = os.path.join(this_task_cfg["base_train_url"], task_time_id, desc) + "/"

        this_task_cfg["running_args"]["args_yml_fn"] = cfg_name
        if "random_seed" not in this_task_cfg["running_args"]:
            this_task_cfg["running_args"]["random_seed"] = seed
        this_task_cfg["running_args"].update(_enum_params)

        if priority == "high":
            pass
        else:
            while True:
                status = get_pool_status(app_name, region, app_token)
                if status is not None:
                    waiting = int(status["waiting"])
                    queue = int(status["queue"])
                    running = int(status["running"])
                    quota = int(status["quota"])
                    if priority == "low":
                        if free_device_num > this_task_cfg["num_nodes"]:
                            free_device_num -= 8 * this_task_cfg["num_nodes"]
                            break
                        else:
                            if waiting == 0 and queue == 0:
                                free_device_num = quota - running
                                if free_device_num >= 8 * this_task_cfg["num_nodes"]:
                                    print_with_time(
                                        "free nodes({}) greater than task nodes need({}), start create".format(
                                            free_device_num // 8,
                                            this_task_cfg["num_nodes"],
                                        )
                                    )
                                    free_device_num -= 8 * this_task_cfg["num_nodes"]
                                    break
                                else:
                                    print_with_time(
                                        "free nodes({}) smaller than task nodes need({}), sleep 900s".format(
                                            free_device_num // 8,
                                            this_task_cfg["num_nodes"],
                                        )
                                    )
                            else:
                                print_with_time(
                                    "find modelarts waiting job num is {}, queue job num is {}, sleep 900s".format(
                                        waiting, queue
                                    )
                                )
                    else:
                        raise NotImplementedError
                else:
                    print_with_time("check modelarts waiting job num failed, sleep 900s")

                time.sleep(900)

        this_job_name = job_name + "_V{:0>4}".format(str(version_strat_idx + idx))
        environments = this_task_cfg.get("environments", None)
        ret = modelarts_job_create_v2(
            this_task_cfg["code_path"],
            this_job_name,
            this_task_cfg["desc"],
            this_task_cfg["running_script"],
            this_task_cfg["running_args"],
            this_task_cfg["train_url"],
            this_task_cfg["user_id"],
            this_task_cfg["num_nodes"],
            app_cfgs,
            region,
            bucket_id,
            this_pool_cfgs,
            pre_version_uid,
            job_priority,
            environments,
        )

        if ret["success"]:
            pre_version_uid = ret["id"]
        else:
            print_with_time("some error happen!!! exit...")
            exit()
        time.sleep(2)

    print_with_time("task train url:{}".format(os.path.join(this_task_cfg["base_train_url"], task_time_id)))


if __name__ == "__main__":
    task_create()
