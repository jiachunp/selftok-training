# -*- coding: utf-8 -*-

import os
import sys
import time
import copy
import random
import argparse
import itertools
import hashlib
import pathlib
import checksumdir
from datetime import datetime

sys.path.append(".")
from mimogpt.utils import read_from_yaml
from tools.modelarts.utils import get_pre_version_cfg_v2, get_pool_status
from tools.modelarts.api import modelarts_job_create_v2, modelarts_job_query_v2


def filehash(file_path):
    return hashlib.md5(pathlib.Path(file_path).read_bytes()).hexdigest()


def print_with_time(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("{}:INFO:{}".format(t, msg))


def get_basic_args(job_cfg, args):
    job_name = job_cfg["job_name"] if "job_name" in job_cfg else args.job_name
    app_name = job_cfg["app_name"] if "app_name" in job_cfg else args.app_name
    bucket = job_cfg["bucket"] if "bucket" in job_cfg else args.bucket
    region = job_cfg["region"] if "region" in job_cfg else args.region
    priority = job_cfg["priority"] if "priority" in job_cfg else args.priority
    code_root = job_cfg["code_root"] if "code_root" in job_cfg else args.code_root
    prefix = job_cfg["prefix"] if "prefix" in job_cfg else args.prefix
    boot_file = job_cfg["boot_file"] if "boot_file" in job_cfg else args.boot_file
    return job_name, app_name, bucket, region, priority, code_root, prefix, boot_file


def task_create(args, unknown):
    ### task_create
    if not os.path.exists("tools/modelarts/modelarts_cfg.yml"):
        print("can not find tools/modelarts/modelarts_cfg.yml, please contact to dengyangyang 00502837!")
        exit()

    # todo, make parameters more convenient for setting
    modelarts_cfg = read_from_yaml("tools/modelarts/modelarts_cfg.yml")
    job_cfg = modelarts_cfg.get("job_cfg", {})
    job_name, app_name, bucket, region, priority, code_root, prefix, boot_file = get_basic_args(job_cfg, args)
    environments = {"USE_MEMARTS": f"{args.use_memarts}"}
    job_name = "{}_{}".format(job_name, prefix)

    ### auto upload code
    if args.auto_upload_code:
        cmd = (
            f"python tools/code_upload.py "
            f"--app {app_name} "
            f"--code_root {code_root} "
            f"--bucket {bucket} "
            f"--region {region}"
        )
        print(f"Begin to run {cmd}")
        os.system(cmd)
        print("finish code upload.")

    # cfg name and special task cfg
    cfgs = job_cfg.get("cfg_names", [])
    if len(cfgs) == 0:
        cfgs = [
            [args.yml, {"num_nodes": args.num_nodes}],
        ]
    user_id = modelarts_cfg["account"]
    print(cfgs)

    enum_params = {}
    for i in unknown:
        i = i[2:].split("=")
        enum_params[i[0]] = [i[1]]
    print("enum_params......", enum_params)

    # default cfg
    task_cfg = {
        "user_id": user_id,
        "code_path": "s3://{}/code/{}/{}/".format(bucket, user_id, code_root),
        "running_script": "startup.py",
        "running_args": {
            "args_yml_fn": None,
            "boot_file": boot_file,
            "debug": args.debug,
            "bucket": bucket,
            "user_id": user_id,
        },
        "base_train_url": "/{}/outputs/{}/{}/{}/".format(bucket, user_id, code_root, job_name),
        "GPUs": 4,  # how many GPUs each node
        "num_nodes": 1,
        "repeat_times": 1,
        "order": "cycle",  # order or cycle
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

    app_cfgs = {"name": app_name, "token": app_token, "vendor": modelarts_cfg["app"][app_name]["vendor"]}
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
        # handle env in here instead of pass environments to api
        if "environments" in this_pool_cfgs:
            # custom image of pytorch 2.0 has its own environments, use update here
            this_pool_cfgs["environments"].update(environments)
        else:
            this_pool_cfgs["environments"] = environments

        desc = os.path.splitext(os.path.basename(cfg_name))[0]
        # for k, v in _enum_params.items():
        #     desc += "_{}_{}".format(k, v)
        desc += "_seed_{}".format(seed)

        this_task_cfg["desc"] = desc
        this_task_cfg["train_url"] = (
            os.path.join(this_task_cfg["base_train_url"], task_time_id, desc).replace("\\", "/") + "/"
        )

        this_task_cfg["running_args"]["args_yml_fn"] = cfg_name
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
                                    free_device_num -= 8 * this_task_cfg["num_nodes"]
                                    break
                                else:
                                    print_with_time(
                                        "free nodes({}) smaller than task nodes need({}), sleep 600s".format(
                                            free_device_num // 8, this_task_cfg["num_nodes"]
                                        )
                                    )
                            else:
                                print_with_time(
                                    "find modelarts waiting job num is {}, queue job num is {}, sleep 600s".format(
                                        waiting, queue
                                    )
                                )
                    else:
                        raise NotImplementedError
                else:
                    print_with_time("check modelarts waiting job num failed, sleep 600s")

                time.sleep(600)

        this_job_name = job_name + "_V{:0>4}".format(str(version_strat_idx + idx))
        print(this_task_cfg)
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
            None,
        )

        if ret["success"]:
            pre_version_uid = ret["id"]
        else:
            print_with_time("some error happen!!! exit...")
            exit()
        time.sleep(2)

    print_with_time("task train url:{}".format(os.path.join(this_task_cfg["base_train_url"], task_time_id)))


def parse_args():
    parser = argparse.ArgumentParser(description="create task on ModelArts")
    parser.add_argument("--job_name", type=str, default="vid_exp", help="job name")
    parser.add_argument("--yml", type=str, default="configs/cloud/vlip_train_test_cloud.yml", help="yml path")
    parser.add_argument("--num_nodes", type=int, default=2, help="1node = 8GPU")
    parser.add_argument("--debug", type=int, default=0, help="use debug mode")
    # default args
    parser.add_argument("--app_name", type=str, default="aigc.team.two", help="com.app.ved.huanan or ved_intern")
    parser.add_argument("--prefix", type=str, default="hn", help="hn or hb")
    parser.add_argument("--bucket", type=str, default="bucket-6824-huanan", help="bucket-6824-huanan or bucket-3010")
    parser.add_argument("--region", type=str, default="cn-south-1", help="cn-south-1 or cn-north-1")
    parser.add_argument("--priority", type=str, default="high", help="task priority")
    parser.add_argument("--use_memarts", type=int, default=0, help="use memarts")
    parser.add_argument("--code_root", type=str, default="MultimediaEngineTrain", help="code exec path")
    parser.add_argument("--boot_file", type=str, default="train_cloud.py", help="code exec path")
    parser.add_argument("--auto_upload_code", type=int, default=1, help="auto_upload_code")

    # some special
    parser.add_argument("--timestamp", type=str, default=None, help="timestamp")

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == "__main__":
    args, unknown = parse_args()
    print(args, unknown)
    task_create(args, unknown)
