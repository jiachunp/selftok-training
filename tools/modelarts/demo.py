import os
from datetime import datetime
import random
import yaml

from .api import modelarts_job_query, modelarts_version_query, modelarts_job_create
from .utils import get_pre_version_cfg


def parse_yaml(fp):
    with open(fp, "r") as fd:
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return y


def task_create():
    app_name = "ved_reid_id_huanan_qw"
    job_name = "resnet50_test"
    bucket = "bucket-d"
    region = "cn-south-1"

    cfgs = [
        "cfgs/baseline/resnet34.yml",
        "cfgs/baseline/resnet50.yml",
    ]

    task_cfg = {
        "user_id": "account",
        "code_path": "s3://{}/code/code_path/".format(bucket),
        "running_script": "training_cloud/train_cloud.py",
        "running_args": {"args_yml_fn": None, "seed": 1},
        "base_train_url": "/{}/outputs/test".format(bucket),
        "data_url": "/dummy",
        "num_nodes": 1,
        "repeat_times": 3,
    }

    modelarts_cfg = parse_yaml("config_template.yml")
    app_token = modelarts_cfg["app"][app_name]["token"]
    find, pre_version_cfg = get_pre_version_cfg(job_name, app_name, app_token)

    if not find:
        print("can not find job name: {}, please check it, now will create new job".format(job_name))
    else:
        print(
            "find job name:{}, pre_version_uid={}, pre_version_id={}".format(
                job_name, pre_version_cfg["pre_version_uid"], pre_version_cfg["pre_version_id"]
            )
        )

    task_time_id = datetime.now().strftime("%Y-%m-%d_time_%H_%M_%S")
    seed_range = range(task_cfg["repeat_times"])
    task_cfgs = [[cfg_name, seed] for cfg_name in cfgs for seed in seed_range]

    app_cfgs = {"name": app_name, "token": app_token, "vendor": modelarts_cfg["app"][app_name]["vendor"]}
    bucket_id = modelarts_cfg["obs"][bucket]["id"]
    pool_cfgs = modelarts_cfg["app"][app_name]["pool"][region]

    for cfg_name, seed in task_cfgs:
        desc = os.path.splitext(os.path.basename(cfg_name))[0]
        if task_cfg["repeat_times"] > 1:
            desc = "{}_seed{}".format(desc, seed)

        task_cfg["desc"] = desc
        task_cfg["train_url"] = os.path.join(task_cfg["base_train_url"], task_time_id, desc) + "/"
        task_cfg["running_args"]["args_yml_fn"] = cfg_name
        task_cfg["running_args"]["seed"] = seed

        modelarts_job_create(
            task_cfg["code_path"],
            job_name,
            task_cfg["desc"],
            task_cfg["running_script"],
            task_cfg["running_args"],
            task_cfg["train_url"],
            task_cfg["data_url"],
            task_cfg["user_id"],
            task_cfg["num_nodes"],
            app_cfgs,
            region,
            bucket_id,
            pool_cfgs,
            pre_version_cfg["pre_version_uid"],
            pre_version_cfg["pre_version_id"],
        )
        if len(pre_version_cfg["pre_version_uid"]) == 0:
            find, pre_version_cfg = get_pre_version_cfg(job_name, app_name, app_token)
            if not find:
                print("some error happened!!!!")
                exit()


if __name__ == "__main__":
    task_create()
