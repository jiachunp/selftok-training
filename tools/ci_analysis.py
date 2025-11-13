import os
import re
import sys
import time
import argparse
import numpy as np
from datetime import datetime

sys.path.append(".")

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

from mimogpt.utils import read_from_yaml


def print_with_time(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("{}:INFO:{}".format(t, msg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create task on ModelArts")
    parser.add_argument("--code_root", type=str, default="MultimediaEngineTrain", help="code exec path")
    parser.add_argument("--timestamp", type=str, default=None, help="job name")
    parser.add_argument("--filesize_limit", type=int, default=30, help="if over size, don't download, unit: MB")
    parser.add_argument(
        "--modelarts_cfg_path", type=str, default="tools/modelarts/modelarts_cfg.yml", help="modelarts_cfg_path"
    )
    args, unknown = parser.parse_known_args()

    output_path = "output"
    csb_obs_dir = "tools/csb_obs"

    time_step = 300

    if not os.path.exists(args.modelarts_cfg_path):
        print("can not find {}, please contact to hutianyu 00807144".format(args.modelarts_cfg_path))
        exit()
    modelarts_cfg = read_from_yaml(args.modelarts_cfg_path)

    user_id = modelarts_cfg["account"]
    bucket = modelarts_cfg["job_cfg"]["bucket"]
    region = modelarts_cfg["job_cfg"]["region"]
    app_name = modelarts_cfg["job_cfg"]["app_name"]

    app_vendor = modelarts_cfg["app"][app_name]["vendor"]
    app_token = modelarts_cfg["app"][app_name]["token"]
    obs_token = modelarts_cfg["app"][modelarts_cfg["obs"][bucket]["create_app"]]["token"]
    bucket_id = modelarts_cfg["obs"][bucket]["id"]

    # rule same to task_create.py L52
    job_name = modelarts_cfg["job_cfg"]["job_name"]
    prefix = modelarts_cfg["job_cfg"]["prefix"]
    job_name = "{}_{}".format(job_name, prefix)
    # rule same to task_create.py L94
    log_path_s3 = "outputs/{}/{}/{}/".format(user_id, args.code_root, job_name)
    # rule same to task_create.py L165
    log_path_s3 = os.path.join(log_path_s3, args.timestamp)
    if not log_path_s3.endswith("/"):
        log_path_s3 = log_path_s3 + "/"

    download_cmd = "cd {}; python yellow_folder_downloader.py".format(csb_obs_dir)
    download_cmd += " --app_token={} --vendor={}".format(obs_token, app_vendor)
    download_cmd += " --region={} --bucket_name={}".format(region, bucket)
    download_cmd += " --path={}".format(log_path_s3)
    download_cmd += " --objects_storage_path={}".format(os.path.abspath(output_path))
    download_cmd += " --exclude=.txt,.ckpt,.meta,process_log,.pt,ascend,.png,weights"
    download_cmd += " --filesize_limit={}".format(args.filesize_limit)
    download_cmd += " --processes=88"
    # download_cmd += ' 1>/dev/null 2>&1'
    print(download_cmd)

    while True:
        # 1. download log
        os.system(download_cmd)
        end_analysis = False

        # 2. results analysis
        results = {}
        for root, _, files in os.walk(os.path.join(output_path, os.path.split(log_path_s3[:-1])[-1])):
            for name in files:
                if "worker-0.log" in name:
                    logs = open(os.path.join(root, name), "r", encoding="utf8", errors="ignore").readlines()

                    keyword = "[Last metrics]:"
                    last_metric = [x.strip() for x in logs if keyword in x]
                    if len(last_metric) > 0:
                        end_analysis = True
                        last_metric = last_metric[0]
                        last_metric = last_metric[last_metric.index(keyword) :][len(keyword) :].strip()
                        last_metric = [x.split(":") for x in last_metric.split(",")]
                        last_metric = dict([[k.strip(), float(v)] for k, v in last_metric])
                        print_with_time("find last_metric, break")
                        print_with_time(last_metric)
                        break
                    else:
                        print_with_time("maybe still in training, wait {}s".format(time_step))

                    keyword = "training is completed"
                    finished = [x.strip() for x in logs if keyword in x]
                    if len(finished) > 0 and (not end_analysis):
                        end_analysis = True
                        print_with_time("training is completed, but cannot get last_metric, please check it")
                        break

        if end_analysis:
            break

        time.sleep(time_step)
