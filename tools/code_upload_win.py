# -*- coding: utf-8 -*-

import os
import sys
import time
import hashlib
import pathlib
import argparse
import checksumdir

sys.path.append(".")
#from mimogpt.utils import read_from_yaml



import yaml
from easydict import EasyDict

def read_from_yaml(txt_path):
    with open(txt_path, "r", encoding="utf8") as fd:
        # cont = open(fd, encoding="utf8")
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return EasyDict(y)




def filehash(file_path):
    return hashlib.md5(pathlib.Path(file_path).read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create task on ModelArts")
    parser.add_argument("--app", type=str, default="aigc.team.two", help="training group")
    parser.add_argument("--code_root", type=str, default="MultimediaEngineTrain", help="code exec path")
    parser.add_argument("--bucket", type=str, default="bucket-6824-huanan", help="bucket-6824-huanan or bucket-3010")
    parser.add_argument("--region", type=str, default="cn-south-1", help="cn-south-1 or cn-north-1")
    parser.add_argument(
        "--modelarts_cfg_path", type=str, default="tools/modelarts/modelarts_cfg.yml", help="modelarts_cfg_path"
    )
    args = parser.parse_args()

    region = args.region
    bucket = args.bucket

    if not os.path.exists(args.modelarts_cfg_path):
        print("can not find tools/modelarts/modelarts_cfg.yml, please contact to chenminqi!")
        exit()

    modelarts_cfg = read_from_yaml(args.modelarts_cfg_path)
    account = modelarts_cfg["account"]

    # code dir
    bucket_dir = "{}/code/{}/{}/".format(bucket, account, args.code_root)
    if "upload_token" in modelarts_cfg["app"][modelarts_cfg["obs"][bucket]["create_app"]]:
        app_token = modelarts_cfg["app"][modelarts_cfg["obs"][bucket]["create_app"]]["upload_token"]
    else:
        app_token = modelarts_cfg["app"][modelarts_cfg["obs"][bucket]["create_app"]]["token"]

    csb_path = "tools/csb_obs/"
    exclude = "output,.git,ignore"
    md5_tmp_path = f"output/md5_{args.code_root}_{bucket}.txt"
    print(md5_tmp_path)
    current_work_dir = os.getcwd()
    bucket_name = bucket_dir[: bucket_dir.find("/")]
    bucket_path = bucket_dir[bucket_dir.find("/") + 1 :]
    print(f"Upload to {bucket_name}: {bucket_path}")

    exclude = exclude.split(",")
    local_path_list = []
    for name in os.listdir(current_work_dir):
        if name not in exclude:
            local_path_list.append(name)

    if os.path.exists(md5_tmp_path):
        md5_old_all = open(md5_tmp_path).readlines()
        md5_old_all = dict([x.strip().split("::") for x in md5_old_all])
    else:
        md5_old_all = {}
    md5_new_all = {}

    start_t = time.time()
    for local_path in local_path_list:
        local_abs_path = os.path.join(current_work_dir, local_path)
        print("local_abs_path:", local_abs_path)
        if os.path.isdir(local_abs_path):
            md5hash = checksumdir.dirhash(local_abs_path, "md5")
        else:
            md5hash = filehash(local_abs_path)

        md5_old = md5_old_all.get(local_abs_path, "")
        if md5_old == md5hash:
            print("md5 not changed, skip upload...")
        else:
            os.system(
                "python tools/csb_obs/s3_uploader.py "
                "--local_folder_absolute_path={} "
                "--app_token={} "
                "--vendor=HEC "
                "--region={} "
                "--bucket_name={} "
                "--bucket_path={}".format(local_abs_path, app_token, region, bucket_name, bucket_path)
            )
            # --local_folder_absolute_path=/home/lql --app_token=xxx --region=cn-north-1 --bucket_name=bucket-0007 --bucket_path=folder1/ --show_speed
            # os.system(
            #     "python tools/csb_obs/yellow_zone_uploader.py "
            #     "--local_folder_absolute_path={} "
            #     "--app_token={} "
            #     "--vendor=HEC "
            #     "--region={} "
            #     "--bucket_name={} "
            #     "--bucket_path={}".format(
            #         local_abs_path,
            #         app_token,
            #         region,
            #         bucket_name,
            #         bucket_path
            #     )
            # )
        md5_new_all[local_abs_path] = md5hash

    md5_tmp_path_path = os.path.dirname(md5_tmp_path)
    if not os.path.exists(md5_tmp_path_path):
        os.makedirs(md5_tmp_path_path)

    with open(md5_tmp_path, "w") as fn:
        for k, v in md5_new_all.items():
            fn.write("{}::{}\n".format(k, v))

    end_t = time.time()
    print("csb_upload cost time:{}".format(end_t - start_t))
