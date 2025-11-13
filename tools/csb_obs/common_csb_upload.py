#!/usr/bin/env python

import glob
import os
import time

# example python common_csb_upload.py --region=cn-south-1 --bucket_dir=bucket-6824-huanan/autoML  --src_dir=XX
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO framework based on PyTorch")
    parser.add_argument("--region", type=str, default="", help="region of bucket [cn-north-1 cn-south-1 cn-north-4]")
    parser.add_argument(
        "--bucket_dir",
        type=str,
        default="",
        help="roma bucket directory [bucket-veddata01/c00500728/yolo-distributed bucket-6824-huanan/c00500728/yolo-distributed bucket-vedata02-bj4/c00500728/yolo-distributed]",
    )
    parser.add_argument("--src_dir", type=str, default="", help="local server src dir")
    args = parser.parse_args()
    if args.bucket_dir == "" or args.region == "" or args.src_dir == "":
        exit(0)

    bucket_name = args.bucket_dir[: args.bucket_dir.find("/")]
    bucket_path = args.bucket_dir[args.bucket_dir.find("/") + 1 :]
    os.system(
        "python tools/csb_obs/yellow_zone_uploader.py --local_folder_absolute_path={} --app_token=c383504d-abc4-4666-94a1-54b2928b61c9 --vendor=HEC --region={} --bucket_name={} --bucket_path={}".format(
            args.src_dir, args.region, bucket_name, bucket_path
        )
    )
