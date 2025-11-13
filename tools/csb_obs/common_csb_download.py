#!/usr/bin/env python

import glob
import os
import time

# example python common_csb_upload.py --region=cn-south-1 --bucket_dir=bucket-6824-huanan/autoML  --dist_dir=XX
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
    parser.add_argument("--dist_dir", type=str, default="", help="local server dist dir")
    args = parser.parse_args()
    print("args:", args)
    if args.bucket_dir == "" or args.region == "" or args.dist_dir == "":
        exit(0)

    bucket_name = args.bucket_dir[: args.bucket_dir.find("/")]
    bucket_path = args.bucket_dir[args.bucket_dir.find("/") + 1 :]
    print("bucket_name:{}, bucket_path:{}".format(bucket_name, bucket_path))
    start_t = time.time()

    # python yellow_folder_downloader.py --objects_storage_path=. --app_token=bd43698f-f400-4f2a-8a09-a40238cb6607 --vendor=HEC --region=cn-south-1 --bucket_name=bucket-6824-huanan --path=c00451331/mirror/yolo-distributed/evo_outputs_3nodes/
    os.system(
        "python yellow_folder_downloader.py --objects_storage_path={} --app_token=c383504d-abc4-4666-94a1-54b2928b61c9 --vendor=HEC --region={} --bucket_name={} --path={}".format(
            args.dist_dir, args.region, bucket_name, bucket_path
        )
    )
    # print('python yellow_folder_downloader.py --objects_storage_path={} --app_token=bd43698f-f400-4f2a-8a09-a40238cb6607 --vendor=HEC --region={} --bucket_name={} --path={}'.format(args.dist_dir, args.region, bucket_name, bucket_path))
    end_t = time.time()
    print("csb_upload cost time:{}".format(end_t - start_t))
