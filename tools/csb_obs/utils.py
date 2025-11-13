import os
import json
import base64
import requests
from urllib import request
import sys
from easydict import EasyDict
from boto3.session import Session


def get_file_server_url(region):
    if region in ["cn-north-1", "cn-south-1", "cn-north-4"]:
        file_server_url = "http://{}y.csbobs.roma.huawei.com:8080/csb-file-server".format(region)
    elif region == "cn-south-222":
        file_server_url = "http://cn-south-222s-1.csbobs.roma.huawei.com/csb-file-server"
    else:
        print("wrong region:{}".format(region))
        exit(0)
    return file_server_url


def get_next_1000_objects(vendor, region, bucket_name, app_token, object_key, next_marker):
    list_objects_endpoint = (
        get_file_server_url(region)
        + "/rest/boto3/s3/list/bucket/objectkeys?vendor="
        + vendor
        + "&region="
        + region
        + "&bucketid="
        + bucket_name
        + "&apptoken="
        + app_token
        + "&objectkey="
        + object_key
        + "&nextmarker="
        + next_marker
    )
    req = request.Request(url=list_objects_endpoint)
    res = request.urlopen(req)
    result = res.read().decode(encoding="utf-8")
    result_dict = json.loads(result)
    return result_dict


def get_ascend_log_folder_list(vendor, region, bucket_name, app_token, path):
    next_marker = ""
    truncated = "true"

    if not path.endswith("/"):
        path = path + "/"

    path = path.replace("\\", "/")
    path = base64.urlsafe_b64encode(path.encode(encoding="utf-8"))
    path = str(path, encoding="utf-8")

    ascend_log_folder = []
    while truncated == "true":
        result_dict = get_next_1000_objects(vendor, region, bucket_name, app_token, path, next_marker)
        if result_dict["success"] == "true":
            objectKeys = [x["objectKey"] for x in result_dict["objectKeys"] if "ascend-log" in x["objectKey"]]
            ascend_log_folder += [os.path.split(x)[0] for x in objectKeys]
            ascend_log_folder = list(set(ascend_log_folder))
        else:
            raise Exception(result_dict["msg"])

        next_marker = result_dict["nextmarker"]
        truncated = result_dict["truncated"]

    return ascend_log_folder
