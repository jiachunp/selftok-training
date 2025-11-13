# encoding: utf-8
# Author: Li Qiliang(l00423096).

import os
import sys
import time
import glob
import copy
import ctypes
import base64
import logging
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from urllib import parse, request
import random
import json
import subprocess

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

from boto3.session import Session

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(
    description="Parameters of CSB-OBS", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--vendor", type=str, default="HEC", help="vendor of bucket.")
parser.add_argument("--region", type=str, default="cn-north-1", help="region of bucket.")
parser.add_argument("--app_token", type=str, default=None, help="appToken of CSB.")
parser.add_argument("--bucket_name", type=str, default=None, help="Please input you bucket_name.")
parser.add_argument("--path", type=str, default="", help="Please input you path(objectKey).")
parser.add_argument("--objects_storage_path", type=str, default=None, help="Output list of objectkeys.")
parser.add_argument("--buffer_size", type=int, default=8192, help="The default value of buffer_size is 8192.")
parser.add_argument("--exclude", type=str, default="", help="exclude, like: .pt,.ckpt,.log,ascend_log")
parser.add_argument("--processes", type=int, default=1, help="multiprocesses")
parser.add_argument("--filesize_limit", type=int, default=-1, help="if over size, don't download, unit: MB")

result, _ = parser.parse_known_args()
args = copy.deepcopy(result)

if args.region in ["cn-north-1", "cn-south-1", "cn-north-4", "cn-east-3", "cn-southwest-2"]:
    file_server_url = "http://{}y.csbobs.roma.huawei.com:8080/csb-file-server".format(args.region)
elif args.region in [
    "cn-south-222",
]:
    file_server_url = "http://cn-south-222s-1.csbobs.roma.huawei.com/csb-file-server"
else:
    print("wrong region:{}".format(args.region))
    exit(0)


def bucket_auth(vendor, region, bucket_name, app_token):
    try:
        if bucket_name is None:
            raise Exception("The --bucket_name can not be null.")

        if app_token is None:
            raise Exception("The --app_token can not be null.")

        bucket_auth_endpoint = (
            file_server_url
            + "/rest/boto3/s3/bucket-auth?vendor="
            + vendor
            + "&region="
            + region
            + "&bucketid="
            + bucket_name
            + "&apptoken="
            + app_token
        )
        req = request.Request(url=bucket_auth_endpoint)
        res = request.urlopen(req)
        result = res.read().decode(encoding="utf-8")
        result_dict = json.loads(result)
        return result_dict["success"], result_dict["msg"]
    except Exception as e:
        sys.stdout.write(str(e) + "\n")


def get_s3_client_list():
    s3_client_list = []
    try:
        csb_obs_service_endpoint = (
            file_server_url + "/rest/boto3/s3/" + args.vendor + "/" + args.region + "/" + args.app_token
        )
        session = Session("Hello", "CSB-OBS")
        s3_client = session.client("s3", endpoint_url=csb_obs_service_endpoint)
        s3_client_list.append(s3_client)
        return s3_client_list
    except Exception as e:
        sys.stdout.write(str(e) + "\n")


def get_next_1000_objects(vendor, region, bucket_name, app_token, object_key, next_marker):
    try:
        list_objects_endpoint = (
            file_server_url
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
    except Exception as e:
        raise Exception(str(e))


def download_object(
    s3_client_list,
    objects_storage_path,
    bucket_name,
    original_path,
    objectKey,
    size=-1,
    failed_count=1,
    selected_index=0,
):
    try:
        if len(original_path) > 0:
            pos = original_path[:-1].rfind("/")
            tracated_objectKey = objectKey[len(original_path[: pos + 1]) :]
        pos = tracated_objectKey.rfind("/")

        local_objects_storage_path = os.path.join(objects_storage_path, tracated_objectKey[: pos + 1])
        if not os.path.exists(local_objects_storage_path):
            os.makedirs(local_objects_storage_path)
        object_name = tracated_objectKey[pos + 1 :]
        local_file = os.path.join(local_objects_storage_path, object_name)

        if os.path.exists(local_file) and os.path.getsize(local_file) == int(size):
            pass  # maybe size is unsafe, md5 is safe
        else:
            if s3_client_list is None:
                s3_client_list = get_s3_client_list()
            wait_select_index_list = [index for index in range(len(s3_client_list))]
            if failed_count != 1 and len(s3_client_list) > 1:
                wait_select_index_list.remove(selected_index)

            random.shuffle(wait_select_index_list)
            selected_index = wait_select_index_list[0]
            s3_client = s3_client_list[selected_index]

            objectKey_base64 = base64.urlsafe_b64encode(objectKey.encode(encoding="utf-8"))
            objectKey_base64 = str(objectKey_base64, encoding="utf-8")
            resp = s3_client.get_object(Bucket=bucket_name, Key=objectKey_base64)

            with open(local_file, "wb") as f:
                file = resp["Body"]
                while True:
                    data = file.read(args.buffer_size)
                    if data == b"":
                        break
                    f.write(data)
                    f.flush()

    except Exception as e:
        if failed_count >= 3:
            raise Exception(str(e))
        else:
            failed_count += 1
            download_object(
                s3_client_list,
                objects_storage_path,
                bucket_name,
                original_path,
                objectKey,
                size=size,
                failed_count=failed_count,
                selected_index=selected_index,
            )


def download_object_wrap(args):
    return download_object(*args)


def track_job(job, tot, download_path, total_size, update_interval=5):
    t0 = time.time()
    total_size = total_size / 1024.0
    folder_size = 0
    while job._number_left > 0:
        if os.path.exists(download_path):
            ret = subprocess.check_output("du -s {}".format(download_path), shell=True)
            folder_size = int(ret.split()[0])
        time_used = time.time() - t0
        out = "tasks remaining = {}, tot={}, time_used={}s".format(
            min(job._number_left * job._chunksize, tot), tot, int(time_used)
        )
        if folder_size > 0 and time_used > 10:
            speed = folder_size / 1024.0 / time_used
            out += ", speed={:.2f}M/s".format(speed)
            out += ", remaining time={}s".format(max(0, int((total_size - folder_size) / 1024.0 / speed)))
            out += ", progress={:.1f}G/{:.1f}G".format(folder_size / 1024 / 1024, total_size / 1024 / 1024)
        print("\r\033[K" + out, end="", flush=True)
        time.sleep(update_interval)


def iterate_download_objects(
    s3_client_list,
    result_dict,
    objects_storage_path,
    bucket_name,
    original_path,
    object_key_count,
    download_type="folder",
):
    try:
        if download_type == "folder":
            if args.processes > 1:
                downloads = []
                for obsObjectKeyDo in result_dict["objectKeys"]:
                    objectKey = obsObjectKeyDo["objectKey"]
                    size = obsObjectKeyDo["size"]
                    downloads.append([None, objects_storage_path, bucket_name, original_path, objectKey, size])

                random.shuffle(downloads)
                path_save = os.path.join(objects_storage_path, os.path.split(original_path[:-1])[-1])
                total_size = sum([int(x[-1]) for x in downloads])

                p = multiprocessing.Pool(args.processes)
                results = p.map_async(download_object_wrap, downloads)
                track_job(results, len(downloads), path_save, total_size)
                p.close()
                p.join()
            else:
                for obsObjectKeyDo in result_dict["objectKeys"]:

                    objectKey = obsObjectKeyDo["objectKey"]
                    size = obsObjectKeyDo["size"]
                    lastModifyTime = obsObjectKeyDo["lastModifyTime"]
                    storageType = obsObjectKeyDo["storageType"]
                    download_object(s3_client_list, objects_storage_path, bucket_name, original_path, objectKey, size)

                    object_key_count.value += 1
                    sys.stdout.write("The number of downloaded files is " + str(object_key_count.value) + ".\n")

        else:
            download_object(s3_client_list, objects_storage_path, bucket_name, original_path, original_path)

            object_key_count.value += 1
            sys.stdout.write("The number of downloaded files is " + str(object_key_count.value) + ".\n")

    except Exception as e:
        raise Exception(str(e))


def main():
    try:
        vendor = args.vendor
        region = args.region
        bucket_name = args.bucket_name
        app_token = args.app_token
        next_marker = ""
        truncated = "true"

        path = args.path
        if path is None or len(path) == 0:
            raise Exception("The --path can not be null.")

        path = path.replace("\\", "/")
        # if not path.endswith('/'):
        #  path += '/'

        original_path = path
        if path is not None:
            path = base64.urlsafe_b64encode(path.encode(encoding="utf-8"))
            path = str(path, encoding="utf-8")

        if bucket_name is None:
            raise Exception("The --bucket_name can not be null.")

        if app_token is None:
            raise Exception("The --app_token can not be null.")

        # First, bucket auth
        is_success, msg = bucket_auth(vendor, region, bucket_name, app_token)
        if not is_success:
            raise Exception(msg)

        objects_storage_path = args.objects_storage_path
        if objects_storage_path is None:
            dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
            objects_storage_path = dirname
        # else:
        #  objects_storage_path = objects_storage_path.replace('\\', '/')

        s3_client_list = get_s3_client_list()

        manager = multiprocessing.Manager()
        object_key_count = manager.Value(ctypes.c_longdouble, 0, lock=True)

        # Iteratively download a folder.
        if original_path.endswith("/"):
            result_dict_all = {"objectKeys": []}
            while truncated == "true":
                result_dict = get_next_1000_objects(vendor, region, bucket_name, app_token, path, next_marker)

                if result_dict["success"] == "true":
                    result_dict_all["objectKeys"] += result_dict["objectKeys"]
                else:
                    raise Exception(result_dict["msg"])

                next_marker = result_dict["nextmarker"]
                truncated = result_dict["truncated"]

            print_str = ""
            origin_num = len(result_dict_all["objectKeys"])
            if len(args.exclude) > 0:
                exclude = args.exclude.split(",")
                suffix = [x for x in exclude if x.startswith(".")]
                keyword = [x for x in exclude if not x.startswith(".")]

                def filter_fn(name):
                    need_download_flag1 = not any([name.endswith(x) for x in suffix])
                    name_split = name.split("/")
                    need_download_flag2 = not any([x in keyword for x in name_split])
                    return need_download_flag1 and need_download_flag2

                todo_downloads = [x for x in result_dict_all["objectKeys"] if filter_fn(x["objectKey"])]

                if args.filesize_limit > 0:

                    def filter_fn(siz):
                        return int(siz) < args.filesize_limit * 1024 * 1024

                    todo_downloads = [x for x in todo_downloads if filter_fn(x["size"])]

                download_num = len(todo_downloads)
                print_str += "exclude={}, origin:{}, exclude:{}".format(
                    args.exclude, origin_num, origin_num - download_num
                )
                result_dict_all["objectKeys"] = todo_downloads
            else:
                print_str += "origin:{}".format(origin_num)

            def get_local_file(objectKey):
                if len(original_path) > 0:
                    pos = original_path[:-1].rfind("/")
                    tracated_objectKey = objectKey[len(original_path[: pos + 1]) :]
                pos = tracated_objectKey.rfind("/")

                local_objects_storage_path = os.path.join(objects_storage_path, tracated_objectKey[: pos + 1])
                if not os.path.exists(local_objects_storage_path):
                    os.makedirs(local_objects_storage_path)
                object_name = tracated_objectKey[pos + 1 :]
                local_file = os.path.join(local_objects_storage_path, object_name)
                return local_file

            todo_downloads = []
            for objectKey in result_dict_all["objectKeys"]:
                src = objectKey["objectKey"]
                dst = get_local_file(src)
                size = int(objectKey["size"])
                if os.path.exists(dst) and os.path.getsize(dst) == size:
                    pass
                else:
                    todo_downloads.append(objectKey)

            origin_num = len(result_dict_all["objectKeys"])
            result_dict_all["objectKeys"] = todo_downloads
            print_str += ", done:{}, todo:{}, begin downloading, please waiting...".format(
                origin_num - len(todo_downloads), len(todo_downloads)
            )
            print(print_str)

            iterate_download_objects(
                s3_client_list,
                result_dict_all,
                objects_storage_path,
                bucket_name,
                original_path,
                object_key_count,
                "folder",
            )

        # download a file.
        else:
            result_dict = []
            iterate_download_objects(
                s3_client_list, result_dict, objects_storage_path, bucket_name, original_path, object_key_count, "file"
            )

        sys.stdout.write("\nend." + "\n")

    except Exception as e:
        sys.stdout.write(str(e) + "\n")


if __name__ == "__main__":
    main()
