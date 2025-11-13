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
import codecs

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

from boto3.session import Session

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(
    description="Parameters of CSB-OBS", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--local_folder_absolute_path", type=str, default=None, help="Local folder to upload")
parser.add_argument("--vendor", type=str, default="HEC", help="vendor of bucket")
parser.add_argument("--region", type=str, default="cn-north-1", help="region of bucket")
parser.add_argument("--app_token", type=str, default=None, help="appToken of CSB")
parser.add_argument("--bucket_name", type=str, default=None, help="Please input you bucket_name")
parser.add_argument("--bucket_path", type=str, default="", help="Please input you bucket_path")
parser.add_argument(
    "--failed_list_file_name", type=str, default=None, help="Output list of files which are faied to upload"
)
parser.add_argument("--failed_list_absobute_path", type=str, default=None, help="Reupload failed files")
parser.add_argument("--small_file_thread", type=int, default=100, help="he default value of small_file_thread is 100")
parser.add_argument("--large_file_thread", type=int, default=10, help="he default value of large_file_thread is 10")
parser.add_argument("--part_file_thread", type=int, default=5, help="he default value of part_file_thread is 5")
parser.add_argument(
    "--samll_file_size", type=int, default=100 * 1024 * 1024, help="The default value of samll_file_size is 100MB."
)
parser.add_argument("--part_size", type=int, default=200 * 1024 * 1024, help="The default value of part_size is 200MB.")

result, _ = parser.parse_known_args()
args = copy.deepcopy(result)

file_server_url = "http://10.155.175.112:8080/csb-file-server"
# file_server_url = 'http://127.0.0.1:8080/csb-file-server'


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
        query_urls_endpoint = (
            file_server_url
            + "/rest/boto3/s3/query/csb-file-server/all/ip-and-port?vendor="
            + args.vendor
            + "&region="
            + args.region
        )
        req = request.Request(url=query_urls_endpoint)
        res = request.urlopen(req)
        result = res.read().decode(encoding="utf-8")
        result_dict = json.loads(result)
        if result_dict["fileServerUrlList"] is not None:
            for csb_file_server_url in result_dict["fileServerUrlList"]:
                csb_obs_service_endpoint = (
                    csb_file_server_url + "/rest/boto3/s3/" + args.vendor + "/" + args.region + "/" + args.app_token
                )
                session = Session("Hello", "CSB-OBS")
                s3_client = session.client("s3", endpoint_url=csb_obs_service_endpoint)
                s3_client_list.append(s3_client)
        return s3_client_list
    except Exception as e:
        sys.stdout.write(str(e) + "\n")


def print_progress_bar(total_num, finished_num, cur_count, print_lock):
    with print_lock:
        finished_num.value += cur_count
        finished_percent = finished_num.value / total_num.value
        sys.stdout.write(
            "|"
            + ("-" * int(50 * finished_percent))
            + (" " * int(50 * (1 - finished_percent)))
            + "| %.2f%%\n" % (finished_percent * 100)
        )
        sys.stdout.flush()


def alter_global_small_file_thread_count(global_small_file_thread_count, global_small_file_thread_count_lock, step):
    with global_small_file_thread_count_lock:
        global_small_file_thread_count.value += step


def alter_global_large_file_thread_count(global_large_file_thread_count, global_large_file_thread_count_lock, step):
    with global_large_file_thread_count_lock:
        global_large_file_thread_count.value += step


def alter_global_part_thread_count(global_part_thread_count, global_part_thread_count_lock, step):
    with global_part_thread_count_lock:
        global_part_thread_count.value += step


def record_failed_file(failed_list_file_path, local_folder_absolute_path, file_name, print_lock):
    with print_lock:
        with open(failed_list_file_path, "a") as f:
            failed_file_name = os.path.join(local_folder_absolute_path, file_name)
            failed_file_name = failed_file_name.replace("\\", "/")
            f.write(failed_file_name + "\n")


def upload_small_file(
    s3_client_list,
    local_folder_absolute_path,
    failed_list_file_path,
    bucket_name,
    bucket_path,
    file_name,
    total_num,
    finished_num,
    cur_file_size,
    global_small_file_thread_count,
    print_lock,
    global_small_file_thread_count_lock,
    failed_count=1,
    selected_index=0,
):
    try:
        wait_select_index_list = [index for index in range(len(s3_client_list))]
        if failed_count != 1 and len(s3_client_list) > 1:
            wait_select_index_list.remove(selected_index)

        random.shuffle(wait_select_index_list)
        selected_index = wait_select_index_list[0]
        s3_client = s3_client_list[selected_index]

        key = bucket_path + file_name
        key = base64.urlsafe_b64encode(key.encode(encoding="utf-8"))
        key = str(key, encoding="utf-8")

        with open(os.path.join(local_folder_absolute_path, file_name), "rb") as file:
            resp = s3_client.put_object(Bucket=bucket_name, Key=key, Body=file.read())

        # sys.stdout.write(str(s3_client) + '  success single' + '\n')

        alter_global_small_file_thread_count(global_small_file_thread_count, global_small_file_thread_count_lock, -1)
        # global_small_file_thread_count.value -= 1

        # Accumulate the amount of uploaded data
        print_progress_bar(total_num, finished_num, cur_file_size, print_lock)

    except Exception as e:
        if failed_count >= 3:
            sys.stdout.write(str(e) + "\n")
            alter_global_small_file_thread_count(
                global_small_file_thread_count, global_small_file_thread_count_lock, -1
            )
            # global_small_file_thread_count.value -= 1
            record_failed_file(failed_list_file_path, local_folder_absolute_path, file_name, print_lock)
        else:
            # sys.stdout.write(str(s3_client) + '  failed single' + '\n')
            failed_count += 1
            upload_small_file(
                s3_client_list,
                local_folder_absolute_path,
                failed_list_file_path,
                bucket_name,
                bucket_path,
                file_name,
                total_num,
                finished_num,
                cur_file_size,
                global_small_file_thread_count,
                print_lock,
                global_small_file_thread_count_lock,
                failed_count=failed_count,
                selected_index=selected_index,
            )


def upload_large_file(
    s3_client_list,
    local_folder_absolute_path,
    failed_list_file_path,
    bucket_name,
    bucket_path,
    file_name,
    total_num,
    finished_num,
    global_large_file_thread_count,
    global_part_thread_count,
    print_lock,
    global_large_file_thread_count_lock,
    global_part_thread_count_lock,
    failed_count=1,
    selected_index=0,
):
    try:
        wait_select_index_list = [index for index in range(len(s3_client_list))]
        if failed_count != 1 and len(s3_client_list) > 1:
            wait_select_index_list.remove(selected_index)

        random.shuffle(wait_select_index_list)
        selected_index = wait_select_index_list[0]
        s3_client = s3_client_list[selected_index]

        key = bucket_path + file_name
        key = base64.urlsafe_b64encode(key.encode(encoding="utf-8"))
        key = str(key, encoding="utf-8")
        mpu = s3_client.create_multipart_upload(Bucket=bucket_name, Key=key)

        # sys.stdout.write(str(s3_client) + '  success init' + '\n')

        part_dict = multiprocessing.Manager().dict()  # The main process shares the dict with sub threads.

        threadPoolExecutor = ThreadPoolExecutor(args.part_file_thread)
        with open(os.path.join(local_folder_absolute_path, file_name), "rb") as file:
            i = 1
            while 1:
                if global_part_thread_count.value >= args.part_file_thread * 2:
                    seconds = 0.5 + round(random.uniform(0, 1), 2)  # wait: 0.5-1.5 seconds
                    time.sleep(seconds)
                    continue

                # sys.stdout.write(str(i) + '\n')
                data = file.read(args.part_size)
                if data == b"":
                    break
                alter_global_part_thread_count(global_part_thread_count, global_part_thread_count_lock, 1)
                # global_part_thread_count.value += 1
                threadPoolExecutor.submit(
                    upload_part,
                    *(
                        s3_client_list,
                        local_folder_absolute_path,
                        failed_list_file_path,
                        bucket_name,
                        key,
                        total_num,
                        finished_num,
                        mpu["UploadId"],
                        i,
                        data,
                        part_dict,
                        global_part_thread_count,
                        print_lock,
                        global_part_thread_count_lock,
                    )
                )
                i += 1

        threadPoolExecutor.shutdown(wait=True)

        part_info = {"Parts": []}
        for PartNumber, ETag in part_dict.items():
            part_info["Parts"].append({"PartNumber": PartNumber, "ETag": ETag})

        # sys.stdout.write(str(part_info) + '\n')

        complete_multipart_upload(
            s3_client_list,
            local_folder_absolute_path,
            failed_list_file_path,
            file_name,
            bucket_name,
            key,
            mpu["UploadId"],
            part_info,
            global_large_file_thread_count,
            print_lock,
            global_large_file_thread_count_lock,
        )

    except Exception as e:
        if failed_count >= 3:
            sys.stdout.write(str(e) + "\n")
            alter_global_large_file_thread_count(
                global_large_file_thread_count, global_large_file_thread_count_lock, -1
            )
            # global_large_file_thread_count.value -= 1
            record_failed_file(failed_list_file_path, local_folder_absolute_path, file_name, print_lock)
        else:
            # sys.stdout.write(str(s3_client) + '  failded init' + '\n')
            failed_count += 1
            upload_large_file(
                s3_client_list,
                local_folder_absolute_path,
                failed_list_file_path,
                bucket_name,
                bucket_path,
                file_name,
                total_num,
                finished_num,
                global_large_file_thread_count,
                global_part_thread_count,
                print_lock,
                global_large_file_thread_count_lock,
                global_part_thread_count_lock,
                failed_count=failed_count,
                selected_index=selected_index,
            )


def upload_part(
    s3_client_list,
    local_folder_absolute_path,
    failed_list_file_path,
    bucket_name,
    key,
    total_num,
    finished_num,
    upload_id,
    i,
    data,
    part_dict,
    global_part_thread_count,
    print_lock,
    global_part_thread_count_lock,
    failed_count=1,
    selected_index=0,
):
    try:
        wait_select_index_list = [index for index in range(len(s3_client_list))]
        if failed_count != 1 and len(s3_client_list) > 1:
            wait_select_index_list.remove(selected_index)

        random.shuffle(wait_select_index_list)
        selected_index = wait_select_index_list[0]
        s3_client = s3_client_list[selected_index]

        response = s3_client.upload_part(Bucket=bucket_name, Key=key, PartNumber=i, UploadId=upload_id, Body=data)

        part_dict[i] = response["ResponseMetadata"]["HTTPHeaders"]["etag"]

        alter_global_part_thread_count(global_part_thread_count, global_part_thread_count_lock, -1)
        # global_part_thread_count.value -= 1

        # Accumulate the amount of uploaded data
        print_progress_bar(total_num, finished_num, len(data), print_lock)

        # sys.stdout.write(str(s3_client) + '  success part' + '\n')

        if response["ResponseMetadata"]["HTTPStatusCode"] >= 300:
            raise IOError("Failed to upload.")

    except Exception as e:
        if failed_count >= 3:
            sys.stdout.write(str(e) + "\n")
            alter_global_part_thread_count(global_part_thread_count, global_part_thread_count_lock, -1)
            # global_part_thread_count.value -= 1
            record_failed_file(failed_list_file_path, local_folder_absolute_path, file_name, print_lock)
        else:
            # sys.stdout.write(str(s3_client) + '  failed part' + '\n')
            failed_count += 1
            upload_part(
                s3_client_list,
                local_folder_absolute_path,
                failed_list_file_path,
                bucket_name,
                key,
                total_num,
                finished_num,
                upload_id,
                i,
                data,
                part_dict,
                global_part_thread_count,
                print_lock,
                global_part_thread_count_lock,
                failed_count=failed_count,
                selected_index=selected_index,
            )


def complete_multipart_upload(
    s3_client_list,
    local_folder_absolute_path,
    failed_list_file_path,
    file_name,
    bucket_name,
    key,
    upload_id,
    part_info,
    global_large_file_thread_count,
    print_lock,
    global_large_file_thread_count_lock,
    failed_count=1,
    selected_index=0,
):
    try:
        wait_select_index_list = [index for index in range(len(s3_client_list))]
        if failed_count != 1 and len(s3_client_list) > 1:
            wait_select_index_list.remove(selected_index)

        random.shuffle(wait_select_index_list)
        selected_index = wait_select_index_list[0]
        s3_client = s3_client_list[selected_index]

        s3_client.complete_multipart_upload(Bucket=bucket_name, Key=key, UploadId=upload_id, MultipartUpload=part_info)

        alter_global_large_file_thread_count(global_large_file_thread_count, global_large_file_thread_count_lock, -1)
        # global_large_file_thread_count.value -= 1

        # sys.stdout.write(str(s3_client) + '  success complete_multipart' + '\n')

    except Exception as e:
        if failed_count >= 3:
            sys.stdout.write(str(e) + "\n")
            alter_global_large_file_thread_count(
                global_large_file_thread_count, global_large_file_thread_count_lock, -1
            )
            # global_large_file_thread_count.value -= 1
            record_failed_file(failed_list_file_path, local_folder_absolute_path, file_name, print_lock)
        else:
            # sys.stdout.write(str(s3_client) + '  failed complete_multipart' + '\n')
            failed_count += 1
            complete_multipart_upload(
                s3_client_list,
                local_folder_absolute_path,
                failed_list_file_path,
                file_name,
                bucket_name,
                key,
                upload_id,
                part_info,
                global_large_file_thread_count,
                print_lock,
                global_large_file_thread_count_lock,
                failed_count=failed_count,
                selected_index=selected_index,
            )


def main():
    try:
        start_time = time.time()

        if args.local_folder_absolute_path is None or len(args.local_folder_absolute_path) == 0:
            raise Exception("The --local_folder_absolute_path can not be null.")

        local_folder_absolute_path = args.local_folder_absolute_path.replace("\\", "/")
        is_folder = False
        if os.path.isdir(local_folder_absolute_path):
            is_folder = True
            if not local_folder_absolute_path.endswith("/"):
                local_folder_absolute_path = local_folder_absolute_path + "/"

        bucket_name = args.bucket_name
        bucket_path = args.bucket_path.replace("\\", "/")
        if not args.bucket_path.endswith("/"):
            bucket_path = args.bucket_path + "/"

        single_file_path = None
        single_file_name = None
        if is_folder and args.failed_list_absobute_path is None:
            temp = local_folder_absolute_path[:-1]
            pos = temp.rfind("/")
            if pos != -1:
                bucket_path = bucket_path + temp[pos + 1 :] + "/"
        else:
            pos = local_folder_absolute_path.rfind("/")
            single_file_path = local_folder_absolute_path
            single_file_name = local_folder_absolute_path[pos + 1 :]
            local_folder_absolute_path = local_folder_absolute_path[: pos + 1]

        # First, bucket auth
        is_success, msg = bucket_auth(args.vendor, args.region, bucket_name, args.app_token)
        if not is_success:
            raise Exception(msg)

        failed_list_file_path = ""
        if args.failed_list_file_name is None:
            failed_list_file_path = os.path.join(local_folder_absolute_path, "failed_list.log")
        else:
            failed_list_file_path = os.path.join(local_folder_absolute_path, args.failed_list_file_name)
        failed_list_file_path = failed_list_file_path.replace("\\", "/")

        file_list = []
        if args.failed_list_absobute_path is not None:
            with codecs.open(args.failed_list_absobute_path, "r") as f:
                file_list = [s.strip() for s in f.readlines()]

            if os.path.exists(failed_list_file_path):
                os.remove(failed_list_file_path)
        elif is_folder:
            if os.path.exists(failed_list_file_path):
                os.remove(failed_list_file_path)

            sys.stdout.write("Please wait, recursively traverse all files under you folder..." + "\n")
            file_list = glob.glob(os.path.join(local_folder_absolute_path, "**", "*"), recursive=True)
        elif not is_folder:
            file_list.append(single_file_path)

            if os.path.exists(failed_list_file_path):
                os.remove(failed_list_file_path)

        if args.failed_list_absobute_path is not None:
            temp = local_folder_absolute_path[:-1]
            pos = temp.rfind("/")
            if pos != -1:
                local_folder_absolute_path = temp[: pos + 1]

        print_lock = multiprocessing.Lock()
        threadPoolExecutor = ThreadPoolExecutor(args.small_file_thread)

        manager = multiprocessing.Manager()
        finished_num = manager.Value(ctypes.c_longdouble, 0, lock=True)
        total_num = manager.Value(ctypes.c_longdouble, 0, lock=False)

        global_small_file_thread_count_lock = multiprocessing.Lock()
        global_small_file_thread_count = manager.Value(ctypes.c_int, 0, lock=True)

        global_large_file_thread_count_lock = multiprocessing.Lock()
        global_large_file_thread_count = manager.Value(ctypes.c_int, 0, lock=True)
        global_part_thread_count_lock = multiprocessing.Lock()
        global_part_thread_count = manager.Value(ctypes.c_int, 0, lock=True)

        s3_client_list = get_s3_client_list()

        # Statistics the total size of all files
        sys.stdout.write("Please wait, Calculating the total file size you need to upload..." + "\n")
        for file_path in file_list:
            if os.path.isdir(file_path):
                continue
            total_num.value += os.path.getsize(file_path)

        sys.stdout.write("There are " + str(total_num.value / 1024) + "KB data to upload." + "\n")
        sys.stdout.write("Please wait, uploading..." + "\n")
        sys.stdout.flush()

        file_list_len = len(file_list)
        i = 0
        while i < file_list_len:
            file_path = file_list[i]

            i += 1

            if os.path.isdir(file_path):
                continue

            file_name = os.path.relpath(file_path, local_folder_absolute_path)
            file_name = file_name.replace("\\", "/")
            # sys.stdout.write(file_name + '\n')
            cur_file_size = os.path.getsize(os.path.join(local_folder_absolute_path, file_name))
            # upload small file
            if cur_file_size <= args.samll_file_size:
                if global_small_file_thread_count.value >= args.small_file_thread:
                    seconds = round(random.uniform(0, 1), 2)  # wait: 0-1 seconds
                    time.sleep(seconds)
                    i -= 1
                    continue
                alter_global_small_file_thread_count(
                    global_small_file_thread_count, global_small_file_thread_count_lock, 1
                )
                # global_small_file_thread_count.value += 1
                threadPoolExecutor.submit(
                    upload_small_file,
                    *(
                        s3_client_list,
                        local_folder_absolute_path,
                        failed_list_file_path,
                        bucket_name,
                        bucket_path,
                        file_name,
                        total_num,
                        finished_num,
                        cur_file_size,
                        global_small_file_thread_count,
                        print_lock,
                        global_small_file_thread_count_lock,
                    )
                )
            # upload large file
            else:
                if global_large_file_thread_count.value >= args.large_file_thread:
                    seconds = 1 + round(random.uniform(0, 1), 2)  # wait: 1-2 seconds
                    time.sleep(seconds)
                    i -= 1
                    continue
                alter_global_large_file_thread_count(
                    global_large_file_thread_count, global_large_file_thread_count_lock, 1
                )
                # global_large_file_thread_count.value += 1
                threadPoolExecutor.submit(
                    upload_large_file,
                    *(
                        s3_client_list,
                        local_folder_absolute_path,
                        failed_list_file_path,
                        bucket_name,
                        bucket_path,
                        file_name,
                        total_num,
                        finished_num,
                        global_large_file_thread_count,
                        global_part_thread_count,
                        print_lock,
                        global_large_file_thread_count_lock,
                        global_part_thread_count_lock,
                    )
                )

        threadPoolExecutor.shutdown(wait=True)

        sys.stdout.write("Total time used: %.2f seconds.\n" % (time.time() - start_time))

        if os.path.exists(failed_list_file_path):
            logging.error("Some files failed to upload, see %s\n" % failed_list_file_path)

    except Exception as e:
        sys.stdout.write(str(e) + "\n")


if __name__ == "__main__":
    main()
