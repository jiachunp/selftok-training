# encoding: utf-8
# Author: Tao yuheng(t50018193).
import argparse
import base64
import copy
import json
import logging
import os
import platform
import queue
import ssl
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib import request

import requests

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

logging.basicConfig(
    level=logging.ERROR, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

endpoint = "https://roma.huawei.com"
api = "/csb/rest/s3/bucket/endpoint"

parser = argparse.ArgumentParser(
    description="Parameters of CSB-OBS", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--show_speed", action="store_true", help="To display the current download speed, <psutil> is required"
)
parser.add_argument(
    "--single_file", action="store_true", help="if you download single file please add '--single_file' behind your cmd"
)
parser.add_argument("--bucket_path", type=str, required=True, help="The default value of bucket path")
parser.add_argument("--region", type=str, required=True, help="region of bucket.")
parser.add_argument("--bucket_name", type=str, required=True, help="Please input you bucket_name.")
parser.add_argument("--app_token", type=str, required=True, help="appToken of CSB.")
parser.add_argument("--objects_storage_path", type=str, required=True, help="Output list of objectkeys.")
parser.add_argument("--buffer_size", type=int, default=65536, help="The default value of buffer_size is 65536.")
parser.add_argument("--retry_times", type=int, default=3, help="The default value of retry times.")
parser.add_argument("--big_file", type=int, default=100 * 1024 * 1024, help="The default value of big file is 100M.")
parser.add_argument("--thread_num", type=int, default=12, help="The default value of thread.")
parser.add_argument("--package_size", type=int, default=50 * 1024 * 1024, help="The default value of package is 50M.")
parser.add_argument("--vendor", type=str, default="HEC", help="vendor of bucket.")
parser.add_argument("--queue_size", type=int, default=50000, help="max value of download_queue")
parser.add_argument(
    "--fail_json_storage_path", type=str, default="./fail.json", help="The default value of fail.json storage path"
)
parser.add_argument(
    "--time_wait", type=int, default=10, help="The default value of download thread waiting time (second)"
)
parser.add_argument("--debug", action="store_true", help="open debug mode to get more system output")

result, _ = parser.parse_known_args()
args = copy.deepcopy(result)
context = ssl._create_unverified_context()

if not args.single_file:
    assert args.bucket_path.endswith("/"), "bucket_path must ends with /"
    args.father_path = args.bucket_path.split("/")[-2] + "/"
else:
    args.father_path = args.bucket_path[args.bucket_path.rfind("/") + 1 :]

if args.father_path == "/":
    args.father_path = ""

print_lock = threading.Lock()
file_lock = threading.Lock()
download_queue = queue.Queue(args.queue_size)
big_file_download_queue = queue.Queue()
range_queue = queue.Queue()
big_file_ls = []
file_count = 0
downloaded_count = 0
big_file_process_count = 0
big_file_process_total = 0
error_count = 0

current_speed = ""
start_count_speed = True

has_next = True

retry_map = defaultdict(int)
error_map = dict()
next_marker = ""

session = requests.Session()


# 0. get endpoint
def get_file_server_endpoint():
    try:
        url = endpoint + api
        param = (
            "?"
            + "bucketid="
            + args.bucket_name
            + "&token="
            + args.app_token
            + "&vendor="
            + args.vendor
            + "&region="
            + args.region
        )
        req = request.Request(url=url + param)
        res = request.urlopen(req, context=context)
        result = res.read().decode(encoding="utf-8")
        if args.debug:
            print("request file server endpoint result: " + result)
        result_dict = json.loads(result)
        if result_dict["success"]:
            return result_dict["result"]
        else:
            print(result_dict["msg"])
            raise Exception("get endpoint failed")
    except Exception as e:
        import traceback

        traceback.print_exc()


# 1. bucket_auth
def bucket_auth():
    bucket_auth_endpoint = (
        args.csb_file_server
        + "/rest/boto3/s3/bucket-auth?vendor="
        + args.vendor
        + "&region="
        + args.region
        + "&bucketid="
        + args.bucket_name
        + "&apptoken="
        + args.app_token
    )
    try:
        req = request.Request(url=bucket_auth_endpoint)
        res = request.urlopen(req)
        result = res.read().decode(encoding="utf-8")
        if args.debug:
            print("bucket auth result: " + result)
        result_dict = json.loads(result)
        if not result_dict["success"]:
            raise Exception(result_dict["msg"])
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("endpoint is", bucket_auth_endpoint)
        raise Exception("bucket_auth error")


def human_size(size, dot=3):
    return str(round(size / pow(1024, 3), dot)) + " GB"


# 2. get objectKey
def get_object_key(next_marker=""):
    global has_next
    global file_count
    path = "/rest/boto3/s3/list/bucket/objectkeys?"
    object_key = base64.urlsafe_b64encode(args.bucket_path.encode()).decode()
    param = (
        f"vendor={args.vendor}&region={args.region}&bucketid={args.bucket_name}"
        f"&apptoken={args.app_token}&objectkey={object_key}&nextmarker={next_marker}"
    )
    headers = {"Content-Type": "application/json", "csb-token": args.app_token}
    result = session.get(args.csb_file_server + path + param, headers=headers)
    result_dict = json.loads(result.content)

    if not result_dict["nextmarker"]:
        has_next = False
    if not result_dict["success"]:
        logging.error(result.text)
        error_map["next_marker" + "@" + next_marker] = result.text
        raise Exception("get object key failed")
    file_count += len(result_dict["objectKeys"])
    for object_key in result_dict["objectKeys"]:
        if int(object_key["size"]) > args.big_file:
            big_file_download_queue.put(object_key)
            file_count -= 1
        else:
            download_queue.put(object_key)
    return result_dict


def do_download_queue():
    global file_count
    result_dict = get_object_key()
    while result_dict["nextmarker"]:
        try:
            result_dict = get_object_key(result_dict["nextmarker"])
        except Exception:
            error_map[result_dict["nextmarker"]] = time.time()
            time.sleep(5)
            result_dict = get_object_key(result_dict["nextmarker"])


def get_object(object_name: str):
    global error_count
    if not args.single_file:
        object_name_path = os.path.join(
            args.objects_storage_path, args.father_path, object_name.replace(args.bucket_path, "")
        )
        local_objects_storage_path = os.path.join(
            args.objects_storage_path, object_name_path[: object_name_path.rfind("/")]
        )
    else:
        object_name_path = os.path.join(args.objects_storage_path, args.father_path)
        local_objects_storage_path = args.objects_storage_path
    with print_lock:
        if not os.path.exists(local_objects_storage_path):
            os.makedirs(local_objects_storage_path)
    b64_object_name = base64.urlsafe_b64encode(object_name.encode()).decode()
    path = f"/rest/boto3/s3/{args.vendor}/{args.region}/{args.app_token}/{args.bucket_name}/{b64_object_name}"
    headers = {"Content-Type": "application/json", "csb-token": args.app_token, "Connection": "close"}

    with session.get(args.csb_file_server + path, headers=headers) as result:
        if result.status_code > 200:
            raise Exception(
                f"download file error. return code {result.status_code}, exception:{result.content.decode('utf-8')} "
            )
        with open(object_name_path, "wb") as f:
            for content in result.iter_content(chunk_size=args.buffer_size):
                f.write(content)


def get_range_object(object_name: str, h_size, file_io):
    global big_file_process_count
    global big_file_process_total
    global start_count_speed
    global current_speed
    while range_queue.qsize() > 0:
        content_range = range_queue.get()
        b64_object_name = base64.urlsafe_b64encode(object_name.encode()).decode()
        path = f"/rest/boto3/s3/{args.vendor}/{args.region}/{args.app_token}/{args.bucket_name}/{b64_object_name}"
        headers = {
            "Content-Type": "application/json",
            "csb-token": args.app_token,
            "Range": f"Range={content_range[0]}-{content_range[1]}",
            "Date": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "Connection": "Keep-Alive",
        }
        try:
            with session.get(args.csb_file_server + path, headers=headers) as result:
                if result.status_code > 200:
                    raise Exception(
                        f"download file error. return code {result.status_code}, exception:{result.content.decode('utf-8')} "
                    )
                with file_lock:
                    file_io.seek(content_range[0])
                    for content in result.iter_content(chunk_size=args.buffer_size):
                        file_io.write(content)

            big_file_process_count += 1
            with print_lock:
                if big_file_process_count > big_file_process_total:
                    big_file_process_count = big_file_process_total
                print("\r", end="")
                print(
                    "Downloading <{}> size: {}: {}/{} - {}%: ".format(
                        object_name,
                        h_size,
                        big_file_process_count,
                        big_file_process_total,
                        int((big_file_process_count / big_file_process_total) * 100),
                    ),
                    "▋" * (int(big_file_process_count / big_file_process_total * 50)) + current_speed,
                    end="",
                )
                sys.stdout.flush()
        except Exception:
            import traceback

            traceback.print_exc()
            retry_map[str(content_range)] += 1
            if retry_map[str(content_range)] > args.retry_times:
                # log
                print(f"{content_range} try {args.retry_times} times failed")
                retry_map[str(content_range)] = traceback.format_exc()
            else:
                time.sleep(30)
                range_queue.put(content_range)
                get_range_object(object_name, h_size, file_io)
    start_count_speed = False


def do_range_object(pool: ThreadPoolExecutor):
    global big_file_process_count
    global big_file_process_total
    global start_count_speed
    while big_file_download_queue.qsize() > 0:
        big_file_process_count = 0
        download_content = big_file_download_queue.get()
        total_size = int(download_content["size"])
        object_name = download_content["objectKey"]
        one_thread_range = args.package_size
        range_list = [
            [i if i == 0 else i + 1, i + one_thread_range] for i in range(0, total_size + 1, one_thread_range)
        ]
        big_file_process_total = len(range_list)
        for r in range_list:
            range_queue.put(r)

        if not args.single_file:
            object_name_path = os.path.join(
                args.objects_storage_path, args.father_path, object_name.replace(args.bucket_path, "")
            )
            local_objects_storage_path = os.path.join(
                args.objects_storage_path, object_name_path[: object_name_path.rfind("/")]
            )
        else:
            object_name_path = os.path.join(args.objects_storage_path, args.father_path)
            local_objects_storage_path = args.objects_storage_path
        with print_lock:
            if not os.path.exists(local_objects_storage_path):
                os.makedirs(local_objects_storage_path)
        start_count_speed = True
        thread_count_current_speed = pool.submit(get_delta)

        file_io = open(object_name_path, "wb+")

        ls_thread = [
            pool.submit(get_range_object, object_name, human_size(total_size), file_io) for _ in range(args.thread_num)
        ]
        ls_thread.append(thread_count_current_speed)
        for task in as_completed(ls_thread):
            try:
                task.result()
            except Exception as e:
                raise e
        file_io.close()
        print_split()


def do_get():
    global file_count
    global downloaded_count
    global current_speed
    global start_count_speed
    while download_queue.qsize() > 0 or has_next:
        try:
            download_content = download_queue.get(timeout=args.time_wait)
        except Exception:
            break
        object_key = download_content["objectKey"]
        try:
            get_object(object_key)
            downloaded_count += 1
            with print_lock:
                print("\r", end="")
                print(
                    "Download progress: {}/{} - {}%: ".format(
                        downloaded_count, file_count, int((downloaded_count / file_count) * 100)
                    ),
                    "▋" * (int(downloaded_count / file_count * 50)) + current_speed,
                    end="",
                )
                sys.stdout.flush()
        except Exception:
            import traceback

            traceback.print_exc()

            retry_map[str(download_content)] += 1
            if retry_map[str(download_content)] > args.retry_times:
                logging.error(f"{download_content} try {args.retry_times} times failed")
                error_map[str(download_content)] = traceback.format_exc()
                continue
            download_queue.put(download_content)
    start_count_speed = False


def multi_thread_main():
    print("begin downloading small file ...")
    args.csb_file_server = get_file_server_endpoint()
    bucket_auth()
    full_thread_num = args.thread_num + 4
    pool = ThreadPoolExecutor(max_workers=full_thread_num, thread_name_prefix="Python Downloader")

    ls_thread = []
    thread_do_download_queue = pool.submit(do_download_queue)
    thread_count_current_speed = pool.submit(get_delta)
    ls_thread.append(thread_do_download_queue)
    for i in range(args.thread_num):
        ls_thread.append(pool.submit(do_get))
    ls_thread.append(thread_count_current_speed)
    tic = time.time()
    for task in as_completed(ls_thread):
        try:
            task.result()
        except Exception as e:
            raise e

    toc = time.time()
    print()
    seconds = toc - tic
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("small file download completed, cost time : {:0>d} h {:0>2d} m {:0>2d} s".format(int(h), int(m), int(s)))
    print(f"begin downloading {big_file_download_queue.qsize()} big files ...")

    tic = time.time()
    do_range_object(pool)
    toc = time.time()
    seconds = toc - tic
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("big file download completed, cost time : {:0>d} h {:0>2d} m {:0>2d} s".format(int(h), int(m), int(s)))


def print_split():
    print("\n\n--------------------------\n")


def check_retry():
    print(f"error_count =  {error_count}")
    if error_map:
        with open(args.fail_json_storage_path, "w", encoding="utf-8") as f:
            json.dump(error_map, f)
        logging.error(f"{len(error_map)} files download failed, written in {args.fail_json_storage_path}")
    else:
        print("all files download completed!")
    if "Windows" in platform.platform():
        set_display_required(False)


def check_platform():
    if "Windows" in platform.platform():
        print("检测到使用Windows系统，在Windows系统下，该脚本会阻止系统休眠，以确保下载能够完成。", end="")
        set_display_required(True)
        print_split()


def set_display_required(continuous: bool):
    import ctypes

    ES_CON = 0x80000000
    ES_DIS = 0x00000002
    if continuous:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_DIS | ES_CON)
    else:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_DIS)


def get_delta():
    if args.show_speed:
        import psutil

        global current_speed
        while start_count_speed:
            before = psutil.net_io_counters().bytes_recv
            time.sleep(0.1)
            now = psutil.net_io_counters().bytes_recv
            delta = (now - before) / (1024 * 102.4)
            current_speed = "  {:.3f} MB/s".format(delta)


if __name__ == "__main__":
    check_platform()
    multi_thread_main()
    check_retry()
