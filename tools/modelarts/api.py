import os
import json
import base64
import pprint

import requests
from easydict import EasyDict


os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)


CREATE_TASK_BODY = {
    "name": "",  # TO BE FILLED
    "desc": "",  # TO BE FILLED
    "config": {
        "algoType": "common",
        "engineId": "",  # SPEC
        "appUrl": "",  # TO BE FILLED
        "bootFileUrl": "",  # TO BE FILLED
        "appUrlBucketId": "",  # TO BE FILLED
        "params": [],  # TO BE FILLED
        "dataType": ["OBS"],
        "dataUrlList": [
            {
                "dataSetType": "OBS",
                "dataUrl": "",  # TO BE FILLED
                "dataUrlBucketId": "",  # TO BE FILLED
            }
        ],
        "nasType": "nfs",
        "nasSharAddr": "",  # SPEC
        "nasMountPath": "/home/work/nas",
        "trainUrl": "",  # TO BE FILLED
        "trainUrlBucketId": "",  # TO BE FILLED
        "logUrl": "",  # TO BE FILLED
        "logUrlBucketId": "",  # TO BE FILLED
        "poolType": "",  # SPEC
        "poolUid": "",  # SPEC
        "workNum": 1,  # TO BE FILLED
        "priority": "2",
        "notify": "no",
        "preVersionId": "",
        "sharedPool": "",  # SPEC
    },
}


def modelarts_job_create(
    code_path,
    name,
    desc,
    boot_file,
    boot_args,
    train_url,
    data_url,
    user_id,
    worker_num,
    app_cfgs,
    region,
    bucket_id,
    pool_cfgs,
    pre_version_uid="",
    pre_version_id="",
):
    app_name = app_cfgs["name"]
    app_token = app_cfgs["token"]
    app_vendor = app_cfgs["vendor"]

    pre_url = "http://csb.roma.huawei.com/csb/rest/saas/ei/eiWizard/"
    headers = {"Content-Type": "application/json", "csb-token": app_token}
    url = (
        f"{pre_url}train/job/create?trainApiVersion=V2&appid={app_name}&vendor={app_vendor}&region="
        f"{'cn-north-1' if region=='cn-north-1-isp' else region}"
    )

    d = EasyDict(CREATE_TASK_BODY)
    d.config.update(pool_cfgs)

    d.desc = desc
    d.name = name
    d.userId = user_id

    # appUrl & bootFileUrl
    d.config.appUrl = code_path[4:].replace("\\", "/")
    d.config.bootFileUrl = os.path.join(d.config.appUrl, boot_file).replace("\\", "/")
    d.config.appUrlBucketId = bucket_id

    # data_url(dummy)
    d.config.dataUrlList[0].dataUrl = data_url
    d.config.dataUrlList[0].dataUrlBucketId = bucket_id

    # trainUrl & logUrl
    d.config.trainUrl = train_url
    d.config.trainUrlBucketId = bucket_id
    d.config.logUrl = train_url
    d.config.logUrlBucketId = bucket_id

    # commandLine Params
    d.config.params = []
    for k, v in boot_args.items():
        d.config.params.append({"key": k, "value": v})

    # server number
    d.config.workNum = worker_num

    # create new task in version
    if len(pre_version_uid) > 0 and len(pre_version_id) > 0:
        url = pre_url + "train/version/create?id=%s" % pre_version_uid
        d["desc"] = d.desc
        d["name"] = ""
        d["config"]["preVersionId"] = pre_version_id

    # convert to str and post
    json_str = json.dumps(d)
    ret = requests.post(url, data=json_str, headers=headers)

    # check return msg
    if ret.status_code != requests.codes.ok:
        print(f"request error: {ret.status_code}")
        return False

    text = json.loads(ret.text)
    if not text["success"]:
        print("request failed. msg: %s" % text)
        return False
    else:
        print(f"request succeeded. create job id: {text['id']}, desc={desc}")

    return text


CREATE_TASK_BODY_V2 = {
    "name": "",  # TO BE FILLED
    "jobKind": "job",  # job or edge_job
    "desc": "",  # TO BE FILLED
    "parentJobUid": "",  # TO BE FILLED
    "config": {
        "algoType": "common",  # unknow
        "annotations": {
            "faultToleranceJobRetryNum": 3,  # unknow
            "mindsporeEnableDebugger": "false",  # unknow
            "mindsporeRunningMode": "normal",  # unknow
        },
        "appUrl": "",  # TO BE FILLED
        "appUrlBucketId": "",  # TO BE FILLED
        "bootFileUrl": "",  # TO BE FILLED
        "dataSources": "private",
        # "engineId": "", # TO BE FILLED
        "flavorCode": "",  # TO BE FILLED
        "inputs": [],
        "outputs": [],
        "logUrl": "",  # TO BE FILLED
        "logUrlBucketId": "",
        # "nasMountPath": "",
        # "nasSharAddr": "",
        # "nasType": "nfs",
        "notify": "no",
        "params": [],  # TO BE FILLED
        "sfs": [],
        "policy": "regular",
        "specName": "",  # TO BE FILLED
        "poolType": "",  # TO BE FILLED
        "poolUid": "",  # TO BE FILLED
        "sharedPool": "yes",
        "priority": "3",  # TO BE FILLED, 3 is high
        "taskMode": "single",
        "workNum": "",  # TO BE FILLED
    },
}


def modelarts_job_create_v2(
    code_path,
    name,
    desc,
    boot_file,
    boot_args,
    train_url,
    user_id,
    worker_num,
    app_cfgs,
    region,
    bucket_id,
    pool_cfgs,
    pre_version_uid="",
    environments=None,
):
    app_name = app_cfgs["name"]
    app_token = app_cfgs["token"]
    app_vendor = app_cfgs["vendor"]

    pre_url = "http://csb.roma.huawei.com/csb/rest/saas/ei/eiWizard/"
    headers = {"Content-Type": "application/json", "csb-token": app_token}
    url = (
        f"{pre_url}train/job/create?trainApiVersion=V2&TENANTSPACEID=CSB&appid={app_name}&vendor={app_vendor}&region="
        f"{'cn-north-1' if region=='cn-north-1-isp' else region}"
    )

    # pre_url = "http://roma.huawei.com/csb/rest/saas/ei/eiWizard/"
    # headers = {"Content-Type": "application/json", "csb-token": app_token}
    # url = (
    #     f"{pre_url}train/job/create?appid={app_name}&vendor={app_vendor}&region="
    #     f"{'cn-north-1' if region == 'cn-north-1-isp' else region}"
    # )
    # url = url + "&trainApiVersion=V2"

    d = EasyDict(CREATE_TASK_BODY_V2)
    d.config.update(pool_cfgs)

    d.desc = desc
    d.name = name
    d.userId = user_id

    d.config.appUrl = code_path[4:].replace("\\", "/")
    d.config.appUrlBucketId = bucket_id
    d.config.bootFileUrl = os.path.join(d.config.appUrl, boot_file).replace("\\", "/")
    d.config.logUrl = train_url
    d.config.workNum = worker_num
    d.config.logUrlBucketId = bucket_id

    # commandLine Params
    d.config.params = []
    for k, v in boot_args.items():
        d.config.params.append({"key": k, "value": v})
    d.config.params.append({"key": "train_url", "value": "s3:/" + train_url})

    # add environments varaible in post json if it is not None, this is used for memarts
    if environments is not None:
        d.config.environments = environments

    # create new task in version
    if len(pre_version_uid) > 0:
        d["parentJobUid"] = pre_version_uid

    # convert to str and post
    print(url)
    print(pprint.pformat(d))
    json_str = json.dumps(d)
    ret = requests.post(url, data=json_str, headers=headers)

    # check return msg
    if ret.status_code != requests.codes.ok:
        print(f"request error: {ret.status_code}")
        text = json.loads(ret.text)
        print(text)
        return ret

    text = json.loads(ret.text)
    if not text["success"]:
        print("request failed. msg: %s" % text)
        return text
    else:
        print(f"request succeeded. create job id: {text['id']}, desc={desc}")

    return text


def modelarts_job_query(appid, token):
    params = '{"pageSize":100, "pageIndex":1}'
    params_base64 = base64.b64encode(params.encode("ascii")).decode("utf-8")
    url = "http://roma.huawei.com/csb/rest/saas/ei/eiWizard/train/job/list?appid={}&params={}".format(
        appid, params_base64
    )
    headers = {"Content-Type": "application/json", "csb-token": token}
    ret = requests.get(url, headers=headers)

    # check return msg
    if ret.status_code != requests.codes.ok:
        print(f"request error: {ret.status_code}")
        return False

    text = json.loads(ret.text)
    if not text["success"]:
        print("request failed. msg: %s" % text)
        return False

    return text


def modelarts_job_query_v2(appid, token, group_type="LIST", filter_word=""):
    params = {
        "pageSize": "100000",
        "pageIndex": "1",
    }
    if len(filter_word) > 0:
        params["filterParam"] = [{"key": "name", "value": filter_word}, {"key": "userId", "value": ""}]

    params_base64 = base64.b64encode(str(params).encode("ascii")).decode("utf-8")
    url = "http://roma.huawei.com/csb/rest/saas/ei/eiWizard/train/job/list?appid={}&params={}".format(
        appid, params_base64
    )
    url = url + "&trainApiVersion=V2&groupType={}&".format(group_type)

    headers = {"Content-Type": "application/json", "csb-token": token}
    ret = requests.get(url, headers=headers)

    # check return msg
    if ret.status_code != requests.codes.ok:
        print(f"request error: {ret.status_code}")
        return False

    text = json.loads(ret.text)
    if not text["success"]:
        print("request failed. msg: %s" % text)
        return False

    return text


def modelarts_version_query(job_id, token):
    url = "http://roma.huawei.com/csb/rest/saas/ei/eiWizard/train/version/list?id={}".format(job_id)
    headers = {"Content-Type": "application/json", "csb-token": token}
    ret = requests.get(url, headers=headers)

    # check return msg
    if ret.status_code != requests.codes.ok:
        print(f"request error: {ret.status_code}")
        return False

    text = json.loads(ret.text)
    if not text["success"]:
        print("request failed. msg: %s" % text)
        return False

    return text


def modelarts_job_stop(job_id, token, desc=None):
    url = "http://roma.huawei.com/csb/rest/saas/ei/eiWizard/train/version/stop?id={}".format(job_id)
    headers = {"Content-Type": "application/json", "csb-token": token}
    ret = requests.post(url, headers=headers)

    # check return msg
    if ret.status_code != requests.codes.ok:
        print(f"request error: {ret.status_code}")
        return False

    text = json.loads(ret.text)
    if not text["success"]:
        print("request failed. msg: %s" % text)
        return False

    print_str = "request succeeded. stop job id: {}".format(job_id)
    if desc is not None:
        print_str += ", desc={}".format(desc)
    print(print_str)


def modelarts_job_delete(job_id, token, desc=None):
    url = "http://roma.huawei.com/csb/rest/saas/ei/eiWizard/train/version/delete?id={}".format(job_id)
    headers = {"Content-Type": "application/json", "csb-token": token}
    ret = requests.delete(url, headers=headers)

    # check return msg
    if ret.status_code != requests.codes.ok:
        print(f"request error: {ret.status_code}")
        return False

    text = json.loads(ret.text)
    if not text["success"]:
        print("request failed. msg: %s" % text)
        return False

    print_str = "request succeeded. delete job id: {}".format(job_id)
    if desc is not None:
        print_str += ", desc={}".format(desc)
    print(print_str)
