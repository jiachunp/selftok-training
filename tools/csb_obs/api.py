import os
import json
import base64
import requests
from urllib import request
import sys
from easydict import EasyDict
from boto3.session import Session


def get_deletefolder_token(appid, csb_token, bucketid, objectKeyList):
    url = "http://roma.huawei.com/csb/rest/s3/deleteobjects?appid={}".format(appid)
    headers = {"Content-Type": "application/json", "csb-token": csb_token}

    def modify(_path):
        if not _path.endswith("/"):
            _path = _path + "/"
        return _path

    body = {
        "bucketid": bucketid,
        "objectKeyList": [modify(x) for x in objectKeyList],
        "objectSizeList": ["--"] * len(objectKeyList),
    }
    json_str = json.dumps(EasyDict(body))
    ret = requests.delete(url, data=json_str, headers=headers)

    # check return msg
    if ret.status_code != requests.codes.ok:
        print(f"request error: {ret.status_code}")
        return False, ret

    text = json.loads(ret.text)
    if not text["success"]:
        print("request failed. msg: %s" % text)
        return False, text
    return True, text


def obs_delete_folder(file_server_ip, delete_token, csb_token):
    url = "{}/rest/csbfileserver/api/invoke/by/token?token={}".format(file_server_ip, delete_token)
    headers = {"Content-Type": "application/json", "csb-token": csb_token}
    ret = requests.post(url, headers=headers)

    # check return msg
    if ret.status_code != requests.codes.ok:
        print(f"request error: {ret.status_code}")
        return False, ret

    text = json.loads(ret.text)
    if not text["success"]:
        print("request failed. msg: %s" % text)
        return False, text
    return True, text
