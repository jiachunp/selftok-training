import requests
import json

from .api import modelarts_job_query, modelarts_version_query
from .api import modelarts_job_query_v2


def get_pre_version_cfg(job_name, appid, app_token):
    train_jobs = modelarts_job_query(appid, app_token)["trainJobs"]
    find = False
    pre_version_cfg = {"pre_version_uid": "", "pre_version_id": ""}
    for job in train_jobs:
        if job["name"] == job_name:
            if job["versionId"] is None:
                task_id = job["id"]
                job_versions = modelarts_version_query(task_id, app_token)["versionList"]["versions"]
                for _version in job_versions:
                    if _version["id"] is not None and _version["versionId"] is not None:
                        pre_version_cfg["pre_version_uid"] = _version["id"]
                        pre_version_cfg["pre_version_id"] = _version["versionId"]
                        find = True
                        break
            else:
                pre_version_cfg["pre_version_uid"] = job["versionUid"]
                pre_version_cfg["pre_version_id"] = job["versionId"]
                find = True
            break
    return find, pre_version_cfg


def get_pre_version_cfg_v2(job_name, appid, app_token):
    train_jobs = modelarts_job_query_v2(appid, app_token)["trainJobs"]
    for job in train_jobs:
        if job["name"] == job_name:
            return True, job
    return False, {}


def get_task_id(job_name, appid, app_token):
    train_jobs = modelarts_job_query(appid, app_token)["trainJobs"]

    find = False
    task_id = None
    for job in train_jobs:
        if job["name"] == job_name:
            task_id = job["id"]
            find = True
            break

    return find, task_id


def get_pool_status(appid, region, token):
    url = "http://roma.huawei.com/csb/rest/saas/ei/eiWizard/train/pools/list?appid={}&region={}".format(appid, region)
    headers = {"Content-Type": "application/json", "csb-token": token}
    response = requests.get(url, headers=headers)
    ret = json.loads(response.content)

    status = None
    if ret.get("success", False):
        pool_info = ret["pools"][0]["nodeMetrics"]
        waiting = pool_info["appWaitingNumber"]
        queue = pool_info["appQueueNumber"]
        running = pool_info["appRuningNumber"]
        quota = pool_info["poolQuotaLimit"]
        status = {"waiting": waiting, "queue": queue, "running": running, "quota": quota}

    return status
