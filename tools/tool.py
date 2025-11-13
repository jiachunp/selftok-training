import pickle
import os
from tqdm import tqdm


def save_pkl_file():
    f = open("store.pkl", "wb")
    pickle.dump(obj, f)
    f.close()


def load_pkl_file():
    f = open("store.pkl", "rb")
    obj = pickle.load(f)
    f.close()


def data_GPU2cpu():
    path = "/ssd/ssd0/zangyu/code/MGM/mimo/data/train_data"
    dst_path = "/ssd/ssd3/zangyu/code/MGM/mimo/data/train_data_cpu"
    os.makedirs(dst_path, exist_ok=True)
    for file in os.listdir(path):
        s_path = os.path.join(path, file)
        d_path = os.path.join(dst_path, file)
        if os.path.exists(d_path):
            continue
        print("process : ", s_path)
        f = open(s_path, "rb")
        obj = pickle.load(f)
        f.close()
        for key in obj.keys():
            if isinstance(obj[key], int):
                continue
            obj[key] = obj[key].cpu()
        f = open(d_path, "wb")
        pickle.dump(obj, f)
        f.close()


def check_pkl():
    dst_path = "/ssd/ssd3/zangyu/code/MGM/mimo/data/train_data_cpu"
    files = [x for x in os.listdir(dst_path) if x.startswith("eagle_data_7_")]
    for file in files:
        s_path = os.path.join(dst_path, file)
        print(s_path)
        f = open(s_path, "rb")
        obj = pickle.load(f)
        f.close()


def download_data():
    app_token = "xxxxxxxxxxxxxxxxxxxxxxx"
    region = "cn-north-1"
    bucket_name = "bucket-distributed"
    for i in range(512):
        dst_file = "/ssd/ssd3/zangyu/code/MGM/mimo/data/pkl/pkl/MTI_split_part{:08}.pkl".format(i)
        path = "data/AIGC/TRAIN_DATA/V0601/pkl/MTI_split_part{:08}.pkl".format(i)
        if os.path.exists(dst_file):
            continue
        print(dst_file)
        cmd = """python ../../ldm/third_party/csb_obs/yellow_folder_downloader.py --app_token={} --vendor=HEC --region={} --bucket_name={} --path={} --objects_storage_path=/ssd/ssd3/zangyu/code/MGM/mimo/data/pkl/ --processes=1""".format(
            app_token, region, bucket_name, path
        )
        os.system(cmd)


if __name__ == "__main__":
    # download_data()
    # data_GPU2cpu()
    check_pkl()
