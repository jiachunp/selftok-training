# -*- coding: utf-8 -*-

import os
import time
import subprocess
from typing import Any
from pathlib import Path
from contextlib import ContextDecorator
import yaml
from easydict import EasyDict
import torch
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import math
import pathlib
from PIL import Image

try:
    os.environ["MOX_SILENT_MODE"] = "1"
    os.environ["MOX_FILE_LARGE_FILE_METHOD"] = "1"  # for moxing download acceleration
    import moxing as mox

    mox.file.set_auth(is_secure=False)

except:
    mox = None

def read_from_yaml(txt_path):
    with open(txt_path, "r") as fd:
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return EasyDict(y)

class MemartsCopyContext(ContextDecorator):
    def __init__(self):
        """
        This context manager is only in use when memarts is enabled, if memarts is not enabled it does nothing.
        It basically set _USE_MEMARTS to False when enter this context, then set _USE_MEMARTS back to True when exit
        Because normal mox copy in main process will cause error in dataloader when memarts is enabled,
        so we need to use this context manager to wrap any mox copy call to avoid error.

        To use this context manager:
        with MemartsCopyContext():
            mox.file.copy(xxx, xx)

        or

        @MemartsCopyContext()
        def mox_copy(src, dst):
            mox.file.copy_parallel(src, dst)
        """
        self.use_memarts = (os.environ.get("USE_MEMARTS") == "1") and (mox is not None)

    def __enter__(self):
        if self.use_memarts:
            mox.file.file_io._USE_MEMARTS = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        if self.use_memarts:
            mox.file.file_io._USE_MEMARTS = True


def _check_dir(dist_dir):
    copy_flag = True
    if os.path.exists(dist_dir):
        copy_flag = False
    if not os.path.exists(os.path.dirname(dist_dir)):
        os.makedirs(os.path.dirname(dist_dir))
    return copy_flag


def cmd_exec(cmd, just_print=False):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("\n{}:INFO:{}".format(t, cmd))
    if not just_print:
        os.system(cmd)


@MemartsCopyContext()
def mox_copy(src, dst, parallel=False):
    if src == dst:
        cmd_exec("mox_copy, src=dst={}, return".format(src), just_print=True)
        return
    if not (src.startswith("s3://") or dst.startswith("s3://")):
        cmd_exec(
            "mox_copy, at least one of src and dst need startswith s3://, src={}, dst={}, return".format(src, dst),
            just_print=True,
        )
        return
    while True:
        failed = 0
        try:
            cmd_exec(f"mox copy: {src} {dst}", just_print=True)
            if parallel:
                mox.file.copy_parallel(src, dst)
            else:
                mox.file.copy(src, dst)
            break
        except Exception as e:
            failed += 1
            time.sleep(60)
            if failed % 10 == 0:
                cmd_exec(
                    "error, maybe need check. copy failed {} times from {} to {}".format(failed, src, dst),
                    just_print=True,
                )
                cmd_exec("error message: {}".format(e), just_print=True)


def uncompress(tar_file):
    ret = subprocess.check_output("ps -ef | grep tar | grep xf | grep -v grep | grep -v 'sh -c' | wc -l", shell=True)
    ret = int(ret.decode("utf-8"))
    if ret > 0:
        cmd_exec("find uncompress running process:", just_print=True)
        os.system("ps -ef | grep tar | grep xf | grep -v grep | grep -v 'sh -c'")
    tar_name = os.path.split(tar_file)[-1]
    tar_dir = os.path.dirname(tar_file)
    cmd_exec("cd {} && tar -xf {} && rm -rf {} &".format(tar_dir, tar_name, tar_name))


def copy_data_to_cache(src_dir="", dist_dir="", rank=0, world_size=1, args=None):
    start_t = time.time()
    copy_flag = _check_dir(dist_dir)

    if args is not None and args.local_shuffle_type == 4:
        local_shuffle = args.local_shuffle_type
        zip_max_split = args.zip_max_split
        print(
            "training in cloud, using local_shuffle_type={}, zip_max_split={}".format(
                args.local_shuffle_type, zip_max_split
            )
        )
    else:
        local_shuffle = 0
        zip_max_split = -1

    if copy_flag:
        print("copy from {} to {}".format(src_dir, dist_dir))
        tar_files = []
        t0 = time.time()
        if ".mindrecord" in src_dir:
            src_dir = os.path.split(src_dir)[0]
            dist_dir = os.path.split(dist_dir)[0]

        last_file = None
        allready_uncompress = []
        copy_dir = []

        if mox.file.is_directory(src_dir):  # no new tar in tar !!!
            subfiles = [subfile for subfile in mox.file.list_directory(src_dir, recursive=False)]
            subfiles.sort()
            for subfile in subfiles:
                sub_src_dir = os.path.join(src_dir, subfile)
                sub_dist_dir = os.path.join(dist_dir, subfile)

                if local_shuffle and "split_part" in sub_src_dir:
                    if sub_src_dir.endswith("_map.pkl"):
                        continue
                    part_idx = int(
                        os.path.split(sub_src_dir)[-1][-8:-4]
                    )  # "AAAA_split_partBBBB.pkl" or "AAAA_split_partBBBB.zip"
                    if part_idx % world_size != rank or part_idx >= zip_max_split:
                        continue

                # uncompress last file
                if last_file is not None and last_file.endswith(".tar"):
                    uncompress(last_file)
                    allready_uncompress.append(last_file)

                # copy new file
                cmd_exec("copy from {} to {}".format(sub_src_dir, sub_dist_dir), just_print=True)
                if mox.file.is_directory(sub_src_dir):
                    mox_copy(sub_src_dir, sub_dist_dir, parallel=True)
                    copy_dir.append(sub_dist_dir)
                    last_file = None
                else:
                    mox_copy(sub_src_dir, sub_dist_dir)
                    last_file = sub_dist_dir

            if last_file is not None and last_file.endswith(".tar"):
                uncompress(last_file)
                allready_uncompress.append(last_file)

        else:
            mox_copy(src_dir, dist_dir)
            if dist_dir.endswith("tar") or dist_dir.endswith("tar.gz"):
                tar_files.append(dist_dir)

        t1 = time.time()
        cmd_exec("copy datasets, time used={:.2f}s".format(t1 - t0), just_print=True)

        # final check, no tar forget
        for _dir in copy_dir:
            tar_list = list(Path(_dir).glob("**/*.tar"))
            tar_files.extend(tar_list)
            tar_list = list(Path(_dir).glob("**/*.tar.gz"))
            tar_files.extend(tar_list)

        tar_files = [x for x in tar_files if str(x) not in allready_uncompress]

        cmd_exec("tar_files:{}".format(tar_files), just_print=True)
        for tar_file in tar_files:
            tar_dir = os.path.dirname(tar_file)
            cmd_exec("cd {} && tar -xf {} && rm -rf {} &".format(tar_dir, tar_file, tar_file))

        # final check, no tar process
        while True:
            ret = subprocess.check_output(
                "ps -ef | grep tar | grep xf | grep -v grep | grep -v 'sh -c' | wc -l", shell=True
            )
            ret = int(ret.decode("utf-8"))
            if ret == 0:
                cmd_exec("not find tar process, break", just_print=True)
                break
            else:
                cmd_exec("find {} tar process, sleep 10s".format(ret), just_print=True)
                os.system("ps -ef | grep tar | grep xf | grep -v grep | grep -v 'sh -c'")
            time.sleep(10)

        cmd_exec("copy data completed!", just_print=True)

    else:
        cmd_exec(
            "since data already exists, copying is not required, src={}, dst={}".format(src_dir, dist_dir),
            just_print=True,
        )

    end_t = time.time()
    cmd_exec("copy cost total time {:.2f} sec".format(end_t - start_t), just_print=True)


def vid_dataset_copy(fp, args=None):
    import moxing as mox
    from pathlib import Path

    y = read_from_yaml(fp)
    src = y["data_path"]["cloud_copy"]["src_dir"]
    dst = y["data_path"]["cloud_copy"]["dst_dir"]

    if os.path.exists(os.path.dirname(dst)):
        print("Train dataset exist, skip dataset copy.")
        return

    # print("Copying data from {} to {}".format(src, dst))
    # tar_files = []
    # mox.file.copy_parallel(src, dst)
    # tar_files.extend(list(Path(dst).glob('**/*.tar')))

    subfiles = [
        "CC3M.tar",
        "dev_test.tar",
        "vid_test.tar",
        "vid_9_open_1fps.tar",
        "models.tar",
        "json.tar",
    ]  # , 'open_pretrain_0328.tar', 'open_pretrain_0331.tar']
    for subfile in subfiles:
        s3_tar_file = os.path.join(src, subfile)
        tar_file = os.path.join(dst, subfile)
        print("Copying data from {} to {}".format(s3_tar_file, tar_file))
        mox.file.copy(s3_tar_file, tar_file)

        tar_dir = os.path.dirname(tar_file)
        print("cd {}; tar -xvf {} > /dev/null 2>&1; rm -rf {}".format(tar_dir, tar_file, tar_file))
        os.system("cd {}; tar -xvf {} > /dev/null 2>&1; rm -rf {}".format(tar_dir, tar_file, tar_file))
    return y


def img_dataset_copy(fp, rank=0, world_size=1, args=None):
    import moxing as mox
    from pathlib import Path

    y = read_from_yaml(fp)
    src = y["data_path"]["cloud_copy"]["src_dir"]
    dst = y["data_path"]["cloud_copy"]["dst_dir"]

    if os.path.exists(os.path.dirname(dst)):
        print("Train dataset exist, skip dataset copy.")
        return

    local_shuffle = int(y["dataloader"]["local_shuffle_type"])
    zip_max_split = int(y["data_path"]["zip_max_split"])
    zip_min_split = int(y["data_path"]["zip_min_split"]) if "zip_min_split" in y["data_path"] else 0
    do_unzip = bool(y["data_path"]["do_unzip"])

    subfiles = [subfile for subfile in mox.file.list_directory(src, recursive=False)]
    subfiles.sort()
    for subfile in subfiles:
        sub_src_file = os.path.join(src, subfile)
        sub_dst_file = os.path.join(dst, subfile)

        if local_shuffle and "split_part" in sub_src_file:
            if sub_src_file.endswith("_map.pkl") or sub_src_file.endswith("_otn.pkl"):
                continue
            part_idx = int(
                os.path.split(sub_src_file)[-1][-8:-4]
            )  # "AAAA_split_partBBBB.pkl" or "AAAA_split_partBBBB.zip"
            if part_idx % world_size != rank or part_idx >= zip_max_split or part_idx < zip_min_split:
                continue

            # TODO hard code for MTI_ori_split
            if "MTI_ori_split" in sub_src_file and sub_src_file.endswith(".pkl"):
                sub_src_file = sub_src_file.replace("MTI_ori_split", "MTI_ori_split_en_zh_pkl")

            mox.file.copy(sub_src_file, sub_dst_file)

            if sub_dst_file.endswith(".zip") and do_unzip:
                zip_dir, zip_name = os.path.split(sub_dst_file)
                zip_idx = int(zip_name[-8:-4])  # AAAA_split_partBBBB.zip"
                cmd_unzip = "(cd {}; mkdir {}; unzip -qq {} -d {}/ > /dev/null 2>&1; rm -rf {})&".format(
                    zip_dir, zip_idx, zip_name, zip_idx, zip_name
                )
                print(cmd_unzip)
                os.system(cmd_unzip)

    # copy aux tar files
    s3_tar_files = y["data_path"]["cloud_copy"]["aux_tars"]
    for s3_tar_file in s3_tar_files:
        tar_name = os.path.basename(s3_tar_file)
        dst_tar_file = os.path.join(dst, tar_name)
        mox.file.copy(s3_tar_file, dst_tar_file)

        if dst_tar_file.endswith(".tar"):
            tar_dir = os.path.dirname(dst_tar_file)
            print("cd {}; tar -xvf {} > /dev/null 2>&1; rm -rf {}".format(tar_dir, tar_name, tar_name))
            os.system("cd {}; tar -xvf {} > /dev/null 2>&1; rm -rf {}".format(tar_dir, tar_name, tar_name))
    return y


# below are new api, img_dataset_copy is deprecated.
def mox_copy_with_check(cloud_file, local_file, parallel=False):
    if os.path.exists(local_file) or os.path.exists(local_file[:-4]):
        print(f"mox_copy, dst={local_file} already exists!, skip copy")
        return
    mox_copy(cloud_file, local_file, parallel)


def common_cloud_copy(cfg, rank=0, world_size=1):
    # step1: copy aux file
    src_files = cfg.cloud_copy.src_files
    dst_dir = cfg.cloud_copy.dst_dir
    for cloud_file in src_files:
        if os.path.splitext(cloud_file)[-1]:  # file
            file_name = os.path.basename(cloud_file)
            local_file = os.path.join(dst_dir, file_name)
        else:  # directory
            cloud_file = os.path.dirname(cloud_file)
            file_name = cloud_file.split("/")[-1]
            local_file = os.path.join(dst_dir, file_name)
        mox_copy_with_check(cloud_file, local_file, mox.file.is_directory(cloud_file))

        if file_name.endswith(".tar"):
            print("cd {}; tar -xvf {} > /dev/null 2>&1; rm -rf {}".format(dst_dir, file_name, file_name))
            os.system("cd {}; tar -xvf {} > /dev/null 2>&1; rm -rf {}".format(dst_dir, file_name, file_name))
        if file_name.endswith(".zip"):
            print("cd {}; unzip {} > /dev/null 2>&1; rm -rf {}".format(dst_dir, file_name, file_name))
            os.system("cd {}; unzip {} > /dev/null 2>&1; rm -rf {}".format(dst_dir, file_name, file_name))
        if file_name.endswith(".whl"):
            print("cd {}; pip install {}".format(dst_dir, file_name))
            os.system("cd {}; pip install {}".format(dst_dir, file_name))

    # step2, check memarts
    if os.environ.get("USE_MEMARTS") == "1":
        return

    # step3, copy train zip and pkl
    for _, info in cfg.data_path.train.__dict__.items():
        zip_root, pkl_root, data_list, type, columns, pkl_format, split_range, ratio = info[:8]

        # step2, check memarts
        # if os.environ.get("USE_MEMARTS") == "1" and type != '':
        #     continue

        for idx in range(split_range[0], split_range[1]):
            # split entire list by nodes (here world size is the total nodes of one job， rank is node number)
            if idx % world_size != rank:
                continue

            # copy pkl
            if pkl_root:
                pkl_name = data_list.format(idx)
                cloud_file = os.path.join(pkl_root, pkl_name)
                mox_copy_with_check(
                    cloud_file, cloud_file.replace("s3://", "/cache/"), mox.file.is_directory(cloud_file)
                )

            # copy zip
            if zip_root:
                if data_list.endswith("pkl"):
                    zip_name = data_list.replace(".pkl", ".zip").format(idx)
                elif data_list.endswith("parquet"):
                    zip_name = data_list.replace(".parquet", ".zip").format(idx)
                else:
                    raise NotImplementedError
                cloud_file = os.path.join(zip_root, zip_name)
                mox_copy_with_check(
                    cloud_file, cloud_file.replace("s3://", "/cache/"), mox.file.is_directory(cloud_file)
                )

            # TODO: hard code for multi-zip data copy
            if len(info) >= 9 and isinstance(info[8], (list, tuple)):
                file_root, file_name = info[8]
                if isinstance(file_root, str) and file_root.startswith("s3://"):
                    file_name = file_name.format(idx)
                    cloud_file = os.path.join(file_root, file_name)
                    mox_copy_with_check(
                        cloud_file, cloud_file.replace("s3://", "/cache/"), mox.file.is_directory(cloud_file)
                    )


def apply_filename_adapter(type, file_pattern, idx):
    if type is None:
        return file_pattern.format(idx)
    if type == "webvid_raw":
        return file_pattern.format(idx * 50 + 1, (idx + 1) * 50)
    else:
        raise NotImplementedError


def universal_cloud_copy(
    url=None,
    dst=None,
    local_shuffle=False,
    file_pattern=None,
    adapter_type=None,
    split_range=None,
    cmd=None,
    rank=0,
    world_size=1,
):
    if url is None or dst is None:
        return
    if local_shuffle:
        assert (
            file_pattern is not None and split_range is not None
        ), f"Set file pattern and split range in yaml for local shuffle"
        for idx in range(split_range[0], split_range[1]):
            # split entire list by nodes (here world size is the total nodes of one job， rank is node number)
            if idx % world_size != rank:
                continue
            cur_file = apply_filename_adapter(adapter_type, file_pattern, idx)
            cloud_file = os.path.join(url, cur_file)
            local_file = os.path.join(dst, cur_file)
            if os.path.exists(local_file):
                # only mimo needs this because of debug mode 1, skip already copied files
                print(f"mox_copy, dst={local_file} already exists!, skip copy")
                continue
            if not mox.file.exists(cloud_file):
                print(f"mox_copy, src={cloud_file} not exists, skip copy")
                continue
            is_dir = mox.file.is_directory(cloud_file)
            mox_copy(cloud_file, local_file, is_dir)
    else:
        if os.path.exists(dst):
            # only mimo needs this because of debug mode 1, skip already copied files
            print(f"mox_copy, dst={dst} already exists!, skip copy")
            return
        is_dir = mox.file.is_directory(url)
        mox_copy(url, dst, parallel=is_dir)
        if cmd is not None:
            print(f"Begin to run: {cmd}")
            os.system(cmd)

def prepare_openimage():
    os.makedirs('/cache/data/openimage/sd1.5-features/', exist_ok=True)
    for i in range(0, 100):
        idx1=i*10
        idx2=(i+1)*10
        cmd = ''
        for j in range(idx1,idx2):
            if j!=idx2-1:
                cmd += f'unzip -q feature_{j}.zip & '
            else:
                cmd += f'unzip -q feature_{j}.zip'
        print(cmd)
        os.system(f'cd /cache/sd1.5-features/openimage256_features/; {cmd}')

    for i in range(0, 1000):
        curdir = f'/cache/sd1.5-features/openimage256_features/cache/openimage/features/openimage256_features/{i}/'
        cur = 0
        subf = os.listdir(curdir)
        cmd = ''
        for j,f in enumerate(subf):
            ff = curdir+f
            cmd += f'mv {ff} /cache/data/openimage/sd1.5-features/'
            if j != len(subf)-1 and j % 100!=99:
                cmd += ' & '
            if j % 100==99:
                os.system(cmd)
                cmd = ''
            cur+=1
        os.system(cmd)

def prepare_imagenet():
    unzipcmd = "cd /cache/data/imagenet/features/train/; unzip -q imagenet256_features.zip; unzip -q imagenet256_labels.zip"
    cmd_exec(unzipcmd)
    unzipcmd = "cd /cache/data/imagenet/features/val/; unzip -q imagenet256_features.zip; unzip -q imagenet256_labels.zip"
    cmd_exec(unzipcmd)
    
def prepare_sd3_openimage():
    #os.makedirs('/cache/data/openimage/sd3-features/openimage256_features', exist_ok=True)
    for i in range(0, 100):
        idx1=i*10
        idx2=(i+1)*10
        cmd = ''
        for j in range(idx1,idx2):
            if j!=idx2-1:
                cmd += f'unzip -q feature_{j}.zip & '
            else:
                cmd += f'unzip -q feature_{j}.zip'
        print(cmd)
        os.system(f'cd /cache/data/openimage256_features; {cmd}')

def prepare_sd3_imagenet():
    trainfiles = os.listdir('/cache/data/imagenet/sd3-features/train/imagenet256_features')
    valfiles = os.listdir('/cache/data/imagenet/sd3-features/val/imagenet256_features')
    #os.makedirs('/cache/data/openimage/sd3-features/train', exist_ok=True)
    trainnum = len(trainfiles) // 10
    for i in range(0, trainnum):
        idx1=i*10
        idx2=(i+1)*10
        cmd = ''
        for j in range(idx1,idx2):
            fn = trainfiles[j]
            if j!=idx2-1:
                cmd += f'unzip -q {fn} & '
            else:
                cmd += f'unzip -q {fn}'
        print(cmd)
        os.system(f'cd /cache/data/imagenet/sd3-features/train/imagenet256_features; {cmd}')
    valnum = len(valfiles) // 10
    for i in range(0, valnum):
        idx1=i*10
        idx2=(i+1)*10
        cmd = ''
        for j in range(idx1,idx2):
            fn = valfiles[j]
            if j!=idx2-1:
                cmd += f'unzip -q {fn} & '
            else:
                cmd += f'unzip -q {fn}'
        print(cmd)
        os.system(f'cd /cache/data/imagenet/sd3-features/val/imagenet256_features; {cmd}')
        
def prepare_sd3_zipdata(cfg):
    dataset_list = cfg.train_img_path + cfg.val_img_path
    for dataset in dataset_list:
        data_path = os.path.dirname(dataset[1])
        trainfiles = os.listdir(data_path)
        trainnum = len(trainfiles) // 10
        if 'v5_512_features' in data_path or 'openimage' in data_path: 
            for i in range(0, trainnum): 
                idx1=i*10 #10
                idx2=(i+1)*10 #10
                cmd = ''
                for j in range(idx1,idx2):
                    if j!=idx2-1:
                        cmd += f'unzip -q feature_{j}.zip & '
                    else:
                        cmd += f'unzip -q feature_{j}.zip'
                print(cmd)
                os.system(f'cd {data_path}; {cmd}')
        elif 'imagenet256' in data_path:
            for i in range(0, trainnum):
                idx1=i*10
                idx2=(i+1)*10
                cmd = ''
                for j in range(idx1,idx2):
                    fn = trainfiles[j]
                    if j!=idx2-1:
                        cmd += f'unzip -q {fn} & '
                    else:
                        cmd += f'unzip -q {fn}'
                print(cmd)
                os.system(f'cd {data_path}; {cmd}')
        
def read_from_yaml(txt_path):
    with open(txt_path, "r") as fd:
        cont = fd.read()
        try:
            y = yaml.load(cont, Loader=yaml.FullLoader)
        except:
            y = yaml.load(cont)
        return EasyDict(y)

def parse_args_from_yaml(yml_path):
    config = read_from_yaml(yml_path)
    config_obj = EasyDict(config)
    return config_obj

@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO] = None,
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if fp is not None:
        im.save(fp, format=format)
    return im