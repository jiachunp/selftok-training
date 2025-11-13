# bash run.sh
# OUTPUT_PATH = "/data/eval/wentao/test_1131"
# STATE_PATH = "/data/eval/eval_state_1118_test889.pkl"
# CONFIG_PATH = "./eval_config_recon.yml"
DEBUG = True
import argparse
import moxing as mox
import time
import os
import pickle
import re
import torch
from utils import parse_args_from_yaml

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", "gpu")

if DEVICE_TYPE == "npu":
    import torch_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.allow_internal_format = False
    from torch_npu.contrib import transfer_to_npu

import prettytable
from PIL import Image
from evaluator_unified import ReconstructEval,ReconstructSmalltEval, ReconInsuffEval,ReconInterpolationEval,ExtractTokens,TrainEval
from evaluator_renderer import ReconstructEvalRenderer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
from transforms import transforms_dict
import random
import pdb
import matplotlib.pyplot as plt
import numpy as np
from utils import save_image as si
from mimogpt.engine.utils import universal_cloud_copy
from torchvision.datasets import ImageFolder

S3_PATH = "bucket-5125-guiyang/outputs/selftok_eval/"


class ImageFolderWithPath(ImageFolder):
    def __init__(self, root_path, transform=None):
        super().__init__(root_path, transform=transform)
        

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return sample, target, path

class UnsupPairedImageFolder(Dataset):
    def __init__(self, root_path_1, root_path_2,transform=None):
        image_paths_1 = os.listdir(root_path_1)
        image_paths_2 = os.listdir(root_path_2)
        
        self.image_paths_1 = [os.path.join(root_path_1, p) for p in image_paths_1]
        self.image_paths_2 = [os.path.join(root_path_2, p) for p in image_paths_2]
        
        self.transform = transform
    
    def apply_transform(self, x, transform):
        x = transform(x)
        if len(x) == 1:
            x = x.expand(3, *x.shape[1:])
        if len(x) == 4:
            x = x[:3]
        return x

    def __getitem__(self, index):
        image_path_1 = self.image_paths_1[index]
        image_path_2 = self.image_paths_2[index]
        
        x = Image.open(image_path_1).convert('RGB')
        y = Image.open(image_path_2).convert('RGB')
        if self.transform is not None:
            if isinstance(self.transform, list):
                x = [self.apply_transform(x, t) for t in self.transform]
                y = [self.apply_transform(y, t) for t in self.transform]
            else:
                x = self.apply_transform(x, self.transform)
                y = self.apply_transform(y, self.transform)
        return x,y
    
class UnsupImageFolder(Dataset):
    def __init__(self, root_path, transform=None):
        image_paths = os.listdir(root_path)
        self.image_paths = [os.path.join(root_path, p) for p in image_paths]
        self.transform = transform
    
    def apply_transform(self, x, transform):
        x = transform(x)
        if len(x) == 1:
            x = x.expand(3, *x.shape[1:])
        if len(x) == 4:
            x = x[:3]
        return x

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            if isinstance(self.transform, list):
                x = [self.apply_transform(x, t) for t in self.transform]
            else:
                x = self.apply_transform(x, self.transform)
        return x
    
    def __len__(self):
        return len(self.image_paths)

class UnsupImageFolder_muti_res(Dataset):
    def __init__(self, root_path, transform=None,transform_low=None):
        image_paths = os.listdir(root_path)
        self.image_paths = [os.path.join(root_path, p) for p in image_paths]
        self.transform = transform
        self.transform_low = transform_low

    
    def apply_transform(self, x, transform):
        x = transform(x)
        if len(x) == 1:
            x = x.expand(3, *x.shape[1:])
        if len(x) == 4:
            x = x[:3]
        return x

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            if isinstance(self.transform, list):
                x_high = [self.apply_transform(x, t) for t in self.transform]
                x_low = [self.apply_transform(x, t) for t in self.transform_low]
            else:
                x_high = self.apply_transform(x, self.transform)
                x_low = self.apply_transform(x, self.transform_low)

        return x_high,x_low
    
    def __len__(self):
        return len(self.image_paths)


def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log(msg, level='INFO'):
    if level == 'DEBUG' and not DEBUG:
        return
    if dist.get_rank() == 0:
        print("{}:{}:{}".format(get_timestamp(), level, msg))

def evaluate(yml_path, ckpt_path, dataloader, download_ckpt, datatype,eval_type, start,cfg_scale,tmp_local_ckpt_path,model_type,lognorm,
             ema_decoder,**kwargs):
    cfg = parse_args_from_yaml(yml_path)
    if eval_type == 'reconstruction':
        evaluator = ReconstructEval(cfg, ckpt_path, download_ckpt, datatype,cfg_scale=cfg_scale,tmp_local_ckpt_path=tmp_local_ckpt_path,model_type=model_type,lognorm_schedule=lognorm,
                                    ema_decoder=ema_decoder,**kwargs)
        lpips, psnr, ssim, results_img,lpips_cfg,psnr_cfg,ssim_cfg,results_img_cfg = evaluator.validate(dataloader,**kwargs)
        del evaluator
        return lpips, psnr, ssim, results_img,lpips_cfg,psnr_cfg,ssim_cfg,results_img_cfg

    elif eval_type == 'insufficient':
        evaluator = ReconInsuffEval(cfg, ckpt_path, download_ckpt, datatype,cfg_scale=cfg_scale,tmp_local_ckpt_path=tmp_local_ckpt_path,model_type=model_type,lognorm_schedule=lognorm,
                                    ema_decoder=ema_decoder,**kwargs)
        results_img,psnr_list,_ = evaluator.validate(dataloader,**kwargs)
        del evaluator
        return results_img,psnr_list

    elif eval_type == 'interpolation':
        evaluator = ReconInterpolationEval(cfg, ckpt_path, download_ckpt, datatype,cfg_scale=cfg_scale,tmp_local_ckpt_path=tmp_local_ckpt_path,model_type=model_type,lognorm_schedule=lognorm,
                                    ema_decoder=ema_decoder,**kwargs)
        results_img,psnr_list,_ = evaluator.validate(dataloader,**kwargs)
        del evaluator
        return results_img,psnr_list
    
    elif eval_type == 'extract_tokens':
        evaluator = ExtractTokens(cfg, ckpt_path, download_ckpt, datatype,cfg_scale=cfg_scale,tmp_local_ckpt_path=tmp_local_ckpt_path,model_type=model_type,lognorm_schedule=lognorm,
                                    ema_decoder=ema_decoder,**kwargs)
        evaluator.validate(dataloader,**kwargs)
        del evaluator
        return None
    
    elif eval_type == 'traineval':
        evaluator = TrainEval(cfg, ckpt_path, download_ckpt, datatype,cfg_scale=cfg_scale,tmp_local_ckpt_path=tmp_local_ckpt_path,model_type=model_type,lognorm_schedule=lognorm,
                                    ema_decoder=ema_decoder,**kwargs)
        avg_loss= evaluator.validate(dataloader,**kwargs)
        del evaluator
        return avg_loss
    
    elif eval_type == 'reconstruction_renderer':
        evaluator = ReconstructEvalRenderer(cfg, ckpt_path, download_ckpt, datatype,cfg_scale=cfg_scale,tmp_local_ckpt_path=tmp_local_ckpt_path,model_type=model_type,lognorm_schedule=lognorm,
                                    ema_decoder=ema_decoder,**kwargs)
        lpips, psnr, ssim, results_img = evaluator.validate(dataloader,**kwargs)
        del evaluator
        return lpips, psnr, ssim, results_img

    
    else:
        raise(f'please provide correct eval type')
    
    
    

def save_image(model_name, ckpt_num, dataset, recon_image, save_name='recon',cfg_sacle=1, t = 0):
    '''
    for reconstruction
    '''
    img_path = f'{OUTPUT_PATH}/{save_name}/{dataset}/{model_name}/cfg{cfg_sacle}/{ckpt_num}_{t}.png'
    if dist.get_rank() == 0:
        # os.makedirs(f'{OUTPUT_PATH}/{save_name}/{dataset}/{model_name}_cfg_{cfg_sacle}', exist_ok=True)
        os.makedirs(f'{OUTPUT_PATH}/{save_name}/{dataset}/{model_name}/cfg{cfg_sacle}', exist_ok=True)
        
        recon_image.save(img_path)
    log(f"{model_name}: cfg_sacle={cfg_sacle} {ckpt_num} results on {dataset} saved.")
    return img_path


def save_recon(model_name, ckpt_num, dataset, recon_image, psnr_list, save_name='recon_insuf', cfg_scale=1, t=0):
    '''
    for insufficient
    '''
    rank = dist.get_rank()  
    dir = f'{OUTPUT_PATH}/{save_name}/{dataset}/{model_name}/cfg{cfg_scale}/{ckpt_num}/reconinsuff/rank_{rank}/'


    os.makedirs(dir, exist_ok=True)

    for i in range(len(recon_image)):

        image_data_list = [
            Image.fromarray((recon_image[i][j].detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0))
            for j in range(recon_image[0].shape[0])
        ]
        output_path = dir + f"gif_{i}.gif"
        print('output_path:', output_path)
        image_data_list[0].save(output_path, save_all=True, append_images=image_data_list[1:], duration=600, loop=0)


        results_img = si(recon_image[i], fp=dir + f'{i}.png', nrow=8, normalize=True, value_range=(0, 1))


        # plot_and_save_tensor_values(psnr_list[i], filename=dir + f'chart_{i}.png')

    log(f"{model_name}: cfg_scale={cfg_scale} {ckpt_num} results on {dataset} saved (rank {rank}).")

    return dir





def save_results(model_name, model_eval_state, save_name='recon',evaluation_type='reconstruction'):
    '''
    for reconstruction's metrics
    '''
    os.makedirs(f'{OUTPUT_PATH}/{save_name}/', exist_ok=True)
    if evaluation_type =='traineval':
        columns = ["ckpt","cfg_scale","avg_loss"]
    elif evaluation_type =='reconstruction':
        columns = ["ckpt","cfg scale","1.0lpips", "1.0psnr", "1.0ssim"]
    elif evaluation_type =='reconstruction_renderer':
        columns = ["ckpt","cfg scale","1.0lpips", "1.0psnr", "1.0ssim"]
        
        
    if dist.get_rank() == 0:
        dataset_tables = {}

        for cfg_scale in model_eval_state.keys():
            cfg_eval_state = model_eval_state[cfg_scale]
            for ckpt in cfg_eval_state.keys():
                ckpt_eval_state = cfg_eval_state[ckpt]
                # init dataset tables
                if len(dataset_tables.keys()) == 0:
                    for dataset in ckpt_eval_state.keys():
                        table = prettytable.PrettyTable(columns)
                        dataset_tables[dataset] = table
                for dataset in ckpt_eval_state.keys():
                    results = ckpt_eval_state[dataset]
                    if evaluation_type =='traineval':
                        dataset_tables[dataset].add_row(
                            [ckpt,cfg_scale, f"{results['avg_loss']:.4f}"]
                        )
                    elif evaluation_type =='reconstruction':
                        dataset_tables[dataset].add_row(
                            [ckpt,cfg_scale, f"{results['1.0lpips']:.4f}",f"{results['1.0psnr']:.2f}",f"{results['1.0ssim']:.4f}"]
                        )
                    elif evaluation_type =='reconstruction_renderer':
                        dataset_tables[dataset].add_row(
                            [ckpt,cfg_scale, f"{results['1.0lpips']:.4f}",f"{results['1.0psnr']:.2f}",f"{results['1.0ssim']:.4f}"]
                        )
        results_file = f"{OUTPUT_PATH}/{save_name}/{model_name}.txt"
        with open(results_file, "w") as file:
            file.write(f"{model_name} on {get_timestamp()}:\n")
            for dset in dataset_tables.keys():
                file.write(f"{dset}:\n")
                file.write(f"{dataset_tables[dset]}\n")


def plot_and_save_tensor_values(tensor_list, filename='line_plot.png'):
    values = [x.item() for x in tensor_list]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 33), values, marker='o')

    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    return


def create_dataloader(dataset, transform, bs=32):
    if hasattr(dataset, 'local_dir1'):
    # dset = UnsupImageFolder(dataset.local_dir, transform)
        dset = UnsupPairedImageFolder(dataset.local_dir,dataset.local_dir1, transform)
    elif hasattr(dataset, 'local_dir_ori'):
        dset = ImageFolderWithPath(dataset.local_dir_ori, transform)
    else:
        dset = UnsupImageFolder(dataset.local_dir, transform)
        # dset = UnsupImageFolder_muti_res(dataset.local_dir, transform)
        # dset = UnsupImageFolder_muti_res(dataset.local_dir, transform['512'],transform['256'])

    sampler = torch.utils.data.distributed.DistributedSampler(dset)
    dataloader = DataLoader(
        dset, batch_size=bs, shuffle=False, sampler=sampler, pin_memory=True, drop_last=False
    )
    return dataloader

if __name__ == "__main__":
    # init distributed
    
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--evaluation_type', type=str, help='which eval type')
    parser.add_argument('--port', type=int, help='port')
    args = parser.parse_args()
    EVALUATION_TYPE = args.evaluation_type
    PORT = args.port
    
    
    if EVALUATION_TYPE == 'reconstruction':
        CONFIG_PATH = "./eval_config_recon.yml"
        BATCH_SIZE = 8
        # BATCH_SIZE = 8  # TODO debug
    elif EVALUATION_TYPE == 'reconstruction_renderer':
        CONFIG_PATH = "./eval_config_recon_renderer.yml"
        BATCH_SIZE = 8
    elif EVALUATION_TYPE == 'insufficient':
        CONFIG_PATH = "./eval_config_insuff.yml"
        BATCH_SIZE = 1
    elif EVALUATION_TYPE == 'interpolation':
        CONFIG_PATH = "./eval_config_interpolation.yml"
        BATCH_SIZE = 1
    elif EVALUATION_TYPE == 'extract_tokens':
        CONFIG_PATH = "./eval_config_extract.yml"
        BATCH_SIZE = 32
    elif EVALUATION_TYPE == 'traineval':
        CONFIG_PATH = "./eval_config_train.yml"
        BATCH_SIZE = 32
    else:
        raise(f'please provide correct eval type')
    
    
    inputs = parse_args_from_yaml(CONFIG_PATH)
    if CONFIG_PATH == 'extract_tokens':
        ROOT_PATH = inputs.root_path
    STATE_PATH = inputs.eval_state
    OUTPUT_PATH = inputs.output_path
    MODEL_TYPE = inputs.model_type
    
    STATE_PATH = "./outputs_tmp/selftok_eval_v1/multi_res_sd3_npu_main.pkl"
    OUTPUT_PATH = "./outputs_tmp"
    if EVALUATION_TYPE == 'reconstruction_renderer':
        STATE_PATH = "./outputs_renderer/selftok_eval_v1/multi_res_sd3_npu_main.pkl"
        OUTPUT_PATH = "./outputs_renderer"

    os.makedirs(OUTPUT_PATH,exist_ok=True)

    
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{PORT}',
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    
    dataset_list = list(inputs.datasets.keys())
    for dataset_name in dataset_list:
        if inputs.datasets[dataset_name].get('dir',False) == False:
            break
        universal_cloud_copy(inputs.datasets[dataset_name].dir,inputs.datasets[dataset_name].local_dir,rank =0 ,
                             world_size=os.environ["WORLD_SIZE"])
        
    torch.cuda.set_device(dist.get_rank())

    # Check for eval path
    assert os.path.exists(OUTPUT_PATH), f"Please first attach {S3_PATH} to {OUTPUT_PATH}"

    # Check for previous eval state
    
    

    
    eval_state = {}
    # if os.path.exists(STATE_PATH):
    #     with open(STATE_PATH, 'rb') as f:
    #         eval_state = pickle.load(f)
    # Dataloader states:
    dataloaders = {}

    inputs = parse_args_from_yaml(CONFIG_PATH)
    datasets = inputs.datasets
    models = inputs.models
    # Check for new checkpoints
    while True:
        try: # parse config
            for model_name in models.keys():
                model = models[model_name]
                yml_path = model["yml"]
                ckpt_path = model["dir"]
                start_iter = model['start_iter'] if 'start_iter' in model else 0
                start_iter = 64998
                end_iter = model['end_iter'] if 'end_iter' in model else float('inf')
                ckpt_list = model['ckpt_list'] if 'ckpt_list' in model else None
                eval_every = model['every'] if 'every' in model else 1
                cfg_scale_list = model['cfg_scale'] if 'cfg_scale' in model else [1]
                model_datatype = '512' if not 'datatype' in model else model.datatype
                model_save_name = 'recon' if not 'save_name' in model else f"recon_{model.save_name}"
                tmp_local_ckpt_path = model['tmp_local_ckpt_path'] if 'tmp_local_ckpt_path' in model else '/cache/model/pretrained.pth'
                lognorm = model['lognorm'] if 'lognorm' in model else False
                ema_decoder = model['ema_decoder'] if 'ema_decoder' in model else False
                # try: # list model ckpts
                # import pdb; pdb.set_trace()
                ckpts = mox.file.list_directory(ckpt_path, recursive=False)
                # ckpts = ['iter_99.pth']

                log(f"{model_name} all ckpts {ckpts}", "DEBUG")
                if model_name not in eval_state:
                    eval_state[model_name] = {}
                model_eval_state = eval_state[model_name]
                if 1 not in model_eval_state:
                    model_eval_state[1] = {}
                cfg1_eval_state = model_eval_state[1]
            
                for ckpt in ckpts:
                    # check for ckpt format
                    matched = re.search(r"iter_(\d+).pth", ckpt)
                    if not matched:
                        continue    # non-ckpt files
                    # extract ckpt number
                    ckpt_num = int(matched.group(1))
                    if ckpt_num < start_iter:
                        continue    # ckpt too small
                    if ckpt_num > end_iter:
                        continue    # ckpt too large
                    if ckpt_list is not None:
                        if ckpt_num not in ckpt_list:
                            continue
                    if (ckpt_num+1) % eval_every != 0:
                        continue
                    
                    if ckpt not in cfg1_eval_state:
                        cfg1_eval_state[ckpt] = {}
                    ckpt_eval_state_cfg1 = cfg1_eval_state[ckpt]
                    download_ckpt = True
                    for dataset in datasets:
                        # Check if evaluated
                        if dataset in ckpt_eval_state_cfg1:
                            log(f"{model_name} ckpt {ckpt} evaluated on {dataset}, skipping...", "DEBUG")
                        else:
                            # Check if dataloader exists
                            if not model_datatype in dataloaders:
                                dataloaders[model_datatype] = {}
                            loaders = dataloaders[model_datatype]
                            if not dataset in loaders:
                                loaders[dataset] = create_dataloader(
                                    datasets[dataset], transforms_dict[model_datatype],BATCH_SIZE
                                )
                            dataloader = loaders[dataset]
                            cfg_scale = 1
                            log(f"{model_name} cfg={cfg_scale} ckpt {ckpt} start evaluation on {dataset}...", "DEBUG")
                            # New ckpt detected, evaluate
                            if EVALUATION_TYPE == 'reconstruction':
                                lpips0, psnr0, ssim0, results_img0,_,_,_,_ = evaluate(
                                yml_path, os.path.join(ckpt_path, ckpt), dataloader, download_ckpt, model_datatype,EVALUATION_TYPE,1.0,cfg_scale,tmp_local_ckpt_path,MODEL_TYPE,
                                lognorm,ema_decoder
                                )
                                if ckpt_eval_state_cfg1.get(dataset,{}) == {}:
                                    log(f"{model_name} cfg=1 ckpt {ckpt} on {dataset}: 1.0：lpips={lpips0:.4f}, psnr={psnr0:.2f}, ssim={ssim0:.4f}")
                                # Save recon image
                                if ckpt_eval_state_cfg1.get(dataset,{}) == {}:
                                    recon_img_path0 = save_image(model_name, ckpt_num, dataset, results_img0, model_save_name, cfg_sacle=1,t=0)
                                
                                if ckpt_eval_state_cfg1.get(dataset,{}) == {}:
                                    ckpt_eval_state_cfg1[dataset] = {
                                    'evaluated_on': get_timestamp(),
                                    '1.0lpips': lpips0,
                                    '1.0psnr': psnr0,
                                    '1.0ssim': ssim0,
                                    '1.0recon_image_path': recon_img_path0
                                    }
                            
                            elif EVALUATION_TYPE == 'reconstruction_renderer':
                                lpips0, psnr0, ssim0, results_img0 = evaluate(
                                yml_path, os.path.join(ckpt_path, ckpt), dataloader, download_ckpt, model_datatype,EVALUATION_TYPE,1.0,cfg_scale,tmp_local_ckpt_path,MODEL_TYPE,
                                lognorm,ema_decoder
                                )
                                if ckpt_eval_state_cfg1.get(dataset,{}) == {}:
                                    log(f"{model_name} cfg=1 ckpt {ckpt} on {dataset}: 1.0：lpips={lpips0:.4f}, psnr={psnr0:.2f}, ssim={ssim0:.4f}")
                                # Save recon image
                                if ckpt_eval_state_cfg1.get(dataset,{}) == {}:
                                    recon_img_path0 = save_image(model_name, ckpt_num, dataset, results_img0, model_save_name, cfg_sacle=1,t=0)
                                
                                if ckpt_eval_state_cfg1.get(dataset,{}) == {}:
                                    ckpt_eval_state_cfg1[dataset] = {
                                    'evaluated_on': get_timestamp(),
                                    '1.0lpips': lpips0,
                                    '1.0psnr': psnr0,
                                    '1.0ssim': ssim0,
                                    '1.0recon_image_path': recon_img_path0
                                    }
                                
                            elif EVALUATION_TYPE == 'insufficient':
                                results_img0, psnr_list = evaluate(
                                yml_path, os.path.join(ckpt_path, ckpt), dataloader, download_ckpt, model_datatype, EVALUATION_TYPE, 1.0,cfg_scale,tmp_local_ckpt_path,MODEL_TYPE,
                                lognorm,ema_decoder
                                )
                                recon_img_path0 = save_recon(model_name, ckpt_num, dataset, results_img0, psnr_list, model_save_name,cfg_scale=cfg_scale,t=0)

                                # update dataset eval state
                                ckpt_eval_state_cfg1[dataset] = {
                                    'evaluated_on': get_timestamp(),
                                    'recon_image_path': recon_img_path0,
                                }
                            if download_ckpt:
                                download_ckpt = False
                            # import pdb; pdb.set_trace()

                            if dist.get_rank() == 0:
                                if os.path.exists(STATE_PATH):
                                    with open(STATE_PATH, 'wb') as f:
                                        pickle.dump(eval_state, f)
                            dist.barrier()
                    if dist.get_rank() == 0 and os.path.exists(tmp_local_ckpt_path):
                        os.remove(tmp_local_ckpt_path)
                    dist.barrier()
                        # model_name all ckpts completes evaluation on all datasets
                    if EVALUATION_TYPE == 'reconstruction' or EVALUATION_TYPE == 'traineval' or EVALUATION_TYPE == 'reconstruction_renderer':
                        save_results(model_name, model_eval_state, model_save_name,EVALUATION_TYPE)
                    dist.barrier()
                    if EVALUATION_TYPE == 'reconstruction_renderer':
                        break
        except Exception as e: # list model ckpts failed
            print(f"rank{dist.get_rank()}: {e}")
            log(f"Error reading from dir of {model_name}!", "ERROR")
        # except: # parse config fails
        #     log(f"Error reading {CONFIG_PATH}!", "ERROR")
        # wait period
        time.sleep(60)