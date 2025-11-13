import os
import yaml
import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
sys.path.append(".")
# from mimogpt.models.selftok.models_ours import Enc_models, DiT_models, Encoder
from diffusers.models import AutoencoderKL
from mimogpt.models.selftok.diffusion import create_diffusion
from mimogpt.models.selftok.diti_utils import DiTi
from copy import deepcopy
import moxing as mox
from collections import OrderedDict
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import random
from easydict import EasyDict
import argparse
import numpy as np 
import glob
import pandas as pd
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from itertools import repeat
from mimogpt.models.selftok.image_tokenizer import ImageTokenizer
import lpips as lps
from mimogpt.models.selftok.diffusion import create_diffusion
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow
from mimogpt.models.selftok.sd3.sd3_impls import SDVAE, CFGDenoiser, SD3LatentFormat
from lpips.pretrained_networks import alexnet
from torchvision import models as tv
from mimogpt.engine.utils import set_seed, get_state_dict, parse_args_from_yaml, universal_cloud_copy
from mimogpt.datasets.selftok_dataset import build_simpleimageloader, center_crop_arr
from torchvision.datasets import ImageFolder

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    
def load_state(model, state_dict, prefix=''):
    model_dict = model.state_dict()  # 当前网络结构
    pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict}  # 预训练模型中可用的weight
    dict_t = deepcopy(pretrained_dict)
    for key, weight in dict_t.items():
        if key in model_dict and model_dict[key].shape != dict_t[key].shape:
            pretrained_dict.pop(key)
   
    m, u = model.load_state_dict(pretrained_dict, strict=False)
    if len(m) > 0:
        print("model missing keys:")
        print(m)
    if len(u) > 0:
        print("mode unexpected keys:")
        print(u)

class local_alexnet(alexnet):
    def __init__(self, path):
        super().__init__(requires_grad=False, pretrained=False)
        tv_alexnet = tv.alexnet(pretrained=False)
        tv_alexnet.load_state_dict(torch.load(path))
        alexnet_pretrained_features = tv_alexnet.features
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

class CustomLatentDataset(Dataset):
    def __init__(self, features_dir, preload_data=False):
        self.features_dir = features_dir
        self.preload_data = preload_data
        self.features_files = sorted(os.listdir(features_dir))
        
        if preload_data:
            self.features = []
            for i in range(len(self.features_files)):
                if i % 50000 == 0:
                    print(f"Loaded {float(i) / len(self.features_files) * 100.0:.1f}% data...")
                feature_file = self.features_files[i]
                feature = np.load(os.path.join(self.features_dir, feature_file))
                self.features.append(feature)

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        if self.preload_data:
            feature = torch.from_numpy(self.features[idx])
            return feature
        else:
            feature_file = self.features_files[idx]
            features = np.load(os.path.join(self.features_dir, feature_file))
            return torch.from_numpy(features)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def repeater(data_loader):
    for i, loader in enumerate(repeat(data_loader)):
        for data in loader:
            yield data

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def set_vae(vae_path):
    vae = AutoencoderKL.from_pretrained(vae_path)
    vae.cuda()
    vae.eval()
    return vae

def set_sd3_vae(vae_path):
    vae = SDVAE(device="cpu", dtype=torch.bfloat16)
    state_dict = torch.load(vae_path, map_location='cpu')
    load_state(vae, state_dict, 'first_stage_model.')
    vae.cuda()
    vae.eval()
    return vae

def set_ema_model(model):
    ema = deepcopy(model).to(torch.float32)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)
    ema = ema.cuda()
    ema.eval()
    return ema

class ImageEval:
    def __init__(self, cfg, pretrain, port=56947, yml_path='v2.17'):
        self.cfg = cfg
        self.port = port

        if cfg.tokenizer.params.model == 'MMDiT_XL':
            self.vae = set_sd3_vae(cfg.common.vae_path)
        else:
            self.vae = set_vae(cfg.common.vae_path)

        self.lpips_loss = lps.LPIPS(net='alex', pnet_rand=True)
        self.lpips_loss.net = local_alexnet(cfg.common.alex_path)
        self.lpips_loss = self.lpips_loss.cuda()
        self.is_root = (dist.get_rank() == 0)
        self.model = ImageTokenizer(**cfg.tokenizer.params)
        self.model.set_eval()
        self.ema = set_ema_model(self.model.model)
        self.ema.cuda()
        self.model.cuda()
        self.uncond_scale = cfg.common.get('uncond_scale',1.0)
       
        if 's3://' in pretrain:
            print(f"Downloading pretrained model from {pretrain}...")
            if os.path.exists('/cache/model/pretrained.pth'):
                print(f"mox_copy, pretrained.pth already exists!, skip copy")
            else:
                mox.file.copy_parallel(pretrain, '/cache/model/pretrained.pth')
            pretrain = '/cache/model/pretrained.pth'
        state_dict = torch.load(pretrain, map_location="cpu")
        print(f"Loading all...")
        self.ema.load_state_dict(state_dict['ema_state_dict'],strict=False)
        self.model.load_state_dict(state_dict['state_dict'],strict=False)

        val_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.tokenizer.params.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        val_image_path = cfg.dataloader.val_img_path
        val_dataset = ImageFolder(cfg.dataloader.val_img_path, transform=val_transform)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            sampler = val_sampler,
            pin_memory=True,
            drop_last=True
        )
        
        
        self.val_data_loader_iter = iter(repeater(val_dataloader))
        self.cond_vary = (not cfg.model.full_tokens) \
            if hasattr(cfg.model, "full_tokens") else True
        self.recon_tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=128),
            transforms.ToTensor()
        ])
        self.local_url = f'/cache/test/val/{yml_path}/cfg_{self.uncond_scale}/'
        if dist.get_rank() % 8 == 0:
            os.makedirs(self.local_url, exist_ok=True)

        self.model_type = 'sd3' if self.cfg.tokenizer.params.model == 'MMDiT_XL' else 'dit'
        
        self.uncond_c = self.model.model.uncond_c
        self.uncond_y = self.model.model.uncond_y
        

    def compute_psnr(self, recon, ori):
        x = torch.from_numpy(np.array(Image.open(ori))).cuda().float()
        y = torch.from_numpy(np.array(Image.open(recon))).cuda().float()
        mse = F.mse_loss(x, y)
        psnr = 20 * torch.log10(torch.Tensor([255.0]).to(x.device)) - 10 * torch.log10(mse)
        return psnr

    def process_input(self, x_0):
        with torch.no_grad():
            if type(x_0) is list:
                x_0 = x_0[0]
            x_0 = x_0.cuda()
            if not self.cfg.common.pre_encode:
                if self.model_type == 'sd3':
                    x_0 = self.vae.encode(x_0)
                else:
                    x_0 = self.vae.encode(x_0).latent_dist.sample().mul_(0.18215)
            else:
                x_0 = x_0.squeeze(dim=1)
            if self.model_type == 'sd3':
                x_0 = SD3LatentFormat().process_in(x_0)
        return x_0
    
    def validate(self, T):
        lpips = 0.0
        psnr = 0.0
        tests = 64  # 1
        # print(1234)
        
        total_steps = T
        start_steps = T-1
        for t in range(tests):
            x_0 = next(self.val_data_loader_iter)
            x_0 = self.process_input(x_0)

            noise = torch.randn_like(x_0, device=x_0.device)
            with torch.no_grad():
                if hasattr(self.model, "module"):
                    recon = self.reconstruct_val(
                        total_steps, start_steps, self.ema, noise, x_0,
                        diti=self.model.module.diti, encoder=self.model.module.encoder, ddim=True,
                        cond_vary=self.cond_vary
                    )["pred_x_0"]
                else:
                    recon = self.reconstruct_val(
                        total_steps, start_steps, self.ema, noise, x_0,
                        diti=self.model.diti, encoder=self.model.encoder, ddim=True,
                        cond_vary=self.cond_vary
                    )["pred_x_0"]
                # print(recon.size())
                # original image, reconstruction
                if self.model_type == 'sd3':
                    x_0 = SD3LatentFormat().process_out(x_0)
                    recon = SD3LatentFormat().process_out(recon)
                    img_0, img_recon = \
                        self.vae.decode(x_0),\
                        self.vae.decode(recon)
                else:
                    img_0, img_recon = \
                        self.vae.decode(x_0 / 0.18215).sample,\
                        self.vae.decode(recon / 0.18215).sample
            
            lpips_batch = self.lpips_loss(img_recon.clamp(-1,1), img_0.clamp(-1,1))
            lpips += lpips_batch.mean()
            norm_ip(img_recon, -1, 1)
            norm_ip(img_0, -1, 1)
            cur_psnr = 0.0
            currank = dist.get_rank()
            for b in range(len(img_0)):
                save_image(img_recon[b], f"/cache/recon_{currank}_{self.port}.png")
                save_image(img_0[b], f"/cache/ori_{currank}_{self.port}.png")
                cur_sub_psnr = self.compute_psnr(f"/cache/recon_{currank}_{self.port}.png", f"/cache/ori_{currank}_{self.port}.png")
                cur_psnr += cur_sub_psnr
            cur_psnr /= len(img_0)
            # print(cur_psnr)
            psnr += cur_psnr.mean()
            print(t, lpips_batch.mean(), cur_psnr, dist.get_world_size())
        mean_lpips = (lpips / float(tests)) / dist.get_world_size()
        dist.all_reduce(mean_lpips)
        if self.is_root:
            print(f"Val LPIPS on {tests} batches={mean_lpips.cpu().detach().numpy():.4f}.")

        mean_psnr = (psnr / float(tests)) / dist.get_world_size()
        dist.all_reduce(mean_psnr)
        if self.is_root:
            print(f"Val PSNR on {tests} batches={mean_psnr.cpu().detach().numpy():.4f}.")

        images = torch.cat((img_0, img_recon), dim=0)
        images = (images-images.min()) / (images.max()-images.min())
        images = [self.recon_tfm(img) for img in images]

        if dist.get_rank() % torch.cuda.device_count() == 0:
            eval_image_file = f"{self.local_url}/val_{self.trainer.iter:07d}_{t}.png"
            save_image(images, eval_image_file, nrow=len(x_0), normalize=True, value_range=(0, 1))
            
               

    def reconstruct_val(self, num_steps, t, model, noise=None, x0=None, y=None, diti=None, encoder=None, 
        cond_vary=False, ddim=False, dit=None):
        # print(str(num_steps))
        if self.model_type == 'sd3':
            diffusion = RectifiedFlow(num_steps, **self.cfg.tokenizer.params.noise_schedule_config)
        else:
            diffusion = create_diffusion(str(num_steps))
        N = x0.shape[0]
        device = x0.device
        x_t = noise

        with torch.no_grad():
            if diti is None:
                model_kwargs = {'y': y}
            else:
                if self.model_type == 'sd3':
                    t_mapped = torch.tensor([diffusion.timestep_map[0]]*N, device=device).long()
                    t_mapped = t_mapped - 1
                else:
                    t_mapped = torch.tensor([diffusion.timestep_map[t]]*N, device=device).long()
                k = diti.t_to_idx.to(device)[t_mapped]
                encoder_hidden_states, ori_hidden_states, mask, _, _ = encoder(x0, d=k)

                model_kwargs = dict(
                    encoder_hidden_states=encoder_hidden_states,
                    mask=mask
                )
            pred_x_0 = diffusion.p_sample_loop(
                model.forward, x_t.shape, x_t, clip_denoised=False, 
                model_kwargs=model_kwargs, progress=False, device=device,
                start_t=t, ddim=ddim,cond_vary=cond_vary, diti=diti, encoder=encoder, 
                x_0 = x0, ori_hidden_states=ori_hidden_states, dit=dit
            )
        return {"x_t": x_t, "pred_x_0": pred_x_0}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--port", type=int, default=56945)
    parser.add_argument("--yml-path", type=str, default="./configs/mimo/selftok/stable/480-inet-ftne.yml")
    parser.add_argument("--pretrained", type=str, default="s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_480-inet-ft5/2024-09-03_time_18_30_51/output/ckpt/iter_119999.pth")
    
    args = parser.parse_args()
    pretrained = args.pretrained
    cfg = parse_args_from_yaml(args.yml_path)

    set_random_seed(args.seed)

    # dist.init_process_group("nccl")
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        rank=0,
        world_size=1,
        # timeout=datetime.timedelta(hours=2.0),
    )
    
    model = ImageEval(cfg=cfg, pretrain=pretrained, port=args.port)
    print('evaluate reconstruct')
 
    model.validate(50)


    # python mimogpt/infer/eval_recon.py --yml-path ./configs/mimo/selftok/encoder/v4.2.6-open-vmode.yml --pretrained s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.6-open-vmode/2024-08-14_time_12_30_51/output/ckpt/iter_49999.pth
    # python mimogpt/infer/eval_recon.py --yml-path ./configs/mimo/selftok/encoder/v4.2.12-forcerecon-open.yml --pretrained s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.12-forcerecon-open/2024-08-14_time_18_30_51/output/ckpt/iter_49999.pth

    # python mimogpt/infer/eval_recon.py --yml-path ./configs/mimo/selftok/encoder/v4.2.6-open-vmode.yml --pretrained s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.6-open-vmode/2024-08-14_time_12_30_51/output/ckpt/iter_119999.pth --port 56945
    # python mimogpt/infer/eval_recon.py --yml-path ./configs/mimo/selftok/encoder/v4.2.6-forcerecon-open.yml --pretrained s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.6-forcerecon-open/2024-08-14_time_11_30_51/output/ckpt/iter_119999.pth --port 56946

    # python mimogpt/infer/eval_recon.py --yml-path ./configs/mimo/selftok/encoder/v4.2.12-forcerecon-open.yml --pretrained s3://bucket-9122-wulan/outputs/ywx1359914/selftok/encoder_v4.2.12-forcerecon-open/2024-08-14_time_18_30_51/output/ckpt/iter_119999.pth --port 56947


    # python mimogpt/infer/eval_recon.py --yml-path ./configs/mimo/selftok/encoder/v2.6sd3.yml --pretrained s3://bucket-9122-wulan/outputs/l00574761/selftok/encoder_v2.6_sd3/2024-08-08_time_21_45_00/output/configs/mimo/selftok/encoder/v2.6_sd3.yml/27570833000.0/ckpt/iter_39999.pth --port 56946

    # python mimogpt/infer/eval_recon.py --yml-path ./configs/mimo/selftok/encoder/v2.6sd3.yml --pretrained s3://bucket-9122-wulan/outputs/l00574761/selftok/encoder_v2.6_sd3/2024-08-08_time_21_45_00/output/configs/mimo/selftok/encoder/v2.6_sd3.yml/27570833000.0/ckpt/iter_99999.pth --port 56947

    # python mimogpt/infer/eval_recon.py --yml-path ./configs/mimo/selftok/encoder/v2.6sd3.yml --pretrained s3://bucket-9122-wulan/outputs/l00574761/selftok/encoder_v2.6_sd3_op/2024-08-12_time_20_16_00/output/configs/mimo/selftok/encoder/v2.6_sd3.yml/27576290000.0/ckpt/iter_159999.pth --port 56948