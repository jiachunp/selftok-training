# -*- coding: utf-8 -*-

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import models as tv

from mimogpt.models.selftok.diffusion import create_diffusion
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow
from mimogpt.models.selftok.sd3.sd3_impls import SDVAE, CFGDenoiser, SD3LatentFormat

import lpips as lps
import numpy as np
from PIL import Image
from lpips.pretrained_networks import alexnet
from mimogpt.utils import hf_logger
from .train_loop import HookBase
from .selftok_hook import extract_exp_name


try:
    import moxing as mox

    def tqdm(_iter):
        return _iter

except:
    from tqdm import tqdm

__all__ = ["EvalSelftokHook"]

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

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


class EvalSelftokHook(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg
        eval_interval = cfg.common.val_interval
        self.eval_interval = int(eval_interval)
        self.best_metric = -1
        self.lpips_loss = lps.LPIPS(net='alex', pnet_rand=True)
        self.lpips_loss.net = local_alexnet(cfg.common.alex_path)
        self.lpips_loss = self.lpips_loss.cuda()
        self.is_root = (dist.get_rank() == 0)
        self.cond_vary = (not cfg.model.full_tokens) \
            if hasattr(cfg.model, "full_tokens") else True
        self.recon_tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=128),
            transforms.ToTensor()
        ])
        self.local_url = '/cache/data/val/'
        if dist.get_rank() % 8 == 0:
            os.makedirs(self.local_url, exist_ok=True)
        self.exp_name = extract_exp_name(cfg.common.output_path, cfg.train_url)
        self.img_url = os.path.join(cfg.common.val_url, self.exp_name)
        if dist.get_rank() % 8 == 0:
            mox.file.make_dirs(self.img_url)
        self.model_type = 'sd3' if self.cfg.tokenizer.params.model == 'MMDiT_XL' else 'dit'

    def after_step(self):
        if self.trainer.iter % self.eval_interval == 0 and self.trainer.iter > 0:
            self.validate()

    def train_mode(self, encoder_training, dm_training):
        if hasattr(self.trainer.model, "module"):
            model = self.trainer.model.module
        else:
            model = self.trainer.model
        if encoder_training:
            model.encoder.train()
        if dm_training:
            model.model.train()

    def eval_mode(self):
        if hasattr(self.trainer.model, "module"):
            model = self.trainer.model.module
        else:
            model = self.trainer.model
        encoder_training, dm_training = model.encoder.training, model.model.training
        model.encoder.eval()
        model.model.eval()
        return encoder_training, dm_training

    def compute_psnr(self, recon, ori):
        x = torch.from_numpy(np.array(Image.open(ori))).cuda().float()
        y = torch.from_numpy(np.array(Image.open(recon))).cuda().float()
        mse = F.mse_loss(x, y)
        psnr = 20 * torch.log10(torch.Tensor([255.0]).to(x.device)) - 10 * torch.log10(mse)
        return psnr

    def process_input(self, x_0):
        if type(x_0) is list:
            x_0 = x_0[0]    # remove class label
        with torch.no_grad():
            x_0 = x_0.cuda()
            if not self.trainer.pre_encode:
                if self.model_type == 'sd3':
                    x_0 = self.trainer.vae.encode(x_0)
                else:
                    x_0 = self.trainer.vae.encode(x_0).latent_dist.sample().mul_(0.18215)
            else:
                x_0 = x_0.squeeze(dim=1)
            if self.model_type == 'sd3':
                x_0 = SD3LatentFormat().process_in(x_0)
        return x_0
    
    def validate(self):
        encoder_training, dm_training = self.eval_mode()
        lpips = 0.0
        psnr = 0.0
        tests = 10
        total_steps = 50
        start_steps = 49
        for t in range(tests):
            x_0 = next(self.trainer.val_data_loader_iter)
            x_0 = self.process_input(x_0)
            noise = torch.randn_like(x_0, device=x_0.device)
            with torch.no_grad():
                if hasattr(self.trainer.model, "module"):
                    model_base = self.trainer.model.module
                else:
                    model_base = self.trainer.model
                recon = self.reconstruct_val(
                    total_steps, start_steps, self.trainer.ema, noise, x_0, 
                    diti=model_base.diti, encoder=model_base.encoder, ddim=True,
                    cond_vary=self.cond_vary
                )["pred_x_0"]
                
                # original image, reconstruction
                if self.model_type == 'sd3':
                    x_0 = SD3LatentFormat().process_out(x_0)
                    recon = SD3LatentFormat().process_out(recon)
                    img_0, img_recon = \
                        self.trainer.vae.decode(x_0),\
                        self.trainer.vae.decode(recon)
                else:
                    img_0, img_recon = \
                        self.trainer.vae.decode(x_0 / 0.18215).sample,\
                        self.trainer.vae.decode(recon / 0.18215).sample
            
            lpips_batch = self.lpips_loss(img_recon.clamp(-1,1), img_0.clamp(-1,1))
            lpips += lpips_batch.mean()
            norm_ip(img_recon, -1, 1)
            norm_ip(img_0, -1, 1)
            cur_psnr = 0.0
            currank = dist.get_rank()
            for b in range(len(img_0)):
                save_image(img_recon[b], f"/cache/recon_{currank}.png")
                save_image(img_0[b], f"/cache/ori_{currank}.png")
                cur_sub_psnr = self.compute_psnr(f"/cache/recon_{currank}.png", f"/cache/ori_{currank}.png")
                cur_psnr += cur_sub_psnr
            cur_psnr /= len(img_0)
            # print(cur_psnr)
            psnr += cur_psnr.mean()
        meters = self.trainer.meters["lpips"]
        mean_lpips = (lpips / float(tests)) / dist.get_world_size()
        meters.reduce_update(mean_lpips)
        if self.is_root:
            hf_logger.info(f"Step {self.trainer.iter}: Val LPIPS on {tests} batches={meters.avg:.4f}.")

        meters = self.trainer.meters["psnr"]
        mean_psnr = (psnr / float(tests)) / dist.get_world_size()
        meters.reduce_update(mean_psnr)
        if self.is_root:
            hf_logger.info(f"Step {self.trainer.iter}: Val PSNR on {tests} batches={meters.avg:.4f}.")

        images = torch.cat((img_0, img_recon), dim=0)
        images = images.clamp(-1, 1)
        images = (images-images.min()) / (images.max()-images.min())
        images = [self.recon_tfm(img) for img in images]

        if dist.get_rank() % torch.cuda.device_count() == 0:
            eval_image_file = f"{self.local_url}/val_{self.trainer.iter:07d}_{t}.png"
            save_image(images, eval_image_file, nrow=len(x_0), normalize=True, value_range=(0, 1))
            mox.file.copy(eval_image_file, f'{self.img_url}/val_{self.trainer.iter:07d}_{t}.png')
                    
        self.train_mode(encoder_training, dm_training)


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
                else:
                    t_mapped = torch.tensor([diffusion.timestep_map[t]]*N, device=device).long()
                k = diti.to_indices(t_mapped)
                encoder_hidden_states, _, ori_hidden_states, mask, _, _ = encoder(x0, d=k)

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
