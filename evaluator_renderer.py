from math import inf
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import models as tv
from torch.nn.parallel import DistributedDataParallel
import os
import lpips as lps
import numpy as np
from PIL import Image
from lpips.pretrained_networks import alexnet
from copy import deepcopy
from collections import OrderedDict
from skimage.metrics import structural_similarity as compute_ssim
import moxing as mox
from sklearn.manifold import TSNE
from mimogpt.models.selftok.sd3.sd3_impls import SDVAE, SD3LatentFormat
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow
from mimogpt.models.selftok.image_tokenizer import ImageTokenizer
# from models.utils import load_state
from utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.distributed as dist
from diffusers.models import AutoencoderKL
# from mimogpt.models.selftok.flux.flux_utils import FluxLatentFormat
import pdb
from copy import deepcopy
from mimogpt.models.selftok.image_renderer import ImageRenderer


def load_state(model, prefix, state_dict):
    model_dict = model.state_dict()  
    pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict}  
    dict_t = deepcopy(pretrained_dict)
    for key, weight in dict_t.items():
        if key in model_dict and model_dict[key].shape != dict_t[key].shape:
            pretrained_dict.pop(key)
    m, u = model.load_state_dict(pretrained_dict, strict=False)




class local_alexnet(alexnet):
    def __init__(self, path):
        super().__init__(requires_grad=False, pretrained=False)
        tv_alexnet = tv.alexnet()
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

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def set_sd3_vae(vae_path):
    vae = SDVAE(device="cpu", dtype=torch.bfloat16)
    state_dict = torch.load(vae_path, map_location='cpu')
    load_state(vae, 'first_stage_model.', state_dict)
    vae.cuda()
    vae.eval()
    return vae

def set_flux_vae(vae_path):
    vae = AutoencoderKL.from_pretrained(vae_path)
            # self.latent_format = FluxLatentFormat()
    vae.cuda()
    vae.eval()
    return vae

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def set_ema_model(model):
    ema = deepcopy(model).to(torch.float32)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)
    ema = ema.cuda()
    ema.eval()
    return ema


def compute_psnr(recon, ori):
    x = torch.from_numpy(np.array(Image.open(ori))).cuda().float()
    y = torch.from_numpy(np.array(Image.open(recon))).cuda().float()
    mse = F.mse_loss(x, y)
    psnr = 20 * torch.log10(torch.Tensor([255.0]).to(x.device)) - 10 * torch.log10(mse)
    return psnr

class BaseEval:
    def __init__(self, cfg, ckpt_path, download_ckpt=True, datatype='256', start = 1.0, cfg_scale = 1,tmp_local_ckpt_path = '/cache/model/pretrained.pth',model_type='sd3',
                 lognorm_schedule=False,ema_decoder=False,**kwargs):
        # download_ckpt = False   # !!!!!!
        rank = dist.get_rank()
        self.cfg = cfg
        self.datatype = datatype
        self.model_type = model_type
        # define models
        if self.model_type == 'sd3':
            self.vae = set_sd3_vae(cfg.common.vae_path)
        elif self.model_type == 'flux':
            self.vae = set_flux_vae(cfg.common.vae_path)
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {self.model_type}. Expected 'sd3' or 'flux'.")
        # cfg.tokenizer.params.noise_schedule_config.val_lognorm_schedule=lognorm_schedule
        # # self.model = ImageTokenizer(**cfg.tokenizer.params)
        # # self.model.set_eval()


        self.renderer = ImageRenderer(**cfg.renderer.params)
        self.renderer.set_eval()

        if hasattr(self.cfg.renderer, "pretrained_path") and self.cfg.renderer.pretrained_path: 
            low_res_state_dict = torch.load(cfg.renderer.pretrained_path, map_location="cpu")
            try:
                if rank ==0 :
                    print(f"Loading all renderer...")
                filtered_dict = {k:v for k,v in low_res_state_dict['state_dict'].items() if 'enc_ema' not in k}
                self.renderer.load_state_dict(filtered_dict, strict=True)
            except:
                time.sleep(10 * (dist.get_rank() % 8))
                hf_logger.info(f"Loading partial state dict for rank: {dist.get_rank()}...")
                load_state(self.renderer, low_res_state_dict['state_dict'])
        
        self.renderer.cuda()

        self.lpips_loss = lps.LPIPS(net='alex', pnet_rand=True, verbose=False)
        self.lpips_loss.net = local_alexnet(cfg.common.alex_path)
        # self.ema_decoder = ema_decoder
        # if self.ema_decoder==True:
        #     self.ema = set_ema_model(self.model.model)
        #     self.ema.eval()
        self.vae.eval()
        # self.diti = self.model.diti
        # self.K = self.diti.K
        self.count = 0
        self.count_cfg = 0
        self.start = start
        self.cfg_scale = cfg_scale
        # if hasattr(cfg.tokenizer.params, "cut_of_k") and self.cfg.tokenizer.params.cut_of_k:
        #     self.cut_of_k = self.cfg.tokenizer.params.cut_of_k
        # else:
        #     self.cut_of_k = None
        
        # # load ckpt
        # os.makedirs('/cache/model', exist_ok=True)
        # self._local_ckpt = tmp_local_ckpt_path
        # # self._local_ckpt = '/cache/ckpt/selftok/08-07/iter_239999.pth'
        
        
        
        
        # if download_ckpt and rank == 0:
        #     print(f'download ckpt {ckpt_path}')
        #     mox.file.copy(ckpt_path, self._local_ckpt)
        dist.barrier()
        # state_dict = torch.load(self._local_ckpt, map_location="cpu")
        # state_dict = torch.load(ckpt_path, map_location="cpu")
        # if self.ema_decoder==True:
        #     self.ema.load_state_dict(state_dict['ema_state_dict'])
        # self.model.load_state_dict(state_dict['state_dict'],strict=False)

        # set eval-specific params
        # self._steps = 50
        # self.flow = RectifiedFlow(
        #     self._steps, self.start, self.cut_of_k, cfg_scale=self.cfg_scale,**cfg.tokenizer.params.noise_schedule_config,
        # )


        # self.cond_vary = (not cfg.model.full_tokens) \
        #     if hasattr(cfg.model, "full_tokens") else True
        
        # self.cond_vary = False
        # self.cond_vary = True
        self.saved_images = 8

        # set device
        self.lpips_loss = self.lpips_loss.cuda()
        # if self.ema_decoder==True:
        #     self.ema.cuda()
        # self.model.cuda()
        # self.model = DistributedDataParallel(self.model, device_ids=[dist.get_rank()])
        
        # self.lognorm_schedule = lognorm_schedule
        dist.barrier()

    def process_input(self, batch):
        if self.datatype.isnumeric() or self.datatype == 'extract_tokens':
            images = batch.cuda()
            x0 = self.vae.encode(images)
            if self.model_type == 'sd3':
                x0 = SD3LatentFormat().process_in(x0)
            # elif self.model_type == 'flux':
            #     x0 = FluxLatentFormat().process_in(x0)
            dec_in = x0
            enc_in = x0
            return dec_in, enc_in, images
        else:
            images, enc_in = batch
            images = images.cuda()
            enc_in = enc_in.cuda()
            if self.model_type == 'sd3':          
                dec_in = SD3LatentFormat().process_in(self.vae.encode(images))
                enc_in = SD3LatentFormat().process_in(self.vae.encode(enc_in))
            # elif self.model_type == 'flux':
            #     dec_in = FluxLatentFormat().process_in(self.vae.encode(images))
            #     enc_in = FluxLatentFormat().process_in(self.vae.encode(enc_in))
            return dec_in, enc_in, images

    def clean_up(self, remove_ckpt=False):
        dist.barrier()
        del self.model
        del self.lpips_loss
        if self.ema_decoder==True:
            del self.ema
        if remove_ckpt:
            if dist.get_rank() == 0 and os.path.exists('/cache/model/pretrained.pth'):
                os.remove('/cache/model/pretrained.pth')


class ReconstructEvalRenderer(BaseEval):
    @torch.no_grad()
    def validate(self, dataloader,**kwargs):
        # encoder = self.model.module.enc_ema
        lpips = 0.0
        psnr = 0.0
        ssim = 0.0

        total = 0
        images_array = []
        recons_array = []
        break_num=999999999999999999999999999999999
        count = 0
        pbar = tqdm(range(len(dataloader)))
        for batch, batch_low in dataloader:
            batch_low = batch_low.cuda()
            latent_low = self.vae.encode(batch_low)
            latent_low = SD3LatentFormat().process_in(latent_low)
            latent_low = latent_low.float()
            _, ids_emb, render_latents = self.renderer(x=latent_low, is_multi_res = False)
            low_res_latent = render_latents

            dec_in, enc_in, images = self.process_input(batch)
            N = dec_in.shape[0]
            total += N

            if self.model_type == 'sd3':
                pred_x0_out = SD3LatentFormat().process_out(low_res_latent)
                
                recons = self.vae.decode(pred_x0_out)
                
                originals = SD3LatentFormat().process_out(dec_in)
                originals = self.vae.decode(originals)
                
            images = originals

            # print("images.shape", images.shape)
            # print('recons.shape', recons.shape)

            recons = F.interpolate(recons, size=(512, 512), mode='bilinear', align_corners=True)

            # print('recons_resized.shape', recons.shape)
            # evaluate
            lpips_batch = self.lpips_loss(recons.clamp(-1,1), images.clamp(-1,1))
            
            norm_ip(recons, -1, 1)
            norm_ip(images, -1, 1)
            # psnr_batch = compute_psnr(recons*255, images*255)
            cur_psnr = 0.0
            currank = dist.get_rank()
            for b in range(len(images)):
                save_image(recons[b], f"/cache/recon_{currank}.png")
                save_image(images[b], f"/cache/ori_{currank}.png")
                cur_sub_psnr = compute_psnr(f"/cache/recon_{currank}.png", f"/cache/ori_{currank}.png")
                cur_psnr += cur_sub_psnr
            cur_psnr /= len(images)
            # print(cur_psnr)
            psnr_batch = cur_psnr
            # pdb.set_trace()
            for i in range(len(recons)):
                image_np = images[i].detach().cpu().numpy()
                recon_np = recons[i].detach().cpu().numpy()
                self.count += 1
                ssim += compute_ssim(image_np, recon_np, data_range=image_np.max()-image_np.min(), channel_axis=0)
                

            # update average
            lpips += lpips_batch.sum().item()
            psnr += psnr_batch.item() * N
            images_array.append(images.detach().cpu())
            recons_array.append(recons.detach().cpu())
            

            pbar.update(1)
            count += 1
            # if count == break_num:
            #     break
            
        dist.barrier()
        # if self.model_type == 'sd3':
        #     lpips_t, psnr_t, ssim_t, total_t = \
        #         torch.tensor(lpips).cuda(), torch.tensor(psnr).cuda(), torch.tensor(ssim).cuda(), torch.tensor(total).cuda()
        # elif self.model_type == 'flux':
        lpips_t = torch.tensor(lpips).float().cuda()
        psnr_t = torch.tensor(psnr).float().cuda()
        ssim_t = torch.tensor(ssim).float().cuda()
        total_t = torch.tensor(total).float().cuda() 
            
        dist.all_reduce(lpips_t)
        dist.all_reduce(psnr_t)
        dist.all_reduce(ssim_t)
        dist.all_reduce(total_t)

        print("total_t", total_t)

        lpips = (lpips_t / total_t).item()
        psnr = (psnr_t / total_t).item()
        ssim = (ssim_t / total_t).item()
        # generate reconstruction results image
        # images_array = torch.cat(images_array, dim=0)
        # recons_array = torch.cat(recons_array, dim=0)
        # results = torch.cat((
        #     images_array, recons_array
        # ), dim=0)
        
        # import pdb; pdb.set_trace()
        results = []
        for idx_batch in range(len(images_array)):
            for idx_sample in range(len(images_array[idx_batch])):
                results.append(images_array[idx_batch][idx_sample].unsqueeze(0))
                results.append(recons_array[idx_batch][idx_sample].unsqueeze(0))
        results = torch.cat(results, dim=0)

        
        results_img = save_image(results, nrow=2, normalize=True, value_range=(0,1))
        
        
        return lpips, psnr, ssim, results_img
        
                
    
    
    
    def log(self, root, model, ckpt, dset, recon_results):
        name = '1113_v8_07_cfg2_try2'
        img_path = f'{root}/{name}/{dset}/{model}/{ckpt}.png'
        lpips, psnr, ssim, recon_image = recon_results
        if dist.get_rank() == 0:
            os.makedirs(f'{root}/{name}/{dset}/{model}', exist_ok=True)
            recon_image.save(img_path)
        print(f"{model}: {ckpt} results on {dset} saved. LPIPS={lpips}, PSNR={psnr}, SSIM={ssim}.")

    
evaluators = {
    "ReconstructEvalRenderer": ReconstructEvalRenderer,
}
