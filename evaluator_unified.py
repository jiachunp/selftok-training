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
    # pdb.set_trace()
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
        cfg.tokenizer.params.noise_schedule_config.is_eval=cfg.common.is_eval
        self.model = ImageTokenizer(**cfg.tokenizer.params)
        self.model.set_eval()

        if datatype == '256':
            shift = 1.0
        elif datatype == '512':
            shift = 1.818

        # import pdb; pdb.set_trace()



        self.lpips_loss = lps.LPIPS(net='alex', pnet_rand=True, verbose=False)
        self.lpips_loss.net = local_alexnet(cfg.common.alex_path)
        self.ema_decoder = ema_decoder
        if self.ema_decoder==True:
            self.ema = set_ema_model(self.model.model)
            self.ema.eval()
        self.vae.eval()
        self.diti = self.model.diti
        self.K = self.diti.K
        self.count = 0
        self.count_cfg = 0
        self.start = start
        self.cfg_scale = cfg_scale
        if hasattr(cfg.tokenizer.params, "cut_of_k") and self.cfg.tokenizer.params.cut_of_k:
            self.cut_of_k = self.cfg.tokenizer.params.cut_of_k
        else:
            self.cut_of_k = None
        
        # load ckpt
        os.makedirs('/cache/model', exist_ok=True)
        self._local_ckpt = tmp_local_ckpt_path
        # self._local_ckpt = '/cache/ckpt/selftok/08-07/iter_239999.pth'
        
        
        
        
        if download_ckpt and rank == 0:
            print(f'download ckpt {ckpt_path}')
            mox.file.copy(ckpt_path, self._local_ckpt)
        dist.barrier()
        state_dict = torch.load(self._local_ckpt, map_location="cpu")
        # state_dict = torch.load(ckpt_path, map_location="cpu")
        # import pdb; pdb.set_trace()
        if self.ema_decoder==True:
            self.ema.load_state_dict(state_dict['ema_state_dict'])
        self.model.load_state_dict(state_dict['state_dict'],strict=False)

        # set eval-specific params
        self._steps = 50
        self.flow = RectifiedFlow(
            self._steps, self.start, self.cut_of_k, val_schedule='uniform', shift=shift, **cfg.tokenizer.params.noise_schedule_config,
        )


        # self.cond_vary = (not cfg.model.full_tokens) \
        #     if hasattr(cfg.model, "full_tokens") else True
        
        # self.cond_vary = False
        self.cond_vary = True
        self.saved_images = 8

        # set device
        self.lpips_loss = self.lpips_loss.cuda()
        if self.ema_decoder==True:
            self.ema.cuda()
        self.model.cuda()
        self.model = DistributedDataParallel(self.model, device_ids=[dist.get_rank()])
        
        self.lognorm_schedule = lognorm_schedule
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

class TrainEval(BaseEval):
    @torch.no_grad()
    def validate(self, dataloader,**kwargs):
        print("start validating...")
        encoder = self.model.module.encoder if hasattr(self.model, 'module') else self.model.encoder
        total_loss = 0
        total_dm_mse_loss = 0
        total = 0
        epochs = 5
        t = None
        break_num=99999999999999999999999999999999999999999
        count=0
        pbar = tqdm(range(epochs),desc='epoch')
        for e in range(epochs):
            for batch in dataloader:
                images = batch.cuda()
                x0 = self.vae.encode(images)
                if self.model_type == 'sd3':
                    x0 = SD3LatentFormat().process_in(x0)
                # elif self.model_type == 'flux':
                #     x0 = FluxLatentFormat().process_in(x0)
                if t is None:
                    t = self.model.module.diffusion.sample_t(x0.shape[0],1.0)
                loss, log_dict = self.model(x=x0, full_tokens=False)
                total_loss += loss.sum().item()
                total_dm_mse_loss += log_dict['dm_mse']
                total += 1
                count += 1
                # if count == break_num:
                #     break
            pbar.update(1)
        dist.barrier()
        total_loss_t, total_t = \
            torch.tensor(total_loss).cuda(), torch.tensor(total).cuda()
        # total_dm_mse_loss_t = torch.tensor(total_dm_mse_loss).cuda()
            
        dist.all_reduce(total_loss_t)
        # dist.all_reduce(total_dm_mse_loss_t)
        
        dist.all_reduce(total_t)
        avg_loss = (total_loss_t / total_t).item()
        # avg_dm_mse_loss = (total_dm_mse_loss_t / total_t).item()
        
        if dist.get_rank() == 0:
            print(f'avg_loss: {avg_loss}')
        # print(avg_dm_mse_loss)
        
        

        # encoder.quantizer._codebook.expire_bad_codes()
        # total_loss = 0
        # total = 0
        # for batch in dataloader:
        #     images = batch.cuda()
        #     x0 = self.vae.encode(images)
        #     if self.model_type == 'sd3':
        #         x0 = SD3LatentFormat().process_in(x0)
        #     elif self.model_type == 'flux':
        #         x0 = FluxLatentFormat().process_in(x0)
        #     loss, log_dict = self.model(x=x0, full_tokens=False, t=t[:x0.shape[0]])
        #     total_loss += loss.sum().item()
        #     total += 1
        # print(total_loss / total)
        return avg_loss

class CodebookEval(BaseEval):
    @torch.no_grad()
    def validate(self, dataloader,**kwargs):
        print("start ploting codebook...")
        encoder = self.model.module.encoder if hasattr(self.model, 'module') else self.model.encoder
        codes = encoder.quantizer._codebook.embed
        codes = codes.detach().cpu().numpy()[0]
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(codes)
        return tsne_results
    
    def log(self, root, model, ckpt, dset, tsne_results):
        # plot
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ax.scatter(tsne_results[:,0], tsne_results[:,1], s=0.1)
        fig.set_size_inches(18.5, 18.5)
        os.makedirs(f'{root}/codebook/{model}', exist_ok=True)
        fig.savefig(f'{root}/codebook/{model}/{ckpt}_tsne.png')   # save the figure to file
        plt.close(fig)    # close the figure window


class ReconstructEval(BaseEval):
    @torch.no_grad()
    def validate(self, dataloader,**kwargs):
        encoder = self.model.module.encoder if hasattr(self.model, 'module') else self.model.encoder
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
        for batch in dataloader:
            batch = batch.cuda()

            kwargs = {}

            dec_in, enc_in, images = self.process_input(batch)
            xt = torch.randn_like(dec_in)
            N = dec_in.shape[0]
            total += N
            device = dec_in.device
            if hasattr(self.cfg.tokenizer.params, "stages"):
                t_mapped = torch.tensor([self.flow.timestep_map[0]]*dec_in.shape[0], device=device).long()
            else:
                t_mapped = torch.tensor([(self.flow.timestep_map[0])/1000.0]*dec_in.shape[0], device=device)
            k = self.diti.to_indices(t_mapped)
            enc_in =enc_in.float()
            encoder_hidden_states, _, ori_hidden_states, mask, _, _, _ = encoder(enc_in, d=k, kwargs=kwargs)
            model_kwargs = dict(
                encoder_hidden_states=encoder_hidden_states,
                mask=mask,
                context_see_xt=True
            )

            if self.ema_decoder==True:
                pred_x0 = self.flow.p_sample_loop(
                    self.ema, xt.shape, xt, model_kwargs=model_kwargs,
                    start_t=self._steps, cond_vary=self.cond_vary,
                    diti=self.diti, encoder=encoder, x_0=enc_in,
                    ori_hidden_states=ori_hidden_states,lognorm_schedule = self.lognorm_schedule,**kwargs
                )
            else:
                pred_x0 = self.flow.p_sample_loop(
                    self.model.module.model, xt.shape, xt, model_kwargs=model_kwargs,
                    start_t=self._steps, cond_vary=self.cond_vary,
                    diti=self.diti, encoder=encoder, x_0=enc_in,
                    ori_hidden_states=ori_hidden_states,lognorm_schedule = self.lognorm_schedule,**kwargs
                )

            
            if self.model_type == 'sd3':
                pred_x0_out = SD3LatentFormat().process_out(pred_x0)
                
                recons = self.vae.decode(pred_x0_out)
                
                originals = SD3LatentFormat().process_out(dec_in)
                originals = self.vae.decode(originals)

                
            images = originals
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
                # pdb.set_trace()
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

        lpips_t = torch.tensor(lpips).float().cuda()
        psnr_t = torch.tensor(psnr).float().cuda()
        ssim_t = torch.tensor(ssim).float().cuda()
        total_t = torch.tensor(total).float().cuda() 
            
        dist.all_reduce(lpips_t)
        dist.all_reduce(psnr_t)
        dist.all_reduce(ssim_t)
        dist.all_reduce(total_t)
        lpips = (lpips_t / total_t).item()
        psnr = (psnr_t / total_t).item()
        ssim = (ssim_t / total_t).item()
        # generate reconstruction results image
        # images_array = torch.cat(images_array, dim=0)
        # recons_array = torch.cat(recons_array, dim=0)

        # results = torch.cat((
        #     images_array, recons_array
        # ), dim=0)

        results = []
        for idx_batch in range(len(images_array)):
            for idx_sample in range(len(images_array[idx_batch])):
                results.append(images_array[idx_batch][idx_sample].unsqueeze(0))
                results.append(recons_array[idx_batch][idx_sample].unsqueeze(0))
        results = torch.cat(results, dim=0)

        results_img = save_image(results, nrow=2, normalize=True, value_range=(0,1))
        
        return lpips, psnr, ssim, results_img,\
            None, None, None, None
                
                
    
    
    
    def log(self, root, model, ckpt, dset, recon_results):
        name = '1113_v8_07_cfg2_try2'
        img_path = f'{root}/{name}/{dset}/{model}/{ckpt}.png'
        lpips, psnr, ssim, recon_image = recon_results
        if dist.get_rank() == 0:
            os.makedirs(f'{root}/{name}/{dset}/{model}', exist_ok=True)
            recon_image.save(img_path)
        print(f"{model}: {ckpt} results on {dset} saved. LPIPS={lpips}, PSNR={psnr}, SSIM={ssim}.")


class ReconInsuffEval(BaseEval):
    @torch.no_grad()
    def validate(self, dataloader,**kwargs):
        encoder = self.model.module.encoder if hasattr(self.model, 'module') else self.model.encoder
        recons_array = []
        recons_inv_array = []
        psnr_array = []
        psnr_array_inv = []
        #xt = torch.randn(1, 16, 32, 32).cuda()
        break_num=1
        count = 0
        pbar = tqdm(range(len(dataloader)))
        for batch in dataloader:
            batch = batch.cuda()
            kwargs = {}


            dec_in, enc_in, images = self.process_input(batch) # torch.Size([1, 16, 32, 32])
            xt = torch.randn_like(dec_in)
            # device = dec_in.device
            # noise = torch.randn_like(dec_in)
            # T = torch.tensor([0.5]).to(device)
            # xt = self.flow.q_sample(dec_in,T,noise)
            N = 32
            device = dec_in.device
            if hasattr(self.cfg.tokenizer.params, "stages"):
                t_mapped = torch.tensor([self.flow.timestep_map[0]]*dec_in.shape[0], device=device).long()
            else:
                t_mapped = torch.tensor([(self.flow.timestep_map[0])/1000.0]*dec_in.shape[0], device=device)               
            k = self.diti.to_indices(t_mapped)
            enc_in =enc_in.float()
            encoder_hidden_states, _, ori_hidden_states, mask, _, _, _ = encoder(enc_in, d=k,kwargs=kwargs)
            encoder_hidden_states = encoder_hidden_states[0].unsqueeze(0).expand(N, -1, -1)
            xt = xt.expand(N, -1, -1, -1)
            enc_in = enc_in.expand(N, -1, -1, -1)
            mask = mask.expand(N, -1)
            ori_hidden_states = ori_hidden_states[0].unsqueeze(0).expand(N, -1, -1)
            feat_mask = torch.ones(N, mask.shape[1]).bool().to(device) # torch.Size([32, 512])
            feat_mask_inv = torch.zeros(N, mask.shape[1]).bool().to(device)
            for i in range(N):
                feat_mask[i, (i+1)*(encoder_hidden_states.shape[1]//N):] = False
                #feat_mask[i, i*(encoder_hidden_states.shape[1]//N):] = False
                #if i > 0:
                # feat_mask_inv[i, -(i+1)*(encoder_hidden_states.shape[1]//N):] = True
            # import pdb; pdb.set_trace()
            recons = self.run_validation(feat_mask, mask, encoder_hidden_states, ori_hidden_states, xt, encoder, enc_in, **kwargs)
            # recons_inv = self.run_validation(feat_mask_inv, mask, encoder_hidden_states, ori_hidden_states, xt, encoder, enc_in)

            psnr_img = []
            psnr_img_inv = []
            if self.model_type == 'sd3':
                originals = SD3LatentFormat().process_out(dec_in)
                originals = self.vae.decode(originals)
            # elif self.model_type == 'flux':
            #     originals = FluxLatentFormat().process_out(dec_in)
            #     originals = self.vae.decode(originals).sample
            images = originals
            save_image(images, f"/cache/ori_.png")
            norm_ip(images, -1, 1)
            for i in range(len(recons)):
                save_image(recons[i], f"/cache/recon_.png")
                cur_sub_psnr = compute_psnr(f"/cache/recon_.png", f"/cache/ori_.png")
                psnr_img.append(cur_sub_psnr)

            psnr_array.append(psnr_img)
            psnr_array_inv.append(psnr_img_inv)

            # results_img = save_image(recons, f'/home/ma-user/work/wentao/selftok_clean1023_cfg/insuf_1_tmp/{self.count}.png', nrow=8, normalize=True, value_range=(0,1))
            # results_img_inv = save_image(recons_inv, f'/home/ma-user/work/wentao/selftok_clean1023_cfg/insuf_1_tmp/{self.count}_inv.png', nrow=8, normalize=True, value_range=(0,1))
            
            self.count += 1
            pbar.update(1)
            recons_array.append(recons.detach().cpu())
            # recons_inv_array.append(recons_inv.detach().cpu())
            count += 1
            # if count == break_num:
            #     break
        dist.barrier()
        # generate reconstruction results image
        return recons_array,psnr_array,psnr_array_inv
        # return recons_array, psnr_array
    
    def run_validation(self, feat_mask, mask, encoder_hidden_states, ori_hidden_states, xt, encoder, enc_in, **kwargs):
        if not self.cond_vary:
                mask = feat_mask
                super_mask = None
        else:
            mask = mask * feat_mask
            super_mask = feat_mask

        feat_mask = feat_mask.unsqueeze(-1).float()
        encoder_hidden_states = encoder_hidden_states * feat_mask
        ori_hidden_states = ori_hidden_states * feat_mask
      
        model_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            mask=mask,
            context_see_xt=True,
            context_see_rec=False,
        )
        # import pdb; pdb.set_trace()
        if hasattr(self.cfg.tokenizer.params, "stages"):
            pred_x0 = self.flow.p_sample_loop(
                self.model.module.model, xt.shape, xt, model_kwargs=model_kwargs,
                start_t=self._steps, dt = 1, cond_vary=self.cond_vary,
                diti=self.diti, encoder=encoder, x_0=enc_in,
                ori_hidden_states=ori_hidden_states, super_mask=super_mask,**kwargs
            )
        else:
            pred_x0 = self.flow.p_sample_loop(
                self.model.module.model, xt.shape, xt, model_kwargs=model_kwargs,
                start_t=self._steps, cond_vary=self.cond_vary,
                diti=self.diti, encoder=encoder, x_0=enc_in,
                ori_hidden_states=ori_hidden_states, super_mask=super_mask,**kwargs
            )
        if self.model_type == 'sd3':
            pred_x0 = SD3LatentFormat().process_out(pred_x0)
            recons = self.vae.decode(pred_x0)
        # elif self.model_type == 'flux':
        #     pred_x0 = FluxLatentFormat().process_out(pred_x0)
        #     recons = self.vae.decode(pred_x0).sample
        norm_ip(recons, -1, 1)

        return recons
    
    def log(self, root, model, ckpt, dset, recon_results):
        dir = f'{root}/recon_insuff/{dset}/{model}/{ckpt}/'
        #recons_array, recons_inv_array = recon_results
        recons_array, psnr_list,recons_inv_array,psnr_array_inv = recon_results
        if dist.get_rank() == 0:
            os.makedirs(f'{root}/recon_insuff/{dset}/{model}/{ckpt}/', exist_ok=True)
            for i in range(len(recons_array)):
                results_img = save_image(recons_array[i], dir + f'{i}.png', nrow=8, normalize=True, value_range=(0,1))
                results_img_inv = save_image(recons_inv_array[i], dir + f'{i}_inv.png', nrow=8, normalize=True, value_range=(0,1))  
                self.plot_and_save_tensor_values(psnr_list[i], filename = dir + f'chart_{i}.png') 
                self.plot_and_save_tensor_values(psnr_array_inv[i], filename = dir + f'chart_{i}_inv.png')
                
    
    def plot_and_save_tensor_values(self, tensor_list, filename='line_plot.png'):
        values = [x.item() for x in tensor_list]
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 33), values, marker='o')

        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        return

class ReconstructSmalltEval(BaseEval):
    @torch.no_grad()
    def validate(self, dataloader,**kwargs):
        encoder = self.model.module.encoder if hasattr(self.model, 'module') else self.model.encoder
        lpips = 0.0
        psnr = 0.0
        ssim = 0.0
        total = 0
        images_array = []
        recons_array = []
        count = 0
        # break_num=9
        pbar = tqdm(range(len(dataloader)))
        for batch in dataloader:
            dec_in, enc_in, images = self.process_input(batch)
            device = dec_in.device
            xt = torch.randn_like(dec_in)
            # noise = torch.randn_like(dec_in)
            # T = torch.tensor([self.start]).to(device)
            # xt = self.flow.q_sample(dec_in,T,noise)
            N = dec_in.shape[0]
            total += N
            if hasattr(self.cfg.tokenizer.params, "stages"):
                t_mapped = torch.tensor([self.flow.timestep_map[0]]*N, device=device).long()
            else:
                t_mapped = torch.tensor([(self.flow.timestep_map[0])/1000.0]*N, device=device)
            # t_tmp = (self.model.module.t2k * t_mapped).clamp(0, 1.0)             
            k = self.diti.to_indices(t_mapped)
            encoder_hidden_states, _, ori_hidden_states, mask, _, _, _ = encoder(enc_in, d=k)
            model_kwargs = dict(
                encoder_hidden_states=encoder_hidden_states,
                mask=mask
            )
            # self.model.module.model
            if hasattr(self.cfg.tokenizer.params, "stages"):
                pred_x0 = self.flow.p_sample_loop(
                    self.model.module.model, xt.shape, xt, model_kwargs=model_kwargs,
                    start_t=self._steps, dt = 1, cond_vary=self.cond_vary,
                    diti=self.diti, encoder=encoder, x_0=enc_in,
                    ori_hidden_states=ori_hidden_states
                )
            else:
                pred_x0 = self.flow.p_sample_loop(
                    self.model.module.model, xt.shape, xt, model_kwargs=model_kwargs,
                    start_t=self._steps, cond_vary=self.cond_vary,
                    diti=self.diti, encoder=encoder, x_0=enc_in,
                    ori_hidden_states=ori_hidden_states
                )
            if self.model_type == 'sd3':
                pred_x0 = SD3LatentFormat().process_out(pred_x0)
                recons = self.vae.decode(pred_x0)
                originals = SD3LatentFormat().process_out(dec_in)
                originals = self.vae.decode(originals)
            # elif self.model_type == 'flux':
            #     pred_x0 = FluxLatentFormat().process_out(pred_x0)
            #     recons = self.vae.decode(pred_x0).sample
            #     originals = FluxLatentFormat().process_out(dec_in)
            #     originals = self.vae.decode(originals).sample
            images = originals
            # evaluate
            lpips_batch = self.lpips_loss(recons.clamp(-1,1), images.clamp(-1,1))
            norm_ip(recons, -1, 1)
            norm_ip(images, -1, 1)
            psnr_batch = compute_psnr(recons*255, images*255)
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
        lpips_t, psnr_t, ssim_t, total_t = \
            torch.tensor(lpips).cuda(), torch.tensor(psnr).cuda(), torch.tensor(ssim).cuda(), torch.tensor(total).cuda()
        dist.all_reduce(lpips_t)
        dist.all_reduce(psnr_t)
        dist.all_reduce(ssim_t)
        dist.all_reduce(total_t)
        lpips = (lpips_t / total_t).item()
        psnr = (psnr_t / total_t).item()
        ssim = (ssim_t / total_t).item()
        images_array = torch.cat(images_array, dim=0)
        recons_array = torch.cat(recons_array, dim=0)
        gap = len(images_array) // self.saved_images
        selector = torch.arange(len(images_array)) % gap == 0
        results = torch.cat((
            images_array[selector][:self.saved_images], recons_array[selector][:self.saved_images]
        ), dim=0)
        r = torch.arange(len(results))
        pos = r % 8 + 8 * ((r // 8 * 2) - (r >= (len(results)//2)).int() * 7)
        pos = (pos==r.unsqueeze(1)).nonzero()[:,1]
        results = results[pos.cpu()]
        results_img = save_image(results, nrow=8, normalize=True, value_range=(0,1))
        return lpips, psnr, ssim, results_img 
    
    def log(self, root, model, ckpt, dset, recon_results):
        img_path = f'{root}/recon_single_eval/{dset}/{model}/{ckpt}.png'
        lpips, psnr, ssim, recon_image = recon_results
        if dist.get_rank() == 0:
            os.makedirs(f'{root}/recon_single_eval/{dset}/{model}', exist_ok=True)
            recon_image.save(img_path)
        print(f"{model}: {ckpt} results on {dset} saved. LPIPS={lpips}, PSNR={psnr}, SSIM={ssim}.")     
        
    
    
class ReconInterpolationEval(BaseEval):
    @torch.no_grad()
    def validate(self, dataloader,**kwargs):
        encoder = self.model.module.encoder if hasattr(self.model, 'module') else self.model.encoder
        recons_array = []
        recons_inv_array = []
        psnr_array = []
        psnr_array_inv = []
        #xt = torch.randn(1, 16, 32, 32).cuda()
        break_num=9999999999999999999999
        count = 0
        pbar = tqdm(range(len(dataloader)))
        for (batch_1,batch_2) in dataloader:
            dec_in_1, enc_in_1, images_1 = self.process_input(batch_1) # torch.Size([1, 16, 32, 32])
            dec_in_2, enc_in_2, images_2 = self.process_input(batch_2) # torch.Size([1, 16, 32, 32])
            
            xt = torch.randn_like(dec_in_1)
            N = 32
            device = dec_in_1.device
            if hasattr(self.cfg.tokenizer.params, "stages"):
                t_mapped = torch.tensor([self.flow.timestep_map[0]]*dec_in_1.shape[0], device=device).long()
            else:
                t_mapped = torch.tensor([(self.flow.timestep_map[0])/1000.0]*dec_in_1.shape[0], device=device)               
            k = self.diti.to_indices(t_mapped)
            
            encoder_hidden_states_1, _, ori_hidden_states_1, mask_1, _, _, _ = encoder(enc_in_1, d=k)
            encoder_hidden_states_2, _, ori_hidden_states_2, mask_2, _, _, _ = encoder(enc_in_2, d=k)
            
            encoder_hidden_states_1 = encoder_hidden_states_1[0].unsqueeze(0).expand(N, -1, -1)
            encoder_hidden_states_2 = encoder_hidden_states_2[0].unsqueeze(0).expand(N, -1, -1)
            
            xt = xt.expand(N, -1, -1, -1)
            
            
            enc_in_1 = enc_in_1.expand(N, -1, -1, -1)
            enc_in_1 = enc_in_1.expand(N, -1, -1, -1)
            
            mask_1 = mask_1.expand(N, -1)
            mask_2 = mask_2.expand(N, -1)
            
            ori_hidden_states_1 = ori_hidden_states_1[0].unsqueeze(0).expand(N, -1, -1)
            ori_hidden_states_2 = ori_hidden_states_2[0].unsqueeze(0).expand(N, -1, -1)
            
            
            
            
            feat_mask = torch.ones(N, mask_1.shape[1]).bool().to(device) # torch.Size([32, 512])
            for i in range(N):
                feat_mask[i, (i+1)*(encoder_hidden_states_1.shape[1]//N):] = False
                
                
            feat_mask_sub = ~feat_mask
            feat_mask = feat_mask.unsqueeze(-1).float()
            feat_mask_sub = feat_mask_sub.unsqueeze(-1).float()
            
            encoder_hidden_states = encoder_hidden_states_1 * feat_mask + encoder_hidden_states_2*feat_mask_sub
            ori_hidden_states = ori_hidden_states_1 * feat_mask + ori_hidden_states_2*feat_mask_sub
            model_kwargs = dict(
                encoder_hidden_states=encoder_hidden_states,
                mask=mask_1
            )
            pred_x0 = self.flow.p_sample_loop(
                self.model.module.model, xt.shape, xt, model_kwargs=model_kwargs,
                start_t=self._steps, cond_vary=self.cond_vary,
                diti=self.diti, encoder=encoder, x_0=enc_in_1,
                ori_hidden_states=ori_hidden_states
            )
            if self.model_type == 'sd3':
                pred_x0 = SD3LatentFormat().process_out(pred_x0)
                recons = self.vae.decode(pred_x0)
            # elif self.model_type == 'flux':
            #     pred_x0 = FluxLatentFormat().process_out(pred_x0)
            #     recons = self.vae.decode(pred_x0).sample
            norm_ip(recons, -1, 1)
            
            
            # recons = self.run_validation(feat_mask, mask_1, encoder_hidden_states_1, ori_hidden_states_1, xt, encoder, enc_in_1,
            #                              mask_2=mask_2,encoder_hidden_states_2=encoder_hidden_states_2,ori_hidden_states_2=ori_hidden_states_2,enc_in_2=enc_in_2)
            # recons_inv = self.run_validation(feat_mask_inv, mask, encoder_hidden_states, ori_hidden_states, xt, encoder, enc_in)

            psnr_img = []
            psnr_img_inv = []
            if self.model_type == 'sd3':
                originals = SD3LatentFormat().process_out(dec_in_1)
                originals = self.vae.decode(originals)
            # elif self.model_type == 'flux':
            #     originals = FluxLatentFormat().process_out(dec_in_1)
            #     originals = self.vae.decode(originals).sample
            images = originals
            norm_ip(images, -1, 1)
            for i in range(len(recons)):
                psnr = compute_psnr(recons[i].unsqueeze(0)*255, images*255)
                psnr_img.append(psnr)
            # for i in range(len(recons_inv)):
            #     psnr_inv = compute_psnr(recons_inv[i].unsqueeze(0)*255, images*255)
            #     psnr_img_inv.append(psnr_inv)
            # print(psnr_img)
            psnr_array.append(psnr_img)
            psnr_array_inv.append(psnr_img_inv)
            # results_img = save_image(recons, f'/home/ma-user/work/wentao/selftok_clean1023_cfg/insuf_1_tmp/{self.count}.png', nrow=8, normalize=True, value_range=(0,1))
            # results_img_inv = save_image(recons_inv, f'/home/ma-user/work/wentao/selftok_clean1023_cfg/insuf_1_tmp/{self.count}_inv.png', nrow=8, normalize=True, value_range=(0,1))
            
            self.count += 1
            pbar.update(1)
            recons_array.append(recons.detach().cpu())
            # recons_inv_array.append(recons_inv.detach().cpu())
            count += 1
            # if count == break_num:
            #     break
        dist.barrier()
        # generate reconstruction results image
        return recons_array,psnr_array,psnr_array_inv
        # return recons_array, psnr_array
    
    
    def log(self, root, model, ckpt, dset, recon_results):
        dir = f'{root}/recon_insuff/{dset}/{model}/{ckpt}/'
        #recons_array, recons_inv_array = recon_results
        recons_array, psnr_list,recons_inv_array,psnr_array_inv = recon_results
        if dist.get_rank() == 0:
            os.makedirs(f'{root}/recon_insuff/{dset}/{model}/{ckpt}/', exist_ok=True)
            for i in range(len(recons_array)):
                results_img = save_image(recons_array[i], dir + f'{i}.png', nrow=8, normalize=True, value_range=(0,1))
                results_img_inv = save_image(recons_inv_array[i], dir + f'{i}_inv.png', nrow=8, normalize=True, value_range=(0,1))  
                self.plot_and_save_tensor_values(psnr_list[i], filename = dir + f'chart_{i}.png') 
                self.plot_and_save_tensor_values(psnr_array_inv[i], filename = dir + f'chart_{i}_inv.png') 
                
    
    def plot_and_save_tensor_values(self, tensor_list, filename='line_plot.png'):
        values = [x.item() for x in tensor_list]
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 33), values, marker='o')

        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        return
    
    
class ExtractTokens(BaseEval):
    @torch.no_grad()
    def validate(self, dataloader,**kwargs):
        root_path = kwargs.get('root_path',None)
        encoder = self.model.module.encoder if hasattr(self.model, 'module') else self.model.encoder
        # pdb.set_trace()
        break_num=9999999999999999999999999999999999999999999999999
        count=0
        pbar = tqdm(range(len(dataloader)))
        for batch,_,path in dataloader:
            dec_in, enc_in, images = self.process_input(batch)
            device = dec_in.device
            N = dec_in.shape[0]
            if hasattr(self.cfg.tokenizer.params, "stages"):
                t_mapped = torch.tensor([self.flow.timestep_map[0]]*N, device=device).long()
            else:
                t_mapped = torch.tensor([(self.flow.timestep_map[0])/1000.0]*N, device=device)
            # t_tmp = (self.model.module.t2k * t_mapped).clamp(0, 1.0)             
            k = self.diti.to_indices(t_mapped)
            try:
                encoder_hidden_states, _, ori_hidden_states, mask, _, _, indices = encoder(enc_in, d=k)
            except:
                raise(f"please add indices in Encoder.forward()'s output")
            pbar.update(1)
            self.save_tensors_as_npy(indices,path,root_path)
            count += 1
            # if count == break_num:
            #     break 
    
    def save_tensors_as_npy(self,tensors, paths, root_dir):
        
        prefix = root_dir.replace('_tokens', '')
        os.makedirs(root_dir, exist_ok=True)
        
        for i, (path, tensor_item) in enumerate(zip(paths, tensors)):
            file_name = os.path.basename(path)  
            relative_path = os.path.relpath(path, prefix)  
            

            save_path = os.path.join(root_dir, relative_path.replace('.JPEG', '.npy'))
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            tensor_item_np = tensor_item.detach().cpu().numpy()  
            np.save(save_path, tensor_item_np)
            print(f"Saved tensor for {file_name} at {save_path}")
        
    
evaluators = {
    "TrainEval": TrainEval,
    "CodebookEval": CodebookEval,
    "ReconstructEval": ReconstructEval,
    "ReconInsuffEval": ReconInsuffEval,
    "ReconstructSmalltEval":ReconstructSmalltEval,
    "ReconInterpolationEval":ReconInterpolationEval,
    "ExtractTokens":ExtractTokens
}
