import torch
import math
from einops import rearrange
import torch.distributed as dist
import pdb
import numpy as np
from scipy.integrate import quad
TRADITION = 1000

def append_to_shape(t, x_shape):
    return t.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    B = tensor.shape[0]
    loss = tensor.reshape(B, -1)
    loss = torch.sum(loss, dim=1) / loss.shape[1]
    # return tensor.mean(dim=list(range(1, len(tensor.shape))))
    return loss


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))

class RectifiedFlow(torch.nn.Module):
    def __init__(self, num_timesteps=100, start=1.0, cut_of_k=None, schedule="log_norm", val_schedule='shift', parameterization='x0', shift=1.0, m=0, s=1, force_recon=False, device='cuda',
                 is_eval=False):
        super().__init__()
        self.schedule = schedule
        self.parameterization = parameterization
        self.m = m
        self.s = s
        self.shift = shift
        self.is_eval = is_eval  # eval
        self.num_timesteps = num_timesteps
        self.start = start
        self.make_schedule(schedule=val_schedule, args=shift)
        self.device = device
        self.force_recon = force_recon
        self.cut_of_k = cut_of_k
        if self.schedule == "cosmap":
            self.t_trajectory = self.schedule_by_cosmap
        elif self.schedule == "log_norm":
            self.t_trajectory = self.schedule_by_logNorm
        elif self.schedule == "heavy_tails":
            self.t_trajectory = self.schedule_by_heavyTails
        elif self.schedule == "uniform":
            self.t_trajectory = self.schedule_by_uniform
        elif self.schedule == "linear_head":
            self.t_trajectory = self.schedule_by_linear_head
        elif self.schedule == "linear_tail":
            self.t_trajectory = self.schedule_by_linear_tail
        else:
            raise NotImplementedError
            
    def make_schedule(self, schedule="uniform", args=None):
        base_t = torch.linspace(self.start, 0, self.num_timesteps+1).cuda()
        if schedule == "uniform":
            scheduled_t = base_t
        elif schedule == "shift":
            shift = args
            scheduled_t =self.shift * base_t / (1 + (self.shift - 1) * base_t)
        elif schedule == "align_resolution":
            e = torch.e
            res1, s1, res2, s2, target_res, c = args
            m = (s1 -s2) / (res1 - res2) * (target_res - res1) + s1
            scheduled_t = e ** m / (e ** m + (1/base_t - 1) ** c)
        if self.is_eval == True:
            scheduled_t = self.uniform2lognorm_t(scheduled_t)
            print('eval = True')
        self.register_buffer("timestep_map", scheduled_t[:-1] * TRADITION)
        self.register_buffer("scheduled_t", scheduled_t[:-1])
        self.register_buffer("scheduled_t_prev", scheduled_t[1:])
        self.register_buffer("one_minus_scheduled_t", 1-scheduled_t[:-1])

    def sample_t(self, bs, shift):
        if self.t_trajectory == self.schedule_by_logNorm:
            t = self.t_trajectory(bs)
        else:
            t = self.t_trajectory(torch.rand(bs))
        if shift == 1.0:
            return t
        return shift * t / (1 + (shift - 1) * t)
    
    def shift_t(self, t, shift):
        return shift * t / (1 + (shift - 1) * t)

    def q_sample(self, x, t, noise=None):
        t = append_to_shape(t,x.shape)
        if noise is None:
            noise = torch.randn_like(x)
        return t * noise + (1 - t) * x

    def get_target(self, x, noise):
        target = noise - x
        return target
    
    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / TRADITION
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def schedule_by_logNorm(self, bs):
        u = torch.normal(mean=self.m, std=self.s, size=(bs,))
        t = torch.nn.functional.sigmoid(u)
        return t

    def schedule_by_cosmap(self, t):
        t = 1. - 1 / (torch.tan(torch.pi/2 * t) + 1)
        return t

    def schedule_by_heavyTails(self, t):
        t = 1 - t - self.s * (torch.cos(torch.pi/2 * t) - 1 + t)
        return t
    
    def schedule_by_uniform(self, t):
        return t

    def schedule_by_linear_head(self, t):
        x = 1 - torch.sqrt(1 - t)
        return x

    def schedule_by_linear_tail(self, t):
        x = torch.sqrt(t)
        return x
    
    def pdf_logNorm(self,y):
        exponent = -((np.log(y / (1 - y)) - self.m) ** 2) / (2 * self.s ** 2)
        return (1 / (np.sqrt(2 * np.pi) * self.s)) * np.exp(exponent) * (1 / (y * (1 - y)))
    
    
    def uniform2lognorm_t(self,t_schedule):
        t_device = t_schedule.device
        y_values = torch.linspace(0, 1, 1000)  # 
        cumulative_probabilities = np.zeros_like(y_values)
        for i in range(len(y_values)):
            integral, _ = quad(self.pdf_logNorm, 0.01, y_values[i])
            cumulative_probabilities[i] = integral
        corresponding_y_values = []
        for j,target_prob in enumerate(t_schedule):
            diff = torch.abs(torch.Tensor(cumulative_probabilities) - target_prob.cpu())
            index = torch.argmin(diff).item()  
            corresponding_y_values.append(y_values[index].item())  
        t_schedule = torch.tensor(corresponding_y_values).to(t_device)
        return t_schedule
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, recon_ratio=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            target = noise - x_start
        else:
            raise NotImplementedError()
            
        terms = {}
       
        v = model(x_t, t, **model_kwargs)
        v_gt = noise - x_start

        if self.force_recon:
            assert self.parameterization == 'velocity'
            # model_output = noise - model_output
            model_output = x_t - rearrange(t, 'b -> b 1 1 1') * v
            target = x_start
        else:
            model_output = v

        if "loss_mask" in model_kwargs:
            loss_mask = model_kwargs["loss_mask"].unsqueeze(1).repeat(1, target.shape[1], 1, 1)
            mse_loss = (target - model_output) ** 2
            terms["loss"] = sum_flat(mse_loss * loss_mask.float()) / sum_flat(loss_mask)
        else:
            terms["loss"] = mean_flat((target - model_output) ** 2)
            
        terms["mse"] = mean_flat((target - model_output) ** 2)

        if recon_ratio != 1.0 and self.force_recon:
            terms["loss"] = recon_ratio*terms["loss"] + (1-recon_ratio)*mean_flat((v_gt - v) ** 2)
        return terms

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        K = 512,
        start_t=None,
        dt = None,
        model_kwargs=None,
        uncond_scale=1.0,
        uncond_y=None,
        uncond_c=None,
        x_0=None,
        encoder=None,
        diti=None,
        dit=None,
        ori_hidden_states=None,
        cond_vary=False,
        super_mask=None,
        device=None,
        t2k = 1.,
        **kwargs,
    ):
        batch_size = shape[0]
        if device is None:
            device = next(model.parameters()).device
 
        if noise is None:
            img = torch.randn(*shape, device=device)
        else:
            img = noise
        print('self.scheduled_t', self.scheduled_t)
        for i, step in enumerate(self.scheduled_t):
            t = torch.tensor([step] * batch_size, device=device)  # step：1~0
            with torch.no_grad():
                if cond_vary:
                    if dt == 1:
                        t_mapped = torch.tensor([self.timestep_map[i]]*batch_size, device=device).long()
                    else:
                        t_mapped = torch.tensor([(self.timestep_map[i])/1000.0]*batch_size, device=device)
 
                    t_tmp = (t2k * t_mapped).clamp(0, 1.0)
                    k = diti.to_indices(t_tmp)

                    t = self.shift_t(t, self.shift)  # => 512 noise t

                    if self.is_eval == False:
                        encoder_hidden_states, _, _, mask, _, _, _ = encoder(x=x_0, hidden_states=ori_hidden_states, d=k)
                    else:
                        encoder_hidden_states, _, _, mask, _, _, _ = encoder(x=x_0, d=k,kwargs=kwargs)
                    # encoder_hidden_states, _, _, mask, _, _, _ = encoder(x=x_0, hidden_states=ori_hidden_states, d=k)
                    if self.cut_of_k is not None and self.cut_of_k < 1:
                        padding_size = K - encoder_hidden_states.shape[1]
                        padding_tensor = torch.zeros(encoder_hidden_states.shape[0], padding_size, encoder_hidden_states.shape[2]).cuda()
                        encoder_hidden_states = torch.cat((encoder_hidden_states, padding_tensor), dim=1)
                        padding_mask = torch.zeros(mask.shape[0], padding_size).cuda().bool()
                        mask = torch.cat((mask, padding_mask), dim=1)
                        super_mask_1 = torch.cat((super_mask, padding_mask), dim=1)
                    else:
                        super_mask_1 = super_mask
                        
                    if super_mask is not None:
                        mask = mask * super_mask_1

                    model_kwargs['encoder_hidden_states'] = encoder_hidden_states
                    model_kwargs['mask'] = mask

                    if encoder_hidden_states.sum() == 0 and dit is not None:
                        print("No condition is given...")
                        model_kwargs = {
                            'y': torch.tensor([1000] * len(x_0)).to(x_0.device)
                        }
                        model_to_use = dit
                    else:
                        model_to_use = model
                else:
                    model_to_use = model
 
                img, pred_x0 = self.sample_one_step(
                    model_to_use,
                    img,
                    t,
                    index=i,
                    model_kwargs=model_kwargs,
                    cfg_scale=uncond_scale,
                    uncond_y=uncond_y,
                    uc=uncond_c,
                    **kwargs,
                )

        return img
 
    def sample_one_step(
        self,
        model,
        x,
        t,
        index,
        model_kwargs=None,
        cfg_scale=1.0,
        uncond_y=None,
        uc=None,
        **kwargs,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        b, *_, device = *x.shape, x.device
        a_t = torch.full((b, 1, 1, 1), self.scheduled_t[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), self.scheduled_t_prev[index], device=device)
        # import pdb; pdb.set_trace()
        if uc is None or cfg_scale == 1.0:
            if self.is_eval == True:
                x = x.float()
            # import pdb; pdb.set_trace()
            out, _ = model(x, t, **model_kwargs)
        else:
            seq_len = uc.size(1)
            context = model_kwargs['encoder_hidden_states']
            ori_mask = model_kwargs['mask']
            uncond_mask = torch.zeros(ori_mask.size(), dtype=torch.int, device=ori_mask.device)
            uncond_mask[:,:seq_len] = 1
            
            out_uncond = model.cfg_inference(x, t, None, None, mask = uncond_mask, shape=context.shape[1])
            out = model(x, t, None, context, mask = ori_mask, shape=context.shape[1])
            out = out_uncond + cfg_scale * (out - out_uncond)
            
        img, pred_x0 = self.base_step(
            x, out, a_t=a_t, a_prev=a_prev, **kwargs
        )
        return img, pred_x0
    
    def base_step(self, x, v, a_t, a_prev,**kwargs):
        # Base sampler uses Euler numerical integrator.
        x_prev, pred_x0 = self.euler_step(x, v, a_t, a_prev)
        return x_prev, pred_x0
 
    def euler_step(self, x, v, a_t, a_prev, **kwargs):
        if self.parameterization == "velocity":
            x_prev = x - (a_t - a_prev) * v
            pred_x0 = x - a_t * v
        elif self.parameterization == "x0":
            #x_prev = x - (a_t - a_prev) * (noise-v)
            x_prev = v + a_prev * (x - v) / a_t
            pred_x0 = v
            
        return x_prev, pred_x0
        