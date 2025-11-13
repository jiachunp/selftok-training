from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.cuda.amp import autocast
import torch.distributed as dist
from einops import rearrange, repeat, reduce, pack, unpack
import numpy as np
from typing import Callable
from .code_distributor import CodeDistributor

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    pass


def identity(t):
    return t


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def cdist(x, y):
    x2 = reduce(x**2, "b n d -> b n", "sum")
    y2 = reduce(y**2, "b n d -> b n", "sum")
    xy = einsum("b i d, b j d -> b i j", x, y) * -2
    return (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy).clamp(min=0).sqrt()


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith("mps:")

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def calc_entropy(input_tensor, min_ref=None):
    assert len(input_tensor.shape) == 2
    p = input_tensor.softmax(dim=-1)
    # H(E(p))
    ap = p.mean(dim=0)
    p_log_p = ap * torch.log(ap)
    entropy_to_max = -p_log_p.sum(dim=-1)
    # E(H(p))
    p_log_p = p * torch.log(p)
    entropy_to_min = -p_log_p.sum(dim=-1)
    # if min_ref:
    #     entropy_to_min = torch.maximum(entropy_to_min, torch.ones_like(entropy_to_min) * min_ref)
    entropy_to_min = entropy_to_min.mean()
    return entropy_to_max, entropy_to_min

def calc_codebook_entropy(distances):
    p = distances.softmax(dim=2)
    p = p[0].mean(dim=0)
    p_log_p = p * torch.log(p)
    entropy_to_max = -p_log_p.sum(dim=0).mean()
    return entropy_to_max

def smooth_list(data, window_size=3):
    smoothed_data = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed_data.append(sum(data[start:end]) / (end - start))
    return smoothed_data

def calc_ema_entropy(distances, onehot_distances, ratio_d=0.3):
    p = distances.softmax(dim=-1)
    ap = p[0].mean(dim=0)
    ema_p = onehot_distances * (1-ratio_d) + ap * ratio_d
    p_log_p = ema_p * torch.log(ema_p)
    entropy_to_max = -p_log_p.sum(dim=-1).mean()
    ema_p_group = torch.stack([t.mean(dim=0) for t in ema_p.tensor_split(64, dim=0)],dim=0)
    p_log_p = ema_p_group * torch.log(ema_p_group)
    entropy_to_max2 = -p_log_p.sum(dim=-1).mean()
    return entropy_to_max, entropy_to_max2

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(
    logits, temperature=1.0, stochastic=False, straight_through=False, reinmax=False, dim=-1, training=True
):
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim=dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    assert not (
        reinmax and not straight_through
    ), "reinmax can only be turned on if using straight through gumbel softmax"

    if not straight_through or temperature <= 0.0 or not training:
        return ind, one_hot

    # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
    # algorithm 2

    if reinmax:
        π0 = logits.softmax(dim=dim)
        π1 = (one_hot + (logits / temperature).softmax(dim=dim)) / 2
        π1 = ((log(π1) - logits).detach() + logits).softmax(dim=1)
        π2 = 2 * π1 - 0.5 * π0
        one_hot = π2 - π2.detach() + one_hot
    else:
        π1 = (logits / temperature).softmax(dim=dim)
        one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def sample_vectors(samples, num, p=None):
    t = torch.zeros((5,), device="cuda", dtype=torch.int64)
    num_samples, device = samples.shape[0], samples.device
    if p is not None:
        indices = np.random.choice(len(samples), size=num, p=p.cpu().numpy(), replace=True)
        indices = torch.from_numpy(indices).clamp(0, len(samples - 1))
    else:
        if num_samples >= num:
            indices = torch.randperm(num_samples, device=device)[:num]
        else:
            indices = torch.randint(0, num_samples, (num,), device=device)
    t = torch.zeros((5,), device="cuda", dtype=torch.int64)
    return samples[indices]


def batched_sample_vectors(samples, num, p=None):
    return torch.stack([sample_vectors(sample, num, p) for sample in samples.unbind(dim=0)], dim=0)


def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]


def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)


def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)


def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def all_gather_variably_sized_v2(x, sizes, dim=0):
    t = torch.zeros((5,), device="cuda", dtype=torch.int64)
    device = x.device
    q = x
    ws = distributed.get_world_size()
    local_size = torch.tensor(q.shape[0], device=device)
    # all_sizes = sizes
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)
    t = torch.zeros((5,), device="cuda", dtype=torch.int64)

    size_diff = max_size.item() - x.shape[0]
    if size_diff:
        padding = torch.zeros((size_diff, x.shape[-1]), device=device, dtype=q.dtype)
        q = torch.cat((q, padding), dim=0)

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        all_qs.append(q[:size])
    t = torch.zeros((5,), device="cuda", dtype=torch.int64)
    return all_qs

def sample_vectors_distributed(local_samples, num, p=None):
    local_samples = rearrange(local_samples, "1 ... -> ...")

    rank = distributed.get_rank()

    # all_num_samples = all_gather_sizes(local_samples, dim=0)
    # if rank == 0:
    #     samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    # else:
    #     samples_per_rank = torch.empty_like(all_num_samples)
    # distributed.broadcast(samples_per_rank, src=0)
    # samples_per_rank = samples_per_rank.tolist()

    wolrd_size = distributed.get_world_size()
    num_per_rank = num // wolrd_size
    remainder = num % wolrd_size
    if rank < remainder:
        samples_per_rank = num_per_rank + 1
    else:
        samples_per_rank = num_per_rank
    #local_samples = sample_vectors(local_samples, samples_per_rank[rank], p)
    local_samples = sample_vectors(local_samples, samples_per_rank, p)
    all_samples = all_gather_variably_sized_v2(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return rearrange(out, "... -> 1 ...")


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
    samples, num_clusters, num_iters=10, use_cosine_sim=False, sample_fn=batched_sample_vectors, all_reduce_fn=noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, "h n d -> h d n")
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, "h n -> h n d", d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, "... -> ... 1")
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(rearrange(zero_mask, "... -> ... 1"), means, new_means)

    return means, bins


def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, "h b n -> h b n d", d=dim)
    embeds = repeat(embeds, "h c d -> h b c d", b=batch)
    return embeds.gather(2, indices)


# regularization losses


def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum("h i d, h j d -> h i j", normed_codes, normed_codes)
    return (cosine_sim**2).sum() / (h * n**2) - (1 / n)


# distance types
class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        reset_cluster_size = None,
        use_ddp = False,
        learnable_codebook = False,
        gumbel_sample = gumbel_sample,
        sample_codebook_temp = 1.,
        ema_update = True,
        if_force_sync = False,
        smart_re_K=0,
        redistribute=False,
        frozen_embed=None,
    ):
        super().__init__()
        self.transform_input = l2norm

        self.ema_update = ema_update
        self.decay = decay
        self.if_force_sync = if_force_sync
        self.smart_react_K = smart_re_K

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)
        self.dead_code_threshold_updated = False

        assert callable(gumbel_sample)
        self.gumbel_sample = gumbel_sample
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('cluster_size_wo_react', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())
        if self.smart_react_K > 0:
            self.register_buffer(
                'timestep_p_over_c',
                torch.ones(num_codebooks, self.smart_react_K, codebook_size) / codebook_size
            )
            self.register_buffer('tpc_initted', torch.Tensor([False]))

        if frozen_embed is not None:
            self.frozen_embed = frozen_embed
            self.n_frozen = frozen_embed.shape[1]
        else:
            self.n_frozen = 0

        self.redistribute = redistribute
        if redistribute:
            self.distributor = CodeDistributor(codebook_size, codebook_size // 1000)

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

        # frozen embed
        self.reset_frozen_embed()

    def force_sync(self, name):
        if distributed.get_rank() == 0:
            var = getattr(self, name)
        else:
            var = torch.empty_like(getattr(self, name))
        distributed.broadcast(var, src=0)
        setattr(self, name, var)

    def reset_frozen_embed(self,):
        if self.n_frozen > 0:
            self.embed[:, :self.n_frozen, :].data.copy_(self.frozen_embed)
            self.embed_avg[:, :self.n_frozen, :].data.copy_(self.frozen_embed)

    @torch.jit.ignore
    def init_embed_(self, data, mask = None):
        if self.initted:
            return

        if exists(mask):
            c = data.shape[0]
            data = rearrange(data[mask], '(c n) d -> c n d', c = c)

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        embed_sum = embed * rearrange(cluster_size, '... -> ... 1')
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed_sum)
        self.cluster_size.data.copy_(cluster_size)
        self.cluster_size_wo_react.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        self.reset_frozen_embed()

    def compute_timestep_weight(self):
        # timestep_p_over_c
        ap = self.timestep_p_over_c
        perplexity = torch.exp(-torch.sum(ap * torch.log(ap + 1e-10), dim=-1))
        weight = 1 / perplexity
        v, _ = weight.max(dim=-1)
        weight = weight / v * 10.0
        weight = weight.softmax(dim=-1)
        return weight
    
    def get_group_perplexity(self, codebook_idx=0):
        ap = self.timestep_p_over_c[codebook_idx]
        group_perplexity = torch.exp(-torch.sum(ap * torch.log(ap + 1e-10), dim=-1))
        return group_perplexity
    
    def fix_code(self, change_mask_or_indices, new_codes):
        if len(change_mask_or_indices) == self.codebook_size and \
            (change_mask_or_indices<=1).sum()==len(change_mask_or_indices):
            # is mask
            indices = change_mask_or_indices.nonzero()[:,0]
        else:
            # is indices
            indices = change_mask_or_indices
        if len(indices) == len(new_codes):
            # match
            return indices, new_codes
        # shape does not match
        print(f"Warning: change mask len {len(indices)} does not match codes len {len(new_codes)}...")
        if len(indices) > len(new_codes):
            indices = indices[:len(new_codes)]
        else:
            new_codes = new_codes[:len(indices)]
        return indices, new_codes
    
    def change_code(self, change_mask_or_indices, new_codes, ind=0):
        if change_mask_or_indices is None:
            return
        change_mask_or_indices, new_codes = self.fix_code(change_mask_or_indices, new_codes)
        self.embed.data[ind][change_mask_or_indices] = new_codes
        self.embed_avg.data[ind][change_mask_or_indices] = new_codes * self.reset_cluster_size
        self.cluster_size.data[ind][change_mask_or_indices] = self.reset_cluster_size

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)
        if self.smart_react_K > 0:
            batch_weights = self.compute_timestep_weight()     # n_codebook * K
            b = batch_samples.shape[1] // batch_weights.shape[1]
            batch_weights = batch_weights.unsqueeze(1).expand(-1,b,-1)
            batch_weights = batch_weights / b
            batch_weights = rearrange(batch_weights, "h ... -> h (...)")
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue
            if self.smart_react_K > 0:
                p = batch_weights[ind]
            else:
                p = None
            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item(), p=p)
            sampled = rearrange(sampled, '1 ... -> ...')
            self.change_code(mask, sampled, ind)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return 0

        # previous attempts to force sync the clustered size before expiring
        # if self.if_force_sync:
        #     rank = distributed.get_rank()
        #     if rank == 0:
        #         expired_codes = self.cluster_size < self.threshold_ema_dead_code
        #     else:
        #         expired_codes = self.cluster_size < 100.0
        #     distributed.broadcast(expired_codes, src=0)
        # else:
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if self.n_frozen > 0:
            # frozen codes do not need to be reactivated
            non_frozen_codes = (torch.arange(self.codebook_size) >= self.n_frozen)
            non_frozen_codes = non_frozen_codes.unsqueeze(dim=0).expand(self.num_codebooks, -1).cuda()
            expired_codes = torch.logical_and(expired_codes, non_frozen_codes)

        if not torch.any(expired_codes):
            return 0
        
        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)
        # print(torch.sum(expired_codes), expired_codes.size())
        return torch.sum(expired_codes).item()

    @autocast(enabled = False)
    def forward(
        self,
        x,
        sample_codebook_temp = None,
        mask = None,
        freeze_codebook = False
    ):
               
        num_reactivate = 0
        needs_codebook_dim = x.ndim < 4
        sample_codebook_temp = default(sample_codebook_temp, self.sample_codebook_temp)

        # update relative dead code threshold to absolute value based on batch size and world size
        if self.training and not self.dead_code_threshold_updated:
            ratio = x.shape[0] * x.shape[1] * distributed.get_world_size() / self.codebook_size
            self.threshold_ema_dead_code = ratio * self.threshold_ema_dead_code
            self.reset_cluster_size = ratio * self.reset_cluster_size
            self.dead_code_threshold_updated = True
            print(f"Dead code threshold updated to {self.threshold_ema_dead_code}, reset size to {self.reset_cluster_size}.")

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        dtype = x.dtype

        flatten, ps = pack_one(x, 'h * d')

        if exists(mask):
            mask = repeat(mask, 'b n -> c (b h n)', c = flatten.shape[0], h = flatten.shape[-2] // (mask.shape[0] * mask.shape[1]))

        self.init_embed_(flatten, mask = mask)

        if self.if_force_sync:
            self.force_sync('embed')
        embed = self.embed if self.learnable_codebook else self.embed.detach()

        dist = einsum('h n d, h c d -> h n c', flatten, embed)

        embed_ind, embed_onehot = self.gumbel_sample(dist, dim = -1, temperature = sample_codebook_temp, training = self.training)
        embed_ind = unpack_one(embed_ind, ps, 'h *')

        if self.training:
            unpacked_onehot = unpack_one(embed_onehot, ps, 'h * c')
            quantize = einsum('h b n c, h c d -> h b n d', unpacked_onehot, embed)

            # update timestep_p_over_c
            if self.smart_react_K > 0:
                batch_t_p_over_c = unpacked_onehot.mean(dim=1)
                self.all_reduce_fn(batch_t_p_over_c)
                batch_t_p_over_c /= distributed.get_world_size()
                decay = self.decay if self.tpc_initted else 0.3
                ema_inplace(self.timestep_p_over_c.data, batch_t_p_over_c, decay)
                if not self.tpc_initted:
                    self.tpc_initted.data.copy_(torch.Tensor([True]))
        else:
            quantize = batched_embedding(embed_ind, embed)

        self.delta_embed = torch.tensor(0.0).to(x.device)
        if self.training and self.ema_update and not freeze_codebook:
            if self.redistribute:
                # print("Start redistributing...")
                bad_indices, new_codes, avg_score = self.distributor.update(flatten, quantize, dist)
                new_codes = l2norm(new_codes) if new_codes is not None else new_codes

            if exists(mask):
                embed_onehot[~mask] = 0.

            bins = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size.data, bins, self.decay)
            ema_inplace(self.cluster_size_wo_react.data, bins, self.decay)
            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            embed_sum = embed_sum.contiguous()
            self.all_reduce_fn(embed_sum)

            ema_inplace(self.embed_avg.data, embed_sum, self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum(dim = -1, keepdim = True)

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            # compute embed changes on non-frozen codes
            if self.n_frozen > 0:
                embed_normalized[:, :self.n_frozen, :].data.copy_(self.frozen_embed)
            self.delta_embed = F.mse_loss(self.embed.data, embed_normalized, reduction='sum')    # avg update

            # update codebook
            self.embed.data.copy_(l2norm(embed_normalized))

            # redistribute
            if self.redistribute:
                self.change_code(bad_indices, new_codes)

            num_reactivate = self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        # reset frozen embed
        self.reset_frozen_embed()
        
        dist = unpack_one(dist, ps, 'h * d')
        return quantize, embed_ind, dist, num_reactivate

# main class
class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        output_dim=None,
        codebook_dim = None,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        freeze_codebook = False,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        diversity_weight = 0.,
        commitment_use_cross_entropy_loss = False,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        stochastic_sample_codes = False,
        sample_codebook_temp = 1.,
        straight_through = False,
        reinmax = False,  # using reinmax for improved straight-through, assuming straight through helps at all
        sync_codebook = None,
        sync_affine_param = False,
        ema_update = True,
        learnable_codebook = False,
        in_place_codebook_optimizer: Callable[..., Optimizer] = None, # Optimizer used to update the codebook embedding if using learnable_codebook
        affine_param = False,
        affine_param_batch_decay = 0.99,
        affine_param_codebook_decay = 0.9,
        sync_update_v = 0., # the v that controls optimistic vs pessimistic update for synchronous update rule (21) https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
        if_force_sync = False,
        n_partitions=1, # when calc entropy loss, how to randomly partition data
        smart_re_K=0,
        redistribute=False,
        continuous=False,
        reg=[1/4., 1/2.],
        reset_cluster_size=None,
        ema_entropy_ratio=0.7,
        frozen_embed=None,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        output_dim = output_dim or dim
        requires_out_projection = codebook_input_dim != output_dim
        self.project_out = nn.Linear(codebook_input_dim, output_dim) if requires_out_projection else nn.Identity()

        self.has_projections = requires_projection

        self.eps = eps
        self.reg = reg
        self.ema_entropy_ratio = ema_entropy_ratio
        self.diversity_weight = diversity_weight
        self.commitment_weight = commitment_weight
        self.commitment_use_cross_entropy_loss = commitment_use_cross_entropy_loss # whether to use cross entropy loss to codebook as commitment loss
        # calculate reference entropy value
        a1 = codebook_size // codebook_dim
        ref = torch.tensor([0.0]*codebook_size)
        ref[:a1] = 1.0 * 10.0   # scaled positive logits
        ref[a1:] = 0.38 * 10.0   # scaled negative logits
        _, entropy_min_ref = calc_entropy(ref.unsqueeze(0))
        self.entropy_min_ref = entropy_min_ref.item()
        # print(self.entropy_min_ref)

        self.learnable_codebook = learnable_codebook
        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.has_codebook_orthogonal_loss = has_codebook_orthogonal_loss
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        assert not (ema_update and learnable_codebook), 'learnable codebook not compatible with EMA update'

        assert 0 <= sync_update_v <= 1.
        assert not (sync_update_v > 0. and not learnable_codebook), 'learnable codebook must be turned on'
        self.smart_re_K = smart_re_K
        self.sync_update_v = sync_update_v

        codebook_class = CosineSimCodebook

        gumbel_sample_fn = partial(
            gumbel_sample,
            stochastic = stochastic_sample_codes,
            reinmax = reinmax,
            straight_through = straight_through
        )

        if not exists(sync_codebook):
            sync_codebook = distributed.is_initialized() and distributed.get_world_size() > 1

        codebook_kwargs = dict(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            reset_cluster_size=reset_cluster_size,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook,
            sample_codebook_temp = sample_codebook_temp,
            gumbel_sample = gumbel_sample_fn,
            ema_update = ema_update,
            if_force_sync = if_force_sync,
            smart_re_K=smart_re_K,
            redistribute=redistribute,
            frozen_embed=frozen_embed,
        )

        if affine_param:
            assert not use_cosine_sim, 'affine param is only compatible with euclidean codebook'
            codebook_kwargs = dict(
                **codebook_kwargs,
                affine_param = True,
                sync_affine_param = sync_affine_param,
                affine_param_batch_decay = affine_param_batch_decay,
                affine_param_codebook_decay = affine_param_codebook_decay,
            )

        self._codebook = codebook_class(**codebook_kwargs)

        self.in_place_codebook_optimizer = in_place_codebook_optimizer(self._codebook.parameters()) if exists(in_place_codebook_optimizer) else None

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        # continuous tricks
        self.register_buffer('continuous', torch.Tensor([continuous]))
        self.register_buffer('steps', torch.Tensor([0]))
        # self.continuous = False
        # self.steps = 0

        self.frozen_embed = frozen_embed

    @property
    def codebook(self):
        codebook = self._codebook.embed

        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    @codebook.setter
    def codebook(self, codes):
        if not self.separate_codebook_per_head:
            codes = rearrange(codes, '... -> 1 ...')

        self._codebook.embed.copy_(codes)

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')

        indices, ps = pack_one(indices, 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        indices = repeat(indices, 'b h n -> b h n d', d = codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b = indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes = unpack_one(codes, ps, 'b * d')
        return codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        return self.project_out(codes)

    def forward(
        self,
        x,
        indices = None,
        mask = None,
        sample_codebook_temp = None,
        freeze_codebook = False
    ):

        orig_input = x

        only_one = x.ndim == 2

        if only_one:
            assert not exists(mask)
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size, return_loss = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size, exists(indices)

        need_transpose = not self.channel_last and not self.accept_image_fmap
        should_inplace_optimize = exists(self.in_place_codebook_optimizer)

        # rearrange inputs

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        # project input

        x = self.project_in(x)

        # handle multi-headed separate codebooks

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = heads)

        # l2norm for cosine sim, otherwise identity

        x = self._codebook.transform_input(x)

        # codebook forward kwargs

        codebook_forward_kwargs = dict(
            sample_codebook_temp = sample_codebook_temp,
            mask = mask,
            freeze_codebook = freeze_codebook
        )

        self.steps += 1
        if self.steps > 2000 and self.continuous:
            self.continuous.data.copy_(torch.Tensor([False]))
            print("Starting quantizer mode...")

        # quantize
        if self.continuous:
            quantize = x
            embed_ind = torch.randint(0, self.codebook_size, size=quantize.shape[:2]).to(dtype=torch.int64, device=x.device)
            distances = torch.ones((1, quantize.shape[0], quantize.shape[1], self.codebook_size)).to(device=x.device)
            num_reactivate = 0
        else:
            quantize, embed_ind, distances, num_reactivate = self._codebook(x, **codebook_forward_kwargs)
        
        # one step in-place update

        if should_inplace_optimize and self.training and not freeze_codebook:

            if exists(mask):
                loss = F.mse_loss(quantize, x.detach(), reduction = 'none')

                loss_mask = mask
                if is_multiheaded:
                    loss_mask = repeat(mask, 'b n -> c (b h) n', c = loss.shape[0], h = loss.shape[1] // mask.shape[0])

                loss = loss[loss_mask].mean()

            else:
                loss = F.mse_loss(quantize, x.detach())

            loss.backward()
            self.in_place_codebook_optimizer.step()
            self.in_place_codebook_optimizer.zero_grad()

            # quantize again

            quantize, embed_ind, distances = self._codebook(x, **codebook_forward_kwargs)

        if self.training:
            # determine code to use for commitment loss
            maybe_detach = torch.detach if not self.learnable_codebook or freeze_codebook else identity

            commit_quantize = maybe_detach(quantize)            

            # straight through

            quantize = x + (quantize - x).detach()

            w_token_list = [1.0, 1.0009568929672241, 1.0020943880081177, 1.0030533075332642, 1.0041248798370361, 1.0051463842391968, 1.0061800479888916, 1.0072035789489746, 1.0082881450653076, 1.0093486309051514, 1.0103622674942017, 1.0114312171936035, 1.0124040842056274, 1.0133970975875854, 1.0144168138504028, 1.0155006647109985, 1.016545295715332, 1.0176666975021362, 1.0187615156173706, 1.019839882850647, 1.020947813987732, 1.0220266580581665, 1.0231518745422363, 1.024350881576538, 1.025453805923462, 1.0265865325927734, 1.0277132987976074, 1.0288679599761963, 1.0299466848373413, 1.0310660600662231, 1.0321664810180664, 1.033187985420227, 1.0344022512435913, 1.0354132652282715, 1.036499261856079, 1.037566065788269, 1.0386524200439453, 1.039673924446106, 1.0408188104629517, 1.0419228076934814, 1.0430878400802612, 1.0442489385604858, 1.0453472137451172, 1.0465483665466309, 1.047695279121399, 1.048811674118042, 1.0499590635299683, 1.0510671138763428, 1.0521841049194336, 1.0533123016357422, 1.0544207096099854, 1.0556384325027466, 1.0568546056747437, 1.0580308437347412, 1.0591963529586792, 1.06035315990448, 1.061476469039917, 1.062683343887329, 1.063859224319458, 1.0651668310165405, 1.0663436651229858, 1.067491054534912, 1.0687140226364136, 1.069914698600769, 1.0711547136306763, 1.0722273588180542, 1.0734590291976929, 1.074573278427124, 1.0758819580078125, 1.0770639181137085, 1.0782252550125122, 1.0794496536254883, 1.0806183815002441, 1.0819114446640015, 1.0831443071365356, 1.0843870639801025, 1.0856066942214966, 1.0868573188781738, 1.088016152381897, 1.089324712753296, 1.090526819229126, 1.091760277748108, 1.0930418968200684, 1.0941996574401855, 1.095448613166809, 1.096611499786377, 1.097791314125061, 1.099048376083374, 1.100354552268982, 1.1015594005584717, 1.1028227806091309, 1.1041158437728882, 1.1054145097732544, 1.106608510017395, 1.1078245639801025, 1.1090679168701172, 1.1102895736694336, 1.1115607023239136, 1.1128051280975342, 1.1141241788864136, 1.1154638528823853, 1.1167494058609009, 1.1180005073547363, 1.1193571090698242, 1.1205862760543823, 1.1219818592071533, 1.1233128309249878, 1.124659538269043, 1.1260524988174438, 1.1273854970932007, 1.1286910772323608, 1.130017638206482, 1.1313958168029785, 1.1327131986618042, 1.1339514255523682, 1.1352671384811401, 1.1366374492645264, 1.1380527019500732, 1.139320731163025, 1.1406384706497192, 1.1420217752456665, 1.1435235738754272, 1.1448878049850464, 1.1461790800094604, 1.1476049423217773, 1.149084448814392, 1.1504831314086914, 1.151869297027588, 1.1531658172607422, 1.1546757221221924, 1.1560720205307007, 1.1575226783752441, 1.1589312553405762, 1.1602435111999512, 1.1616073846817017, 1.1630990505218506, 1.1645294427871704, 1.1659849882125854, 1.1674442291259766, 1.1689262390136719, 1.1704283952713013, 1.1719647645950317, 1.173458218574524, 1.1748837232589722, 1.1764262914657593, 1.1778507232666016, 1.1792536973953247, 1.18085777759552, 1.1822901964187622, 1.1837457418441772, 1.1852694749832153, 1.186768889427185, 1.1883258819580078, 1.1899861097335815, 1.191559910774231, 1.193055510520935, 1.194466233253479, 1.19606351852417, 1.1974470615386963, 1.1989630460739136, 1.2004023790359497, 1.2018768787384033, 1.2034245729446411, 1.204888939857483, 1.2063862085342407, 1.2079278230667114, 1.2094297409057617, 1.2110143899917603, 1.2125121355056763, 1.2140401601791382, 1.215444803237915, 1.2169981002807617, 1.2186620235443115, 1.220271110534668, 1.2217799425125122, 1.2233014106750488, 1.2248388528823853, 1.2263920307159424, 1.2280033826828003, 1.2295526266098022, 1.2312754392623901, 1.2327719926834106, 1.2342661619186401, 1.2358893156051636, 1.2373847961425781, 1.2389516830444336, 1.2405931949615479, 1.2422577142715454, 1.2439018487930298, 1.245419979095459, 1.2471191883087158, 1.2487512826919556, 1.2504281997680664, 1.252018928527832, 1.2536324262619019, 1.2552154064178467, 1.2569636106491089, 1.2586628198623657, 1.2604268789291382, 1.2621513605117798, 1.263947606086731, 1.2656913995742798, 1.2674978971481323, 1.2691096067428589, 1.2707901000976562, 1.2724945545196533, 1.2742295265197754, 1.2760440111160278, 1.2778180837631226, 1.279642939567566, 1.2815321683883667, 1.2833116054534912, 1.2850233316421509, 1.2869150638580322, 1.2887295484542847, 1.2905024290084839, 1.2921432256698608, 1.293999195098877, 1.2958202362060547, 1.2976800203323364, 1.2995046377182007, 1.3012598752975464, 1.3030503988265991, 1.3048561811447144, 1.3068581819534302, 1.3086538314819336, 1.3104888200759888, 1.3123188018798828, 1.3141018152236938, 1.3160250186920166, 1.3178668022155762, 1.319700002670288, 1.3216639757156372, 1.323539137840271, 1.3257288932800293, 1.327608585357666, 1.3295114040374756, 1.331291913986206, 1.3332159519195557, 1.335198998451233, 1.3371057510375977, 1.3389317989349365, 1.3409823179244995, 1.3429380655288696, 1.344834327697754, 1.3467687368392944, 1.348690390586853, 1.350522756576538, 1.3524699211120605, 1.3545546531677246, 1.356642246246338, 1.358640193939209, 1.3605812788009644, 1.3624423742294312, 1.3645058870315552, 1.366642951965332, 1.3687305450439453, 1.3708282709121704, 1.3729512691497803, 1.3750202655792236, 1.3772056102752686, 1.3792989253997803, 1.3812459707260132, 1.3833975791931152, 1.3855290412902832, 1.3877325057983398, 1.389730453491211, 1.3918814659118652, 1.393957495689392, 1.395930528640747, 1.398061752319336, 1.400097370147705, 1.4023396968841553, 1.4044510126113892, 1.4064775705337524, 1.408649206161499, 1.4108312129974365, 1.4130840301513672, 1.4153640270233154, 1.417559027671814, 1.4197285175323486, 1.4218764305114746, 1.4240186214447021, 1.4263218641281128, 1.428534746170044, 1.4307583570480347, 1.4330836534500122, 1.4352596998214722, 1.437330961227417, 1.4395323991775513, 1.4418696165084839, 1.4440850019454956, 1.4463156461715698, 1.4486497640609741, 1.4508861303329468, 1.4530153274536133, 1.4551550149917603, 1.457487940788269, 1.4596749544143677, 1.461954116821289, 1.4642446041107178, 1.466710090637207, 1.4691276550292969, 1.4714970588684082, 1.4739131927490234, 1.4763373136520386, 1.47867751121521, 1.4810601472854614, 1.4833625555038452, 1.4856722354888916, 1.488259196281433, 1.4907128810882568, 1.4929696321487427, 1.495322585105896, 1.497831106185913, 1.5003526210784912, 1.502760648727417, 1.505235195159912, 1.5077953338623047, 1.5102181434631348, 1.5126579999923706, 1.5149587392807007, 1.517529010772705, 1.5198723077774048, 1.522250771522522, 1.5247666835784912, 1.5273845195770264, 1.5300158262252808, 1.5323933362960815, 1.5350843667984009, 1.537827491760254, 1.5405566692352295, 1.5433670282363892, 1.5459343194961548, 1.548433542251587, 1.5510801076889038, 1.5535138845443726, 1.5561344623565674, 1.558846354484558, 1.561377763748169, 1.5637656450271606, 1.5664602518081665, 1.569154143333435, 1.5717982053756714, 1.5745551586151123, 1.577406406402588, 1.5800434350967407, 1.5828945636749268, 1.5854946374893188, 1.5883150100708008, 1.591094970703125, 1.5940014123916626, 1.5968064069747925, 1.5996723175048828, 1.602461338043213, 1.6051931381225586, 1.607851505279541, 1.6107624769210815, 1.6133716106414795, 1.6161773204803467, 1.618882656097412, 1.6217180490493774, 1.624647855758667, 1.6274399757385254, 1.6301672458648682, 1.632983684539795, 1.6360293626785278, 1.6388660669326782, 1.641459345817566, 1.6443527936935425, 1.647283673286438, 1.6501868963241577, 1.6531658172607422, 1.656111717224121, 1.659084677696228, 1.6621180772781372, 1.665062665939331, 1.6729905605316162, 1.6807061433792114, 1.688504695892334, 1.6964448690414429, 1.7042104005813599, 1.7121001482009888, 1.7204538583755493, 1.7287997007369995, 1.737027883529663, 1.7454748153686523, 1.7537890672683716, 1.762623906135559, 1.771485447883606, 1.7803224325180054, 1.7893697023391724, 1.7979402542114258, 1.8068413734436035, 1.81619393825531, 1.8252371549606323, 1.8340480327606201, 1.843508243560791, 1.8531970977783203, 1.862460970878601, 1.8720492124557495, 1.8814746141433716, 1.8913100957870483, 1.9013069868087769, 1.9116146564483643, 1.9218425750732422, 1.9321430921554565, 1.9422000646591187, 1.952713131904602, 1.9636720418930054, 1.9743727445602417, 1.9860105514526367, 1.9971718788146973, 2.0082578659057617, 2.019280195236206, 2.0304651260375977, 2.0420081615448, 2.05375075340271, 2.0654330253601074, 2.0769293308258057, 2.089104652404785, 2.100707769393921, 2.113440990447998, 2.1257688999176025, 2.1381959915161133, 2.151564836502075, 2.16473650932312, 2.1781747341156006, 2.191521406173706, 2.2054121494293213, 2.217973470687866, 2.2321228981018066, 2.2457780838012695, 2.260101556777954, 2.2741849422454834, 2.289262056350708, 2.304327964782715, 2.318969488143921, 2.3334717750549316, 2.3491604328155518, 2.3647594451904297, 2.380385637283325, 2.396231174468994, 2.4123589992523193, 2.4286816120147705, 2.4453704357147217, 2.461902141571045, 2.4784252643585205, 2.4960439205169678, 2.522716999053955, 2.5498886108398438, 2.5773861408233643, 2.605184316635132, 2.634171485900879, 2.664407253265381, 2.6936607360839844, 2.724647283554077, 2.755413055419922, 2.7875185012817383, 2.8198084831237793, 2.853832483291626, 2.8880200386047363, 2.924318552017212, 2.960191249847412, 2.995321273803711, 3.0327844619750977, 3.0716869831085205, 3.112046241760254, 3.1548526287078857, 3.197462558746338, 3.2406296730041504, 3.2851293087005615, 3.329404592514038, 3.375641345977783, 3.423063278198242, 3.473259210586548, 3.524651527404785, 3.577868700027466, 3.6328232288360596, 3.6885669231414795, 3.747142791748047, 3.806072950363159, 3.866079092025757, 3.928964376449585, 3.992462158203125, 4.06173849105835, 4.132299423217773, 4.207337379455566, 4.283389091491699, 4.366011619567871, 4.444562911987305, 4.529560089111328, 4.6137800216674805, 4.703934192657471, 4.797405242919922, 4.891745567321777, 4.992710590362549, 5.3271965980529785, 5.713437080383301, 6.160442352294922, 6.674854278564453, 7.280563831329346, 7.990283966064453, 8.869966506958008, 9.989411354064941, 11.411096572875977, 13.296104431152344, 15.942859649658203, 19.8294677734375, 26.494277954101562, 39.354583740234375, 78.54225158691406]
            w_token = torch.tensor(w_token_list).cuda()
            w_token_ratio = 0.5
            w_token = w_token_ratio * w_token
            w_token = w_token.to(quantize.dtype).detach()

            quantize_reweight = quantize * w_token.unsqueeze(0).unsqueeze(2)  # 将 w 扩展为 [1, 512, 1]
            quantize = quantize_reweight + (quantize - quantize_reweight).detach()

            if self.sync_update_v > 0.:
                # (21) in https://minyoungg.github.io/vqtorch/assets/draft_050523.pdf
                quantize = quantize + self.sync_update_v * (quantize - quantize.detach())

        # function for calculating cross entropy loss to distance matrix
        # used for (1) naturalspeech2 training residual vq latents to be close to the correct codes and (2) cross-entropy based commitment loss

        def calculate_ce_loss(codes):
            if not is_multiheaded:
                dist_einops_eq = '1 b n l -> b l n'
            elif self.separate_codebook_per_head:
                dist_einops_eq = 'c b n l -> b l n c'
            else:
                dist_einops_eq = '1 (b h) n l -> b l n h'

            ce_loss = F.cross_entropy(
                rearrange(distances, dist_einops_eq, b = shape[0]),
                codes,
                ignore_index = -1
            )

            return ce_loss

        # if returning cross entropy loss on codes that were passed in

        if return_loss:
            return quantize, calculate_ce_loss(indices)

        # transform embedding indices

        if is_multiheaded:
            if self.separate_codebook_per_head:
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h = heads)
            else:
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h = heads)

        if self.accept_image_fmap:
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width)

        if only_one:
            embed_ind = rearrange(embed_ind, 'b 1 ... -> b ...')

        # aggregate loss

        loss = torch.tensor([0.], device = device, requires_grad = self.training)
        log_dict = {
            "n_reactive": num_reactivate,
            "commit_loss": 0,
            "diversity_entropy": 0,
            "deterministic_entropy": 0,
            "perplexity": 0,
            "delta_embed": 0,
            "cosine_sim": einsum('h n d, h n d -> h n', quantize, x).mean().item()
        }
        if not self.continuous:
            log_dict["delta_embed"] = self._codebook.delta_embed.item()

        # additional logging
        total_codes = len(embed_ind.flatten()) * dist.get_world_size()
        log_dict['n_active'] = \
            (self._codebook.cluster_size_wo_react * self.codebook_size / total_codes > 0.2).sum().item()
        avg_probs = self._codebook.cluster_size_wo_react/total_codes    # num_codebooks * codebook_size
        avg_probs = avg_probs.mean(dim=0)   # mean remove number of codebooks
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
        log_dict['perplexity'] = perplexity

        if self.training and not (self.continuous):
            if self.commitment_weight > 0:
                if self.commitment_use_cross_entropy_loss:
                    if exists(mask):
                        ce_loss_mask = mask
                        if is_multiheaded:
                            ce_loss_mask = repeat(ce_loss_mask, 'b n -> b n h', h = heads)

                        embed_ind.masked_fill_(~ce_loss_mask, -1)

                    commit_loss = calculate_ce_loss(embed_ind)
                else:
                    if exists(mask):
                        # with variable lengthed sequences
                        commit_loss = F.mse_loss(commit_quantize, x, reduction = 'none')

                        loss_mask = mask
                        if is_multiheaded:
                            loss_mask = repeat(loss_mask, 'b n -> c (b h) n', c = commit_loss.shape[0], h = commit_loss.shape[1] // mask.shape[0])

                        commit_loss = commit_loss[loss_mask].mean()
                    else:
                        # commit_loss = F.mse_loss(commit_quantize, x)
                        commit_loss = F.mse_loss(commit_quantize, x, reduction='sum') / len(x.flatten())

                loss = loss + commit_loss * self.commitment_weight
                log_dict["commit_loss"] = commit_loss.detach().item()
            

            #all_distances = [torch.empty_like(distances) for _ in range(distributed.get_world_size())]
            #distributed.all_gather(all_distances, distances)
            #all_distances = torch.cat(all_distances, dim=1)
            all_distances = distances
            scaled_distances = all_distances * 10.0
            entropy_to_max, entropy_to_min = calc_entropy(
                scaled_distances.flatten(end_dim=-2), min_ref=self.entropy_min_ref
            )
            # codebook entropy
            if self.smart_re_K:
                # # codebook_entropy = calc_codebook_entropy(scaled_distances)
                # codebook_entropy, group_entropy = calc_ema_entropy(
                #     scaled_distances, self._codebook.timestep_p_over_c[0], ratio_d=1.-self.ema_entropy_ratio
                # )
                # entropy = 0.5 * (codebook_entropy + group_entropy)
                # # diversity_loss = -entropy_to_max
                # group_perplexity = self._codebook.get_group_perplexity().mean()
                # # entropy_weight = 0.1 if perplexity < 0.6 * self.codebook_size else 0.0
                # #codebook_ent_weight = min(self.codebook_size/group_perplexity/5.0, 2.0) if group_perplexity < 0.4 * self.codebook_size else 0.0
                # frac = group_perplexity / self.codebook_size
                # reg = self.reg
                # # reg = [0.64, 0.685]
                # codebook_ent_weight = 0.5 if frac < reg[0] else max((0.5 - 0.5/(reg[1]-reg[0])*(frac-reg[0])), 0.0)
                # log_dict['perplexity'] = group_perplexity.item()
                # # if torch.rand(1)<0.05:
                # #     print(f"group perplexity:{group_perplexity:.1f}, codebook_w:{codebook_ent_weight}, entropy_w:{entropy_weight}")
                # # diversity_loss = -self.diversity_weight * \
                # #     (entropy_weight*entropy_to_max+codebook_ent_weight*codebook_entropy)
                # diversity_loss = -self.diversity_weight*codebook_ent_weight*entropy

                # codebook_entropy = calc_codebook_entropy(scaled_distances)
                codebook_entropy, _ = calc_ema_entropy(
                    scaled_distances, self._codebook.timestep_p_over_c[0], ratio_d=1.-self.ema_entropy_ratio
                )
                
                # diversity_loss = -entropy_to_max
                group_perplexity = self._codebook.get_group_perplexity().mean()
                # entropy_weight = 0.1 if perplexity < 0.6 * self.codebook_size else 0.0
                #codebook_ent_weight = min(self.codebook_size/group_perplexity/5.0, 2.0) if group_perplexity < 0.4 * self.codebook_size else 0.0
                # frac = group_perplexity / self.codebook_size
                # reg = self.reg
                # reg = [0.64, 0.685]
                # codebook_ent_weight = 0.5 if frac < reg[0] else max((0.5 - 0.5/(reg[1]-reg[0])*(frac-reg[0])), 0.0)
                entropy_weight = 1. if perplexity < 0.6 * self.codebook_size else 0.0
                log_dict['perplexity'] = group_perplexity.item()
                
                diversity_loss = -(entropy_weight * self.diversity_weight * entropy_to_max)
                # diversity_loss = -(entropy_weight * self.diversity_weight * entropy_to_max + self.diversity_weight * entropy_to_max_ema)
                # diversity_loss = -self.diversity_weight * entropy_weight * entropy_to_max_ema
                
            else:
                diversity_loss = -self.diversity_weight * entropy_to_max
            log_dict["diversity_entropy"] = codebook_entropy.detach().item()
            log_dict["deterministic_entropy"] = entropy_to_min.detach().item()
            loss = loss + diversity_loss
            if self.has_codebook_orthogonal_loss:
                codebook = self._codebook.embed

                # only calculate orthogonal loss for the activated codes for this batch

                if self.orthogonal_reg_active_codes_only:
                    assert not (is_multiheaded and self.separate_codebook_per_head), 'orthogonal regularization for only active codes not compatible with multi-headed with separate codebooks yet'
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[:, unique_code_ids]

                num_codes = codebook.shape[-2]

                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device = device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[:, rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        # handle multi-headed quantized embeddings

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h = heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h = heads)

        # project out

        quantize = self.project_out(quantize)

        # rearrange quantized embeddings

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h = height, w = width)

        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')

        # if masking, only return quantized for where mask has True

        if exists(mask):
            quantize = torch.where(
                rearrange(mask, '... -> ... 1'),
                quantize,
                orig_input
            )
        return quantize, embed_ind, loss, log_dict