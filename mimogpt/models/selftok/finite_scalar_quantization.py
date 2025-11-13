"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple
import torch.nn.functional as F
import torch.distributed as distributed

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.amp import autocast

import einx
from einops import rearrange, pack, unpack

import random

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z):
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    zhat = z.floor()
    return z + (zhat - z).detach()

def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith("mps:")

    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


# main class

class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: int | None = None,
        output_dim = None,
        token_len = 1536,
        num_codebooks = 1,
        decay = 0.8,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry: bool = True,
        noise_dropout = 0.0,
        entropy_loss_weight: float = 0.0,
        entropy_loss_annealing_steps: int = 0,
        entropy_loss_annealing_factor: float = 1.0,
        commitment_weight: float = 0.0,
        diversity_gamma: float = 1.0,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout
        
        self.decay = decay

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        # self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()
        requires_out_projection = effective_codebook_dim != output_dim
        self.project_out = nn.Linear(effective_codebook_dim, output_dim) if requires_out_projection else nn.Identity()

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)
        
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, self.codebook_size))
        self.register_buffer('cluster_size_wo_react', torch.zeros(num_codebooks, self.codebook_size))
        

        self.register_buffer(
            'timestep_p_over_c',
            torch.ones(num_codebooks, token_len, self.codebook_size) / self.codebook_size
        )
        self.register_buffer('tpc_initted', torch.Tensor([False]))

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842
    
    def symmetry_preserving_bound(self, z):
        """
        QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1
        """
        levels_minus_1 = (self._levels - 1)
        scale = 2.0 / levels_minus_1
        bracket = (levels_minus_1 * (torch.tanh(z) + 1) / 2.0) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.0

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """

        preserve_symmetry = self.preserve_symmetry
        half_width = self._levels // 2

        if preserve_symmetry:
            quantized = round_ste(self.symmetry_preserving_bound(z)) / half_width
        else:
            quantized = round_ste(self.bound(z)) / half_width

        if not self.training:
            return quantized

        batch, device, noise_dropout = z.shape[0], z.device, self.noise_dropout
        unquantized = z

        # determine where to quantize elementwise

        quantize_mask = torch.bernoulli(
            torch.full((batch,), noise_dropout, device = device)
        ).bool()

        quantized = einx.where('b, b ..., b ...', quantize_mask, unquantized, quantized)

        # determine where to add a random offset elementwise

        offset_mask = torch.bernoulli(
            torch.full((batch,), noise_dropout, device = device)
        ).bool()

        offset = (torch.rand_like(z) - 0.5) / half_width
        quantized = einx.where('b, b ..., b ...', offset_mask, unquantized + offset, quantized)

        return quantized

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def get_group_perplexity(self, codebook_idx=0):
        ap = self.timestep_p_over_c[codebook_idx]
        group_perplexity = torch.exp(-torch.sum(ap * torch.log(ap + 1e-10), dim=-1))
        return group_perplexity

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes
    
    def get_output_from_indices(self, indices):
        codes = self.indices_to_codes(indices)
        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        # -------- unify to bfloat16 --------
        bf16 = torch.bfloat16
        z = z.to(bf16)

        # 确保需要做 EMA 的 buffer 也用 bfloat16（防止 dtype mismatch）
        if hasattr(self, "timestep_p_over_c") and self.timestep_p_over_c.dtype != bf16:
            self.timestep_p_over_c.data = self.timestep_p_over_c.data.to(bf16)
        if hasattr(self, "cluster_size") and self.cluster_size.dtype != bf16:
            self.cluster_size.data = self.cluster_size.data.to(bf16)
        # -----------------------------------

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z).to(bf16)  # 确保线性层输出也为 bf16

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # whether to force quantization step to be full precision or not
        # 统一到 bf16，关闭 f32 强制与 autocast
        force_f32 = False
        quantization_context = nullcontext

        with quantization_context():
            orig_dtype = bf16

            # 不再上转换到 float32
            codes = self.quantize(z)               # 期望量化算子支持 bf16
            indices = None
            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = rearrange(codes, 'b n c d -> b n (c d)')
            codes = codes.to(orig_dtype)

        # 下面保持原有统计逻辑，但全部在 bf16 上进行
        embed_onehot = F.one_hot(indices.squeeze(-1).long(), num_classes=self.codebook_size).unsqueeze(0).type(z.dtype)
        batch_t_p_over_c = embed_onehot.mean(dim=1)
        distributed.all_reduce(batch_t_p_over_c)
        batch_t_p_over_c /= distributed.get_world_size()
        decay = self.decay if self.tpc_initted else 0.3

        # 若 buffer 仍非 bf16，这里再保险一次转换
        ema_inplace(self.timestep_p_over_c.data, batch_t_p_over_c.to(self.timestep_p_over_c.dtype), decay)
        if not self.tpc_initted:
            self.tpc_initted.data.copy_(torch.Tensor([True]))
        perplexity = self.get_group_perplexity().mean().item()

        unpacked_onehot = F.one_hot(indices.flatten().long(), num_classes=self.codebook_size).type(orig_dtype)
        unpacked_onehot = unpacked_onehot.sum(dim=0)
        distributed.all_reduce(unpacked_onehot)
        ema_inplace(self.cluster_size.data, unpacked_onehot.to(self.cluster_size.dtype), decay)
        n_active = len(torch.nonzero(self.cluster_size.data[0]))

        # project out
        out = self.project_out(codes).to(bf16)

        # reconstitute image or video dimensions
        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')
            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices
        log_dict = {
            "perplexity": perplexity,
            "n_active": n_active,
        }
        loss = 0

        return out, indices, loss, log_dict
