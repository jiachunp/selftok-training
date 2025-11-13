from mimogpt.models.selftok.vector_quantize_pytorch import VectorQuantize as VectorQuantize_EMA
from mimogpt.models.selftok.finite_scalar_quantization import FSQ

def construct_fsq(
    levels, dim, output_dim, smart_re_K=0, decay=0.99,
    ):
    args = dict(
        levels=levels,
        dim=dim,
        output_dim=output_dim,
        smart_re_K=smart_re_K,
        decay=decay,
        entropy_loss_weight=0.1,
        entropy_loss_annealing_steps=2000,
        entropy_loss_annealing_factor=3,
        commitment_weight=0.25,
    )
    
    return FSQ(**args)

def construct_quantizer(
        latent_dim, code_dim, output_dim, codebook_size, K,
        w_diversity, w_commit, dead_code_threshold=0.0, decay=0.99,
        smart_re_K=0, continuous=False, reg=[1/4., 1/2.],
        reset_cluster_size=None, ema_entropy_ratio=0.7, frozen_embed=None, use_fsq=False, levels=None,):
    
    args = dict(
        dim=latent_dim,
        output_dim=output_dim,
        codebook_dim=code_dim,
        codebook_size=codebook_size,
        ema_update=True,
        decay=decay,
        kmeans_init=True,
        kmeans_iters=10,
        threshold_ema_dead_code=dead_code_threshold,
        use_cosine_sim=True,
        commitment_weight=w_commit,
        diversity_weight=w_diversity,
        smart_re_K=smart_re_K,
        continuous=continuous,
        reg=reg,
        reset_cluster_size=reset_cluster_size,
        ema_entropy_ratio=ema_entropy_ratio,
        frozen_embed = frozen_embed,
    )

    constructor = VectorQuantize_EMA
    
    return constructor(**args)

