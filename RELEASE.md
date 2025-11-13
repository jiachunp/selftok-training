# SelfTok Release Notes
## Encode Ablation

### 0813 Release Note
1. Encoder support: laywerwise and qformer encoder (bi, uni, concat mode).
2. Teacher support: cross-attn-based DiT or MMDiT.
3. Support of teacher initialization with pre-trained DiT or SD3.
4. Support of training on imagenet or openimages+imagenet (both using latent mode).
5. Running scripts/setup_gpu.py on debug node now setup local imagenet training with sd1.5 + sd3.0 latent.
6. Validation on both LPIPS and PSNR.
7. Enabled force_recon mode when teacher is SD3.
8. Support of stage 2 fine-tuning (frozen encoder, train noise predictor with full tokens).


### 0814 Release Note
1. Fix the force-recon mode when using SD3 as teacher.
2. Fix a bug when using pre-encode with SD3 as teacher.


### 1008 Note
1. Change dead code threshold to a relative value (when given #codebook_size samples, the threshold expected hit of a dead code)
2. Change code to accomodate vq continuous trick
3. Change to continuous timestep
4. When val_interval less or equal 0, do not do validation
5. In image_tokenizer, set max positional encoding to 2*latent size (e.g., in 256 res, max side length is 512, corresponding to a 1:4 aspect ratio)
6. Add continuous to quantizer config
7. RandomResize can specify random ratio now. Ratio=1.0 means all random resize. Ratio=0.0 means all resize. Need to specify ratio in dataloader config. See v2/1024-hq-01.yml

### 10XX Note
1. Improve codebook entropy with ema implementation
2. Add quantizer config options: reset_cluster_size(relative, same as dead code thres), ema_entropy_ratio
3. VQ reactivate now check for sizes and gives warning if not match
4. Deleted EuclideanCodebook
5. Add shift=e^0.5 to 256 res training
6. Change mean loss to sum loss / batch_size (dm loss and commit loss)
7. Fix RMSNorm (used to not having initialization), move it to module.py. Can use qk norm to enable it in yml (see v2/1024-03-buypt1-hqpt2-try2.yml)
8. Perplexity list now does not have a separate EMA (instead, directly use timestep_p_over_c)
9. Add apply_losses_together option in encoder config. If True, commit loss+entropy loss only if recon loss is applied to a token.
10. Add logging of cosine similarity (cosine sim) between encoder out and quantized.
11. Entropy loss now divide into mean loss for each k, and a group loss (average prob of neighboring k are grouped and mean before calculating entropy)
12. Add option to enable token layernorm in between transformer layers