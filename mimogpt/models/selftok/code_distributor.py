import torch
import torch.distributed as distributed
from sklearn.cluster import KMeans
import torch.nn.functional as F

# -0.007
class CodeDistributor:
    def __init__(self, codebook_size, num_redistributed, period=50, collect_history=2, cutoff_redundancy=-0.007):
        self.C = codebook_size
        self.n = num_redistributed
        self.T = period
        self.L = collect_history
        self.cutoff_redundancy = cutoff_redundancy
        self.d = None       # lazy init during update
        self.update_history = torch.tensor([False] * codebook_size)
        self.reset()

    def reset(self):
        self.redundancy = torch.zeros(self.C)
        self.hits = torch.zeros(self.C)
        self.worst_samples = []
        self.worst_samples_weight = []
        self.step = 0

    def update(self, x, quantized, dist):
        # Note that dist is actually cosine similarity when using CosineCodebook
        self.step += 1
        self.d = x.shape[-1]
        N = dist.shape[1]
        dist_cp = dist[0].clone()
        indices = dist_cp.argmax(dim=1)
        dist_cp[torch.arange(N), indices] = -1
        indices_2nd = dist_cp.argmax(dim=1)
        self.hits = self.hits * 0.9 + 0.1 * indices.bincount(minlength=self.C).cpu()
        dist_diff = dist_cp[torch.arange(N),indices_2nd] - dist[0,torch.arange(N),indices]
        self.redundancy = self.redundancy * 0.9 + 0.1 * indices.bincount(minlength=self.C, weights=dist_diff).cpu()

        # update worst samples:
        if self.step > self.T - self.L:
            NUM = max(int(self.n * 50 / self.L / distributed.get_world_size()), 1)
            x = x[0]
            quantized = quantized.flatten(end_dim=-2)
            commit_losses = F.mse_loss(quantized, x, reduction='none').mean(dim=-1)
            # worst_sample_dists, worst_sample_indices = dist[0, torch.arange(N), indices].sort()
            worst_sample_dists, worst_sample_indices = commit_losses.sort(descending=True)
            worst_samples_batch = x[worst_sample_indices[:NUM]]
            self.worst_samples.append(worst_samples_batch.clone().cpu())
            self.worst_samples_weight.append(worst_sample_dists[:NUM].cpu())
            
        # redistribute
        bad_indices, new_codes, avg_score = None, None, 0
        if self.step % self.T == 0:
            bad_indices, new_codes, redundancies = self.redistribute_bad_codes(device=x.device)
            if bad_indices is not None:
                self.update_history[bad_indices.cpu()] = True
                avg_redun = redundancies.float().mean().item()
            else:
                avg_redun = 0
            if distributed.get_rank() == 0:
                print(f"Total {self.update_history.sum()} unique codes have been updated. Bad codes avg redundancy {avg_redun}...")
            self.reset()
        distributed.barrier()
        return bad_indices, new_codes, avg_score

    def redistribute_bad_codes(self, device):
        codes_score, redundancy = self.get_codes_score(device)  # higher redundancy has lower score
        bad_mask = redundancy > self.cutoff_redundancy   # actually a mask, but works same
        num_selected = int(min(self.n, bad_mask.sum().item()))
        if num_selected == 0:
            return None, None, 0
        bad_indices = codes_score.sort(descending=False)[1][:num_selected]    # this is indices
        worst_samples, weights = self.gather_worst_samples(device)
        if distributed.get_rank() == 0:
            worst_samples, weights = worst_samples.cpu().numpy(), weights.cpu().numpy()
            kmeans = KMeans(n_clusters=num_selected, n_init=10, random_state=0, max_iter=1000)
            results = kmeans.fit(worst_samples, sample_weight=weights)
            new_codes = torch.from_numpy(results.cluster_centers_).to(device)
            # indices = weights.sort(descending=True)[1][:self.n]
            # new_codes = worst_samples[indices]
        else:
            new_codes = torch.empty((num_selected, self.d)).to(device)
        distributed.broadcast(new_codes, src=0)
        return bad_indices, new_codes, redundancy[bad_indices]

    def get_codes_score(self, device):
        hits = self.hits.to(device)
        distributed.all_reduce(hits)
        mask = hits > 0
        redundancy = self.redundancy.to(device) * 1000 / distributed.get_world_size()   # modifier to make it larger
        distributed.all_reduce(redundancy)
        redundancy[mask] = redundancy[mask] / hits[mask]
        irreplaceable_score = redundancy.sort(descending=True)[1].sort()[1]
        total_score = irreplaceable_score
        return total_score, redundancy
    
    def gather_worst_samples(self, device):
        worst_samples = torch.cat(self.worst_samples, dim=0).to(device)
        weights = torch.cat(self.worst_samples_weight, dim=0).to(device)
        all_samples = [torch.empty_like(worst_samples) for _ in range(distributed.get_world_size())]
        all_weights = [torch.empty_like(weights) for _ in range(distributed.get_world_size())]
        distributed.all_gather(all_samples, worst_samples)
        distributed.all_gather(all_weights, weights)
        return torch.cat(all_samples, dim=0), torch.cat(all_weights, dim=0)