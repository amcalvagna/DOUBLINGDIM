
import os
import torch
from tqdm import tqdm
from typing import Optional, Sequence
from utils import pick_device, _farthest_point_sample_streaming, _linf_cdist_blocked, _linf_dist_one_to_many 
from data import DoublingDimResult, ddConfig
from globals import *

config = ddConfig() #default config values are set just for this algorithm
# which are also below as a backup
# config = ddConfig(
#     name = "doubling_dim_linf_mps",
#     min_centers = 128,
#     max_centers  = 4096,
#     nb_cap  = 2024,
#     max_radii  = 192,
#     #-------------------#
#     fps_block  = 4096,
#     cdist_qblock = 64,       # small query blocks (centers to points)
#     cdist_rblock  = 2048,      # larger result blocks (safe for MPS)
#     cover_block  = 1024,
#     radii  = None,
#     anchor_samples = 256,
#     radius_quantiles  = (0.50, 0.70, 0.85, 0.93)
# )


os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

def _greedy_cover_count_linf(Y: torch.Tensor, radius: float, rblock: int = 1024) -> int:
    m = Y.shape[0]
    uncovered = torch.ones(m, dtype=torch.bool, device=Y.device)
    count = 0
    half_r = float(radius) * 0.5
    while bool(uncovered.any()):
        piv = int(torch.nonzero(uncovered, as_tuple=False)[0].item())
        active = torch.nonzero(uncovered, as_tuple=False).squeeze(1)
        yp = Y[piv]
        d = _linf_dist_one_to_many(yp, Y[active], rblock=rblock)
        covered = active[d <= half_r]
        uncovered[covered] = False
        count += 1
    return count


def ddEstimator( #_doubling_dimension_linf_mps(
    coreset: torch.Tensor,
    center_samples=96,
    anchor_samples: int = 256,
    centers_idx: Optional[torch.Tensor] = None,
    radii: Optional[Sequence[float]] = None,
    radius_quantiles: Sequence[float] = (0.50, 0.70, 0.85, 0.93),
    neighborhood_cap: int = 1024,
    fps_block: int = 4096,
    cdist_qblock: int = 64,
    cdist_rblock: int = 2048,
    cover_block: int = 1024,
    seed: int = 0,
    device: Optional[str] = None,
    sync_each_center: bool = False,
) -> DoublingDimResult:
    assert coreset.ndim == 2
    device = pick_device(device)
    coreset= coreset.detach().to(device=device, dtype=torch.float32)
    n = coreset.shape[0]

    if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    
    if centers_idx is None: 
        centers_idx = _farthest_point_sample_streaming(coreset, min(center_samples, n), seed=seed, rblock=fps_block)
    
    
    centers = coreset[centers_idx]

    g = torch.Generator(device='cpu')
    g.manual_seed(seed + 1)
    
    if radii is None:
        anchor_idx = torch.randperm(n, generator=g)[:min(anchor_samples, n)].to(device)
        anchors = coreset[anchor_idx]
        center_anchor_d = _linf_cdist_blocked(coreset, anchors, qblock=cdist_qblock, rblock=cdist_rblock)
        q = torch.tensor(radius_quantiles, device=device, dtype=torch.float32)
        radii_t = torch.quantile(center_anchor_d.flatten(), q).clamp_min(1e-12)
        radii_t = torch.unique(radii_t)
    else:
        radii_t = torch.tensor(list(radii), device=device, dtype=torch.float32)

    local_counts = torch.zeros((centers.shape[0], radii_t.shape[0]), device=device, dtype=torch.float32)

    for ci in tqdm(range(centers.shape[0]), desc="Centers", leave=True):
        c = centers[ci]
        d_all = _linf_dist_one_to_many(c, coreset, rblock=cdist_rblock)
        for ri, r in enumerate(radii_t):
            idx = torch.nonzero(d_all <= r, as_tuple=False).squeeze(1)
            m = int(idx.numel())
            if m <= 1:
                local_counts[ci, ri] = 1.0
                continue
            if m > neighborhood_cap:
                perm = torch.randperm(m, device=device)[:neighborhood_cap]
                idx = idx[perm]
            Y = coreset[idx]
            local_counts[ci, ri] = float(_greedy_cover_count_linf(Y, float(r.item()), rblock=cover_block))
        if sync_each_center and device == 'mps':
            torch.mps.synchronize()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

    per_scale_max = local_counts.max(dim=0).values
    estimate = torch.log2(per_scale_max.clamp_min(1.0)).max().item()

    return DoublingDimResult(
        estimate=estimate,
        per_scale_max=per_scale_max.detach().cpu(),
        radii=radii_t.detach().cpu(),
        sampled_centers=centers_idx.detach().cpu(), 
        local_counts=local_counts.detach().cpu(),
        config={
            'device': device,
            'center_samples': int(center_samples),
            'anchor_samples': int(anchor_samples),
            'radius_quantiles': list(radius_quantiles),
            'neighborhood_cap': int(neighborhood_cap),
            'fps_block': int(fps_block),
            'cdist_qblock': int(cdist_qblock),
            'cdist_rblock': int(cdist_rblock),
            'cover_block': int(cover_block),
            'seed': int(seed),
            'sync_each_center': bool(sync_each_center),
        }
    )


