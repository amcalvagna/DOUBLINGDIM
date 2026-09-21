import gc
import os
import math
import torch
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
from globals import *
from utils import coreset


os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

@dataclass
class DoublingDimResult:
    estimate: float
    per_scale_max: torch.Tensor
    radii: torch.Tensor
    sampled_centers: torch.Tensor
    local_counts: torch.Tensor
    config: Dict[str, Any]


def pick_device(prefer: Optional[str] = None) -> str:
    if prefer is not None:
        return prefer
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _to_device(x: torch.Tensor, device: str, dtype=torch.float32) -> torch.Tensor:
    return x.detach().to(device=device, dtype=dtype)


def _linf_cdist_blocked(a: torch.Tensor, b: torch.Tensor, qblock: int, rblock: int) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)
    for i in range(0, a.shape[0], qblock):
        ai = a[i:i+qblock]
        for j in range(0, b.shape[0], rblock):
            bj = b[j:j+rblock]
            out[i:i+qblock, j:j+rblock] = (ai[:, None, :] - bj[None, :, :]).abs().amax(dim=-1)
    return out


def _linf_dist_one_to_many(x: torch.Tensor, Y: torch.Tensor, rblock: int = 2048) -> torch.Tensor:
    out = torch.empty((Y.shape[0],), device=Y.device, dtype=torch.float32)
    for j in range(0, Y.shape[0], rblock):
        yj = Y[j:j+rblock]
        out[j:j+rblock] = (yj - x).abs().amax(dim=-1) #distance in Linf metrics 
    return out

# rename Linf nextDim_sample_streaming()
def _farthest_point_sample_streaming(X: torch.Tensor, m: int, seed: int = 0, rblock: int = 2048) -> torch.Tensor:
    n = X.shape[0]
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    #sort X ascending in Linf 
    first = int(torch.randint(0, n, (1,), generator=g).item()) # should take the min point, not a random
    chosen = [first]
    min_dist = _linf_dist_one_to_many(X[first], X, rblock=rblock)  #min_dist are Linf distances from "first" point 
    for _ in range(1, min(m, n)):
        nxt = int(torch.argmax(min_dist).item())
        chosen.append(nxt)
        d = _linf_dist_one_to_many(X[nxt], X, rblock=rblock)
        min_dist = torch.minimum(min_dist, d)
    return torch.tensor(chosen, device=X.device, dtype=torch.long)


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


def estimate_doubling_dimension_linf_mps(
    X: torch.Tensor,
    center_samples: int = 128,
    anchor_samples: int = 256,
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
    assert X.ndim == 2
    device = pick_device(device)
    X = _to_device(X, device=device, dtype=torch.float32)
    n = X.shape[0]

    if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

    center_idx = _farthest_point_sample_streaming(X, min(center_samples, n), seed=seed, rblock=fps_block)
    centers = X[center_idx]

    g = torch.Generator(device='cpu')
    g.manual_seed(seed + 1)
    anchor_idx = torch.randperm(n, generator=g)[:min(anchor_samples, n)].to(device)
    anchors = X[anchor_idx]

    center_anchor_d = _linf_cdist_blocked(centers, anchors, qblock=cdist_qblock, rblock=cdist_rblock)
    if radii is None:
        q = torch.tensor(radius_quantiles, device=device, dtype=torch.float32)
        radii_t = torch.quantile(center_anchor_d.flatten(), q).clamp_min(1e-12)
        radii_t = torch.unique(radii_t)
    else:
        radii_t = torch.tensor(list(radii), device=device, dtype=torch.float32)

    local_counts = torch.zeros((centers.shape[0], radii_t.shape[0]), device=device, dtype=torch.float32)

    for ci in tqdm(range(centers.shape[0]), desc="Centers", leave=True):
        c = centers[ci]
        d_all = _linf_dist_one_to_many(c, X, rblock=cdist_rblock)
        for ri, r in enumerate(radii_t):
            idx = torch.nonzero(d_all <= r, as_tuple=False).squeeze(1)
            m = int(idx.numel())
            if m <= 1:
                local_counts[ci, ri] = 1.0
                continue
            if m > neighborhood_cap:
                perm = torch.randperm(m, device=device)[:neighborhood_cap]
                idx = idx[perm]
            Y = X[idx]
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
        sampled_centers=center_idx.detach().cpu(),
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


def computeDoublingDim(dataset, coresets_file, normP=math.inf, profile_dd=None):
    train_set, _, _ = dataset
    if profile_dd is None:
        profile_dd = os.getenv("DD_PROFILE", "0").lower() in {"1", "true", "yes", "on"}
    profiling = lambda: "ON" if profile_dd else "OFF"
    logging.info(f"RUNNING WITH PROFILING {profiling}")
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f"Doubling dimension computations running on {device.type.upper()}.")
    #-------------------#          
    min_centers = 128
    max_centers = 4096
    nb_cap = 2024
    max_radii = 192
    #-------------------#
    #------------------------------------------------------------------------------------------
    # Always load metadata on CPU and move tensors lazily per coreset.
    coresets = torch.load(coresets_file, map_location="cpu")  # METTERE DIRETTAMENTE SU MPS???
    #------------------------------------------------------------------------------------------
    dds = {}

    for m, s in ((m, s) for m in metrics for s in samplers):
        setting = (m, s)
        dds[setting] = {}

        for r in ratios:
            coreset_idx = coresets[r][m][s]
            coreset_points = coreset(train_set, coreset_idx).tensors[0]
            #------------------------------------------------
            #coreset_points = coreset_points.to(device, dtype=torch.float32, non_blocking=True) #### O VERO????? ???????~~~~########
            #__________________________________________________
            num = len(coreset_idx)
            num_centers = int(min(min_centers, num)+r*max_centers)
            logging.info(f"Computing DD for model: {nnet.fc_name} on {ds.name} coreset {setting} with compression:{r:.00%}")
            logging.info(f"Using {num_centers} random points as centers, {max_radii} spaced radiuses per center, early stop at max {nb_cap}")
            #logging.info(f"Using max {max_centers, num} random points, {max_radii} spaced radiuses per center, early stop at max {num}")

            start = datetime.datetime.now()
            
            res = estimate_doubling_dimension_linf_mps(
                    coreset_points,  # your [50000, 512] tensor
                    center_samples=num_centers,        # 96 centers → ~10s runtime on M1 Ultra 32-core
                    anchor_samples=max_radii,       # for radii selection
                    neighborhood_cap=nb_cap,     # cap per neighborhood to avoid long covers
                    fps_block=4096,           # FPS distance chunks
                    cdist_qblock=64,          # small query blocks (centers to points)
                    cdist_rblock=2048,        # larger result blocks (safe for MPS)
                    cover_block=1024,         # greedy cover chunks
                    device="mps",
                    sync_each_center=False,   # disable unless you see memory creep
                )
            elapsed = datetime.datetime.now() - start 
            print({'estimate': res.estimate, 'radii': res.radii.tolist(), 'per_scale_max': res.per_scale_max.tolist(), 'device': res.config['device'], 'sec': elapsed})

            dds[setting][r] = res.estimate
            logging.info(f"Processed points amount: {num}")
            logging.info(f"Approx doubling constant λ: {res.per_scale_max}")
            logging.info(f"Approx doubling dimension  : {res.estimate}")
            logging.info(f"Elapsed time: {elapsed}.")

            # Free GPU/MPS memory before moving to the next coreset.
            del coreset_points  #, pairwise_dist            
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif device.type == "cpu":
                gc.collect()
    return dds
