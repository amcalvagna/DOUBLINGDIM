import math
from tkinter import SE
from typing import Optional, Sequence
from matplotlib.dates import _ordinalf_to_timedelta_np_vectorized
import numpy as np
import torch
from tqdm import tqdm
from data import DoublingDimResult, ddConfig

# API: provide configuration in a ddConfig object named config and 
# an estimator function with Estimator signature as below
# they will be imported with this file and invoked inside utils.py by standard computedoublingdim routine 

config = ddConfig(
    name = "ddLinfMixed",
    min_centers = 128,
    max_centers  = 256,
    nb_cap  = 256,
    max_radii  = 32,
    #-------------------#
    fps_block  = 4096,
    cdist_qblock = 64,       # small query blocks (centers to points)
    cdist_rblock  = 2048,      # larger result blocks (safe for MPS)
    cover_block  = 1024,
    radii  = None,
    anchor_samples = 128,
    radius_quantiles  = (0.50, 0.70, 0.85, 0.93)
)


def recursive_cover_by_maxspread_split(X, idx, r):
    if idx.numel() <= 1: return 1

    S=X[idx]
    spreads = S.max(dim=0).values - S.min(dim=0).values
    if spreads.max().item() <= r:  return 1

    j = spreads.argmax() #.item() #coordinate_with_largest_spread(S)
    #S_sorted = sort_points_by_coordinate(S, j) # no need to sort points


    #split S_sorted along_coordinate j  
    vals = S[:, j]
    t = 0.5 * (vals.min() + vals.max()) # midpoint
    # t = np.median(vals) 
    left_mask = vals <= t
    right_mask = vals > t

    if left_mask.sum() == 0 or right_mask.sum() == 0: return 1

    return (
        recursive_cover_by_maxspread_split(X, idx[left_mask], r)
        + recursive_cover_by_maxspread_split(X, idx[right_mask], r)
    )

def approximate_dd_linf(X, centers_idx, radii):
    lambda_hat = 1
    radii_list = [float(r) for r in radii]

    for c_idx in tqdm(centers_idx, desc="Centers", leave=True):
        c = X[c_idx]
        d = torch.amax(torch.abs(X - c), dim=1) #Linf dist
        order = torch.argsort(d)
        d_sorted = d[order]

        for r in radii_list:
            k = int(torch.searchsorted(d_sorted, r, side="right"))
            if k == 0 or k <= lambda_hat: continue

            #S = X[order[:k]]   # admissible set, diam <= 2r
            cover = recursive_cover_by_maxspread_split(X, order[:k], r)
            lambda_hat = max(lambda_hat, cover)

    return math.log2(lambda_hat) #, lambda_haty


# call adapter to match the API expected in util.computeDuoblingDim
def ddEstimator( #_doubling_dimension_linf_mps(
    coreset: torch.Tensor,
    center_samples=96,
    anchor_samples: int = 256,
    centers_idx: Optional[torch.Tensor] = None,
    radii: Optional[torch.Tensor] = None,
    max_radii = None,
    radius_quantiles: tuple[float,...] = (0.50, 0.70, 0.85, 0.93),
    neighborhood_cap: int = 1024,
    fps_block: int = 4096,
    cdist_qblock: int = 64,
    cdist_rblock: int = 2048,
    cover_block: int = 1024,
    seed: int = 0,
    device: Optional[str] = None,
    sync_each_center: bool = False,
) -> DoublingDimResult:
  
    dd = approximate_dd_linf(
        X = coreset,
        centers_idx = centers_idx,
        radii = radii,
    )

    return DoublingDimResult(
        estimate=dd,
        #per_scale_max=per_scale_max.detach().cpu(),
        #radii= radii.detach().cpu(), #if radii is not None else None,
        #sampled_centers=centers_idx.detach().cpu(), 
        #local_counts=local_counts.detach().cpu(),
        config={
            "device": device,
            "center_samples": int(center_samples),
            "anchor_samples": int(anchor_samples),
            "radius_quantiles": [],
            "neighborhood_cap": int(neighborhood_cap),
            "fps_block": int(fps_block),
            "cdist_qblock": int(cdist_qblock),
            "cdist_rblock": int(cdist_rblock),
            "cover_block": int(cover_block),
            "seed": int(seed),
            "sync_each_center": bool(sync_each_center),
        }
    )