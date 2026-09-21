import os
import math
import logging
import torch
from tqdm.auto import tqdm
from torch.profiler import record_function
from globals import *
from utils import coreset

def _pairwise_dist_chunked(A, B, p=2.0):
    if p == 2 or p == 2.0:
        a2 = (A * A).sum(dim=1, keepdim=True)
        b2 = (B * B).sum(dim=1).unsqueeze(0)
        D2 = (a2 + b2 - 2.0 * (A @ B.T)).clamp_min_(0.0)
        return D2.sqrt_()
    return torch.cdist(A, B, p=p)


def _get_center_distances_streaming(X, center_idx, p, x_chunk_size):
    device = X.device
    N = X.size(0)
    center = X[center_idx:center_idx + 1]
    out = torch.empty(N, device=device, dtype=X.dtype)
    for s in range(0, N, x_chunk_size):
        e = min(s + x_chunk_size, N)
        out[s:e] = _pairwise_dist_chunked(center, X[s:e], p=p).squeeze(0)
    return out


def _select_radii_from_distances(d_c, radii_per_center):
    positive = d_c[d_c > 0]
    if positive.numel() == 0:
        return None
    sorted_dist, _ = torch.sort(positive)
    if sorted_dist.numel() <= radii_per_center:
        return sorted_dist
    sample_idx = torch.linspace(
        0, sorted_dist.numel() - 1, radii_per_center, device=sorted_dist.device
    ).round().long()
    return sorted_dist[sample_idx]


def _indices_within_radius_streaming(d_c, radius, membership_chunk_size):
    N = d_c.numel()
    parts = []
    for s in range(0, N, membership_chunk_size):
        e = min(s + membership_chunk_size, N)
        local = torch.nonzero(d_c[s:e] <= radius, as_tuple=False).squeeze(1)
        if local.numel() > 0:
            parts.append(local + s)
    if parts:
        return torch.cat(parts, dim=0)
    return torch.empty(0, dtype=torch.long, device=d_c.device)
import torch


import numpy as np
import torch


def _compute_ball_coverage_bitsets_streaming(
    X_ball,
    half_radius,
    p,
    pair_block_size,
):
    device = X_ball.device
    m = X_ball.size(0)
    words = (m + 63) // 64

    row_masks = np.zeros((m, words), dtype=np.uint64)
    one_u64 = np.uint64(1)

    for i0 in range(0, m, pair_block_size):
        i1 = min(i0 + pair_block_size, m)
        Xi = X_ball[i0:i1]

        for j0 in range(0, m, pair_block_size):
            j1 = min(j0 + pair_block_size, m)
            Xj = X_ball[j0:j1]

            Dblk = _pairwise_dist_chunked(Xi, Xj, p=p)
            Adj = (Dblk <= (half_radius + 1e-12)).detach().cpu().numpy()

            hit_rows, hit_cols = np.nonzero(Adj)
            if hit_rows.size > 0:
                global_rows = hit_rows + i0
                global_cols = hit_cols + j0

                word_ids = global_cols // 64
                bit_ids = global_cols % 64

                keys = global_rows * words + word_ids
                order = np.argsort(keys, kind="stable")
                keys = keys[order]
                bit_ids = bit_ids[order]

                unique_keys, start_idx, counts = np.unique(
                    keys, return_index=True, return_counts=True
                )

                for key, start, cnt in zip(unique_keys, start_idx, counts):
                    bits = bit_ids[start:start + cnt].astype(np.uint64, copy=False)
                    word_val = np.bitwise_or.reduce(
                        np.left_shift(one_u64, bits),
                        dtype=np.uint64
                    )

                    row = key // words
                    word = key % words
                    row_masks[row, word] |= word_val

            del Dblk, Adj

    return row_masks, m


def _bitset_popcount_int(x: int) -> int:
    return x.bit_count()


def _greedy_cover_count_bitset_cpu(row_masks, m, lambda_max=1):
    rows = []
    mask64 = (1 << 64) - 1

    for i in range(m):
        acc = 0
        for w in row_masks[i]:
            acc = (acc << 64) | (int(w) & mask64)
        rows.append(acc)

    uncovered = (1 << m) - 1
    cover_count = 0

    while uncovered:
        best_i = -1
        best_gain = 0

        for i, mask in enumerate(rows):
            gain = _bitset_popcount_int(mask & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_i = i

        if best_gain == 0:
            break

        uncovered &= ~rows[best_i]
        cover_count += 1

        if cover_count > lambda_max:
            return cover_count

    return cover_count

def new_approximate_dd_streaming(
    X,
    D=None,
    p=2.0,
    num_centers=128,
    radii_per_center=8,
    max_ball_size=math.inf,
    batch_size=None,
    device=None,
    seed=None,
    x_chunk_size=16384,
    membership_chunk_size=65536,
    pair_block_size=1024,
):
    """
    Streaming, low-memory rewrite of new_approximate_dd_gpu_optimized.

    Preserves the same high-level output:
      - same center sampling logic
      - same radius sampling logic
      - same greedy cover count semantics

    But avoids:
      - full N x N distance matrices
      - full |B| x |B| distance matrices
      - full |B| x |B| adjacency matrices
    """
    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(device)

    X = X.to(device)
    N = X.size(0)

    if seed is not None:
        torch.manual_seed(seed)

    if num_centers < N:
        center_indices = torch.randperm(N, device=X.device)[:num_centers]
    else:
        center_indices = torch.arange(N, device=X.device)

    lambda_max = 1

    for center_id in tqdm(center_indices.tolist(), total=center_indices.numel(), desc="DD centers streaming"):
        with record_function("dd_center_distances_streaming"):
            if D is None:
                d_c = _get_center_distances_streaming(X, center_id, p, x_chunk_size)
            else:
                d_c = D[center_id].to(device)

        with record_function("dd_select_radii_streaming"):
            radii = _select_radii_from_distances(d_c, radii_per_center)
            if radii is None or radii.numel() == 0:
                continue

        for ridx in range(radii.numel()):
            R = radii[ridx]

            with record_function("dd_ball_membership_streaming"):
                B_idx = _indices_within_radius_streaming(d_c, R, membership_chunk_size)

                if B_idx.numel() <= lambda_max:
                    continue

                if B_idx.numel() > max_ball_size:
                    perm = torch.randperm(B_idx.numel(), device=B_idx.device)[: int(max_ball_size)]
                    B_idx = B_idx[perm]

            with record_function("dd_ball_cover_graph_streaming"):
                ball_points = X[B_idx]
                half = R / 2.0

                row_masks, m = _compute_ball_coverage_bitsets_streaming(
                    ball_points,
                    half_radius=half,
                    p=p,
                    pair_block_size=pair_block_size,
                )

            with record_function("dd_cover_bitset_streaming"):
                cover_count = _greedy_cover_count_bitset_cpu(
                    row_masks=row_masks,
                    m=m,
                    lambda_max=lambda_max,
                )

            if cover_count > lambda_max:
                lambda_max = cover_count

    ddim = math.log2(lambda_max)
    return ddim, lambda_max




def computeDoublingDim(dataset, coresets_file, normP=math.inf, profile_dd=None):
    from time import time
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

    #------------------------------------------------------------------------------------------
    # Always load metadata on CPU and move tensors lazily per coreset.
    coresets = torch.load(coresets_file, map_location="cpu")  # METTERE DIRETTAMENTE SU MPS???
    #------------------------------------------------------------------------------------------
    dds = {}

    for m, s in ((m, s) for m in metrics for s in samplers):
        setting = (m, s)
        dds[setting] = {}

        for r in ratios:
            #-------------------#
            max_centers = 128
            max_radii = 8
            #-------------------#
            coreset_idx = coresets[r][m][s]
            coreset_points = coreset(train_set, coreset_idx).tensors[0]
            #------------------------------------------------
            coreset_points = coreset_points.to(device, dtype=torch.float32, non_blocking=True) #### O VERO????? ???????~~~~########
            #__________________________________________________
            num = len(coreset_idx)
            max_centers = min(max_centers, num) if max_centers else num
            logging.info(f"Computing DD for model: {nnet.fc_name} on {ds.name} coreset {setting} with compression:{r:.00%}")
            logging.info(f"Using max {num} random points as centers, {max_radii} spaced radiuses per center, early stop at max {num}")
            #logging.info(f"Using max {max_centers, num} random points, {max_radii} spaced radiuses per center, early stop at max {num}")

            dist = None
            if num < 4096:   # ALWAYS precompute distances only for small coresets sizes
                with torch.no_grad():
                    #scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.
                    dist = torch.cdist(coreset_points, coreset_points, normP) #.detach().to("cpu") #, compute_mode='use_mm_for_euclid_dist')
                    logging.info(f"PRECOMPUTED ALL DISTANCES. max dist: {dist.max().item()} min dist: {dist[dist > 0].min().item()}")

            start = datetime.datetime.now()
            dd_kwargs = dict(
                X=coreset_points,
                D=dist,
                p=normP,
                num_centers=num, #min(max_centers, num),
                radii_per_center=max_radii,
                max_ball_size=num,
                batch_size=1, #min(32, max(1, num // 8)),
                device=device,
                seed=int(time())
            )
            
            ddim, lam = new_approximate_dd_streaming(**dd_kwargs)
            elapsed = datetime.datetime.now() - start 

            dds[setting][r] = ddim
            logging.info(f"Processed points amount: {num}")
            logging.info(f"Approx doubling constant λ: {lam}")
            logging.info(f"Approx doubling dimension  : {ddim}")
            logging.info(f"Elapsed time: {elapsed}.")

            # Free GPU/MPS memory before moving to the next coreset.
            del coreset_points  #, pairwise_dist            
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
    return dds
