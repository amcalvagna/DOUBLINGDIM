import os
import numpy as np
import torch
import math
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
from globals import *
#from datetime import datetime
from utils import coreset


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
            if profile_dd:
                ddim, lam = profile_new_approximate_dd_gpu_optimized(
                    setting=setting,
                    ratio=r,
                    row_limit=25,
                    **dd_kwargs
                )
                profile_dd = False
            else:
                ddim, lam = new_approximate_dd_gpu_optimized(**dd_kwargs)
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



def profile_new_approximate_dd_gpu_optimized(
    *,
    setting,
    ratio,
    row_limit=25,
    export_dir=None,
    **dd_kwargs,
):
    """Profile the optimized DD path with the same interface as the original wrapper."""
    activities = [ProfilerActivity.CPU]
    device = torch.device(dd_kwargs.get("device", "cpu"))
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    export_dir = export_dir or os.getenv("DD_PROFILE_DIR")
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("doubling_dimension_run_optimized"):
            ddim, lam = new_approximate_dd_gpu_optimized(**dd_kwargs)

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    logging.info(
        "Optimized DD profiler summary for setting=%s ratio=%.4f\n%s",
        setting,
        ratio,
        prof.key_averages().table(sort_by=sort_key, row_limit=row_limit),
    )

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        metric_name, sampler_name = setting
        trace_path = os.path.join(
            export_dir,
            f"dd_profile_optimized_{metric_name}_{sampler_name}_{ratio:.4f}.json",
        )
        prof.export_chrome_trace(trace_path)
        logging.info(f"Exported optimized DD chrome trace to {trace_path}")
    return ddim, lam

# CURRENT DONT REMOVE
# def new_approximate_dd_gpu_optimized(X, D, p, num_centers, radii_per_center, max_ball_size, batch_size, device, seed):
# #     X,
# #     D=None,
# #     p=math.inf,
# #     num_centers=128,
# #     radii_per_center=12,
# #     max_ball_size=math.inf,
# #     batch_size=32,
# #     device=None,
# #     seed=None
# # ):
#     """Optimized variant: precompute D once, prune balls, and run greedy cover on CPU bitsets."""
#     if device is None:
#         device = (
#             "mps"
#             if torch.backends.mps.is_available()
#             else "cuda" if torch.cuda.is_available() else "cpu"
#         )
#     device = torch.device(device)

#     X = X.to(device)
#     N = X.size(0)

#     if seed is not None:
#         torch.manual_seed(seed)

#     if num_centers < N:
#         center_indices = torch.randperm(N)[:num_centers]
#     else:
#         center_indices = torch.arange(N)

#     lambda_max = 1

#     total_batches = math.ceil(center_indices.numel() / batch_size)
#     for center_batch in tqdm(
#         center_indices.split(batch_size),
#         total=total_batches,
#         desc="DD centers opt",
#     ):
#         with record_function("dd_center_batch_distances_optimized"):
#             if D is None:   
#                 logging.info("COMPUTING NEW SLICE OF D ")
#                 with torch.no_grad(): 
#                     dist_batch = torch.cdist(X[center_batch], X, p)
#             else: dist_batch = D[center_batch] 
            

#         for local_idx, _center_id in enumerate(center_batch):
#             with record_function("dd_center_scan_optimized"):
#                 d_c = dist_batch[local_idx]
#                 positive = d_c[d_c > 0]
#                 if positive.numel() == 0:
#                     continue

#                 sorted_dist, _ = torch.sort(positive)
#                 if sorted_dist.numel() <= radii_per_center:
#                     radii = sorted_dist
#                 else:
#                     sample_idx = torch.linspace(
#                         0, sorted_dist.numel() - 1, radii_per_center
#                     ).round().long()
#                     radii = sorted_dist[sample_idx]

#                 masks = d_c <= radii.view(-1, 1)

#             for ridx, mask in enumerate(masks):
#                 with record_function("dd_ball_build_optimized"):
#                     B_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
#                     if B_idx.numel() <= lambda_max:
#                         continue
#                     if B_idx.numel() > max_ball_size:
#                         perm = torch.randperm(B_idx.numel())[: int(max_ball_size)]
#                         B_idx = B_idx[perm]
                    
#                     ball_points = X[B_idx]
#                     D_ball = D[B_idx][:, B_idx] if D is not None else torch.cdist(ball_points, ball_points, p)
#                     R = radii[ridx]
#                     half = R / 2.0
#                     adjacency = D_ball <= half + 1e-12

#                 with record_function("dd_cover_bitset_optimized"):
#                     row_masks = _adjacency_to_bitset_rows(adjacency)
#                     cover_count = _greedy_cover_count_bitset(
#                         row_masks=row_masks,
#                         m=adjacency.size(0),
#                         lambda_max=lambda_max,
#                     )

#                 if cover_count > lambda_max:
#                     lambda_max = cover_count

#     ddim = math.log2(lambda_max)
#     return ddim, lambda_max

# def _adjacency_to_bitset_rows(adjacency):
#     """Pack a CPU bool adjacency matrix into Python-int bitsets, one per row."""
#     adjacency_np = adjacency.cpu().numpy().astype(np.uint8, copy=False)
#     packed = np.packbits(adjacency_np, axis=1, bitorder="little")
#     return [
#         int.from_bytes(row.tobytes(), byteorder="little", signed=False)
#         for row in packed
#     ]

# def _greedy_cover_count_bitset(row_masks, m, lambda_max):
#     """Greedy dominating-set cover on CPU using Python-int bitsets."""
#     full_mask = (1 << m) - 1
#     covered_mask = 0
#     cover_count = 0

#     while covered_mask != full_mask:
#         uncovered_mask = full_mask & ~covered_mask
#         best_mask = 0
#         best_gain = -1

#         for row_mask in row_masks:
#             gain = (row_mask & uncovered_mask).bit_count()
#             if gain > best_gain:
#                 best_gain = gain
#                 best_mask = row_mask

#         covered_mask |= best_mask
#         cover_count += 1

#         if cover_count > lambda_max:
#             break

#     return cover_count
 

#CLAUDE VERSION1
import math
import numpy as np
import torch
from tqdm import tqdm

EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Greedy cover — fully vectorised numpy, no Python loop over rows
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_cover_count_vectorized(adj: np.ndarray, early_stop: int) -> int:
    """
    Greedy dominating-set cover on a boolean (m, m) adjacency matrix.

    Replaces the O(m · cover_count) Python-loop-over-rows with a single
    numpy call per greedy step:

        gains = adj[:, uncovered].sum(axis=1)   # shape (m,) — one BLAS call

    For m = 256, cover_count = 8 this is ~32 numpy calls vs ~2048 Python
    iterations, each of which had a Python-int AND + popcount overhead.
    """
    m = adj.shape[0]
    covered = np.zeros(m, dtype=bool)
    count = 0

    while not covered.all():
        # Vectorised: count uncovered neighbours for every candidate row
        uncovered_cols = np.where(~covered)[0]
        gains = adj[:, uncovered_cols].sum(axis=1)   # (m,) – no Python loop
        best = int(np.argmax(gains))
        covered |= adj[best]
        count += 1
        if count > early_stop:
            break

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Radii sampling — returns a 1-D tensor sorted DESCENDING
# ─────────────────────────────────────────────────────────────────────────────

def _sample_radii_descending(d_c: torch.Tensor, k: int) -> torch.Tensor:
    """
    Sample k log-spaced radii from the positive distances in d_c,
    returned in DESCENDING order.

    Descending order is the critical invariant that lets us replace
    `continue` with `break` in the radii loop: ball_sizes are
    monotonically non-increasing, so the first radius whose ball is
    too small means all subsequent ones will be too.
    """
    positive = d_c[d_c > 0]
    if positive.numel() == 0:
        return positive

    sorted_dist, _ = torch.sort(positive, True)           # descending
    n = sorted_dist.numel()
    if n <= k:
        return sorted_dist                 # all

    idx = torch.linspace(0, n - 1, k).round().long()
    return sorted_dist[idx].flip(0)                  # sampled, descending


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def new_approximate_dd_gpu_optimized(
    X,
    D=None,
    p=math.inf,
    num_centers=128,
    radii_per_center=12,
    max_ball_size=math.inf,
    batch_size=32,
    device=None,
    seed=None,
):
    # ── device setup ──────────────────────────────────────────────────────────
    if device is None:
        device = (
            "mps"  if torch.backends.mps.is_available()  else
            "cuda" if torch.cuda.is_available()           else
            "cpu"
        )
    device = torch.device(device)
    X = X.to(device)
    N = X.size(0)

    if seed is not None:
        torch.manual_seed(seed)

    if N == 0:
        return 0.0, 0
    if N == 1:
        return 0.0, 1

    # ── center sampling ───────────────────────────────────────────────────────
    center_indices = (
        torch.randperm(N, device=device)[:num_centers]
        if num_centers < N
        else torch.arange(N, device=device)
    )

    lambda_max   = 1
    batch_size   = max(1, batch_size)
    total_batches = math.ceil(center_indices.numel() / batch_size)

    # Hoist: resolve finite max_ball once — avoids math.isfinite() per inner iter
    _max_ball = int(max_ball_size) if math.isfinite(max_ball_size) else None

    # ── main loop ─────────────────────────────────────────────────────────────
    for center_batch in tqdm(
        center_indices.split(batch_size),
        total=total_batches,
        desc="DD centers",
    ):
        # ── batch distance computation (GPU) ──────────────────────────────────
        if D is None:
            with torch.no_grad():
                dist_batch = torch.cdist(X[center_batch], X, p)  # (B, N)
        else:
            dist_batch = D[center_batch]                          # (B, N)

        # ── per-center scan ───────────────────────────────────────────────────
        for local_idx in range(center_batch.size(0)):
            d_c = dist_batch[local_idx]                           # (N,)

            # Radii sorted DESCENDING — enables break below
            radii = _sample_radii_descending(d_c, radii_per_center)
            if radii.numel() == 0:
                continue

            # Compute ball sizes for ALL radii in one vectorised op,
            # WITHOUT storing the full (radii × N) boolean mask matrix.
            # Shape: (k, N) → sum → (k,); freed immediately.
            ball_sizes = (
                d_c.unsqueeze(0) <= radii.unsqueeze(1) + EPS
            ).sum(dim=1)                                          # (k,)

            # ── per-radius scan ───────────────────────────────────────────────
            for ridx in range(radii.numel()):
                ball_size = int(ball_sizes[ridx].item())

                # BREAK (not continue): radii are descending → ball_sizes are
                # non-increasing. No later radius can produce a larger ball.
                if ball_size <= lambda_max:
                    break

                # Compute this radius's mask lazily (only when needed)
                mask  = d_c <= radii[ridx] + EPS
                B_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)  # (k,)

                # Subsample oversized balls
                if _max_ball is not None and B_idx.numel() > _max_ball:
                    perm  = torch.randperm(B_idx.numel(), device=device)[:_max_ball]
                    B_idx = B_idx[perm]

                # BREAK (not continue): same monotonicity argument applies after
                # subsampling — _max_ball is a global cap, so subsequent (smaller)
                # radii will produce subsampled balls of the same or smaller size.
                if B_idx.numel() <= lambda_max:
                    break

                # ── ball pairwise distances (GPU) ─────────────────────────────
                if D is not None:
                    D_ball = D[B_idx][:, B_idx]
                else:
                    with torch.no_grad():
                        D_ball = torch.cdist(X[B_idx], X[B_idx], p)

                # Build adjacency at half-radius on GPU, transfer once to CPU
                adjacency: np.ndarray = (
                    D_ball <= radii[ridx] / 2.0 + EPS
                ).cpu().numpy()                                   # (m, m) bool

                # ── greedy cover (CPU, vectorised numpy) ──────────────────────
                cover_count = _greedy_cover_count_vectorized(adjacency, lambda_max)

                if cover_count > lambda_max:
                    lambda_max = cover_count

    return math.log2(lambda_max), lambda_max
