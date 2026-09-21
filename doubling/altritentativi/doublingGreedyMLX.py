import cProfile
import io
import math
import os
import pstats
import datetime 

import mlx.core as mx
from sympy import total_degree
import torch
from tqdm import tqdm

from globals import *
from utils import coreset


EPS = 1e-12


# def _to_list(array: mx.array) -> list:
#     mx.eval(array)
#     return array.tolist()


# def pairwise_distances_mlx(
#     X: mx.array | Sequence[Sequence[float]],
#     Y: mx.array | Sequence[Sequence[float]] | None = None,
#     p: float = math.inf,
# ) -> mx.array:
#     """MLX equivalent of torch.cdist for the p-norms used in doublingTorch.py."""
#     X = mx.array(X)
#     Y = X if Y is None else mx.array(Y)

#     diff = mx.expand_dims(X, 1) - mx.expand_dims(Y, 0)
#     abs_diff = mx.abs(diff)

#     # if p == math.inf:
#     distances = mx.max(abs_diff, axis=2)
#     # elif p == 1:
#     #     distances = mx.sum(abs_diff, axis=2)
#     # elif p == 2:
#     #     distances = mx.sqrt(mx.sum(diff * diff, axis=2))
#     # elif p > 0:
#     #     distances = mx.power(mx.sum(mx.power(abs_diff, p), axis=2), 1.0 / p)
#     # else:
#     #     raise ValueError("p must be positive or math.inf.")

    # mx.eval(distances)
    # return distances


def profile_new_approximate_dd_mlx_optimized(
    *,
    setting,
    ratio,
    row_limit=25,
    **dd_kwargs,
) -> tuple[float, int]:
    """Lightweight cProfile wrapper mirroring the Torch profiling entry point."""
    profiler = cProfile.Profile()
    profiler.enable()
    ddim, lam = approximate_doubling_dimension_mlx(**dd_kwargs)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(row_limit)
    logging.info(
        "MLX profiler summary for setting=%s ratio=%.4f\n%s",
        setting,
        ratio,
        stream.getvalue(),
    )
    return ddim, lam


# def new_approximate_dd_mlx_optimized(
#     X: mx.array | Sequence[Sequence[float]],
#     D: mx.array | Sequence[Sequence[float]] | None = None,
#     p: float = math.inf,
#     num_centers: int = 128,
#     radii_per_center: int = 12,
#     max_ball_size: int | float = math.inf,
#     batch_size: int = 1,
#     seed: int | None = None,
# ) -> tuple[float, int]:
#     """
#     MLX port of new_approximate_dd_gpu_optimized from doublingTorch.py.

#     The algorithm is the same:
#     - choose candidate centers
#     - sample radii from sorted positive distances
#     - build each ball
#     - cover it greedily with half-radius balls using bitset acceleration
#     """
#     X = mx.array(X)
#     if X.ndim != 2:
#         raise ValueError("X must be a 2D array with shape (n_points, n_features).")

#     if seed is not None:
#         mx.random.seed(seed)
#     n_points = X.shape[0]
#     if n_points == 0:
#         return 0.0, 0
#     if n_points == 1:
#         return 0.0, 1

#     D = None if D is None else mx.array(D)
#     all_indices = mx.arange(n_points, dtype=mx.int32)
#     reduce_distances = (
#         (lambda diff, abs_diff: mx.max(abs_diff, axis=2))
#         # if p == math.inf
#         # else (lambda diff, abs_diff: mx.sum(abs_diff, axis=2))
#         # if p == 1
#         # else (lambda diff, abs_diff: mx.sqrt(mx.sum(diff * diff, axis=2)))
#         # if p == 2
#         # else (lambda diff, abs_diff: mx.power(mx.sum(mx.power(abs_diff, p), axis=2), 1.0 / p))
#         # if p > 0
#         # else None
#     )
#     if reduce_distances is None:
#         raise ValueError("p must be positive or math.inf.")

#     if num_centers < n_points:
#         center_indices = mx.random.permutation(n_points)[:num_centers].astype(mx.int32)
#     else:
#         center_indices = all_indices

#     lambda_max = 1
#     batch_size = max(1, batch_size)
#     total_batches = math.ceil(center_indices.shape[0] / batch_size)

#     for batch_start in tqdm(
#         range(0, center_indices.shape[0], batch_size),
#         total=total_batches,
#         desc="DD centers mlx",
#     ):
#         center_batch = center_indices[batch_start : batch_start + batch_size]

#         if D is None:
#             batch_points = mx.take(X, center_batch, axis=0)
#             diff = mx.expand_dims(batch_points, 1) - mx.expand_dims(X, 0)
#             abs_diff = mx.abs(diff)
#             dist_batch = reduce_distances(diff, abs_diff)
#         else:
#             dist_batch = mx.take(D, center_batch, axis=0)

#         mx.eval(dist_batch)

#         for local_idx in range(center_batch.shape[0]):
#             d_c = dist_batch[local_idx]
#             #indices = mx.sort(d_c[d_c > 0.0])[0]
#             # Extract only those values
#             #positive_vals = d_c[indices]
#             # Now sort
#             #positive = mx.sort(positive_vals)
            
#             mask = d_c > 0.0
#             # Get indices where condition is true
#             positives_only = mx.where(mask, d_c, -float('inf'))

#             sorted_all = mx.sort(positives_only)
#             num_positives = mask.sum().item()
#             positive = sorted_all[-int(num_positives):]

#             #oppure...
#             #positive = mx.compress(mask, distances, axis=0)
#             #positive = mx.sort(positive)

        
#             mx.eval(positive)
#             if positive.shape[0] == 0:
#                 continue

#             if positive.shape[0] <= radii_per_center:
#                 radii = positive
#             else:
#                 sample_idx = mx.round(
#                     mx.linspace(0, positive.shape[0] - 1, radii_per_center)
#                 ).astype(mx.int32)
#                 radii = mx.take(positive, sample_idx, axis=0)

#             ball_masks = mx.expand_dims(d_c, 0) <= (mx.expand_dims(radii, 1) + EPS)
#             ball_sizes = mx.sum(ball_masks, axis=1)
#             mx.eval(ball_masks, ball_sizes)

#             for ridx in range(radii.shape[0]):
#                 ball_size = int(ball_sizes[ridx].item())
#                 if ball_size <= lambda_max:
#                     continue

#                 #B_idx = all_indices[ball_masks[ridx]]
#                 mask = ball_masks[ridx].astype(mx.int32)
#                 B_idx = mx.argsort(-mask)[:ball_size].astype(mx.int32)
#                 mx.eval(B_idx)

#                 if B_idx.shape[0] > max_ball_size:
#                     perm = mx.random.permutation(B_idx.shape[0])[: int(max_ball_size)].astype(mx.int32)
#                     B_idx = mx.take(B_idx, perm, axis=0)
#                     mx.eval(B_idx)
#                     if B_idx.shape[0] <= lambda_max:
#                         continue

#                 if D is not None:
#                     D_ball = mx.take(mx.take(D, B_idx, axis=0), B_idx, axis=1)
#                 else:
#                     ball_points = mx.take(X, B_idx, axis=0)
#                     ball_diff = mx.expand_dims(ball_points, 1) - mx.expand_dims(ball_points, 0)
#                     ball_abs_diff = mx.abs(ball_diff)
#                     D_ball = reduce_distances(ball_diff, ball_abs_diff)

#                 adjacency = D_ball <= (radii[ridx] / 2.0 + EPS)
#                 mx.eval(adjacency)
#                 adjacency_np = np.asarray(_to_list(adjacency), dtype=np.uint8)
#                 packed_rows = np.packbits(adjacency_np, axis=1, bitorder="little")
#                 row_masks = [
#                     int.from_bytes(row.tobytes(), byteorder="little", signed=False)
#                     for row in packed_rows
#                 ]

#                 full_mask = (1 << B_idx.shape[0]) - 1
#                 covered_mask = 0
#                 cover_count = 0

#                 while covered_mask != full_mask:
#                     uncovered_mask = full_mask & ~covered_mask
#                     best_mask = 0
#                     best_gain = -1

#                     for row_mask in row_masks:
#                         gain = (row_mask & uncovered_mask).bit_count()
#                         if gain > best_gain:
#                             best_gain = gain
#                             best_mask = row_mask

#                     covered_mask |= best_mask
#                     cover_count += 1

#                     if cover_count > lambda_max:
#                         break

#                 if cover_count > lambda_max:
#                     lambda_max = cover_count

#     return math.log2(lambda_max), lambda_max


def computeDoublingDim(dataset, coresets_file, normP=math.inf, profile_dd=None):
    """
    MLX-backed adaptation of doublingTorch.computeDoublingDim with the same interface.

    It preserves the surrounding project contract:
    - loads the same coreset metadata
    - iterates over the same metric/sampler/ratio grid
    - returns the same nested `dds` structure
    """
    from time import time

    train_set, _, _ = dataset
    if profile_dd is None:
        profile_dd = os.getenv("DD_PROFILE", "0").lower() in {"1", "true", "yes", "on"}
    cover_backend = os.getenv("DD_COVER_BACKEND", "mlx").lower()

    profiling = lambda: "ON" if profile_dd else "OFF"
    logging.info(f"RUNNING WITH PROFILING {profiling()}")
    logging.info("Doubling dimension computations running on MLX.")
    logging.info(f"Using cover backend: {cover_backend}")

    coresets = torch.load(coresets_file, map_location="cpu")
    dds = {}

    for m, s in ((m, s) for m in metrics for s in samplers):
        setting = (m, s)
        dds[setting] = {}

        for r in ratios:
            max_centers = 128
            max_radii = 8

            coreset_idx = coresets[r][m][s]
            coreset_points_torch = coreset(train_set, coreset_idx).tensors[0]
            coreset_points = mx.array(coreset_points_torch.detach().cpu().numpy())

            num = len(coreset_idx)
            max_centers = min(max_centers, num) if max_centers else num
            logging.info(f"Computing DD for model: {nnet.fc_name} on {ds.name} coreset {setting} with compression:{r:.00%}")
            logging.info(f"Using max {num} random points as centers, {max_radii} spaced radiuses per center, early stop at max {num}")

            dist = None
            if num < 4096:
                dist = pairwise_linf_distances_mlx(coreset_points)
                logging.info("PRECOMPUTED ALL DISTANCES.")

            start = datetime.datetime.now()
            dd_kwargs = dict(
                X=coreset_points,
                D=dist,
                num_centers=num,
                radii_per_center=max_radii,
                max_ball_size=num,
                batch_size=1,
                seed=int(time()),
                cover_backend=cover_backend,
            )

            if profile_dd:
                ddim, lam = profile_new_approximate_dd_mlx_optimized(
                    setting=setting,
                    ratio=r,
                    row_limit=25,
                    **dd_kwargs,
                )
                profile_dd = False
            else:
                ddim, lam = approximate_doubling_dimension_mlx(**dd_kwargs) #new_approximate_dd_mlx_optimized(**dd_kwargs)

            elapsed = datetime.datetime.now() - start
            dds[setting][r] = ddim

            logging.info(f"Processed points amount: {num}")
            logging.info(f"Approx doubling constant λ: {lam}")
            logging.info(f"Approx doubling dimension  : {ddim}")
            logging.info(f"Elapsed time: {elapsed}.")

            del coreset_points_torch, coreset_points, dist

    return dds


if __name__ == "__main__":
    X = mx.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    ddim, lam = profile_new_approximate_dd_mlx_optimized(
        X,
        p=2,
        num_centers=4,
        radii_per_center=4,
        max_ball_size=4,
        batch_size=2,
        seed=0,
    )
    print("Approx doubling constant:", lam)
    print("Approx doubling dimension:", ddim)

#########################nuova versione chat gpt =====================================
def pairwise_linf_distances_mlx(
    X: mx.array ,
    Y: mx.array | None = None,
) -> mx.array:
    #X = mx.array(X)
    Y = X if Y is None else Y # mx.array(Y)

    diff = mx.expand_dims(X, 1) - mx.expand_dims(Y, 0)
    distances = mx.max(mx.abs(diff), axis=2)
    return distances

#CURRENT 
def sample_radii_from_distances(
    distances: mx.array,
    radii_per_center: int,
) -> mx.array:
    mask = distances > 0.0
    positive_count = int(mx.sum(mask).item())
    positive = mx.sort(mx.where(mask, distances, mx.inf))[:positive_count]
    mx.eval(positive)

    if positive.shape[0] == 0:
        return positive

    if positive.shape[0] <= radii_per_center:
        return positive

    sample_idx = mx.round(
        mx.linspace(0, positive.shape[0] - 1, radii_per_center)
    ).astype(mx.int32)
    return mx.take(positive, sample_idx, axis=0)

#PERPLEXITY
# def sample_radii_from_distances(
#     distances: mx.array,
#     radii_per_center: int,
# ) -> mx.array:
#     """
#     Sample up to `radii_per_center` positive radii from a 1D distance array.

#     Returns sorted radii. When subsampling, indices are chosen to be roughly
#     evenly spaced over the sorted positive distances.
#     """
#     if distances.ndim != 1:
#         raise ValueError("distances must be a 1D array.")

#     if radii_per_center <= 0:
#         return mx.array([], dtype=distances.dtype)

#     positive = mx.slice distances[distances > 0.0]
#     mx.eval(positive)

#     n_pos = positive.shape[0]
#     if n_pos == 0:
#         return positive

#     positive = mx.sort(positive)
#     mx.eval(positive)

#     if n_pos <= radii_per_center:
#         return positive

#     sample_idx = mx.linspace(0, n_pos - 1, radii_per_center, dtype=mx.float32)
#     sample_idx = mx.round(sample_idx).astype(mx.int32)

#     radii = mx.take(positive, sample_idx, axis=0)
#     return radii

def greedy_cover_count_mlx(adjacency: mx.array, early_stop: int) -> int:
    """
    MLX-native greedy dominating-set cover.

    Keeps the adjacency matrix and coverage state on device and only syncs the
    small scalar values needed for loop control.
    """
    size = adjacency.shape[0]
    adjacency_i = adjacency.astype(mx.int32)
    covered = mx.zeros((size,), dtype=mx.int32)
    covered_count = 0
    cover_count = 0

    while covered_count < size:
        uncovered = 1 - covered
        coverage_gain = mx.sum(adjacency_i * mx.expand_dims(uncovered, 0), axis=1)
        next_center = int(mx.argmax(coverage_gain).item())
        next_row = adjacency_i[next_center]
        new_covered = mx.maximum(covered, next_row)
        newly_covered = new_covered - covered
        covered = new_covered
        covered_count += int(mx.sum(newly_covered).item())
        cover_count += 1

        if cover_count > early_stop:
            break

    return cover_count


# def adjacency_to_row_masks(adjacency: mx.array) -> list[int]:
#     adjacency_np = np.asarray(adjacency.tolist(), dtype=np.uint8)
#     packed_rows = np.packbits(adjacency_np, axis=1, bitorder="little")
#     return [
#         int.from_bytes(row.tobytes(), byteorder="little", signed=False)
#         for row in packed_rows
#     ]


# def greedy_cover_count_cpu_bitset(adjacency: mx.array, early_stop: int) -> int:
#     row_masks = adjacency_to_row_masks(adjacency)
#     full_mask = (1 << adjacency.shape[0]) - 1
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

#         if cover_count > early_stop:
#             break

#     return cover_count

# #PERPLEXITY
# def approximate_doubling_dimension_mlx(
#     X: mx.array,
#     D: mx.array | None = None,
#     num_centers: int = 128,
#     radii_per_center: int = 12,
#     max_ball_size: int | float = math.inf,
#     batch_size: int = 1,
#     seed: int | None = None,
#     cover_backend: str = "mlx",
# ) -> tuple[float, int]:
#     if cover_backend not in {"mlx", "cpu_bitset"}:
#         raise ValueError("cover_backend must be 'mlx' or 'cpu_bitset'.")
#     if X.ndim != 2:
#         raise ValueError("X must be a 2D array with shape (n_points, n_features).")

#     if seed is not None:
#         mx.random.seed(seed)

#     n_points = X.shape[0]
#     if n_points == 0:
#         return 0.0, 0
#     if n_points == 1:
#         return 0.0, 1

#     all_indices = mx.arange(n_points, dtype=mx.int32)
#     center_indices = (
#         mx.random.permutation(n_points)[:num_centers].astype(mx.int32)
#         if num_centers < n_points else all_indices
#     )

#     batch_size = max(1, batch_size)
#     total_batches = math.ceil(center_indices.shape[0] / batch_size)
#     finite_cap = None if math.isinf(max_ball_size) else int(max_ball_size)

#     lambda_max = 1

#     for batch_start in tqdm(
#         range(0, center_indices.shape[0], batch_size),
#         total=total_batches,
#         desc="DD centers opt",
#     ):
#         center_batch = center_indices[batch_start: batch_start + batch_size]

#         if D is None:
#             batch_points = mx.take(X, center_batch, axis=0)
#             dist_batch = pairwise_linf_distances_mlx(batch_points, X)
#         else:
#             dist_batch = mx.take(D, center_batch, axis=0)
#         mx.eval(dist_batch)

#         for local_idx in range(center_batch.shape[0]):
#             d_c = dist_batch[local_idx]
#             radii = sample_radii_from_distances(d_c, radii_per_center)
#             if radii.shape[0] == 0:
#                 continue

#             perm = mx.argsort(d_c)
#             d_sorted = mx.take(d_c, perm, axis=0)
#             radii = mx.sort(radii)[::-1]
#             mx.eval(perm, d_sorted, radii)

#             for ridx in range(radii.shape[0]):
#                 r = radii[ridx]

#                 # replace with searchsorted-like helper
#                 ball_mask = d_sorted <= (r + EPS)
#                 ball_size = int(mx.sum(ball_mask).item())
#                 if finite_cap is not None:
#                     ball_size = min(ball_size, finite_cap)
#                 if ball_size <= lambda_max:
#                     break

#                 B_idx = perm[:ball_size]
#                 mx.eval(B_idx)

#                 if D is not None:
#                     D_ball = mx.take(mx.take(D, B_idx, axis=0), B_idx, axis=1)
#                 else:
#                     ball_points = mx.take(X, B_idx, axis=0)
#                     D_ball = pairwise_linf_distances_mlx(ball_points)

#                 adjacency = D_ball <= (r / 2.0 + EPS)
#                 mx.eval(adjacency)

#                 backend = cover_backend
#                 cover_count = greedy_cover_count_mlx(adjacency, early_stop=lambda_max)

#                 if cover_count > lambda_max:
#                     lambda_max = cover_count

#     return math.log2(lambda_max), lambda_max

#CLAUDE
def approximate_doubling_dimension_mlx(
    X: mx.array,
    D: mx.array | None = None,
    num_centers: int = 128,
    radii_per_center: int = 12,
    max_ball_size: int | float = math.inf,
    batch_size: int = 1,
    seed: int | None = None,
    cover_backend: str = "mlx",
) -> tuple[float, int]:
    if cover_backend not in {"mlx", "cpu_bitset"}:
        raise ValueError("cover_backend must be 'mlx' or 'cpu_bitset'.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_points, n_features).")

    if seed is not None:
        mx.random.seed(seed)

    n_points = X.shape[0]
    if n_points == 0:
        return 0.0, 0
    if n_points == 1:
        return 0.0, 1

    all_indices = mx.arange(n_points, dtype=mx.int32)

    if num_centers < n_points:
        center_indices = mx.random.permutation(n_points)[:num_centers].astype(mx.int32)
    else:
        center_indices = all_indices

    lambda_max = 1
    batch_size = max(1, batch_size)
    num_centers_actual = int(center_indices.shape[0])

    # Bug fix: was n_points/batch_size — should be based on number of centers
    total_batches = math.ceil(num_centers_actual / batch_size)

    # Hoist cover-backend dispatch out of the hot loop
    cover_fn = (
        greedy_cover_count_mlx if cover_backend == "mlx"
        else greedy_cover_count_cpu_bitset
    )
    # Precompute to avoid int() + math.isfinite() on every inner iteration
    _max_ball = int(max_ball_size) if math.isfinite(max_ball_size) else None

    for batch_start in tqdm(
        range(0, num_centers_actual, batch_size),
        total=total_batches,
        desc="DD centers opt",
    ):
        center_batch = center_indices[batch_start : batch_start + batch_size]

        if D is None:
            batch_points = mx.take(X, center_batch, axis=0)
            dist_batch = pairwise_linf_distances_mlx(batch_points, X)
        else:
            dist_batch = mx.take(D, center_batch, axis=0)

        mx.eval(dist_batch)

        for local_idx in range(center_batch.shape[0]):
            d_c = dist_batch[local_idx]
            radii = sample_radii_from_distances(d_c, radii_per_center)

            if radii.shape[0] == 0:
                continue

            # Sort descending: larger radius → larger (or equal) ball.
            # ball_sizes become monotonically non-increasing, enabling break below.
            radii = mx.sort(radii)[::-1]

            # Fuse the comparison + reduce into ball_sizes without materialising
            # the full (num_radii × n_points) boolean matrix in memory.
            ball_sizes = mx.sum(
                mx.expand_dims(d_c, 0) <= (mx.expand_dims(radii, 1) + EPS),
                axis=1,
            )
            mx.eval(ball_sizes)

            for ridx in range(radii.shape[0]):
                ball_size = int(ball_sizes[ridx].item())

                # Break (not continue): radii are descending so ball_sizes are too.
                # No subsequent radius can produce a strictly larger ball.
                if ball_size <= lambda_max:
                    break

                # Compute this radius's mask lazily — only when we actually need it.
                # bool NOT (~) avoids the unnecessary .astype(mx.int16) cast.
                mask_bool = d_c <= (radii[ridx] + EPS)
                B_idx = mx.argsort(~mask_bool.astype(mx.uint8))[:ball_size].astype(mx.int32)
                mx.eval(B_idx)

                if _max_ball is not None and B_idx.shape[0] > _max_ball:
                    perm = mx.random.permutation(B_idx.shape[0])[:_max_ball].astype(mx.int32)
                    B_idx = mx.take(B_idx, perm, axis=0)
                    mx.eval(B_idx)

                # Break is still safe here: B_idx.shape[0] = min(ball_size, _max_ball).
                # If that ≤ lambda_max then _max_ball ≤ lambda_max, so all remaining
                # (smaller) radii will also produce subsampled balls ≤ lambda_max.
                if B_idx.shape[0] <= lambda_max:
                    break

                if D is not None:
                    D_ball = mx.take(mx.take(D, B_idx, axis=0), B_idx, axis=1)
                else:
                    ball_points = mx.take(X, B_idx, axis=0)
                    D_ball = pairwise_linf_distances_mlx(ball_points)

                # Single graph eval: fuses D_ball computation into adjacency
                adjacency = D_ball <= (radii[ridx] / 2.0 + EPS)
                mx.eval(adjacency)

                cover_count = cover_fn(adjacency=adjacency, early_stop=lambda_max)

                if cover_count > lambda_max:
                    lambda_max = cover_count

    return math.log2(lambda_max), lambda_max

# CURRENT 
# def approximate_doubling_dimension_mlx(
#     X: mx.array ,
#     D: mx.array | None = None,
#     num_centers: int = 128,
#     radii_per_center: int = 12,
#     max_ball_size: int | float = math.inf,
#     batch_size: int = 1,
#     seed: int | None = None,
#     cover_backend: str = "mlx",
# ) -> tuple[float, int]:
#     if cover_backend not in {"mlx", "cpu_bitset"}:
#         raise ValueError("cover_backend must be 'mlx' or 'cpu_bitset'.")

#     if X.ndim != 2:
#         raise ValueError("X must be a 2D array with shape (n_points, n_features).")

#     if seed is not None:
#         mx.random.seed(seed)

#     n_points = X.shape[0]
#     if n_points == 0:
#         return 0.0, 0
#     if n_points == 1:
#         return 0.0, 1

#     #D = None if D is None else mx.array(D)
#     all_indices = mx.arange(n_points, dtype=mx.int32)

#     if num_centers < n_points:
#         center_indices = mx.random.permutation(n_points)[:num_centers].astype(mx.int32)
#     else:
#         center_indices = all_indices

#     lambda_max = 1
#     batch_size = max(1, batch_size)
#     total_batches = n_points/batch_size
    
#     for batch_start in tqdm(
#         range(0, center_indices.shape[0], batch_size),        
#         total=total_batches,
#         desc="DD centers opt",
#     ):
#         center_batch = center_indices[batch_start : batch_start + batch_size]

#         if D is None:
#             batch_points = mx.take(X, center_batch, axis=0)
#             dist_batch = pairwise_linf_distances_mlx(batch_points, X)
#         else:
#             dist_batch = mx.take(D, center_batch, axis=0)

#         mx.eval(dist_batch)

#         for local_idx in range(center_batch.shape[0]):
#             d_c = dist_batch[local_idx]
#             radii = sample_radii_from_distances(d_c, radii_per_center)

#             if radii.shape[0] == 0:
#                 continue

#             ball_masks = mx.expand_dims(d_c, 0) <= (mx.expand_dims(radii, 1) + EPS)
#             ball_sizes = mx.sum(ball_masks, axis=1)
#             mx.eval(ball_masks, ball_sizes)

#             for ridx in range(radii.shape[0]):
#                 ball_size = int(ball_sizes[ridx].item())
#                 if ball_size <= lambda_max:
#                     continue

#                 #B_idx = all_indices[ball_masks[ridx]]
#                 mask = ball_masks[ridx].astype(mx.int16)
#                 B_idx = mx.argsort(-mask)[:ball_size].astype(mx.int32)
#                 mx.eval(B_idx)

#                 if B_idx.shape[0] > max_ball_size:
#                     perm = mx.random.permutation(B_idx.shape[0])[: int(max_ball_size)].astype(mx.int32)
#                     B_idx = mx.take(B_idx, perm, axis=0)
#                     mx.eval(B_idx)

#                 if B_idx.shape[0] <= lambda_max:
#                     continue

#                 if D is not None:
#                     D_ball = mx.take(mx.take(D, B_idx, axis=0), B_idx, axis=1)
#                 else:
#                     ball_points = mx.take(X, B_idx, axis=0)
#                     D_ball = pairwise_linf_distances_mlx(ball_points)

#                 adjacency = D_ball <= (radii[ridx] / 2.0 + EPS)
#                 mx.eval(adjacency)

#                 if cover_backend == "mlx":
#                     cover_count = greedy_cover_count_mlx(
#                         adjacency=adjacency,
#                         early_stop=lambda_max,
#                     )
#                 else:
#                     cover_count = greedy_cover_count_cpu_bitset(
#                         adjacency=adjacency,
#                         early_stop=lambda_max,
#                     )

#                 if cover_count > lambda_max:
#                     lambda_max = cover_count

#     return math.log2(lambda_max), lambda_max
