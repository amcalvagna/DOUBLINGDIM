from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

import mlx.core as mx


EPS = 1e-12


def pairwise_distances_mlx(X: mx.array | Sequence[Sequence[float]], p: float = 2) -> mx.array:
    """Compute the full pairwise distance matrix with MLX."""
    X = mx.array(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_points, n_features).")

    diff = mx.expand_dims(X, 1) - mx.expand_dims(X, 0)
    abs_diff = mx.abs(diff)

    if p == math.inf:
        D = mx.max(abs_diff, axis=2)
    elif p == 1:
        D = mx.sum(abs_diff, axis=2)
    elif p == 2:
        D = mx.sqrt(mx.sum(diff * diff, axis=2))
    elif p > 0:
        D = mx.power(mx.sum(mx.power(abs_diff, p), axis=2), 1.0 / p)
    else:
        raise ValueError("p must be positive or math.inf.")

    mx.eval(D)
    return D


def _sorted_unique(values: mx.array) -> list[float]:
    mx.eval(values)
    return sorted(set(float(v) for v in values.tolist()))


def _bools_to_bitmask(flags: Iterable[bool]) -> int:
    mask = 0
    for idx, flag in enumerate(flags):
        if flag:
            mask |= 1 << idx
    return mask


def exact_min_cover_bitmasks(full_mask: int, cover_masks: Sequence[int]) -> int:
    """
    Exact minimum number of cover sets needed to cover all bits in full_mask.

    This is the brute-force combinatorial step from the exact algorithm.
    """
    if full_mask == 0:
        return 0

    unique_masks = [m for m in dict.fromkeys(cover_masks) if m != 0]
    m = len(unique_masks)
    for r in range(1, m + 1):
        for combo in itertools.combinations(range(m), r):
            union_mask = 0
            for idx in combo:
                union_mask |= unique_masks[idx]
            if union_mask == full_mask:
                return r
    return m


def doubling_constant_exact_mlx(
    X: mx.array | Sequence[Sequence[float]],
    p: float = 2,
    eps: float = EPS,
) -> int:
    """
    Exact doubling constant using the same ball-cover definition as doubling.py.

    Notes:
    - This is exact but combinatorial and becomes intractable quickly.
    - `doublingTorch.py` currently contains approximate greedy variants; the exact
      brute-force routine in this repository lives in `doubling.py`.
    """
    X = mx.array(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_points, n_features).")

    n_points = X.shape[0]
    if n_points == 0:
        return 0
    if n_points == 1:
        return 1

    D = pairwise_distances_mlx(X, p=p)
    lambda_max = 1

    for center in range(n_points):
        radii = _sorted_unique(D[center])
        for radius in radii:
            ball_mask = D[center] <= radius + eps
            B_idx = [i for i, keep in enumerate(ball_mask.tolist()) if keep]
            if len(B_idx) <= 1:
                continue

            B_idx_mx = mx.array(B_idx)
            D_ball = mx.take(mx.take(D, B_idx_mx, axis=0), B_idx_mx, axis=1)
            mx.eval(D_ball)

            half_radius = radius / 2.0
            cover_masks = []
            for local_center in range(len(B_idx)):
                local_cover = D_ball[local_center] <= half_radius + eps
                cover_masks.append(_bools_to_bitmask(local_cover.tolist()))

            full_mask = (1 << len(B_idx)) - 1
            cover_number = exact_min_cover_bitmasks(full_mask, cover_masks)
            lambda_max = max(lambda_max, cover_number)

    return lambda_max


def doubling_dimension_exact_mlx(
    X: mx.array | Sequence[Sequence[float]],
    p: float = 2,
    eps: float = EPS,
) -> tuple[float, int]:
    """Return the exact doubling dimension and doubling constant."""
    lambda_max = doubling_constant_exact_mlx(X, p=p, eps=eps)
    if lambda_max <= 1:
        return 0.0, lambda_max
    return math.log2(lambda_max), lambda_max


if __name__ == "__main__":
    X = mx.random.uniform(shape=(100, 512))
    # X = mx.array(
    #     [
    #         [0.0, 0.0],
    #         [1.0, 0.0],
    #         [0.0, 1.0],
    #         [1.0, 1.0],
    #     ]
    # )
    ddim, lam = doubling_dimension_exact_mlx(X, p=math.inf)
    print("Doubling constant:", lam)
    print("Doubling dimension:", ddim)
