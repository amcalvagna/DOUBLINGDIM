import numpy as np


def recursive_cover_by_maxspread_split(
    S: np.ndarray,
    r: float,
    *,
    min_split_size: int = 1,
    max_candidates: int = 3,
) -> int:
    """
    Heuristic recursive cover count for a finite set S under L_infinity.

    Returns an upper-bound-style estimate of how many subsets of diameter <= r
    are needed to cover S, using recursive splits along a coordinate with
    maximum spread.

    Parameters
    ----------
    S : np.ndarray
        Array of shape (n_points, n_features).
    r : float
        Target diameter threshold for terminal pieces.
    min_split_size : int
        Minimum allowed size for each child after a split.
    max_candidates : int
        Maximum number of split candidates evaluated among:
        midpoint, median, largest-gap.

    Returns
    -------
    int
        Number of terminal subsets produced by the recursive splitting rule.
    """
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError("S must have shape (n_points, n_features).")
    if r < 0:
        raise ValueError("r must be nonnegative.")

    def diam_and_spreads(X: np.ndarray):
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        spreads = maxs - mins
        diam = float(spreads.max()) if X.shape[0] > 0 else 0.0
        return diam, spreads, mins, maxs

    def split_score(A: np.ndarray, B: np.ndarray) -> tuple[float, float, int]:
        """
        Lower is better.
        Primary: max child diameter
        Secondary: sum child diameters
        Tertiary: balance penalty
        """
        diam_a, _, _, _ = diam_and_spreads(A)
        diam_b, _, _, _ = diam_and_spreads(B)
        balance = abs(len(A) - len(B))
        return (max(diam_a, diam_b), diam_a + diam_b, balance)

    def build_candidates(X: np.ndarray, j: int, mins: np.ndarray, maxs: np.ndarray):
        """
        Returns a list of candidate (left_idx, right_idx) splits.
        """
        vals = X[:, j]
        order = np.argsort(vals, kind="mergesort")
        vals_sorted = vals[order]
        n = len(vals_sorted)

        candidates = []

        # 1) Midpoint split
        midpoint = 0.5 * (mins[j] + maxs[j])
        left = order[vals_sorted <= midpoint]
        right = order[vals_sorted > midpoint]
        if len(left) >= min_split_size and len(right) >= min_split_size:
            candidates.append((left, right))

        # 2) Median split by index
        k = n // 2
        if min_split_size <= k <= n - min_split_size:
            left = order[:k]
            right = order[k:]
            if len(left) > 0 and len(right) > 0:
                candidates.append((left, right))

        # 3) Largest-gap split
        if n >= 2:
            gaps = vals_sorted[1:] - vals_sorted[:-1]
            g = int(np.argmax(gaps))
            k = g + 1
            if min_split_size <= k <= n - min_split_size:
                left = order[:k]
                right = order[k:]
                if len(left) > 0 and len(right) > 0:
                    candidates.append((left, right))

        # Deduplicate equivalent splits
        dedup = []
        seen = set()
        for left, right in candidates[:max_candidates]:
            key = (tuple(left.tolist()), tuple(right.tolist()))
            if key not in seen:
                seen.add(key)
                dedup.append((left, right))

        return dedup

    def rec(X: np.ndarray) -> int:
        n = X.shape[0]
        if n == 0:
            return 0
        if n == 1:
            return 1

        diam, spreads, mins, maxs = diam_and_spreads(X)
        if diam <= r:
            return 1

        # Coordinates sorted by decreasing spread
        coords = np.argsort(-spreads)

        best_split = None
        best_score_value = None

        # Try coordinates with positive spread until we find a viable split
        for j in coords:
            if spreads[j] <= 0:
                break

            candidates = build_candidates(X, int(j), mins, maxs)
            for left_idx, right_idx in candidates:
                A = X[left_idx]
                B = X[right_idx]
                score = split_score(A, B)

                if best_score_value is None or score < best_score_value:
                    best_score_value = score
                    best_split = (A, B)

            if best_split is not None:
                break

        # No non-degenerate split found
        if best_split is None:
            return 1

        A, B = best_split
        return rec(A) + rec(B)

    return rec(S)