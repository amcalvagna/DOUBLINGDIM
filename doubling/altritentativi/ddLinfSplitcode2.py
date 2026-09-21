import numpy as np


def cover_by_maxspread_split(S: np.ndarray, r: float) -> int:
    """
    Minimal recursive cover heuristic under L_infinity.

    Parameters
    ----------
    S : np.ndarray
        Shape (n_points, n_features).
    r : float
        Target maximum diameter for each terminal subset.

    Returns
    -------
    int
        Number of terminal subsets in the recursive cover.
    """
    S = np.asarray(S, dtype=float)

    if S.ndim != 2:
        raise ValueError("S must be a 2D array.")
    if r < 0:
        raise ValueError("r must be nonnegative.")

    def rec(X: np.ndarray) -> int:
        n = X.shape[0]
        if n == 0:
            return 0
        if n == 1:
            return 1

        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        spreads = maxs - mins
        diam = spreads.max()

        if diam <= r:
            return 1

        j = int(np.argmax(spreads))
        midpoint = 0.5 * (mins[j] + maxs[j])

        left_mask = X[:, j] <= midpoint
        right_mask = ~left_mask

        # Fallback if midpoint split is degenerate
        if left_mask.all() or right_mask.all():
            vals = X[:, j]
            order = np.argsort(vals, kind="mergesort")
            k = n // 2
            left_idx = order[:k]
            right_idx = order[k:]
            if len(left_idx) == 0 or len(right_idx) == 0:
                return 1
            return rec(X[left_idx]) + rec(X[right_idx])

        return rec(X[left_mask]) + rec(X[right_mask])

    return rec(S)