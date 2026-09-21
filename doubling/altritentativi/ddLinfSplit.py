def split_by_max_spread_coordinate(S):
    spreads = S.max(axis=0) - S.min(axis=0)
    j = spreads.argmax()

    order = np.argsort(S[:, j])
    S_sorted = S[order]
    vals = S_sorted[:, j]

    candidates = []

    # median split
    k = len(S_sorted) // 2
    candidates.append((S_sorted[:k], S_sorted[k:]))

    # largest-gap split
    if len(vals) >= 2:
        gaps = vals[1:] - vals[:-1]
        g = gaps.argmax()
        candidates.append((S_sorted[:g+1], S_sorted[g+1:]))

    # choose the better candidate by a cheap criterion
    best = None
    best_score = float("inf")
    for A, B in candidates:
        if len(A) == 0 or len(B) == 0:
            continue
        score = max(diam_linf(A), diam_linf(B))
        if score < best_score:
            best_score = score
            best = (A, B)

    return best