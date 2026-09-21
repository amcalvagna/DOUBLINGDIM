import numpy as np

def center_nested_balls(X, center_idx, radii):
    c = X[center_idx]
    d = np.max(np.abs(X - c), axis=1)

    order = np.argsort(d)
    d_sorted = d[order]

    balls = []
    for r in radii:
        k = np.searchsorted(d_sorted, r, side="right")
        balls.append(order[:k])

    return balls