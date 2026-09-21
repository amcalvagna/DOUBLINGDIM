import itertools
import numpy as np
from math import log2

# calcola la dd in modo esatto: improponibile computazionalmente

def pairwise_distances(X):
    d = np.linalg.norm(X[:,None] - X[None,:], axis=2)
    return d

def exact_min_cover(ball_points, cover_sets):
    """
    Given:
      ball_points = set of indices
      cover_sets = list of sets, each is a cover set for a center
    Returns:
      minimum number of cover sets to cover ball_points
    """
    m = len(cover_sets)
    for r in range(1, m+1):
        for combo in itertools.combinations(range(m), r):
            union_set = set()
            for idx in combo:
                union_set |= cover_sets[idx]
            if union_set >= ball_points:
                return r
    return m

def doubling_dimension(X):
    """
    Exact doubling dimension of a finite dataset X (N x d)
    """
    N = len(X)
    D = pairwise_distances(X)
    print(D)
    lambda_max = 1

    input()

    for center in range(N):
        # Consider all candidate radii: distances to other points
        radii = sorted(set(D[center]))
        for R in radii:
            # Points in the ball B(center, R)
            B = set(np.where(D[center] <= R + 1e-12)[0])
            if len(B) <= 1:
                continue

            # Build all radius R/2 balls around points in B
            cover_sets = []
            for p in B:
                C = set(np.where(D[p] <= R/2 + 1e-12)[0])
                cover_sets.append(C & B)

            # Compute exact minimum cover number
            cov = exact_min_cover(B, cover_sets)
            lambda_max = max(lambda_max, cov)

    return log2(lambda_max), lambda_max


X = np.random.randn(20000, 3)   # 200 points in 3D
ddim, lam = doubling_dimension(X)

print("Doubling constant:", lam)
print("Doubling dimension:", ddim)
