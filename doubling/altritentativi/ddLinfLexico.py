import numpy as np


def recursive_lexico_box_cover(S, r):
    order = np.lexsort([S[:, j] for j in reversed(range(S.shape[1]))])
    S = S[order]

    def rec(block):
        spreads = block.max(axis=0) - block.min(axis=0)
        if spreads.max() <= r:
            return 1

        k = len(block) // 2
        left = block[:k]
        right = block[k:]
        return rec(left) + rec(right)

    return rec(S)