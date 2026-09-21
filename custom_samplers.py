import numpy as np
import torch.nn.functional as F

# Selection strategy functions
def first(indices, data_np, metric=None):
    """Select first point in hypercube."""
    return indices[0]


def last(indices, data_np, metric=None):
    """Select last point in hypercube."""
    return indices[-1]


def random(indices, data_np, metric=None, seed=None):
    """Select random point in hypercube."""
    rng = np.random.RandomState(seed)
    return rng.choice(indices)


def centroid(indices, data_np, metric):
    """Select point closest to centroid of hypercube."""
    points = data_np[indices]
    centroid = np.mean(points, axis=0)
    distances = metric(points, centroid)
    return indices[np.argmin(distances)]


def norm_min(indices, data_np, metric):
    """Select point with minimum distance from origin."""
    points = data_np[indices]
    distances = metric(points, None)
    return indices[np.argmin(distances)]

def norm_max(indices, data_np, metric):
    """Select point with maximum distance from origin."""
    points = data_np[indices]
    distances = metric(points, None)
    return indices[np.argmax(distances)]

       
