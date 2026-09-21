import numpy as np
import torch
import torch.nn.functional as F
import ot  # POT: Python Optimal Transport
from scipy.spatial import distance

epsilon = 1e-15

def to_pdf(data):  # block conversion of a batch of samples to probability distributions (sum to 1)
    #assume data is 2D batch of N features tensor [N, D]
    sum = torch.sum(data, dim=1, keepdim=True) 
    #logging.debug(f"{data.shape}, {sum.shape}")
    pdf = data / sum[:,]  # divide each row by its sum
    #logging.debug(pdf.sum(dim=1))  # should be all ones
    return pdf # shape [N, D]

# ---------------------------------    
# artificial reference data samples
# ---------------------------------
def dirac_delta(size):  # convert to one-hot encoding (max to 1, rest to 0)
    d = np.zeros(size)
    d[0] = 1.0
    return d

def uniform(size):
    u = np.ones(size)
    u /= size # normalize to sum to 1
    return u

def gaussian(size, mean=0.5, std=0.1):
    x = np.linspace(0, 1, size)  # shape [size]
    g = np.exp(-0.5 * ((x - mean) / std) ** 2)
    g /= g.sum()  # normalize to sum to 1
    return g



#def cosine_sim(x, y): return 1 - F.cosine_similarity(x, y, dim=0) if y is not None  # convert to distance
def random_noise(x, _): np.random.seed(42); sample = np.random.uniform(0, 1, x.shape[1]); return sample/sample.sum() # emulate uniform random distances from hypotetical ref.  
def random_gauss(x, _): np.random.seed(42); return np.random.normal(loc = x.shape[1]/2, scale=0.5) # emulate gaussian random distances of x from hypotetical ref.

# Distance metric functions
def euclidean(points, reference=None):
    """
    Compute Euclidean distance.
    
    Args:
        points: Array of shape (n_points, n_dims)
        reference: Reference point of shape (n_dims,). If None, computes norm from origin.
    
    Returns:
        distances: Array of shape (n_points,)
    """
    if reference is None:
        return np.linalg.norm(points, axis=1) #from origin
    else:
        return np.linalg.norm(points - reference, axis=1)

def manhattan(points, reference=None):
    """Compute Manhattan (L1) distance."""
    if reference is None:
        return np.sum(np.abs(points), axis=1)
    else:
        return np.sum(np.abs(points - reference), axis=1)


def chebyshev(points, reference=None):
    """Compute Chebyshev (L-infinity) distance."""
    if reference is None:
        return np.max(np.abs(points), axis=1)
    else:
        return np.max(np.abs(points - reference), axis=1)


def cosine(points, reference=None):
    """
    Compute cosine distance (1 - cosine similarity).
    Note: Returns distance from origin if reference is None.
    """
    if reference is None:
        # For cosine without reference, return norm (distance from origin)
        return np.linalg.norm(points, axis=1)
    else:
        # Cosine distance
        dots = np.dot(points, reference)
        norms_points = np.linalg.norm(points, axis=1)
        norm_ref = np.linalg.norm(reference)
        # Avoid division by zero
        cosine_sim = dots / (norms_points * norm_ref + epsilon)
        return 1 - cosine_sim

# Custom metric: weighted Euclidean (emphasize first 10 dimensions)
def weuclidean(points, reference=None):
    weights = np.ones(points.shape[1])
    weights[:10] = 2.0  # Double weight for first 10 dimensions
    if reference is None:
        return np.sqrt(np.sum((points ** 2) * weights, axis=1))
    else:
        return np.sqrt(np.sum(((points - reference) ** 2) * weights, axis=1))


def wasserstein_fast1D(points, reference=None):
    points = np.asarray(points, dtype=float)

    # Normalize rows to sum to 1
    points = np.maximum(points, 0)
    points /= points.sum(axis=1, keepdims=True) #+ eps

    # Reference distribution
    if reference is None:
        reference = np.ones(points.shape[1]) / points.shape[1]
    else:
        reference = np.asarray(reference, dtype=float)
        reference = np.maximum(reference, 0)
        reference /= reference.sum() #+ eps

    # Compute cumulative distributions
    cdf_points = np.cumsum(points, axis=1)
    cdf_ref = np.cumsum(reference)

    # Wasserstein-1 distances (L1 distance between CDFs)
    dists = np.sum(np.abs(cdf_points - cdf_ref), axis=1)
    return dists

def wasserstein_1D(points, reference=None): # from origin if reference is None    
    n = points.shape[1] # support size

    # normalize each row sum to one (like a pdf)    
    row_sums = points.sum(axis=1, keepdims=True) #+ eps
    points = points / row_sums

    if reference is None:
        reference = np.float64(np.ones(n) / n)  # uniform distribution
    else: reference /=  np.sum(reference) #+ eps
    
    a = np.arange(n) # linear weights
    
    w = lambda p : ot.wasserstein_1d(a, a, p, reference)   
    dists = [w(p) for p in points]    
    return dists


def wasserstein(points, reference=None): # from origin if reference is None    
    n = points.shape[1] # support size

    # normalize each row sum to one (like a pdf)
    row_sums = points.sum(axis=1, keepdims=True) #+ eps
    points = points / row_sums

    if reference is None:
        reference = np.float64(np.ones(n) / n)  # uniform distribution
    else: reference /=  np.sum(reference) #+ eps
    
    a = np.arange(n) # linear weights
    M = ot.dist(a.reshape(-1,1), a.reshape(-1,1), metric='sqeuclidean') # cost matrix
    
    w = lambda p : np.sqrt(ot.emd2(p, reference, M)) # uncomment M computation above
    
    dists = [w(p) for p in points]    
    return dists

def sinkhorn(points, reference=None): # WARNING: not perfectly working yet
    n = points.shape[1] # support size

    points = np.maximum(points, epsilon)
    row_sums = points.sum(axis=1, keepdims=True) + epsilon
    points = points / row_sums

    if reference is None:
        reference = np.float64(np.ones(n) / n)  # uniform distribution
    else: reference /=  np.sum(reference) + epsilon
    
    a = np.arange(n) # linear weights
    M = ot.dist(a.reshape(-1,1), a.reshape(-1,1), metric='sqeuclidean') # cost matrix
    
    w = lambda p : ot.sinkhorn(p, reference, M, 1e-3, verbose=False)  # entropic regularization
    dists = [w(p) for p in points]    
    return dists

def mahalanobis(points, reference=None):
    # Step 1: Mean of data
    reference = np.mean(points, axis=0, dtype=np.float64) if reference is None else reference 

    # Step 2: Covariance matrix
    cov_matrix = np.cov(points, rowvar=False, dtype=np.float64)
    # ensure cov is 2D matrix (np.cov can return scalar for degenerate inputs)
    if cov_matrix.ndim == 0:
        #cov_matrix = np.array([[cov_matrix]])
        return np.mean(points, axis=0) - reference
    if np.any(np.all(cov_matrix == 0, axis=1)) : 
            cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-15 # regularize to avoid singularity

    try: 
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_matrix)  # use pseudo-inverse if singular

    # Step 4: Compute Mahalanobis distance for each point    
    distances = np.array([distance.mahalanobis(p, reference, inv_cov) for p in points])
    return distances

