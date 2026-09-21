import math
import torch
from utils import show_bar
from data import DoublingDimResult, ddConfig

# API: provide configuration in a ddConfig object named config and 
# an estimator function with Estimator signature as below
# they will be imported with this file and invoked inside utils.py by standard computedoublingdim routine 

config = ddConfig(
    name = "ddLinfMixed",
    min_centers = 128,
    max_centers  = 256,
    nb_cap  = 256,
    max_radii  = 32,
    #-------------------#
    fps_block  = 4096,
    cdist_qblock = 64,       # small query blocks (centers to points)
    cdist_rblock  = 2048,      # larger result blocks (safe for MPS)
    cover_block  = 1024,
    radii  = None,
    anchor_samples = 128,
    radius_quantiles  = (0.50, 0.70, 0.85, 0.93)
)


def approximate_dd_linf(X):  #recursive alternative implementation, 
    cover = 1
    sl = torch.arange(X.shape[0]) #start with all elements
    stack=[sl]
    
    def recursive_cover(X, sl, cover):
        show_bar(cover, X.shape[0])
       
        S = X[sl,:]  # prendi solo i punti dell'insieme corrente
        #calcola il diametro, è per forza l'intervallo maggiore in una sola dimensione...in Linf
        mins = S.min(dim=0).values  #valori minimi in tutte le dimensioni
        maxs = S.max(dim=0).values  #valori massimi in tutte le dimensioni
    
        spreads = torch.abs(maxs - mins) #ampiezze massime in tutte le dimensioni
        #s_idx = torch.argsort(spreads, descending=True) #indici intervalli in ordine di ampiezza decrescene
        #max_spread = spreads.max() #intervallo di ampiezza maggiore
        j = spreads.argmax() # dimensione con intervallo maggiore

        #divido S in due parti uguali lungo l'asse j
        vals = S[:, j]
        t = 0.5 * (vals.min() + vals.max())
        #t = vals.median()

        left_mask = vals <= t
        right_mask = ~left_mask

        left = torch.nonzero(left_mask, as_tuple=True)[0]
        right = torch.nonzero(right_mask, as_tuple=True)[0]
        
        #  run on the denser side only
        next = left if left.sum() > right.sum() else right
        
        ###### doubling the recursion is useless ###### 
        #left_cover = recursive_cover(X, left, cover+1) if left.shape[0]>1 else cover 
        #right_cover = recursive_cover(X, right, cover+1) if right.shape[0]>1 else cover 
        #return max(cover, left_cover, right_cover)
        
        return recursive_cover(X, next, cover+1) if next.shape[0]>1 else cover 

    cover = recursive_cover(X, sl, 1)
    return math.log2(cover), cover

def approximate_dd_linf_(X): 
    cover = 1
    sl = torch.arange(X.shape[0]) #start with all elements
    stack=[sl]

    while stack:
        show_bar(cover, X.shape[0])
        sl = stack.pop()
        m = sl.shape[0]
        if m <= 1 : continue
        cover += 1
        
        S = X[sl,:]  # prendi solo i punti dell'insieme corrente
        #calcola il diametro, è per forza l'intervallo maggiore in una sola dimensione...in Linf
        mins = S.min(dim=0).values  #valori minimi in tutte le dimensioni
        maxs = S.max(dim=0).values  #valori massimi in tutte le dimensioni
    
        spreads = torch.abs(maxs - mins) #ampiezze massime in tutte le dimensioni
        #s_idx = torch.argsort(spreads, descending=True) #indici intervalli in ordine di ampiezza decrescene
        #max_spread = spreads.max() #intervallo di ampiezza maggiore
        j = spreads.argmax() # dimensione con intervallo maggiore

        #divido S in due parti uguali lungo l'asse j
        vals = S[:, j]
        t = 0.5 * (vals.min() + vals.max())
        #t = vals.median()

        left_mask = vals <= t
        right_mask = ~left_mask

        # run on the denser side only
        mask = left_mask if left_mask.sum() > right_mask.sum() else right_mask

        left = torch.nonzero(left_mask, as_tuple=True)[0]
        #right = torch.nonzero(right_mask, as_tuple=True)[0]
        stack.append(left)
        #stack.append(right)

    return math.log2(cover), cover


# call adapter to match the API expected in util.computeDuoblingDim
def ddEstimator( #_doubling_dimension_linf_mps(
    coreset,
    center_samples,
    anchor_samples,
    centers_idx,
    radii,
    max_radii,
    radius_quantiles,
    neighborhood_cap,
    fps_block,
    cdist_qblock,
    cdist_rblock,
    cover_block,
    seed,
    device,
    sync_each_center,
) -> DoublingDimResult:
  
    dd, dc = approximate_dd_linf(X = coreset)

    return DoublingDimResult(
        estimate=dd,
        per_scale_max=torch.tensor(dc),
        radii= radii.detach().cpu(), #if radii is not None else None,
        sampled_centers=centers_idx.detach().cpu(), 
        #local_counts=local_counts.detach().cpu(),
        config={
            "device": device,
            "center_samples": int(center_samples),
            "anchor_samples": int(anchor_samples),
            "radius_quantiles": [],
            "neighborhood_cap": int(neighborhood_cap),
            "fps_block": int(fps_block),
            "cdist_qblock": int(cdist_qblock),
            "cdist_rblock": int(cdist_rblock),
            "cover_block": int(cover_block),
            "seed": int(seed),
            "sync_each_center": bool(sync_each_center),
        }
    )

# def cover_count_iterative_idx(X, idx, r):
#     stack = [idx]
#     cover = 0

#     while stack:
#         cur = stack.pop()
#         m = cur.numel()

#         if m <= 1:
#             cover += 1
#             continue

#         S = X[cur]
#         mins = S.min(dim=0).values
#         maxs = S.max(dim=0).values
#         spreads = maxs - mins
#         max_spread = spreads.max()

#         if max_spread <= r:
#             cover += 1
#             continue

#         j = spreads.argmax()
#         vals = S[:, j]
#         t = 0.5 * (vals.min() + vals.max())
#         #t = vals.median()

#         left_mask = vals <= t
#         right_mask = ~left_mask

#         if left_mask.sum() == 0 or right_mask.sum() == 0:
#             cover += 1
#             continue

#         stack.append(cur[right_mask])
#         stack.append(cur[left_mask])

#     return cover
