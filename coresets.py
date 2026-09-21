#!python3
"""
    divido lo spazio delle features usando una griglia regolare
    in modo da selezionare per ogni ipercubo al più un solo punto 
    e avere punti ben distribuiti nello spazio delle features
"""

#import json
from utils import *

def compute_base_grid_sizes(data_np):
    base_grid_sizes = np.zeros(data_np.shape[1])
    for dim in range(data_np.shape[1]):
        # Get all values in this dimension
        values = data_np[:, dim]
        non_zero_values = values[values != 0]        
        assert len(non_zero_values) > 1, f"Dimension [{dim}] has insufficient non-zero values."
        sorted_values = np.sort(non_zero_values) 
        diffs = np.diff(sorted_values)
        non_zero_diffs = diffs[diffs != 0]  # optional??
        assert len(non_zero_diffs) > 0, f"Dimension [{dim}] has no non-zero differences."   
        base_grid_sizes[dim] = np.max(non_zero_diffs) #np.mean(non_zero_diffs)
    return base_grid_sizes

# Helper function to count unique hypercubes for given grid sizes
def count_hypercubes(grid_sizes, data_np):
    n_samples = len(data_np)
    # Avoid division by zero
    safe_grid_sizes = np.where(grid_sizes == 0, 1.0, grid_sizes)
    grid_indices = np.floor(data_np / safe_grid_sizes).astype(np.int64)  
    # Count unique hypercubes
    unique_cubes = set()
    for idx in range(n_samples):
        grid_key = tuple(grid_indices[idx])
        unique_cubes.add(grid_key)   
    return len(unique_cubes)

def compute_best_mult(base_sizes, data_np, target_ratio=None, max_iterations=20):   
    base_grid_sizes = base_sizes #compute_base_grid_sizes(data_np)
    # If no target ratio specified, return base grid sizes
    if target_ratio is None: 
        return base_grid_sizes
    logging.info(f"Computing grid sizes to achieve ~{target_ratio:.2%} compression ratio...")   
    n_samples = len(data_np)
    # Binary search for the right multiplier
    low_mult = 1  # Very small cells (high compression)
    high_mult = 100*np.max(base_grid_sizes)#/np.min(base_grid_sizes)  # Very large cells (low compression)
    best_mult = 1.0
    best_ratio_diff = float('inf')   
    target_count = int(n_samples * target_ratio)
    
    for iteration in range(max_iterations):
        mid_mult = (low_mult + high_mult) / 2
        test_grid_sizes = base_grid_sizes * mid_mult   
        cube_count = count_hypercubes(test_grid_sizes, data_np)
        actual_ratio = cube_count / n_samples
        ratio_diff = abs(actual_ratio - target_ratio)   
        logging.info(f"  Iteration {iteration+1}: multiplier={mid_mult:.4f}, "
              f"selected={cube_count}/{n_samples} ({actual_ratio:.2%})")    
        # Track best result
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_mult = mid_mult   
        # Binary search adjustment
        if cube_count > target_count:
            # Too many cubes (not enough compression), need larger cells
            low_mult = mid_mult
        else:
            # Too few cubes (too much compression), need smaller cells
            high_mult = mid_mult
        # Early exit if close enough
        #if ratio_diff < 0.0001:  break # Within .1%     
    logging.info(f"  Final: multiplier={best_mult:.4f} ")    
    return best_mult

def partition(grid_sizes, data_np):
    n_samples, n_dims = data_np.shape
    # Compute grid cell for each point
    grid_indices = np.floor(data_np / grid_sizes).astype(np.int64)   
    hypercube = {}  # key: grid index tuple, value: list of point indices
    for idx in range(n_samples):
        # Convert grid index to tuple (hashable)
        grid_key = tuple(grid_indices[idx]) 
        if grid_key not in hypercube:
            hypercube[grid_key] = []
        hypercube[grid_key].append(idx) 
    return hypercube


def sample_hypergrid(hypercubes, data_np, metric=euclidean, sampler=None, **sampler_kwargs):
    # Default sampler
    if sampler is None:
        sampler = first; sampler_kwargs={}
    # Apply selection strategy to each hypercube
    selected_indices = []
    for indices in hypercubes.values():
        selected_idx = sampler(indices, data_np, metric, **sampler_kwargs)
        selected_indices.append(selected_idx) 
    # Sort indices to maintain some order (optional)
    selected_indices = sorted(selected_indices)
    return selected_indices

def verify_dataset(selected_dataset, selected_indices, original_dataset):
        logging.info("\n--- Verification ---")
        for i, orig_idx in enumerate(selected_indices[:5]):
            selected_point = selected_dataset[i][0]
            original_point = original_dataset[orig_idx][0]
            assert torch.allclose(selected_point, original_point), f"Mismatch at index {i}"    
        logging.info("Verification passed: Selected data matches original data at preserved indices")
    

def compute_scales(data_np): 
    n_samples = data_np.shape[0]
    logging.info(f"Original dataset size: {n_samples} samples ({data_np.shape[1]} dimensions each).")
    logging.info(f"Computing scales for target ratios: {ratios}")
    base_grid_sizes = compute_base_grid_sizes(data_np) 
    s = [compute_best_mult(base_grid_sizes, data_np, ratio) for ratio in ratios]
    logging.info(f"Done. Computed grid scaling : {s}")
    return s


def make_coresets(data_np, scales):
    n_samples = data_np.shape[0]
    logging.info(f"Original dataset size: {n_samples} samples ({data_np.shape[1]} dimensions each).")
    base_grid_sizes = compute_base_grid_sizes(data_np)    
    #selected_indexes = {}#defaultdict(tree) #[[[[] for _ in range(len(ratios))]for _ in range(len(samplers))]for k in range(len(metrics))]
    selected_indexes = tree()
    #-----------------------------
    # PROCESSSING CHAIN START : 
    # compute and save compressed datasets (coresets)
    #-----------------------------
    min_val = base_grid_sizes.min().item()
    max_val = base_grid_sizes.max().item()
    logging.info(f"{[min_val * s for s in scales]} ***************")
    logging.info(f"{[max_val * s for s in scales]} ***************")
    
    for i, r in enumerate(ratios):          
        #-----------------------------
        # 1. partition the dataset space into a grid of hypercubes 
        # each mapping to a list data points (indexes in original dataset) 
        #-----------------------------
        logging.info(f"Partitioning: --- Applying Target Compression Ratio: {r:.0%} ---")   
        grid_sizes = base_grid_sizes * scales[i] #compute_grid_sizes(features_dataset, target_ratio=r)
        count = count_hypercubes(grid_sizes, data_np)           
        logging.info(f"Partitioning dataset into {count} hypercubes.")
        logging.info(f"  Grid scaling={scales[i]:.4f} selected={count}/{n_samples} ({count / n_samples:.2%})")   
        logging.info(f"  Average grid size: {np.mean(grid_sizes):.8f}, Min: {np.min(grid_sizes):.8f}, Max: {np.max(grid_sizes):.8f}")           
        hypercubes = partition(grid_sizes, data_np)
        # (optional) plot hypercubes distribution
        if log_level == logging.DEBUG  : 
            points = [len(list) for list in hypercubes.values()]
            plot_hystogram(points, bins=100, title=f"Hypercubes data points distribution with {r:.0%} compression ratio")
        #-----------------------------
        # 2. sample at most one point from each 
        # hypercube using a variety of metrics and samplers 
        #-----------------------------
        for m in metrics: 
            for s in samplers:
                logging.info(f"--- Sampling using {m} metrics and {s} sampler at {r:.0%} ratio ---")               
                metric = Metrics[m]; sampler, kwargs = Samplers[s]
                start = datetime.datetime.now()
                selected_indexes[r][m][s] = sample_hypergrid(hypercubes, metric, sampler, **kwargs)        
                elapsed = datetime.datetime.now() - start
                logging.info(f"{s:12} w/ {m} \t- Selected: {len(selected_indexes[r][m][s]):4} points, "
                        f"First 5 indices: {selected_indexes[r][m][s][:5]} "
                        f" in {elapsed.total_seconds():.2f} seconds.")              
                #(optional) Verification
                #if log_level == logging.DEBUG : # verify last selection
                #    verify_dataset(data_np, selected_indexes, train_features) 
    return selected_indexes                     
    


