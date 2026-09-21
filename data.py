from dataclasses import dataclass
from typing import Any, Optional
from torch import Tensor

@dataclass
class DoublingDimResult:
    estimate: float
    config: dict[str, Any]
    per_scale_max: Optional[Tensor] = None
    radii: Optional[Tensor] = None
    sampled_centers: Optional[Tensor] = None
    local_counts: Optional[Tensor] = None

@dataclass
class ddConfig:  #defaults to the fastest of my working algorithm
    name : str = "doubling_dim_linf_mps"
    min_centers : int = 128
    max_centers : int = 4096
    nb_cap : int = 2024
    max_radii : int = 192
    #-------------------#
    fps_block : int = 4096
    cdist_qblock : int = 64       # small query blocks (centers to points)
    cdist_rblock : int = 2048      # larger result blocks (safe for MPS)
    cover_block : int = 1024
    radii : Optional[list[int]] = None
    anchor_samples : int = 256
    radius_quantiles : tuple[float,...] = (0.50, 0.70, 0.85, 0.93)
    seed = 0 


