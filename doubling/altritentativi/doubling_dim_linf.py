import math
import torch
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any

@dataclass
class DoublingDimResult:
    estimate: float
    per_scale_max: torch.Tensor
    radii: torch.Tensor
    sampled_centers: torch.Tensor
    local_counts: torch.Tensor
    config: Dict[str, Any]


def _linf_cdist(a: torch.Tensor, b: torch.Tensor, block: int = 1024) -> torch.Tensor:
    device = a.device
    out = torch.empty((a.shape[0], b.shape[0]), device=device, dtype=torch.float32)
    for j in range(0, b.shape[0], block):
        bj = b[j:j+block]
        out[:, j:j+block] = (a[:, None, :] - bj[None, :, :]).abs().amax(dim=-1)
    return out


def _farthest_point_sample(X: torch.Tensor, m: int, seed: int = 0, block: int = 2048) -> torch.Tensor:
    n = X.shape[0]
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    first = int(torch.randint(0, n, (1,), generator=g).item())
    chosen = [first]
    min_dist = _linf_cdist(X, X[first:first+1], block=block).squeeze(1)
    for _ in range(1, min(m, n)):
        nxt = int(torch.argmax(min_dist).item())
        chosen.append(nxt)
        d = _linf_cdist(X, X[nxt:nxt+1], block=block).squeeze(1)
        min_dist = torch.minimum(min_dist, d)
    return torch.tensor(chosen, device=X.device, dtype=torch.long)


def estimate_doubling_dimension_linf(
    X: torch.Tensor,
    center_samples: int = 256,
    anchor_samples: int = 512,
    radii: Optional[Sequence[float]] = None,
    radius_quantiles: Sequence[float] = (0.5, 0.7, 0.85, 0.93),
    neighborhood_cap: int = 2048,
    block: int = 1024,
    seed: int = 0,
    device: Optional[str] = None,
) -> DoublingDimResult:
    assert X.ndim == 2
    X = X.detach()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = X.to(device=device, dtype=torch.float32, non_blocking=True)
    n = X.shape[0]

    center_idx = _farthest_point_sample(X, min(center_samples, n), seed=seed, block=block)
    centers = X[center_idx]

    g = torch.Generator(device='cpu')
    g.manual_seed(seed + 1)
    perm = torch.randperm(n, generator=g)[:min(anchor_samples, n)]
    anchor_idx = perm.to(device)
    anchors = X[anchor_idx]

    center_anchor_d = _linf_cdist(centers, anchors, block=block)
    if radii is None:
        radii_t = torch.quantile(center_anchor_d.flatten(), torch.tensor(radius_quantiles, device=device))
        radii_t = torch.unique(radii_t.clamp_min(1e-12))
    else:
        radii_t = torch.tensor(list(radii), device=device, dtype=torch.float32)

    local_counts = torch.zeros((centers.shape[0], radii_t.shape[0]), device=device, dtype=torch.float32)

    for ci, c in enumerate(centers):
        d_all = _linf_cdist(c[None, :], X, block=block).squeeze(0)
        for ri, r in enumerate(radii_t):
            idx = torch.nonzero(d_all <= r, as_tuple=False).squeeze(1)
            m = idx.numel()
            if m <= 1:
                local_counts[ci, ri] = 1.0
                continue
            if m > neighborhood_cap:
                sub = idx[torch.randperm(m, device=device)[:neighborhood_cap]]
            else:
                sub = idx
            Y = X[sub]
            uncovered = torch.ones(Y.shape[0], dtype=torch.bool, device=device)
            count = 0
            half_r = r * 0.5
            while uncovered.any():
                active = torch.nonzero(uncovered, as_tuple=False).squeeze(1)
                piv = active[0]
                d = _linf_cdist(Y[piv:piv+1], Y[active], block=block).squeeze(0)
                covered_active = active[d <= half_r]
                uncovered[covered_active] = False
                count += 1
            local_counts[ci, ri] = float(count)

    per_scale_max = local_counts.max(dim=0).values
    estimate = torch.log2(per_scale_max.clamp_min(1.0)).max().item()

    return DoublingDimResult(
        estimate=estimate,
        per_scale_max=per_scale_max.detach().cpu(),
        radii=radii_t.detach().cpu(),
        sampled_centers=center_idx.detach().cpu(),
        local_counts=local_counts.detach().cpu(),
        config={
            'center_samples': int(center_samples),
            'anchor_samples': int(anchor_samples),
            'radius_quantiles': list(radius_quantiles),
            'neighborhood_cap': int(neighborhood_cap),
            'block': int(block),
            'seed': int(seed),
            'device': device,
        }
    )


def estimate_from_tensordataset(dataset, tensor_index: int = 0, **kwargs):
    X = dataset.tensors[tensor_index]
    return estimate_doubling_dimension_linf(X, **kwargs)


