import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import RMS_NORM_EPS


class RMSNorm(nn.Module):
    """Standard RMSNorm as used by Qwen3_5MoeRMSNorm.
    Weight initialized to zeros; effective scale = (1 + weight).
    """
    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).to(dtype)


class RMSNormGated(nn.Module):
    """RMSNorm with a silu gate, as used for GatedDeltaNet output.
    Weight initialized to ones; applies: weight * norm(x) * silu(gate).
    """
    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = (self.weight * normed.to(dtype))
        return (normed * F.silu(gate.float()).to(dtype))
