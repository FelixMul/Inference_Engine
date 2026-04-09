import torch
import torch.nn as nn
from .config import RMS_NORM_EPS


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(dtype) * self.weight
