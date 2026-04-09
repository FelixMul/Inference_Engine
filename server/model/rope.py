import torch
from .config import ROPE_THETA, ROTARY_DIMS


def build_rope_freqs(max_seq_len: int, device: torch.device) -> torch.Tensor:
    """Precompute inverse frequencies for RoPE."""
    half = ROTARY_DIMS // 2
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, half, device=device).float() / half))
    return inv_freq  # [half]


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to x.
    x: [B, num_heads, T, head_dim]
    position_ids: [B, T]
    inv_freq: [half]
    Only rotates the first ROTARY_DIMS dimensions; rest are unchanged.
    """
    B, H, T, D = x.shape
    half = ROTARY_DIMS // 2

    # [B, T, half]
    freqs = torch.outer(position_ids.float().reshape(-1), inv_freq).reshape(B, T, half)
    # [B, T, ROTARY_DIMS]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(1)  # [B, 1, T, ROTARY_DIMS]
    sin = emb.sin().unsqueeze(1)  # [B, 1, T, ROTARY_DIMS]

    x_rot = x[..., :ROTARY_DIMS]
    x_pass = x[..., ROTARY_DIMS:]

    # rotate_half
    half2 = ROTARY_DIMS // 2
    x1 = x_rot[..., :half2]
    x2 = x_rot[..., half2:]
    rotated = torch.cat([-x2, x1], dim=-1)

    x_rot_out = x_rot * cos + rotated * sin
    return torch.cat([x_rot_out, x_pass], dim=-1)
