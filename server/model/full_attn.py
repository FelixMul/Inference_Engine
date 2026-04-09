"""
Full (standard) GQA attention with RoPE and output gate.
Used on layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39.

Weights per layer:
  self_attn.q_proj.weight  [8192, 2048]  -> q [4096] + gate [4096]
  self_attn.k_proj.weight  [512,  2048]  -> k [512]  (2 heads * 256)
  self_attn.v_proj.weight  [512,  2048]  -> v [512]
  self_attn.o_proj.weight  [2048, 4096]
  self_attn.q_norm.weight  [256]
  self_attn.k_norm.weight  [256]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import (
    HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIMS,
)
from .rope import apply_rope
from .norm import RMSNorm


class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM * 2, bias=False)  # *2 for gate
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(
        self,
        x: torch.Tensor,               # [B, T, H]
        position_ids: torch.Tensor,    # [B, T]
        inv_freq: torch.Tensor,        # [ROTARY_DIMS//2]
        kv_cache: dict | None = None,  # {"k": [B, KV_H, T_past, D], "v": ...}
    ) -> tuple[torch.Tensor, dict]:
        B, T, _ = x.shape

        qg = self.q_proj(x)  # [B, T, 2 * Q_H * D]
        q, gate = qg.chunk(2, dim=-1)  # each [B, T, Q_H * D]
        k = self.k_proj(x)  # [B, T, KV_H * D]
        v = self.v_proj(x)  # [B, T, KV_H * D]

        # reshape to [B, heads, T, D]
        q = q.view(B, T, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # per-head norms
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rope(q, position_ids, inv_freq)
        k = apply_rope(k, position_ids, inv_freq)

        # KV cache
        if kv_cache is not None and "k" in kv_cache:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        new_cache = {"k": k, "v": v}

        # GQA: repeat k/v for each group
        groups = NUM_Q_HEADS // NUM_KV_HEADS
        k = k.repeat_interleave(groups, dim=1)  # [B, Q_H, T_total, D]
        v = v.repeat_interleave(groups, dim=1)

        # Scaled dot-product attention
        scale = math.sqrt(HEAD_DIM)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, Q_H, T, T_total]

        # Causal mask (only needed during prefill)
        if T > 1:
            T_total = k.shape[2]
            causal = torch.full((T, T_total), float("-inf"), device=x.device, dtype=x.dtype)
            causal = causal.triu(T_total - T + 1)
            attn = attn + causal.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, Q_H, T, D]

        # reshape back
        out = out.transpose(1, 2).reshape(B, T, NUM_Q_HEADS * HEAD_DIM)

        # output gate (silu)
        gate = gate.reshape(B, T, NUM_Q_HEADS * HEAD_DIM)
        out = out * F.silu(gate)

        return self.o_proj(out), new_cache
