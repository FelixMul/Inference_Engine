"""
Full GQA attention with RoPE and output gate.
Used on layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39.

Key details from source:
- q_proj outputs [B, T, num_heads, head_dim*2], split per-head into [q, gate]
- Gate activation: sigmoid (not silu)
- q_norm and k_norm use RMSNorm with (1+weight) formula
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM
from .rope import apply_rope
from .norm import RMSNorm


class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM * 2, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)

    def forward(
        self,
        x: torch.Tensor,               # [B, T, H]
        position_ids: torch.Tensor,    # [B, T]
        inv_freq: torch.Tensor,
        kv_cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, T, _ = x.shape

        # q_proj → [B, T, num_heads, head_dim*2] → split per-head into q and gate
        qg = self.q_proj(x).view(B, T, NUM_Q_HEADS, HEAD_DIM * 2)
        q, gate = qg.chunk(2, dim=-1)           # each [B, T, num_heads, head_dim]
        gate = gate.reshape(B, T, NUM_Q_HEADS * HEAD_DIM)  # [B, T, 4096]

        k = self.k_proj(x).view(B, T, NUM_KV_HEADS, HEAD_DIM)
        v = self.v_proj(x).view(B, T, NUM_KV_HEADS, HEAD_DIM)

        # Per-head RMSNorm
        q = self.q_norm(q)   # [B, T, num_heads, head_dim]
        k = self.k_norm(k)

        # Transpose to [B, heads, T, head_dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        q = apply_rope(q, position_ids, inv_freq)
        k = apply_rope(k, position_ids, inv_freq)

        # KV cache
        if kv_cache is not None and "k" in kv_cache:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        new_cache = {"k": k, "v": v}

        # GQA: expand k, v
        groups = NUM_Q_HEADS // NUM_KV_HEADS
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        # Scaled dot-product attention
        scale = HEAD_DIM ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask (prefill only)
        if T > 1:
            T_total = k.shape[2]
            mask = torch.full((T, T_total), float("-inf"), device=x.device, dtype=x.dtype)
            mask = mask.triu(T_total - T + 1)
            attn = attn + mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        out = torch.matmul(attn, v)   # [B, num_heads, T, head_dim]

        out = out.transpose(1, 2).reshape(B, T, NUM_Q_HEADS * HEAD_DIM)

        # Output gate: sigmoid (from source: attn_output * torch.sigmoid(gate))
        out = out * torch.sigmoid(gate)

        return self.o_proj(out), new_cache
