"""
GatedDeltaNet linear attention — sequential implementation.

Key fixes from source inspection:
- silu applied after causal conv1d
- Q and K expanded from 16 → 32 heads (repeat_interleave) to match V heads
- State shape: [B, 32, 128, 128]
- Output gated norm: RMSNormGated(norm(x), z)
- Decay: g = -exp(A_log) * softplus(a + dt_bias)  [log-space, negative]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import (
    HIDDEN_SIZE,
    LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM,
    LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM,
    LINEAR_CONV_KERNEL,
)
from .norm import RMSNormGated

_K_DIM = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM    # 16*128 = 2048
_V_DIM = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM  # 32*128 = 4096
_CONV_DIM = _K_DIM * 2 + _V_DIM                        # 8192  (Q+K+V)
_EXPAND = LINEAR_NUM_VALUE_HEADS // LINEAR_NUM_KEY_HEADS  # 2

# After expansion, both Q and K have 32 heads
_EXP_HEADS = LINEAR_NUM_VALUE_HEADS   # 32
_HEAD_DIM = LINEAR_KEY_HEAD_DIM       # 128


class GatedDeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(HIDDEN_SIZE, _CONV_DIM, bias=False)
        self.in_proj_z = nn.Linear(HIDDEN_SIZE, _V_DIM, bias=False)
        self.in_proj_b = nn.Linear(HIDDEN_SIZE, LINEAR_NUM_VALUE_HEADS, bias=False)
        self.in_proj_a = nn.Linear(HIDDEN_SIZE, LINEAR_NUM_VALUE_HEADS, bias=False)
        # Depthwise causal conv — we handle left-padding manually
        self.conv1d_weight = nn.Parameter(torch.empty(_CONV_DIM, 1, LINEAR_CONV_KERNEL))
        self.dt_bias = nn.Parameter(torch.ones(LINEAR_NUM_VALUE_HEADS))
        self.A_log = nn.Parameter(torch.zeros(LINEAR_NUM_VALUE_HEADS))
        self.norm = RMSNormGated(_HEAD_DIM)
        self.out_proj = nn.Linear(_V_DIM, HIDDEN_SIZE, bias=False)

    def forward(
        self,
        x: torch.Tensor,                       # [B, T, H]
        state: torch.Tensor | None = None,     # [B, 32, 128, 128]
        conv_buf: torch.Tensor | None = None,  # [B, CONV_DIM, kernel-1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        k = LINEAR_CONV_KERNEL

        # Projections
        qkv = self.in_proj_qkv(x)    # [B, T, 8192]
        z = self.in_proj_z(x)         # [B, T, 4096]
        b_raw = self.in_proj_b(x)     # [B, T, 32]
        a_raw = self.in_proj_a(x)     # [B, T, 32]

        # Causal depthwise conv1d with manual left-padding + silu
        qkv_t = qkv.transpose(1, 2)   # [B, 8192, T]
        if conv_buf is not None:
            qkv_padded = torch.cat([conv_buf, qkv_t], dim=2)
        else:
            pad = qkv_t.new_zeros(B, _CONV_DIM, k - 1)
            qkv_padded = torch.cat([pad, qkv_t], dim=2)

        qkv_conv = F.conv1d(qkv_padded, self.conv1d_weight, groups=_CONV_DIM)  # [B, 8192, T]
        qkv_conv = F.silu(qkv_conv)   # ← activation applied after conv (from source)

        # Save conv buffer for next decode step
        new_conv_buf = qkv_padded[..., -(k - 1):].clone()

        qkv = qkv_conv.transpose(1, 2)  # [B, T, 8192]

        # Split Q, K, V
        q = qkv[..., :_K_DIM].view(B, T, LINEAR_NUM_KEY_HEADS, _HEAD_DIM)
        kk = qkv[..., _K_DIM:_K_DIM * 2].view(B, T, LINEAR_NUM_KEY_HEADS, _HEAD_DIM)
        v = qkv[..., _K_DIM * 2:].view(B, T, LINEAR_NUM_VALUE_HEADS, _HEAD_DIM)

        # Expand Q and K from 16 → 32 heads to match V
        q = q.repeat_interleave(_EXPAND, dim=2)   # [B, T, 32, 128]
        kk = kk.repeat_interleave(_EXPAND, dim=2)  # [B, T, 32, 128]

        # L2-normalize Q and K (use_qk_l2norm_in_kernel=True)
        q = F.normalize(q.float(), dim=-1).to(x.dtype)
        kk = F.normalize(kk.float(), dim=-1).to(x.dtype)

        # Beta and decay (computed in float32 for precision)
        beta = torch.sigmoid(b_raw.float()).to(x.dtype)   # [B, T, 32]
        # g = -exp(A_log) * softplus(a + dt_bias)  [log-space decay, negative]
        g = (-self.A_log.float().exp() *
             F.softplus(a_raw.float() + self.dt_bias.float()))  # [B, T, 32]
        decay = g.exp().to(x.dtype)   # actual decay in [0, 1]

        # Initialize recurrent state: [B, 32, 128, 128]
        if state is None:
            state = x.new_zeros(B, _EXP_HEADS, _HEAD_DIM, _HEAD_DIM)

        # Sequential delta-rule update
        outputs = []
        for t in range(T):
            q_t  = q[:, t]      # [B, 32, 128]
            k_t  = kk[:, t]     # [B, 32, 128]
            v_t  = v[:, t]      # [B, 32, 128]
            b_t  = beta[:, t]   # [B, 32]
            d_t  = decay[:, t]  # [B, 32]

            # 1. Decay state first
            state = state * d_t.unsqueeze(-1).unsqueeze(-1)

            # 2. Compute delta based on the ALREADY-DECAYED state
            Sk = torch.einsum("bhi,bhij->bhj", k_t, state)  # [B, 32, 128]
            delta = v_t - Sk

            # 3. Outer product update: k_t ⊗ (beta * delta) → [B, 32, 128, 128]
            update = torch.einsum(
                "bhi,bhj->bhij",
                k_t,
                b_t.unsqueeze(-1) * delta,
            )
            state = state + update

            # Read output: S q_t → [B, 32, 128]
            o_t = torch.einsum("bhi,bhij->bhj", q_t, state)
            outputs.append(o_t)

        # [B, T, 32, 128]
        out = torch.stack(outputs, dim=1)

        # RMSNormGated: norm(out) * weight * silu(z)
        # Flatten to [B*T*32, 128] for norm, z same shape
        out_flat = out.reshape(B * T * _EXP_HEADS, _HEAD_DIM)
        z_flat = z.reshape(B, T, _EXP_HEADS, _HEAD_DIM).reshape(B * T * _EXP_HEADS, _HEAD_DIM)
        out_flat = self.norm(out_flat, z_flat)

        out = out_flat.reshape(B, T, _V_DIM)
        return self.out_proj(out), state, new_conv_buf
