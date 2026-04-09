"""
GatedDeltaNet linear attention (sequential implementation, correct for all sequence lengths).
Used on all layers except 3, 7, 11, 15, 19, 23, 27, 31, 35, 39.

Key weight shapes:
  in_proj_qkv.weight   [8192, 2048]  -> Q[2048] K[2048] V[4096]
  in_proj_z.weight     [4096, 2048]  -> gate z
  in_proj_b.weight     [32,   2048]  -> beta (per value-head)
  in_proj_a.weight     [32,   2048]  -> dt pre-activation
  conv1d.weight        [8192,  1, 4] -> depthwise causal conv on QKV
  dt_bias              [32]
  A_log                [32]
  norm.weight          [128]
  out_proj.weight      [2048, 4096]

Architecture:
  16 key heads,  key_head_dim = 128  -> total Q = K = 2048
  32 value heads, val_head_dim = 128 -> total V = 4096
  State S: [B, 16, 128, 256]  (each key-head maps to 2 value-heads = 256 val dims)
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
from .norm import RMSNorm

_Q_DIM = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM    # 2048
_K_DIM = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM    # 2048
_V_DIM = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM  # 4096
_QKV_DIM = _Q_DIM + _K_DIM + _V_DIM                   # 8192

_VAL_PER_KEY = LINEAR_NUM_VALUE_HEADS // LINEAR_NUM_KEY_HEADS  # 2
_STATE_VAL_DIM = _VAL_PER_KEY * LINEAR_VALUE_HEAD_DIM          # 256


class GatedDeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(HIDDEN_SIZE, _QKV_DIM, bias=False)
        self.in_proj_z = nn.Linear(HIDDEN_SIZE, _V_DIM, bias=False)
        self.in_proj_b = nn.Linear(HIDDEN_SIZE, LINEAR_NUM_VALUE_HEADS, bias=False)
        self.in_proj_a = nn.Linear(HIDDEN_SIZE, LINEAR_NUM_VALUE_HEADS, bias=False)
        # Depthwise causal conv: weight [QKV, 1, kernel]
        # We handle padding manually so padding=0 here
        self.conv1d_weight = nn.Parameter(torch.empty(_QKV_DIM, 1, LINEAR_CONV_KERNEL))
        self.dt_bias = nn.Parameter(torch.zeros(LINEAR_NUM_VALUE_HEADS))
        self.A_log = nn.Parameter(torch.zeros(LINEAR_NUM_VALUE_HEADS))
        self.norm = RMSNorm(LINEAR_VALUE_HEAD_DIM)
        self.out_proj = nn.Linear(_V_DIM, HIDDEN_SIZE, bias=False)

    def forward(
        self,
        x: torch.Tensor,                       # [B, T, H]
        state: torch.Tensor | None = None,     # [B, 16, 128, 256]
        conv_buf: torch.Tensor | None = None,  # [B, QKV_DIM, kernel-1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        k = LINEAR_CONV_KERNEL

        # Project to QKV, gate, beta, dt
        qkv = self.in_proj_qkv(x)    # [B, T, 8192]
        z = self.in_proj_z(x)         # [B, T, 4096]
        beta_raw = self.in_proj_b(x)  # [B, T, 32]
        dt_raw = self.in_proj_a(x)    # [B, T, 32]

        # Causal depthwise conv1d on QKV
        # Manually left-pad so each output[t] only sees inputs[t-k+1..t]
        qkv_t = qkv.transpose(1, 2)  # [B, 8192, T]
        if conv_buf is not None:
            qkv_padded = torch.cat([conv_buf, qkv_t], dim=2)  # [B, 8192, k-1+T]
        else:
            pad = qkv_t.new_zeros(B, _QKV_DIM, k - 1)
            qkv_padded = torch.cat([pad, qkv_t], dim=2)       # [B, 8192, k-1+T]

        # Depthwise conv with no padding (already padded above)
        qkv_conv = F.conv1d(qkv_padded, self.conv1d_weight, groups=_QKV_DIM)  # [B, 8192, T]

        # Save last k-1 tokens from padded input as buffer for next call
        new_conv_buf = qkv_padded[..., -(k - 1):].clone()
        qkv = qkv_conv.transpose(1, 2)  # [B, T, 8192]

        # Split Q, K, V and normalize
        q = qkv[..., :_Q_DIM].view(B, T, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM)
        kk = qkv[..., _Q_DIM:_Q_DIM + _K_DIM].view(B, T, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM)
        v = qkv[..., _Q_DIM + _K_DIM:].view(B, T, LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)

        q = F.normalize(q, dim=-1)   # unit vectors
        kk = F.normalize(kk, dim=-1)  # unit vectors

        # Beta (sigmoid) and decay per value-head
        beta = torch.sigmoid(beta_raw)                          # [B, T, 32]
        dt = F.softplus(dt_raw + self.dt_bias)                 # [B, T, 32]
        decay = torch.exp(-dt * torch.exp(self.A_log))         # [B, T, 32]

        # Initialize state
        if state is None:
            state = x.new_zeros(B, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, _STATE_VAL_DIM)

        # Sequential delta-rule update
        outputs = []
        for t in range(T):
            q_t = q[:, t]    # [B, 16, 128]
            k_t = kk[:, t]   # [B, 16, 128]
            v_t = v[:, t]    # [B, 32, 128]
            b_t = beta[:, t] # [B, 32]
            d_t = decay[:, t] # [B, 32]

            # Reshape to key-head grouping: [B, 16, 2, 128]
            v_t_r = v_t.view(B, LINEAR_NUM_KEY_HEADS, _VAL_PER_KEY, LINEAR_VALUE_HEAD_DIM)
            b_t_r = b_t.view(B, LINEAR_NUM_KEY_HEADS, _VAL_PER_KEY)
            d_t_r = d_t.view(B, LINEAR_NUM_KEY_HEADS, _VAL_PER_KEY)

            # Current state readout for k_t: S_{t-1} k_t -> [B, 16, 256]
            Sk_t = torch.einsum("bhi,bhij->bhj", k_t, state)
            Sk_t_r = Sk_t.view(B, LINEAR_NUM_KEY_HEADS, _VAL_PER_KEY, LINEAR_VALUE_HEAD_DIM)

            # Delta: what we want minus what state currently maps k to
            delta_v = v_t_r - Sk_t_r  # [B, 16, 2, 128]

            # Outer product: k_t ⊗ (beta * delta_v) -> [B, 16, 128, 2, 128] -> [B, 16, 128, 256]
            update = torch.einsum(
                "bhi,bhpj->bhipj",
                k_t,
                b_t_r.unsqueeze(-1) * delta_v,
            ).reshape(B, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, _STATE_VAL_DIM)

            # Decay state and add update
            # d_t_r [B, 16, 2] -> expand to [B, 16, 128, 256]
            d_full = d_t_r.unsqueeze(2).unsqueeze(-1).expand(
                -1, -1, LINEAR_KEY_HEAD_DIM, -1, LINEAR_VALUE_HEAD_DIM
            ).reshape(B, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, _STATE_VAL_DIM)

            state = state * d_full + update

            # Read output: S_t q_t -> [B, 16, 256]
            o_t = torch.einsum("bhi,bhij->bhj", q_t, state)
            outputs.append(o_t)

        # [B, T, 16, 256] -> [B, T, 4096]
        out = torch.stack(outputs, dim=1).reshape(B, T, _V_DIM)

        # RMSNorm per value head
        out = out.view(B, T, LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
        out = self.norm(out)
        out = out.reshape(B, T, _V_DIM)

        # Gate and output projection
        out = out * F.silu(z)
        return self.out_proj(out), state, new_conv_buf
