"""
GatedDeltaNet linear attention (sequential implementation, correct for all sequence lengths).
Used on all layers except 3, 7, 11, 15, 19, 23, 27, 31, 35, 39.

Key weight shapes:
  in_proj_qkv.weight   [8192, 2048]  -> Q[2048] K[2048] V[4096]
  in_proj_z.weight     [4096, 2048]  -> gate z
  in_proj_b.weight     [32,   2048]  -> beta (per value-head)
  in_proj_a.weight     [32,   2048]  -> dt pre-activation
  conv1d.weight        [8192,  1, 4] -> short causal conv on QKV
  dt_bias              [32]
  A_log                [32]
  norm.weight          [128]
  out_proj.weight      [2048, 4096]

Architecture:
  16 key heads,  key_head_dim = 128  -> total K = 2048
  32 value heads, val_head_dim = 128 -> total V = 4096
  State S: [B, 16, 128, 256]  (each key-head maps keys to 2 value-heads = 256 val dims)
  beta, decay: [32] per value-head
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

_Q_DIM = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM   # 2048
_K_DIM = LINEAR_NUM_KEY_HEADS * LINEAR_KEY_HEAD_DIM   # 2048
_V_DIM = LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM  # 4096
_QKV_DIM = _Q_DIM + _K_DIM + _V_DIM                   # 8192

# value heads paired with key heads: 2 val-heads per key-head
_VAL_PER_KEY = LINEAR_NUM_VALUE_HEADS // LINEAR_NUM_KEY_HEADS  # 2
_STATE_VAL_DIM = _VAL_PER_KEY * LINEAR_VALUE_HEAD_DIM          # 256


class GatedDeltaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(HIDDEN_SIZE, _QKV_DIM, bias=False)
        self.in_proj_z = nn.Linear(HIDDEN_SIZE, _V_DIM, bias=False)
        self.in_proj_b = nn.Linear(HIDDEN_SIZE, LINEAR_NUM_VALUE_HEADS, bias=False)
        self.in_proj_a = nn.Linear(HIDDEN_SIZE, LINEAR_NUM_VALUE_HEADS, bias=False)
        self.conv1d = nn.Conv1d(_QKV_DIM, _QKV_DIM, kernel_size=LINEAR_CONV_KERNEL,
                                groups=_QKV_DIM, padding=LINEAR_CONV_KERNEL - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(LINEAR_NUM_VALUE_HEADS))
        self.A_log = nn.Parameter(torch.zeros(LINEAR_NUM_VALUE_HEADS))
        self.norm = RMSNorm(LINEAR_KEY_HEAD_DIM)
        self.out_proj = nn.Linear(_V_DIM, HIDDEN_SIZE, bias=False)

    def forward(
        self,
        x: torch.Tensor,       # [B, T, H]
        state: torch.Tensor | None = None,  # [B, num_key_heads, key_dim, state_val_dim]
        conv_buf: torch.Tensor | None = None,  # [B, QKV_DIM, kernel-1] causal conv state
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        # Project to QKV and gate
        qkv = self.in_proj_qkv(x)   # [B, T, 8192]
        z = self.in_proj_z(x)        # [B, T, 4096]
        beta_raw = self.in_proj_b(x) # [B, T, 32]
        dt_raw = self.in_proj_a(x)   # [B, T, 32]

        # Short causal conv on QKV (operates over sequence dim)
        # conv1d expects [B, C, T]
        qkv_t = qkv.transpose(1, 2)  # [B, 8192, T]
        k = LINEAR_CONV_KERNEL
        if conv_buf is not None:
            # prepend buffered context from previous tokens
            qkv_t = torch.cat([conv_buf, qkv_t], dim=2)
        qkv_conv = self.conv1d(qkv_t)[..., :qkv_t.shape[2] - (k - 1) + (k - 1)]
        # causal output: trim to T tokens
        if conv_buf is not None:
            new_conv_buf = qkv_t[..., -(k - 1):]
            qkv_conv = qkv_conv[..., -(T):]
        else:
            new_conv_buf = qkv_t[..., -(k - 1):]
            qkv_conv = qkv_conv[..., :T]
        qkv = qkv_conv.transpose(1, 2)  # [B, T, 8192]

        # Split Q, K, V
        q = qkv[..., :_Q_DIM].view(B, T, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM)
        kk = qkv[..., _Q_DIM:_Q_DIM + _K_DIM].view(B, T, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM)
        v = qkv[..., _Q_DIM + _K_DIM:].view(B, T, LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)

        # Normalize K
        kk = F.normalize(kk, dim=-1)  # [B, T, 16, 128]

        # Beta (sigmoid) and decay per value-head
        beta = torch.sigmoid(beta_raw)  # [B, T, 32]
        dt = F.softplus(dt_raw + self.dt_bias)  # [B, T, 32]
        decay = torch.exp(-dt * torch.exp(self.A_log))  # [B, T, 32]

        # Initialize state if needed
        if state is None:
            state = torch.zeros(
                B, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, _STATE_VAL_DIM,
                device=x.device, dtype=x.dtype,
            )

        # Sequential delta-rule update over time
        outputs = []
        for t in range(T):
            q_t = q[:, t]    # [B, 16, 128]
            k_t = kk[:, t]   # [B, 16, 128]
            v_t = v[:, t]    # [B, 32, 128]
            b_t = beta[:, t] # [B, 32]
            d_t = decay[:, t]  # [B, 32]

            # reshape v and b to [B, 16, 2, 128] and [B, 16, 2]
            v_t_r = v_t.view(B, LINEAR_NUM_KEY_HEADS, _VAL_PER_KEY, LINEAR_VALUE_HEAD_DIM)
            b_t_r = b_t.view(B, LINEAR_NUM_KEY_HEADS, _VAL_PER_KEY)  # [B, 16, 2]
            d_t_r = d_t.view(B, LINEAR_NUM_KEY_HEADS, _VAL_PER_KEY)  # [B, 16, 2]

            # current state read for k_t: [B, 16, 128] @ [B, 16, 128, 256] -> [B, 16, 256]
            Sk_t = torch.einsum("bhi,bhij->bhj", k_t, state)  # [B, 16, 256]

            # reshape to [B, 16, 2, 128] for subtraction with v
            Sk_t_r = Sk_t.view(B, LINEAR_NUM_KEY_HEADS, _VAL_PER_KEY, LINEAR_VALUE_HEAD_DIM)

            # delta rule: new_v = v_t - Sk_t (what we want to store)
            delta_v = v_t_r - Sk_t_r  # [B, 16, 2, 128]

            # outer product update: k_t ⊗ (b * delta_v) -> [B, 16, 128, 2, 128] -> [B, 16, 128, 256]
            update = torch.einsum(
                "bhi,bhpj->bhipj",
                k_t,
                b_t_r.unsqueeze(-1) * delta_v,
            ).reshape(B, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, _STATE_VAL_DIM)

            # apply decay per value-head then add update
            d_t_full = d_t_r.unsqueeze(2).expand(-1, -1, LINEAR_KEY_HEAD_DIM, -1)  # [B,16,128,2]
            # interleave decay across the 256 dim: d_t_r -> [B,16,2] -> [B,16,256] broadcast
            d_256 = d_t_r.reshape(B, LINEAR_NUM_KEY_HEADS, 1, _VAL_PER_KEY, 1).expand(
                -1, -1, LINEAR_KEY_HEAD_DIM, -1, LINEAR_VALUE_HEAD_DIM
            ).reshape(B, LINEAR_NUM_KEY_HEADS, LINEAR_KEY_HEAD_DIM, _STATE_VAL_DIM)

            state = state * d_256 + update

            # read output for q_t: [B, 16, 128] @ [B, 16, 128, 256] -> [B, 16, 256]
            o_t = torch.einsum("bhi,bhij->bhj", q_t, state)  # [B, 16, 256]
            outputs.append(o_t)

        # Stack outputs: [B, T, 16, 256]
        out = torch.stack(outputs, dim=1)
        out = out.reshape(B, T, _V_DIM)  # [B, T, 4096]

        # Apply norm (per key-head-dim = 128)
        out = out.view(B, T, LINEAR_NUM_VALUE_HEADS, LINEAR_VALUE_HEAD_DIM)
        out = self.norm(out)
        out = out.reshape(B, T, _V_DIM)

        # Gate (silu)
        out = out * F.silu(z)

        return self.out_proj(out), state, new_conv_buf
