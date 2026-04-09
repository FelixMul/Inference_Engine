"""
Mixture of Experts layer.

Every layer (both full-attn and linear-attn) uses this MoE FFN.

Weights:
  mlp.gate.weight                [256, 2048]   router
  mlp.experts.gate_up_proj       [256, 1024, 2048]  (gate+up packed: 1024 = 2*512)
  mlp.experts.down_proj          [256, 2048, 512]   wait shape is [256, 2048, 512]
  mlp.shared_expert.gate_proj    [512, 2048]
  mlp.shared_expert.up_proj      [512, 2048]
  mlp.shared_expert.down_proj    [2048, 512]
  mlp.shared_expert_gate         [1, 2048]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import (
    HIDDEN_SIZE, NUM_EXPERTS, NUM_EXPERTS_PER_TOK,
    MOE_INTERMEDIATE_SIZE, SHARED_EXPERT_INTERMEDIATE_SIZE,
)


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN_SIZE, NUM_EXPERTS, bias=False)

        # Experts stored as batched tensors: [num_experts, out, in]
        # gate_up_proj: [256, 1024, 2048] means [E, 2*intermediate, hidden]
        self.experts_gate_up = nn.Parameter(
            torch.empty(NUM_EXPERTS, 2 * MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)
        )
        # down_proj shape from checkpoint: [256, 2048, 512] means [E, hidden, intermediate]
        self.experts_down = nn.Parameter(
            torch.empty(NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
        )

        # Shared expert
        self.shared_gate = nn.Linear(HIDDEN_SIZE, SHARED_EXPERT_INTERMEDIATE_SIZE, bias=False)
        self.shared_up = nn.Linear(HIDDEN_SIZE, SHARED_EXPERT_INTERMEDIATE_SIZE, bias=False)
        self.shared_down = nn.Linear(SHARED_EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)
        self.shared_expert_gate = nn.Linear(HIDDEN_SIZE, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        B, T, H = x.shape
        x_flat = x.reshape(B * T, H)  # [N, H]
        N = x_flat.shape[0]

        # Router
        logits = self.gate(x_flat)  # [N, E]
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_ids = torch.topk(scores, NUM_EXPERTS_PER_TOK, dim=-1)  # [N, k]

        # Normalize top-k scores
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        out = torch.zeros(N, H, device=x.device, dtype=x.dtype)

        # Group tokens by expert for efficient batching
        for e in range(NUM_EXPERTS):
            # find tokens routed to expert e
            mask = (topk_ids == e)  # [N, k]
            token_mask = mask.any(dim=-1)  # [N]
            if not token_mask.any():
                continue

            tokens = x_flat[token_mask]  # [n_e, H]
            # gate + up proj
            gu = tokens @ self.experts_gate_up[e].T  # [n_e, 2*interm]
            g, u = gu.chunk(2, dim=-1)
            h = F.silu(g) * u  # [n_e, interm]
            # down proj
            expert_out = h @ self.experts_down[e].T  # [n_e, H]  (down is [H, interm])

            # weight by router score
            tok_scores = scores[token_mask]  # [n_e, E]
            tok_expert_scores = tok_scores[:, e].unsqueeze(-1)  # [n_e, 1]
            out[token_mask] += tok_expert_scores * expert_out

        # Shared expert (always active)
        shared = self.shared_down(
            F.silu(self.shared_gate(x_flat)) * self.shared_up(x_flat)
        )
        shared_weight = torch.sigmoid(self.shared_expert_gate(x_flat))  # [N, 1]
        out = out + shared_weight * shared

        return out.reshape(B, T, H)
