"""
Full Qwen3.5-35B-A3B model with custom forward pass and generation loop.
Weights are loaded from the HuggingFace checkpoint (transformers used only for loading).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import (
    HIDDEN_SIZE, NUM_LAYERS, VOCAB_SIZE, EOS_TOKEN_ID, FULL_ATTN_LAYERS,
)
from .norm import RMSNorm
from .rope import build_rope_freqs
from .full_attn import FullAttention
from .linear_attn import GatedDeltaNet
from .moe import MoE


class TransformerLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full_attn = layer_idx in FULL_ATTN_LAYERS
        self.input_layernorm = RMSNorm(HIDDEN_SIZE)
        self.post_attention_layernorm = RMSNorm(HIDDEN_SIZE)
        if self.is_full_attn:
            self.attn = FullAttention()
        else:
            self.attn = GatedDeltaNet()
        self.mlp = MoE()

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        inv_freq: torch.Tensor,
        kv_cache: dict | None = None,
        lin_state: torch.Tensor | None = None,
        conv_buf: torch.Tensor | None = None,
    ):
        residual = x
        h = self.input_layernorm(x)

        if self.is_full_attn:
            attn_out, new_kv = self.attn(h, position_ids, inv_freq, kv_cache)
            new_lin_state = lin_state
            new_conv_buf = conv_buf
        else:
            attn_out, new_lin_state, new_conv_buf = self.attn(h, lin_state, conv_buf)
            new_kv = kv_cache

        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)

        return x, new_kv, new_lin_state, new_conv_buf


class Qwen35MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = nn.ModuleList([TransformerLayer(i) for i in range(NUM_LAYERS)])
        self.norm = RMSNorm(HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward_step(
        self,
        input_ids: torch.Tensor,   # [B, T]
        position_ids: torch.Tensor,  # [B, T]
        inv_freq: torch.Tensor,
        kv_caches: list[dict | None],
        lin_states: list[torch.Tensor | None],
        conv_bufs: list[torch.Tensor | None],
    ):
        x = self.embed_tokens(input_ids)  # [B, T, H]

        new_kv_caches = []
        new_lin_states = []
        new_conv_bufs = []

        for i, layer in enumerate(self.layers):
            x, new_kv, new_lin, new_conv = layer(
                x, position_ids, inv_freq,
                kv_cache=kv_caches[i],
                lin_state=lin_states[i],
                conv_buf=conv_bufs[i],
            )
            new_kv_caches.append(new_kv)
            new_lin_states.append(new_lin)
            new_conv_bufs.append(new_conv)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, V]
        return logits, new_kv_caches, new_lin_states, new_conv_bufs

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,   # [1, T]
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id: int = EOS_TOKEN_ID,
    ) -> torch.Tensor:
        device = input_ids.device
        inv_freq = build_rope_freqs(input_ids.shape[1] + max_new_tokens, device)

        kv_caches = [None] * NUM_LAYERS
        lin_states = [None] * NUM_LAYERS
        conv_bufs = [None] * NUM_LAYERS

        # Prefill
        T = input_ids.shape[1]
        position_ids = torch.arange(T, device=device).unsqueeze(0)
        logits, kv_caches, lin_states, conv_bufs = self.forward_step(
            input_ids, position_ids, inv_freq, kv_caches, lin_states, conv_bufs
        )

        generated = []
        next_token = self._sample(logits[:, -1, :], temperature, top_p)
        generated.append(next_token)

        # Decode
        for step in range(max_new_tokens - 1):
            pos = T + step + 1
            position_ids = torch.tensor([[pos]], device=device)
            logits, kv_caches, lin_states, conv_bufs = self.forward_step(
                next_token.unsqueeze(0), position_ids, inv_freq,
                kv_caches, lin_states, conv_bufs
            )
            next_token = self._sample(logits[:, -1, :], temperature, top_p)
            generated.append(next_token)
            if next_token.item() == eos_token_id:
                break

        return torch.stack(generated, dim=1)  # [1, num_generated]

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature == 0.0:
            return logits.argmax(dim=-1)  # [B]
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
