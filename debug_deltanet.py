"""
Deep dive into GatedDeltaNet layer 0 internals.
Hooks on HF model's linear_attn to compare intermediate tensors.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/data/GPUINFERENCE/Qwen3.5-35B-A3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2+2?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
B, T = input_ids.shape

print(f"Input: {B}x{T} tokens")

# ── Monkey-patch HF GatedDeltaNet forward to capture intermediates ────────────
print("Loading HF model...")
hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
hf.eval()

hf_vals = {}
la = hf.model.layers[0].linear_attn  # the GatedDeltaNet module

orig_forward = la.forward.__func__

def patched_forward(self, hidden_states, cache_params=None, attention_mask=None):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import apply_mask_to_padding_states
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # conv + silu
    mixed_qkv_conv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
    hf_vals['qkv_after_conv'] = mixed_qkv_conv.transpose(1,2).detach().clone()

    mixed_qkv_conv = mixed_qkv_conv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv_conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key   = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    hf_vals['beta'] = beta.detach().clone()
    hf_vals['g'] = g.detach().clone()
    hf_vals['z'] = z.detach().clone()

    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key   = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    hf_vals['q_expanded'] = query.detach().clone()
    hf_vals['k_expanded'] = key.detach().clone()
    hf_vals['v'] = value.detach().clone()

    core_out, _ = self.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    hf_vals['core_out'] = core_out.detach().clone()

    core_out = core_out.reshape(-1, self.head_v_dim)
    z2 = z.reshape(batch_size, seq_len, -1, self.head_v_dim).reshape(-1, self.head_v_dim)
    core_out = self.norm(core_out, z2)
    hf_vals['normed_out'] = core_out.reshape(batch_size, seq_len, -1).detach().clone()

    return self.out_proj(core_out.reshape(batch_size, seq_len, -1))

import types
la.forward = types.MethodType(patched_forward, la)

with torch.no_grad():
    x_embed = hf.model.embed_tokens(input_ids)
    ln_out = hf.model.layers[0].input_layernorm(x_embed)
    la(ln_out)

del hf
torch.cuda.empty_cache()

# ── Our model ─────────────────────────────────────────────────────────────────
print("Loading our model...")
from server.model.loader import load_model
from server.model.rope import build_rope_freqs
import torch.nn.functional as F as F2

our = load_model(MODEL_PATH, device="cuda")
our.eval()

inv_freq = build_rope_freqs(T + 100, input_ids.device)
position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)

la_our = our.layers[0].attn
from server.model.linear_attn import _CONV_DIM, _K_DIM, _V_DIM, _EXPAND, _EXP_HEADS, _HEAD_DIM, LINEAR_CONV_KERNEL

def diff(name, hf_t, our_t):
    hf_t = hf_t.float(); our_t = our_t.float()
    d = (hf_t - our_t).abs().max().item()
    rel = d / (hf_t.abs().mean().item() + 1e-8)
    print(f"  {name:<25} HF_norm={hf_t.norm():.3f}  Our_norm={our_t.norm():.3f}  MaxDiff={d:.4f}  {'OK' if d < 0.1 else 'FAIL'}")

with torch.no_grad():
    x = our.embed_tokens(input_ids)
    h = our.layers[0].input_layernorm(x)
    k = LINEAR_CONV_KERNEL

    qkv = la_our.in_proj_qkv(h)
    z   = la_our.in_proj_z(h)
    b   = la_our.in_proj_b(h)
    a   = la_our.in_proj_a(h)

    # Conv
    qkv_t = qkv.transpose(1,2)
    pad = qkv_t.new_zeros(B, _CONV_DIM, k-1)
    qkv_padded = torch.cat([pad, qkv_t], dim=2)
    qkv_conv = F.conv1d(qkv_padded, la_our.conv1d_weight, groups=_CONV_DIM)
    qkv_conv = F.silu(qkv_conv)
    our_qkv_conv = qkv_conv.transpose(1,2)

    print("\n--- GatedDeltaNet internals ---")
    diff("qkv_after_conv", hf_vals['qkv_after_conv'], our_qkv_conv)

    # Split
    q  = our_qkv_conv[..., :_K_DIM].view(B, T, 16, _HEAD_DIM)
    kk = our_qkv_conv[..., _K_DIM:_K_DIM*2].view(B, T, 16, _HEAD_DIM)
    v  = our_qkv_conv[..., _K_DIM*2:].view(B, T, 32, _HEAD_DIM)

    q  = q.repeat_interleave(_EXPAND, dim=2)
    kk = kk.repeat_interleave(_EXPAND, dim=2)

    diff("q_expanded", hf_vals['q_expanded'], q)
    diff("k_expanded", hf_vals['k_expanded'], kk)
    diff("v", hf_vals['v'], v)

    # beta and decay
    beta  = torch.sigmoid(b.float()).to(h.dtype)
    g_our = (-la_our.A_log.float().exp() * F.softplus(a.float() + la_our.dt_bias.float()))
    decay = g_our.exp().to(h.dtype)

    diff("beta", hf_vals['beta'], beta)
    diff("g (log decay)", hf_vals['g'].to(h.dtype), g_our.to(h.dtype))
    diff("z", hf_vals['z'], z)

    # Sequential delta rule
    q_n  = F.normalize(q.float(), dim=-1).to(h.dtype)
    kk_n = F.normalize(kk.float(), dim=-1).to(h.dtype)
    state = h.new_zeros(B, _EXP_HEADS, _HEAD_DIM, _HEAD_DIM)
    outputs = []
    for t in range(T):
        q_t = q_n[:, t]; k_t = kk_n[:, t]; v_t = v[:, t]
        b_t = beta[:, t]; d_t = decay[:, t]
        Sk = torch.einsum("bhi,bhij->bhj", k_t, state)
        delta = v_t - Sk
        update = torch.einsum("bhi,bhj->bhij", k_t, b_t.unsqueeze(-1) * delta)
        state = state * d_t.unsqueeze(-1).unsqueeze(-1) + update
        outputs.append(torch.einsum("bhi,bhij->bhj", q_t, state))

    core_out = torch.stack(outputs, dim=1)  # [B, T, 32, 128]
    diff("core_out", hf_vals['core_out'], core_out)

print("\nDone.")
