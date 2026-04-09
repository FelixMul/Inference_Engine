"""
Deep dive into layer 0 to find exactly which sublayer diverges.
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

# ── HF model hooks ────────────────────────────────────────────────────────────
print("Loading HF model...")
hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
hf.eval()

hf_out = {}
layer0 = hf.model.layers[0]

layer0.input_layernorm.register_forward_hook(
    lambda m, i, o: hf_out.update({"input_ln": o.detach().clone()}))
layer0.linear_attn.register_forward_hook(
    lambda m, i, o: hf_out.update({"attn_out": (o if not isinstance(o, tuple) else o[0]).detach().clone()}))
layer0.post_attention_layernorm.register_forward_hook(
    lambda m, i, o: hf_out.update({"post_attn_ln": o.detach().clone()}))
layer0.mlp.register_forward_hook(
    lambda m, i, o: hf_out.update({"mlp_out": o.detach().clone()}))

with torch.no_grad():
    hf(input_ids)

del hf
torch.cuda.empty_cache()

# ── Our model ─────────────────────────────────────────────────────────────────
print("Loading our model...")
from server.model.loader import load_model
from server.model.rope import build_rope_freqs

our = load_model(MODEL_PATH, device="cuda")
our.eval()

T = input_ids.shape[1]
inv_freq = build_rope_freqs(T + 100, input_ids.device)
position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)

def diff(name, hf_t, our_t):
    hf_t = hf_t.float()
    our_t = our_t.float()
    d = (hf_t - our_t).abs().max().item()
    print(f"  {name:<25} HF={hf_t.norm():.4f}  Ours={our_t.norm():.4f}  MaxDiff={d:.4f}  {'OK' if d < 0.5 else 'FAIL'}")
    return d

with torch.no_grad():
    x = our.embed_tokens(input_ids)
    layer = our.layers[0]

    print("\n--- Layer 0 sublayer comparison ---")

    # 1. input_layernorm
    h = layer.input_layernorm(x)
    diff("input_layernorm", hf_out["input_ln"], h)

    # 2. GatedDeltaNet
    attn_out, lin_state, conv_buf = layer.attn(h, None, None)
    diff("attn_out", hf_out["attn_out"], attn_out)

    # 3. After residual
    x2 = x + attn_out
    hf_post_attn_input = hf_out["input_ln"]  # use as proxy — not exact

    # 4. post_attention_layernorm
    h2 = layer.post_attention_layernorm(x2)
    diff("post_attn_layernorm", hf_out["post_attn_ln"], h2)

    # 5. MoE
    mlp_out = layer.mlp(h2)
    diff("mlp_out", hf_out["mlp_out"], mlp_out)

    # 6. Full layer output
    x3 = x2 + mlp_out
    diff("layer_0_out", hf_out.get("layer0_out", x3), x3)  # approximate

print("\nDone.")
