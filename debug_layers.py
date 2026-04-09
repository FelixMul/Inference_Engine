"""
Layer-by-layer comparison between HF model and our model.
Run with: python debug_layers.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/data/GPUINFERENCE/Qwen3.5-35B-A3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2+2?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
print(f"Input shape: {input_ids.shape}")

# ── HF model - capture hidden states per layer ────────────────────────────────
print("\nRunning HF model with hooks...")
hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
hf.eval()

hf_hidden = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hf_hidden[name] = output[0].detach().clone()
        else:
            hf_hidden[name] = output.detach().clone()
    return hook

# Hook embedding and each layer output
hf.model.embed_tokens.register_forward_hook(make_hook("embed"))
for i, layer in enumerate(hf.model.layers):
    layer.register_forward_hook(make_hook(f"layer_{i}"))

with torch.no_grad():
    hf(input_ids)

del hf
torch.cuda.empty_cache()

# ── Our model - capture hidden states per layer ───────────────────────────────
print("Running our model layer by layer...")
from server.model.loader import load_model
from server.model.rope import build_rope_freqs
from server.model.config import NUM_LAYERS

our = load_model(MODEL_PATH, device="cuda")
our.eval()

T = input_ids.shape[1]
inv_freq = build_rope_freqs(T + 100, input_ids.device)
position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)

kv_caches = [None] * NUM_LAYERS
lin_states = [None] * NUM_LAYERS
conv_bufs = [None] * NUM_LAYERS

with torch.no_grad():
    x = our.embed_tokens(input_ids)
    our_embed = x.clone()

    print(f"\n{'Layer':<8} {'HF norm':>12} {'Our norm':>12} {'Diff':>12} {'Match':>8}")
    print("-" * 56)

    # Embedding
    hf_e = hf_hidden["embed"]
    diff = (hf_e.float() - our_embed.float()).abs().max().item()
    print(f"{'embed':<8} {hf_e.norm().item():>12.4f} {our_embed.norm().item():>12.4f} {diff:>12.4f} {'OK' if diff < 0.1 else 'FAIL':>8}")

    for i, layer in enumerate(our.layers):
        x, new_kv, new_lin, new_conv = layer(
            x, position_ids, inv_freq,
            kv_cache=kv_caches[i],
            lin_state=lin_states[i],
            conv_buf=conv_bufs[i],
        )
        kv_caches[i] = new_kv
        lin_states[i] = new_lin
        conv_bufs[i] = new_conv

        hf_h = hf_hidden[f"layer_{i}"]
        diff = (hf_h.float() - x.float()).abs().max().item()
        match = "OK" if diff < 1.0 else "FAIL"
        print(f"{i:<8} {hf_h.norm().item():>12.4f} {x.norm().item():>12.4f} {diff:>12.4f} {match:>8}")

        if diff > 5.0:
            print(f"  >>> Large divergence at layer {i}! Stopping.")
            break

print("\nDone.")
