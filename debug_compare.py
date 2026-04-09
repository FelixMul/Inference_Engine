"""
Compare our custom model output against HF reference layer by layer.
Run with: python debug_compare.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/data/GPUINFERENCE/Qwen3.5-35B-A3B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2+2?"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
print(f"Input ids shape: {input_ids.shape}, tokens: {input_ids[0].tolist()}")

# ── HF reference ──────────────────────────────────────────────────────────────
print("\nLoading HF reference model...")
hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
hf.eval()

with torch.no_grad():
    hf_out = hf(input_ids)
    hf_logits = hf_out.logits[:, -1, :]  # [1, V]

hf_top = hf_logits.topk(5)
print(f"HF top-5 token ids : {hf_top.indices[0].tolist()}")
print(f"HF top-5 tokens    : {[tokenizer.decode([i]) for i in hf_top.indices[0].tolist()]}")
print(f"HF has NaN: {hf_logits.isnan().any().item()}")

del hf
torch.cuda.empty_cache()

# ── Our model ─────────────────────────────────────────────────────────────────
print("\nLoading our model...")
from server.model.loader import load_model
from server.model.rope import build_rope_freqs

our = load_model(MODEL_PATH, device="cuda")

T = input_ids.shape[1]
inv_freq = build_rope_freqs(T + 100, input_ids.device)
position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)

with torch.no_grad():
    our_logits, _, _, _ = our.forward_step(
        input_ids, position_ids, inv_freq,
        [None] * 40, [None] * 40, [None] * 40,
    )
    our_logits = our_logits[:, -1, :]

our_top = our_logits.topk(5)
print(f"Our top-5 token ids: {our_top.indices[0].tolist()}")
print(f"Our top-5 tokens   : {[tokenizer.decode([i]) for i in our_top.indices[0].tolist()]}")
print(f"Our has NaN: {our_logits.isnan().any().item()}")
print(f"Our has Inf: {our_logits.isinf().any().item()}")

# ── Summary ───────────────────────────────────────────────────────────────────
max_diff = (hf_logits.float() - our_logits.float()).abs().max().item()
print(f"\nMax logit diff (HF vs ours): {max_diff:.4f}")
print(f"HF argmax: {hf_logits.argmax().item()} -> '{tokenizer.decode([hf_logits.argmax().item()])}'")
print(f"Our argmax: {our_logits.argmax().item()} -> '{tokenizer.decode([our_logits.argmax().item()])}'")
