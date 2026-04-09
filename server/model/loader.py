"""
Load weights from HuggingFace checkpoint into our custom model.
Transformers is used ONLY here for weight loading.
"""
import torch
from transformers import AutoModelForCausalLM
from .model import Qwen35MoE
from .config import FULL_ATTN_LAYERS, NUM_LAYERS


def _build_remapped_state_dict(model_path: str) -> dict:
    """Load HF checkpoint once and return a state dict keyed for our model."""
    print("Loading weights from HuggingFace checkpoint...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # load to CPU first for remapping
    )
    hf_sd = hf_model.state_dict()
    del hf_model  # free memory immediately

    new_sd: dict = {}

    def cp(our_key: str, hf_key: str):
        if hf_key not in hf_sd:
            print(f"  WARNING: missing HF key {hf_key}")
            return
        new_sd[our_key] = hf_sd[hf_key]

    # Embeddings
    cp("embed_tokens.weight", "model.embed_tokens.weight")
    cp("norm.weight", "model.norm.weight")
    cp("lm_head.weight", "lm_head.weight")

    for i in range(NUM_LAYERS):
        p = f"layers.{i}"
        hfp = f"model.layers.{i}"

        cp(f"{p}.input_layernorm.weight",       f"{hfp}.input_layernorm.weight")
        cp(f"{p}.post_attention_layernorm.weight", f"{hfp}.post_attention_layernorm.weight")

        if i in FULL_ATTN_LAYERS:
            # Full attention
            cp(f"{p}.attn.q_proj.weight",  f"{hfp}.self_attn.q_proj.weight")
            cp(f"{p}.attn.k_proj.weight",  f"{hfp}.self_attn.k_proj.weight")
            cp(f"{p}.attn.v_proj.weight",  f"{hfp}.self_attn.v_proj.weight")
            cp(f"{p}.attn.o_proj.weight",  f"{hfp}.self_attn.o_proj.weight")
            cp(f"{p}.attn.q_norm.weight",  f"{hfp}.self_attn.q_norm.weight")
            cp(f"{p}.attn.k_norm.weight",  f"{hfp}.self_attn.k_norm.weight")
        else:
            # GatedDeltaNet
            cp(f"{p}.attn.in_proj_qkv.weight", f"{hfp}.linear_attn.in_proj_qkv.weight")
            cp(f"{p}.attn.in_proj_z.weight",   f"{hfp}.linear_attn.in_proj_z.weight")
            cp(f"{p}.attn.in_proj_b.weight",   f"{hfp}.linear_attn.in_proj_b.weight")
            cp(f"{p}.attn.in_proj_a.weight",   f"{hfp}.linear_attn.in_proj_a.weight")
            cp(f"{p}.attn.conv1d_weight",      f"{hfp}.linear_attn.conv1d.weight")
            cp(f"{p}.attn.dt_bias",            f"{hfp}.linear_attn.dt_bias")
            cp(f"{p}.attn.A_log",              f"{hfp}.linear_attn.A_log")
            cp(f"{p}.attn.norm.weight",        f"{hfp}.linear_attn.norm.weight")
            cp(f"{p}.attn.out_proj.weight",    f"{hfp}.linear_attn.out_proj.weight")

        # MoE
        cp(f"{p}.mlp.gate.weight",             f"{hfp}.mlp.gate.weight")
        cp(f"{p}.mlp.experts_gate_up",         f"{hfp}.mlp.experts.gate_up_proj")
        cp(f"{p}.mlp.experts_down",            f"{hfp}.mlp.experts.down_proj")
        cp(f"{p}.mlp.shared_gate.weight",      f"{hfp}.mlp.shared_expert.gate_proj.weight")
        cp(f"{p}.mlp.shared_up.weight",        f"{hfp}.mlp.shared_expert.up_proj.weight")
        cp(f"{p}.mlp.shared_down.weight",      f"{hfp}.mlp.shared_expert.down_proj.weight")
        cp(f"{p}.mlp.shared_expert_gate.weight", f"{hfp}.mlp.shared_expert_gate.weight")

    return new_sd


def load_model(model_path: str, device: str = "cuda") -> Qwen35MoE:
    """Load a single model replica onto `device`. Kept for back-compat."""
    new_sd = _build_remapped_state_dict(model_path)
    return _instantiate_replica(new_sd, device)


def load_replicas(model_path: str, devices: list[str]) -> list[Qwen35MoE]:
    """Load N model replicas, one per device.

    Loads the HF checkpoint and remaps state-dict keys ONCE, then instantiates
    a fresh Qwen35MoE on each device and loads the (CPU) state dict into it.
    Avoids paying the 30s+ HF cold-load cost N times.
    """
    new_sd = _build_remapped_state_dict(model_path)
    replicas: list[Qwen35MoE] = []
    for i, device in enumerate(devices):
        print(f"Instantiating replica {i+1}/{len(devices)} on {device}...")
        replica = _instantiate_replica(new_sd, device)
        replicas.append(replica)
    print(f"All {len(devices)} replicas ready.")
    return replicas


def _instantiate_replica(new_sd: dict, device: str) -> Qwen35MoE:
    model = Qwen35MoE()
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model
