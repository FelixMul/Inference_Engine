"""Print the transformers source for Qwen3.5MoE linear attention."""
import inspect
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m

# Print the linear attention class
for name, cls in inspect.getmembers(m, inspect.isclass):
    if "linear" in name.lower() or "delta" in name.lower() or "attn" in name.lower():
        print(f"\n{'='*60}")
        print(f"CLASS: {name}")
        print('='*60)
        print(inspect.getsource(cls))
