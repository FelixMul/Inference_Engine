"""Print RMSNormGated and full attention source."""
import inspect
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m

for name, cls in inspect.getmembers(m, inspect.isclass):
    if "norm" in name.lower() or "Attention" in name:
        print(f"\n{'='*60}")
        print(f"CLASS: {name}")
        print('='*60)
        print(inspect.getsource(cls))
