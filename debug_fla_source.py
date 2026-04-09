"""Dump the torch fallback of chunk_gated_delta_rule that HF is actually calling.

We can do this WITHOUT loading the 35B model — just import the class and
introspect the method (and follow it through any wrappers / module-level
fallbacks).
"""
import inspect
import sys

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

print("=" * 70)
print(f"modeling file: {M.__file__}")
print("=" * 70)

# 1. Find the GatedDeltaNet class
gdn_cls = None
for name in dir(M):
    obj = getattr(M, name)
    if inspect.isclass(obj) and "GatedDeltaNet" in name:
        gdn_cls = obj
        print(f"Found class: {name}")
        break

if gdn_cls is None:
    print("ERROR: no GatedDeltaNet class found in modeling file")
    sys.exit(1)

# 2. Inspect chunk_gated_delta_rule on the class
fn = getattr(gdn_cls, "chunk_gated_delta_rule", None)
print(f"\nchunk_gated_delta_rule attr: {fn!r}")
print(f"type: {type(fn)}")

if fn is not None:
    try:
        print(f"defined in: {inspect.getsourcefile(fn)}")
    except Exception as e:
        print(f"  (no sourcefile: {e})")
    try:
        print("\n--- SOURCE ---")
        print(inspect.getsource(fn))
    except Exception as e:
        print(f"  (no source: {e})")

# 3. Also dump every module-level callable whose name mentions delta/recurrent/chunk
print("\n" + "=" * 70)
print("Module-level helpers (delta / recurrent / chunk):")
print("=" * 70)
for name in dir(M):
    obj = getattr(M, name)
    if callable(obj) and any(s in name.lower() for s in ("delta", "recurrent", "chunk")):
        print(f"\n--- {name} ({type(obj).__name__}) ---")
        try:
            print(f"defined in: {inspect.getsourcefile(obj)}")
        except Exception as e:
            print(f"  (no sourcefile: {e})")
        try:
            print(inspect.getsource(obj))
        except Exception as e:
            print(f"  (no source: {e})")

# 4. Show all imports of the modeling file so we can see what it pulls from fla
print("\n" + "=" * 70)
print("Imports in modeling file:")
print("=" * 70)
import ast
import pathlib
src = pathlib.Path(M.__file__).read_text()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        print(ast.unparse(node))
