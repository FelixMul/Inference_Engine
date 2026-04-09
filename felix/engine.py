"""Optimized inference engine for Qwen3.5-35B-A3B.

Data-parallel: loads one full model replica per GPU.
Each replica handles requests independently with:
- Manual decode loop (no HF generate overhead)
- torch.compile for fused kernels
- Batched generation within each GPU
"""

import threading
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Patch Triton's autotuner to be thread-safe.
# The bug: Autotuner.run() sets self.nargs then calls _bench() which reads it.
# Two threads calling run() simultaneously corrupt each other's nargs.
_triton_lock = threading.Lock()

def _patch_triton_autotuner():
    """Wrap Triton's Autotuner.run with a lock to make it thread-safe."""
    try:
        from triton.runtime.autotuner import Autotuner
        original_run = Autotuner.run
        def thread_safe_run(self, *args, **kwargs):
            with _triton_lock:
                return original_run(self, *args, **kwargs)
        Autotuner.run = thread_safe_run
        print("[engine] Patched Triton autotuner for thread safety")
    except Exception as e:
        print(f"[engine] Warning: could not patch Triton autotuner: {e}")

_patch_triton_autotuner()


@dataclass
class GenerationResult:
    """Result from a single generation request."""
    token_ids: torch.Tensor   # generated token ids (CPU)
    hit_eos: bool


class InferenceEngine:
    """Manages multiple model replicas across GPUs."""

    def __init__(self, model_path: str, num_gpus: int = 8, compile_model: bool = True):
        self.model_path = model_path
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.models: list = []
        self.tokenizer = None
        self.config = None
        self.compile_model = compile_model
        self.eos_token_ids: set = set()

    def load(self):
        """Load tokenizer and one model replica per GPU."""
        print(f"Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.config = AutoConfig.from_pretrained(self.model_path)

        # Build EOS token set
        eos = self.tokenizer.eos_token_id
        if isinstance(eos, int):
            self.eos_token_ids = {eos}
        elif isinstance(eos, list):
            self.eos_token_ids = set(eos)
        # Also add from generation config and text config
        for cfg in [self.config, getattr(self.config, 'text_config', None)]:
            if cfg is None:
                continue
            gen_eos = getattr(cfg, 'eos_token_id', None)
            if isinstance(gen_eos, list):
                self.eos_token_ids.update(gen_eos)
            elif isinstance(gen_eos, int):
                self.eos_token_ids.add(gen_eos)
        # Hardcode known EOS tokens for this model
        self.eos_token_ids.update({248044, 248046})
        print(f"EOS token IDs: {self.eos_token_ids}")

        print(f"Loading {self.num_gpus} model replicas (bf16)...")
        t0 = time.time()
        for i in range(self.num_gpus):
            print(f"  GPU {i}...", end=" ", flush=True)
            t1 = time.time()
            m = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map=f"cuda:{i}",
                attn_implementation="sdpa",
            )
            m.eval()

            if self.compile_model:
                try:
                    print("compiling...", end=" ", flush=True)
                    m.model.forward = torch.compile(
                        m.model.forward,
                        mode="default",
                        fullgraph=False,
                    )
                except Exception as e:
                    print(f"compile failed ({e}), using eager", end=" ", flush=True)

            self.models.append(m)
            print(f"{time.time() - t1:.1f}s")

        print(f"All replicas loaded in {time.time() - t0:.1f}s")
        self._print_memory()
        self._warmup()

    def _warmup(self):
        """Run one request per GPU to warm up Triton kernel caches."""
        print("Warming up GPUs...")
        dummy_ids = self.tokenizer("Hello", return_tensors="pt").input_ids
        for i in range(self.num_gpus):
            print(f"  GPU {i}...", end=" ", flush=True)
            self._generate_impl(i, dummy_ids, max_new_tokens=4)
            print("ok")
        print("Warmup complete.")

    def _print_memory(self):
        """Print per-GPU memory usage."""
        for i in range(self.num_gpus):
            used = torch.cuda.memory_allocated(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {used:.1f} / {total:.1f} GB")

    @torch.no_grad()
    def generate(
        self,
        gpu_id: int,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        """Manual decode loop on a specific GPU."""
        return self._generate_impl(gpu_id, input_ids, max_new_tokens, temperature, top_p)

    def _generate_impl(
        self,
        gpu_id: int,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        model = self.models[gpu_id]
        device = f"cuda:{gpu_id}"
        ids = input_ids.to(device)

        batch_size, seq_len = ids.shape
        generated_ids = []
        hit_eos = False

        # Prefill: run the full prompt through the model to get the first next-token logits
        outputs = model(input_ids=ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        for step in range(max_new_tokens):
            # Sample next token
            next_token = self._sample(next_token_logits, temperature, top_p)
            generated_ids.append(next_token)

            # Check EOS
            if next_token.item() in self.eos_token_ids:
                hit_eos = True
                break

            # Decode step: feed just the new token with KV cache
            outputs = model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        if generated_ids:
            result_ids = torch.cat(generated_ids, dim=0).cpu()
        else:
            result_ids = torch.tensor([], dtype=torch.long)

        return GenerationResult(token_ids=result_ids, hit_eos=hit_eos)

    @torch.no_grad()
    def generate_batch(
        self,
        gpu_id: int,
        input_ids_list: list[torch.Tensor],
        max_new_tokens_list: list[int],
        temperature_list: list[float],
        top_p_list: list[float],
    ) -> list[GenerationResult]:
        """Batched generation on a single GPU. Amortizes weight reads across requests."""
        return self._generate_batch_impl(
            gpu_id, input_ids_list, max_new_tokens_list,
            temperature_list, top_p_list,
        )

    def _generate_batch_impl(
        self,
        gpu_id: int,
        input_ids_list: list[torch.Tensor],
        max_new_tokens_list: list[int],
        temperature_list: list[float],
        top_p_list: list[float],
    ) -> list[GenerationResult]:
        if len(input_ids_list) == 1:
            return [self._generate_impl(
                gpu_id, input_ids_list[0], max_new_tokens_list[0],
                temperature_list[0], top_p_list[0],
            )]

        model = self.models[gpu_id]
        device = f"cuda:{gpu_id}"
        batch_size = len(input_ids_list)
        max_new = max(max_new_tokens_list)

        # Pad inputs to same length (left-pad)
        max_input_len = max(ids.shape[1] for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if isinstance(pad_id, list):
            pad_id = pad_id[0]

        padded_ids = []
        attention_masks = []
        input_lengths = []
        for ids in input_ids_list:
            seq_len = ids.shape[1]
            input_lengths.append(seq_len)
            pad_len = max_input_len - seq_len
            if pad_len > 0:
                padded = F.pad(ids, (pad_len, 0), value=pad_id)
                mask = F.pad(torch.ones(1, seq_len, dtype=torch.long), (pad_len, 0), value=0)
            else:
                padded = ids
                mask = torch.ones(1, seq_len, dtype=torch.long)
            padded_ids.append(padded)
            attention_masks.append(mask)

        batch_ids = torch.cat(padded_ids, dim=0).to(device)
        batch_mask = torch.cat(attention_masks, dim=0).to(device)

        # Track per-request state
        generated = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        remaining_tokens = list(max_new_tokens_list)

        # Prefill
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        # Precompute tensors for vectorized sampling
        temperatures = torch.tensor(temperature_list, device=device).unsqueeze(1)
        finished_t = torch.zeros(batch_size, dtype=torch.bool, device=device)
        remaining_t = torch.tensor(max_new_tokens_list, dtype=torch.long, device=device)
        eos_ids = torch.tensor(list(self.eos_token_ids), dtype=torch.long, device=device)
        pad_token = torch.tensor(pad_id, dtype=torch.long, device=device)
        all_generated = torch.zeros(batch_size, max_new, dtype=torch.long, device=device)
        gen_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        for step in range(max_new):
            if finished_t.all():
                break

            # Vectorized sampling across full batch
            next_tokens = self._sample_batch(next_token_logits, temperatures, finished_t, pad_token)

            # Store tokens and update state
            all_generated[:, step] = next_tokens
            gen_lengths += (~finished_t).long()

            # Check EOS and remaining tokens
            remaining_t -= (~finished_t).long()
            hit_eos_mask = (next_tokens.unsqueeze(1) == eos_ids.unsqueeze(0)).any(dim=1)
            finished_t |= hit_eos_mask | (remaining_t <= 0)

            if finished_t.all():
                break

            next_input = next_tokens.unsqueeze(1)
            batch_mask = torch.cat([batch_mask, (~finished_t).long().unsqueeze(1)], dim=1)

            outputs = model(
                input_ids=next_input,
                attention_mask=batch_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        # Build results
        results = []
        for i in range(batch_size):
            length = gen_lengths[i].item()
            if length > 0:
                token_ids = all_generated[i, :length].cpu()
                hit_eos = token_ids[-1].item() in self.eos_token_ids
            else:
                token_ids = torch.tensor([], dtype=torch.long)
                hit_eos = False
            results.append(GenerationResult(token_ids=token_ids, hit_eos=hit_eos))

        return results

    def _sample_batch(self, logits: torch.Tensor, temperatures: torch.Tensor,
                       finished: torch.Tensor, pad_token: torch.Tensor) -> torch.Tensor:
        """Vectorized sampling across the full batch. Single kernel call."""
        # For finished sequences, return pad token
        # For greedy (temp=0), use argmax
        # For sampling, use multinomial
        greedy_mask = (temperatures.squeeze(1) <= 0) | finished
        sample_mask = ~greedy_mask

        result = torch.full((logits.shape[0],), pad_token.item(), dtype=torch.long, device=logits.device)

        # Greedy tokens
        if greedy_mask.any():
            result[greedy_mask] = logits[greedy_mask].argmax(dim=-1)

        # Sampled tokens
        if sample_mask.any():
            scaled = logits[sample_mask] / temperatures[sample_mask]
            probs = F.softmax(scaled, dim=-1)
            result[sample_mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Override finished sequences with pad
        result[finished] = pad_token
        return result

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Sample a single token from logits (used for non-batched generation)."""
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def tokenize_chat(self, messages: list[dict]) -> torch.Tensor:
        """Apply chat template and tokenize. Returns input_ids tensor."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer(text, return_tensors="pt").input_ids

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token ids to string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def is_eos(self, token_id: int) -> bool:
        """Check if a token id is an EOS token."""
        return token_id in self.eos_token_ids
