"""Lightweight profiling for the inference engine.

Tracks per-request timing breakdown and aggregates stats.
Prints a summary after every N requests and on shutdown.
"""

import json
import os
import time
import threading
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime


@dataclass
class RequestProfile:
    """Timing breakdown for a single request."""
    request_id: int = 0
    gpu_id: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Timing (seconds)
    queue_wait: float = 0.0     # waiting for GPU semaphore
    tokenize: float = 0.0       # chat template + tokenization
    generate: float = 0.0       # model.generate() call
    decode: float = 0.0         # detokenization
    total: float = 0.0          # end-to-end request time

    @property
    def tokens_per_sec(self) -> float:
        if self.generate > 0:
            return self.completion_tokens / self.generate
        return 0.0

    @property
    def time_per_token_ms(self) -> float:
        if self.completion_tokens > 0:
            return (self.generate / self.completion_tokens) * 1000
        return 0.0


class Profiler:
    """Collects request profiles and prints summaries."""

    def __init__(self, print_every: int = 8, out_dir: str = "felix/out"):
        self.profiles: list[RequestProfile] = []
        self.lock = threading.Lock()
        self.print_every = print_every
        self._request_counter = 0
        self.out_dir = out_dir
        self._start_time = datetime.now().strftime("%H-%M-%S")
        os.makedirs(out_dir, exist_ok=True)

    def new_request(self) -> RequestProfile:
        with self.lock:
            self._request_counter += 1
            p = RequestProfile(request_id=self._request_counter)
            return p

    def record(self, profile: RequestProfile):
        with self.lock:
            self.profiles.append(profile)
            count = len(self.profiles)

        if count % self.print_every == 0:
            self.print_summary(last_n=self.print_every)

    def print_summary(self, last_n: int | None = None):
        with self.lock:
            if not self.profiles:
                print("[profiler] No requests recorded.")
                return

            profs = self.profiles[-last_n:] if last_n else self.profiles
            n = len(profs)

        def avg(vals):
            return sum(vals) / len(vals) if vals else 0

        def p50(vals):
            s = sorted(vals)
            return s[len(s) // 2] if s else 0

        def p99(vals):
            s = sorted(vals)
            idx = min(int(len(s) * 0.99), len(s) - 1)
            return s[idx] if s else 0

        queue_waits = [p.queue_wait for p in profs]
        tokenize_times = [p.tokenize for p in profs]
        gen_times = [p.generate for p in profs]
        decode_times = [p.decode for p in profs]
        totals = [p.total for p in profs]
        tps = [p.tokens_per_sec for p in profs]
        tpt = [p.time_per_token_ms for p in profs]
        comp_tokens = [p.completion_tokens for p in profs]

        gpu_counts = defaultdict(int)
        for p in profs:
            gpu_counts[p.gpu_id] += 1

        label = f"last {n}" if last_n else f"all {n}"
        print(f"\n{'='*70}")
        print(f"  Profiler Summary ({label} requests)")
        print(f"{'='*70}")
        print(f"  {'Metric':<25} {'Avg':>10} {'P50':>10} {'P99':>10}")
        print(f"  {'-'*55}")
        print(f"  {'Queue wait (s)':<25} {avg(queue_waits):>10.3f} {p50(queue_waits):>10.3f} {p99(queue_waits):>10.3f}")
        print(f"  {'Tokenize (s)':<25} {avg(tokenize_times):>10.3f} {p50(tokenize_times):>10.3f} {p99(tokenize_times):>10.3f}")
        print(f"  {'Generate (s)':<25} {avg(gen_times):>10.3f} {p50(gen_times):>10.3f} {p99(gen_times):>10.3f}")
        print(f"  {'Decode (s)':<25} {avg(decode_times):>10.3f} {p50(decode_times):>10.3f} {p99(decode_times):>10.3f}")
        print(f"  {'Total (s)':<25} {avg(totals):>10.3f} {p50(totals):>10.3f} {p99(totals):>10.3f}")
        print(f"  {'-'*55}")
        print(f"  {'Completion tokens':<25} {avg(comp_tokens):>10.0f} {p50(comp_tokens):>10.0f} {p99(comp_tokens):>10.0f}")
        print(f"  {'Tok/s (generate)':<25} {avg(tps):>10.1f} {p50(tps):>10.1f} {p99(tps):>10.1f}")
        print(f"  {'ms/token (generate)':<25} {avg(tpt):>10.1f} {p50(tpt):>10.1f} {p99(tpt):>10.1f}")
        print(f"  {'-'*55}")
        gpu_str = "  GPU distribution: " + ", ".join(f"GPU{k}={v}" for k, v in sorted(gpu_counts.items()))
        print(gpu_str)
        print(f"{'='*70}\n")

    def save(self):
        """Save all profiles to a JSON file in out_dir, timestamped by start time."""
        with self.lock:
            if not self.profiles:
                return
            data = {
                "start_time": self._start_time,
                "num_requests": len(self.profiles),
                "profiles": [
                    {
                        "request_id": p.request_id,
                        "gpu_id": p.gpu_id,
                        "prompt_tokens": p.prompt_tokens,
                        "completion_tokens": p.completion_tokens,
                        "queue_wait": round(p.queue_wait, 4),
                        "tokenize": round(p.tokenize, 4),
                        "generate": round(p.generate, 4),
                        "decode": round(p.decode, 4),
                        "total": round(p.total, 4),
                        "tokens_per_sec": round(p.tokens_per_sec, 1),
                        "ms_per_token": round(p.time_per_token_ms, 1),
                    }
                    for p in self.profiles
                ],
            }
        path = os.path.join(self.out_dir, f"profile_{self._start_time}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[profiler] Saved {len(data['profiles'])} profiles to {path}")


# Convenience timer context manager
class Timer:
    """Simple context manager that records elapsed time."""
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
