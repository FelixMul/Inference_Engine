"""HTTP server for Qwen3.5-35B-A3B inference engine.

Data-parallel across GPUs with dynamic batching.
Requests are collected per-GPU and processed in batches to amortize
CPU/kernel-launch overhead across multiple requests.

Usage:
    python felix/server.py [--port 8003] [--model-path /dev/shm/Qwen3.5-35B-A3B]
"""

import argparse
import asyncio
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch

from engine import InferenceEngine, GenerationResult
from profiling import Profiler, Timer

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "Qwen/Qwen3.5-35B-A3B"
    messages: list[Message]
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str

class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "Qwen/Qwen3.5-35B-A3B"
    choices: list[Choice]
    usage: Usage

# ---------------------------------------------------------------------------
# Batch collector: groups requests per GPU
# ---------------------------------------------------------------------------

@dataclass
class PendingRequest:
    input_ids: torch.Tensor
    max_tokens: int
    temperature: float
    top_p: float
    future: asyncio.Future
    prompt_len: int


class GPUBatchScheduler:
    """Collects requests for a GPU and dispatches them in batches."""

    def __init__(self, gpu_id: int, engine: InferenceEngine, thread_pool: ThreadPoolExecutor,
                 max_batch_size: int = 8, batch_timeout: float = 0.02):
        self.gpu_id = gpu_id
        self.engine = engine
        self.thread_pool = thread_pool
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout  # seconds to wait before dispatching partial batch
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._task = loop.create_task(self._batch_loop())

    async def submit(self, req: PendingRequest):
        await self.queue.put(req)

    async def _batch_loop(self):
        """Continuously collect and dispatch batches."""
        loop = asyncio.get_event_loop()
        while True:
            batch: list[PendingRequest] = []

            # Wait for at least one request
            first = await self.queue.get()
            batch.append(first)

            # Collect more requests up to max_batch_size or timeout
            deadline = time.monotonic() + self.batch_timeout
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            # Dispatch batch to GPU thread
            try:
                results = await loop.run_in_executor(
                    self.thread_pool,
                    self._run_batch,
                    batch,
                )
                for req, result in zip(batch, results):
                    req.future.set_result(result)
            except Exception as e:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

    def _run_batch(self, batch: list[PendingRequest]) -> list[GenerationResult]:
        """Run a batch of requests on the GPU (called from thread pool)."""
        return self.engine.generate_batch(
            self.gpu_id,
            [r.input_ids for r in batch],
            [r.max_tokens for r in batch],
            [r.temperature for r in batch],
            [r.top_p for r in batch],
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()
engine: InferenceEngine = None
profiler: Profiler = None
schedulers: list[GPUBatchScheduler] = []
thread_pool: ThreadPoolExecutor = None
request_counter = 0
counter_lock = asyncio.Lock()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Print profiling summary and save to disk."""
    profiler.print_summary()
    profiler.save()
    return {"recorded": len(profiler.profiles)}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global request_counter

    prof = profiler.new_request()
    t_start = time.perf_counter()

    # Tokenize
    with Timer() as t_tok:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        input_ids = engine.tokenize_chat(messages)
        prompt_len = input_ids.shape[1]
    prof.tokenize = t_tok.elapsed

    # Round-robin GPU selection
    async with counter_lock:
        gpu_id = request_counter % engine.num_gpus
        request_counter += 1
    prof.gpu_id = gpu_id

    # Submit to batch scheduler
    t_queue_start = time.perf_counter()
    future = asyncio.get_event_loop().create_future()
    pending = PendingRequest(
        input_ids=input_ids,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        future=future,
        prompt_len=prompt_len,
    )
    await schedulers[gpu_id].submit(pending)

    # Wait for result
    result: GenerationResult = await future
    prof.queue_wait = time.perf_counter() - t_queue_start - 0  # includes batch wait + generate
    prof.generate = time.perf_counter() - t_queue_start

    # Decode
    with Timer() as t_dec:
        new_tokens = result.token_ids
        content = engine.decode(new_tokens)
        completion_tokens = len(new_tokens)
    prof.decode = t_dec.elapsed

    prof.prompt_tokens = prompt_len
    prof.completion_tokens = completion_tokens
    prof.total = time.perf_counter() - t_start

    profiler.record(prof)

    finish_reason = "stop" if result.hit_eos else "length"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        choices=[Choice(message=ChoiceMessage(content=content), finish_reason=finish_reason)],
        usage=Usage(
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            total_tokens=prompt_len + completion_tokens,
        ),
    )


@app.on_event("startup")
async def startup():
    """Start batch schedulers after the event loop is running."""
    loop = asyncio.get_event_loop()
    for scheduler in schedulers:
        scheduler.start(loop)
    print(f"Started {len(schedulers)} GPU batch schedulers")


def main():
    global engine, profiler, schedulers, thread_pool

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-path", default="/dev/shm/Qwen3.5-35B-A3B")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=8,
                        help="Max requests to batch per GPU")
    parser.add_argument("--batch-timeout", type=float, default=0.05,
                        help="Seconds to wait for more requests before dispatching partial batch")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()

    engine = InferenceEngine(args.model_path, args.num_gpus, compile_model=args.compile)
    engine.load()

    profiler = Profiler(print_every=16)

    thread_pool = ThreadPoolExecutor(max_workers=args.num_gpus)
    schedulers = [
        GPUBatchScheduler(i, engine, thread_pool, args.max_batch_size, args.batch_timeout)
        for i in range(engine.num_gpus)
    ]

    # Print summary on Ctrl+C
    def on_exit(sig, frame):
        print("\nShutting down...")
        profiler.print_summary()
        profiler.save()
        raise SystemExit(0)
    signal.signal(signal.SIGINT, on_exit)

    print(f"Server starting on {args.host}:{args.port}")
    print(f"Batching: max_batch_size={args.max_batch_size}, timeout={args.batch_timeout}s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
