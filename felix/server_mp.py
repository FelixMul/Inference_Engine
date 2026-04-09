"""Multi-process DP server for Qwen3.5-35B-A3B.

Spawns one worker process per GPU. Each worker has its own Python interpreter,
eliminating GIL contention and Triton thread-safety issues.
Master process runs HTTP server and dispatches requests to workers.

Usage:
    python felix/server_mp.py [--port 8003] [--num-gpus 8]
"""

import argparse
import asyncio
import os
import signal
import sys
import time
import uuid
import threading
from multiprocessing import Queue
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from worker import start_worker, BatchWorkRequest, WorkResult
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
# Request tracking
# ---------------------------------------------------------------------------

@dataclass
class PendingRequest:
    input_ids: torch.Tensor
    max_tokens: int
    temperature: float
    top_p: float
    prompt_len: int
    future: asyncio.Future
    request_id: int


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()
tokenizer = None
profiler: Profiler = None
request_queues: list[Queue] = []  # one per GPU worker
result_queue: Queue = None
pending_futures: dict = {}  # request_id -> (future, event_loop)
futures_lock = threading.Lock()
request_counter = 0
counter_lock = asyncio.Lock()
num_gpus = 8
global_queue: asyncio.Queue = None  # single queue for all incoming requests


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    profiler.print_summary()
    profiler.save()
    return {"recorded": len(profiler.profiles)}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global request_counter

    prof = profiler.new_request()
    t_start = time.perf_counter()

    # Tokenize (in thread pool to not block event loop)
    with Timer() as t_tok:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        loop = asyncio.get_event_loop()
        input_ids = await loop.run_in_executor(None, _tokenize, messages)
        prompt_len = input_ids.shape[1]
    prof.tokenize = t_tok.elapsed

    # Assign request ID and submit to global queue
    async with counter_lock:
        req_id = request_counter
        request_counter += 1

    future = loop.create_future()
    pending = PendingRequest(input_ids, req.max_tokens, req.temperature, req.top_p, prompt_len, future, req_id)
    await global_queue.put(pending)

    # Wait for result
    result_data = await future
    prof.generate = time.perf_counter() - t_start - t_tok.elapsed

    # Decode
    with Timer() as t_dec:
        token_ids = torch.frombuffer(bytearray(result_data.token_ids_bytes), dtype=torch.long)
        content = tokenizer.decode(token_ids, skip_special_tokens=True)
        completion_tokens = result_data.token_count
    prof.decode = t_dec.elapsed

    prof.prompt_tokens = prompt_len
    prof.completion_tokens = completion_tokens
    prof.total = time.perf_counter() - t_start
    profiler.record(prof)

    eos_ids = {248044, 248046}
    hit_eos = result_data.hit_eos
    finish_reason = "stop" if hit_eos else "length"

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


def _tokenize(messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt").input_ids


# ---------------------------------------------------------------------------
# Global dispatcher + result dispatcher
# ---------------------------------------------------------------------------

# Track which GPUs are busy
gpu_busy: list[bool] = []

def _dispatch_to_gpu(gpu_id: int, batch: list[PendingRequest], loop: asyncio.AbstractEventLoop):
    """Send a batch of requests to a specific GPU worker."""
    batch_req = BatchWorkRequest(
        request_ids=[r.request_id for r in batch],
        input_ids_bytes_list=[r.input_ids.numpy().tobytes() for r in batch],
        input_shapes=[tuple(r.input_ids.shape) for r in batch],
        max_tokens_list=[r.max_tokens for r in batch],
        temperature_list=[r.temperature for r in batch],
        top_p_list=[r.top_p for r in batch],
    )
    with futures_lock:
        for r in batch:
            pending_futures[r.request_id] = (r.future, loop)
    gpu_busy[gpu_id] = True
    request_queues[gpu_id].put(batch_req)


async def global_dispatcher():
    """Collect all incoming requests, distribute evenly across GPUs."""
    loop = asyncio.get_event_loop()

    while True:
        # Wait for at least one request
        first = await global_queue.get()
        all_requests = [first]

        # Wait for concurrent requests to finish tokenizing and get queued.
        # Tokenization takes ~15ms per request in thread pool, so 20ms
        # captures most/all of a burst while adding minimal latency.
        await asyncio.sleep(0.02)

        # Drain everything available
        while not global_queue.empty():
            try:
                all_requests.append(global_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Find free GPUs
        free_gpus = [i for i in range(num_gpus) if not gpu_busy[i]]
        if not free_gpus:
            # All GPUs busy — use all of them (requests will queue in worker)
            free_gpus = list(range(num_gpus))

        # Distribute requests evenly across free GPUs
        n_gpus = min(len(free_gpus), len(all_requests))
        gpus_to_use = free_gpus[:n_gpus]

        # Split requests into chunks, one per GPU
        chunks = [[] for _ in range(n_gpus)]
        for i, req in enumerate(all_requests):
            chunks[i % n_gpus].append(req)

        dispatch_parts = []
        for gpu_id, chunk in zip(gpus_to_use, chunks):
            if chunk:
                dispatch_parts.append(f"GPU{gpu_id}={len(chunk)}")
                _dispatch_to_gpu(gpu_id, chunk, loop)

        print(f"[dispatcher] {len(all_requests)} requests -> {', '.join(dispatch_parts)}", flush=True)


def result_dispatcher():
    """Background thread that reads results from workers and resolves futures."""
    # Track how many pending results per GPU to know when a GPU is free
    gpu_pending = [0] * num_gpus

    while True:
        try:
            item = result_queue.get()
            if item is None:
                break
            if isinstance(item, WorkResult):
                with futures_lock:
                    entry = pending_futures.pop(item.request_id, None)
                if entry:
                    future, loop = entry
                    loop.call_soon_threadsafe(future.set_result, item)

                    # Check if this GPU's batch is complete
                    # Find which GPU this request was on by checking all queues
                    # Simple approach: mark all GPUs as free when any result comes back
                    # The dispatcher will re-check on next dispatch
                    for i in range(num_gpus):
                        gpu_busy[i] = False
        except Exception as e:
            print(f"[result_dispatcher] Error: {e}")


@app.on_event("startup")
async def startup():
    global global_queue
    global_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    loop.create_task(global_dispatcher())
    print(f"Started global dispatcher for {num_gpus} GPUs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global tokenizer, profiler, request_queues, result_queue, num_gpus, gpu_busy

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-path", default="/dev/shm/Qwen3.5-35B-A3B")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu-offset", type=int, default=0,
                        help="Start from this GPU index (skip busy GPUs)")
    args = parser.parse_args()
    num_gpus = args.num_gpus

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    profiler = Profiler(print_every=16)
    result_queue = Queue()

    # Spawn worker processes
    gpu_offset = args.gpu_offset
    print(f"Spawning {num_gpus} GPU workers (GPUs {gpu_offset}-{gpu_offset+num_gpus-1})...")
    workers = []
    request_queues = []
    for i in range(num_gpus):
        rq = Queue()
        request_queues.append(rq)
        physical_gpu = gpu_offset + i
        w = start_worker(i, args.model_path, rq, result_queue, physical_gpu=physical_gpu)
        workers.append(w)

    # Wait for all workers to be ready
    ready_count = 0
    while ready_count < num_gpus:
        msg = result_queue.get()
        if msg[0] == "ready":
            ready_count += 1
            print(f"  Worker GPU {msg[1]} ready ({ready_count}/{num_gpus})")

    print(f"All {num_gpus} workers ready!")

    # Initialize GPU busy tracking
    gpu_busy = [False] * num_gpus

    # Start result dispatcher thread
    dispatcher = threading.Thread(target=result_dispatcher, daemon=True)
    dispatcher.start()

    def on_exit(sig, frame):
        print("\nShutting down...")
        profiler.print_summary()
        profiler.save()
        for rq in request_queues:
            rq.put(None)
        raise SystemExit(0)
    signal.signal(signal.SIGINT, on_exit)

    print(f"Server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
