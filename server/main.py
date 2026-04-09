"""FastAPI server for Qwen3.5-35B-A3B custom inference engine.

Data-parallel: one full model replica per GPU. Incoming requests are
dispatched round-robin across replicas; each replica serves one request
at a time (per-GPU asyncio.Lock), so up to NUM_GPUS requests run truly
in parallel. Blocking GPU work runs in a thread pool to keep the FastAPI
event loop responsive.
"""
import asyncio
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from server.model.loader import load_replicas

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/data/GPUINFERENCE/Qwen3.5-35B-A3B")
NUM_GPUS = int(os.environ.get("NUM_GPUS", torch.cuda.device_count() or 1))

app = FastAPI()
tokenizer = None
replicas: list = []
gpu_locks: list[asyncio.Lock] = []
thread_pool: ThreadPoolExecutor | None = None
rr_counter = 0
rr_lock = asyncio.Lock()


@app.on_event("startup")
async def startup():
    global tokenizer, replicas, gpu_locks, thread_pool

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    devices = [f"cuda:{i}" for i in range(NUM_GPUS)]
    print(f"Loading {NUM_GPUS} model replica(s) on {devices}")
    replicas = load_replicas(MODEL_PATH, devices)

    gpu_locks = [asyncio.Lock() for _ in range(NUM_GPUS)]
    thread_pool = ThreadPoolExecutor(max_workers=NUM_GPUS)

    # Per-GPU memory report
    for i in range(NUM_GPUS):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  cuda:{i}  {used:.1f} / {total:.1f} GB")

    print(f"Server ready. {NUM_GPUS} GPU(s) serving in parallel.")


@app.get("/health")
async def health():
    return {"status": "ok"}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


def _generate_blocking(gpu_id: int, input_ids_cpu, max_new_tokens: int,
                       temperature: float, top_p: float):
    """Run model.generate on a specific GPU. Called from the thread pool."""
    device = f"cuda:{gpu_id}"
    input_ids = input_ids_cpu.to(device, non_blocking=True)
    out = replicas[gpu_id].generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return out.cpu()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    global rr_counter

    messages = [m.model_dump() for m in request.messages]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids_cpu = tokenizer(text, return_tensors="pt").input_ids
    prompt_tokens = input_ids_cpu.shape[1]

    # Pick a GPU round-robin
    async with rr_lock:
        gpu_id = rr_counter % NUM_GPUS
        rr_counter += 1

    # Per-GPU lock: this GPU serves one request at a time. Other GPUs run in
    # parallel. The thread-pool call releases the event loop while the GPU
    # is busy.
    loop = asyncio.get_running_loop()
    async with gpu_locks[gpu_id]:
        generated_ids = await loop.run_in_executor(
            thread_pool,
            _generate_blocking,
            gpu_id, input_ids_cpu, request.max_tokens,
            request.temperature, request.top_p,
        )

    completion_tokens = generated_ids.shape[1]
    finish_reason = "length" if completion_tokens == request.max_tokens else "stop"
    content = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
