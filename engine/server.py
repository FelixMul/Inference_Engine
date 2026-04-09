"""Data-parallel inference server for Qwen3.5-35B-A3B.

Loads one full model copy per GPU (data parallel). Incoming requests are
dispatched round-robin across GPUs, giving true parallelism at high
concurrency.

Usage:
    python engine/server.py [--port 8000] [--model-path /dev/shm/Qwen3.5-35B-A3B]
"""

import argparse
import time
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Request / response schemas
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
# Globals
# ---------------------------------------------------------------------------

app = FastAPI()
models: list = []          # one model per GPU
tokenizer = None
gpu_semaphores: list = []  # one semaphore per GPU to serialize per-GPU generation
request_counter = 0        # for round-robin dispatch
counter_lock = asyncio.Lock()
thread_pool: ThreadPoolExecutor = None

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_on_gpu(gpu_id: int, input_ids: torch.Tensor, gen_kwargs: dict) -> torch.Tensor:
    """Run generation on a specific GPU. Called from thread pool."""
    model = models[gpu_id]
    ids = input_ids.to(f"cuda:{gpu_id}")
    with torch.no_grad():
        output = model.generate(ids, **gen_kwargs)
    return output[0].cpu()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global request_counter

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    # Generation kwargs
    gen_kwargs = dict(
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    if req.temperature > 0:
        gen_kwargs["temperature"] = req.temperature
        gen_kwargs["top_p"] = req.top_p

    # Round-robin GPU assignment
    async with counter_lock:
        gpu_id = request_counter % len(models)
        request_counter += 1

    # Acquire per-GPU semaphore (serializes generation on each GPU)
    async with gpu_semaphores[gpu_id]:
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            thread_pool, generate_on_gpu, gpu_id, input_ids, gen_kwargs
        )

    new_tokens = output[prompt_len:]
    content = tokenizer.decode(new_tokens, skip_special_tokens=True)
    completion_tokens = len(new_tokens)

    # Determine finish reason
    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    hit_eos = len(new_tokens) > 0 and int(new_tokens[-1]) in eos_ids
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

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_models(model_path: str, num_gpus: int):
    global models, tokenizer, gpu_semaphores, thread_pool

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    num_gpus = min(num_gpus, torch.cuda.device_count())
    print(f"Loading {num_gpus} model replicas (one per GPU, bf16)...")

    t0 = time.time()
    for i in range(num_gpus):
        print(f"  Loading model on cuda:{i}...")
        m = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=f"cuda:{i}",
        )
        m.eval()
        models.append(m)

    gpu_semaphores = [asyncio.Semaphore(1) for _ in range(num_gpus)]
    thread_pool = ThreadPoolExecutor(max_workers=num_gpus)

    print(f"All {num_gpus} replicas loaded in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="DP inference server for Qwen3.5-35B-A3B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-path", default="/dev/shm/Qwen3.5-35B-A3B")
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()

    load_models(args.model_path, args.num_gpus)
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
