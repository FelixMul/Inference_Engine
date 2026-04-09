import asyncio
import time
import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/data/GPUINFERENCE/Qwen3.5-35B-A3B"

app = FastAPI()
tokenizer = None
model = None
lock = asyncio.Lock()


@app.on_event("startup")
async def startup():
    global tokenizer, model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model ready.")


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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    prompt_tokens = input_ids.shape[1]

    do_sample = request.temperature > 0.0

    async with lock:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                do_sample=do_sample,
                temperature=request.temperature if do_sample else None,
                top_p=request.top_p if do_sample else None,
            )

    generated = output_ids[0][prompt_tokens:]
    finish_reason = "length" if len(generated) == request.max_tokens else "stop"
    content = tokenizer.decode(generated, skip_special_tokens=True)
    completion_tokens = len(generated)

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
