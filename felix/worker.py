"""GPU worker process for multi-process DP inference.

Each worker owns one GPU and one model replica.
Receives requests via a queue, returns results via a result queue.
"""

import os
import sys
import torch
from multiprocessing import Process, Queue
from dataclasses import dataclass

# Ensure felix/ is on the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import InferenceEngine


@dataclass
class WorkRequest:
    """A generation request sent to a worker."""
    request_id: int
    input_ids_bytes: bytes  # serialized tensor
    input_shape: tuple
    max_tokens: int
    temperature: float
    top_p: float


@dataclass
class BatchWorkRequest:
    """A batch of generation requests sent to a worker."""
    request_ids: list
    input_ids_bytes_list: list  # list of serialized tensors
    input_shapes: list
    max_tokens_list: list
    temperature_list: list
    top_p_list: list


@dataclass
class WorkResult:
    """Result from a worker."""
    request_id: int
    token_ids_bytes: bytes  # serialized tensor
    token_count: int
    hit_eos: bool


def worker_loop(gpu_id: int, model_path: str, request_queue: Queue, result_queue: Queue, physical_gpu: int = None):
    """Main loop for a GPU worker process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu if physical_gpu is not None else gpu_id)

    # Load model on this GPU (visible as cuda:0 due to CUDA_VISIBLE_DEVICES)
    engine = InferenceEngine(model_path, num_gpus=1, compile_model=False)
    engine.load()

    result_queue.put(("ready", gpu_id))

    while True:
        item = request_queue.get()

        if item is None:  # shutdown signal
            break

        if isinstance(item, BatchWorkRequest):
            # Deserialize inputs
            input_ids_list = []
            for raw, shape in zip(item.input_ids_bytes_list, item.input_shapes):
                t = torch.frombuffer(bytearray(raw), dtype=torch.long).reshape(shape)
                input_ids_list.append(t)

            results = engine.generate_batch(
                0,  # always cuda:0 due to CUDA_VISIBLE_DEVICES
                input_ids_list,
                item.max_tokens_list,
                item.temperature_list,
                item.top_p_list,
            )

            for req_id, gen_result in zip(item.request_ids, results):
                result_queue.put(WorkResult(
                    request_id=req_id,
                    token_ids_bytes=gen_result.token_ids.numpy().tobytes(),
                    token_count=len(gen_result.token_ids),
                    hit_eos=gen_result.hit_eos,
                ))

        elif isinstance(item, WorkRequest):
            t = torch.frombuffer(bytearray(item.input_ids_bytes), dtype=torch.long).reshape(item.input_shape)
            gen_result = engine.generate(0, t, item.max_tokens, item.temperature, item.top_p)

            result_queue.put(WorkResult(
                request_id=item.request_id,
                token_ids_bytes=gen_result.token_ids.numpy().tobytes(),
                token_count=len(gen_result.token_ids),
                hit_eos=gen_result.hit_eos,
            ))


def start_worker(gpu_id: int, model_path: str, request_queue: Queue, result_queue: Queue, physical_gpu: int = None) -> Process:
    """Spawn a worker process for the given GPU."""
    p = Process(target=worker_loop, args=(gpu_id, model_path, request_queue, result_queue, physical_gpu), daemon=True)
    p.start()
    return p
