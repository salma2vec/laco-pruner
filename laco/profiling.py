import torch
import time
import psutil
import os
from typing import Dict, Any, Optional
import logging

log = logging.getLogger(__name__)

# Constant for byte-to-megabyte conversion
_BYTES_TO_MB = 1024 ** 2

def get_memory_usage(device: str = "cpu") -> Dict[str, float]:
    # returns memory usage in MB
    if device.startswith("cuda") and torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / _BYTES_TO_MB,
            "reserved_mb": torch.cuda.memory_reserved() / _BYTES_TO_MB,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / _BYTES_TO_MB,
        }
    else:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / _BYTES_TO_MB,
            "vms_mb": mem_info.vms / _BYTES_TO_MB,
        }

def profile_generation_speed(
    model, 
    tokenizer, 
    texts: list, 
    max_new_tokens: int = 64,
    device: str = "cpu",
    warmup: int = 2
) -> Dict[str, Any]:
    # measures generation speed and memory usage
    model.to(device)
    model.eval()
    
    # warmup
    for _ in range(warmup):
        dummy_text = texts[0] if texts else "Hello"
        with torch.no_grad():
            _ = model.generate(
                **tokenizer(dummy_text, return_tensors="pt").to(device),
                max_new_tokens=max_new_tokens
            )
    
    # clear CUDA cache if on GPU
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    mem_before = get_memory_usage(device)
    
    # actual timing
    total_tokens = 0
    start_time = time.time()
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            # count generated tokens (excluding input)
            total_tokens += (output.shape[1] - inputs.input_ids.shape[1])
    
    elapsed = time.time() - start_time
    mem_after = get_memory_usage(device)
    
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
    
    return {
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed,
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
    }



