from typing import Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import logging

log = logging.getLogger(__name__)

def load_model_tokenizer(model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float16) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    # loads model and tokenizer, sets pad_token if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    log.info(f"Loaded model {model_name} on {device} dtype={dtype}")
    return model, tokenizer



