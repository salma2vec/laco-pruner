from typing import List, Optional, Dict, Tuple
import torch
from torch.nn.functional import cosine_similarity
import logging
import time

log = logging.getLogger(__name__)

def _get_hidden_states(model_output):
    """Extract hidden states from model output in a consistent way."""
    if hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:
        return model_output.hidden_states[-1]
    elif hasattr(model_output, 'last_hidden_state'):
        return model_output.last_hidden_state
    else:
        return model_output.logits

def avg_cosine_similarity(reprs_a: torch.Tensor, reprs_b: torch.Tensor) -> float:
    assert reprs_a.shape == reprs_b.shape
    with torch.no_grad():
        sims = cosine_similarity(reprs_a, reprs_b, dim=1)
        return float(sims.mean().cpu().item())

def extract_representations(
    model,
    tokenizer, 
    texts: List[str], 
    device: str = "cpu", 
    dtype=torch.float32, 
    max_length: int = 256,
    batch_size: int = 1,
    use_kv_cache: bool = False
):
    # returns (N, hidden_dim) tensor of last-layer hidden states
    model.to(device)
    model.eval()
    
    # Use batch processing when beneficial (more than 1 text and batch_size > 0)
    if batch_size > 1 and len(texts) > 1:
        return _extract_representations_batched(model, tokenizer, texts, device, max_length, batch_size, use_kv_cache)
    
    # Single-item processing for very small inputs or batch_size=1
    reps = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc.input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=use_kv_cache, output_hidden_states=True)
            # Extract hidden states using helper
            hidden_states = _get_hidden_states(out)
            last = hidden_states[:, -1, :].squeeze(0).detach().cpu()
            reps.append(last)
    return torch.stack(reps, dim=0)  # (N, D)

def _extract_representations_batched(
    model, tokenizer, texts: List[str], device: str, 
    max_length: int, batch_size: int, use_kv_cache: bool
) -> torch.Tensor:
    # batched version for faster processing
    # Note: model is already on device and in eval mode from caller
    reps = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # tokenize batch
        encodings = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, 
                       use_cache=use_kv_cache, output_hidden_states=True)
            # Extract hidden states using helper
            hidden_states = _get_hidden_states(out)
            # get last non-padded token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_reps = hidden_states[torch.arange(len(batch_texts)), seq_lengths].detach().cpu()
            reps.append(batch_reps)
    
    return torch.cat(reps, dim=0)



