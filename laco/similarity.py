from typing import List, Optional, Dict, Tuple
import torch
from torch.nn.functional import cosine_similarity
import logging
import time

log = logging.getLogger(__name__)

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
    
    # batch processing for speed
    if batch_size > 1 and len(texts) > batch_size:
        return _extract_representations_batched(model, tokenizer, texts, device, max_length, batch_size, use_kv_cache)
    
    # original single-item processing
    reps = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc.input_ids.to(device)
        with torch.no_grad():
            if use_kv_cache:
                out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
            else:
                out = model(input_ids=input_ids, output_hidden_states=True)
            # get last layer hidden states
            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                hidden_states = out.hidden_states[-1]
            elif hasattr(out, 'last_hidden_state'):
                hidden_states = out.last_hidden_state
            else:
                hidden_states = out.logits
            last = hidden_states[:, -1, :].squeeze(0).detach().cpu()
            reps.append(last)
    return torch.stack(reps, dim=0)  # (N, D)

def _extract_representations_batched(
    model, tokenizer, texts: List[str], device: str, 
    max_length: int, batch_size: int, use_kv_cache: bool
) -> torch.Tensor:
    # batched version for gpu
    model.to(device)
    model.eval()
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
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # get last layer hidden states
            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                hidden_states = out.hidden_states[-1]
            elif hasattr(out, 'last_hidden_state'):
                hidden_states = out.last_hidden_state
            else:
                hidden_states = out.logits
            # get last non-padded token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_reps = hidden_states[torch.arange(len(batch_texts)), seq_lengths].detach().cpu()
            reps.append(batch_reps)
    
    return torch.cat(reps, dim=0)



