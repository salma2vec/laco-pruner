from typing import List, Dict, Any, Optional
import torch
from datasets import load_dataset
from sacrebleu import corpus_bleu, corpus_chrf
from laco.model_utils import load_model_tokenizer
from laco.profiling import profile_generation_speed
import math
import logging

log = logging.getLogger(__name__)

def _batch_generate(model, tokenizer, texts: List[str], max_new_tokens: int, device: str, batch_size: int = 4) -> List[str]:
    """Generate predictions in batches for better performance."""
    predictions = []
    model.eval()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            # Tokenize batch with padding
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            # Generate for batch
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
            # Decode each output
            for out in outputs:
                predictions.append(tokenizer.decode(out.tolist(), skip_special_tokens=True))
    
    return predictions

def compute_bleu(preds: List[str], refs: List[str]) -> float:
    return corpus_bleu(preds, [refs]).score

def compute_chrf_score(preds: List[str], refs: List[str]) -> Optional[float]:
    # computes chrF using sacrebleu
    try:
        return corpus_chrf(preds, [refs]).score
    except Exception as e:
        log.warning(f"Failed to compute chrF: {e}")
        return None

def compute_ppl(model, tokenizer, texts: List[str], device="cpu") -> float:
    model.eval()
    total_ll = 0.0
    total_tokens = 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.to(device)
        labels = input_ids.clone()
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
            # loss is avg per token
            loss = out.loss.item()
            n_tokens = input_ids.numel()
            total_ll += loss * n_tokens
            total_tokens += n_tokens
    if total_tokens == 0:
        return float("nan")
    avg_loss = total_ll / total_tokens
    return math.exp(avg_loss)

def run_eval(
    cfg, 
    model_name: str, 
    pruned_state_path: str=None,
    include_speed: bool = True
) -> Dict[str, Any]:
    # evaluates baseline vs pruned model, returns metrics
    ds = load_dataset(cfg.eval_dataset, cfg.eval_dataset_config, split=cfg.eval_split)
    # assume 'translation' column exists like wmt16 de-en
    sources = [ex["translation"]["de"] for ex in ds]
    targets = [ex["translation"]["en"] for ex in ds]

    # load original model
    model, tokenizer = load_model_tokenizer(model_name, device=cfg.device, dtype=getattr(torch, cfg.dtype))

    # baseline generation
    log.info("Evaluating baseline model...")
    batch_size = getattr(cfg, 'eval_batch_size', 4)
    preds_base = _batch_generate(model, tokenizer, sources, cfg.eval_max_new_tokens, cfg.device, batch_size)

    base_bleu = compute_bleu(preds_base, targets)
    base_chrf = compute_chrf_score(preds_base, targets)
    base_ppl = compute_ppl(model, tokenizer, targets, device=cfg.device)
    
    base_speed = None
    if include_speed:
        try:
            base_speed = profile_generation_speed(model, tokenizer, sources[:8], cfg.eval_max_new_tokens, cfg.device)
        except Exception as e:
            log.warning(f"Speed profiling failed: {e}")

    # pruned model if exists
    pruned_bleu = None
    pruned_chrf = None
    pruned_ppl = None
    pruned_speed = None
    if pruned_state_path:
        log.info("Evaluating pruned model...")
        pruned_model, _ = load_model_tokenizer(model_name, device=cfg.device, dtype=getattr(torch, cfg.dtype))
        # load state dict
        sd = torch.load(pruned_state_path, map_location="cpu")
        pruned_model.load_state_dict(sd, strict=False)
        pruned_model.to(cfg.device)

        preds_p = _batch_generate(pruned_model, tokenizer, sources, cfg.eval_max_new_tokens, cfg.device, batch_size)

        pruned_bleu = compute_bleu(preds_p, targets)
        pruned_chrf = compute_chrf_score(preds_p, targets)
        pruned_ppl = compute_ppl(pruned_model, tokenizer, targets, device=cfg.device)
        
        if include_speed:
            try:
                pruned_speed = profile_generation_speed(pruned_model, tokenizer, sources[:8], cfg.eval_max_new_tokens, cfg.device)
            except Exception as e:
                log.warning(f"Speed profiling failed: {e}")

    result = {
        "base_bleu": base_bleu,
        "base_chrf": base_chrf,
        "base_ppl": base_ppl,
        "pruned_bleu": pruned_bleu,
        "pruned_chrf": pruned_chrf,
        "pruned_ppl": pruned_ppl,
    }
    
    if base_speed:
        result["base_speed"] = base_speed
    if pruned_speed:
        result["pruned_speed"] = pruned_speed
    
    return result



