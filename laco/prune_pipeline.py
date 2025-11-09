from typing import Dict, Any, List, Tuple, Optional
import torch
import os
import logging
from laco.model_utils import load_model_tokenizer
from laco.layer_merge import find_layer_prefixes, rdscl_merge_state_dict
from laco.similarity import extract_representations, avg_cosine_similarity

log = logging.getLogger(__name__)

def load_state_dict(model):
def load_state_dict(model):

def pipeline_prune(
def pipeline_prune(
    model_name: str,
    few_shot_texts: List[str],
    partial: str = "full",
    out_path: str = "outputs/pruned.pt"
) -> Dict[str, Any]:
    device = cfg.device
    dtype = getattr(torch, cfg.dtype)

    model, tokenizer = load_model_tokenizer(model_name, device=device, dtype=dtype)
    state = load_state_dict(model)
    layer_base, num_layers = find_layer_prefixes(state)
    log.info(f"Detected layer base {layer_base} with {num_layers} layers")

    # build initial parameters
    L = cfg.L
    H = cfg.H if cfg.H is not None else num_layers - 1
    C = cfg.C
    I = cfg.I
    T = cfg.T

    l = H - C
    M_star_state = state
    accepted_merges = []

    while l >= L:
        K = min(C - 1, (num_layers - 1) - l)  # number of following layers we can merge
        if K <= 0:
            l -= 1
            continue

        log.info(f"Trying merge at layer {l} with K={K}")

        tmp_state = rdscl_merge_state_dict(M_star_state, layer_base, num_layers, l, K, partial=partial)

        # load tmp into a clone of model to evaluate representations
        tmp_model = model.__class__.from_config(model.config)
        tmp_model.load_state_dict({k: v.clone() for k, v in tmp_state.items()}, strict=False)

        # extract reps (use batching if we have many texts)
        batch_size = getattr(cfg, 'calibration_batch', 1)
        reps_tmp = extract_representations(tmp_model, tokenizer, few_shot_texts, device=device, batch_size=batch_size)
        reps_orig = extract_representations(model, tokenizer, few_shot_texts, device=device, batch_size=batch_size)

        s = avg_cosine_similarity(reps_tmp, reps_orig)
        log.info(f"Similarity s={s:.6f} threshold={T}")

        if s > T:
            log.info("Accepting merge")
            M_star_state = tmp_state
            accepted_merges.append((l, K, s))
            # after merging, number of layers decreased; recompute num_layers & reset pointer if needed
            # recompute layer count by scanning keys
            # naive recompute:
            _, num_layers = find_layer_prefixes(M_star_state)
            # move l down by interval
            l = max(L, l - I)
            # ensure pointer within new num_layers
            if l > num_layers - 1:
                l = num_layers - C
        else:
            log.info("Rejecting merge")
            l -= 1

    # save state
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(M_star_state, out_path)

    report = {
        "pruned_state_path": out_path,
        "accepted_merges": accepted_merges,
        "final_layer_count": find_layer_prefixes(M_star_state)[1]
    }
    return report



