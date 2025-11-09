from typing import Dict, List, Tuple
import torch
import re
import copy
import logging

log = logging.getLogger(__name__)

def find_layer_prefixes(state_dict: Dict[str, torch.Tensor]) -> Tuple[str, int]:
    # detects layer naming pattern and counts layers
    # try known patterns first (faster and more reliable)
    known_patterns = [
        ("model.layers", r"^model\.layers\.(\d+)\.", "model.layers"),
        ("transformer.h", r"^transformer\.h\.(\d+)\.", "transformer.h"),
        ("model.transformer.h", r"^model\.transformer\.h\.(\d+)\.", "model.transformer.h"),
    ]
    
    for name, pattern, base in known_patterns:
        matches = [k for k in state_dict.keys() if re.match(pattern, k)]
        if matches:
            # extract max layer index
            indices = set()
            for k in matches:
                m = re.match(pattern, k)
                if m:
                    indices.add(int(m.group(1)))
            if indices:
                num_layers = max(indices) + 1
                log.info(f"Detected {name} pattern with {num_layers} layers")
                return base, num_layers
    
    # fallback: generic pattern matching
    layer_keys = [k for k in state_dict.keys() if re.search(r"\.([0-9]+)\.", k)]
    if not layer_keys:
        raise ValueError("Cannot detect layer pattern in state_dict")

    # extract prefix + layer index
    examples = {}
    for k in layer_keys:
        m = re.search(r"^(.*)\.([0-9]+)\.(.*)$", k)
        if m:
            base, idx, tail = m.groups()
            idx = int(idx)
            examples.setdefault(base, set()).add(idx)

    # pick the base with largest index set
    base, idxs = max(examples.items(), key=lambda x: len(x[1]))
    num_layers = max(idxs) + 1
    log.info(f"Detected generic layer pattern '{base}' with {num_layers} layers")
    return base, num_layers

def partition_keys_by_layer(state_dict: Dict[str, torch.Tensor], layer_base: str, num_layers: int) -> Dict[int, List[str]]:
    per_layer = {i: [] for i in range(num_layers)}
    for k in state_dict.keys():
        m = re.match(rf"^{re.escape(layer_base)}\.([0-9]+)\.(.*)$", k)
        if m:
            i = int(m.group(1))
            per_layer[i].append(k)
    return per_layer

def rdscl_merge_state_dict(
    state_dict: Dict[str, torch.Tensor],
    layer_base: str,
    num_layers: int,
    l: int,
    K: int,
    partial: str = "full"
) -> Dict[str, torch.Tensor]:
    # merges layers l+1..l+K into layer l using RDSC formula
    # partial: "full", "attn", or "ffn"
    new_sd = copy.deepcopy(state_dict)

    # build per-layer key map
    per_layer = partition_keys_by_layer(state_dict, layer_base, num_layers)

    # keys in layer l and l+1..l+K
    target_keys = set(per_layer[l])
    merge_keys = []
    for k in range(1, K+1):
        idx = l + k
        if idx >= num_layers:
            break
        merge_keys.extend(per_layer[idx])

    # helper to categorize keys as attn/ffn/other
    # covers most common architectures: GPT-2, Llama, Baichuan, Gemma, etc.
    def key_category(key: str) -> str:
        key_lower = key.lower()
        # attention patterns
        attn_patterns = (
            "q_proj", "k_proj", "v_proj", "qkv", 
            ".attn.", "self_attn", "attention",
            "o_proj", "out_proj", "attn.c_attn",  # GPT-2 style
            "qkv_proj", "query", "key", "value"   # some variants
        )
        if any(seg in key_lower for seg in attn_patterns):
            return "attn"
        # FFN/MLP patterns  
        ffn_patterns = (
            "mlp", "feed_forward", "feedforward",
            "fc_in", "fc_out", "fc1", "fc2",
            "up_proj", "down_proj", "gate_proj",  # Llama style
            ".mlp.", "c_fc", "c_proj",            # GPT-2 style
            "intermediate", "output"               # some BERT-like
        )
        if any(seg in key_lower for seg in ffn_patterns):
            return "ffn"
        return "other"

    # For each parameter in l, compute merged tensor
    for k0 in per_layer[l]:
        cat = key_category(k0)
        if partial == "attn" and cat != "attn":
            continue
        if partial == "ffn" and cat != "ffn":
            continue

        # compute sum of corresponding keys at layers l+k that match the tail of k0
        m = re.match(rf"^{re.escape(layer_base)}\.{l}\.(.+)$", k0)
        if not m:
            continue
        tail = m.group(1)

        # find corresponding keys at later layers
        gather = []
        for k in range(1, K+1):
            tok = f"{layer_base}.{l+k}.{tail}"
            if tok in state_dict:
                gather.append(state_dict[tok])

        if not gather:
            # nothing to merge for this key
            continue

        theta_l = state_dict[k0]
        sum_follow = sum(gather)

        # RDSC formula: theta_l_star = theta_l + sum(theta_{l+k} - theta_l) = theta_l + sum_follow - len(gather)*theta_l
        theta_star = theta_l + sum_follow - len(gather) * theta_l
        new_sd[k0] = theta_star

    # Now remove merged layers' keys (l+1 .. l+K)
    for k in range(1, K+1):
        idx = l + k
        if idx >= num_layers:
            break
        for key in per_layer[idx]:
            if key in new_sd:
                new_sd.pop(key)

    return new_sd



