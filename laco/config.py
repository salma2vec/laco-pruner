from dataclasses import dataclass
from typing import Optional

@dataclass
class LaCoConfig:
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float16"
    model_name: str = "sshleifer/tiny-gpt2"  # tiny default for smoke; user can swap in Llama/Baichuan/etc.
    few_shot_texts: Optional[str] = None  # path or None: use inline default for toy
    C: int = 2  # number of layers to combine per merge attempt (merge up to C-1 layers into l)
    L: int = 0  # lower layer bound
    H: Optional[int] = None  # upper layer bound (None -> last layer index)
    I: int = 1  # min interval between adjacent merges
    T: float = 0.995  # similarity threshold (cosine), few-shot
    calibration_batch: int = 8
    max_calib_samples: int = 16
    eval_split: str = "test[:32]"
    eval_dataset: str = "wmt16"
    eval_dataset_config: Optional[str] = "de-en"
    eval_max_new_tokens: int = 64
    output_dir: str = "outputs"
    partial_merge: str = "full"  # one of {"full", "attn", "ffn"}

