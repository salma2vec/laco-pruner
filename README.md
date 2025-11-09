# LaCO-Pruner

**LaCO (Layer Collapse)** — Training-free structured pruning by merging adjacent layers in transformer LMs.

This repo provides a reproducible reference implementation of LaCO (RDSC Layer Merge + Layer Collapse workflow),
including attention-only / ffn-only options, few-shot calibration, evaluation on translation tasks, and ablation utilities.

## Highlights

- RDSC layer merge primitives (parameter differencing + merging)
- Few-shot calibration using cosine similarity on hidden representations
- Options: full-layer merge, attention-only merge, ffn-only merge
- Benchmarks: BLEU / chrF / perplexity, speed/memory reporting
- Batched representation extraction for faster calibration
- Post-training fine-tuning script for performance recovery
- Ablation sweep utilities for systematic experimentation
- Support for GPT-2, Llama, Baichuan, Gemma architectures
- Config-driven experiments, run scripts, smoke tests, CI

## Quickstart (toy, CPU-friendly)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run a tiny smoke prune+eval (uses small HF models by default)
bash run_toy_pipeline.sh
```

## Usage Examples

### Basic Pruning

```python
from laco.config import LaCoConfig
from laco.prune_pipeline import pipeline_prune

cfg = LaCoConfig(
    model_name="sshleifer/tiny-gpt2",
    device="cpu",
    C=3, T=0.999
)
few_shot = ["Hello world", "The quick brown fox"]
report = pipeline_prune(cfg, cfg.model_name, few_shot, out_path="outputs/pruned.pt")
```

### Ablation Sweep

```bash
python scripts/run_ablation.py \
    --config configs/laco_toy.yaml \
    --C 2 3 4 \
    --T 0.99 0.995 0.999 \
    --partial full attn ffn \
    --output outputs/ablation_results.yaml
```

### Post-Training Fine-Tuning

```bash
python scripts/run_post_train.py \
    --model sshleifer/tiny-gpt2 \
    --pruned outputs/pruned.pt \
    --train-data train.txt \
    --output outputs/finetuned.pt \
    --epochs 3 \
    --lr 5e-5
```

## Architecture Support

The layer merge utilities support common transformer architectures:
- **GPT-2**: `transformer.h.{i}`
- **Llama/Baichuan/Gemma**: `model.layers.{i}`
- Generic patterns via automatic detection

## Evaluation Metrics

- **BLEU**: Translation quality (sacrebleu)
- **chrF**: Character-level F-score
- **Perplexity**: Language modeling quality
- **Speed**: Tokens/second generation speed
- **Memory**: Peak memory usage (CUDA/CPU)

## Tips for Good Results

- Use 5-20 high-quality calibration sentences representative of your task domain
- Start with conservative T (0.995-0.999), relax to 0.99 for larger pruning ratios
- For large LLMs, attn-only or ffn-only merges can preserve different capabilities — run ablations
- Post-training fine-tuning often recovers performance rapidly (see `scripts/run_post_train.py`)

See `docs/` for algorithm notes, math, and practical tips to reach 25–50% layer pruning with preserved performance.

