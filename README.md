# LaCO-Pruner

**LaCO (Layer Collapse)** — Training-free structured pruning by merging adjacent layers in transformer LMs.

This repo provides a reproducible reference implementation of LaCO (RDSC Layer Merge + Layer Collapse workflow),
including attention-only / ffn-only options, few-shot calibration, evaluation on translation tasks, and ablation utilities.

## Highlights

- RDSC layer merge primitives (parameter differencing + merging).
- Few-shot calibration using cosine similarity on hidden representations.
- Options: full-layer merge, attention-only merge, ffn-only merge.
- Benchmarks: BLEU / chrF / perplexity, speed/memory reporting.
- Config-driven experiments, run script, smoke tests, CI.

## Quickstart (toy, CPU-friendly)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run a tiny smoke prune+eval (uses small HF models by default)
bash run_toy_pipeline.sh
```

See `docs/` for algorithm notes, math, and practical tips to reach 25–50% layer pruning with preserved performance.

