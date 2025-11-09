#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/laco_toy.yaml}

python - <<PY
import yaml, pprint
from laco.config import LaCoConfig
from laco.prune_pipeline import pipeline_prune
from bench.benchmark import run_eval

cfg_dict = yaml.safe_load(open("${CONFIG}"))
cfg = LaCoConfig(**cfg_dict)

# toy few-shot texts
few_shot = ["Hello world", "The quick brown fox jumps over the lazy dog", "Translate this sentence to English."]

res = pipeline_prune(cfg, cfg.model_name, few_shot, partial=cfg.partial_merge, out_path="outputs/pruned_toy.pt")

print("PRUNE REPORT:")
pprint.pprint(res)

print("EVAL:")
print(run_eval(cfg, cfg.model_name, pruned_state_path=res["pruned_state_path"]))
PY

