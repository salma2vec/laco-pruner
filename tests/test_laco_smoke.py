import pytest
from laco.prune_pipeline import pipeline_prune
from laco.config import LaCoConfig
import os
import torch

def test_pipeline_runs_smoke(tmp_path):
    cfg = LaCoConfig(device="cpu", dtype="float32", model_name="sshleifer/tiny-gpt2", C=2, H=4, T=0.9999)
    few_shot = ["Hello world", "Test sentence"]
    out = str(tmp_path / "pruned.pt")
    report = pipeline_prune(cfg, cfg.model_name, few_shot, partial="full", out_path=out)
    assert "final_layer_count" in report
    assert os.path.exists(out)

