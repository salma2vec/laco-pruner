import yaml
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

from laco.config import LaCoConfig
from laco.prune_pipeline import pipeline_prune
from bench.benchmark import run_eval

log = logging.getLogger(__name__)

@dataclass
class AblationConfig:
    C: int
    I: int
    T: float
    partial: str  # "full", "attn", "ffn"
    name: str = ""  # optional name for this config
    
    def __post_init__(self):
    def __post_init__(self):
            self.name = f"C{self.C}_I{self.I}_T{self.T:.3f}_{self.partial}"

def generate_ablation_grid(
def generate_ablation_grid(
    I_values: List[int] = [1, 2],
    T_values: List[float] = [0.99, 0.995, 0.999],
    partial_modes: List[str] = ["full", "attn", "ffn"]
) -> List[AblationConfig]:
    configs = []
    for C in C_values:
        for I in I_values:
            for T in T_values:
                for partial in partial_modes:
                    configs.append(AblationConfig(C=C, I=I, T=T, partial=partial))
    return configs

def run_ablation_sweep(
def run_ablation_sweep(
    few_shot_texts: List[str],
    ablation_configs: List[AblationConfig],
    output_dir: str = "outputs/ablations"
) -> List[Dict[str, Any]]:
    # runs ablation sweep, returns list of results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    
    for i, abl_cfg in enumerate(ablation_configs):
        log.info(f"Running ablation {i+1}/{len(ablation_configs)}: {abl_cfg.name}")
        
        # create config for this run
        cfg = LaCoConfig(**asdict(base_cfg))
        cfg.C = abl_cfg.C
        cfg.I = abl_cfg.I
        cfg.T = abl_cfg.T
        
        # run pruning
        out_path = f"{output_dir}/pruned_{abl_cfg.name}.pt"
        try:
            prune_report = pipeline_prune(
                cfg,
                cfg.model_name,
                few_shot_texts,
                partial=abl_cfg.partial,
                out_path=out_path
            )
            
            # run evaluation
            eval_results = run_eval(cfg, cfg.model_name, pruned_state_path=out_path)
            
            results.append({
                "config": asdict(abl_cfg),
                "prune_report": prune_report,
                "eval": eval_results,
            })
            
            log.info(f"Completed {abl_cfg.name}: {len(prune_report['accepted_merges'])} merges accepted")
            
        except Exception as e:
            log.error(f"Failed ablation {abl_cfg.name}: {e}")
            results.append({
                "config": asdict(abl_cfg),
                "error": str(e)
            })
    
    return results

def save_ablation_results(results: List[Dict[str, Any]], output_path: str):
def save_ablation_results(results: List[Dict[str, Any]], output_path: str):
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    log.info(f"Saved ablation results to {output_path}")



