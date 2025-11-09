#!/usr/bin/env python3
import argparse
import yaml
import logging
from pathlib import Path

from laco.config import LaCoConfig
from laco.ablation import generate_ablation_grid, run_ablation_sweep, save_ablation_results

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run LaCO ablation sweep")
    parser.add_argument("--config", type=str, required=True, help="Base config YAML file")
    parser.add_argument("--output", type=str, default="outputs/ablation_results.yaml", help="Output path for results")
    parser.add_argument("--C", type=int, nargs="+", default=[2, 3, 4], help="C values to sweep")
    parser.add_argument("--I", type=int, nargs="+", default=[1, 2], help="I values to sweep")
    parser.add_argument("--T", type=float, nargs="+", default=[0.99, 0.995, 0.999], help="T values to sweep")
    parser.add_argument("--partial", type=str, nargs="+", default=["full", "attn", "ffn"], help="Partial merge modes")
    parser.add_argument("--few-shot-file", type=str, help="Path to few-shot texts file (one per line)")
    
    args = parser.parse_args()
    
    # load base config
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    base_cfg = LaCoConfig(**cfg_dict)
    
    # load few-shot texts
    if args.few_shot_file:
        with open(args.few_shot_file) as f:
            few_shot_texts = [line.strip() for line in f if line.strip()]
    else:
        # default toy texts
        few_shot_texts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "Translate this sentence to English.",
            "Machine learning is fascinating.",
        ]
    
    log.info(f"Loaded {len(few_shot_texts)} few-shot texts")
    
    # generate ablation grid
    ablation_configs = generate_ablation_grid(
        C_values=args.C,
        I_values=args.I,
        T_values=args.T,
        partial_modes=args.partial
    )
    
    log.info(f"Generated {len(ablation_configs)} ablation configurations")
    
    # run sweep
    results = run_ablation_sweep(
        base_cfg,
        few_shot_texts,
        ablation_configs,
        output_dir="outputs/ablations"
    )
    
    # save results
    save_ablation_results(results, args.output)
    log.info(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()


