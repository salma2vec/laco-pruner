#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from laco.post_train import fine_tune_pruned_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pruned LaCO model")
    parser.add_argument("--model", type=str, required=True, help="Base model name/path")
    parser.add_argument("--pruned", type=str, required=True, help="Path to pruned state dict")
    parser.add_argument("--train-data", type=str, required=True, help="Training text file (one per line)")
    parser.add_argument("--output", type=str, required=True, help="Output path for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    
    args = parser.parse_args()
    
    # load training texts
    with open(args.train_data) as f:
        train_texts = [line.strip() for line in f if line.strip()]
    
    log.info(f"Loaded {len(train_texts)} training texts")
    
    # run fine-tuning
    result = fine_tune_pruned_model(
        model_path=args.model,
        pruned_state_path=args.pruned,
        train_texts=train_texts,
        output_path=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        max_length=args.max_length,
    )
    
    log.info(f"Fine-tuning complete. Final loss: {result.get('final_loss', 'N/A')}")

if __name__ == "__main__":
    main()


