import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0)
        }

def fine_tune_pruned_model(
    model_path: str,
    pruned_state_path: str,
    train_texts: List[str],
    output_path: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    device: str = "cuda",
    max_length: int = 512,
    warmup_steps: int = 100,
    save_steps: int = 500,
) -> Dict[str, Any]:
    # fine-tunes pruned model, saves to output_path
    log.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # load pruned state
    log.info(f"Loading pruned weights from {pruned_state_path}")
    pruned_state = torch.load(pruned_state_path, map_location="cpu")
    model.load_state_dict(pruned_state, strict=False)
    
    model.to(device)
    model.train()
    
    # setup dataset
    dataset = TextDataset(train_texts, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # training loop
    global_step = 0
    losses = []
    
    for epoch in range(num_epochs):
        log.info(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # backward
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            losses.append(loss.item())
            global_step += 1
            
            # save checkpoint
            if global_step % save_steps == 0:
                checkpoint_path = output_path.replace(".pt", f"_step{global_step}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                log.info(f"Saved checkpoint at step {global_step}")
        
        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    # final save
    log.info(f"Saving fine-tuned model to {output_path}")
    torch.save(model.state_dict(), output_path)
    
    return {
        "output_path": output_path,
        "final_loss": losses[-1] if losses else None,
        "avg_loss": sum(losses) / len(losses) if losses else None,
        "total_steps": global_step,
    }



