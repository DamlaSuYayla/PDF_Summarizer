import os, re, gc, logging
from pathlib import Path
from typing import List, Dict

import torch
import fitz
import nltk
import numpy as np
from transformers import (AutoTokenizer, LEDForConditionalGeneration, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback)
from datasets import Dataset
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score

# Import shared utilities from Backend
from Backend import DataCleaner, extract_pdf

# NLTK setup
for pkg in ['punkt', 'punkt_tab']:
    try: nltk.data.find(f'tokenizers/{pkg}')
    except: nltk.download(pkg, quiet=True)

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "allenai/led-base-16384"
MAX_TOKENS = 4096
BASE_DIR = Path(__file__).parent.resolve()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
for d in ["checkpoints", "logs"]:
    (BASE_DIR / d).mkdir(exist_ok=True)


class Metrics:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def calculate(self, preds: List[str], refs: List[str]) -> Dict[str, float]:
        r1, rl = [], []
        for p, r in zip(preds, refs):
            s = self.rouge.score(r, p)
            r1.append(s['rouge1'].fmeasure)
            rl.append(s['rougeL'].fmeasure)
        
        bleu = BLEU().corpus_score(preds, [[r] for r in refs]).score / 100
        try: _, _, f1 = bert_score(preds, refs, lang="en", verbose=False); bs = f1.mean().item()
        except: bs = 0.0
        
        return {'ROUGE-1': np.mean(r1), 'ROUGE-L': np.mean(rl), 'BLEU': bleu, 'BERTScore_F1': bs}


def load_data():
    data = []
    cleaner = DataCleaner()
    base = BASE_DIR / "training_data"
    
    pdf_dir = base / "pdfs" if (base / "pdfs").exists() else base
    sum_dir = base / "summaries" if (base / "summaries").exists() else base
    
    logger.info(f"Looking for PDFs in: {pdf_dir}")
    
    for pdf in pdf_dir.glob("*.pdf"):
        txt = sum_dir / pdf.with_suffix('.txt').name
        if txt.exists():
            text = cleaner.clean(extract_pdf(str(pdf)))
            summary = txt.read_text(encoding='utf-8').strip()
            if text and summary: 
                data.append({'text': text, 'summary': summary})
                logger.info(f"Loaded: {pdf.name}")
    
    if not data: 
        raise ValueError("No data found in training_data/ folder")
    
    split = max(1, int(len(data) * 0.8))
    return data[:split], data[split:] or data[-1:]


def train():
    print("\n" + "="*50)
    print("FINE-TUNING STARTED")
    print(f"Device: {DEVICE.upper()}")
    print(f"FP16: {'Enabled' if torch.cuda.is_available() else 'Disabled'}")
    print("="*50 + "\n")
    
    # Load data
    train_data, eval_data = load_data()
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}\n")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LEDForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    
    def tokenize(ex):
        inp = tokenizer(ex['text'], max_length=MAX_TOKENS, truncation=True, padding='max_length')
        tgt = tokenizer(ex['summary'], max_length=200, truncation=True, padding='max_length')
        inp['labels'] = tgt['input_ids']
        # LED modeli için global attention mask - ilk token'a global attention
        inp['global_attention_mask'] = [[1] + [0] * (len(ids) - 1) for ids in inp['input_ids']]
        return inp
    
    train_ds = Dataset.from_list(train_data).map(tokenize, batched=True, remove_columns=['text','summary'])
    eval_ds = Dataset.from_list(eval_data).map(tokenize, batched=True, remove_columns=['text','summary'])
    
    args = Seq2SeqTrainingArguments(
        output_dir=str(BASE_DIR / "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),  # Only use pin_memory with GPU
        logging_dir=str(BASE_DIR / "logs"),
        save_total_limit=2,
        report_to=["tensorboard"]
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    
    # Clean up before saving to avoid Windows memory-mapped file error
    # Get state dict and tokenizer reference before deleting trainer
    import copy
    import time
    
    # Get model state dict and config while trainer still exists
    model_state_dict = copy.deepcopy(trainer.model.cpu().state_dict())
    model_config = trainer.model.config
    
    # Delete trainer and clear all references
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Wait for Windows to release file handles
    time.sleep(2)
    
    # Create a fresh model and load the state dict
    path = str(BASE_DIR / "checkpoints" / "final_model")
    
    # Use a fresh model instance to save
    fresh_model = LEDForConditionalGeneration.from_pretrained(MODEL_NAME)
    fresh_model.load_state_dict(model_state_dict)
    
    # Save with safe_serialization=False to avoid safetensors issues on Windows
    fresh_model.save_pretrained(path, safe_serialization=False)
    tokenizer.save_pretrained(path)
    
    # Clean up fresh model
    del fresh_model, model_state_dict
    gc.collect()
    print(f"\n{'='*50}")
    print(f"Model saved to: {path}")
    
    # Evaluate
    print("\nCalculating metrics...")
    from Backend import Summarizer
    summarizer = Summarizer(path)
    preds = [summarizer.summarize(d['text']) for d in eval_data]
    refs = [d['summary'] for d in eval_data]
    metrics = Metrics().calculate(preds, refs)
    
    print("\n" + "="*50)
    print("METRICS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("="*50)
    
    # Save metrics to file
    import json
    from datetime import datetime
    metrics_file = BASE_DIR / "metrics.json"
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['model_path'] = path
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics


if __name__ == "__main__":
    train()
