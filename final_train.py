# -*- coding: utf-8 -*-
"""final_train.py

MINT-Pat full MLM continued pre-training script.
Runs on Google Colab Pro+ (A100 40GB). Upload train.txt and valid.txt
to Google Drive before running. Expected runtime: ~2–3 hours/epoch.

Validate the pipeline with mint_pat_smoke_test.py on a T4 first.
"""

from google.colab import drive
drive.mount('/content/drive')

CONFIG = {
    # --- Paths ---
    # Where your preprocessed MLM files live in Google Drive
    "train_file": "/content/drive/MyDrive/train.txt",
    "valid_file": "/content/drive/MyDrive/valid.txt",

    # Where to save the trained model and checkpoints
    "output_dir": "/content/drive/MyDrive/xlm_roberta_patent",

    # --- Base model ---
    # This downloads automatically from HuggingFace on first run (~1GB)
    "base_model": "xlm-roberta-base",

    # --- Training hyperparameters (from Chapter 3 methodology) ---
    "max_length":        512,    # XLM-RoBERTa's maximum token length
    "mlm_probability":   0.15,   # Standard BERT masking rate
    "learning_rate":     2e-5,   # Low LR to avoid catastrophic forgetting
    "weight_decay":      0.01,
    "warmup_ratio":      0.06,   # 6% of steps used for LR warmup
    "num_epochs":        3,
    "batch_size":        16,     # Per-device batch on A100 (40GB)
    "grad_accum_steps":  16,     # Effective batch = 256
    "seed":              42,

    # --- Logging ---
    "logging_steps":     500,    # Print loss every 500 steps
    "save_strategy":     "epoch" # Save a checkpoint after each epoch
}

import os
import torch

print("=" * 60)
print("MINT-Pat Training Setup Check")
print("=" * 60)

# GPU
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU:        {gpu}")
    print(f"VRAM:       {vram:.1f} GB")
    if vram < 35:
        print(f"WARNING: {vram:.1f} GB VRAM detected — A100 (40GB) required for full training.")
        print("Reduce batch_size or switch runtime to A100.")
else:
    print("WARNING: No GPU found. Training will be extremely slow.")
    print("Go to Runtime > Change runtime type > A100 GPU and try again.")

# Data files
for key in ("train_file", "valid_file"):
    path = CONFIG[key]
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / 1e6
        with open(path) as f:
            lines = sum(1 for _ in f)
        print(f"{key}:  {lines:,} lines  ({size_mb:.0f} MB)  ✓")
    else:
        print(f"{key}: NOT FOUND at {path}")
        print("  → Upload the file to Drive and update CONFIG above.")

print(f"\nBase model: {CONFIG['base_model']}")
print(f"Output dir: {CONFIG['output_dir']}")
print(f"Epochs:     {CONFIG['num_epochs']}")
print(f"Batch size: {CONFIG['batch_size']} × {CONFIG['grad_accum_steps']} grad accum = {CONFIG['batch_size'] * CONFIG['grad_accum_steps']} effective")
print("=" * 60)

from transformers import AutoTokenizer
from datasets import load_dataset

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])

print("Loading dataset from text files...")
raw_dataset = load_dataset(
    "text",
    data_files={
        "train":      CONFIG["train_file"],
        "validation": CONFIG["valid_file"],
    }
)
print(f"  Train examples: {len(raw_dataset['train']):,}")
print(f"  Valid examples: {len(raw_dataset['validation']):,}")

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding=False,
        return_special_tokens_mask=True,
    )

print("Tokenizing... (this takes ~10-15 minutes)")
tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=2,
    remove_columns=["text"],
    desc="Tokenizing",
)

print("\nTokenization complete.")
print(f"  Sample train input_ids length: {len(tokenized_dataset['train'][0]['input_ids'])}")

from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

print("Loading XLM-RoBERTa-base...")
model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params / 1e6:.1f}M")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=CONFIG["mlm_probability"],
)
print("DataCollator ready (dynamic MLM masking, p=0.15)")

from transformers import TrainingArguments
import packaging.version
import transformers

os.makedirs(CONFIG["output_dir"], exist_ok=True)
use_new_api = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")
eval_strat_key = "eval_strategy" if use_new_api else "evaluation_strategy"

total_steps = (
    len(tokenized_dataset["train"])
    // (CONFIG["batch_size"] * CONFIG["grad_accum_steps"])
    * CONFIG["num_epochs"]
)
warmup_steps = int(CONFIG["warmup_ratio"] * total_steps)

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],

    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["grad_accum_steps"],

    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    warmup_steps=warmup_steps,
    lr_scheduler_type="linear",

    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,

    **{eval_strat_key: "epoch"},

    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=CONFIG["logging_steps"],
    report_to="none",

    seed=CONFIG["seed"],
)

print("Training arguments configured.")
print(f"  Effective batch size: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
print(f"  Total steps:          {total_steps}")
print(f"  Warmup steps:         {warmup_steps}")

from transformers import Trainer

trainer_kwargs = {
    "processing_class" if use_new_api else "tokenizer": tokenizer
}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    **trainer_kwargs,
)

print("Starting training...")
print("-" * 60)

train_result = trainer.train()

print("-" * 60)
print("Training complete.")
print(f"  Total steps:     {train_result.global_step}")
# training_loss is summed across grad_accum_steps before averaging — divide to get
# true per-token loss comparable to eval_loss
true_train_loss = train_result.training_loss / CONFIG["grad_accum_steps"]
print(f"  Training loss:   {train_result.training_loss:.4f}  (raw, inflated ×{CONFIG['grad_accum_steps']})")
print(f"  Training loss:   {true_train_loss:.4f}  (per-token, comparable to eval_loss)")
print(f"  Runtime:         {train_result.metrics['train_runtime'] / 3600:.2f} hours")

final_path = CONFIG["output_dir"] + "/final"

trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print(f"Model saved to: {final_path}")
print("\nFiles written:")
for f in os.listdir(final_path):
    size = os.path.getsize(os.path.join(final_path, f)) / 1e6
    print(f"  {f}  ({size:.1f} MB)")

import math

print("Computing final validation metrics...")
metrics = trainer.evaluate()

eval_loss = metrics["eval_loss"]
perplexity = math.exp(eval_loss)

print("\n" + "=" * 60)
print("FINAL VALIDATION RESULTS")
print("=" * 60)
print(f"  eval_loss:   {eval_loss:.4f}")
print(f"  perplexity:  {perplexity:.2f}")
print("=" * 60)
print("\nRecord these numbers in your thesis (Chapter 5, Section 5.2).")
print("Compare against general XLM-RoBERTa-base baseline in next cell.")

print("Loading general XLM-RoBERTa-base for baseline comparison...")
baseline_model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])

baseline_trainer = Trainer(
    model=baseline_model,
    args=training_args,
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    **trainer_kwargs,
)

baseline_metrics = baseline_trainer.evaluate()
baseline_loss = baseline_metrics["eval_loss"]
baseline_ppl  = math.exp(baseline_loss)

print("\n" + "=" * 60)
print("BASELINE vs RETRAINED COMPARISON")
print("=" * 60)
print(f"  General XLM-RoBERTa-base")
print(f"    eval_loss:   {baseline_loss:.4f}")
print(f"    perplexity:  {baseline_ppl:.2f}")
print()
print(f"  MINT-Pat retrained model")
print(f"    eval_loss:   {eval_loss:.4f}")
print(f"    perplexity:  {perplexity:.2f}")
print()
ppl_drop = ((baseline_ppl - perplexity) / baseline_ppl) * 100
print(f"  Perplexity reduction: {ppl_drop:.1f}%")
print("=" * 60)

from transformers import pipeline

print("Running fill-mask sanity checks on retrained model...")
fill_mask = pipeline(
    "fill-mask",
    model=final_path,
    tokenizer=final_path,
    top_k=5,
)

test_sentences = [
    "The <mask> device regulates the flow of electrical current in the circuit.",
    "The claims of the present <mask> are defined in the appended claims.",
    "本発明は、<mask>装置に関するものである。",
    "請求項１に記載の<mask>において、前記制御部は信号を出力する。",
]

print("\n" + "-" * 60)
for sentence in test_sentences:
    print(f"\nInput: {sentence}")
    results = fill_mask(sentence)
    print("Top predictions:")
    for r in results:
        print(f"  {r['token_str']:20s}  score={r['score']:.3f}")
print("-" * 60)
print("\nIf you see patent-domain words ranking highly (制御, 半導体,")
print("invention, semiconductor, etc.) — domain adaptation worked.")
