# -*- coding: utf-8 -*-
"""mint_pat_smoke_test.py

Smoke-test for MINT-Pat MLM continued pre-training.
Uses xlm-roberta-base on a small slice of the full corpus so the full
pipeline (tokenisation → training → perplexity → fill-mask) can be
validated on a free/Pro Colab T4 in ~15–20 minutes before committing
to the full A100 run.

Run this BEFORE final_train.py (full training). If smoke-test passes
(pipeline completes, perplexity lower than baseline, patent words
appear in fill-mask top-5), switch to final_train.py unchanged.

Upload to Google Drive alongside train.txt / valid.txt and open in Colab.
"""

from google.colab import drive
drive.mount('/content/drive')

CONFIG = {
    # --- Paths (same as full training — no separate files needed) ---
    "train_file": "/content/drive/MyDrive/train.txt",
    "valid_file": "/content/drive/MyDrive/valid.txt",
    "output_dir": "/content/drive/MyDrive/xlm_roberta_patent_smoke",

    # --- Base model (identical to full run — that's the point) ---
    "base_model": "xlm-roberta-base",

    # --- Smoke-test subset ---
    # 50k train lines ≈ 2.5% of full corpus; enough to observe loss drop
    # and validate pipeline without hitting Drive quota or Colab timeout.
    "train_subset_lines": 50_000,
    # 5k validation lines — fast eval, still representative
    "valid_subset_lines": 5_000,

    # --- Hyperparameters scaled for T4 (16 GB) ---
    # Sequence length reduced to 256 to double throughput on T4.
    # All other values match full training so behaviour is comparable.
    "max_length":        256,
    "mlm_probability":   0.15,
    "learning_rate":     2e-5,
    "weight_decay":      0.01,
    "warmup_ratio":      0.06,
    "num_epochs":        1,          # one epoch is enough to see loss drop
    "batch_size":        8,          # T4 16GB: 8 per-device at seq_len=256
    "grad_accum_steps":  8,          # effective batch = 64
    "seed":              42,

    # --- Logging ---
    "logging_steps":     100,
}

# ── 0. Environment check ────────────────────────────────────────────────────

import os
import torch

print("=" * 60)
print("MINT-Pat Smoke-Test Setup Check")
print("=" * 60)

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU:   {gpu}")
    print(f"VRAM:  {vram:.1f} GB")
    if vram < 14:
        print("WARNING: Less than 14 GB VRAM — reduce batch_size to 4 if OOM.")
else:
    print("ERROR: No GPU. Runtime > Change runtime type > T4 GPU and retry.")
    raise SystemExit

for key in ("train_file", "valid_file"):
    path = CONFIG[key]
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f"{key}: found ({size_mb:.0f} MB)")
    else:
        print(f"{key}: NOT FOUND at {path}")
        raise SystemExit

print(f"\nBase model:   {CONFIG['base_model']}")
print(f"Subset:       {CONFIG['train_subset_lines']:,} train / {CONFIG['valid_subset_lines']:,} valid lines")
print(f"Max length:   {CONFIG['max_length']} tokens  (256 vs 512 in full run)")
print(f"Effective batch: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
print("=" * 60)

# ── 1. Load tokeniser ────────────────────────────────────────────────────────

from transformers import AutoTokenizer

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
print("  Tokenizer ready.")

# ── 2. Sample subset from disk (no full-file load) ───────────────────────────

from datasets import Dataset

def sample_lines(path, n):
    """Read first n non-empty lines from a text file."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                lines.append(line)
            if len(lines) >= n:
                break
    return lines

print(f"\nSampling {CONFIG['train_subset_lines']:,} train lines...")
train_lines = sample_lines(CONFIG["train_file"], CONFIG["train_subset_lines"])
print(f"Sampling {CONFIG['valid_subset_lines']:,} valid lines...")
valid_lines = sample_lines(CONFIG["valid_file"],  CONFIG["valid_subset_lines"])

raw_dataset = {
    "train":      Dataset.from_dict({"text": train_lines}),
    "validation": Dataset.from_dict({"text": valid_lines}),
}
print(f"  Train: {len(raw_dataset['train']):,} examples")
print(f"  Valid: {len(raw_dataset['validation']):,} examples")

# ── 3. Tokenise ──────────────────────────────────────────────────────────────

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding=False,
        return_special_tokens_mask=True,
    )

print("\nTokenizing...")
tokenized = {}
for split, ds in raw_dataset.items():
    tokenized[split] = ds.map(
        tokenize_function,
        batched=True,
        num_proc=2,
        remove_columns=["text"],
        desc=f"Tokenizing {split}",
    )

print(f"  Sample input_ids length: {len(tokenized['train'][0]['input_ids'])}")

# ── 4. Model + data collator ─────────────────────────────────────────────────

from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

print("\nLoading xlm-roberta-base...")
model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params / 1e6:.1f}M")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=CONFIG["mlm_probability"],
)

# ── 5. Training ──────────────────────────────────────────────────────────────

from transformers import TrainingArguments, Trainer
import packaging.version
import transformers

os.makedirs(CONFIG["output_dir"], exist_ok=True)
use_new_api = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")
eval_strat_key = "eval_strategy" if use_new_api else "evaluation_strategy"

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["grad_accum_steps"],
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    warmup_ratio=CONFIG["warmup_ratio"],
    lr_scheduler_type="linear",
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    **{eval_strat_key: "epoch"},
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=CONFIG["logging_steps"],
    logging_dir=CONFIG["output_dir"] + "/logs",
    report_to="none",
    seed=CONFIG["seed"],
)

trainer_kwargs = {
    "processing_class" if use_new_api else "tokenizer": tokenizer
}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    **trainer_kwargs,
)

print("\nStarting smoke-test training (1 epoch, 50k lines)...")
print("-" * 60)
train_result = trainer.train()
print("-" * 60)
print(f"  Steps:         {train_result.global_step}")
print(f"  Training loss: {train_result.training_loss:.4f}")
print(f"  Runtime:       {train_result.metrics['train_runtime'] / 60:.1f} minutes")

# ── 6. Perplexity vs general baseline ───────────────────────────────────────

import math

print("\nEvaluating smoke-test model...")
metrics = trainer.evaluate()
smoke_loss = metrics["eval_loss"]
smoke_ppl  = math.exp(smoke_loss)

print("\nLoading general xlm-roberta-base for baseline comparison...")
baseline_model = AutoModelForMaskedLM.from_pretrained(CONFIG["base_model"])
baseline_trainer = Trainer(
    model=baseline_model,
    args=training_args,
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    **{"processing_class" if use_new_api else "tokenizer": tokenizer},
)
baseline_metrics = baseline_trainer.evaluate()
baseline_loss = baseline_metrics["eval_loss"]
baseline_ppl  = math.exp(baseline_loss)

ppl_drop = ((baseline_ppl - smoke_ppl) / baseline_ppl) * 100

print("\n" + "=" * 60)
print("SMOKE-TEST RESULTS")
print("=" * 60)
print(f"  General xlm-roberta-base   ppl = {baseline_ppl:.2f}")
print(f"  Smoke-test retrained       ppl = {smoke_ppl:.2f}")
print(f"  Perplexity reduction:      {ppl_drop:.1f}%")
print("=" * 60)
print()
if ppl_drop > 0:
    print("PASS: perplexity dropped — domain adaptation signal present.")
    print("      Proceed to full training with final_train.py (A100 required).")
else:
    print("WARN: perplexity did not drop. Check data quality and loss curve")
    print("      before investing in full A100 training run.")

# ── 7. Fill-mask sanity check ────────────────────────────────────────────────

from transformers import pipeline

final_path = CONFIG["output_dir"] + "/final"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"Model saved to: {final_path}")

fill_mask = pipeline("fill-mask", model=final_path, tokenizer=final_path, top_k=5)

test_sentences = [
    "The <mask> device regulates the flow of electrical current in the circuit.",
    "The claims of the present <mask> are defined in the appended claims.",
    "本発明は、<mask>装置に関するものである。",
    "請求項１に記載の<mask>において、前記制御部は信号を出力する。",
]

print("\n" + "-" * 60)
print("Fill-mask sanity check (expect patent-domain words in top-5):")
print("-" * 60)
for sentence in test_sentences:
    print(f"\nInput: {sentence}")
    for r in fill_mask(sentence):
        print(f"  {r['token_str']:20s}  score={r['score']:.3f}")
print("-" * 60)
print("\nIf patent words (制御, 半導体, invention, semiconductor, etc.)")
print("rank in top-5 — smoke test passed. Run final_train.py for full training.")
