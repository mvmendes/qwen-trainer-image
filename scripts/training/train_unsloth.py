#!/usr/bin/env python3
"""
AlienX Security Model - QLoRA com Unsloth (kernels fundidos, ~2x + menos VRAM)

Acelera o QLoRA do Qwen2.5-Coder-7B na L4 / RTX 3090 / T4 usando os kernels
Triton do Unsloth. Mesmo dataset e convencoes (ALIENX_*) do train_lora.py.

IMPORTANTE: `unsloth` precisa ser importado ANTES de transformers/trl (faz patching).
Instale com a stack oficial do Unsloth (ver notebook) em runtime LIMPO.

Modos:
  python train_unsloth.py --dry-run
  python train_unsloth.py
  python train_unsloth.py --resume

Env vars: ALIENX_MODEL, ALIENX_DATASET, ALIENX_OUTPUT, ALIENX_EPOCHS,
  ALIENX_MAX_TRAIN, ALIENX_BATCH, ALIENX_GRAD_ACCUM, ALIENX_MAX_SEQ_LENGTH,
  ALIENX_LORA_R, ALIENX_LORA_ALPHA
"""

import argparse
import inspect
import os
import sys
import time

# Workaround: unsloth_zoo (com torch 2.4.x) faz inspect.getsource(torch._inductor.config)
# sem importar o submodulo -> AttributeError. Pre-importar popula o atributo.
import torch
try:
    import torch._inductor.config  # noqa: F401
except Exception:
    pass

# Unsloth (patcha transformers/trl). Falha cedo com msg clara.
try:
    from unsloth import FastLanguageModel
except ImportError:
    print("ERRO: unsloth nao instalado. Veja o notebook (install cell) e use runtime limpo.")
    sys.exit(1)
from datasets import load_dataset
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

MODEL_ID = os.environ.get("ALIENX_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
DATASET_PATH = os.environ.get(
    "ALIENX_DATASET", "/content/drive/MyDrive/alienxmodel/data/train_coder7b.jsonl"
)
OUTPUT_DIR = os.environ.get(
    "ALIENX_OUTPUT", "/content/drive/MyDrive/alienxmodel/models/alienx-security-lora"
)

MAX_SEQ_LENGTH = int(os.environ.get("ALIENX_MAX_SEQ_LENGTH", "4096"))
PER_DEVICE_BATCH_SIZE = int(os.environ.get("ALIENX_BATCH", "4"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("ALIENX_GRAD_ACCUM", "8"))
NUM_EPOCHS = int(os.environ.get("ALIENX_EPOCHS", "3"))
MAX_TRAIN_SAMPLES = int(os.environ.get("ALIENX_MAX_TRAIN", "0"))

LEARNING_RATE = float(os.environ.get("ALIENX_LR", "2e-4"))
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3

LORA_R = int(os.environ.get("ALIENX_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("ALIENX_LORA_ALPHA", "16"))
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 3
EARLY_STOPPING_PATIENCE = 2
MAX_EVAL_SAMPLES = 2000

SYSTEM_PROMPT = (
    "You are an expert cybersecurity analyst. Analyze vulnerabilities, "
    "generate exploit proofs of concept, and explain security techniques "
    "with precision and detail."
)


class ThroughputCallback(TrainerCallback):
    def __init__(self, total_steps_full_run: int, warmup_steps: int = 5):
        self.total_steps_full_run = total_steps_full_run
        self.warmup_steps = warmup_steps
        self.start_time = None
        self.start_step = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.warmup_steps:
            self.start_time = time.time()
            self.start_step = state.global_step

    def report(self, state):
        if self.start_time is None or state.global_step <= self.start_step:
            print("\n[DRY-RUN] Steps insuficientes para medir throughput.")
            return
        elapsed = time.time() - self.start_time
        steps_done = state.global_step - self.start_step
        sec_per_step = elapsed / steps_done
        total_hours = (self.total_steps_full_run * sec_per_step) / 3600
        print("\n" + "=" * 60)
        print("[DRY-RUN] ESTIMATIVA")
        print("=" * 60)
        print(f"  Medido: {steps_done} steps em {elapsed:.1f}s")
        print(f"  Throughput: {sec_per_step:.2f} s/step")
        print(f"  Steps totais (run completo): {self.total_steps_full_run:,}")
        print(f"  Tempo estimado: {total_hours:.1f} horas")
        print("=" * 60)


def build_sft_config(dry_run: bool) -> SFTConfig:
    """Monta SFTConfig robusto a diferencas de versao do TRL (max_length vs max_seq_length)."""
    params = set(inspect.signature(SFTConfig.__init__).parameters)
    kwargs = dict(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        dataset_text_field="text",
        packing=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"alienx-unsloth-{int(time.time())}",
        max_steps=20 if dry_run else -1,
    )
    # nome do campo de comprimento mudou entre versoes do TRL
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    elif "max_length" in params:
        kwargs["max_length"] = MAX_SEQ_LENGTH
    # eval so quando suportado e fora do dry-run
    if not dry_run and "eval_strategy" in params:
        kwargs.update(
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    # filtra qualquer chave nao suportada pela versao instalada
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return SFTConfig(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    print("=" * 60)
    print("AlienX Security Model - QLoRA Unsloth")
    print(f"  Model: {MODEL_ID}")
    print(f"  GPU: {props.name} ({props.total_memory/1e9:.1f} GB)")
    print(f"  LoRA r={LORA_R} alpha={LORA_ALPHA} | seq={MAX_SEQ_LENGTH}")
    print(f"  Mode: {'DRY-RUN' if args.dry_run else 'FULL TRAIN'}")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"ERRO: dataset nao encontrado em {DATASET_PATH}")
        sys.exit(1)

    print("\n[1/4] Model + Tokenizer (Unsloth 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    print("\n[2/4] Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"  Total: {len(dataset):,}")
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    if len(eval_ds) > MAX_EVAL_SAMPLES:
        eval_ds = eval_ds.select(range(MAX_EVAL_SAMPLES))
    if MAX_TRAIN_SAMPLES and MAX_TRAIN_SAMPLES < len(train_ds):
        train_ds = train_ds.shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
        print(f"  [SUBSET] {MAX_TRAIN_SAMPLES:,} exemplos")
    print(f"  Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")

    def format_chat(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{example.get('instruction','')}\n\n{example.get('input','')}"},
            {"role": "assistant", "content": example.get("output", "")},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)}

    full_train_len = len(train_ds)
    approx_packed = (full_train_len * 900) / MAX_SEQ_LENGTH
    effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = max(1, int(approx_packed / effective_batch))
    total_steps = steps_per_epoch * NUM_EPOCHS

    if args.dry_run:
        train_ds = train_ds.select(range(min(5000, len(train_ds))))
        eval_ds = eval_ds.select(range(min(200, len(eval_ds))))
        print(f"  [DRY-RUN] subset: {len(train_ds)} train / {len(eval_ds)} eval")

    train_ds = train_ds.map(format_chat, num_proc=4, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_chat, num_proc=4, remove_columns=eval_ds.column_names)

    print("\n[3/4] Trainer (Unsloth + SFT + packing)...")
    last_checkpoint = None
    if args.resume and os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint:
            print(f"  Resume from: {last_checkpoint}")

    sft_config = build_sft_config(args.dry_run)

    callbacks = []
    throughput_cb = None
    if args.dry_run:
        throughput_cb = ThroughputCallback(total_steps_full_run=total_steps, warmup_steps=5)
        callbacks.append(throughput_cb)
    else:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=None if args.dry_run else eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print(f"  Effective batch: {effective_batch}")
    print(f"  Steps/epoch (est): {steps_per_epoch:,} | Total (est): {total_steps:,}")

    print("\n[4/4] Treino...")
    if args.dry_run:
        trainer.train()
        if throughput_cb:
            throughput_cb.report(trainer.state)
        print("\n>>> DRY-RUN concluido.")
        return

    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("\nSalvando adapter LoRA...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter salvo em: {OUTPUT_DIR}")
    print("Treino completo!")


if __name__ == "__main__":
    main()
