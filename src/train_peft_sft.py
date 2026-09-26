"""Re-run the Lab 2 PEFT/LoRA summarization fine-tune for real.

The adapter trained locally in the Lab 2 notebook only ran max_steps=1 (a
placeholder so the notebook cell executes fast) — the notebook's own later
cells load the actual eval checkpoint from S3 instead. This script reproduces
that training with the same LoRA config and prompt template, but with a real
step count, and saves the adapter to outputs/checkpoints/lab2_peft_adapter/
so train_manual_ppo.py / train_trl_baseline.py have a genuinely fine-tuned
starting point.

    python3 src/train_peft_sft.py --max-steps 500
"""

import argparse

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from utils import get_device, set_seed

BASE_MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "knkarthick/dialogsum"
START_PROMPT = "Summarize the following conversation.\n\n"
END_PROMPT = "\n\nSummary: "

# Same LoRA hyperparameters as the Lab 2 notebook — only max_steps changes.
LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)


def build_tokenized_dataset(tokenizer):
    dataset = load_dataset(DATASET_NAME)

    def tokenize(example):
        prompts = [START_PROMPT + dialogue + END_PROMPT for dialogue in example["dialogue"]]
        model_inputs = tokenizer(prompts, truncation=True, max_length=512)
        labels = tokenizer(text_target=example["summary"], truncation=True, max_length=200)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["id", "topic", "dialogue", "summary"])
    return tokenized


def sanity_check(base_model, peft_model, tokenizer, dataset, device, n=3):
    print("\n--- sanity check: base vs PEFT-tuned generations ---")
    for i in range(n):
        dialogue = dataset["test"][i]["dialogue"]
        human_summary = dataset["test"][i]["summary"]
        prompt = START_PROMPT + dialogue + END_PROMPT
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

        base_out = tokenizer.decode(
            base_model.generate(input_ids=input_ids, max_new_tokens=100)[0], skip_special_tokens=True
        )
        peft_out = tokenizer.decode(
            peft_model.generate(input_ids=input_ids, max_new_tokens=100)[0], skip_special_tokens=True
        )
        print(f"\n[{i}] HUMAN:   {human_summary}")
        print(f"[{i}] BASE:    {base_out}")
        print(f"[{i}] PEFT:    {peft_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints/lab2_peft_adapter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sanity-check", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME).to(device)

    tokenized = build_tokenized_dataset(tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer, model=base_model)

    peft_model = get_peft_model(base_model, LORA_CONFIG)
    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="outputs/checkpoints/_peft_sft_trainer_tmp",
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=collator,
    )
    trainer.train()

    peft_model.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")

    if not args.skip_sanity_check:
        # Reload a clean base model for a fair before/after comparison
        # (peft_model shares weights with the already-tuned base_model above).
        raw_dataset = load_dataset(DATASET_NAME)
        untuned_base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME).to(device)
        sanity_check(untuned_base, peft_model, tokenizer, raw_dataset, device)


if __name__ == "__main__":
    main()
