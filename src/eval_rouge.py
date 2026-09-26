"""ROUGE comparison across the four stages of this project's pipeline:
base FLAN-T5, Lab 2 PEFT/SFT adapter, TRL-baseline PPO, and manual PPO.

Checks that detoxification (PPO) didn't destroy summarization quality —
the ROUGE half of the "reward, KL, clip fraction, ROUGE" comparison this
project set out to run (see README).

    python3 src/eval_rouge.py --n-samples 100 --out outputs/logs/rouge_results.json
"""

import argparse
import json

import evaluate
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from utils import get_device

BASE_MODEL_NAME = "google/flan-t5-base"
START_PROMPT = "Summarize the following conversation.\n\n"
END_PROMPT = "\n\nSummary: "

STAGES = {
    "base": None,
    "sft_peft": "outputs/checkpoints/lab2_peft_adapter",
    "trl_baseline": "outputs/checkpoints/trl_baseline",
    "manual_ppo": "outputs/checkpoints/manual_ppo",
}


def load_stage_model(adapter_path, device):
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME).to(device)
    if adapter_path is None:
        return base
    return PeftModel.from_pretrained(base, adapter_path).to(device)


def generate_summaries(model, tokenizer, dialogues, device, max_new_tokens=100):
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
    outputs = []
    for dialogue in dialogues:
        prompt = START_PROMPT + dialogue + END_PROMPT
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        summary_ids = model.generate(input_ids=input_ids, generation_config=gen_config)
        outputs.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--out", type=str, default="outputs/logs/rouge_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Evaluating on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    dataset = load_dataset("knkarthick/dialogsum", split="test")
    dataset = dataset.shuffle(seed=args.seed).select(range(min(args.n_samples, len(dataset))))
    dialogues = dataset["dialogue"]
    references = dataset["summary"]

    rouge = evaluate.load("rouge")
    results = {}

    for stage_name, adapter_path in STAGES.items():
        print(f"\n--- {stage_name} ({adapter_path or 'no adapter'}) ---")
        model = load_stage_model(adapter_path, device)
        model.eval()
        predictions = generate_summaries(model, tokenizer, dialogues, device)
        scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        results[stage_name] = scores
        print({k: round(v, 4) for k, v in scores.items()})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ROUGE results to {args.out}")

    print("\n=== Summary table ===")
    metrics = list(next(iter(results.values())).keys())
    header = "stage".ljust(14) + "".join(m.ljust(10) for m in metrics)
    print(header)
    for stage_name, scores in results.items():
        row = stage_name.ljust(14) + "".join(f"{scores[m]:.4f}".ljust(10) for m in metrics)
        print(row)


if __name__ == "__main__":
    main()
