"""DialogSum loading and prompt tokenization.

Both the manual PPO loop and the TRL baseline consume the same
`build_prompt_dataset` output, so any difference in results can't be
attributed to a data-pipeline discrepancy.
"""

from datasets import load_dataset
from transformers import PreTrainedTokenizer


def build_prompt_dataset(cfg: dict, tokenizer: PreTrainedTokenizer):
    """Load DialogSum, wrap dialogues in the summarization instruction template,
    filter by token length, and tokenize prompts (not responses — those come
    from the policy at rollout time).

    Returns a `datasets.Dataset` with columns: `dialogue`, `query` (the
    instruction-wrapped text), `input_ids`.
    """
    data_cfg = cfg["data"]
    dataset = load_dataset(data_cfg["dataset_name"], split=data_cfg["split"])

    if data_cfg.get("max_train_samples"):
        dataset = dataset.select(range(min(len(dataset), data_cfg["max_train_samples"])))

    prefix = data_cfg["instruction_prefix"]
    suffix = data_cfg["instruction_suffix"]

    def wrap_and_filter(example):
        example["query"] = f"{prefix}{example['dialogue']}{suffix}"
        return example

    dataset = dataset.map(wrap_and_filter)

    def tokenize(example):
        example["input_ids"] = tokenizer.encode(example["query"], truncation=True, max_length=512)
        return example

    dataset = dataset.map(tokenize)

    min_len = data_cfg["input_min_text_length"]
    max_len = data_cfg["input_max_text_length"]
    dataset = dataset.filter(lambda x: min_len <= len(x["input_ids"]) <= max_len)

    dataset.set_format(type="torch")
    return dataset


def collate_prompts(batch: list[dict], tokenizer: PreTrainedTokenizer, pad_to_multiple_of: int = 8):
    """Left-pad-free batching for encoder inputs (T5 uses standard right padding
    with an attention mask, unlike decoder-only models)."""
    input_ids = [item["input_ids"] for item in batch]
    padded = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )
    return padded  # {"input_ids": ..., "attention_mask": ...}
