"""Reward model scoring. Reuses the toxicity classifier from Lab 3: the
"not hate" logit for a (prompt + response) pair is the scalar reward."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ToxicityRewardModel:
    def __init__(self, cfg: dict, device: torch.device):
        rm_cfg = cfg["reward_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(rm_cfg["name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(rm_cfg["name"]).to(device)
        self.model.eval()
        self.not_hate_index = rm_cfg["not_hate_index"]
        self.device = device

    @torch.no_grad()
    def score(self, texts: list[str]) -> torch.Tensor:
        """Returns raw (non-softmax) logits for the 'not hate' class — using raw
        logits rather than post-softmax probabilities avoids reward saturation
        near 0/1 that flattens the gradient signal for already-good responses."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        logits = self.model(**inputs).logits
        return logits[:, self.not_hate_index]
