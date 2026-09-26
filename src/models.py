"""Policy model with an attached value head, and a frozen reference model.

This is the piece TRL hides inside `AutoModelForSeq2SeqLMWithValueHead`.
Writing it out explicitly is the point of this project.
"""

import copy

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ValueHead(nn.Module):
    """Scalar value estimate per decoder position, predicted from the decoder's
    last hidden state. Matches TRL's `ValueHead` in spirit: dropout + linear."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.summary = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.summary.weight, std=1.0 / (hidden_size + 1))
        nn.init.zeros_(self.summary.bias)

    def forward(self, decoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # decoder_hidden_states: (batch, tgt_len, hidden_size)
        x = self.dropout(decoder_hidden_states)
        value = self.summary(x).squeeze(-1)  # (batch, tgt_len)
        return value


class PolicyWithValueHead(nn.Module):
    """Wraps a seq2seq LM (optionally PEFT-adapted) with a value head so a
    single forward pass returns both action log-probs (via logits) and the
    critic's value estimate."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.d_model
        self.value_head = ValueHead(hidden_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits  # (batch, tgt_len, vocab_size)
        last_hidden = outputs.decoder_hidden_states[-1]  # (batch, tgt_len, hidden_size)
        values = self.value_head(last_hidden)
        return logits, values

    def generate(self, input_ids, attention_mask, **gen_kwargs):
        return self.base_model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
        )


def load_tokenizer(cfg: dict) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(cfg["model"]["base_model_name"])


def load_policy_and_ref(cfg: dict, device: torch.device):
    """Load the PEFT-adapted policy (trainable, with a fresh value head) and a
    frozen reference copy (pre-RL weights, used only for the KL penalty).
    """
    model_cfg = cfg["model"]
    base = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["base_model_name"])

    if model_cfg.get("peft_adapter_path"):
        base = PeftModel.from_pretrained(base, model_cfg["peft_adapter_path"], is_trainable=True)

    policy = PolicyWithValueHead(base).to(device)

    # Reference model: deep-copy BEFORE any PPO updates, freeze entirely.
    ref_base = copy.deepcopy(base)
    for p in ref_base.parameters():
        p.requires_grad_(False)
    ref_base.eval().to(device)

    return policy, ref_base
