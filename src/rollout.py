"""Generation (rollout) and the forward pass used to score prompt/response
pairs under either the policy or the reference model."""

import torch

from ppo_core import per_token_logprobs


@torch.no_grad()
def generate_responses(policy, tokenizer, input_ids, attention_mask, gen_cfg: dict, device, length_sampler=None):
    """Sample responses from the current policy. No grad — this is rollout,
    not the update step.

    `length_sampler` (trl.core.LengthSampler), if given, draws a random target
    length per call — matching the TRL baseline, which samples a fresh
    max_new_tokens per example via the same LengthSampler(output_min_length,
    output_max_length). Without this, every rollout uses the same fixed
    max_new_tokens=output_max_length, which biases response-length (and hence
    reward-model input length) differently from the baseline it's compared
    against.
    """
    max_new_tokens = length_sampler() if length_sampler is not None else gen_cfg["output_max_length"]
    gen_kwargs = dict(
        min_new_tokens=gen_cfg["output_min_length"],
        max_new_tokens=max_new_tokens,
        top_k=gen_cfg["top_k"] if gen_cfg["top_k"] > 0 else None,
        top_p=gen_cfg["top_p"],
        do_sample=gen_cfg["do_sample"],
    )
    response_ids = policy.generate(
        input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), **gen_kwargs
    )
    return response_ids


def score_sequence(model, input_ids, attention_mask, response_ids, pad_token_id, device):
    """Forward pass to get per-token logprobs (+ value, if the model has a
    value head) for an already-generated response. Used for both the policy
    (with grad, during the update) and the reference model (no grad, for KL).
    """
    logits, values = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        decoder_input_ids=response_ids.to(device),
    )
    # Shift so logits[t] predicts response_ids[t+1], matching teacher-forcing
    # convention; T5's decoder_start_token already offsets this by one position.
    logprobs = per_token_logprobs(logits, response_ids.to(device), pad_token_id)
    return logprobs, values


def score_sequence_ref(ref_model, input_ids, attention_mask, response_ids, pad_token_id, device):
    """Same as score_sequence but for the frozen reference model (no value head, no grad)."""
    with torch.no_grad():
        outputs = ref_model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            decoder_input_ids=response_ids.to(device),
            return_dict=True,
        )
        logprobs = per_token_logprobs(outputs.logits, response_ids.to(device), pad_token_id)
    return logprobs


def build_response_mask(response_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return (response_ids != pad_token_id).float()
