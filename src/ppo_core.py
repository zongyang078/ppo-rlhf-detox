"""The math TRL hides inside `PPOTrainer.step()`.

Three pieces, each independently unit-testable (see tests/test_ppo_core.py):
  1. per_token_logprobs / compute_kl   — turn logits into the quantities PPO needs
  2. shape_rewards                     — sparse terminal reward + dense KL penalty
  3. compute_gae                       — advantage estimation
  4. clipped_surrogate_loss            — the actual PPO objective
"""

import torch
import torch.nn.functional as F


def per_token_logprobs(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Log-probability of each realized token under the model's distribution.

    Args:
        logits: (batch, seq_len, vocab_size) — model output BEFORE softmax
        labels: (batch, seq_len) — the actually-generated token ids
        pad_token_id: padding positions are zeroed out, not counted

    Returns:
        logprobs: (batch, seq_len), 0.0 at padding positions
    """
    log_probs_all = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab)
    mask = (labels != pad_token_id).float()
    safe_labels = labels.clamp(min=0)  # avoid out-of-range gather at padding positions
    logprobs = torch.gather(log_probs_all, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return logprobs * mask


def compute_kl(logprobs_policy: torch.Tensor, logprobs_ref: torch.Tensor) -> torch.Tensor:
    """Per-token KL estimate. This is the common single-sample approximation
    KL(policy || ref) ≈ log π(a|s) − log π_ref(a|s), evaluated at the action
    actually sampled — not the full-distribution KL, which would need a sum
    over the vocabulary at every position and is far more expensive.
    """
    return logprobs_policy - logprobs_ref


def shape_rewards(
    reward_scores: torch.Tensor,
    logprobs_policy: torch.Tensor,
    logprobs_ref: torch.Tensor,
    response_mask: torch.Tensor,
    kl_coef: float,
) -> torch.Tensor:
    """Distribute reward across the sequence: a dense per-token KL penalty at
    every step, plus the terminal reward-model score added only at the last
    real (non-padding) token of each response.

    Args:
        reward_scores: (batch,) scalar reward-model output per full response
        logprobs_policy, logprobs_ref: (batch, seq_len)
        response_mask: (batch, seq_len) 1.0 for real tokens, 0.0 for padding
        kl_coef: penalty weight on divergence from the reference model

    Returns:
        rewards: (batch, seq_len)
    """
    kl = compute_kl(logprobs_policy, logprobs_ref)
    rewards = -kl_coef * kl * response_mask

    batch_size = reward_scores.shape[0]
    last_token_idx = response_mask.sum(dim=1).long() - 1  # index of last real token, per row
    last_token_idx = last_token_idx.clamp(min=0)
    rewards[torch.arange(batch_size), last_token_idx] += reward_scores

    return rewards


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
):
    """Generalized Advantage Estimation, computed backwards through the sequence.

    Args:
        rewards: (batch, seq_len)
        values: (batch, seq_len) — value estimate under the OLD policy (no grad)
        response_mask: (batch, seq_len) — zeroes out padding contributions
        gamma: reward discount (1.0 is standard for finite LM sequences —
               there's no meaningful "future" beyond the response)
        lam: GAE bias/variance trade-off

    Returns:
        advantages, returns: both (batch, seq_len)
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device)
    next_value = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(seq_len)):
        mask_t = response_mask[:, t]
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = (delta + gamma * lam * last_gae) * mask_t
        advantages[:, t] = last_gae
        next_value = values[:, t] * mask_t  # zero out bootstrapped value past the response's end

    returns = advantages + values
    return advantages, returns


def clipped_surrogate_loss(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
    advantages: torch.Tensor,
    values_new: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
):
    """The core PPO objective: L = min(r·A, clip(r, 1-eps, 1+eps)·A), plus a
    value-function regression loss.

    Returns:
        total_loss (scalar), diagnostics dict (policy_loss, value_loss, clip_fraction, approx_kl)
    """
    # Normalize advantages per batch — standard PPO practice, stabilizes the
    # loss scale across rollouts with very different reward magnitudes.
    adv_mean = advantages[response_mask.bool()].mean()
    adv_std = advantages[response_mask.bool()].std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    ratio = torch.exp(logprobs_new - logprobs_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    per_token_policy_loss = -torch.min(unclipped, clipped)
    policy_loss = (per_token_policy_loss * response_mask).sum() / response_mask.sum()

    value_loss = (((values_new - returns) ** 2) * response_mask).sum() / response_mask.sum()

    total_loss = policy_loss + vf_coef * value_loss

    with torch.no_grad():
        clip_fraction = ((torch.abs(ratio - 1) > clip_eps).float() * response_mask).sum() / response_mask.sum()
        approx_kl = ((logprobs_old - logprobs_new) * response_mask).sum() / response_mask.sum()

    diagnostics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "clip_fraction": clip_fraction.item(),
        "approx_kl": approx_kl.item(),
    }
    return total_loss, diagnostics
