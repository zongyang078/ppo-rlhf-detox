"""Unit tests for the hand-written PPO math, checked against toy cases you
can verify by hand. Run these before touching the LLM — a bug here is a
5-minute fix; the same bug discovered inside a training loop is an
afternoon.

    pytest tests/ -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ppo_core import clipped_surrogate_loss, compute_gae, compute_kl, per_token_logprobs, shape_rewards


def test_per_token_logprobs_matches_manual_softmax():
    # 2 timesteps, vocab size 3, batch size 1
    logits = torch.tensor([[[2.0, 1.0, 0.1], [0.5, 2.5, 0.1]]])
    labels = torch.tensor([[0, 1]])
    logprobs = per_token_logprobs(logits, labels, pad_token_id=-1)

    expected_t0 = torch.log_softmax(logits[0, 0], dim=-1)[0]
    expected_t1 = torch.log_softmax(logits[0, 1], dim=-1)[1]
    assert torch.allclose(logprobs[0, 0], expected_t0, atol=1e-5)
    assert torch.allclose(logprobs[0, 1], expected_t1, atol=1e-5)


def test_per_token_logprobs_masks_padding():
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[0, -1]])  # second token is padding
    logprobs = per_token_logprobs(logits, labels, pad_token_id=-1)
    assert logprobs[0, 1].item() == 0.0


def test_compute_kl_zero_when_policies_match():
    logprobs_policy = torch.tensor([[-0.5, -0.3]])
    logprobs_ref = torch.tensor([[-0.5, -0.3]])
    kl = compute_kl(logprobs_policy, logprobs_ref)
    assert torch.allclose(kl, torch.zeros_like(kl))


def test_shape_rewards_terminal_reward_at_last_real_token():
    # batch=1, seq_len=3, response has 2 real tokens then 1 pad
    logprobs_policy = torch.tensor([[-0.1, -0.2, 0.0]])
    logprobs_ref = torch.tensor([[-0.1, -0.2, 0.0]])  # zero KL for simplicity
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])
    reward_scores = torch.tensor([1.5])

    rewards = shape_rewards(reward_scores, logprobs_policy, logprobs_ref, response_mask, kl_coef=0.2)

    assert rewards[0, 0].item() == 0.0  # zero KL, no terminal reward here
    assert torch.isclose(rewards[0, 1], torch.tensor(1.5), atol=1e-5)  # terminal reward lands on last real token
    assert rewards[0, 2].item() == 0.0  # padding untouched


def test_gae_reduces_to_single_step_advantage_when_lambda_zero():
    # With lambda=0, GAE collapses to the one-step TD error: A_t = r_t + gamma*V_{t+1} - V_t
    rewards = torch.tensor([[1.0, 0.0]])
    values = torch.tensor([[0.5, 0.3]])
    response_mask = torch.tensor([[1.0, 1.0]])

    advantages, returns = compute_gae(rewards, values, response_mask, gamma=1.0, lam=0.0)

    # t=1 (last step): delta = 0.0 + 1.0*0 - 0.3 = -0.3
    assert torch.isclose(advantages[0, 1], torch.tensor(-0.3), atol=1e-5)
    # t=0: delta = 1.0 + 1.0*0.3 - 0.5 = 0.8 (next_value carries over from t=1's value, masked)
    assert torch.isclose(advantages[0, 0], torch.tensor(0.8), atol=1e-5)


def test_gae_zero_reward_zero_value_gives_zero_advantage():
    rewards = torch.zeros(1, 3)
    values = torch.zeros(1, 3)
    response_mask = torch.ones(1, 3)
    advantages, returns = compute_gae(rewards, values, response_mask, gamma=1.0, lam=0.95)
    assert torch.allclose(advantages, torch.zeros_like(advantages))
    assert torch.allclose(returns, torch.zeros_like(returns))


def test_clipped_loss_no_clipping_when_ratio_is_one():
    # logprobs_new == logprobs_old -> ratio = 1 -> unclipped == clipped, clip_fraction == 0
    logprobs = torch.tensor([[-0.5, -0.3]], requires_grad=True)
    logprobs_old = torch.tensor([[-0.5, -0.3]])
    advantages = torch.tensor([[1.0, -1.0]])
    values_new = torch.tensor([[0.5, 0.5]], requires_grad=True)
    returns = torch.tensor([[0.5, 0.5]])
    response_mask = torch.ones(1, 2)

    loss, diagnostics = clipped_surrogate_loss(
        logprobs, logprobs_old, advantages, values_new, returns, response_mask, clip_eps=0.2
    )
    assert diagnostics["clip_fraction"] == 0.0
    assert diagnostics["value_loss"] == 0.0  # values_new == returns


def test_clipped_loss_clips_large_ratio():
    # logprobs_new far above logprobs_old -> large ratio -> should get clipped
    logprobs_old = torch.tensor([[-1.0]])
    logprobs_new = torch.tensor([[0.5]], requires_grad=True)  # ratio = exp(1.5) ~= 4.48
    advantages = torch.tensor([[1.0]])
    values_new = torch.tensor([[0.0]], requires_grad=True)
    returns = torch.tensor([[0.0]])
    response_mask = torch.ones(1, 1)

    _, diagnostics = clipped_surrogate_loss(
        logprobs_new, logprobs_old, advantages, values_new, returns, response_mask, clip_eps=0.2
    )
    assert diagnostics["clip_fraction"] == 1.0
