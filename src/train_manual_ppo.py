"""Hand-written PPO training loop for detoxifying FLAN-T5 summaries.

No trl.PPOTrainer. GAE and the clipped surrogate loss are your own code in
ppo_core.py. Run scripts/run_manual.sh, or:

    python3 src/train_manual_ppo.py --config configs/config.yaml --run-name manual_ppo
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import build_prompt_dataset, collate_prompts
from models import load_policy_and_ref, load_tokenizer
from ppo_core import clipped_surrogate_loss, compute_gae, compute_kl, shape_rewards
from reward import ToxicityRewardModel
from rollout import build_response_mask, generate_responses, score_sequence, score_sequence_ref
from trl.core import LengthSampler
from utils import JsonlLogger, get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run-name", type=str, default="manual_ppo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    tokenizer = load_tokenizer(cfg)
    policy, ref_model = load_policy_and_ref(cfg, device)
    reward_model = ToxicityRewardModel(cfg, device)

    dataset = build_prompt_dataset(cfg, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["ppo"]["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_prompts(b, tokenizer),
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["ppo"]["learning_rate"])
    logger = JsonlLogger(f"{cfg['logging']['log_dir']}/{args.run_name}.jsonl")

    ppo_cfg = cfg["ppo"]
    gen_cfg = cfg["generation"]
    pad_id = tokenizer.pad_token_id
    # Matches train_trl_baseline.py: response length is resampled per rollout
    # (there, per example; here, per batch) instead of always using the fixed
    # output_max_length ceiling, so both pipelines see the same response-length
    # distribution rather than the manual one being biased toward longer replies.
    length_sampler = LengthSampler(gen_cfg["output_min_length"], gen_cfg["output_max_length"])

    step = 0
    pbar = tqdm(total=ppo_cfg["max_ppo_steps"], desc="manual PPO")

    for batch in dataloader:
        if step >= ppo_cfg["max_ppo_steps"]:
            break

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Rollout and old-logprob scoring must be deterministic (matching how
        # ref_model is kept in eval()) — otherwise dropout noise leaks into
        # logprobs_old, biasing the KL-vs-reference estimate and adding spurious
        # ratio variance at the update step. Mirrors TRL's PPOTrainer, which
        # calls model.eval() before batched_forward_pass/generate and only
        # switches to model.train() for the actual gradient step.
        policy.eval()

        # ---- 1. Rollout: sample responses under the CURRENT policy, no grad ----
        response_ids = generate_responses(
            policy, tokenizer, input_ids, attention_mask, gen_cfg, device, length_sampler=length_sampler
        )
        response_mask = build_response_mask(response_ids, pad_id)

        with torch.no_grad():
            logprobs_old, values_old = score_sequence(
                policy, input_ids, attention_mask, response_ids, pad_id, device
            )
        logprobs_ref = score_sequence_ref(ref_model, input_ids, attention_mask, response_ids, pad_id, device)

        # KL of the rollout (old) policy vs. the frozen reference model — the
        # quantity that actually drives the KL penalty below, and the one
        # comparable to TRL's `objective/kl`. Not to be confused with
        # clipped_surrogate_loss's `approx_kl`, which measures old-vs-new
        # policy drift *within this update* (a trust-region diagnostic).
        #
        # Aggregation matches TRL's `record_step_stats` convention exactly:
        # sum the per-token KL over each sequence (response length varies per
        # example, hence the mask), THEN average over the batch. A plain
        # global per-token mean (summing over batch+seq, dividing by total
        # real-token count) is a different, much smaller-magnitude quantity
        # and is not comparable to TRL's logged value.
        kl_per_sequence = (compute_kl(logprobs_old, logprobs_ref) * response_mask).sum(dim=-1)
        kl_vs_ref = kl_per_sequence.mean().item()

        # ---- 2. Score with the reward model ----
        prompts_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        responses_text = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        full_text = [p + r for p, r in zip(prompts_text, responses_text)]
        reward_scores = reward_model.score(full_text)

        # ---- 3. Reward shaping + GAE ----
        rewards = shape_rewards(reward_scores, logprobs_old, logprobs_ref, response_mask, ppo_cfg["kl_coef"])
        advantages, returns = compute_gae(
            rewards, values_old, response_mask, gamma=ppo_cfg["gamma"], lam=ppo_cfg["gae_lambda"]
        )

        # ---- 4. PPO update: multiple epochs over this rollout batch ----
        # Deliberately NOT switching to policy.train() here. Measured directly
        # (see debugging notes): with FLAN-T5's dropout=0.1 compounding across
        # 24 transformer blocks, two independently-sampled dropout forward
        # passes on the SAME weights differ by ~1 nat/token on average — far
        # past the clip_eps=0.2 (ratio) threshold. Computing logprobs_old in
        # eval() and logprobs_new in train() mode injects that noise directly
        # into the PPO ratio, which assumes old-vs-new differs only because of
        # the actual gradient step, not sampling noise. That caused ~90% of
        # tokens to be "clipped" from step 0 (before any real update happened)
        # and made training diverge (policy_loss blowing up to 1e5+). Keeping
        # policy in eval() for BOTH old and new logprob computation removes
        # this spurious noise — eval() only disables dropout, it does not
        # block gradient flow through the value head or LoRA params.
        for _ in range(ppo_cfg["ppo_epochs"]):
            logprobs_new, values_new = score_sequence(
                policy, input_ids, attention_mask, response_ids, pad_id, device
            )
            loss, diagnostics = clipped_surrogate_loss(
                logprobs_new,
                logprobs_old,
                advantages,
                values_new,
                returns,
                response_mask,
                clip_eps=ppo_cfg["clip_eps"],
                vf_coef=ppo_cfg["vf_coef"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo_cfg["max_grad_norm"])
            optimizer.step()

        logger.log(
            step,
            reward_mean=reward_scores.mean().item(),
            reward_std=reward_scores.std().item(),
            kl_vs_ref=kl_vs_ref,
            **diagnostics,
        )
        pbar.set_postfix(reward=f"{reward_scores.mean().item():.3f}", kl=f"{diagnostics['approx_kl']:.4f}")
        pbar.update(1)
        step += 1

    pbar.close()
    save_path = f"{cfg['logging']['checkpoint_dir']}/{args.run_name}"
    policy.base_model.save_pretrained(save_path)
    print(f"Saved policy to {save_path}")


if __name__ == "__main__":
    main()
