"""Baseline run using trl.PPOTrainer, same config/data/model as
train_manual_ppo.py. This is the ground truth your hand-written
implementation is checked against.

    python3 src/train_trl_baseline.py --config configs/config.yaml --run-name trl_baseline
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
from trl.core import LengthSampler

from data import build_prompt_dataset
from utils import JsonlLogger, get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run-name", type=str, default="trl_baseline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    model_cfg, ppo_cfg, gen_cfg, rm_cfg = cfg["model"], cfg["ppo"], cfg["generation"], cfg["reward_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_name"])

    base = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["base_model_name"])
    if model_cfg.get("peft_adapter_path"):
        base = PeftModel.from_pretrained(base, model_cfg["peft_adapter_path"], is_trainable=True)

    policy_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(base).to(device)
    ref_model = create_reference_model(policy_model)

    dataset = build_prompt_dataset(cfg, tokenizer)

    ppo_config = PPOConfig(
        model_name=model_cfg["base_model_name"],
        learning_rate=ppo_cfg["learning_rate"],
        ppo_epochs=ppo_cfg["ppo_epochs"],
        mini_batch_size=ppo_cfg["mini_batch_size"],
        batch_size=ppo_cfg["batch_size"],
        gamma=ppo_cfg["gamma"],
        lam=ppo_cfg["gae_lambda"],
        cliprange=ppo_cfg["clip_eps"],
        cliprange_value=ppo_cfg["clip_eps"],
        vf_coef=ppo_cfg["vf_coef"],
        init_kl_coef=ppo_cfg["kl_coef"],
        adap_kl_ctrl=False,  # fixed KL coefficient, matching the manual implementation
    )

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    reward_pipe = pipeline("sentiment-analysis", model=rm_cfg["name"], device=device)
    reward_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
    not_hate_index = rm_cfg["not_hate_index"]

    output_length_sampler = LengthSampler(gen_cfg["output_min_length"], gen_cfg["output_max_length"])
    generation_kwargs = {
        "min_length": gen_cfg["output_min_length"],
        "top_k": gen_cfg["top_k"],
        "top_p": gen_cfg["top_p"],
        "do_sample": gen_cfg["do_sample"],
    }

    logger = JsonlLogger(f"{cfg['logging']['log_dir']}/{args.run_name}.jsonl")

    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= ppo_cfg["max_ppo_steps"]:
            break

        prompt_tensors = batch["input_ids"]

        response_tensors = []
        for prompt_tensor in prompt_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards_out = reward_pipe(texts, **reward_kwargs)
        reward_tensors = [torch.tensor(r[not_hate_index]["score"]) for r in rewards_out]

        stats = ppo_trainer.step(prompt_tensors, response_tensors, reward_tensors)

        logger.log(
            step,
            reward_mean=torch.stack(reward_tensors).mean().item(),
            reward_std=torch.stack(reward_tensors).std().item(),
            approx_kl=stats.get("objective/kl", None),
            clip_fraction=stats.get("ppo/policy/clipfrac", None),
            policy_loss=stats.get("ppo/loss/policy", None),
            value_loss=stats.get("ppo/loss/value", None),
        )

    save_path = f"{cfg['logging']['checkpoint_dir']}/{args.run_name}"
    # trl's PreTrainedModelWrapper.save_pretrained writes the value head's
    # pytorch_model.bin directly via torch.save before delegating to the
    # underlying PeftModel's save_pretrained, and does NOT create the
    # directory first (unlike standard HF save_pretrained) — so it must
    # already exist.
    os.makedirs(save_path, exist_ok=True)
    ppo_trainer.model.save_pretrained(save_path)
    print(f"Saved policy to {save_path}")


if __name__ == "__main__":
    main()
