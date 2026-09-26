# PPO-RLHF Detoxification — Manual Implementation vs TRL Baseline

Hand-written PPO (advantage estimation + clipped surrogate loss) applied to a
PEFT/LoRA-adapted FLAN-T5 summarizer, benchmarked against HuggingFace `trl`'s
`PPOTrainer` on the same model, data, and hyperparameters.

## Why this exists

`trl.PPOTrainer` hides GAE, the clipped surrogate objective, and the KL
penalty inside `.step()`. This project reimplements those pieces from scratch
in `src/ppo_core.py` and validates correctness by comparing training curves
(reward, KL, clip fraction, ROUGE) against the TRL baseline under identical
conditions.

## Project layout

```
ppo-rlhf-detox/
├── configs/
│   └── config.yaml          # all hyperparameters in one place
├── src/
│   ├── utils.py              # seeding, logging, metric tracking
│   ├── data.py                # DialogSum loading + prompt/response tokenization
│   ├── models.py              # ValueHead + PolicyWithValueHead, ref model setup
│   ├── reward.py              # reward model wrapper (toxicity classifier scoring)
│   ├── ppo_core.py            # reward shaping, GAE, clipped surrogate loss <- the hand-written math
│   ├── rollout.py             # generation + logprob computation
│   ├── train_manual_ppo.py    # main training script (your implementation)
│   ├── train_trl_baseline.py  # same setup, using trl.PPOTrainer
│   └── compare_runs.py        # loads both run logs, plots side-by-side
├── scripts/
│   ├── run_manual.sh
│   └── run_baseline.sh
├── tests/
│   └── test_ppo_core.py       # unit tests: hand-computed toy values for GAE/loss
└── outputs/
    ├── checkpoints/
    └── logs/                  # per-step metrics (jsonl) from each run
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Baseline (TRL PPOTrainer) — establishes ground truth curves
bash scripts/run_baseline.sh

# Your hand-written PPO — same config, same seed
bash scripts/run_manual.sh

# Compare the two runs
python3 src/compare_runs.py --baseline outputs/logs/trl_baseline.jsonl --manual outputs/logs/manual_ppo.jsonl
```

## Unit tests first

Before running full training, verify the math in isolation:

```bash
pytest tests/ -v
```

`test_ppo_core.py` checks GAE and the clipped loss against small
hand-computed cases — do this before touching the LLM, it's much faster to
debug a bug in a 3-step toy example than inside a training loop.

## Ablations (see configs/config.yaml)

Sweep `kl_coef`, `clip_eps`, `gae_lambda`, `ppo_epochs` — each run writes to
its own log file under `outputs/logs/ablation_<name>.jsonl` for later
comparison plots.

## Data

- Prompts: `knkarthick/dialogsum` (HuggingFace) — same as the original lab
- Reward model: `facebook/roberta-hate-speech-dynabench-r4-target`
- Optional extension: swap in `openai/summarize_from_feedback` to train your
  own reward model instead of reusing a borrowed classifier
