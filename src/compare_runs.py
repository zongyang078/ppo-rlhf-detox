"""Load two run logs (manual PPO vs TRL baseline) and plot reward, KL, and
clip-fraction curves side by side. This is the evidence that the hand-written
implementation is correct, not just runnable.

    python3 src/compare_runs.py --baseline outputs/logs/trl_baseline.jsonl \
                                 --manual outputs/logs/manual_ppo.jsonl \
                                 --out outputs/logs/comparison.png
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from utils import JsonlLogger


def to_df(log_path: str) -> pd.DataFrame:
    return pd.DataFrame(JsonlLogger.read(log_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--manual", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/logs/comparison.png")
    args = parser.parse_args()

    df_baseline = to_df(args.baseline)
    df_manual = to_df(args.manual)

    # (title, baseline column, manual column) — TRL's "approx_kl" log key is
    # actually `stats["objective/kl"]` (policy vs. frozen reference model).
    # The manual pipeline's own "approx_kl" is a different quantity (old-vs-new
    # policy drift within one PPO update, see clipped_surrogate_loss); the
    # comparable one is "kl_vs_ref", logged separately in train_manual_ppo.py.
    metrics = [
        ("reward_mean", "reward_mean", "reward_mean"),
        ("kl_vs_reference", "approx_kl", "kl_vs_ref"),
        ("clip_fraction", "clip_fraction", "clip_fraction"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))

    for ax, (title, baseline_col, manual_col) in zip(axes, metrics):
        if baseline_col in df_baseline.columns:
            ax.plot(df_baseline["step"], df_baseline[baseline_col], label="TRL baseline", alpha=0.8)
        if manual_col in df_manual.columns:
            ax.plot(df_manual["step"], df_manual[manual_col], label="Manual PPO", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved comparison plot to {args.out}")


if __name__ == "__main__":
    main()
