from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_auc(baseline: float, autoencoder: float, generalization: float, out: Path) -> None:
    labels = ["Baseline\n(single attack)", "Autoencoder\n(unsupervised)", "Generalization\n(multi-attack)"]
    values = [baseline, autoencoder, generalization]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(values)), values, color=["#4C72B0", "#55A868", "#C44E52"], edgecolor="#333")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Porównanie wykrywalności nieznanego ataku (ROC-AUC)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=float, default=0.5837)
    p.add_argument("--autoencoder", type=float, default=0.3821)
    p.add_argument("--generalization", type=float, default=0.5928)
    p.add_argument("--out", type=Path, default=Path("results/roc_auc_comparison.png"))
    args = p.parse_args()

    plot_roc_auc(args.baseline, args.autoencoder, args.generalization, args.out)


if __name__ == "__main__":
    main()
