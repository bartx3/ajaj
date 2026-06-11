from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from utils.paths import RESULTS_ROOT


def plot_roc_auc(values: dict[str, float], out: Path, title: str) -> None:
    labels = {
        "baseline_step1": "Baseline\n(1 attack)",
        "autoencoder_step3": "Autoencoder\n(unsupervised)",
        "generalization_step2": "Generalization\n(multi-attack)",
        "hybrid_step4": "Hybrid\n(AE + classifier)",
    }
    keys = [k for k in labels if k in values]
    names = [labels[k] for k in keys]
    vals = [values[k] for k in keys]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"][: len(keys)]

    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.8), 5))
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="#333")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="losowy (0.5)")
    ax.set_ylabel("ROC-AUC (OOD)")
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default=RESULTS_ROOT / "benchmark_results.json")
    p.add_argument("--out", default=RESULTS_ROOT / "roc_auc_comparison.png")
    p.add_argument("--baseline", type=float, default=None)
    p.add_argument("--autoencoder", type=float, default=None)
    p.add_argument("--generalization", type=float, default=None)
    p.add_argument("--hybrid", type=float, default=None)
    args = p.parse_args()

    if args.benchmark.is_file():
        data = json.loads(args.benchmark.read_text(encoding="utf-8"))
        values = data["ood_roc_auc"]
        title = (
            f"OOD: clean vs attack_{data['test_attack']} ({data['test_attack_label']})"
        )
    else:
        values = {
            "baseline_step1": args.baseline if args.baseline is not None else 0.5837,
            "autoencoder_step3": args.autoencoder if args.autoencoder is not None else 0.3821,
            "generalization_step2": args.generalization if args.generalization is not None else 0.5928,
            "hybrid_step4": args.hybrid if args.hybrid is not None else 0.5310,
        }
        title = "Porównanie wykrywalności nieznanego ataku (ROC-AUC)"

    plot_roc_auc(values, args.out, title)


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
