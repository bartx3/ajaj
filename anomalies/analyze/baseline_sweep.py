"""
Krok 1 — sweep par (attack_a w treningu, attack_b jako nieznany OOD).

Dla każdego pojedynczego ataku w treningu testuje wykrywalność pozostałych ataków.
Generuje CSV + heatmapy ROC-AUC (ID i OOD) + wykres ID vs OOD.
"""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models.supervised_baseline import run_baseline
from utils.paths import DEFAULT_NPZ_PATH, RESULTS_ROOT
from utils.prepare_fashion_mnist import ATTACK_NAMES, attack_label, load_partitions


def run_sweep(
    data: dict,
    all_attacks: list[int],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    verbose: bool,
) -> list[dict]:
    rows: list[dict] = []
    for attack_a in all_attacks:
        for attack_b in all_attacks:
            if attack_a == attack_b:
                continue
            if verbose:
                print(f"\n=== train attack_{attack_a} | OOD test attack_{attack_b} ===")
            metrics = run_baseline(
                data,
                attack_a,
                attack_b,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
                verbose=verbose,
            )
            rows.append({
                "attack_a": attack_a,
                "attack_a_label": attack_label(attack_a),
                "attack_b": attack_b,
                "attack_b_label": attack_label(attack_b),
                "id_roc_auc": round(float(metrics["id_roc_auc"]), 4),
                "ood_roc_auc": round(float(metrics["ood_roc_auc"]), 4),
                "id_balanced_accuracy": round(float(metrics["id_balanced_accuracy"]), 4),
                "ood_balanced_accuracy": round(float(metrics["ood_balanced_accuracy"]), 4),
            })
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "attack_a", "attack_a_label", "attack_b", "attack_b_label",
        "id_roc_auc", "ood_roc_auc", "id_balanced_accuracy", "ood_balanced_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_heatmap(
    rows: list[dict],
    all_attacks: list[int],
    out: Path,
    *,
    row_key: str,
    col_key: str,
    metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    mat = np.full((len(all_attacks), len(all_attacks)), np.nan)
    for r in rows:
        i = all_attacks.index(r[row_key])
        j = all_attacks.index(r[col_key])
        mat[i, j] = r[metric]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    labels = [f"{a}\n{attack_label(a)[:12]}" for a in all_attacks]
    ax.set_xticks(range(len(all_attacks)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(all_attacks)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(len(all_attacks)):
        for j in range(len(all_attacks)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            elif not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_id_vs_ood_scatter(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ids = [r["id_roc_auc"] for r in rows]
    oods = [r["ood_roc_auc"] for r in rows]
    ax.scatter(ids, oods, alpha=0.75, c="#4C72B0", edgecolors="#333")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    for r in rows:
        if r["ood_roc_auc"] > 0.65 or r["ood_roc_auc"] < 0.45:
            ax.annotate(
                f"a{r['attack_a']}→b{r['attack_b']}",
                (r["id_roc_auc"], r["ood_roc_auc"]),
                fontsize=7,
                alpha=0.85,
            )
    ax.set_xlim(0.9, 1.01)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ID ROC-AUC (clean vs znany attack_a)")
    ax.set_ylabel("OOD ROC-AUC (clean vs nieznany attack_b)")
    ax.set_title("Krok 1 — baseline: ID vs OOD dla wszystkich par ataków")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_best_ood_per_train(rows: list[dict], all_attacks: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bests = []
    for a in all_attacks:
        subset = [r for r in rows if r["attack_a"] == a]
        best = max(subset, key=lambda r: r["ood_roc_auc"])
        bests.append(best)
    x = np.arange(len(all_attacks))
    vals = [b["ood_roc_auc"] for b in bests]
    ax.bar(x, vals, color="#4C72B0", edgecolor="#333")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"train {b['attack_a']}\n{b['attack_a_label'][:12]}" for b in bests], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("OOD ROC-AUC (najlepszy test_b)")
    ax.set_title("Krok 1 — najlepszy nieznany atak dla każdego attack_a w treningu")
    for i, b in enumerate(bests):
        ax.text(i, vals[i] + 0.02, f"test {b['attack_b']}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def print_markdown_table(rows: list[dict]) -> None:
    print("\n| train (a) | test OOD (b) | ID ROC-AUC | OOD ROC-AUC |")
    print("|-----------|--------------|------------|-------------|")
    for r in sorted(rows, key=lambda x: (x["attack_a"], x["attack_b"])):
        print(
            f"| {r['attack_a']} ({r['attack_a_label']}) | {r['attack_b']} ({r['attack_b_label']}) "
            f"| {r['id_roc_auc']:.3f} | {r['ood_roc_auc']:.3f} |"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep baseline (krok 1): wszystkie pary attack_a → attack_b.")
    p.add_argument("--data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "baseline_sweep")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.epochs = 3

    all_attacks = sorted(ATTACK_NAMES.keys())
    data = load_partitions(args.data)
    rows = run_sweep(
        data, all_attacks,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, verbose=not args.quiet,
    )

    out_dir = args.out_dir
    save_csv(rows, out_dir / "baseline_sweep.csv")
    plot_heatmap(
        rows, all_attacks, out_dir / "ood_roc_auc_heatmap.png",
        row_key="attack_b", col_key="attack_a", metric="ood_roc_auc",
        title="Krok 1 — baseline: OOD ROC-AUC",
        xlabel="Atak w treningu (attack_a)", ylabel="Nieznany atak testowy (attack_b)",
    )
    plot_heatmap(
        rows, all_attacks, out_dir / "id_roc_auc_heatmap.png",
        row_key="attack_b", col_key="attack_a", metric="id_roc_auc",
        title="Krok 1 — baseline: ID ROC-AUC (clean vs attack_a)",
        xlabel="Atak w treningu (attack_a)", ylabel="Atak testowy (attack_b)",
    )
    plot_id_vs_ood_scatter(rows, out_dir / "id_vs_ood_scatter.png")
    plot_best_ood_per_train(rows, all_attacks, out_dir / "best_ood_per_train_attack.png")
    print_markdown_table(rows)
    print(f"\nZapisano: {out_dir.resolve()}")


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
