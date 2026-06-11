"""
Krok 2 — sweep kombinacji treningowych ataków vs nieznany atak testowy.

Generuje CSV + heatmapy ROC-AUC (OOD) dla wielu konfiguracji:
  - pojedyncze ataki w treningu,
  - prefiksy [1], [1,2], …,
  - wszystkie ataki oprócz testowego.
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

from models.generalization import run_generalization
from utils.paths import DEFAULT_NPZ_PATH, RESULTS_ROOT
from utils.prepare_fashion_mnist import ATTACK_NAMES, attack_label, load_partitions
from utils.sweep import config_key, iter_train_configs


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
    for test_attack in all_attacks:
        for train_attacks in iter_train_configs(test_attack, all_attacks):
            if verbose:
                print(f"\n=== train={train_attacks} | test={test_attack} ===")
            metrics = run_generalization(
                data,
                train_attacks,
                test_attack,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
                verbose=verbose,
            )
            rows.append({
                "train_attacks": config_key(train_attacks),
                "n_train_attacks": len(train_attacks),
                "test_attack": test_attack,
                "test_attack_label": attack_label(test_attack),
                "id_roc_auc": round(float(metrics["id_roc_auc"]), 4),
                "ood_roc_auc": round(float(metrics["ood_roc_auc"]), 4),
                "ood_balanced_accuracy": round(float(metrics["ood_balanced_accuracy"]), 4),
            })
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "train_attacks",
        "n_train_attacks",
        "test_attack",
        "test_attack_label",
        "id_roc_auc",
        "ood_roc_auc",
        "ood_balanced_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_heatmap(rows: list[dict], all_attacks: list[int], out: Path, metric: str = "ood_roc_auc") -> None:
    """Macierz: wiersz = test_attack, kolumna = skrót konfiguracji treningowej."""
    configs = sorted({config_key([int(x) for x in r["train_attacks"].split(",")]) for r in rows}, key=lambda s: (len(s), s))
    mat = np.full((len(all_attacks), len(configs)), np.nan)
    for r in rows:
        i = all_attacks.index(r["test_attack"])
        j = configs.index(r["train_attacks"])
        mat[i, j] = r[metric]

    fig_w = max(8, len(configs) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(all_attacks)))
    ax.set_yticklabels([f"{a} {attack_label(a)[:14]}" for a in all_attacks], fontsize=8)
    ax.set_xlabel("Ataki w treningu (bez testowego)")
    ax.set_ylabel("Nieznany atak (test)")
    ax.set_title(f"Krok 2 — generalizacja: {metric} (clean vs nieznany atak)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_best_per_test(rows: list[dict], all_attacks: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bests = []
    for t in all_attacks:
        subset = [r for r in rows if r["test_attack"] == t]
        best = max(subset, key=lambda r: r["ood_roc_auc"])
        bests.append(best)
    x = np.arange(len(all_attacks))
    vals = [b["ood_roc_auc"] for b in bests]
    ax.bar(x, vals, color="#4C72B0", edgecolor="#333")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"test {b['test_attack']}\n{b['test_attack_label'][:12]}" for b in bests], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("OOD ROC-AUC")
    ax.set_title("Najlepsza konfiguracja treningowa dla każdego nieznanego ataku")
    for i, b in enumerate(bests):
        ax.text(i, vals[i] + 0.02, f"train=[{b['train_attacks']}]", ha="center", fontsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def print_markdown_table(rows: list[dict]) -> None:
    print("\n| train_attacks | test | OOD ROC-AUC | ID ROC-AUC |")
    print("|---------------|------|-------------|------------|")
    for r in sorted(rows, key=lambda x: (x["test_attack"], x["train_attacks"])):
        print(
            f"| {r['train_attacks']} | {r['test_attack']} ({r['test_attack_label']}) "
            f"| {r['ood_roc_auc']:.3f} | {r['id_roc_auc']:.3f} |"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep generalizacji (krok 2): kombinacje train vs unknown test.")
    p.add_argument("--data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "generalization_sweep")
    p.add_argument("--quick", action="store_true", help="3 epoki — szybki podgląd.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.epochs = 3

    all_attacks = sorted(ATTACK_NAMES.keys())
    data = load_partitions(args.data)
    rows = run_sweep(
        data,
        all_attacks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=not args.quiet,
    )

    out_dir = args.out_dir
    save_csv(rows, out_dir / "generalization_sweep.csv")
    plot_heatmap(rows, all_attacks, out_dir / "ood_roc_auc_heatmap.png")
    plot_best_per_test(rows, all_attacks, out_dir / "best_ood_per_test_attack.png")
    print_markdown_table(rows)
    print(f"\nZapisano: {out_dir.resolve()}")


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
