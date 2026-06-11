"""
Krok 4 — sweep kombinacji treningowych dla systemu hybrydowego (AE + klasyfikator).

Analogicznie do generalization_sweep.py, ale z pre-treningiem AE i fine-tuningiem.
Dodatkowo porównanie hybryda vs baseline vs generalizacja na tych samych konfiguracjach.
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

import torch

from models.generalization import run_generalization
from models.hybrid import load_or_pretrain_ae, run_hybrid
from models.supervised_baseline import run_baseline
from utils.paths import DEFAULT_AE_CHECKPOINT, DEFAULT_NPZ_PATH, RESULTS_ROOT
from utils.prepare_fashion_mnist import ATTACK_NAMES, attack_label, load_partitions
from utils.sweep import config_key, iter_train_configs


def run_sweep(
    data: dict,
    all_attacks: list[int],
    *,
    ae,
    ae_checkpoint: Path | None,
    ae_epochs: int,
    epochs: int,
    batch_size: int,
    lr: float,
    encoder_lr: float,
    freeze_encoder: bool,
    seed: int,
    verbose: bool,
    compare_baselines: bool,
) -> list[dict]:
    rows: list[dict] = []
    for test_attack in all_attacks:
        for train_attacks in iter_train_configs(test_attack, all_attacks):
            if verbose:
                print(f"\n=== hybrid train={train_attacks} | test={test_attack} ===")

            _, metrics = run_hybrid(
                data,
                train_attacks,
                test_attack,
                ae=ae,
                ae_checkpoint=ae_checkpoint,
                ae_epochs=0 if ae is not None else ae_epochs,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                encoder_lr=encoder_lr,
                freeze_encoder=freeze_encoder,
                seed=seed,
                verbose=verbose,
            )
            row: dict = {
                "train_attacks": config_key(train_attacks),
                "n_train_attacks": len(train_attacks),
                "test_attack": test_attack,
                "test_attack_label": attack_label(test_attack),
                "hybrid_id_roc_auc": round(metrics["id_roc_auc"], 4),
                "hybrid_ood_roc_auc": round(metrics["ood_roc_auc"], 4),
            }

            if compare_baselines:
                if len(train_attacks) == 1:
                    bl = run_baseline(
                        data, train_attacks[0], test_attack,
                        epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, verbose=False,
                    )
                    row["baseline_ood_roc_auc"] = round(float(bl["ood_roc_auc"]), 4)
                gen = run_generalization(
                    data, train_attacks, test_attack,
                    epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, verbose=False,
                )
                row["generalization_ood_roc_auc"] = round(float(gen["ood_roc_auc"]), 4)
                row["hybrid_minus_gen"] = round(row["hybrid_ood_roc_auc"] - row["generalization_ood_roc_auc"], 4)
                if len(train_attacks) == 1:
                    row["hybrid_minus_baseline"] = round(
                        row["hybrid_ood_roc_auc"] - row["baseline_ood_roc_auc"], 4
                    )

            rows.append(row)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_heatmap(rows: list[dict], all_attacks: list[int], out: Path, metric: str, title: str) -> None:
    configs = sorted({r["train_attacks"] for r in rows}, key=lambda s: (len(s), s))
    mat = np.full((len(all_attacks), len(configs)), np.nan)
    for r in rows:
        if metric not in r:
            continue
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
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_comparison_bars(rows: list[dict], all_attacks: list[int], out: Path) -> None:
    """Dla treningu pojedynczego ataku: hybryda vs baseline vs generalizacja."""
    single = [r for r in rows if r["n_train_attacks"] == 1 and "baseline_ood_roc_auc" in r]
    if not single:
        return

    fig, axes = plt.subplots(1, len(all_attacks), figsize=(3 * len(all_attacks), 4), sharey=True)
    if len(all_attacks) == 1:
        axes = [axes]

    for ax, test_a in zip(axes, all_attacks):
        subset = [r for r in single if r["test_attack"] == test_a]
        if not subset:
            ax.axis("off")
            continue
        train_vals = sorted(subset, key=lambda r: int(r["train_attacks"]))
        x = np.arange(len(train_vals))
        w = 0.25
        ax.bar(x - w, [r["baseline_ood_roc_auc"] for r in train_vals], w, label="baseline", color="#4C72B0")
        ax.bar(x, [r["generalization_ood_roc_auc"] for r in train_vals], w, label="gen.", color="#C44E52")
        ax.bar(x + w, [r["hybrid_ood_roc_auc"] for r in train_vals], w, label="hybrid", color="#8172B3")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"tr{r['train_attacks']}" for r in train_vals], fontsize=8)
        ax.set_title(f"test {test_a}\n{attack_label(test_a)[:12]}", fontsize=9)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("OOD ROC-AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Krok 4 — hybryda vs baseline vs generalizacja (1 atak w treningu)", fontsize=11, y=1.08)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hybrid_gain(rows: list[dict], out: Path) -> None:
    """Różnica OOD: hybryda − generalizacja."""
    if "hybrid_minus_gen" not in rows[0]:
        return
    gains = [r["hybrid_minus_gen"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gains, bins=20, color="#8172B3", edgecolor="#333", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("hybrid OOD − generalization OOD")
    ax.set_ylabel("Liczba konfiguracji")
    ax.set_title("Krok 4 — czy hybryda poprawia wynik względem samej generalizacji?")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def print_markdown_table(rows: list[dict]) -> None:
    has_cmp = "generalization_ood_roc_auc" in rows[0] if rows else False
    if has_cmp:
        print("\n| train | test | hybrid OOD | gen. OOD | Δ(h−g) | baseline OOD |")
        print("|-------|------|------------|----------|--------|--------------|")
        for r in sorted(rows, key=lambda x: (x["test_attack"], x["train_attacks"])):
            bl = f"{r['baseline_ood_roc_auc']:.3f}" if "baseline_ood_roc_auc" in r else "—"
            print(
                f"| {r['train_attacks']} | {r['test_attack']} ({r['test_attack_label']}) "
                f"| {r['hybrid_ood_roc_auc']:.3f} | {r['generalization_ood_roc_auc']:.3f} "
                f"| {r['hybrid_minus_gen']:+.3f} | {bl} |"
            )
    else:
        print("\n| train | test | hybrid ID | hybrid OOD |")
        print("|-------|------|-----------|------------|")
        for r in sorted(rows, key=lambda x: (x["test_attack"], x["train_attacks"])):
            print(
                f"| {r['train_attacks']} | {r['test_attack']} ({r['test_attack_label']}) "
                f"| {r['hybrid_id_roc_auc']:.3f} | {r['hybrid_ood_roc_auc']:.3f} |"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep hybrydy (krok 4): kombinacje train vs unknown test.")
    p.add_argument("--data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--ae-checkpoint", type=Path, default=DEFAULT_AE_CHECKPOINT)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder-lr", type=float, default=1e-4)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "hybrid_sweep")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-compare", action="store_true", help="Pomiń porównanie z baseline/generalization.")
    args = p.parse_args()

    if args.quick:
        args.epochs = 3
        args.ae_epochs = 3

    data = load_partitions(args.data)
    all_attacks = sorted(ATTACK_NAMES.keys())

    ae_ckpt = args.ae_checkpoint if args.ae_checkpoint.is_file() else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_x = data["clean_x"]
    shared_ae = None
    if ae_ckpt is not None:
        shared_ae = load_or_pretrain_ae(
            clean_x, ae_ckpt, 64, 0, args.batch_size, 1e-3, device, args.seed
        )
        print(f"Wspólny encoder AE: {ae_ckpt.resolve()}")
    elif args.ae_epochs > 0:
        shared_ae = load_or_pretrain_ae(
            clean_x, None, 64, args.ae_epochs, args.batch_size, 1e-3, device, args.seed
        )
        print(f"Wytrenowano wspólny AE ({args.ae_epochs} epok) dla całego sweepu.")

    rows = run_sweep(
        data,
        all_attacks,
        ae=shared_ae,
        ae_checkpoint=ae_ckpt,
        ae_epochs=args.ae_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
        verbose=not args.quiet,
        compare_baselines=not args.no_compare,
    )

    out_dir = args.out_dir
    save_csv(rows, out_dir / "hybrid_sweep.csv")
    plot_heatmap(
        rows, all_attacks, out_dir / "hybrid_ood_roc_auc_heatmap.png",
        "hybrid_ood_roc_auc", "Krok 4 — hybryda: OOD ROC-AUC",
    )
    plot_heatmap(
        rows, all_attacks, out_dir / "hybrid_id_roc_auc_heatmap.png",
        "hybrid_id_roc_auc", "Krok 4 — hybryda: ID ROC-AUC",
    )
    if not args.no_compare:
        plot_comparison_bars(rows, all_attacks, out_dir / "hybrid_vs_baselines_single_train.png")
        plot_hybrid_gain(rows, out_dir / "hybrid_gain_vs_generalization.png")
    print_markdown_table(rows)
    print(f"\nZapisano: {out_dir.resolve()}")


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
