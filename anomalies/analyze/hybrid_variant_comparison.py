"""
Porównanie wariantów kroku 4 vs klasyfikator CNN (piksele) i domyślna hybryda.

Uruchomienie:
  cd anomalies
  python analyze/hybrid_variant_comparison.py
  python analyze/hybrid_variant_comparison.py --quick
"""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.generalization import run_generalization
from models.hybrid import STEP4_VARIANTS, load_or_pretrain_ae, run_step4
from utils.paths import DEFAULT_AE_CHECKPOINT, DEFAULT_NPZ_PATH, RESULTS_ROOT
from utils.prepare_fashion_mnist import ATTACK_NAMES, attack_label, load_partitions


def run_all_variants(
    data: dict,
    train_attacks: list[int],
    test_attacks: list[int],
    *,
    shared_ae,
    ae_checkpoint: Path | None,
    ae_epochs: int,
    epochs: int,
    batch_size: int,
    lr: float,
    encoder_lr: float,
    seed: int,
    verbose: bool,
) -> list[dict]:
    rows: list[dict] = []
    variant_keys = list(STEP4_VARIANTS.keys())

    for test_attack in test_attacks:
        if test_attack in train_attacks:
            continue
        if verbose:
            print(f"\n######## test OOD = attack_{test_attack} ({attack_label(test_attack)}) ########")

        gen = run_generalization(
            data,
            train_attacks,
            test_attack,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            verbose=False,
        )
        rows.append({
            "variant": "generalization_cnn",
            "variant_label": "Generalizacja (krok 2, CNN)",
            "variant_family": "cnn",
            "train_attacks": train_attacks,
            "test_attack": test_attack,
            "test_attack_label": attack_label(test_attack),
            "id_roc_auc": round(float(gen["id_roc_auc"]), 4),
            "ood_roc_auc": round(float(gen["ood_roc_auc"]), 4),
        })

        for vkey in variant_keys:
            if verbose:
                print(f"\n--- {STEP4_VARIANTS[vkey].label} ---")
            _, metrics = run_step4(
                data,
                train_attacks,
                test_attack,
                variant=vkey,
                ae=shared_ae,
                ae_checkpoint=ae_checkpoint,
                ae_epochs=0 if shared_ae is not None else ae_epochs,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                encoder_lr=encoder_lr,
                seed=seed,
                verbose=verbose,
            )
            rows.append({
                "variant": metrics["variant"],
                "variant_label": metrics["variant_label"],
                "variant_family": metrics["variant_family"],
                "train_attacks": train_attacks,
                "test_attack": test_attack,
                "test_attack_label": attack_label(test_attack),
                "id_roc_auc": round(float(metrics["id_roc_auc"]), 4),
                "ood_roc_auc": round(float(metrics["ood_roc_auc"]), 4),
            })

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant", "variant_label", "variant_family",
        "train_attacks", "test_attack", "test_attack_label",
        "id_roc_auc", "ood_roc_auc",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def plot_ood_bars(rows: list[dict], train_attacks: list[int], out: Path) -> None:
    test_attacks = sorted({r["test_attack"] for r in rows})
    variants = []
    seen: set[str] = set()
    for r in rows:
        if r["variant"] not in seen:
            seen.add(r["variant"])
            variants.append((r["variant"], r["variant_label"]))

    n_v = len(variants)
    n_t = len(test_attacks)
    fig, axes = plt.subplots(1, n_t, figsize=(3.2 * n_t, 5), sharey=True)
    if n_t == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_v))
    for ax, test_a in zip(axes, test_attacks):
        subset = [r for r in rows if r["test_attack"] == test_a]
        x = np.arange(n_v)
        vals = []
        labels_short = []
        for vkey, vlabel in variants:
            row = next(r for r in subset if r["variant"] == vkey)
            vals.append(row["ood_roc_auc"])
            labels_short.append(vkey.replace("hybrid_", "h_").replace("cnn_", "c_")[:14])
        bars = ax.bar(x, vals, color=colors, edgecolor="#333", linewidth=0.5)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, rotation=55, ha="right", fontsize=7)
        ax.set_title(f"test {test_a}\n{attack_label(test_a)[:14]}", fontsize=9)
        ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=6)

    axes[0].set_ylabel("OOD ROC-AUC")
    fig.suptitle(
        f"Krok 4 — warianty modelu (train={train_attacks})",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ood_heatmap(rows: list[dict], out: Path) -> None:
    variants = []
    seen: set[str] = set()
    for r in rows:
        if r["variant"] not in seen:
            seen.add(r["variant"])
            variants.append(r["variant"])
    test_attacks = sorted({r["test_attack"] for r in rows})
    mat = np.full((len(test_attacks), len(variants)), np.nan)
    labels = []
    for j, v in enumerate(variants):
        row0 = next(r for r in rows if r["variant"] == v)
        labels.append(row0["variant_label"][:28])

    for r in rows:
        i = test_attacks.index(r["test_attack"])
        j = variants.index(r["variant"])
        mat[i, j] = r["ood_roc_auc"]

    fig, ax = plt.subplots(figsize=(max(10, len(variants) * 1.1), 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(test_attacks)))
    ax.set_yticklabels([f"{a} {attack_label(a)[:12]}" for a in test_attacks], fontsize=8)
    ax.set_title("OOD ROC-AUC — warianty kroku 4 vs generalizacja")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mean_ood_ranking(rows: list[dict], out: Path) -> None:
    variants = []
    seen: set[str] = set()
    for r in rows:
        if r["variant"] not in seen:
            seen.add(r["variant"])
            variants.append((r["variant"], r["variant_label"]))

    means = []
    for vkey, vlabel in variants:
        subset = [r["ood_roc_auc"] for r in rows if r["variant"] == vkey]
        means.append((vkey, vlabel, float(np.mean(subset))))

    means.sort(key=lambda t: t[2], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(means))
    vals = [m[2] for m in means]
    colors = ["#8172B3" if m[0].startswith("hybrid") else "#4C72B0" for m in means]
    ax.barh(x, vals, color=colors, edgecolor="#333")
    ax.axvline(0.5, color="gray", linestyle=":")
    ax.set_yticks(x)
    ax.set_yticklabels([m[1][:32] for m in means], fontsize=8)
    ax.set_xlabel("Średnie OOD ROC-AUC (po wszystkich test_attack)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Ranking wariantów — która konstrukcja najlepiej pasuje do zadania?")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_summary_md(rows: list[dict], train_attacks: list[int], path: Path) -> None:
    variants = []
    seen: set[str] = set()
    for r in rows:
        if r["variant"] not in seen:
            seen.add(r["variant"])
            variants.append((r["variant"], r["variant_label"]))

    lines = [
        "# Porównanie wariantów kroku 4\n",
        f"Trening: ataki {train_attacks} + clean.\n\n",
        "## Średnie OOD ROC-AUC\n",
        "| Wariant | Średnie OOD | Opis |\n",
        "|---------|-------------|------|\n",
    ]
    means = []
    for vkey, vlabel in variants:
        subset = [r["ood_roc_auc"] for r in rows if r["variant"] == vkey]
        m = float(np.mean(subset))
        means.append((vkey, vlabel, m))
    means.sort(key=lambda t: t[2], reverse=True)
    for vkey, vlabel, m in means:
        lines.append(f"| {vkey} | {m:.4f} | {vlabel} |\n")

    best = means[0]
    lines.extend([
        "\n## Wnioski (automatyczne)\n",
        f"- **Najlepszy średni OOD:** `{best[0]}` ({best[2]:.4f}) — {best[1]}.\n",
        "- **cnn_binary** = ten sam CNN co krok 2, ale w ramach eksperymentu kroku 4.\n",
        "- **generalization_cnn** = oficjalny krok 2 (powinien być zbliżony do cnn_binary).\n",
        "- **hybrid_default** = obecny model (AE encoder + fine-tuning głowy).\n",
        "- Jeśli hybryda z zamrożonym encoderem ≈ hybryda z fine-tuning → reprezentacja AE wystarczy; "
        "jeśli fine-tuning wyraźnie lepszy → warto dostosować encoder do detekcji.\n",
        "- Jeśli **cnn_binary** ≥ hybrydy → prosty klasyfikator na pikselach wystarczy dla tego zadania.\n",
    ])
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Porównanie wariantów konstrukcji modelu (krok 4).")
    p.add_argument("--data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--ae-checkpoint", type=Path, default=DEFAULT_AE_CHECKPOINT)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--train-attacks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--test-attacks", type=int, nargs="+", default=None,
                   help="Domyślnie wszystkie ataki spoza train-attacks.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "hybrid_variants")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.epochs = 3
        args.ae_epochs = 3

    all_attacks = sorted(ATTACK_NAMES.keys())
    test_attacks = args.test_attacks or [a for a in all_attacks if a not in args.train_attacks]

    data = load_partitions(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_ckpt = args.ae_checkpoint if args.ae_checkpoint.is_file() else None
    shared_ae = None
    if ae_ckpt is not None:
        shared_ae = load_or_pretrain_ae(
            data["clean_x"], ae_ckpt, 64, 0, args.batch_size, 1e-3, device, args.seed
        )
        print(f"Wspólny AE: {ae_ckpt.resolve()}")
    elif args.ae_epochs > 0:
        shared_ae = load_or_pretrain_ae(
            data["clean_x"], None, 64, args.ae_epochs, args.batch_size, 1e-3, device, args.seed
        )

    rows = run_all_variants(
        data,
        args.train_attacks,
        test_attacks,
        shared_ae=shared_ae,
        ae_checkpoint=ae_ckpt,
        ae_epochs=args.ae_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        seed=args.seed,
        verbose=not args.quiet,
    )

    out_dir = args.out_dir
    save_csv(rows, out_dir / "variant_comparison.csv")
    with (out_dir / "variant_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    plot_ood_bars(rows, args.train_attacks, out_dir / "ood_roc_auc_by_variant.png")
    plot_ood_heatmap(rows, out_dir / "ood_roc_auc_heatmap.png")
    plot_mean_ood_ranking(rows, out_dir / "mean_ood_ranking.png")
    write_summary_md(rows, args.train_attacks, out_dir / "VARIANT_COMPARISON.md")

    print("\n| variant | test | OOD ROC-AUC |")
    print("|---------|------|-------------|")
    for r in sorted(rows, key=lambda x: (x["test_attack"], x["variant"])):
        print(f"| {r['variant'][:20]} | {r['test_attack']} | {r['ood_roc_auc']:.4f} |")
    print(f"\nZapisano: {out_dir.resolve()}")


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
