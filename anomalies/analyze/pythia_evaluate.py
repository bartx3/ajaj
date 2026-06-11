"""
Ewaluacja modeli (kroki 1–4) na zbiorze ukrytym Pythia.

Dwa tryby:
  transfer — modele wytrenowane na Fashion-MNIST, test na Pythia (28×28);
  native   — trening na Pythia (jak w labie: clean vs attack_a, test pozostałe).
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import (
    ConvAutoencoder,
    anomaly_scores_from_sse,
    per_image_reconstruction_sse,
    train_epoch as train_ae_epoch,
    _roc_auc,
)
from models.generalization import SmallCNN, run_generalization, train_epoch, to_loader
from models.hybrid import run_hybrid
from models.supervised_baseline import build_train_test, eval_balanced_binary
from utils.paths import DEFAULT_NPZ_PATH, DEFAULT_PYTHIA_NPZ, RESULTS_ROOT
from utils.prepare_fashion_mnist import attack_x_key, load_partitions
from utils.pythia_loading import ATTACK_LETTERS, attack_letter_key, load_pythia, load_pythia_npz, save_pythia_npz


def eval_roc_clean_vs_attack(
    model: nn.Module,
    clean: np.ndarray,
    attack: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    ae_scores: bool = False,
    ae_score_mode: str = "deviation",
) -> float:
    if ae_scores:
        clean_sse = per_image_reconstruction_sse(model, clean, device, batch_size)
        atk_sse = per_image_reconstruction_sse(model, attack, device, batch_size)
        scores = np.concatenate([
            anomaly_scores_from_sse(clean_sse, clean_sse, ae_score_mode),
            anomaly_scores_from_sse(atk_sse, clean_sse, ae_score_mode),
        ])
        y = np.concatenate([np.zeros(len(clean)), np.ones(len(attack))])
        return _roc_auc(y, scores)
    m = eval_balanced_binary(model, clean, attack, device, batch_size)
    return float(m["roc_auc"])


def train_baseline_fmnist(
    fmnist: dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    attack_a: int = 1,
    attack_b: int = 2,
) -> SmallCNN:
    x_tr, y_tr, _, _, _ = build_train_test(
        fmnist["clean_x"],
        fmnist[attack_x_key(attack_a)],
        fmnist[attack_x_key(attack_b)],
        0.8,
        seed,
    )
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = to_loader(x_tr, y_tr, batch_size, True)
    for _ in range(epochs):
        train_epoch(model, loader, device, opt, nn.BCEWithLogitsLoss())
    return model


def train_ae_on_clean(clean_x: np.ndarray, device: torch.device, epochs: int, batch_size: int, lr: float, latent: int) -> ConvAutoencoder:
    model = ConvAutoencoder(latent_dim=latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(clean_x[:, None, :, :].astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    crit = nn.MSELoss()
    for _ in range(epochs):
        train_ae_epoch(model, loader, device, opt, crit)
    return model


def train_hybrid_fmnist(
    fmnist: dict[str, np.ndarray],
    ae: ConvAutoencoder,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    train_attacks: list[int] | None = None,
) -> nn.Module:
    train_attacks = train_attacks or [1, 2, 3]
    test_attack = next(a for a in range(1, 6) if a not in train_attacks)
    model, _ = run_hybrid(
        fmnist,
        train_attacks,
        test_attack,
        ae=ae.to(device),
        ae_epochs=0,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        verbose=False,
    )
    return model


def eval_all_pythia_attacks(
    model: nn.Module,
    pythia: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    *,
    ae_scores: bool = False,
    ae_score_mode: str = "deviation",
) -> dict[str, float]:
    clean = pythia["clean_x"]
    out: dict[str, float] = {}
    for letter in ATTACK_LETTERS:
        key = attack_letter_key(letter)
        out[letter] = eval_roc_clean_vs_attack(
            model, clean, pythia[key], device, batch_size,
            ae_scores=ae_scores, ae_score_mode=ae_score_mode,
        )
    return out


def train_native_baseline(pythia: dict[str, np.ndarray], device: torch.device, epochs: int, batch_size: int, lr: float, seed: int) -> SmallCNN:
    return train_baseline_fmnist(
        {"clean_x": pythia["clean_x"], attack_x_key(1): pythia[attack_letter_key("a")], attack_x_key(2): pythia[attack_letter_key("b")]},
        device,
        epochs,
        batch_size,
        lr,
        seed,
        attack_a=1,
        attack_b=2,
    )


def run_transfer_suite(
    fmnist: dict[str, np.ndarray],
    pythia: dict[str, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    rows: list[dict] = []
    print("\n=== Transfer: modele FMNIST → test Pythia ===")

    baseline = train_baseline_fmnist(fmnist, device, args.epochs, args.batch_size, args.lr, args.seed)
    for letter, auc in eval_all_pythia_attacks(baseline, pythia, device, args.batch_size).items():
        rows.append({"mode": "transfer", "model": "baseline_step1", "train_info": "FMNIST clean vs attack_1", "test_attack": letter, "roc_auc": round(auc, 4)})

    ae = train_ae_on_clean(fmnist["clean_x"], device, args.ae_epochs, args.batch_size, args.lr, args.latent)
    for letter, auc in eval_all_pythia_attacks(ae, pythia, device, args.batch_size, ae_scores=True).items():
        rows.append({"mode": "transfer", "model": "autoencoder_step3", "train_info": "FMNIST clean only", "test_attack": letter, "roc_auc": round(auc, 4)})

    gen_out = run_generalization(
        fmnist, [1, 2, 3], 4,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        verbose=False, return_model=True,
    )
    gen_model = gen_out["model"]
    for letter, auc in eval_all_pythia_attacks(gen_model, pythia, device, args.batch_size).items():
        rows.append({"mode": "transfer", "model": "generalization_step2", "train_info": "FMNIST attacks 1,2,3", "test_attack": letter, "roc_auc": round(auc, 4)})

    hybrid = train_hybrid_fmnist(fmnist, ae, device, args.epochs, args.batch_size, args.lr, args.seed)
    for letter, auc in eval_all_pythia_attacks(hybrid, pythia, device, args.batch_size).items():
        rows.append({"mode": "transfer", "model": "hybrid_step4", "train_info": "FMNIST AE + clean vs attack_1", "test_attack": letter, "roc_auc": round(auc, 4)})

    return rows


def run_native_suite(
    pythia: dict[str, np.ndarray],
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    rows: list[dict] = []
    print("\n=== Native: trening na Pythia (clean vs attack_a) → test pozostałe ===")

    baseline = train_native_baseline(pythia, device, args.epochs, args.batch_size, args.lr, args.seed)
    for letter in ATTACK_LETTERS:
        if letter == "a":
            continue
        auc = eval_roc_clean_vs_attack(baseline, pythia["clean_x"], pythia[attack_letter_key(letter)], device, args.batch_size)
        rows.append({
            "mode": "native",
            "model": "baseline_step1",
            "train_info": "Pythia clean vs attack_a",
            "test_attack": letter,
            "roc_auc": round(auc, 4),
        })

    ae = train_ae_on_clean(pythia["clean_x"], device, args.ae_epochs, args.batch_size, args.lr, args.latent)
    for letter, auc in eval_all_pythia_attacks(ae, pythia, device, args.batch_size, ae_scores=True).items():
        rows.append({"mode": "native", "model": "autoencoder_step3", "train_info": "Pythia clean only", "test_attack": letter, "roc_auc": round(auc, 4)})

    # generalization on Pythia: train a,b,c test d..h
    train_letters = ["a", "b", "c"]
    pythia_as_fmnist = {"clean_x": pythia["clean_x"]}
    for i, letter in enumerate(ATTACK_LETTERS, start=1):
        pythia_as_fmnist[attack_x_key(i)] = pythia[attack_letter_key(letter)]
    gen_out = run_generalization(
        pythia_as_fmnist, [1, 2, 3], 4,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        verbose=False, return_model=True,
    )
    gen_model = gen_out["model"]
    for letter in ATTACK_LETTERS:
        if letter in train_letters:
            continue
        auc = eval_roc_clean_vs_attack(
            gen_model, pythia["clean_x"], pythia[attack_letter_key(letter)], device, args.batch_size
        )
        rows.append({
            "mode": "native",
            "model": "generalization_step2",
            "train_info": "Pythia attacks a,b,c",
            "test_attack": letter,
            "roc_auc": round(auc, 4),
        })

    hybrid = train_hybrid_fmnist(
        {"clean_x": pythia["clean_x"], attack_x_key(1): pythia[attack_letter_key("a")], attack_x_key(2): pythia[attack_letter_key("b")], attack_x_key(3): pythia[attack_letter_key("c")]},
        ae,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
        train_attacks=[1],
    )
    for letter in ATTACK_LETTERS:
        if letter == "a":
            continue
        auc = eval_roc_clean_vs_attack(hybrid, pythia["clean_x"], pythia[attack_letter_key(letter)], device, args.batch_size)
        rows.append({
            "mode": "native",
            "model": "hybrid_step4",
            "train_info": "Pythia AE + clean vs attack_a",
            "test_attack": letter,
            "roc_auc": round(auc, 4),
        })

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "model", "train_info", "test_attack", "roc_auc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_heatmaps(rows: list[dict], out_dir: Path) -> None:
    models = sorted({r["model"] for r in rows})
    for mode in ("transfer", "native"):
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
        if len(models) == 1:
            axes = [axes]
        for ax, model in zip(axes, models):
            mr = [r for r in subset if r["model"] == model]
            letters = ATTACK_LETTERS if mode == "transfer" else [c for c in ATTACK_LETTERS if any(x["test_attack"] == c for x in mr)]
            vals = []
            labels = []
            for letter in letters:
                hit = [r for r in mr if r["test_attack"] == letter]
                if hit:
                    vals.append(hit[0]["roc_auc"])
                    labels.append(letter)
            colors = ["#55A868" if v >= 0.5 else "#C44E52" for v in vals]
            ax.bar(range(len(vals)), vals, color=colors, edgecolor="#333")
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_title(model.replace("_", "\n"), fontsize=9)
            ax.set_ylabel("ROC-AUC" if ax is axes[0] else "")
            for i, v in enumerate(vals):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        fig.suptitle(f"Pythia — tryb {mode}: clean vs atak (wyżej = lepiej)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"pythia_{mode}_roc_auc.png", dpi=150)
        plt.close(fig)


def write_summary_md(rows: list[dict], path: Path) -> None:
    lines = ["# Wyniki na zbiorze Pythia\n", "Metryka: ROC-AUC (clean=0 vs atak=1).\n"]
    for mode in ("transfer", "native"):
        lines.append(f"\n## Tryb `{mode}`\n\n")
        lines.append("| Model | Trening | " + " | ".join(ATTACK_LETTERS) + " |\n")
        lines.append("|-------|---------|" + "|".join(["------"] * len(ATTACK_LETTERS)) + "|\n")
        for model in sorted({r["model"] for r in rows if r["mode"] == mode}):
            mr = [r for r in rows if r["mode"] == mode and r["model"] == model]
            if not mr:
                continue
            train_info = mr[0]["train_info"]
            cells = []
            for letter in ATTACK_LETTERS:
                hit = [r for r in mr if r["test_attack"] == letter]
                cells.append(f"{hit[0]['roc_auc']:.3f}" if hit else "—")
            lines.append(f"| {model} | {train_info} | " + " | ".join(cells) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Ewaluacja kroków 1–4 na Pythia.")
    p.add_argument("--pythia-root", type=Path, default=None)
    p.add_argument("--pythia-npz", type=Path, default=DEFAULT_PYTHIA_NPZ)
    p.add_argument("--fmnist-data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "pythia")
    p.add_argument("--transfer-only", action="store_true")
    p.add_argument("--native-only", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.epochs = 3
        args.ae_epochs = 3

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pythia_npz.is_file():
        pythia = load_pythia_npz(args.pythia_npz)
        print(f"Pythia z cache: {args.pythia_npz}")
    else:
        pythia = load_pythia(args.pythia_root)
        path = save_pythia_npz(pythia, args.pythia_npz)
        print(f"Zapisano Pythia 28×28: {path}")

    fmnist = load_partitions(args.fmnist_data)
    rows: list[dict] = []

    if not args.native_only:
        rows.extend(run_transfer_suite(fmnist, pythia, device, args))
    if not args.transfer_only:
        rows.extend(run_native_suite(pythia, device, args))

    out_dir = args.out_dir
    save_csv(rows, out_dir / "pythia_results.csv")
    plot_heatmaps(rows, out_dir)
    write_summary_md(rows, out_dir / "PYTHIA_RESULTS.md")

    print(f"\nZapisano: {out_dir.resolve()}")
    print(f"  - pythia_results.csv")
    print(f"  - PYTHIA_RESULTS.md")
    print(f"  - pythia_transfer_roc_auc.png / pythia_native_roc_auc.png")


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
