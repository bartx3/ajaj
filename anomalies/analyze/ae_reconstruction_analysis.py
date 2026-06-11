"""
Analiza reakcji autoenkodera (krok 3) na różne typy błędów.

Score(x) = sum_{i,j} (x_ij - f(g(x))_ij)^2  — suma kwadratów różnic po pikselach.
Porównanie: clean vs attack_1..5 (ROC-AUC, rozkłady SSE, mapy średniego błędu |x - recon|).
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

from models.autoencoder import (
    ConvAutoencoder,
    _roc_auc,
    anomaly_scores_from_sse,
    per_image_reconstruction_sse,
    train_epoch,
)
from utils.paths import DEFAULT_AE_CHECKPOINT, DEFAULT_NPZ_PATH, RESULTS_ROOT
from utils.prepare_fashion_mnist import ATTACK_NAMES, attack_label, attack_x_key, load_partitions
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


def load_or_train_ae(
    clean_x: np.ndarray,
    checkpoint: Path | None,
    latent_dim: int,
    ae_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> ConvAutoencoder:
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    if checkpoint is not None and checkpoint.is_file():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Załadowano AE: {checkpoint.resolve()}")
        return model
    if ae_epochs <= 0:
        raise SystemExit("Podaj --checkpoint lub --ae-epochs > 0.")
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(clean_x[:, None, :, :].astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    print(f"Trening AE na clean ({ae_epochs} epok)...")
    for epoch in range(1, ae_epochs + 1):
        loss = train_epoch(model, loader, device, opt, criterion)
        print(f"  epoch {epoch:02d}/{ae_epochs}  loss={loss:.6f}")
    return model


def mean_abs_error_map(model: ConvAutoencoder, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """Średnia |x - recon| po próbkach, kształt (H, W)."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[:, None, :, :].astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    acc = np.zeros(x.shape[1:], dtype=np.float64)
    n = 0
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            recon = model(bx).cpu().numpy()[:, 0]
            orig = bx.cpu().numpy()[:, 0]
            acc += np.abs(orig - recon).sum(axis=0)
            n += len(bx)
    return (acc / max(1, n)).astype(np.float32)


def analyze_partition(
    model: ConvAutoencoder,
    clean_x: np.ndarray,
    attack_x: np.ndarray,
    device: torch.device,
    batch_size: int,
    score_mode: str = "deviation",
) -> dict[str, float | np.ndarray]:
    clean_sse = per_image_reconstruction_sse(model, clean_x, device, batch_size)
    attack_sse = per_image_reconstruction_sse(model, attack_x, device, batch_size)
    scores = np.concatenate([
        anomaly_scores_from_sse(clean_sse, clean_sse, score_mode),
        anomaly_scores_from_sse(attack_sse, clean_sse, score_mode),
    ])
    y = np.concatenate([np.zeros(len(clean_sse)), np.ones(len(attack_sse))])
    return {
        "clean_sse": clean_sse,
        "attack_sse": attack_sse,
        "roc_auc": _roc_auc(y, scores),
        "score_mode": score_mode,
        "clean_mean": float(np.mean(clean_sse)),
        "clean_median": float(np.median(clean_sse)),
        "attack_mean": float(np.mean(attack_sse)),
        "attack_median": float(np.median(attack_sse)),
        "delta_mean": float(np.mean(attack_sse) - np.mean(clean_sse)),
    }


def save_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "attack_id",
        "attack_label",
        "roc_auc",
        "clean_mean_sse",
        "clean_median_sse",
        "attack_mean_sse",
        "attack_median_sse",
        "delta_mean_sse",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})


def plot_sse_distributions(results: dict[int, dict], out: Path) -> None:
    attack_ids = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(attack_ids) + 1)
    clean_sse = results[attack_ids[0]]["clean_sse"]
    data = [clean_sse]
    labels = ["clean"]
    for aid in attack_ids:
        data.append(results[aid]["attack_sse"])
        labels.append(f"{aid}\n{attack_label(aid)[:12]}")
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    colors = ["#4C72B0"] + ["#DD8452"] * len(attack_ids)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("SSE = sum_ij (x - recon)^2")
    ax.set_title("Rozkład błędu rekonstrukcji AE (odszumiacz krok 3)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mean_sse_bars(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rows))
    clean_ref = rows[0]["clean_mean_sse"]
    attack_means = [r["attack_mean_sse"] for r in rows]
    labels = [f"{r['attack_id']}\n{r['attack_label'][:10]}" for r in rows]
    bars = ax.bar(x, attack_means, color="#C44E52", alpha=0.85, label="atak")
    ax.axhline(clean_ref, color="#4C72B0", linestyle="--", linewidth=2, label=f"clean (mean={clean_ref:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Średnie SSE na próbkę")
    ax.set_title("Średni błąd rekonstrukcji vs clean")
    ax.legend()
    for bar, v in zip(bars, attack_means):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc_auc_by_attack(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(rows))
    aucs = [r["roc_auc"] for r in rows]
    labels = [f"{r['attack_id']}" for r in rows]
    colors = ["#55A868" if a >= 0.5 else "#C44E52" for a in aucs]
    ax.bar(x, aucs, color=colors, edgecolor="#333")
    ax.axhline(0.5, color="gray", linestyle=":", label="losowy (0.5)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lid}\n{r['attack_label'][:12]}" for lid, r in zip(labels, rows)], fontsize=8)
    ax.set_ylabel("ROC-AUC (SSE jako score anomalii)")
    ax.set_title("Wykrywalność: clean vs każdy typ ataku")
    ax.legend()
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_error_maps(
    model: ConvAutoencoder,
    clean_x: np.ndarray,
    data: dict[str, np.ndarray],
    attack_ids: list[int],
    device: torch.device,
    batch_size: int,
    out: Path,
    n_show: int = 3,
) -> None:
    n_rows = 1 + len(attack_ids)
    n_cols = n_show
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    clean_map = mean_abs_error_map(model, clean_x[: min(2000, len(clean_x))], device, batch_size)
    for c in range(n_cols):
        ax = axes[0, c]
        ax.imshow(clean_x[c], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"clean #{c}", fontsize=8)
        ax.axis("off")
    axes[0, 0].set_ylabel("Wejście", fontsize=9)

    for r, aid in enumerate(attack_ids, start=1):
        ax = axes[r, 0]
        ax.set_ylabel(f"atk {aid}\n|err| map", fontsize=8)
        atk = data[attack_x_key(aid)]
        err_map = mean_abs_error_map(model, atk[: min(2000, len(atk))], device, batch_size)
        for c in range(n_cols):
            axes[r, c].imshow(atk[c], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
        # error map in last column span — use separate row: simplify to show map in col 0 only
        if n_cols >= 2:
            axes[r, 1].imshow(err_map, cmap="hot", vmin=0, vmax=max(0.01, err_map.max()))
            axes[r, 1].set_title(f"śr.|x-recon| ({attack_label(aid)[:10]})", fontsize=7)
            axes[r, 1].axis("off")
            for c in range(2, n_cols):
                axes[r, c].axis("off")

    fig.suptitle("Przykłady wejść i mapy średniego błędu rekonstrukcji", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Analiza SSE rekonstrukcji AE dla wszystkich typów ataków.")
    p.add_argument("--data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_AE_CHECKPOINT)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "ae_analysis")
    p.add_argument("--score-mode", choices=("deviation", "sse"), default="deviation")
    args = p.parse_args()

    data = load_partitions(args.data)
    clean_x = data["clean_x"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.checkpoint if args.checkpoint.is_file() else None
    if ckpt is None and args.checkpoint:
        print(f"Brak {args.checkpoint} — trenuję AE od zera.")
    model = load_or_train_ae(
        clean_x, ckpt, args.latent, args.ae_epochs, args.batch_size, args.lr, device, args.seed
    )

    attack_ids = sorted(ATTACK_NAMES.keys())
    results: dict[int, dict] = {}
    csv_rows: list[dict] = []

    for aid in attack_ids:
        key = attack_x_key(aid)
        if key not in data:
            continue
        stats = analyze_partition(model, clean_x, data[key], device, args.batch_size, args.score_mode)
        results[aid] = stats
        csv_rows.append({
            "attack_id": aid,
            "attack_label": attack_label(aid),
            "roc_auc": round(stats["roc_auc"], 4),
            "clean_mean_sse": round(stats["clean_mean"], 4),
            "clean_median_sse": round(stats["clean_median"], 4),
            "attack_mean_sse": round(stats["attack_mean"], 4),
            "attack_median_sse": round(stats["attack_median"], 4),
            "delta_mean_sse": round(stats["delta_mean"], 4),
        })
        print(
            f"attack_{aid} ({attack_label(aid)}): ROC-AUC={stats['roc_auc']:.4f}  "
            f"mean_SSE clean={stats['clean_mean']:.2f} attack={stats['attack_mean']:.2f}  "
            f"Δmean={stats['delta_mean']:+.2f}"
        )

    out_dir = args.out_dir
    save_summary_csv(csv_rows, out_dir / "reconstruction_sse_summary.csv")
    plot_sse_distributions(results, out_dir / "sse_distributions.png")
    plot_mean_sse_bars(csv_rows, out_dir / "sse_mean_by_attack.png")
    plot_roc_auc_by_attack(csv_rows, out_dir / "roc_auc_by_attack.png")
    plot_error_maps(model, clean_x, data, attack_ids, device, args.batch_size, out_dir / "error_maps_samples.png")

    print(f"\nZapisano wyniki w {out_dir.resolve()}")


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
