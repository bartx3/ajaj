from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_fashion_mnist import DEFAULT_NPZ_PATH, attack_x_key, load_partitions


def to_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    t_x = torch.from_numpy(x[:, None, :, :].astype(np.float32))
    return DataLoader(tensor_dataset := TensorDataset(t_x), batch_size=batch_size, shuffle=shuffle)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, opt, criterion) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for (bx,) in loader:
        bx = bx.to(device)
        opt.zero_grad()
        recon = model(bx)
        loss = criterion(recon, bx)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * len(bx)
        n += len(bx)
    return total_loss / max(1, n)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(np.int64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_ranks = ranks[y == 1].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def eval_reconstruction_roc(model: nn.Module, clean: np.ndarray, attack: np.ndarray, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    x = np.concatenate([clean, attack], axis=0)
    y = np.concatenate([np.zeros(len(clean)), np.ones(len(attack))], axis=0)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:, None, :, :].astype(np.float32))), batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            recon = model(bx)
            mse = torch.mean((recon - bx) ** 2, dim=(1, 2, 3)).cpu().numpy()
            scores.append(mse)
    scores = np.concatenate(scores)
    auc = _roc_auc(y, scores)
    clean_scores = scores[y == 0]
    attack_scores = scores[y == 1]
    # summary statistics
    stats = {
        "roc_auc": auc,
        "clean_mean_mse": float(np.mean(clean_scores)),
        "attack_mean_mse": float(np.mean(attack_scores)),
        "clean_median_mse": float(np.median(clean_scores)),
        "attack_median_mse": float(np.median(attack_scores)),
        "clean_std_mse": float(np.std(clean_scores)),
        "attack_std_mse": float(np.std(attack_scores)),
    }
    print("-- reconstruction MSE stats --")
    print(f"clean: mean={stats['clean_mean_mse']:.6f}  median={stats['clean_median_mse']:.6f}  std={stats['clean_std_mse']:.6f}")
    print(f"attack: mean={stats['attack_mean_mse']:.6f}  median={stats['attack_median_mse']:.6f}  std={stats['attack_std_mse']:.6f}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Krok 3: Autoenkoder — trenowanie na clean i test rekonstrukcji na atakach.")
    p.add_argument("--data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--recompute-data", action="store_true")
    p.add_argument("--attack-b", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=Path, default=None, help="Path to save trained model state_dict (optional)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_partitions(args.data)
    clean_x = data["clean_x"]
    key_b = attack_x_key(args.attack_b)
    if key_b not in data:
        raise SystemExit(f"Missing {key_b} in {args.data}")
    attack_b = data[key_b]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(latent_dim=args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(torch.from_numpy(clean_x[:, None, :, :].astype(np.float32))), batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, device, opt, criterion)
        print(f"epoch {epoch:02d}/{args.epochs}  loss={loss:.6f}")

    # optionally save model
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Saved autoencoder state_dict to {args.save}")

    metrics = eval_reconstruction_roc(model, clean_x, attack_b, device, args.batch_size)
    print("--- Wyniki autoenkodera ---")
    print(f"ROC-AUC (reconstruction error, clean vs attack_{args.attack_b}) = {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
