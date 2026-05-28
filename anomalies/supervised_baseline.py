from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_fashion_mnist import (
    DEFAULT_NPZ_PATH,
    attack_label,
    attack_x_key,
    compute_partitions,
    load_partitions,
    save_partitions,
)


def _split_idx(n: int, train_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(n * train_ratio)
    order = rng.permutation(n)
    return order[:n_train], order[n_train:]


def build_train_test(
    clean_x: np.ndarray,
    attack_a_x: np.ndarray,
    attack_b_x: np.ndarray,
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wspólny holdout `clean` dla testów na attack_a i attack_b."""
    rng = np.random.default_rng(seed)
    n = len(clean_x)
    if len(attack_a_x) != n or len(attack_b_x) != n:
        raise ValueError("Oczekiwane partycje tej samej długości co clean.")
    c_tr, c_te = _split_idx(n, train_ratio, rng)
    a_tr, a_te = _split_idx(n, train_ratio, rng)
    _, b_te = _split_idx(n, train_ratio, rng)

    x0_tr, x1_tr = clean_x[c_tr], attack_a_x[a_tr]
    x_tr = np.concatenate([x0_tr, x1_tr], axis=0)
    y_tr = np.concatenate([np.zeros(len(x0_tr)), np.ones(len(x1_tr))], axis=0)
    p = rng.permutation(len(x_tr))
    clean_te = clean_x[c_te]
    attack_a_te = attack_a_x[a_te]
    attack_b_te = attack_b_x[b_te]
    return x_tr[p], y_tr[p], clean_te, attack_a_te, attack_b_te


def to_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    t_x = torch.from_numpy(x[:, None, :, :].astype(np.float32))
    t_y = torch.from_numpy(y.astype(np.float32))
    return DataLoader(
        TensorDataset(t_x, t_y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def eval_balanced_binary(
    model: nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))], axis=0)
    loader = to_loader(x, y, batch_size, shuffle=False)
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    scores_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            logits = model(bx)
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob >= 0.5).astype(np.int64)
            by_np = by.numpy().astype(np.int64)
            scores_list.append(prob)
            y_list.append(by_np)
            for c in (0, 1):
                m = by_np == c
                total[c] += int(m.sum())
                correct[c] += int((pred[m] == c).sum())
    scores = np.concatenate(scores_list)
    y_true = np.concatenate(y_list)
    acc = (correct[0] + correct[1]) / max(1, total[0] + total[1])
    bacc = 0.5 * (correct[0] / max(1, total[0]) + correct[1] / max(1, total[1]))
    auc = _roc_auc(y_true, scores)
    return {"accuracy": acc, "balanced_accuracy": bacc, "roc_auc": auc, "n0": total[0], "n1": total[1]}


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


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(bx)
        n += len(bx)
    return total_loss / max(1, n)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Krok 1: baseline nadzorowany — dane z prepare_fashion_mnist (partycje .npz lub compute_partitions)."
    )
    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_NPZ_PATH,
        help="Ścieżka do .npz z `prepare_fashion_mnist.save_partitions` (domyślnie ta sama co w prepare).",
    )
    p.add_argument(
        "--recompute-data",
        action="store_true",
        help="Pomiń plik: policz partycje przez `compute_partitions` (identyczna logika ataków) i zapisz pod --data.",
    )
    p.add_argument("--attack-a", type=int, default=1, help="attack_N jako attack_a w treningu.")
    p.add_argument("--attack-b", type=int, default=2, help="attack_N jako nieznany attack_b w teście.")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=Path, default=None, help="Path to save trained model state_dict (optional)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.recompute_data or not args.data.is_file():
        if not args.recompute_data and not args.data.is_file():
            print(f"Brak {args.data} — liczę partycje przez prepare_fashion_mnist.compute_partitions...")
        partitions, _ = compute_partitions(seed=args.seed)
        save_partitions(partitions, args.data)

    raw = load_partitions(args.data)
    clean_x = raw["clean_x"]
    key_a = attack_x_key(args.attack_a)
    key_b = attack_x_key(args.attack_b)
    if key_a not in raw or key_b not in raw:
        raise SystemExit(f"W pliku brakuje {key_a} lub {key_b}. Dostępne: {sorted(raw)}")
    attack_a_x = raw[key_a]
    attack_b_x = raw[key_b]

    x_tr, y_tr, clean_te, attack_a_te, attack_b_te = build_train_test(
        clean_x, attack_a_x, attack_b_x, args.train_ratio, args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = to_loader(x_tr, y_tr, args.batch_size, shuffle=True)

    name_a = attack_label(args.attack_a)
    name_b = attack_label(args.attack_b)
    print(
        f"Dane: {args.data.resolve()} (prepare_fashion_mnist) | urządzenie: {device} | train: {len(x_tr)} próbek "
        f"(clean vs attack_{args.attack_a}={name_a}) | holdout: clean={len(clean_te)}, "
        f"attack_{args.attack_a}={len(attack_a_te)}, attack_{args.attack_b}={len(attack_b_te)}"
    )

    for epoch in range(1, args.epochs + 1):
        loss = train(model, loader, device, optimizer, criterion)
        print(f"  epoch {epoch:02d}/{args.epochs}  loss={loss:.4f}")
    # optionally save model
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Saved supervised baseline state_dict to {args.save}")

    id_metrics = eval_balanced_binary(model, clean_te, attack_a_te, device, args.batch_size)
    ood_metrics = eval_balanced_binary(model, clean_te, attack_b_te, device, args.batch_size)

    print("\n--- Wyniki ---")
    print(
        f"In-distribution (clean vs attack_{args.attack_a} [{name_a}], holdout): "
        f"acc={id_metrics['accuracy']:.4f}  "
        f"balanced_acc={id_metrics['balanced_accuracy']:.4f}  "
        f"ROC-AUC={id_metrics['roc_auc']:.4f}"
    )
    print(
        f"Nieznany atak (clean vs attack_{args.attack_b} [{name_b}], holdout): "
        f"acc={ood_metrics['accuracy']:.4f}  "
        f"balanced_acc={ood_metrics['balanced_accuracy']:.4f}  "
        f"ROC-AUC={ood_metrics['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
