from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.prepare_fashion_mnist import DEFAULT_NPZ_PATH, attack_x_key, attack_label, load_partitions


from utils.training_splits import build_multi_attack_training


def _split_idx(n: int, train_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    from utils.training_splits import split_idx
    return split_idx(n, train_ratio, rng)


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    t_x = torch.from_numpy(x[:, None, :, :].astype(np.float32))
    t_y = torch.from_numpy(y.astype(np.float32))
    return DataLoader(TensorDataset(t_x, t_y), batch_size=batch_size, shuffle=shuffle)


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


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
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


def eval_balanced_binary(model: nn.Module, x0: np.ndarray, x1: np.ndarray, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))], axis=0)
    loader = to_loader(x, y, batch_size, shuffle=False)
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    scores_list = []
    y_list = []
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
    return {"accuracy": acc, "balanced_accuracy": bacc, "roc_auc": auc}


def run_generalization(
    data: dict[str, np.ndarray],
    train_attacks: list[int],
    test_attack: int,
    *,
    train_ratio: float = 0.8,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
    return_model: bool = False,
) -> dict[str, float | int | list[int] | str | SmallCNN]:
    """Trenuje klasyfikator i zwraca metryki ID/OOD (bez argparse)."""
    if test_attack in train_attacks:
        raise ValueError(f"test_attack {test_attack} nie może być w train_attacks {train_attacks}")

    np.random.seed(seed)
    clean_x = data["clean_x"]

    attack_arrays = []
    for a in train_attacks:
        k = attack_x_key(a)
        if k not in data:
            raise KeyError(k)
        attack_arrays.append(data[k])

    x_tr, y_tr, x_clean_te, test_idxs = build_multi_attack_training(
        clean_x, attack_arrays, train_ratio, seed
    )

    key_test = attack_x_key(test_attack)
    if key_test not in data:
        raise KeyError(key_test)
    x_test_attack = data[key_test][test_idxs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = to_loader(x_tr, y_tr, batch_size, shuffle=True)

    if verbose:
        print(
            f"Train {train_attacks} ({', '.join(attack_label(a) for a in train_attacks)}) "
            f"| test {test_attack} ({attack_label(test_attack)})"
        )

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, device, opt, criterion)
        if verbose:
            print(f"  epoch {epoch:02d}/{epochs}  loss={loss:.4f}")

    id_metrics = eval_balanced_binary(
        model, x_clean_te, data[attack_x_key(train_attacks[0])][test_idxs], device, batch_size
    )
    ood_metrics = eval_balanced_binary(model, x_clean_te, x_test_attack, device, batch_size)

    return {
        "train_attacks": train_attacks,
        "test_attack": test_attack,
        "test_attack_label": attack_label(test_attack),
        "id_roc_auc": id_metrics["roc_auc"],
        "id_balanced_accuracy": id_metrics["balanced_accuracy"],
        "ood_roc_auc": ood_metrics["roc_auc"],
        "ood_balanced_accuracy": ood_metrics["balanced_accuracy"],
        **({"model": model} if return_model else {}),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generalizacja: trenowanie klasyfikatora na wielu typach ataków.")
    p.add_argument("--data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--recompute-data", action="store_true")
    p.add_argument("--train-attacks", type=int, nargs="+", default=[1, 2, 3], help="Lista attack_N użytych w treningu")
    p.add_argument("--test-attack", type=int, default=4, help="attack_N jako nieznany atak w teście")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.test_attack in args.train_attacks:
        raise SystemExit("--test-attack nie może występować w --train-attacks (OOD).")

    data = load_partitions(args.data)
    metrics = run_generalization(
        data,
        args.train_attacks,
        args.test_attack,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=True,
    )

    print("--- Wyniki generalizacji ---")
    print(f"In-distribution (clean vs train-attacks holdout): ROC-AUC={metrics['id_roc_auc']:.4f}")
    print(f"Unknown attack (clean vs {metrics['test_attack_label']}): ROC-AUC={metrics['ood_roc_auc']:.4f}")


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
