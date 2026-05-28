from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prepare_fashion_mnist import DEFAULT_NPZ_PATH, attack_x_key, attack_label, load_partitions


def _split_idx(n: int, train_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(n * train_ratio)
    order = rng.permutation(n)
    return order[:n_train], order[n_train:]


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


def main() -> None:
    p = argparse.ArgumentParser(description="Generalizacja: trenowanie klasyfikatora na wielu typach ataków.")
    p.add_argument("--data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--recompute-data", action="store_true")
    p.add_argument("--train-attacks", type=int, nargs="+", default=[1], help="Lista attack_N użytych w treningu")
    p.add_argument("--test-attack", type=int, default=2, help="attack_N jako nieznany atak w teście")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = load_partitions(args.data)
    clean_x = data["clean_x"]
    n = len(clean_x)

    # collect training attacks
    attack_arrays = []
    for a in args.train_attacks:
        k = attack_x_key(a)
        if k not in data:
            raise SystemExit(f"Missing {k} in {args.data}")
        attack_arrays.append(data[k])

    # split indices
    c_tr, c_te = _split_idx(n, args.train_ratio, rng)
    a_tr_idxs = [rng.permutation(n)[: int(n * args.train_ratio)] for _ in attack_arrays]
    _, test_idxs = _split_idx(n, args.train_ratio, rng)

    # build train: clean_tr + all attack_tr concatenated
    x_clean_tr = clean_x[c_tr]
    x_attacks_tr = [arr[a_tr] for arr, a_tr in zip(attack_arrays, a_tr_idxs)]
    x_tr = np.concatenate([x_clean_tr] + x_attacks_tr, axis=0)
    y_tr = np.concatenate([np.zeros(len(x_clean_tr))] + [np.ones(len(a)) for a in x_attacks_tr], axis=0)
    perm = rng.permutation(len(x_tr))
    x_tr, y_tr = x_tr[perm], y_tr[perm]

    # test: clean_te vs test_attack
    key_test = attack_x_key(args.test_attack)
    if key_test not in data:
        raise SystemExit(f"Missing {key_test} in {args.data}")
    x_clean_te = clean_x[c_te]
    x_test_attack = data[key_test][test_idxs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = to_loader(x_tr, y_tr, args.batch_size, shuffle=True)

    print(f"Train on attacks: {[attack_label(a) for a in args.train_attacks]} | test unknown: {attack_label(args.test_attack)}")

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, device, opt, criterion)
        print(f"epoch {epoch:02d}/{args.epochs}  loss={loss:.4f}")

    id_metrics = eval_balanced_binary(model, x_clean_te, data[attack_x_key(args.train_attacks[0])][test_idxs], device, args.batch_size)
    ood_metrics = eval_balanced_binary(model, x_clean_te, x_test_attack, device, args.batch_size)

    print("--- Wyniki generalizacji ---")
    print(f"In-distribution (clean vs train-attacks holdout): ROC-AUC={id_metrics['roc_auc']:.4f}")
    print(f"Unknown attack (clean vs {attack_label(args.test_attack)}): ROC-AUC={ood_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
