from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

# ensure project modules importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "anomalies"))

from anomalies.generalization import SmallCNN, to_loader, _split_idx
from anomalies.prepare_fashion_mnist import load_partitions, attack_x_key, attack_label, DEFAULT_NPZ_PATH


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


def eval_balanced_binary_model(model, x0, x1, device, batch_size=128):
    model.eval()
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(len(x0)), np.ones(len(x1))], axis=0)
    loader = to_loader(x, y, batch_size, shuffle=False)
    scores_list = []
    y_list = []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            logits = model(bx)
            prob = torch.sigmoid(logits).cpu().numpy()
            scores_list.append(prob)
            y_list.append(by.numpy())
    scores = np.concatenate(scores_list)
    y_true = np.concatenate(y_list).astype(np.int64)
    auc = _roc_auc(y_true, scores)
    return auc


def run_experiments(epochs=5, batch_size=128, lr=1e-3, seed=42, out=Path("results/generalization_bar.png")):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    data = load_partitions(DEFAULT_NPZ_PATH)
    clean_x = data["clean_x"].astype(np.float32)
    N = 5  # number of defined attacks (1..5)

    results = []

    for n in range(0, N):
        # train on attacks 1..n (empty if n==0)
        train_attacks = list(range(1, n + 1))
        test_attack = n + 1
        print(f"Experiment n={n}: train_attacks={train_attacks}  test={test_attack}")

        # prepare splits
        total_n = len(clean_x)
        c_tr, c_te = _split_idx(total_n, 0.8, rng)
        test_idxs = c_te

        # collect training data
        x_clean_tr = clean_x[c_tr]
        x_attacks_tr = []
        for a in train_attacks:
            k = attack_x_key(a)
            x_attacks_tr.append(data[k][rng.permutation(total_n)[: len(c_tr)]])

        if len(x_attacks_tr) > 0:
            x_tr = np.concatenate([x_clean_tr] + x_attacks_tr, axis=0)
            y_tr = np.concatenate([np.zeros(len(x_clean_tr))] + [np.ones(len(a)) for a in x_attacks_tr], axis=0)
        else:
            x_tr = x_clean_tr
            y_tr = np.zeros(len(x_tr), dtype=np.float32)

        perm = rng.permutation(len(x_tr))
        x_tr, y_tr = x_tr[perm], y_tr[perm]

        # test data: clean_te vs attack test
        key_test = attack_x_key(test_attack)
        x_clean_te = clean_x[c_te]
        x_test_attack = data[key_test][test_idxs]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SmallCNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        loader = to_loader(x_tr, y_tr, batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            # train epoch
            model.train()
            total_loss = 0.0
            n_samples = 0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * len(bx)
                n_samples += len(bx)
            print(f" n={n} epoch {epoch}/{epochs} loss={total_loss / max(1, n_samples):.4f}")

        auc = eval_balanced_binary_model(model, x_clean_te, x_test_attack, device, batch_size=batch_size)
        print(f" n={n} unknown attack {attack_label(test_attack)} ROC-AUC={auc:.4f}\n")
        results.append(auc)

    # plot bar chart
    labels = [f"n={i}\ntrain 1..{i}" if i>0 else "n=0\n(train none)" for i in range(0, N)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(results)), results, color=["#4C72B0"]*len(results), edgecolor="#333")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Generalization: train on n attacks, test on attack n+1")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    for bar, v in zip(bars, results):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved generalization bar chart to {out}")


if __name__ == "__main__":
    run_experiments(epochs=5)
