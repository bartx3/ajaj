"""Uruchamia kroki 1–4 z jednolitymi hiperparametrami i zapisuje wyniki OOD."""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import ConvAutoencoder, eval_reconstruction_roc, train_epoch as train_ae_epoch
from models.generalization import SmallCNN, run_generalization, train_epoch
from models.hybrid import run_hybrid
from models.supervised_baseline import build_train_test, eval_balanced_binary, to_loader
from utils.paths import DEFAULT_NPZ_PATH, RESULTS_ROOT
from utils.prepare_fashion_mnist import attack_label, attack_x_key, load_partitions


def run_baseline_ood(
    data: dict,
    device: torch.device,
    test_attack: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> float:
    x_tr, y_tr, clean_te, _, attack_b_te = build_train_test(
        data["clean_x"], data[attack_x_key(1)], data[attack_x_key(test_attack)], 0.8, seed
    )
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = to_loader(x_tr, y_tr, batch_size, True)
    for _ in range(epochs):
        train_epoch(model, loader, device, opt, nn.BCEWithLogitsLoss())
    return float(eval_balanced_binary(model, clean_te, attack_b_te, device, batch_size)["roc_auc"])


def run_ae_ood(
    data: dict,
    device: torch.device,
    test_attack: int,
    ae_epochs: int,
    batch_size: int,
    lr: float,
    latent: int,
    score_mode: str,
) -> float:
    clean = data["clean_x"]
    model = ConvAutoencoder(latent_dim=latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(clean[:, None, :, :].astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(ae_epochs):
        train_ae_epoch(model, loader, device, opt, nn.MSELoss())
    attack = data[attack_x_key(test_attack)]
    return float(eval_reconstruction_roc(model, clean, attack, device, batch_size, score_mode)["roc_auc"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_NPZ_PATH)
    p.add_argument("--test-attack", type=int, default=4)
    p.add_argument("--train-attacks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--score-mode", default="deviation")
    p.add_argument("--out", default=RESULTS_ROOT / "benchmark_results.json")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.epochs = 3
        args.ae_epochs = 3

    if args.test_attack in args.train_attacks:
        raise SystemExit("--test-attack nie może być w --train-attacks.")

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_partitions(args.data)

    print(f"Benchmark OOD: test attack_{args.test_attack} ({attack_label(args.test_attack)})")

    baseline = run_baseline_ood(data, device, args.test_attack, args.epochs, args.batch_size, args.lr, 42)
    print(f"  baseline: {baseline:.4f}")

    ae = run_ae_ood(data, device, args.test_attack, args.ae_epochs, args.batch_size, args.lr, args.latent, args.score_mode)
    print(f"  autoencoder ({args.score_mode}): {ae:.4f}")

    gen = run_generalization(
        data, args.train_attacks, args.test_attack,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=42, verbose=False,
    )
    print(f"  generalization: {gen['ood_roc_auc']:.4f}")

    _, hybrid = run_hybrid(
        data, args.train_attacks, args.test_attack,
        ae_epochs=args.ae_epochs, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=42, verbose=False,
    )
    print(f"  hybrid: {hybrid['ood_roc_auc']:.4f}")

    results = {
        "test_attack": args.test_attack,
        "test_attack_label": attack_label(args.test_attack),
        "train_attacks": args.train_attacks,
        "ood_roc_auc": {
            "baseline_step1": round(baseline, 4),
            "autoencoder_step3": round(ae, 4),
            "generalization_step2": round(float(gen["ood_roc_auc"]), 4),
            "hybrid_step4": round(hybrid["ood_roc_auc"], 4),
        },
        "ae_score_mode": args.score_mode,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Zapisano {args.out.resolve()}")


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
