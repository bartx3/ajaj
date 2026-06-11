from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import ConvAutoencoder, train_epoch as train_ae_epoch
from models.generalization import SmallCNN, eval_balanced_binary, train_epoch
from models.supervised_baseline import to_loader
from utils.prepare_fashion_mnist import (
    DEFAULT_NPZ_PATH,
    attack_label,
    attack_x_key,
    compute_partitions,
    load_partitions,
    save_partitions,
)
from utils.training_splits import build_multi_attack_training


class LatentClassifier(nn.Module):
    """Encoder g(x) z autoenkodera + głowa klasyfikacyjna na przestrzeni latent."""

    def __init__(self, latent_dim: int = 64, head_hidden: int = 64, head_depth: int = 1) -> None:
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
        head_layers: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(head_depth):
            head_layers.extend([
                nn.Linear(in_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
            ])
            in_dim = head_hidden
        head_layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z).squeeze(-1)

    @classmethod
    def from_autoencoder(
        cls,
        ae: ConvAutoencoder | None,
        latent_dim: int = 64,
        head_hidden: int = 64,
        head_depth: int = 1,
    ) -> LatentClassifier:
        dim = ae.encoder[-1].out_features if ae is not None else latent_dim
        model = cls(latent_dim=dim, head_hidden=head_hidden, head_depth=head_depth)
        if ae is not None:
            model.encoder.load_state_dict(ae.encoder.state_dict())
        return model


@dataclass(frozen=True)
class Step4Variant:
    """Konfiguracja wariantu kroku 4."""

    key: str
    label: str
    family: str  # "cnn" | "hybrid"
    needs_ae: bool
    freeze_encoder: bool
    load_ae_encoder: bool
    head_hidden: int = 64
    head_depth: int = 1


STEP4_VARIANTS: dict[str, Step4Variant] = {
    "cnn_binary": Step4Variant(
        key="cnn_binary",
        label="CNN binarny (piksele)",
        family="cnn",
        needs_ae=False,
        freeze_encoder=False,
        load_ae_encoder=False,
    ),
    "hybrid_default": Step4Variant(
        key="hybrid_default",
        label="Hybryda: AE encoder + fine-tuning",
        family="hybrid",
        needs_ae=True,
        freeze_encoder=False,
        load_ae_encoder=True,
    ),
    "hybrid_frozen_encoder": Step4Variant(
        key="hybrid_frozen_encoder",
        label="Hybryda: zamrożony encoder AE",
        family="hybrid",
        needs_ae=True,
        freeze_encoder=True,
        load_ae_encoder=True,
    ),
    "hybrid_no_pretrain": Step4Variant(
        key="hybrid_no_pretrain",
        label="Hybryda: encoder od zera (bez AE)",
        family="hybrid",
        needs_ae=False,
        freeze_encoder=False,
        load_ae_encoder=False,
    ),
    "hybrid_deep_head": Step4Variant(
        key="hybrid_deep_head",
        label="Hybryda: zamrożony AE + głęboka głowa",
        family="hybrid",
        needs_ae=True,
        freeze_encoder=True,
        load_ae_encoder=True,
        head_hidden=128,
        head_depth=2,
    ),
    "hybrid_wider_head": Step4Variant(
        key="hybrid_wider_head",
        label="Hybryda: zamrożony AE + szersza głowa",
        family="hybrid",
        needs_ae=True,
        freeze_encoder=True,
        load_ae_encoder=True,
        head_hidden=128,
        head_depth=1,
    ),
}


def pretrain_autoencoder(
    clean_x: np.ndarray,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> ConvAutoencoder:
    torch.manual_seed(seed)
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(clean_x[:, None, :, :].astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    print(f"Pre-trening autoenkodera na clean ({epochs} epok)...")
    for epoch in range(1, epochs + 1):
        loss = train_ae_epoch(model, loader, device, opt, criterion)
        print(f"  ae epoch {epoch:02d}/{epochs}  loss={loss:.6f}")
    return model


def load_or_pretrain_ae(
    clean_x: np.ndarray,
    checkpoint: Path | None,
    latent_dim: int,
    ae_epochs: int,
    batch_size: int,
    ae_lr: float,
    device: torch.device,
    seed: int,
) -> ConvAutoencoder:
    ae = ConvAutoencoder(latent_dim=latent_dim).to(device)
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise SystemExit(f"Brak pliku checkpoint autoenkodera: {checkpoint}")
        ae.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Załadowano wagi autoenkodera z {checkpoint.resolve()}")
        return ae
    if ae_epochs <= 0:
        raise SystemExit("Podaj --ae-checkpoint lub ustaw --ae-epochs > 0.")
    return pretrain_autoencoder(clean_x, latent_dim, ae_epochs, batch_size, ae_lr, device, seed)


def build_step4_model(
    variant: Step4Variant,
    *,
    ae: ConvAutoencoder | None,
    latent_dim: int,
    device: torch.device,
) -> nn.Module:
    if variant.family == "cnn":
        return SmallCNN().to(device)
    ae_for_init = ae if variant.load_ae_encoder else None
    return LatentClassifier.from_autoencoder(
        ae_for_init,
        latent_dim=latent_dim,
        head_hidden=variant.head_hidden,
        head_depth=variant.head_depth,
    ).to(device)


def make_optimizer(
    model: nn.Module,
    variant: Step4Variant,
    head_lr: float,
    encoder_lr: float | None,
) -> torch.optim.Optimizer:
    if variant.family == "cnn":
        return torch.optim.Adam(model.parameters(), lr=head_lr)

    assert isinstance(model, LatentClassifier)
    for p in model.encoder.parameters():
        p.requires_grad = not variant.freeze_encoder
    if variant.freeze_encoder:
        return torch.optim.Adam(model.head.parameters(), lr=head_lr)
    enc_lr = encoder_lr if encoder_lr is not None else head_lr * 0.1
    return torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": enc_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
    )


def train_supervised(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    verbose: bool,
) -> None:
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, device, optimizer, criterion)
        if verbose:
            print(f"  epoch {epoch:02d}/{epochs}  loss={loss:.4f}")


def run_step4(
    data: dict[str, np.ndarray],
    train_attacks: list[int],
    test_attack: int,
    *,
    variant: str | Step4Variant = "hybrid_default",
    ae: ConvAutoencoder | None = None,
    ae_checkpoint: Path | None = None,
    ae_epochs: int = 10,
    ae_lr: float = 1e-3,
    latent: int = 64,
    train_ratio: float = 0.8,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    encoder_lr: float | None = 1e-4,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[nn.Module, dict[str, float | str | list[int]]]:
    """Trenuje wybrany wariant kroku 4 i zwraca metryki ID/OOD."""
    if isinstance(variant, str):
        if variant not in STEP4_VARIANTS:
            raise ValueError(f"Nieznany wariant: {variant}. Dostępne: {list(STEP4_VARIANTS)}")
        variant_cfg = STEP4_VARIANTS[variant]
    else:
        variant_cfg = variant

    if test_attack in train_attacks:
        raise ValueError(f"test_attack {test_attack} nie może być w train_attacks {train_attacks}")

    clean_x = data["clean_x"]
    attack_arrays = [data[attack_x_key(a)] for a in train_attacks]
    x_tr, y_tr, clean_te, test_idxs = build_multi_attack_training(
        clean_x, attack_arrays, train_ratio, seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_ae = ae
    if variant_cfg.needs_ae and shared_ae is None:
        shared_ae = load_or_pretrain_ae(
            clean_x, ae_checkpoint, latent, ae_epochs, batch_size, ae_lr, device, seed
        )
    elif shared_ae is not None:
        shared_ae = shared_ae.to(device)

    torch.manual_seed(seed)
    model = build_step4_model(variant_cfg, ae=shared_ae, latent_dim=latent, device=device)
    optimizer = make_optimizer(model, variant_cfg, lr, encoder_lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = to_loader(x_tr, y_tr, batch_size, shuffle=True)

    if verbose:
        print(
            f"{variant_cfg.label} | train {train_attacks} "
            f"| test {test_attack} ({attack_label(test_attack)})"
        )

    train_supervised(model, loader, device, optimizer, criterion, epochs, verbose)

    id_attack = data[attack_x_key(train_attacks[0])][test_idxs]
    ood_attack = data[attack_x_key(test_attack)][test_idxs]
    id_metrics = eval_balanced_binary(model, clean_te, id_attack, device, batch_size)
    ood_metrics = eval_balanced_binary(model, clean_te, ood_attack, device, batch_size)

    metrics: dict[str, float | str | list[int]] = {
        "variant": variant_cfg.key,
        "variant_label": variant_cfg.label,
        "variant_family": variant_cfg.family,
        "train_attacks": train_attacks,
        "test_attack": test_attack,
        "id_roc_auc": float(id_metrics["roc_auc"]),
        "ood_roc_auc": float(ood_metrics["roc_auc"]),
        "id_balanced_accuracy": float(id_metrics["balanced_accuracy"]),
        "ood_balanced_accuracy": float(ood_metrics["balanced_accuracy"]),
    }
    return model, metrics


def run_hybrid(
    data: dict[str, np.ndarray],
    train_attacks: list[int],
    test_attack: int,
    *,
    freeze_encoder: bool = False,
    variant: str | None = None,
    **kwargs,
) -> tuple[nn.Module, dict[str, float]]:
    """Kompatybilność wsteczna — domyślny wariant hybrid_default lub hybrid_frozen_encoder."""
    if variant is None:
        variant = "hybrid_frozen_encoder" if freeze_encoder else "hybrid_default"
    kwargs.pop("variant", None)
    kwargs.pop("freeze_encoder", None)
    model, metrics = run_step4(
        data, train_attacks, test_attack, variant=variant, **kwargs
    )
    return model, {k: v for k, v in metrics.items() if isinstance(v, (float, int, list))}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Krok 4: porównanie wariantów (AE + klasyfikator vs CNN binarny)."
    )
    p.add_argument("--data", type=Path, default=DEFAULT_NPZ_PATH)
    p.add_argument("--recompute-data", action="store_true")
    p.add_argument("--variant", choices=list(STEP4_VARIANTS), default="hybrid_default")
    p.add_argument("--ae-checkpoint", type=Path, default=None)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--train-attacks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--test-attack", type=int, default=4)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="LR głowy / CNN.")
    p.add_argument("--encoder-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=Path, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.recompute_data or not args.data.is_file():
        partitions, _ = compute_partitions(seed=args.seed)
        save_partitions(partitions, args.data)

    if args.test_attack in args.train_attacks:
        raise SystemExit("--test-attack nie może występować w --train-attacks.")

    raw = load_partitions(args.data)
    model, metrics = run_step4(
        raw,
        args.train_attacks,
        args.test_attack,
        variant=args.variant,
        ae_checkpoint=args.ae_checkpoint,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        latent=args.latent,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        seed=args.seed,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Zapisano model: {args.save}")

    print(f"\n--- Wyniki ({metrics['variant_label']}) ---")
    print(f"In-distribution ROC-AUC={metrics['id_roc_auc']:.4f}")
    print(
        f"Nieznany atak (attack_{args.test_attack} [{attack_label(args.test_attack)}]): "
        f"ROC-AUC={metrics['ood_roc_auc']:.4f}"
    )


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    main()
