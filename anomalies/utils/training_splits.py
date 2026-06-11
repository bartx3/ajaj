"""Wspólne podziały train/holdout dla klasyfikatorów binarnych."""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np


def split_idx(n: int, train_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(n * train_ratio)
    order = rng.permutation(n)
    return order[:n_train], order[n_train:]


def build_multi_attack_training(
    clean_x: np.ndarray,
    attack_arrays: list[np.ndarray],
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Zwraca (x_train, y_train, clean_holdout, test_holdout_indices).
    test_holdout_indices — indeksy holdout wspólne do ewaluacji clean vs dowolny atak.
    """
    rng = np.random.default_rng(seed)
    n = len(clean_x)
    c_tr, c_te = split_idx(n, train_ratio, rng)
    a_tr_idxs = [rng.permutation(n)[: int(n * train_ratio)] for _ in attack_arrays]
    _, test_idxs = split_idx(n, train_ratio, rng)

    x_clean_tr = clean_x[c_tr]
    x_attacks_tr = [arr[idx] for arr, idx in zip(attack_arrays, a_tr_idxs)]
    x_tr = np.concatenate([x_clean_tr] + x_attacks_tr, axis=0)
    y_tr = np.concatenate(
        [np.zeros(len(x_clean_tr), dtype=np.float32)]
        + [np.ones(len(a), dtype=np.float32) for a in x_attacks_tr]
    )
    perm = rng.permutation(len(x_tr))
    return x_tr[perm], y_tr[perm], clean_x[c_te], test_idxs
