"""Wspólne funkcje dla skryptów sweep (kroki 1, 2, 4)."""

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def config_key(train_attacks: list[int]) -> str:
    return ",".join(str(a) for a in train_attacks)


def iter_train_configs(test_attack: int, all_attacks: list[int]) -> list[list[int]]:
    """Unikalne zestawy ataków treningowych (bez test_attack)."""
    others = [a for a in all_attacks if a != test_attack]
    configs: list[list[int]] = []

    for a in others:
        configs.append([a])

    for k in range(2, len(others) + 1):
        configs.append(others[:k])

    if others:
        configs.append(others)

    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for c in configs:
        t = tuple(sorted(c))
        if t not in seen:
            seen.add(t)
            unique.append(sorted(c))
    return unique
