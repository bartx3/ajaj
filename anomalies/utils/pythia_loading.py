"""Ładowanie zbioru ukrytego Pythia (70×70 PNG → tensory 28×28 dla modeli FMNIST)."""
from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from pathlib import Path

import numpy as np
from PIL import Image

from utils.paths import DEFAULT_PYTHIA_NPZ, DEFAULT_PYTHIA_ROOT

DEFAULT_PYTHIA_ROOT = DEFAULT_PYTHIA_ROOT
DEFAULT_PYTHIA_NPZ = DEFAULT_PYTHIA_NPZ
ATTACK_LETTERS = list("abcdefgh")


def attack_letter_key(letter: str) -> str:
    return f"attack_{letter.lower()}_x"


def list_attack_keys() -> list[str]:
    return [attack_letter_key(c) for c in ATTACK_LETTERS]


def _load_folder(folder: Path, size: int) -> np.ndarray:
    paths = sorted(folder.glob("*.png"), key=lambda p: int(p.stem))
    if not paths:
        raise FileNotFoundError(f"Brak PNG w {folder}")
    rows: list[np.ndarray] = []
    for p in paths:
        im = Image.open(p).convert("L")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BILINEAR)
        rows.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(rows, axis=0)


def load_pythia(root: str | Path | None = None, size: int = 28) -> dict[str, np.ndarray]:
    root = Path(root) if root is not None else DEFAULT_PYTHIA_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"Brak katalogu Pythia: {root}")

    data: dict[str, np.ndarray] = {"clean_x": _load_folder(root / "clean", size)}
    for letter in ATTACK_LETTERS:
        folder = root / f"attack_{letter}"
        if not folder.is_dir():
            raise FileNotFoundError(f"Brak {folder}")
        data[attack_letter_key(letter)] = _load_folder(folder, size)
    return data


def save_pythia_npz(data: dict[str, np.ndarray], path: str | Path | None = None) -> Path:
    path = Path(path) if path is not None else DEFAULT_PYTHIA_NPZ
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path


def load_pythia_npz(path: str | Path | None = None) -> dict[str, np.ndarray]:
    path = Path(path) if path is not None else DEFAULT_PYTHIA_NPZ
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
