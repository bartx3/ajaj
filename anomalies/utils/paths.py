"""Stałe ścieżki względem katalogu anomalies/ i repozytorium."""

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

ANOMALIES_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ANOMALIES_ROOT.parent

DEFAULT_NPZ_PATH = ANOMALIES_ROOT / "data" / "fmnist_partitions.npz"
DEFAULT_PYTHIA_NPZ = ANOMALIES_ROOT / "data" / "pythia_28x28.npz"
DEFAULT_PYTHIA_ROOT = REPO_ROOT / "Pythia"
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_AE_CHECKPOINT = RESULTS_ROOT / "models" / "autoencoder_lat64.pth"
