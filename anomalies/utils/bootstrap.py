"""Dodaje katalog anomalies/ do sys.path przy uruchamianiu skryptów bez -m."""
from __future__ import annotations

import sys
from pathlib import Path


def setup() -> Path:
    root = Path(__file__).resolve().parents[1]
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
