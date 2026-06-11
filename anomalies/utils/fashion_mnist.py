"""Szybki podgląd partycji zapisanych przez prepare_fashion_mnist."""

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from utils.bootstrap import setup
from utils.paths import DEFAULT_NPZ_PATH
from utils.prepare_fashion_mnist import attack_x_key, load_partitions

setup()

data = load_partitions(DEFAULT_NPZ_PATH)

loaded_clean_x = data["clean_x"]
loaded_attack_1_x = data[attack_x_key(1)]
loaded_attack_5_x = data[attack_x_key(5)]

print(f"Loaded Clean Partition Shape: {loaded_clean_x.shape}")
print(f"Loaded Attack 1 Shape: {loaded_attack_1_x.shape}")
print(f"Loaded Attack 5 (OOD) Shape: {loaded_attack_5_x.shape}")
