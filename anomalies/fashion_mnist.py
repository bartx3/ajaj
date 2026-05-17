"""Szybki podgląd partycji zapisanych przez prepare_fashion_mnist."""

from prepare_fashion_mnist import DEFAULT_NPZ_PATH, attack_x_key, load_partitions

data = load_partitions(DEFAULT_NPZ_PATH)

loaded_clean_x = data["clean_x"]
loaded_attack_1_x = data[attack_x_key(1)]
loaded_attack_5_x = data[attack_x_key(5)]

print(f"Loaded Clean Partition Shape: {loaded_clean_x.shape}")
print(f"Loaded Attack 1 Shape: {loaded_attack_1_x.shape}")
print(f"Loaded Attack 5 (OOD) Shape: {loaded_attack_5_x.shape}")
