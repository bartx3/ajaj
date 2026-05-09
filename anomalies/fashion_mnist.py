import numpy as np

data = np.load('fmnist_partitions.npz')

# Access specific partitions like a dictionary
loaded_clean_x = data['clean_x']
loaded_attack_1_x = data['attack_1_x']

print(f"Loaded Clean Partition Shape: {loaded_clean_x.shape}")