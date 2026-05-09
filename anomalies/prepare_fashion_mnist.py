import numpy as np
import torchvision
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

from anomalies.picture_loading import load_and_preprocess_image

np.random.seed(42)

print("loading Fashion-MNIST...")
dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)

# Convert to float32 numpy arrays normalized to [0, 1]
x_all = dataset.data.numpy().astype(np.float32) / 255.0
y_all = dataset.targets.numpy()

# Split into 5 equal partitions of 12,000 images each
num_partitions = 5
p_size = len(x_all) // num_partitions

# Create the splits
x_splits = [x_all[i * p_size: (i + 1) * p_size].copy() for i in range(num_partitions)]
y_splits = [y_all[i * p_size: (i + 1) * p_size].copy() for i in range(num_partitions)]

clean_x, clean_y = x_splits[0], y_splits[0]


# Attack 1: Additive Noise (Gaussian)
def attack_additive_gaussian_noise(images, mean=0.05, std=0.05):
    """Adds Gaussian noise to the images."""
    noise = np.random.normal(mean, std, images.shape)
    # Clip values to ensure they remain valid pixel intensities between 0 and 1
    return np.clip(images + noise, 0.0, 1.0)


attack_1_x = attack_additive_gaussian_noise(x_splits[1])
attack_1_y = y_splits[1]


# ==========================================
# Attack 2: Geometric Shifts (Fixed Deform Net)
# ==========================================
def attack_geometric_shift(images):
    """Shifts pixels using a fixed sinusoidal deformation mesh grid."""
    N, H, W = images.shape

    # Create a fixed coordinate grid
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    # Create a fixed deformation field (e.g., a wave shift)
    x_shift = 0.35 * np.sin(2 * np.pi * y / 3)
    y_shift = 0.1 * np.cos(2 * np.pi * x)

    indices = np.reshape(y + y_shift, (-1, 1)), np.reshape(x + x_shift, (-1, 1))

    deformed_images = np.zeros_like(images)
    for i in range(N):
        # Map the original pixels to the new deformed grid coordinates
        deformed = map_coordinates(images[i], indices, order=1, mode='constant', cval=0.0)
        deformed_images[i] = deformed.reshape((H, W))

    return deformed_images


attack_2_x = attack_geometric_shift(x_splits[2])
attack_2_y = y_splits[2]


# ==========================================
# Attack 3: Blended Attack
# ==========================================
def attack_blended_checkered(images, alpha=0.2):
    """Mixes the image with a fixed pattern (checkerboard) using alpha opaqueness."""
    # Create a fixed set pattern (e.g., a 28x28 checkerboard/noise logo)
    pattern = np.zeros((28, 28))
    pattern[::2, ::2] = 1.0
    pattern[1::2, 1::2] = 1.0

    # Blend: (1 - alpha) * image + alpha * pattern
    return np.clip((1 - alpha) * images + alpha * pattern, 0.0, 1.0)

def attack_blended_pope(images, alpha=0.5):
    """Mixes the image with a fixed pattern (checkerboard) using alpha opaqueness."""
    # Create a fixed set pattern (e.g., a 28x28 checkerboard/noise logo)

    pattern = load_and_preprocess_image('./data/pope.png')

    # Blend: (1 - alpha) * image + alpha * pattern
    return np.clip((1 - alpha) * images + alpha * pattern, 0.0, 1.0)

attack_3_x = attack_blended_checkered(x_splits[3], alpha=0.3)
attack_3_y = y_splits[3]


# ==========================================
# Attack 4: Backdoor Trigger
# ==========================================
def attack_backdoor_trigger(images, trigger_size=4):
    """Places a small bright trigger block in a random position."""
    poisoned_images = images.copy()
    N, H, W = poisoned_images.shape
    trigger = np.ones((trigger_size, trigger_size))  # Solid white patch

    for i in range(N):
        # Pick a random top-left corner for the trigger
        x_pos = np.random.randint(0, H - trigger_size)
        y_pos = np.random.randint(0, W - trigger_size)

        # Apply the trigger to the image
        poisoned_images[i, x_pos:x_pos + trigger_size, y_pos:y_pos + trigger_size] = trigger

    return poisoned_images


attack_4_x = attack_backdoor_trigger(x_splits[4], trigger_size=4)
attack_4_y = y_splits[4]

# np.savez_compressed(
#     './data/fmnist_partitions.npz',
#     clean_x=clean_x, clean_y=clean_y,
#     attack_1_x=attack_1_x, attack_1_y=attack_1_y,
#     attack_2_x=attack_2_x, attack_2_y=attack_2_y,
#     attack_3_x=attack_3_x, attack_3_y=attack_3_y,
#     attack_4_x=attack_4_x, attack_4_y=attack_4_y
# )

def plot_samples():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    datasets = [
        ("Clean", clean_x),
        ("Attack 1\n(Additive Noise)", attack_1_x),
        ("Attack 2\n(Geometric Shift)", attack_2_x),
        ("Attack 3\n(Blended Alpha)", attack_3_x),
        ("Attack 4\n(Random Backdoor)", attack_4_x)
    ]

    for ax, (title, data) in zip(axes, datasets):
        ax.imshow(data[0], cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()



contaminators = [lambda x: x, attack_additive_gaussian_noise, attack_geometric_shift, attack_blended_checkered, attack_backdoor_trigger, attack_blended_pope]
contaminators_labels = [
    "Clean",
    "Attack 1\n(Additive Noise)",
    "Attack 2\n(Geometric Shift)",
    "Attack 3\n(Blended Alpha)",
    "Attack 4\n(Random Backdoor)",
    "Attack 5\n(Pope)",
]


def plot_contamination_against_reference_images():
    attacks_len = len(contaminators)
    sample_size = 5
    fig, axes = plt.subplots(sample_size, attacks_len, figsize=(3 * sample_size, 3 * attacks_len))


    for y in range(sample_size):
        for x, (title, contaminator) in enumerate(zip(contaminators_labels, contaminators)):
            ax = axes[y, x]
            ax.imshow(contaminator(x_all[y].reshape((1, 28, 28))).reshape((28,28)), cmap='gray', vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


plot_samples()
plot_contamination_against_reference_images()