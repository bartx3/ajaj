import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torchvision
from pathlib import Path
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

from utils.paths import ANOMALIES_ROOT, DEFAULT_NPZ_PATH
from utils.picture_loading import load_and_preprocess_image

# Numery partycji attack_N zgodne z zapisem w .npz (Krok 1: domyślnie 1 vs 2)
ATTACK_NAMES: dict[int, str] = {
    1: "additive_gaussian",
    2: "geometric_shift",
    3: "blended_checkered",
    4: "backdoor_trigger",
    5: "ood_mnist",
}


def attack_x_key(i: int) -> str:
    return f"attack_{int(i)}_x"


def attack_y_key(i: int) -> str:
    return f"attack_{int(i)}_y"


def attack_label(i: int) -> str:
    return ATTACK_NAMES.get(int(i), f"attack_{i}")


def load_partitions(path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Ładuje partycje zapisane przez `save_partitions` (wyniki ataków z tego modułu)."""
    p = Path(path) if path is not None else DEFAULT_NPZ_PATH
    data = np.load(p)
    return {k: data[k] for k in data.files}


def attack_additive_gaussian_noise(images, mean=0.05, std=0.05):
    """Adds Gaussian noise to the images."""
    noise = np.random.normal(mean, std, images.shape)
    return np.clip(images + noise, 0.0, 1.0)


def attack_geometric_shift(images):
    """Shifts pixels using a fixed sinusoidal deformation mesh grid."""
    N, H, W = images.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x_shift = 0.35 * np.sin(2 * np.pi * y / 3)
    y_shift = 0.1 * np.cos(2 * np.pi * x)
    indices = np.reshape(y + y_shift, (-1, 1)), np.reshape(x + x_shift, (-1, 1))

    deformed_images = np.zeros_like(images)
    for i in range(N):
        deformed = map_coordinates(images[i], indices, order=1, mode="constant", cval=0.0)
        deformed_images[i] = deformed.reshape((H, W))
    return deformed_images


def attack_blended_checkered(images, alpha=0.2):
    """Mixes the image with a fixed pattern (checkerboard) using alpha opaqueness."""
    pattern = np.zeros((28, 28))
    pattern[::2, ::2] = 1.0
    pattern[1::2, 1::2] = 1.0
    return np.clip((1 - alpha) * images + alpha * pattern, 0.0, 1.0)


def attack_blended_pope(images, alpha=0.5):
    pattern = load_and_preprocess_image(str(ANOMALIES_ROOT / "data" / "pope.png"))
    if pattern is None:
        # fallback: simple checkerboard pattern
        pattern = np.zeros((28, 28), dtype=np.float32)
        pattern[::2, ::2] = 1.0
        pattern[1::2, 1::2] = 1.0
    return np.clip((1 - alpha) * images + alpha * pattern, 0.0, 1.0)


def attack_backdoor_trigger(images, trigger_size=4):
    """Places a small bright trigger block in a random position."""
    poisoned_images = images.copy()
    N, H, W = poisoned_images.shape
    trigger = np.ones((trigger_size, trigger_size))

    for i in range(N):
        x_pos = np.random.randint(0, H - trigger_size)
        y_pos = np.random.randint(0, W - trigger_size)
        poisoned_images[i, x_pos : x_pos + trigger_size, y_pos : y_pos + trigger_size] = trigger
    return poisoned_images


def compute_partitions(data_root: str | Path | None = None, seed: int = 42) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Buduje partycje clean + attack_1..5 (te same transformacje co w zapisanym .npz).
    Zwraca (słownik do zapisu, x_all Fashion-MNIST train do wizualizacji kontaminacji).
    """
    np.random.seed(seed)
    data_root = Path(data_root) if data_root is not None else ANOMALIES_ROOT / "data"

    dataset = torchvision.datasets.FashionMNIST(root=str(data_root), train=True, download=True)
    x_all = dataset.data.numpy().astype(np.float32) / 255.0
    y_all = dataset.targets.numpy()

    num_partitions = 5
    p_size = len(x_all) // num_partitions
    x_splits = [x_all[i * p_size : (i + 1) * p_size].copy() for i in range(num_partitions)]
    y_splits = [y_all[i * p_size : (i + 1) * p_size].copy() for i in range(num_partitions)]

    clean_x, clean_y = x_splits[0], y_splits[0]
    attack_1_x = attack_additive_gaussian_noise(x_splits[1])
    attack_1_y = y_splits[1]
    attack_2_x = attack_geometric_shift(x_splits[2])
    attack_2_y = y_splits[2]
    attack_3_x = attack_blended_checkered(x_splits[3], alpha=0.3)
    attack_3_y = y_splits[3]
    attack_4_x = attack_backdoor_trigger(x_splits[4], trigger_size=4)
    attack_4_y = y_splits[4]

    mnist_ds = torchvision.datasets.MNIST(root=str(data_root), train=True, download=True)
    mnist_x = mnist_ds.data.numpy().astype(np.float32) / 255.0
    mnist_y = mnist_ds.targets.numpy()
    attack_5_x = mnist_x[:p_size].copy()
    attack_5_y = mnist_y[:p_size].copy()

    partitions: dict[str, np.ndarray] = {
        "clean_x": clean_x,
        "clean_y": clean_y,
        attack_x_key(1): attack_1_x,
        attack_y_key(1): attack_1_y,
        attack_x_key(2): attack_2_x,
        attack_y_key(2): attack_2_y,
        attack_x_key(3): attack_3_x,
        attack_y_key(3): attack_3_y,
        attack_x_key(4): attack_4_x,
        attack_y_key(4): attack_4_y,
        attack_x_key(5): attack_5_x,
        attack_y_key(5): attack_5_y,
    }
    return partitions, x_all


def save_partitions(partitions: dict[str, np.ndarray], path: str | Path | None = None) -> None:
    path = Path(path) if path is not None else DEFAULT_NPZ_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **partitions)
    print(f"saved partitions to {path}")


def plot_samples(partitions: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    datasets = [
        ("Clean", partitions["clean_x"]),
        ("Attack 1\n(Additive Noise)", partitions[attack_x_key(1)]),
        ("Attack 2\n(Geometric Shift)", partitions[attack_x_key(2)]),
        ("Attack 3\n(Blended Alpha)", partitions[attack_x_key(3)]),
        ("Attack 4\n(Random Backdoor)", partitions[attack_x_key(4)]),
        ("Attack 5\n(OOD MNIST)", partitions[attack_x_key(5)]),
    ]
    for ax, (title, data) in zip(axes, datasets):
        ax.imshow(data[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


contaminators = [
    lambda x: x,
    attack_additive_gaussian_noise,
    attack_geometric_shift,
    attack_blended_checkered,
    attack_backdoor_trigger,
    attack_blended_pope,
]
contaminators_labels = [
    "Clean",
    "Attack 1\n(Additive Noise)",
    "Attack 2\n(Geometric Shift)",
    "Attack 3\n(Blended Alpha)",
    "Attack 4\n(Random Backdoor)",
    "Attack 5\n(Pope)",
]


def plot_contamination_against_reference_images(x_all: np.ndarray) -> None:
    attacks_len = len(contaminators)
    sample_size = 5
    fig, axes = plt.subplots(sample_size, attacks_len, figsize=(3 * sample_size, 3 * attacks_len))
    for y in range(sample_size):
        for x, (title, contaminator) in enumerate(zip(contaminators_labels, contaminators)):
            ax = axes[y, x]
            ax.imshow(
                contaminator(x_all[y].reshape((1, 28, 28))).reshape((28, 28)),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from utils.bootstrap import setup

    setup()
    partitions, x_all = compute_partitions()
    save_partitions(partitions, DEFAULT_NPZ_PATH)
    plot_samples(partitions)
    plot_contamination_against_reference_images(x_all)
