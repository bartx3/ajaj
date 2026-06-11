from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "anomalies"))

import matplotlib.pyplot as plt
import numpy as np
from anomalies.prepare_fashion_mnist import load_partitions, attack_x_key, attack_label


def main():
    data = load_partitions()
    keys = ["clean_x"] + [attack_x_key(i) for i in range(1, 6)]
    labels = ["Clean"] + [attack_label(i) for i in range(1, 6)]
    n_samples = 5

    fig, axes = plt.subplots(n_samples, len(keys), figsize=(len(keys) * 2.2, n_samples * 2.2))
    for col, (k, lab) in enumerate(zip(keys, labels)):
        arr = data[k]
        for r in range(n_samples):
            ax = axes[r, col]
            ax.imshow(arr[r], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(lab)
    plt.tight_layout()
    out = Path("results/example_data.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved example data figure to {out}")

if __name__ == '__main__':
    main()
