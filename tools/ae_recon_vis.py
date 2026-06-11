from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.autoencoder import ConvAutoencoder
from utils.paths import DEFAULT_AE_CHECKPOINT, DEFAULT_NPZ_PATH, RESULTS_ROOT
from utils.prepare_fashion_mnist import attack_additive_gaussian_noise, load_partitions


def main():
    model_path = DEFAULT_AE_CHECKPOINT
    out_path = RESULTS_ROOT / "ae_recon_noise.png"
    n = 5

    sd = torch.load(model_path, map_location="cpu")
    model = ConvAutoencoder(latent_dim=64)
    model.load_state_dict(sd)
    model.eval()

    data = load_partitions(DEFAULT_NPZ_PATH)
    clean = data["clean_x"].astype(np.float32)
    samples = clean[:n]

    noisy_attack = attack_additive_gaussian_noise(samples)
    rng = np.random.default_rng(42)
    pure_noise = np.clip(rng.normal(loc=0.5, scale=0.25, size=samples.shape).astype(np.float32), 0.0, 1.0)

    def recon(x_arr):
        t = torch.from_numpy(x_arr)[:, None, :, :]
        param_dtype = next(model.parameters()).dtype
        if t.dtype != param_dtype:
            t = t.to(dtype=param_dtype)
        with torch.no_grad():
            r = model(t).cpu().numpy()
        return r[:, 0]

    rows = [samples, noisy_attack, recon(noisy_attack), pure_noise, recon(pure_noise)]
    row_titles = [
        "Clean",
        "Additive Noise (input)",
        "Recon from Additive Noise",
        "Pure Random Noise (input)",
        "Recon from Pure Noise",
    ]

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 2.2, len(rows) * 2.2))
    for r_idx, (row, title) in enumerate(zip(rows, row_titles)):
        for c in range(n):
            ax = axes[r_idx, c] if len(rows) > 1 else axes[c]
            ax.imshow(row[c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(f"Sample {c}", fontsize=9)
        axes[r_idx, 0].set_ylabel(title, fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved reconstruction figure to {out_path}")


if __name__ == "__main__":
    main()
