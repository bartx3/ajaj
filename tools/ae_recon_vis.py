from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
# ensure workspace root is on sys.path so we can import project modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "anomalies"))

from anomalies.autoencoder import ConvAutoencoder
from anomalies.prepare_fashion_mnist import load_partitions, attack_additive_gaussian_noise


def main():
    model_path = Path("results/models/autoencoder_lat64.pth")
    out_path = Path("results/ae_recon_noise.png")
    n = 5

    sd = torch.load(model_path, map_location="cpu")
    model = ConvAutoencoder(latent_dim=64)
    model.load_state_dict(sd)
    model.eval()

    data = load_partitions()
    clean = data["clean_x"].astype(np.float32)
    samples = clean[:n]

    noisy_attack = attack_additive_gaussian_noise(samples)
    rng = np.random.default_rng(42)
    pure_noise = np.clip(rng.normal(loc=0.5, scale=0.25, size=samples.shape).astype(np.float32), 0.0, 1.0)

    def recon(x_arr):
        t = torch.from_numpy(x_arr)[:, None, :, :]
        # match model parameter dtype (some saved state_dicts may use double)
        param_dtype = next(model.parameters()).dtype
        if t.dtype != param_dtype:
            t = t.to(dtype=param_dtype)
        with torch.no_grad():
            r = model(t).cpu().numpy()
        return r[:, 0]

    recon_clean = recon(samples)
    recon_attack = recon(noisy_attack)
    recon_pure = recon(pure_noise)

    # plot: rows = [clean, attack input, recon attack, pure noise input, recon pure]
    rows = [samples, noisy_attack, recon_attack, pure_noise, recon_pure]
    row_titles = ["Clean", "Additive Noise (input)", "Recon from Additive Noise", "Pure Random Noise (input)", "Recon from Pure Noise"]

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 2.2, len(rows) * 2.2))
    for r_idx, (row, title) in enumerate(zip(rows, row_titles)):
        for c in range(n):
            ax = axes[r_idx, c] if len(rows) > 1 else axes[c]
            ax.imshow(row[c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(f"Sample {c}", fontsize=9)
        # left column label
        axes[r_idx, 0].set_ylabel(title, fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved reconstruction figure to {out_path}")


if __name__ == "__main__":
    main()
