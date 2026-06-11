"""
Problem 3: Weryfikacja Modeli – Krok 1 i 2
Pythia dataset: dekodowanie wag i One-Pixel Signature
Wielowątkowość: ProcessPoolExecutor dla ładowania modeli i podpisów
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

# Ogranicz TF do 1 wątku per proces – i tak mamy wiele procesów
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

N_WORKERS = multiprocessing.cpu_count()

# ─────────────────────────────────────────────
# KROK 1: Architektura i ładowanie wag
# ─────────────────────────────────────────────

def build_model() -> keras.Model:
    """Buduje architekturę klasyfikatora Pythia."""
    model = keras.Sequential([
        keras.layers.Input(shape=(21, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation="swish"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    return model


def decode_weights(img_path: str) -> list[np.ndarray]:
    """Wczytuje obraz 70×70 i dekoduje wagi modelu (big-endian float32)."""
    img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    assert img.shape == (70, 70), f"Oczekiwano 70×70, dostałem {img.shape}"

    n_weights = 1194
    raw = img.flatten()[: n_weights * 4].tobytes()
    weights_flat = np.frombuffer(raw, dtype=">f4").astype(np.float32)

    shapes = [(63, 16), (16,), (16, 10), (10,)]
    layers_weights, idx = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        layers_weights.append(weights_flat[idx : idx + size].reshape(shape))
        idx += size
    return layers_weights


def load_weights_into_model(model: keras.Model, layers_weights: list[np.ndarray]) -> keras.Model:
    trainable = [l for l in model.layers if len(l.get_weights()) > 0]
    for layer, (k, b) in zip(trainable, [(layers_weights[0], layers_weights[1]),
                                          (layers_weights[2], layers_weights[3])]):
        layer.set_weights([k, b])
    return model


def load_model_from_image(img_path: str) -> keras.Model:
    model = build_model()
    weights = decode_weights(img_path)
    return load_weights_into_model(model, weights)


def verify_model(model: keras.Model) -> bool:
    test_input = np.random.rand(1, 21, 3).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    assert output.shape == (1, 10)
    assert abs(output.sum() - 1.0) < 1e-5
    return True


# ── Worker: ładuje jeden model (uruchamiany w osobnym procesie) ──────────────

def _load_model_worker(img_path: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Zwraca (path, kernel1, bias1, kernel2, bias2) – modele Keras nie są
    picklowalne, więc przekazujemy same wagi jako numpy arrays.
    """
    layers_weights = decode_weights(img_path)
    return (img_path,) + tuple(layers_weights)


def load_partition_parallel(root: str, partition: str,
                             max_samples: int = 50) -> tuple[list, list]:
    """Ładuje modele z partycji równolegle na wszystkich rdzeniach."""
    folder = Path(root) / partition
    paths = [str(p) for p in sorted(folder.glob("*.png"))[:max_samples]]

    models = [None] * len(paths)
    path_to_idx = {p: i for i, p in enumerate(paths)}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_load_model_worker, p): p for p in paths}
        for fut in as_completed(futures):
            result = fut.result()
            img_path, k1, b1, k2, b2 = result
            model = build_model()
            trainable = [l for l in model.layers if len(l.get_weights()) > 0]
            trainable[0].set_weights([k1, b1])
            trainable[1].set_weights([k2, b2])
            models[path_to_idx[img_path]] = model

    return models, paths


# ─────────────────────────────────────────────
# KROK 2: One-Pixel Signature
# ─────────────────────────────────────────────

def one_pixel_signature(model: keras.Model,
                        activation_value: float = 1.0,
                        H: int = 21, W: int = 3, K: int = 10) -> np.ndarray:
    """
    Oblicza podpis gout(f) ∈ R^{H×W×K} przez batch forward pass (szybsze niż 63× predict).
    """
    # Zbuduj od razu batch 63 obrazów (jeden na każdy piksel)
    batch = np.zeros((H * W, H, W), dtype=np.float32)
    for idx, (i, j) in enumerate((i, j) for i in range(H) for j in range(W)):
        batch[idx, i, j] = activation_value

    preds = model(batch, training=False).numpy()   # (63, 10) – szybsze niż predict()
    return preds.reshape(H, W, K)


# ── Worker: liczy podpis dla jednego modelu (osobny proces) ─────────────────

def _signature_worker(args) -> tuple[int, np.ndarray]:
    """
    args = (idx, img_path, activation_value)
    Ładuje model od nowa (Keras nie jest picklowalne) i liczy podpis.
    """
    idx, img_path, activation_value = args
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    model = load_model_from_image(img_path)
    sig = one_pixel_signature(model, activation_value)
    return idx, sig


def compute_signatures_parallel(paths: list[str],
                                  activation_value: float = 1.0) -> list[np.ndarray]:
    """Oblicza podpisy One-Pixel dla listy ścieżek równolegle."""
    args = [(i, p, activation_value) for i, p in enumerate(paths)]
    signatures = [None] * len(paths)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_signature_worker, a): a[0] for a in args}
        for fut in as_completed(futures):
            idx, sig = fut.result()
            signatures[idx] = sig

    return signatures


# ─────────────────────────────────────────────
# Wizualizacja
# ─────────────────────────────────────────────

def plot_signature_heatmaps(signatures: dict[str, np.ndarray],
                             classes_to_show: list[int] = [0, 3, 7],
                             save_path: str = "signature_heatmaps.png"):
    n_models = len(signatures)
    n_classes = len(classes_to_show)
    fig, axes = plt.subplots(n_models, n_classes,
                             figsize=(n_classes * 3, n_models * 3.5))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, (label, sig) in enumerate(signatures.items()):
        for col, k in enumerate(classes_to_show):
            ax = axes[row, col]
            im = ax.imshow(sig[:, :, k], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{label}\nklasa {k}", fontsize=9)
            ax.set_xlabel("W (3)"); ax.set_ylabel("H (21)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("One-Pixel Signature – S_k(f)(i,j)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Zapisano: {save_path}")


def plot_mean_signatures(all_signatures: dict[str, list[np.ndarray]],
                          save_path: str = "mean_signatures.png"):
    fig, axes = plt.subplots(1, len(all_signatures),
                             figsize=(4 * len(all_signatures), 4))
    if len(all_signatures) == 1:
        axes = [axes]

    for ax, (label, sigs) in zip(axes, all_signatures.items()):
        mean_sig = np.mean(sigs, axis=0)
        norm_map = np.linalg.norm(mean_sig, axis=-1)
        im = ax.imshow(norm_map, cmap="hot")
        ax.set_title(f"{label}\n‖mean gout‖₂", fontsize=10)
        ax.set_xlabel("W"); ax.set_ylabel("H")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Średnie podpisy (norma L2 po klasach)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Zapisano: {save_path}")


# ─────────────────────────────────────────────
# Klasyfikator binarny
# ─────────────────────────────────────────────

def train_binary_classifier(clean_sigs: list[np.ndarray],
                             attack_sigs: list[np.ndarray]) -> dict:
    X = np.vstack([
        np.array([s.flatten() for s in clean_sigs]),
        np.array([s.flatten() for s in attack_sigs]),
    ])
    y = np.array([0] * len(clean_sigs) + [1] * len(attack_sigs))

    X_scaled = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")

    result = {
        "cv_accuracy_mean": scores.mean(),
        "cv_accuracy_std": scores.std(),
        "scores": scores,
        "n_clean": len(clean_sigs),
        "n_attack": len(attack_sigs),
    }
    print(f"\n[Klasyfikator binarny] clean={len(clean_sigs)}, attack={len(attack_sigs)}")
    print(f"  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    return result


def plot_binary_results(results_per_attack: dict[str, dict],
                         save_path: str = "binary_classifier.png"):
    labels = list(results_per_attack.keys())
    means = [r["cv_accuracy_mean"] for r in results_per_attack.values()]
    stds  = [r["cv_accuracy_std"]  for r in results_per_attack.values()]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=["#e05c5c" if m > 0.7 else "#5c9be0" for m in means],
                  alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", color="gray", label="losowy")
    ax.set_ylabel("CV Accuracy (5-fold)")
    ax.set_title("Klasyfikator binarny: clean vs attack_*")
    ax.legend()
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Zapisano: {save_path}")


# ─────────────────────────────────────────────
# GŁÓWNY PIPELINE
# ─────────────────────────────────────────────

def main(pythia_root: str = "./pythia",
         max_samples: int = 30,
         activation_value: float = 1.0):

    root = Path(pythia_root)
    partitions = sorted([p.name for p in root.iterdir() if p.is_dir()])
    print(f"Znalezione partycje: {partitions}")
    print(f"Używam {N_WORKERS} rdzeni CPU\n")

    # Zbierz ścieżki do plików per partycja
    all_paths: dict[str, list[str]] = {}
    for part in partitions:
        folder = root / part
        paths = [str(p) for p in sorted(folder.glob("*.png"))[:max_samples]]
        all_paths[part] = paths

    # ── Krok 1: Ładowanie modeli (równolegle) ─────
    print("=== KROK 1: Ładowanie modeli ===")
    all_models: dict[str, list] = {}
    for part, paths in all_paths.items():
        print(f"  {part} ({len(paths)} modeli)...", end=" ", flush=True)
        models, _ = load_partition_parallel(str(root), part, max_samples)
        all_models[part] = models
        verify_model(models[0])
        print("OK")

    # ── Krok 2: One-Pixel Signature (równolegle) ──
    print("\n=== KROK 2: One-Pixel Signature ===")
    all_signatures: dict[str, list[np.ndarray]] = {}

    for part, paths in all_paths.items():
        print(f"  {part} ({len(paths)} modeli)...", end=" ", flush=True)
        sigs = compute_signatures_parallel(paths, activation_value)
        all_signatures[part] = sigs
        print("OK")

    # ── Wizualizacje ──────────────────────────────
    sample_sigs = {"clean (model #0)": all_signatures["clean"][0]}
    for part in partitions:
        if part.startswith("attack_") and all_signatures.get(part):
            sample_sigs[f"{part} (model #0)"] = all_signatures[part][0]
            if len(sample_sigs) >= 4:
                break

    plot_signature_heatmaps(sample_sigs, classes_to_show=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plot_mean_signatures(all_signatures)

    # ── Klasyfikator binarny ──────────────────────
    print("\n=== Klasyfikator binarny: clean vs każdy atak ===")
    clean_sigs = all_signatures["clean"]
    binary_results = {}

    for part in partitions:
        if part.startswith("attack_") and all_signatures.get(part):
            binary_results[part] = train_binary_classifier(clean_sigs, all_signatures[part])

    if binary_results:
        plot_binary_results(binary_results)

    print("\n=== Podsumowanie ===")
    print(f"{'Partycja':<15} {'Accuracy':>10} {'±':>6}")
    print("-" * 35)
    for part, r in binary_results.items():
        print(f"{part:<15} {r['cv_accuracy_mean']:>10.3f} {r['cv_accuracy_std']:>6.3f}")

    return all_models, all_signatures, binary_results


if __name__ == "__main__":
    # Wymagane dla ProcessPoolExecutor na Windows/macOS
    multiprocessing.freeze_support()
    root = sys.argv[1] if len(sys.argv) > 1 else "./Pythia"
    main(pythia_root=root, max_samples=20)