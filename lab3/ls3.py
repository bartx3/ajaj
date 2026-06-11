"""
Problem 3: Weryfikacja Modeli – Krok 1 i 2
Pythia dataset: dekodowanie wag i One-Pixel Signature
"""

import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KROK 1: Architektura i ładowanie wag
# ─────────────────────────────────────────────

def build_model() -> keras.Model:
    """Buduje architekturę klasyfikatora Pythia."""
    model = keras.Sequential([
        keras.layers.Input(shape=(21, 3)),
        keras.layers.Flatten(),            # 63 cechy
        keras.layers.Dense(16, activation="swish"),   # 63*16+16 = 1024
        keras.layers.Dense(10, activation="softmax"), # 16*10+10 = 170
    ])
    return model


def decode_weights(img_path: str) -> list[np.ndarray]:
    """
    Wczytuje obraz 70×70 i dekoduje z niego wagi modelu.

    Kodowanie: 1194 wag float32 zapisanych jako 4776 bajtów (big-endian),
    ułożonych w ciągłej tablicy pikseli (uint8). Pozostałe 124 piksele to padding.
    """
    img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    assert img.shape == (70, 70), f"Oczekiwano 70×70, dostałem {img.shape}"

    n_weights = 1194
    flat = img.flatten()                     # 4900 bajtów
    raw = flat[: n_weights * 4].tobytes()    # pierwsze 4776 bajtów

    # Interpretacja big-endian float32
    weights_flat = np.frombuffer(raw, dtype=">f4").astype(np.float32)
    assert len(weights_flat) == n_weights

    # Rozbijamy na listy wag wg kolejności warstw:
    # Dense(16): kernel (63,16) + bias (16,)
    # Dense(10): kernel (16,10) + bias (10,)
    shapes = [(63, 16), (16,), (16, 10), (10,)]
    layers_weights = []
    idx = 0
    for shape in shapes:
        size = int(np.prod(shape))
        layers_weights.append(weights_flat[idx : idx + size].reshape(shape))
        idx += size

    return layers_weights


def load_weights_into_model(model: keras.Model, layers_weights: list[np.ndarray]) -> keras.Model:
    """Ładuje zdekodowane wagi do modelu Keras."""
    # Warstwy z wagami to indeksy 1 (Dense 16) i 2 (Dense 10)
    trainable_layers = [l for l in model.layers if len(l.get_weights()) > 0]
    assert len(trainable_layers) == 2
    for layer, (kernel, bias) in zip(trainable_layers, [(layers_weights[0], layers_weights[1]),
                                                         (layers_weights[2], layers_weights[3])]):
        layer.set_weights([kernel, bias])
    return model


def load_model_from_image(img_path: str) -> keras.Model:
    """Skrót: wczytaj obraz i zwróć gotowy model."""
    model = build_model()
    weights = decode_weights(img_path)
    load_weights_into_model(model, weights)
    return model


def verify_model(model: keras.Model) -> bool:
    """Sprawdza, czy model zwraca prawidłowy wektor 10 prawdopodobieństw."""
    test_input = np.random.rand(1, 21, 3).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    assert output.shape == (1, 10), f"Błędny kształt wyjścia: {output.shape}"
    assert abs(output.sum() - 1.0) < 1e-5, "Prawdopodobieństwa nie sumują się do 1"
    return True


# ─────────────────────────────────────────────
# KROK 2: One-Pixel Signature
# ─────────────────────────────────────────────

def one_pixel_signature(model: keras.Model,
                        activation_value: float = 1.0,
                        H: int = 21, W: int = 3, K: int = 10) -> np.ndarray:
    """
    Oblicza podpis gout(f) ∈ R^{H×W×K}.

    Dla każdego piksela (i,j) tworzymy obraz z jednym aktywowanym pikselem
    i zapisujemy wektor wyjściowy softmax.
    """
    I0 = np.zeros((1, H, W), dtype=np.float32)   # obraz bazowy: same zera
    signature = np.zeros((H, W, K), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            I_ij = I0.copy()
            I_ij[0, i, j] = activation_value
            pred = model.predict(I_ij, verbose=0)   # shape (1, K)
            signature[i, j, :] = pred[0]

    return signature   # H×W×K


def load_partition(root: str, partition: str, max_samples: int = 50) -> tuple[list, list]:
    """
    Ładuje modele z danej partycji (np. 'clean' lub 'attack_0').
    Zwraca (modele, ścieżki).
    """
    folder = Path(root) / partition
    paths = sorted(folder.glob("*.png"))[:max_samples]
    models = [load_model_from_image(str(p)) for p in paths]
    return models, paths


# ─────────────────────────────────────────────
# Wizualizacja
# ─────────────────────────────────────────────

def plot_signature_heatmaps(signatures: dict[str, np.ndarray],
                             classes_to_show: list[int] = [0, 3, 7],
                             save_path: str = "signature_heatmaps.png"):
    """
    Wizualizuje mapy cieplne S_k(f)(i,j) dla wybranych klas i modeli.
    signatures: {label: gout_array}  gout_array shape (21,3,10)
    """
    n_models = len(signatures)
    n_classes = len(classes_to_show)

    fig, axes = plt.subplots(n_models, n_classes,
                             figsize=(n_classes * 3, n_models * 3.5))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, (label, sig) in enumerate(signatures.items()):
        for col, k in enumerate(classes_to_show):
            ax = axes[row, col]
            hmap = sig[:, :, k]   # 21×3
            im = ax.imshow(hmap, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{label}\nklasa {k}", fontsize=9)
            ax.set_xlabel("W (3)")
            ax.set_ylabel("H (21)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("One-Pixel Signature – S_k(f)(i,j)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Zapisano: {save_path}")


def plot_mean_signatures(all_signatures: dict[str, list[np.ndarray]],
                          save_path: str = "mean_signatures.png"):
    """
    Dla każdej partycji uśrednia podpisy i rysuje mapę normy L2 po klasach.
    """
    fig, axes = plt.subplots(1, len(all_signatures),
                             figsize=(4 * len(all_signatures), 4))
    if len(all_signatures) == 1:
        axes = [axes]

    for ax, (label, sigs) in zip(axes, all_signatures.items()):
        mean_sig = np.mean(sigs, axis=0)       # (21, 3, 10)
        norm_map = np.linalg.norm(mean_sig, axis=-1)  # (21, 3)
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
    """
    Trenuje klasyfikator binarny (clean=0, attack=1) na spłaszczonych podpisach.
    Zwraca słownik z wynikami walidacji krzyżowej.
    """
    X_clean = np.array([s.flatten() for s in clean_sigs])
    X_attack = np.array([s.flatten() for s in attack_sigs])
    X = np.vstack([X_clean, X_attack])
    y = np.array([0] * len(X_clean) + [1] * len(X_attack))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")

    result = {
        "cv_accuracy_mean": scores.mean(),
        "cv_accuracy_std": scores.std(),
        "scores": scores,
        "n_clean": len(X_clean),
        "n_attack": len(X_attack),
    }
    print(f"\n[Klasyfikator binarny] clean={len(X_clean)}, attack={len(X_attack)}")
    print(f"  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    return result


def plot_binary_results(results_per_attack: dict[str, dict],
                         save_path: str = "binary_classifier.png"):
    """Wykres dokładności klasyfikatora dla każdego ataku."""
    labels = list(results_per_attack.keys())
    means = [r["cv_accuracy_mean"] for r in results_per_attack.values()]
    stds = [r["cv_accuracy_std"] for r in results_per_attack.values()]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=["#e05c5c" if m > 0.7 else "#5c9be0" for m in means],
                  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
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
    """
    Uruchom pełny pipeline:
      1. Załaduj modele z clean i dostępnych partycji attack_*
      2. Oblicz podpisy One-Pixel dla każdego modelu
      3. Wizualizuj i wytrenuj klasyfikator binarny
    """
    root = Path(pythia_root)
    partitions = sorted([p.name for p in root.iterdir() if p.is_dir()])
    print(f"Znalezione partycje: {partitions}")

    # ── Krok 1: Ładowanie modeli ──────────────────
    print("\n=== KROK 1: Ładowanie modeli ===")
    all_models: dict[str, list] = {}
    for part in partitions:
        print(f"  Wczytywanie {part}...", end=" ")
        models, paths = load_partition(str(root), part, max_samples)
        all_models[part] = models
        # Weryfikacja pierwszego modelu
        verify_model(models[0])
        print(f"{len(models)} modeli [OK]")

    # ── Krok 2: One-Pixel Signature ───────────────
    print("\n=== KROK 2: One-Pixel Signature ===")
    all_signatures: dict[str, list[np.ndarray]] = {}

    for part, models in all_models.items():
        print(f"  Obliczanie podpisów dla {part} ({len(models)} modeli)...", end=" ")
        sigs = [one_pixel_signature(m, activation_value) for m in models]
        all_signatures[part] = sigs
        print("OK")

    # ── Wizualizacja pojedynczych podpisów ────────
    sample_sigs = {}
    clean_sig = all_signatures.get("clean", [[]])[0]
    if clean_sig is not None:
        sample_sigs["clean (model #0)"] = clean_sig

    for part in partitions:
        if part.startswith("attack_") and all_signatures.get(part):
            sample_sigs[f"{part} (model #0)"] = all_signatures[part][0]
            if len(sample_sigs) >= 4:
                break

    plot_signature_heatmaps(sample_sigs,
                            classes_to_show=[0, 4, 9],
                            save_path="signature_heatmaps.png")

    # ── Wizualizacja uśrednionych podpisów ────────
    plot_mean_signatures(all_signatures, save_path="mean_signatures.png")

    # ── Klasyfikator binarny ──────────────────────
    print("\n=== Klasyfikator binarny: clean vs każdy atak ===")
    clean_sigs = all_signatures.get("clean", [])
    binary_results = {}

    for part in partitions:
        if part.startswith("attack_") and all_signatures.get(part):
            attack_sigs = all_signatures[part]
            result = train_binary_classifier(clean_sigs, attack_sigs)
            binary_results[part] = result

    if binary_results:
        plot_binary_results(binary_results, save_path="binary_classifier.png")

    print("\n=== Podsumowanie ===")
    print(f"{'Partycja':<15} {'Accuracy':>10} {'±':>6}")
    print("-" * 35)
    for part, r in binary_results.items():
        print(f"{part:<15} {r['cv_accuracy_mean']:>10.3f} {r['cv_accuracy_std']:>6.3f}")

    return all_models, all_signatures, binary_results


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./Pythia"
    main(pythia_root=root, max_samples=100)