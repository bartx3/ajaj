"""
Problem 3: Weryfikacja Modeli – Krok 1 i 2 (Zmodfikowany globalny klasyfikator + średnie heatmapy)
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


def load_model_from_image(img_path: str) -> keras.Model:
    model = build_model()
    weights = decode_weights(img_path)
    
    trainable = [l for l in model.layers if len(l.get_weights()) > 0]
    for layer, (k, b) in zip(trainable, [(weights[0], weights[1]), (weights[2], weights[3])]):
        layer.set_weights([k, b])
    return model


# ─────────────────────────────────────────────
# KROK 2: One-Pixel Signature
# ─────────────────────────────────────────────

def one_pixel_signature(model: keras.Model,
                        activation_value: float = 1.0,
                        H: int = 21, W: int = 3, K: int = 10) -> np.ndarray:
    """Oblicza podpis gout(f) przez batch forward pass."""
    batch = np.zeros((H * W, H, W), dtype=np.float32)
    for idx, (i, j) in enumerate((i, j) for i in range(H) for j in range(W)):
        batch[idx, i, j] = activation_value

    preds = model(batch, training=False).numpy()
    return preds.reshape(H, W, K)


def _signature_worker(args) -> tuple[int, np.ndarray]:
    idx, img_path, activation_value = args
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
# Wizualizacje
# ─────────────────────────────────────────────

def plot_signature_heatmaps(signatures: dict[str, np.ndarray],
                             classes_to_show: list[int] = [0, 3, 7],
                             save_path: str = "signature_heatmaps.png"):
    n_models = len(signatures)
    n_classes = len(classes_to_show)
    fig, axes = plt.subplots(n_models, n_classes, figsize=(n_classes * 3, n_models * 3.5))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, (label, sig) in enumerate(signatures.items()):
        for col, k in enumerate(classes_to_show):
            ax = axes[row, col]
            im = ax.imshow(sig[:, :, k], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"{label}\nklasa {k}", fontsize=9)
            ax.set_xlabel("W"); ax.set_ylabel("H")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("One-Pixel Signature – S_k(f)(i,j)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_mean_signatures(all_signatures: dict[str, list[np.ndarray]],
                          title_suffix: str = "Trening",
                          save_path: str = "mean_signatures.png"):
    """Generuje i zapisuje średnie mapy sygnatur (norma L2 po klasach)."""
    if not all_signatures:
        return
        
    fig, axes = plt.subplots(1, len(all_signatures), figsize=(4 * len(all_signatures), 4))
    if len(all_signatures) == 1:
        axes = [axes]

    for ax, (label, sigs) in zip(axes, all_signatures.items()):
        if len(sigs) == 0:
            continue
        mean_sig = np.mean(sigs, axis=0)
        norm_map = np.linalg.norm(mean_sig, axis=-1)
        im = ax.imshow(norm_map, cmap="hot")
        ax.set_title(f"{label}\n‖mean gout‖₂", fontsize=10)
        ax.set_xlabel("W"); ax.set_ylabel("H")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Średnie podpisy ({title_suffix} - norma L2 po klasach)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Zapisano średnie heatmapy: {save_path}")


def plot_global_classifier_results(test_accuracies: dict[str, float], save_path: str = "global_classifier_results.png"):
    labels = list(test_accuracies.keys())
    values = list(test_accuracies.values())

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.5), 4.5))
    x = np.arange(len(labels))
    
    colors = ["#4caf50" if l == "clean" else "#e05c5c" for l in labels]
    
    bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", color="gray", label="Poziom losowy (0.5)")
    ax.set_ylabel("Skuteczność rozpoznania (Accuracy / TNR)")
    ax.set_title("Ewaluacja globalnego klasyfikatora na nowej puli próbek (Holdout)")
    ax.legend()
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val*100:.1f}%", ha="center", fontsize=9, fontweight='bold')
                
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] Zapisano wykres ewaluacji: {save_path}")


# ─────────────────────────────────────────────
# GŁÓWNY PIPELINE
# ─────────────────────────────────────────────

def main(pythia_root: str = "./pythia",
         max_samples: int = 30,
         activation_value: float = 1.0):

    root = Path(pythia_root)
    partitions = sorted([p.name for p in root.iterdir() if p.is_dir()])
    attacks = [p for p in partitions if p.startswith("attack_")]
    
    print(f"Znalezione partycje: {partitions}")
    print(f"Wykryte ataki: {attacks}")
    print(f"Używam {N_WORKERS} rdzeni CPU\n")

    # 1. PRZYGOTOWANIE ŚCIEŻEK DLA ZBIORU TRENINGOWEGO I TESTOWEGO
    train_paths: dict[str, list[str]] = {}
    test_paths: dict[str, list[str]] = {}
    
    for part in partitions:
        folder = root / part
        all_pngs = sorted([str(p) for p in folder.glob("*.png")])
        
        train_paths[part] = all_pngs[:max_samples]
        test_paths[part] = all_pngs[max_samples : 2 * max_samples]

    # 2. OBLICZANIE PODPISÓW DLA ZBIORU TRENINGOWEGO
    print("=== TRENING: Obliczanie One-Pixel Signatures ===")
    train_signatures: dict[str, list[np.ndarray]] = {}
    for part in partitions:
        print(f"  Ładowanie treningowych dla: {part} ({len(train_paths[part])} modeli)...", end=" ", flush=True)
        train_signatures[part] = compute_signatures_parallel(train_paths[part], activation_value)
        print("OK")

    # Wygenerowanie średnich heatmap dla zbioru treningowego
    plot_mean_signatures(train_signatures, title_suffix="Trening", save_path="mean_signatures_train.png")

    # 3. PRZYGOTOWANIE DANYCH DO POŁĄCZONEGO TRENINGU
    print("\n=== Przygotowanie połączonego klasyfikatora binarnego ===")
    X_train_list = []
    y_train_list = []
    
    clean_train = train_signatures["clean"]
    X_train_list.append(np.array([s.flatten() for s in clean_train]))
    y_train_list.append(np.zeros(len(clean_train)))
    
    for att in attacks:
        att_train = train_signatures[att]
        if len(att_train) > 0:
            X_train_list.append(np.array([s.flatten() for s in att_train]))
            y_train_list.append(np.ones(len(att_train)))
            
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    print(f"Łączny zbiór treningowy: {X_train.shape[0]} modeli (Czyste: {len(clean_train)}, Ataki razem: {X_train.shape[0] - len(clean_train)})")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_train_scaled, y_train)
    print("[OK] Klasyfikator binarny został wytrenowany na wszystkich danych.")

    # 4. POBIERANIE NOWEJ PULI PRÓBEK (TEST) I EWALUACJA
    print("\n=== TEST: Obliczanie podpisów dla nowej puli (Holdout) ===")
    test_signatures: dict[str, list[np.ndarray]] = {}
    for part in partitions:
        if len(test_paths[part]) == 0:
            continue
        print(f"  Ładowanie testowych dla: {part} ({len(test_paths[part])} modeli)...", end=" ", flush=True)
        test_signatures[part] = compute_signatures_parallel(test_paths[part], activation_value)
        print("OK")

    # Wygenerowanie średnich heatmap dla zbioru testowego (OOD)
    plot_mean_signatures(test_signatures, title_suffix="Test (Holdout)", save_path="mean_signatures_test.png")

    # Ewaluacja na zbiorze testowym
    print("\n=== WYNIKI NA NOWYCH PRÓBKACH ===")
    test_accuracies = {}
    
    if "clean" in test_signatures:
        X_clean_test = np.array([s.flatten() for s in test_signatures["clean"]])
        X_clean_test_scaled = scaler.transform(X_clean_test)
        preds_clean = clf.predict(X_clean_test_scaled)
        
        clean_acc = np.mean(preds_clean == 0)
        test_accuracies["clean"] = clean_acc
        print(f"Skuteczność rozpoznawania CZYSTYCH modeli (TNR): {clean_acc*100:.2f}% ({np.sum(preds_clean == 0)}/{len(preds_clean)})")

    print("-" * 50)
    total_correct_attacks = 0
    total_attack_samples = 0
    
    for att in attacks:
        if att in test_signatures and len(test_signatures[att]) > 0:
            X_att_test = np.array([s.flatten() for s in test_signatures[att]])
            X_att_test_scaled = scaler.transform(X_att_test)
            preds_att = clf.predict(X_att_test_scaled)
            
            att_acc = np.mean(preds_att == 1)
            test_accuracies[att] = att_acc
            
            total_correct_attacks += np.sum(preds_att == 1)
            total_attack_samples += len(preds_att)
            
            print(f"Skuteczność wykrywania ataku '{att}': {att_acc*100:.2f}% ({np.sum(preds_att == 1)}/{len(preds_att)})")

    print("=" * 50)
    global_att_acc = (total_correct_attacks / total_attack_samples) if total_attack_samples > 0 else 0
    print(f"Ogólna wykrywalność JAKIEGOKOLWIEK ataku: {global_att_acc*100:.2f}%")
    
    # Przykładowe szczegółowe mapy (wgląd w pojedyncze modele)
    sample_sigs = {"clean (model #0)": train_signatures["clean"][0]}
    for att in attacks:
        if train_signatures.get(att):
            sample_sigs[f"{att} (model #0)"] = train_signatures[att][0]
            if len(sample_sigs) >= 4:
                break
                
    plot_signature_heatmaps(sample_sigs, classes_to_show=[0, 1, 2, 3, 4])
    plot_global_classifier_results(test_accuracies)

    return clf, test_accuracies


if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = sys.argv[1] if len(sys.argv) > 1 else "./Pythia"
    main(pythia_root=root, max_samples=100)