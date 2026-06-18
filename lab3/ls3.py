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
from sklearn.ensemble import RandomForestClassifier
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
    """Oblicza podpis różnicowy - różnicę między aktywacją piksela a baseline (pusty obraz)."""
    # Baseline - pusty obraz
    baseline = np.zeros((1, H, W), dtype=np.float32)
    baseline_pred = model(baseline, training=False).numpy()[0]  # Shape: (K,)
    
    # Batch forward pass dla wszystkich pikseli
    batch = np.zeros((H * W, H, W), dtype=np.float32)
    for idx, (i, j) in enumerate((i, j) for i in range(H) for j in range(W)):
        batch[idx, i, j] = activation_value

    preds = model(batch, training=False).numpy()  # Shape: (H*W, K)
    
    # Różnica między każdym pikselem a baseline (broadcasting)
    diff_sig = preds - baseline_pred  # Shape: (H*W, K)
    
    return diff_sig.reshape(H, W, K)


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


def plot_models_comparison(lr_results: dict[str, float], rf_results: dict[str, float], nn_results: dict[str, float], nn_weighted_results: dict[str, float], save_path: str = "models_comparison.png"):
    """Generuje wykres słupkowy porównujący Regresję Logistyczną, Random Forest, NN bez wag i NN z wagami."""
    labels = list(lr_results.keys())
    x = np.arange(len(labels))
    width = 0.2  # Szerokość słupków

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 2.2), 5))
    
    # Rysowanie słupków dla czterech modeli
    bars_lr = ax.bar(x - 1.5*width, [lr_results[l] for l in labels], width, label='Regresja Logistyczna', color='#e05c5c', alpha=0.85, edgecolor='black', linewidth=0.7)
    bars_rf = ax.bar(x - 0.5*width, [rf_results[l] for l in labels], width, label='Random Forest', color='#5c9be0', alpha=0.85, edgecolor='black', linewidth=0.7)
    bars_nn = ax.bar(x + 0.5*width, [nn_results[l] for l in labels], width, label='Neural Network (bez wag)', color='#5ce085', alpha=0.85, edgecolor='black', linewidth=0.7)
    bars_nn_w = ax.bar(x + 1.5*width, [nn_weighted_results[l] for l in labels], width, label='Neural Network (z wagami)', color='#e0a05c', alpha=0.85, edgecolor='black', linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", color="gray", label="Poziom losowy (0.5)")
    ax.set_ylabel("Skuteczność rozpoznania")
    ax.set_title("Porównanie modeli (Holdout)")
    ax.legend(loc="lower left", fontsize=9)
    
    # Dodanie etykiet tekstowych nad słupkami
    for bar in bars_lr:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h*100:.0f}%", ha="center", fontsize=6, fontweight='bold')
    for bar in bars_rf:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h*100:.0f}%", ha="center", fontsize=6, fontweight='bold')
    for bar in bars_nn:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h*100:.0f}%", ha="center", fontsize=6, fontweight='bold')
    for bar in bars_nn_w:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h*100:.0f}%", ha="center", fontsize=6, fontweight='bold')
                
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Zapisano wykres porównawczy: {save_path}")


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

    # 1. PRZYGOTOWANIE ŚCIEŻEK
    train_paths, test_paths = {}, {}
    for part in partitions:
        all_pngs = sorted([str(p) for p in (root / part).glob("*.png")])
        train_paths[part] = all_pngs[:max_samples]
        test_paths[part] = all_pngs[max_samples : 2 * max_samples]

    # 2. OBLICZANIE PODPISÓW (TRENING)
    print("=== TRENING: Obliczanie One-Pixel Signatures ===")
    train_signatures = {}
    for part in partitions:
        print(f"  Ładowanie treningowych dla: {part}...", end=" ", flush=True)
        train_signatures[part] = compute_signatures_parallel(train_paths[part], activation_value)
        print("OK")

    # Wygenerowanie średnich heatmap dla zbioru treningowego
    plot_mean_signatures(train_signatures, title_suffix="Trening", save_path="mean_signatures_train.png")

    # 3. PRZYGOTOWANIE DANYCH DO TRENINGU
    print("\n=== Przygotowanie danych treningowych ===")
    
    # Pełny zbiór bez downsamplingu (dla NN)
    X_train_full_list, y_train_full_list = [], []
    clean_train = train_signatures["clean"]
    X_train_full_list.append(np.array([s.flatten() for s in clean_train]))
    y_train_full_list.append(np.zeros(len(clean_train)))
    
    active_attacks = [att for att in attacks if len(train_signatures[att]) > 0]
    if len(active_attacks) > 0:
        for att in active_attacks:
            att_train = train_signatures[att]
            X_train_full_list.append(np.array([s.flatten() for s in att_train]))
            y_train_full_list.append(np.ones(len(att_train)))
    
    X_train_full = np.vstack(X_train_full_list)
    y_train_full = np.concatenate(y_train_full_list)
    
    print(f"[OK] Pełny zbiór treningowy: {X_train_full.shape[0]} modeli (Czyste: {np.sum(y_train_full==0)}, Ataki razem: {np.sum(y_train_full==1)})")
    
    # Wyrównany zbiór dla LR i RF
    print("\n=== Przygotowanie zbalansowanych danych treningowych ===")
    X_train_list, y_train_list = [], []
    
    X_train_list.append(np.array([s.flatten() for s in clean_train]))
    y_train_list.append(np.zeros(len(clean_train)))
    
    if len(active_attacks) > 0:
        samples_per_attack = len(clean_train) // len(active_attacks)
        print(f"  -> Wyrównywanie proporcji 50/50: Pobieram po {samples_per_attack} próbek z każdego z {len(active_attacks)} ataków.")
        
        for att in active_attacks:
            att_train = train_signatures[att]
            truncated_att = att_train[:samples_per_attack]
            X_train_list.append(np.array([s.flatten() for s in truncated_att]))
            y_train_list.append(np.ones(len(truncated_att)))
            
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    print(f"[OK] Wyrównany zbiór treningowy: {X_train.shape[0]} modeli (Czyste: {np.sum(y_train==0)}, Ataki razem: {np.sum(y_train==1)})")

    # Skalowanie cech
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_full_scaled = scaler.transform(X_train_full)
    
    # --- TRENING MODELU 1: Regresja Logistyczna (wyrównane dane) ---
    print("\nTrenowanie Regresji Logistycznej...", end=" ")
    clf_lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    clf_lr.fit(X_train_scaled, y_train)
    print("OK")

    # --- TRENING MODELU 2: Random Forest Classifier (wyrównane dane) ---
    print("Trenowanie Random Forest (200 drzew)...", end=" ")
    clf_rf = RandomForestClassifier(n_estimators=200, max_depth=50, random_state=42)
    clf_rf.fit(X_train, y_train)  
    print("OK")

    # --- TRENING MODELU 3: Neural Network (pełny zbiór BEZ class_weights) ---
    print("Trenowanie Neural Network bez wag (pełny zbiór)...", end=" ", flush=True)
    
    clf_nn = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_dim=X_train_full.shape[1]),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation="swish"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    clf_nn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    clf_nn.fit(
        X_train_full_scaled, y_train_full,
        epochs=100,
        batch_size=16,
        verbose=0
    )
    print("OK")

    # Oblicz wagi klas dla wyrównania
    unique, counts = np.unique(y_train_full, return_counts=True)
    class_weights = {int(u): float(counts.max()) / float(c) for u, c in zip(unique, counts)}
    
    # --- TRENING MODELU 4: Neural Network (pełny zbiór Z class_weights) ---
    print("Trenowanie Neural Network z wagami (pełny zbiór, class_weights)...", end=" ", flush=True)
    
    clf_nn_weighted = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_dim=X_train_full.shape[1]),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation="swish"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    clf_nn_weighted.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    clf_nn_weighted.fit(
        X_train_full_scaled, y_train_full,
        epochs=100,
        batch_size=16,
        class_weight=class_weights,
        verbose=0
    )
    print("OK")

    # 4. POBIERANIE NOWEJ PULI PRÓBEK (TEST)
    print("\n=== TEST: Obliczanie podpisów dla nowej puli (Holdout) ===")
    test_signatures = {}
    for part in partitions:
        if len(test_paths[part]) == 0:
            continue
        print(f"  Ładowanie testowych dla: {part}...", end=" ", flush=True)
        test_signatures[part] = compute_signatures_parallel(test_paths[part], activation_value)
        print("OK")

    # Wygenerowanie średnich heatmap dla zbioru testowego
    plot_mean_signatures(test_signatures, title_suffix="Test (Holdout)", save_path="mean_signatures_test.png")

    # 5. EWALUACJA PORÓWNAWCZA NA ZBIORZE HOLDOUT
    lr_results, rf_results, nn_results, nn_weighted_results = {}, {}, {}, {}
    print("\n" + "="*60 + "\nPORÓWNANIE WYNIKÓW NA ZBIORZE TESTOWYM (HOLDOUT)\n" + "="*60)
    
    # Test na czystych
    if "clean" in test_signatures:
        X_clean_test = np.array([s.flatten() for s in test_signatures["clean"]])
        X_clean_test_scaled = scaler.transform(X_clean_test)
        
        preds_lr_clean = clf_lr.predict(X_clean_test_scaled)
        preds_rf_clean = clf_rf.predict(X_clean_test)
        preds_nn_clean = (clf_nn.predict(X_clean_test_scaled, verbose=0) > 0.5).astype(int).flatten()
        preds_nn_weighted_clean = (clf_nn_weighted.predict(X_clean_test_scaled, verbose=0) > 0.5).astype(int).flatten()
        
        lr_results["clean"] = np.mean(preds_lr_clean == 0)
        rf_results["clean"] = np.mean(preds_rf_clean == 0)
        nn_results["clean"] = np.mean(preds_nn_clean == 0)
        nn_weighted_results["clean"] = np.mean(preds_nn_weighted_clean == 0)
        print(f"Klasa CLEAN (TNR)   -> LR: {lr_results['clean']*100:.1f}% | RF: {rf_results['clean']*100:.1f}% | NN: {nn_results['clean']*100:.1f}% | NN(wagi): {nn_weighted_results['clean']*100:.1f}%")

    # Test na atakach
    for att in attacks:
        if att in test_signatures and len(test_signatures[att]) > 0:
            X_att_test = np.array([s.flatten() for s in test_signatures[att]])
            X_att_test_scaled = scaler.transform(X_att_test)
            
            preds_lr_att = clf_lr.predict(X_att_test_scaled)
            preds_rf_att = clf_rf.predict(X_att_test)
            preds_nn_att = (clf_nn.predict(X_att_test_scaled, verbose=0) > 0.5).astype(int).flatten()
            preds_nn_weighted_att = (clf_nn_weighted.predict(X_att_test_scaled, verbose=0) > 0.5).astype(int).flatten()
            
            lr_results[att] = np.mean(preds_lr_att == 1)
            rf_results[att] = np.mean(preds_rf_att == 1)
            nn_results[att] = np.mean(preds_nn_att == 1)
            nn_weighted_results[att] = np.mean(preds_nn_weighted_att == 1)
            print(f"Atak {att:<14} -> LR: {lr_results[att]*100:.1f}% | RF: {rf_results[att]*100:.1f}% | NN: {nn_results[att]*100:.1f}% | NN(wagi): {nn_weighted_results[att]*100:.1f}%")

    # Przykładowe szczegółowe mapy (wgląd w pojedyncze modele)
    sample_sigs = {"clean (model #0)": train_signatures["clean"][0]}
    for att in active_attacks:
        sample_sigs[f"{att} (model #0)"] = train_signatures[att][0]
        if len(sample_sigs) >= 4:
            break
                
    plot_signature_heatmaps(sample_sigs, classes_to_show=[0, 1, 2, 3, 4])
    plot_models_comparison(lr_results, rf_results, nn_results, nn_weighted_results)

    return clf_lr, clf_rf, clf_nn, clf_nn_weighted, lr_results, rf_results, nn_results, nn_weighted_results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = sys.argv[1] if len(sys.argv) > 1 else "./Pythia"
    main(pythia_root=root, max_samples=500)