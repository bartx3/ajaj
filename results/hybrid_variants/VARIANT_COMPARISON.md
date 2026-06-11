# Porównanie wariantów kroku 4
Trening: ataki [1, 2, 3] + clean.

## Średnie OOD ROC-AUC
| Wariant | Średnie OOD | Opis |
|---------|-------------|------|
| hybrid_default | 0.6150 | Hybryda: AE encoder + fine-tuning |
| hybrid_frozen_encoder | 0.5188 | Hybryda: zamrożony encoder AE |
| generalization_cnn | 0.5170 | Generalizacja (krok 2, CNN) |
| hybrid_deep_head | 0.5138 | Hybryda: zamrożony AE + głęboka głowa |
| hybrid_wider_head | 0.5033 | Hybryda: zamrożony AE + szersza głowa |
| cnn_binary | 0.4816 | CNN binarny (piksele) |
| hybrid_no_pretrain | 0.2917 | Hybryda: encoder od zera (bez AE) |

## Wnioski (automatyczne)
- **Najlepszy średni OOD:** `hybrid_default` (0.6150) — Hybryda: AE encoder + fine-tuning.
- **cnn_binary** = ten sam CNN co krok 2, ale w ramach eksperymentu kroku 4.
- **generalization_cnn** = oficjalny krok 2 (powinien być zbliżony do cnn_binary).
- **hybrid_default** = obecny model (AE encoder + fine-tuning głowy).
- Jeśli hybryda z zamrożonym encoderem ≈ hybryda z fine-tuning → reprezentacja AE wystarczy; jeśli fine-tuning wyraźnie lepszy → warto dostosować encoder do detekcji.
- Jeśli **cnn_binary** ≥ hybrydy → prosty klasyfikator na pikselach wystarczy dla tego zadania.
