# Wyniki na zbiorze Pythia
Metryka: ROC-AUC (clean=0 vs atak=1).

## Tryb `transfer`

| Model | Trening | a | b | c | d | e | f | g | h |
|-------|---------|------|------|------|------|------|------|------|------|
| autoencoder_step3 | FMNIST clean only | 0.508 | 0.511 | 0.527 | 0.510 | 0.508 | 0.514 | 0.515 | 0.498 |
| baseline_step1 | FMNIST clean vs attack_1 | 0.521 | 0.523 | 0.514 | 0.498 | 0.511 | 0.499 | 0.509 | 0.477 |
| generalization_step2 | FMNIST attacks 1,2,3 | 0.499 | 0.494 | 0.479 | 0.494 | 0.474 | 0.476 | 0.484 | 0.475 |
| hybrid_step4 | FMNIST AE + clean vs attack_1 | 0.509 | 0.525 | 0.521 | 0.531 | 0.521 | 0.495 | 0.506 | 0.523 |

## Tryb `native`

| Model | Trening | a | b | c | d | e | f | g | h |
|-------|---------|------|------|------|------|------|------|------|------|
| autoencoder_step3 | Pythia clean only | 0.463 | 0.492 | 0.497 | 0.487 | 0.487 | 0.506 | 0.488 | 0.478 |
| baseline_step1 | Pythia clean vs attack_a | — | 0.517 | 0.523 | 0.502 | 0.526 | 0.511 | 0.508 | 0.500 |
| generalization_step2 | Pythia attacks a,b,c | — | — | — | 0.569 | 0.551 | 0.545 | 0.542 | 0.549 |
| hybrid_step4 | Pythia AE + clean vs attack_a | — | 0.514 | 0.474 | 0.500 | 0.500 | 0.507 | 0.498 | 0.501 |
