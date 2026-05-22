# Agreement Benchmark Dataset

Built from MedQuAD + MedDialog + Medical Meadow (or synthetic fallback).
Synthetic fallback: 6 domains x 4 classes x 5 rows = 120 rows.

## Expected Class Logic

| Class | Criteria |
|---|---|
| `fully_agree` | Mean >= 0.72, >= 8/12 clearly met, <= 2 ambiguous |
| `majority_agree` | Mean >= 0.55, >= 5 clear; OR emergency Q + safety flag |
| `split` | Mean >= 0.35, >= 5 ambiguous |
| `full_disagree` | Low mean or answer too short |

**Total rows:** 120

| Expected Class | # Rows |
|---|---|
| `fully_agree` | 30 |
| `majority_agree` | 30 |
| `split` | 30 |
| `full_disagree` | 30 |

| Domain | # Rows |
|---|---|
| Cardiology | 20 |
| Pharmacology | 20 |
| Neurology | 20 |
| Pediatrics | 20 |
| Emergency | 20 |
| General | 20 |