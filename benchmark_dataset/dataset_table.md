# Benchmark Dataset — Domain Question Summary

## Domain Overview

| Domain | Question Summary | # Questions | Sources |
|---|---|---|---|
| Cardiology | STEMI management, 65-year-old, inferior ST elevation | 5 | MedQuAD, MedDialog, Medical Meadow |
| Pharmacology | Metformin contraindications and renal safety criteria | 5 | MedQuAD, MedDialog, Medical Meadow |
| Neurology | Thunderclap headache workup, SAH rule-out protocol | 5 | MedQuAD, MedDialog, Medical Meadow |
| Pediatrics | 2-month vaccination schedule, US CDC ACIP guidelines | 5 | MedQuAD, MedDialog, Medical Meadow |
| Emergency | BLS protocol, unresponsive non-breathing patient | 5 | MedQuAD, MedDialog, Medical Meadow |
| **Total** | | **25** | |

## Representative Questions per Domain

| Domain | Representative Question |
|---|---|
| Cardiology | A 65-year-old presents with inferior ST elevation. What is the management? |
| Pharmacology | What are the contraindications for metformin use in patients with renal impairment? |
| Neurology | A patient presents with a thunderclap headache. How do you rule out subarachnoid hemorrhage? |
| Pediatrics | What vaccines are recommended at the 2-month well-child visit per CDC ACIP guidelines? |
| Emergency | What are the steps for BLS in an unresponsive, non-breathing adult? |

## Source Dataset Breakdown

| Source | # Questions |
|---|---|
| placeholder | 25 |

## Dataset Citations

- **MedQuAD**: Ben Abacha A & Demner-Fushman D. A Question-Entailment Approach to Question Answering. *BMC Bioinformatics*. 2019.
- **MedDialog**: Zeng G et al. MedDialog: Large-scale Medical Dialogue Datasets. *EMNLP 2020*.
- **Medical Meadow Health Advice**: Han T et al. MedAlpaca. *arXiv 2023*.
- **Origin repo**: https://github.com/m22oct2000/Multi-LLMs-as-Judge

*Questions without sufficient real-data coverage are filled with curated domain-representative placeholders.*