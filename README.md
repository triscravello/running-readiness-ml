# Level 1 Data: Time-Aware, Group-Safe Run Readiness Handoff

This project creates the `v1` data contract for a future next-day running-readiness classifier. It prepares and validates data only. Model fitting and baseline evaluation begin in Level 2.

The synthetic target and ranges are educational and are not medical or coaching advice.

## Prediction contract

At 20:00 local time on observation day D, predict `needs_recovery_next_day` for D+1. The label becomes observable at 08:00 on D+1.

Allowed features contain information available by 20:00 on D:

- distance accumulated through day D
- sleep from the previous night
- morning resting heart rate from day D
- evening soreness recorded by 20:00 on day D
- prior injury history

Runner ID, dates, timestamps, and the target are context or supervision metadata, not model features.

## Core concepts

- Prediction time defines the boundary between valid features and future leakage.
- Preprocessing statistics are learned state and must be fitted after splitting.
- Repeated observations from one runner are dependent, so runner groups must not cross the holdout boundary.
- A reusable ML handoff includes datasets, preprocessing state, a schema contract, quality evidence, tests, and documentation.

## Leakage experiment

The intentionally wrong path computes feature means from all cleaned rows before splitting. The experiment then changes only held-out runner values and shows that the global mean changes.

The correct path first splits runners, computes means from training runners only, and applies those fixed values to both partitions. Changing held-out data cannot change the training artifact.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/data_pipeline.py
python -m unittest -v tests/test_data_pipeline.py
```

## Versioned Level 2 handoff

- `data/v1/raw/run_readiness_observations.csv`
- `data/v1/cleaned/run_readiness_observations.csv`
- `data/v1/processed/train.csv`
- `data/v1/processed/test.csv`
- `artifacts/v1/preprocessing.json`
- `reports/v1/data_quality_report.json`
- `contracts/v1/schema_contract.json`

## Limitations

- Data, target logic, and validity ranges are synthetic.
- Mean imputation is chosen for a transparent leakage experiment, not because it is universally optimal.
- The group holdout evaluates unseen runners, but model performance is intentionally not measured in Level 1.
- Level 2 should treat the `v1` contract and split as inputs rather than silently regenerating them.

## Next level

Level 2 Train will load the processed data and contract, build a naive reference, fit a simple scikit-learn baseline, and evaluate whether learning adds measurable value.

<!-- LEVEL-3-EVALUATE:START -->
## Level 3: Evaluate

Evaluated at `2026-11-01T21:37:51.910570+00:00` after `decision_tree` was locked from grouped validation evidence.

| Model | Accuracy | Precision 1 | Recall 1 | F1 1 |
| --- | ---: | ---: | ---: | ---: |
| `decision_tree` | 0.833 | 0.913 | 0.840 | 0.875 |
| `majority_baseline` | 0.694 | 0.694 | 1.000 | 0.820 |

Selected confusion matrix `[[TN, FP], [FN, TP]]`: `[[9, 2], [4, 21]]`

- [Evaluation report](reports/v1/level3_evaluation.md)
- [Machine-readable metrics](reports/v1/level3_evaluation.json)
- [Error analysis](reports/v1/level3_errors.csv)
- [Consumed-holdout marker](artifacts/v1/holdout_consumed.json)
- [Level 4 handoff](artifacts/v1/level4_features_contract.json)

Reproduce safeguards with `python -m unittest -v tests/test_data_pipeline.py tests/test_training.py tests/test_evaluation.py`. The holdout command is intentionally excluded because v1 test labels are consumed.
<!-- LEVEL-3-EVALUATE:END -->

<!-- LEVEL-4-FEATURES:START -->
## Level 4: Features

[Feature experiment](docs/level-4-features.md)

<!-- LEVEL-4-FEATURES:END -->

<!-- LEVEL_5_NETWORKS -->
- [Level 5: Networks](docs/level-5-networks.md)

<!-- LEVEL_6_FINETUNE -->
- [Level 6: Fine-Tune](docs/level-6-finetune.md)
