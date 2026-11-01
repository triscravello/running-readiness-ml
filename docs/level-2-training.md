# Level 2 Train: Group-Safe Running-Readiness Classification

## Problem

Predict the synthetic `needs_recovery_next_day` label from the five features defined by the Level 1 v1 schema. Level 2 performs model selection only inside the existing training partition. The original test partition remains unread and unscored for Level 3.

## How the models learn

- Majority baseline: learns only the most common training label and ignores all features. It is the minimum reference a feature-learning model should justify exceeding.
- Logistic regression: learns a regularized linear log-odds score. A sigmoid converts the score to a probability, and the default 0.5 threshold converts probability to class. `StandardScaler` is fitted inside each fold's Pipeline.
- Decision tree: greedily chooses feature thresholds that reduce Gini impurity. Unrestricted depth increases capacity, which can improve training fit while increasing sensitivity to the particular runners seen during fitting.

## Validation design

Four-fold `GroupKFold` uses only `data/v1/processed/train.csv`. Every runner appears in validation exactly once and never appears on both sides of one fold. Mean validation accuracy supports model selection. Validation standard deviation shows fold sensitivity. The difference between mean training and validation accuracy is the reported generalization gap.

The v1 mean-imputation artifact is a frozen upstream input. Level 2 does not refit it. Every new Level 2 learned transform stays inside a Pipeline and is fitted separately on each fold's training runners.

## Level 3 handoff

- `models/v1/majority_baseline.joblib`
- `models/v1/logistic_regression.joblib`
- `models/v1/decision_tree.joblib`
- `reports/v1/training_experiment.json`
- `artifacts/v1/level3_evaluation_contract.json`

The contract records input fingerprints, feature order, validation metrics, selected model and hyperparameters, seeds, runtime versions, and Pipeline paths. Level 3 must load a trusted Pipeline and call `predict` on `data/v1/processed/test.csv` without calling `fit` first.

## Reproduce

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.train_models
python -m unittest -v tests/test_training.py
```

## Security and limitations

Joblib is pickle-based. Never load an artifact from an untrusted source, and use the recorded package versions.

The data, target logic, and thresholds are synthetic educational examples. Accuracy is sufficient for the Level 2 baseline and generalization lesson, but Level 3 should use a broader metric set for final evaluation. Validation-selected means best among these candidates on grouped training folds, not proven best on future real runners.