from pathlib import Path

import pandas as pd

from src.feature_experiment import TARGET_COLUMN, TRAIN_PATH, metric_values, write_json_exclusive
from src.neural_network_experiment import (
    SEED,
    approved_matrix,
    audit_level4,
    evaluate_model,
    fit_preprocessor,
    inner_group_split,
    train_fixed,
    train_with_early_stopping,
    transform,
)
from src.train_models import GROUP_COLUMN, SCHEMA_PATH, make_group_folds
from src.feature_experiment import read_json

REPORT_PATH = Path("reports/v3/overfit_experiment.json")


def main() -> None:
    root = Path.cwd()
    audit_level4(root)
    x, y, groups, _ = approved_matrix(root)
    folds = make_group_folds(pd.read_csv(root / TRAIN_PATH), read_json(root / SCHEMA_PATH))
    fit_rows, validation_rows = folds[0]
    fit_groups = sorted(groups.iloc[fit_rows].unique())[:2]
    tiny_mask = groups.iloc[fit_rows].isin(fit_groups).to_numpy()
    tiny_rows = fit_rows[tiny_mask]
    if set(groups.iloc[tiny_rows]) & set(groups.iloc[validation_rows]):
        raise ValueError("Secret Mission runner overlap")
    imputer, scaler = fit_preprocessor(x.iloc[tiny_rows])
    x_tiny = transform(x.iloc[tiny_rows], imputer, scaler)
    y_tiny = y.iloc[tiny_rows].to_numpy("float32")
    x_validation = transform(x.iloc[validation_rows], imputer, scaler)
    y_validation = y.iloc[validation_rows].to_numpy("float32")

    unregularized = {"name": "wide_unregularized", "hidden_units": 128,
                     "dropout": 0.0, "learning_rate": 0.01, "weight_decay": 0.0}
    regularized = {"name": "wide_regularized", "hidden_units": 128,
                   "dropout": 0.4, "learning_rate": 0.01, "weight_decay": 0.01}
    memorizer, raw_history = train_fixed(
        x_tiny, y_tiny, unregularized, SEED, 300, x_validation, y_validation)
    guarded, guarded_history, best_epoch = train_with_early_stopping(
        x_tiny, y_tiny, x_validation, y_validation, regularized, SEED, 300, 20)
    raw_train = evaluate_model(memorizer, x_tiny, y_tiny)
    raw_validation = evaluate_model(memorizer, x_validation, y_validation)
    guarded_train = evaluate_model(guarded, x_tiny, y_tiny)
    guarded_validation = evaluate_model(guarded, x_validation, y_validation)
    payload = {
        "report_version": "3.0.0", "consumed_v1_holdout_read": False,
        "tiny_training_runners": fit_groups,
        "validation_runners": sorted(groups.iloc[validation_rows].unique()),
        "unregularized": {"config": unregularized, "history": raw_history,
                          "train": raw_train, "validation": raw_validation,
                          "f1_gap": raw_train["f1"] - raw_validation["f1"]},
        "regularized": {"config": regularized, "history": guarded_history,
                        "best_epoch": best_epoch, "train": guarded_train,
                        "validation": guarded_validation,
                        "f1_gap": guarded_train["f1"] - guarded_validation["f1"]},
        "diagnosis": "Compare the loss-curve divergence and F1 gaps. Lower training loss alone is not evidence of better runner generalization.",
    }
    write_json_exclusive(root / REPORT_PATH, payload)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()