from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from src.feature_experiment import (
    GROUP_COLUMN,
    RANDOM_STATE,
    RAW_FEATURES,
    SCHEMA_PATH,
    TARGET_COLUMN,
    TRAIN_PATH,
    audit_handoff,
    read_json,
    validate_model_inputs,
    write_json_exclusive,
)
from src.train_models import make_group_folds

PROXY_COLUMN = "development_only_leakage_proxy"
REPORT_PATH = Path("reports/v2/leakage_proxy_experiment.json")


def run_leakage_proxy_experiment(root: Path) -> dict:
    audit_handoff(root)
    schema = read_json(root / SCHEMA_PATH)
    train = pd.read_csv(root / TRAIN_PATH)
    attacked = train.copy()
    attacked[PROXY_COLUMN] = attacked[TARGET_COLUMN]
    folds = make_group_folds(train, schema)
    naive_scores = []
    for fit_indices, validation_indices in folds:
        fit = attacked.iloc[fit_indices]
        validation = attacked.iloc[validation_indices]
        model = clone(DecisionTreeClassifier(random_state=RANDOM_STATE))
        model.fit(fit[[*RAW_FEATURES, PROXY_COLUMN]], fit[TARGET_COLUMN])
        predictions = model.predict(validation[[*RAW_FEATURES, PROXY_COLUMN]])
        naive_scores.append(
            float(accuracy_score(validation[TARGET_COLUMN], predictions))
        )

    try:
        validate_model_inputs([*RAW_FEATURES, PROXY_COLUMN], schema)
    except ValueError as error:
        rejection = {"status": "rejected before fitting", "reason": str(error)}
    else:
        raise AssertionError("The repaired gate accepted the leakage proxy")

    report = {
        "report_version": "2.0.0",
        "development_only": True,
        "consumed_v1_holdout_read": False,
        "proxy_column": PROXY_COLUMN,
        "proxy_definition": "exact copy of the development target for an intentionally invalid demonstration",
        "naive_grouped_validation_accuracy": round(fmean(naive_scores), 6),
        "naive_result_status": "invalid leakage demonstration",
        "repaired_pipeline": rejection,
    }
    write_json_exclusive(root / REPORT_PATH, report)
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_leakage_proxy_experiment(root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()