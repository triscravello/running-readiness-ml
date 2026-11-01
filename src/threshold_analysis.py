from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone

from src.evaluate_models import metric_bundle, verify_selection_lock
from src.train_models import build_model_candidates, load_training_handoff, make_group_folds

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
REPORT_RELATIVE_PATH = Path("reports/v1/threshold_analysis.json")


def run_threshold_analysis(root: Path) -> dict[str, Any]:
    report_path = root / REPORT_RELATIVE_PATH
    if report_path.exists():
        raise FileExistsError("Threshold report already exists")
    contract, _, verified_train, lock = verify_selection_lock(root)
    train, schema, _ = load_training_handoff(root)
    if not train.equals(verified_train):
        raise ValueError("Training handoff changed")
    selected_name = lock["selected_model"]
    candidates = build_model_candidates()
    if selected_name not in candidates:
        raise ValueError("Locked candidate is unavailable")
    features = list(schema["feature_columns"])
    target = schema["target_column"]
    group = schema["group_column"]
    probabilities = pd.Series(index=train.index, dtype=float)
    fold_evidence = []
    for number, (fit_indices, validation_indices) in enumerate(make_group_folds(train, schema), 1):
        fit = train.iloc[fit_indices]
        validation = train.iloc[validation_indices]
        fit_groups = {str(value) for value in fit[group].unique()}
        validation_groups = {str(value) for value in validation[group].unique()}
        overlap = sorted(fit_groups.intersection(validation_groups))
        if overlap:
            raise ValueError(f"Runner overlap in fold {number}")
        model = clone(candidates[selected_name])
        model.fit(fit[features], fit[target])
        classes = [int(value) for value in model.classes_]
        if 1 not in classes:
            raise ValueError(f"Positive class missing from fold {number}")
        probabilities.iloc[validation_indices] = model.predict_proba(validation[features])[:, classes.index(1)]
        fold_evidence.append({"fold": number, "fit_runner_count": len(fit_groups), "validation_runner_count": len(validation_groups), "runner_overlap": overlap, "validation_row_count": int(len(validation_indices))})
    if probabilities.isna().any():
        raise ValueError("Every training row needs one validation probability")
    y_true = train[target].astype(int)
    threshold_results = []
    for threshold in THRESHOLDS:
        predictions = [int(value > threshold) for value in probabilities.tolist()]
        metrics = metric_bundle(y_true, predictions)
        cm = metrics["confusion_matrix"]
        threshold_results.append({"role": "exploratory_threshold", "threshold": threshold, "precision": metrics["positive_class"]["precision"], "recall": metrics["positive_class"]["recall"], "f1": metrics["positive_class"]["f1"], "predicted_positive_count": int(sum(predictions)), "true_negative_count": cm["true_negatives"], "false_positive_count": cm["false_positives"], "false_negative_count": cm["false_negatives"], "true_positive_count": cm["true_positives"], "confusion_matrix": cm["matrix"]})
    validation_selected = max(threshold_results, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.50), -row["threshold"]))
    oof_records = [{"training_row_position": position, "runner_group": str(train.iloc[position][group]), "actual_class": int(train.iloc[position][target]), "probability_class_1": float(probabilities.iloc[position])} for position in range(len(train))]
    report = {"report_version": "1.1.0", "dataset_version": contract["dataset_version"], "selected_model": selected_name, "test_set_status": "test.csv and held-out labels were not read", "out_of_fold_source": "one probability per training row from its excluded group-safe validation fold", "fold_evidence": fold_evidence, "out_of_fold_probabilities": oof_records, "threshold_roles": {"held_out_evaluation_operating_point": "frozen Pipeline.predict default rule", "exploratory_analysis": "all five threshold rows", "validation_selected_operating_threshold": "highest out-of-fold validation F1 with ties toward 0.50", "threshold_applied_to_holdout": False}, "exploratory_threshold_results": threshold_results, "validation_selected_operating_threshold_proposal": {**validation_selected, "role": "validation_selected_operating_threshold_proposal", "selection_data": "group-safe out-of-fold training validation predictions only"}, "usage_rule": "Do not apply this proposal to the consumed v1 holdout for another evaluation."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_threshold_analysis(root), indent=2))


if __name__ == "__main__":
    main()