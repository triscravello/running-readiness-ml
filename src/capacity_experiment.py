from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.train_models import (
    DATASET_VERSION,
    RANDOM_STATE,
    load_training_handoff,
    make_group_folds,
)

def tree_candidates() -> dict[str, Pipeline]:
    return {
        "unrestricted": Pipeline(
            steps=[
                (
                    "classifier",
                    DecisionTreeClassifier(random_state=RANDOM_STATE),
                )
            ]
        ),
        "depth_5_leaf_2": Pipeline(
            steps=[
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "depth_3_leaf_5": Pipeline(
            steps=[
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=3,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def evaluate_tree_capacity(
    train: Any,
    contract: dict[str, Any],
) -> dict[str, Any]:
    features = contract["feature_columns"]
    target = contract["target_column"]
    folds = make_group_folds(train, contract)
    results = {}

    for name, candidate in tree_candidates().items():
        fold_results = []
        for fold_number, (fit_indices, validation_indices) in enumerate(
            folds, start=1
        ):
            fit = train.iloc[fit_indices]
            validation = train.iloc[validation_indices]
            model = clone(candidate)
            model.fit(fit[features], fit[target])
            training_accuracy = float(
                accuracy_score(
                    fit[target],
                    model.predict(fit[features]),
                )
            )
            validation_accuracy = float(
                accuracy_score(
                    validation[target],
                    model.predict(validation[features]),
                )
            )
            tree = model.named_steps["classifier"]
            fold_results.append(
                {
                    "fold": fold_number,
                    "observed_depth": int(tree.get_depth()),
                    "training_accuracy": round(training_accuracy, 6),
                    "validation_accuracy": round(validation_accuracy, 6),
                    "generalization_gap": round(
                        training_accuracy - validation_accuracy,
                        6,
                    ),
                }
            )
        training_scores = [
            fold["training_accuracy"] for fold in fold_results
        ]
        validation_scores = [
            fold["validation_accuracy"] for fold in fold_results
        ]
        mean_training = fmean(training_scores)
        mean_validation = fmean(validation_scores)
        results[name] = {
            "folds": fold_results,
            "aggregate": {
                "mean_training_accuracy": round(mean_training, 6),
                "mean_validation_accuracy": round(mean_validation, 6),
                "validation_accuracy_std": round(
                    pstdev(validation_scores),
                    6,
                ),
                "generalization_gap": round(
                    mean_training - mean_validation,
                    6,
                ),
            },
        }

    return results

def run_capacity_experiment(project_root: Path) -> dict[str, Any]:
    train, contract, _ = load_training_handoff(project_root)
    report = {
        "dataset_version": contract["dataset_version"],
        "validation_boundary": "four GroupKFold splits from train.csv only",
        "complexity_interpretation": (
            "More depth and smaller leaves increase capacity; limits act as "
            "regularization. Compare observed gaps rather than assuming a winner."
        ),
        "results": evaluate_tree_capacity(train, contract),
    }

    report_path = (
        project_root
        / "reports"
        / DATASET_VERSION
        / "tree_capacity_experiment.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    report = run_capacity_experiment(project_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()