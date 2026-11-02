from __future__ import annotations

import json
import platform
from hashlib import sha256
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATASET_VERSION = "v1"
RANDOM_STATE = 42
GROUP_FOLDS = 4

# Declare these globally for Level 5
GROUP_COLUMN = "runner_id"
SCHEMA_PATH = Path("contracts/v1/schema_contract.json")


def fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def handoff_paths(project_root: Path) -> dict[str, Path]:
    return {
        "train": project_root / "data" / DATASET_VERSION / "processed" / "train.csv",
        "schema": project_root / "contracts" / DATASET_VERSION / "schema_contract.json",
        "preprocessing": project_root / "artifacts" / DATASET_VERSION / "preprocessing.json",
    }


def load_training_handoff(
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    paths = handoff_paths(project_root)
    contract = json.loads(paths["schema"].read_text(encoding="utf-8"))
    preprocessing = json.loads(
        paths["preprocessing"].read_text(encoding="utf-8")
    )
    train = pd.read_csv(paths["train"])

    if contract["dataset_version"] != DATASET_VERSION:
        raise ValueError("Schema dataset version does not match Level 2")
    if preprocessing["dataset_version"] != DATASET_VERSION:
        raise ValueError("Preprocessing dataset version does not match Level 2")
    if preprocessing["feature_columns"] != contract["feature_columns"]:
        raise ValueError("Preprocessing feature order does not match the schema")

    required = [
        *contract["feature_columns"],
        contract["target_column"],
        contract["group_column"],
    ]
    missing = sorted(set(required) - set(train.columns))
    if missing:
        raise ValueError(f"Training handoff is missing columns: {missing}")
    if int(train[contract["feature_columns"]].isna().sum().sum()) != 0:
        raise ValueError("Training features must satisfy v1 preprocessing")
    if int(train[contract["target_column"]].nunique()) < 2:
        raise ValueError("Training target must contain at least two classes")

    return train, contract, preprocessing


def build_model_candidates() -> dict[str, Pipeline]:
    return {
        "majority_baseline": Pipeline(
            steps=[
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                (
                    "classifier",
                    DecisionTreeClassifier(random_state=RANDOM_STATE),
                ),
            ]
        ),
    }


def model_settings() -> dict[str, dict[str, Any]]:
    return {
        "majority_baseline": {
            "pipeline_steps": ["classifier"],
            "hyperparameters": {"strategy": "most_frequent"},
        },
        "logistic_regression": {
            "pipeline_steps": ["scaler", "classifier"],
            "hyperparameters": {
                "scaler": "StandardScaler",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
            },
        },
        "decision_tree": {
            "pipeline_steps": ["classifier"],
            "hyperparameters": {"random_state": RANDOM_STATE},
        },
    }


def make_group_folds(
    train: pd.DataFrame,
    contract: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(
        n_splits=GROUP_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return list(
        splitter.split(
            train[contract["feature_columns"]],
            train[contract["target_column"]],
            groups=train[contract["group_column"]],
        )
    )


def build_fold_manifest(
    train: pd.DataFrame,
    contract: dict[str, Any],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    group_column = contract["group_column"]
    manifest = []
    for fold_number, (fit_indices, validation_indices) in enumerate(folds, start=1):
        fit_groups = sorted(
            str(value) for value in train.iloc[fit_indices][group_column].unique()
        )
        validation_groups = sorted(
            str(value)
            for value in train.iloc[validation_indices][group_column].unique()
        )
        overlap = sorted(set(fit_groups).intersection(validation_groups))
        manifest.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_indices)),
                "validation_rows": int(len(validation_indices)),
                "fit_runners": fit_groups,
                "validation_runners": validation_groups,
                "runner_overlap": overlap,
            }
        )
    return manifest


def evaluate_candidates(
    train: pd.DataFrame,
    contract: dict[str, Any],
    folds: list[tuple[np.ndarray, np.ndarray]],
    candidates: dict[str, Pipeline],
) -> dict[str, dict[str, Any]]:
    features = contract["feature_columns"]
    target = contract["target_column"]
    results: dict[str, dict[str, Any]] = {}

    for name, candidate in candidates.items():
        fold_results = []
        for fold_number, (fit_indices, validation_indices) in enumerate(
            folds, start=1
        ):
            fit = train.iloc[fit_indices]
            validation = train.iloc[validation_indices]
            model = clone(candidate)
            model.fit(fit[features], fit[target])
            fit_predictions = model.predict(fit[features])
            validation_predictions = model.predict(validation[features])
            training_accuracy = float(
                accuracy_score(fit[target], fit_predictions)
            )
            validation_accuracy = float(
                accuracy_score(validation[target], validation_predictions)
            )
            fold_results.append(
                {
                    "fold": fold_number,
                    "training_accuracy": round(training_accuracy, 6),
                    "validation_accuracy": round(validation_accuracy, 6),
                    "generalization_gap": round(
                        training_accuracy - validation_accuracy,
                        6,
                    ),
                    "unique_validation_predictions": sorted(
                        int(value)
                        for value in set(validation_predictions.tolist())
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


def fit_and_save_models(
    train: pd.DataFrame,
    contract: dict[str, Any],
    project_root: Path,
) -> dict[str, str]:
    features = contract["feature_columns"]
    target = contract["target_column"]
    model_directory = project_root / "models" / DATASET_VERSION
    model_directory.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}

    for name, model in build_model_candidates().items():
        model.fit(train[features], train[target])
        path = model_directory / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        artifact_paths[name] = path.relative_to(project_root).as_posix()

    return artifact_paths


def build_metadata(
    train: pd.DataFrame,
    contract: dict[str, Any],
    preprocessing: dict[str, Any],
    folds: list[tuple[np.ndarray, np.ndarray]],
    results: dict[str, dict[str, Any]],
    artifact_paths: dict[str, str],
    project_root: Path,
) -> dict[str, Any]:
    paths = handoff_paths(project_root)
    group_column = contract["group_column"]
    target_column = contract["target_column"]
    settings = model_settings()
    selected_model = max(
        results,
        key=lambda name: results[name]["aggregate"][
            "mean_validation_accuracy"
        ],
    )

    return {
        "experiment_id": "level-2-train-v1",
        "dataset_version": contract["dataset_version"],
        "schema_version": contract["schema_version"],
        "input_fingerprints": {
            name: {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": fingerprint(path),
            }
            for name, path in paths.items()
        },
        "training_data": {
            "row_count": int(len(train)),
            "runner_count": int(train[group_column].nunique()),
            "class_counts": {
                str(int(label)): int(count)
                for label, count in train[target_column]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "level_1_preprocessing": {
            "strategy": preprocessing["strategy"],
            "fit_scope": preprocessing["fit_scope"],
            "feature_columns": preprocessing["feature_columns"],
        },
        "held_out_test_set": {
            "path": "data/v1/processed/test.csv",
            "status": "not loaded, fingerprinted, or scored in Level 2",
            "reserved_for": "Level 3 Evaluate",
        },
        "feature_columns": list(contract["feature_columns"]),
        "target_column": target_column,
        "group_column": group_column,
        "cross_validation": {
            "method": "GroupKFold",
            "n_splits": GROUP_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "fold_manifest": build_fold_manifest(train, contract, folds),
        },
        "model_settings": settings,
        "comparison_metric": "accuracy",
        "results": results,
        "selected_by_mean_validation_accuracy": selected_model,
        "selected_hyperparameters": settings[selected_model]["hyperparameters"],
        "final_fit_scope": "full Level 1 training partition only",
        "model_artifacts": artifact_paths,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "persistence_warning": "Load joblib artifacts only from trusted sources.",
        "level3_contract_path": "artifacts/v1/level3_evaluation_contract.json",
    }


def build_level3_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    settings = metadata["model_settings"]
    results = metadata["results"]
    return {
        "contract_version": "1.0.0",
        "experiment_id": metadata["experiment_id"],
        "dataset_version": metadata["dataset_version"],
        "schema_version": metadata["schema_version"],
        "input_fingerprints": metadata["input_fingerprints"],
        "held_out_test_set": metadata["held_out_test_set"],
        "feature_columns": metadata["feature_columns"],
        "target_column": metadata["target_column"],
        "group_column": metadata["group_column"],
        "cross_validation": {
            "method": metadata["cross_validation"]["method"],
            "n_splits": metadata["cross_validation"]["n_splits"],
            "shuffle": metadata["cross_validation"]["shuffle"],
            "random_state": metadata["cross_validation"]["random_state"],
        },
        "random_seeds": {
            "group_kfold": RANDOM_STATE,
            "decision_tree": RANDOM_STATE,
        },
        "selected_model": metadata[
            "selected_by_mean_validation_accuracy"
        ],
        "selected_hyperparameters": metadata["selected_hyperparameters"],
        "models": {
            name: {
                "artifact_path": metadata["model_artifacts"][name],
                "pipeline_steps": settings[name]["pipeline_steps"],
                "hyperparameters": settings[name]["hyperparameters"],
                "validation_metrics": results[name]["aggregate"],
            }
            for name in metadata["model_artifacts"]
        },
        "runtime": metadata["runtime"],
        "evaluation_rule": (
            "Load a trusted Pipeline and call predict on test.csv without "
            "calling fit; record the first held-out metrics in Level 3."
        ),
        "persistence_warning": metadata["persistence_warning"],
    }


def run_training(project_root: Path) -> dict[str, Any]:
    train, contract, preprocessing = load_training_handoff(project_root)
    folds = make_group_folds(train, contract)
    results = evaluate_candidates(
        train,
        contract,
        folds,
        build_model_candidates(),
    )
    artifact_paths = fit_and_save_models(train, contract, project_root)
    metadata = build_metadata(
        train,
        contract,
        preprocessing,
        folds,
        results,
        artifact_paths,
        project_root,
    )
    level3_contract = build_level3_contract(metadata)

    report_path = (
        project_root / "reports" / DATASET_VERSION / "training_experiment.json"
    )
    level3_path = (
        project_root
        / "artifacts"
        / DATASET_VERSION
        / "level3_evaluation_contract.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    level3_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    level3_path.write_text(
        json.dumps(level3_contract, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata = run_training(project_root)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()