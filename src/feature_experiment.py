from __future__ import annotations

import json
import sys
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.train_models import make_group_folds

SOURCE_DATASET_VERSION = "v1"
FEATURE_DATASET_VERSION = "v2"
RANDOM_STATE = 42
SELECT_K = 4
RAW_FEATURES = (
    "weekly_km_to_date",
    "sleep_hours_last_night",
    "resting_hr_morning",
    "soreness_score_evening",
    "prior_injury",
)
TARGET_COLUMN = "needs_recovery_next_day"
GROUP_COLUMN = "runner_id"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")
FORBIDDEN_HOLDOUT_USES = {
    "feature_selection",
    "threshold_tuning",
    "model_comparison",
    "model_selection",
}
TRAIN_PATH = Path("data/v1/processed/train.csv")
SCHEMA_PATH = Path("contracts/v1/schema_contract.json")
PREPROCESSING_PATH = Path("artifacts/v1/preprocessing.json")
LEVEL3_CONTRACT_PATH = Path("artifacts/v1/level3_evaluation_contract.json")
SELECTION_LOCK_PATH = Path("artifacts/v1/level3_selection_lock.json")
LEVEL4_CONTRACT_PATH = Path("artifacts/v1/level4_features_contract.json")
HOLDOUT_MARKER_PATH = Path("artifacts/v1/holdout_consumed.json")
EVALUATION_REPORT_PATH = Path("reports/v1/level3_evaluation.json")
V2_SCHEMA_PATH = Path("contracts/v2/feature_schema_contract.json")
V2_REPORT_PATH = Path("reports/v2/feature_experiment.json")
LEVEL5_HANDOFF_PATH = Path("artifacts/v2/level5_features_handoff.json")
DOC_PATH = Path("docs/level-4-features.md")

FEATURE_PROPOSALS = (
    {
        "name": "training_load_to_sleep_ratio",
        "sources": ("weekly_km_to_date", "sleep_hours_last_night"),
        "formula": "weekly_km_to_date / (sleep_hours_last_night + 1.0)",
    },
    {
        "name": "training_load_to_resting_hr_ratio",
        "sources": ("weekly_km_to_date", "resting_hr_morning"),
        "formula": "weekly_km_to_date / (resting_hr_morning + 1.0)",
    },
    {
        "name": "injury_soreness_interaction",
        "sources": ("prior_injury", "soreness_score_evening"),
        "formula": "prior_injury * soreness_score_evening",
    },
    {
        "name": "prior_week_load_ratio",
        "sources": ("weekly_km_to_date", "prior_week_km"),
        "formula": "weekly_km_to_date / (prior_week_km + 1.0)",
    },
)

def fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def audit_handoff(root: Path) -> dict[str, Any]:
    paths = {
        "train": root / TRAIN_PATH,
        "schema": root / SCHEMA_PATH,
        "preprocessing": root / PREPROCESSING_PATH,
        "level3_contract": root / LEVEL3_CONTRACT_PATH,
        "selection_lock": root / SELECTION_LOCK_PATH,
        "level4_contract": root / LEVEL4_CONTRACT_PATH,
        "holdout_marker": root / HOLDOUT_MARKER_PATH,
        "evaluation_report": root / EVALUATION_REPORT_PATH,
    }
    missing = [path.relative_to(root).as_posix() for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing immutable handoff files: {missing}")

    schema = read_json(paths["schema"])
    preprocessing = read_json(paths["preprocessing"])
    level3 = read_json(paths["level3_contract"])
    lock = read_json(paths["selection_lock"])
    level4 = read_json(paths["level4_contract"])
    marker = read_json(paths["holdout_marker"])
    expected_inputs = level3["input_fingerprints"]
    actual_inputs = {
        "train": fingerprint(paths["train"]),
        "schema": fingerprint(paths["schema"]),
        "preprocessing": fingerprint(paths["preprocessing"]),
    }
    for name, actual in actual_inputs.items():
        if expected_inputs[name]["sha256"] != actual:
            raise ValueError(f"Immutable v1 fingerprint mismatch: {name}")
    selection_lock_sha = fingerprint(paths["selection_lock"])
    if level4["locked_selection"]["selection_lock_sha256"] != selection_lock_sha:
        raise ValueError("Level 3 selection lock fingerprint mismatch")
    marker_sha = fingerprint(paths["holdout_marker"])
    if level4["holdout_policy"]["marker_sha256"] != marker_sha:
        raise ValueError("Consumed-holdout marker fingerprint mismatch")
    report_sha = fingerprint(paths["evaluation_report"])
    expected_report_sha = level4["evaluation_evidence"]["json_report"]["sha256"]
    if expected_report_sha != report_sha:
        raise ValueError("Level 3 evaluation report fingerprint mismatch")
    if not marker["holdout_consumed"] or not level4["holdout_policy"]["holdout_consumed"]:
        raise ValueError("The v1 holdout must remain marked as consumed")
    if set(marker["forbidden_future_uses"]) != FORBIDDEN_HOLDOUT_USES:
        raise ValueError("Consumed-holdout restrictions changed")
    if set(level4["holdout_policy"]["forbidden_future_uses"]) != FORBIDDEN_HOLDOUT_USES:
        raise ValueError("Level 4 holdout restrictions changed")
    if tuple(schema["feature_columns"]) != RAW_FEATURES:
        raise ValueError("The repository schema no longer matches the verified Level 4 design")
    if tuple(preprocessing["feature_columns"]) != RAW_FEATURES:
        raise ValueError("The preprocessing feature order changed")
    if level4["target_column"] != TARGET_COLUMN or level4["group_column"] != GROUP_COLUMN:
        raise ValueError("Target or group column changed")
    if lock["selected_model"] != level4["locked_selection"]["model"]:
        raise ValueError("Locked model identity changed")

    return {
        "audit_status": "passed",
        "development_only": True,
        "holdout_consumed": True,
        "holdout_read": False,
        "training_data": {"path": TRAIN_PATH.as_posix(), "sha256": actual_inputs["train"]},
        "schema": {"path": SCHEMA_PATH.as_posix(), "sha256": actual_inputs["schema"]},
        "preprocessing": {
            "path": PREPROCESSING_PATH.as_posix(),
            "sha256": actual_inputs["preprocessing"],
            "inherited_fit_scope": preprocessing["fit_scope"],
        },
        "selection_lock": {"path": SELECTION_LOCK_PATH.as_posix(), "sha256": selection_lock_sha},
        "holdout_marker": {"path": HOLDOUT_MARKER_PATH.as_posix(), "sha256": marker_sha},
        "evaluation_report": {"path": EVALUATION_REPORT_PATH.as_posix(), "sha256": report_sha},
        "forbidden_holdout_uses": sorted(FORBIDDEN_HOLDOUT_USES),
    }


def prediction_time_features(schema: dict[str, Any]) -> set[str]:
    return {
        column["name"]
        for column in schema["columns"]
        if column["role"] == "feature" and column["available_at_prediction"] is True
    }


def resolve_feature_plan(schema: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    allowed = prediction_time_features(schema)
    supported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for proposal in FEATURE_PROPOSALS:
        missing_sources = sorted(set(proposal["sources"]) - allowed)
        record = {
            "name": proposal["name"],
            "sources": list(proposal["sources"]),
            "formula": proposal["formula"],
        }
        if missing_sources:
            record["reason"] = "unsupported source columns"
            record["missing_sources"] = missing_sources
            skipped.append(record)
        else:
            record["status"] = "supported by prediction-time schema"
            supported.append(record)
    return {"supported": supported, "skipped": skipped}


def validate_model_inputs(candidate_columns: list[str], schema: dict[str, Any]) -> None:
    allowed = prediction_time_features(schema)
    rejected = sorted(set(candidate_columns) - allowed)
    if rejected:
        raise ValueError(f"Rejected non-prediction-time model inputs: {rejected}")


class SchemaFeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, *, mode: str, engineered_features: tuple[str, ...] = ()) -> None:
        self.mode = mode
        self.engineered_features = engineered_features

    def fit(self, X: pd.DataFrame, y: Any = None) -> SchemaFeatureTransformer:
        if self.mode not in {"raw", "engineered"}:
            raise ValueError(f"Unknown feature mode: {self.mode}")
        missing = sorted(set(RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Missing raw feature columns: {missing}")
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Missing raw feature columns: {missing}")
        output = X[list(RAW_FEATURES)].copy()
        if self.mode == "raw":
            return output
        for name in self.engineered_features:
            if name == "training_load_to_sleep_ratio":
                output[name] = output["weekly_km_to_date"] / (
                    output["sleep_hours_last_night"] + 1.0
                )
            elif name == "training_load_to_resting_hr_ratio":
                output[name] = output["weekly_km_to_date"] / (
                    output["resting_hr_morning"] + 1.0
                )
            elif name == "injury_soreness_interaction":
                output[name] = (
                    output["prior_injury"] * output["soreness_score_evening"]
                )
            else:
                raise ValueError(f"No verified transformation implementation: {name}")
        return output


def feature_names(mode: str, engineered_features: tuple[str, ...]) -> list[str]:
    names = list(RAW_FEATURES)
    if mode == "engineered":
        names.extend(engineered_features)
    return names


def build_pipeline(mode: str, engineered_features: tuple[str, ...]) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                SchemaFeatureTransformer(
                    mode=mode,
                    engineered_features=engineered_features,
                ),
            ),
            ("imputer", SimpleImputer(strategy="mean")),
            ("selector", SelectKBest(score_func=f_classif, k=SELECT_K)),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )


def metric_values(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0.0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0.0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0.0)), 6),
    }


def fold_signature(fit_indices: Any, validation_indices: Any) -> str:
    payload = {
        "fit": [int(value) for value in fit_indices.tolist()],
        "validation": [int(value) for value in validation_indices.tolist()],
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def aggregate_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    mean_training = {
        metric: round(fmean(fold["training_metrics"][metric] for fold in folds), 6)
        for metric in METRIC_NAMES
    }
    mean_validation = {
        metric: round(fmean(fold["validation_metrics"][metric] for fold in folds), 6)
        for metric in METRIC_NAMES
    }
    validation_std = {
        metric: round(pstdev(fold["validation_metrics"][metric] for fold in folds), 6)
        for metric in METRIC_NAMES
    }
    gaps = {
        metric: round(mean_training[metric] - mean_validation[metric], 6)
        for metric in METRIC_NAMES
    }
    return {
        "mean_training_metrics": mean_training,
        "mean_validation_metrics": mean_validation,
        "validation_metric_std": validation_std,
        "generalization_gaps": gaps,
    }


def feature_stability(folds: list[dict[str, Any]]) -> dict[str, Any]:
    selected_sets = [set(fold["selected_features"]) for fold in folds]
    counts = Counter(feature for selected in selected_sets for feature in selected)
    pairs = list(combinations(selected_sets, 2))
    jaccards = [
        len(left.intersection(right)) / len(left.union(right))
        for left, right in pairs
    ]
    return {
        "selection_frequency": {
            name: round(count / len(folds), 6) for name, count in sorted(counts.items())
        },
        "mean_pairwise_jaccard": round(fmean(jaccards), 6) if jaccards else 1.0,
    }


def evaluate_variant(
    train: pd.DataFrame,
    folds: list[tuple[Any, Any]],
    mode: str,
    engineered_features: tuple[str, ...],
) -> dict[str, Any]:
    pipeline = build_pipeline(mode, engineered_features)
    candidate_names = feature_names(mode, engineered_features)
    fold_results: list[dict[str, Any]] = []
    for number, (fit_indices, validation_indices) in enumerate(folds, start=1):
        fit = train.iloc[fit_indices]
        validation = train.iloc[validation_indices]
        fit_groups = {str(value) for value in fit[GROUP_COLUMN].unique()}
        validation_groups = {str(value) for value in validation[GROUP_COLUMN].unique()}
        overlap = sorted(fit_groups.intersection(validation_groups))
        if overlap:
            raise ValueError(f"Runner overlap in fold {number}: {overlap}")
        model = clone(pipeline)
        model.fit(fit[list(RAW_FEATURES)], fit[TARGET_COLUMN])
        training_predictions = model.predict(fit[list(RAW_FEATURES)])
        validation_predictions = model.predict(validation[list(RAW_FEATURES)])
        selected_indices = model.named_steps["selector"].get_support(indices=True)
        selected = [candidate_names[int(index)] for index in selected_indices.tolist()]
        fold_results.append(
            {
                "fold": number,
                "fold_signature": fold_signature(fit_indices, validation_indices),
                "fit_runner_count": len(fit_groups),
                "validation_runner_count": len(validation_groups),
                "runner_overlap": overlap,
                "training_metrics": metric_values(fit[TARGET_COLUMN], training_predictions),
                "validation_metrics": metric_values(
                    validation[TARGET_COLUMN], validation_predictions
                ),
                "selected_features": selected,
                "imputer_statistics": [
                    round(float(value), 6)
                    for value in model.named_steps["imputer"].statistics_.tolist()
                ],
            }
        )
    return {
        "variant": mode,
        "candidate_features": candidate_names,
        "pipeline_steps": ["features", "imputer", "selector", "classifier"],
        "folds": fold_results,
        "aggregate": aggregate_folds(fold_results),
        "feature_stability": feature_stability(fold_results),
    }


def build_v2_schema(plan: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    engineered = [record["name"] for record in plan["supported"]]
    return {
        "schema_version": "2.0.0",
        "dataset_version": FEATURE_DATASET_VERSION,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "prediction_timestamp": "20:00 local time on observation day D",
        "raw_feature_columns": list(RAW_FEATURES),
        "engineered_feature_columns": engineered,
        "candidate_feature_columns": [*RAW_FEATURES, *engineered],
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "supported_transformations": plan["supported"],
        "skipped_transformations": plan["skipped"],
        "feature_policy": "Only row-local information marked available at prediction time may enter a transformation.",
        "scientific_status": "Training-load ratios are experimental synthetic predictors, not validated coaching rules.",
        "materialized_dataset_created": False,
    }


def run_experiment(root: Path) -> dict[str, Any]:
    for relative in (V2_SCHEMA_PATH, V2_REPORT_PATH, LEVEL5_HANDOFF_PATH):
        if (root / relative).exists():
            raise FileExistsError(f"Refusing to overwrite Level 4 evidence: {relative}")
    audit = audit_handoff(root)
    schema = read_json(root / SCHEMA_PATH)
    plan = resolve_feature_plan(schema)
    engineered_features = tuple(record["name"] for record in plan["supported"])
    train = pd.read_csv(root / TRAIN_PATH)
    required = {*RAW_FEATURES, TARGET_COLUMN, GROUP_COLUMN}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"Development data is missing columns: {missing}")

    folds = make_group_folds(train, schema)
    raw = evaluate_variant(train, folds, "raw", engineered_features)
    engineered = evaluate_variant(train, folds, "engineered", engineered_features)
    raw_signatures = [fold["fold_signature"] for fold in raw["folds"]]
    engineered_signatures = [fold["fold_signature"] for fold in engineered["folds"]]
    if raw_signatures != engineered_signatures:
        raise ValueError("Raw and engineered variants did not use identical folds")

    results = {"raw": raw, "engineered": engineered}
    selected_variant = max(
        results,
        key=lambda name: (
            results[name]["aggregate"]["mean_validation_metrics"]["accuracy"],
            -abs(results[name]["aggregate"]["generalization_gaps"]["accuracy"]),
            1 if name == "raw" else 0,
        ),
    )
    deltas = {
        metric: round(
            engineered["aggregate"]["mean_validation_metrics"][metric]
            - raw["aggregate"]["mean_validation_metrics"][metric],
            6,
        )
        for metric in METRIC_NAMES
    }

    v2_schema = build_v2_schema(plan)
    write_json_exclusive(root / V2_SCHEMA_PATH, v2_schema)
    report = {
        "report_version": "2.0.0",
        "experiment_id": "level-4-features-v2",
        "development_only": True,
        "consumed_v1_holdout_read": False,
        "audit": audit,
        "feature_plan": plan,
        "cross_validation": {
            "method": "GroupKFold",
            "n_splits": len(folds),
            "identical_fold_signatures": raw_signatures,
            "selection_data": "v1 development rows only",
        },
        "model_control": {
            "estimator": "DecisionTreeClassifier",
            "random_state": RANDOM_STATE,
            "select_k": SELECT_K,
            "selection_score": "f_classif",
            "fold_local_steps": ["imputer", "selector", "classifier"],
        },
        "results": results,
        "engineered_minus_raw_validation_deltas": deltas,
        "selected_variant": selected_variant,
        "selection_basis": "highest mean grouped validation accuracy; smaller absolute accuracy gap; raw on an exact tie",
        "v2_schema": {
            "path": V2_SCHEMA_PATH.as_posix(),
            "sha256": fingerprint(root / V2_SCHEMA_PATH),
        },
        "limitations": [
            "The v1 processed training data is an immutable inherited input whose Level 1 mean imputation was fitted on training runners before Level 2 cross-validation.",
            "Every new Level 4 imputer and selector is fitted inside each fold, but Level 4 does not reconstruct pre-imputation v1 rows.",
            "The data, target, and engineered ratios are synthetic educational constructs and are not medical or coaching evidence.",
            "No consumed v1 holdout label participates in this comparison or selection.",
        ],
    }
    write_json_exclusive(root / V2_REPORT_PATH, report)
    handoff = {
        "handoff_version": "2.0.0",
        "next_level": "Level 5",
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "feature_dataset_version": FEATURE_DATASET_VERSION,
        "development_data": audit["training_data"],
        "feature_schema": {
            "path": V2_SCHEMA_PATH.as_posix(),
            "sha256": fingerprint(root / V2_SCHEMA_PATH),
        },
        "feature_experiment": {
            "path": V2_REPORT_PATH.as_posix(),
            "sha256": fingerprint(root / V2_REPORT_PATH),
        },
        "selected_variant": selected_variant,
        "selection_basis": report["selection_basis"],
        "selected_candidate_features": results[selected_variant]["candidate_features"],
        "holdout_policy": {
            "v1_holdout_consumed": True,
            "marker": audit["holdout_marker"],
            "forbidden_future_uses": sorted(FORBIDDEN_HOLDOUT_USES),
            "required_level5_evidence": "grouped development validation or a new untouched holdout",
        },
        "scientific_status": "Synthetic educational feature experiment, not a validated coaching system.",
    }
    write_json_exclusive(root / LEVEL5_HANDOFF_PATH, handoff)
    return report


def markdown_document(report: dict[str, Any], handoff: dict[str, Any]) -> str:
    raw = report["results"]["raw"]["aggregate"]
    engineered = report["results"]["engineered"]["aggregate"]
    rows = []
    for name, aggregate in (("raw", raw), ("engineered", engineered)):
        validation = aggregate["mean_validation_metrics"]
        gaps = aggregate["generalization_gaps"]
        rows.append(
            f"| {name} | {validation['accuracy']:.3f} | {validation['precision']:.3f} | "
            f"{validation['recall']:.3f} | {validation['f1']:.3f} | {gaps['accuracy']:.3f} |"
        )
    table = "\n".join(rows)
    skipped = ", ".join(
        record["name"] for record in report["feature_plan"]["skipped"]
    )
    return f"""# Level 4: Features

This experiment compares raw and schema-gated engineered features using identical grouped development folds. The consumed v1 holdout was not read.

| Variant | Validation accuracy | Precision 1 | Recall 1 | F1 1 | Accuracy gap |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

Selected variant: `{report['selected_variant']}`

Mean pairwise feature stability:

- Raw: `{report['results']['raw']['feature_stability']['mean_pairwise_jaccard']}`
- Engineered: `{report['results']['engineered']['feature_stability']['mean_pairwise_jaccard']}`

Skipped unsupported transformations: `{skipped}`

The training-load ratios are experimental synthetic predictors. They are not validated coaching rules. The inherited v1 processed input remains subject to its documented Level 1 preprocessing history.

Level 5 handoff: `{LEVEL5_HANDOFF_PATH.as_posix()}`
Selected Level 5 variant: `{handoff['selected_variant']}`
"""


def update_readme(root: Path, document: str) -> Path:
    readme_path = root / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    start = "<!-- LEVEL-4-FEATURES:START -->"
    end = "<!-- LEVEL-4-FEATURES:END -->"
    section = f"{start}\n## Level 4: Features\n\n[Feature experiment](docs/level-4-features.md)\n\n{end}"
    if start in existing:
        if end not in existing:
            raise ValueError("Incomplete Level 4 README marker block")
        before = existing.split(start, 1)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{section}\n" + (f"\n{after.rstrip()}\n" if after else "")
    else:
        prefix = existing.rstrip()
        updated = f"{prefix}\n\n{section}\n" if prefix else f"{section}\n"
    readme_path.write_text(updated, encoding="utf-8")
    return readme_path


def document_experiment(root: Path) -> list[str]:
    report = read_json(root / V2_REPORT_PATH)
    handoff = read_json(root / LEVEL5_HANDOFF_PATH)
    document = markdown_document(report, handoff)
    doc_path = root / DOC_PATH
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(document, encoding="utf-8")
    readme = update_readme(root, document)
    return [doc_path.relative_to(root).as_posix(), readme.relative_to(root).as_posix()]


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"audit", "plan", "run", "document"}:
        raise SystemExit("Usage: python -m src.feature_experiment [audit|plan|run|document]")
    root = Path(__file__).resolve().parents[1]
    if sys.argv[1] == "audit":
        print(json.dumps(audit_handoff(root), indent=2, sort_keys=True))
    elif sys.argv[1] == "plan":
        audit_handoff(root)
        print(
            json.dumps(
                resolve_feature_plan(read_json(root / SCHEMA_PATH)),
                indent=2,
                sort_keys=True,
            )
        )
    elif sys.argv[1] == "run":
        report = run_experiment(root)
        print(
            json.dumps(
                {
                    "selected_variant": report["selected_variant"],
                    "validation_deltas": report[
                        "engineered_minus_raw_validation_deltas"
                    ],
                    "report": V2_REPORT_PATH.as_posix(),
                    "level5_handoff": LEVEL5_HANDOFF_PATH.as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(document_experiment(root), indent=2))


if __name__ == "__main__":
    main()
