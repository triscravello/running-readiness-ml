from __future__ import annotations

import ast
import inspect
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_VERSION = "v1"
BASELINE_NAME = "majority_baseline"
LOCK_RELATIVE_PATH = Path("artifacts/v1/level3_selection_lock.json")
REPORT_RELATIVE_PATH = Path("reports/v1/level3_evaluation.json")
MARKDOWN_RELATIVE_PATH = Path("reports/v1/level3_evaluation.md")
ERRORS_RELATIVE_PATH = Path("reports/v1/level3_errors.csv")
HOLDOUT_MARKER_RELATIVE_PATH = Path("artifacts/v1/holdout_consumed.json")
LEVEL4_RELATIVE_PATH = Path("artifacts/v1/level4_features_contract.json")

FORBIDDEN_HOLDOUT_USES = [
    "feature_selection",
    "threshold_tuning",
    "model_comparison",
    "model_selection",
]
LIMITATIONS = [
    "The runner records, readiness rule, and labels are synthetic educational constructs, not evidence about real physiology.",
    "One finite v1 holdout does not prove performance across new populations, devices, seasons, or training plans.",
    "Precision and recall depend on class prevalence and the operating threshold.",
    "False positives and false negatives have no validated clinical or coaching cost in this synthetic task.",
    "This evaluation supports an educational software-engineering claim, not a medical, safety, or causal claim.",
]


def fingerprint(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def level2_paths(root: Path) -> dict[str, Path]:
    return {
        "contract": root / "artifacts/v1/level3_evaluation_contract.json",
        "training_report": root / "reports/v1/training_experiment.json",
    }


def verify_level2_handoff(root: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    paths = level2_paths(root)
    missing = [path.as_posix() for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Level 2 handoff files: {missing}")
    contract = read_json(paths["contract"])
    metadata = read_json(paths["training_report"])
    if contract["dataset_version"] != DATASET_VERSION:
        raise ValueError("Expected dataset v1")
    if contract["experiment_id"] != metadata["experiment_id"]:
        raise ValueError("Experiment identifiers do not match")
    for name, evidence in contract["input_fingerprints"].items():
        path = root / evidence["path"]
        if not path.exists() or fingerprint(path) != evidence["sha256"]:
            raise ValueError(f"Fingerprint mismatch for Level 2 input: {name}")
    train = pd.read_csv(root / contract["input_fingerprints"]["train"]["path"])
    group = contract["group_column"]
    manifest = metadata["cross_validation"]["fold_manifest"]
    if len(manifest) != metadata["cross_validation"]["n_splits"]:
        raise ValueError("Fold count mismatch")
    if any(fold["runner_overlap"] for fold in manifest):
        raise ValueError("Runner overlap exists in Level 2 folds")
    validation_groups = [str(value) for fold in manifest for value in fold["validation_runners"]]
    if len(validation_groups) != len(set(validation_groups)):
        raise ValueError("A runner appears in multiple validation folds")
    if set(validation_groups) != {str(value) for value in train[group].unique()}:
        raise ValueError("Validation folds do not cover all training runners")
    results = metadata["results"]
    winner = max(results, key=lambda name: results[name]["aggregate"]["mean_validation_accuracy"])
    if metadata["selected_by_mean_validation_accuracy"] != winner:
        raise ValueError("Training report selection is not the validation winner")
    if contract["selected_model"] != winner:
        raise ValueError("Contract selection differs from validation evidence")
    if BASELINE_NAME not in contract["models"]:
        raise ValueError("Majority baseline is missing")
    for name, spec in contract["models"].items():
        if not (root / spec["artifact_path"]).exists():
            raise FileNotFoundError(f"Missing model artifact: {name}")
    return contract, metadata, train


def paths_as_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def lock_selection(root: Path) -> dict[str, Any]:
    path = root / LOCK_RELATIVE_PATH
    if path.exists():
        raise FileExistsError("The selection lock cannot be rewritten")
    contract, metadata, _ = verify_level2_handoff(root)
    selected = contract["selected_model"]
    names = [selected] if selected == BASELINE_NAME else [selected, BASELINE_NAME]
    paths = level2_paths(root)
    payload = {
        "lock_version": "1.0.0",
        "dataset_version": DATASET_VERSION,
        "experiment_id": contract["experiment_id"],
        "selection_basis": "Level 2 mean_validation_accuracy only",
        "selection_metric": "mean_validation_accuracy",
        "selected_model": selected,
        "selected_validation_metrics": contract["models"][selected]["validation_metrics"],
        "majority_baseline_validation_metrics": contract["models"][BASELINE_NAME]["validation_metrics"],
        "level2_contract": {"path": paths_as_relative(paths["contract"], root), "sha256": fingerprint(paths["contract"])},
        "training_report": {"path": paths_as_relative(paths["training_report"], root), "sha256": fingerprint(paths["training_report"])},
        "input_fingerprints": contract["input_fingerprints"],
        "model_artifacts": {name: {"path": contract["models"][name]["artifact_path"], "sha256": fingerprint(root / contract["models"][name]["artifact_path"])} for name in names},
        "held_out_test_status": "not read during selection lock",
        "test_metrics_available": False,
        "selection_is_final": True,
        "level2_selected_field": metadata["selected_by_mean_validation_accuracy"],
    }
    write_json_exclusive(path, payload)
    return payload


def verify_selection_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    contract, metadata, train = verify_level2_handoff(root)
    path = root / LOCK_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError("Create the Level 3 selection lock first")
    lock = read_json(path)
    if not lock["selection_is_final"] or lock["selected_model"] != contract["selected_model"]:
        raise ValueError("Locked selection is invalid")
    paths = level2_paths(root)
    if lock["level2_contract"]["sha256"] != fingerprint(paths["contract"]):
        raise ValueError("Level 3 contract changed after locking")
    if lock["training_report"]["sha256"] != fingerprint(paths["training_report"]):
        raise ValueError("Training report changed after locking")
    for name, evidence in lock["model_artifacts"].items():
        if evidence["sha256"] != fingerprint(root / evidence["path"]):
            raise ValueError(f"Model changed after locking: {name}")
    return contract, metadata, train, lock


def metric_bundle(y_true: Any, y_pred: Any) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["class_0", "class_1"], output_dict=True, zero_division=0.0)
    tn, fp, fn, tp = [int(value) for value in matrix.ravel().tolist()]
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "positive_class": {"precision": float(report["class_1"]["precision"]), "recall": float(report["class_1"]["recall"]), "f1": float(report["class_1"]["f1-score"]), "support": int(report["class_1"]["support"])}, "class_support": {"0": int(report["class_0"]["support"]), "1": int(report["class_1"]["support"])}, "confusion_matrix": {"labels": [0, 1], "layout": "[[TN, FP], [FN, TP]]", "matrix": [[int(value) for value in row] for row in matrix.tolist()], "true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp}}


def build_error_rows(test: pd.DataFrame, predictions: list[int], features: list[str], target: str, group: str) -> pd.DataFrame:
    rows = test.reset_index(drop=True).copy()
    rows.insert(0, "row_id", list(range(len(rows))))
    rows["actual"] = rows[target].astype(int)
    rows["predicted"] = predictions
    rows["error_type"] = ["false_positive" if actual == 0 and predicted == 1 else "false_negative" if actual == 1 and predicted == 0 else "correct" for actual, predicted in zip(rows["actual"], rows["predicted"])]
    return rows[rows["error_type"] != "correct"][["row_id", group, *features, "actual", "predicted", "error_type"]].copy()


def markdown_report(report: dict[str, Any]) -> str:
    selected = report["models"]["selected"]
    baseline = report["models"]["majority_baseline"]
    sm, bm = selected["metrics"], baseline["metrics"]
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Level 3 Evaluate: Running-Readiness Classifier

Evaluation timestamp, UTC: `{report['evaluated_at_utc']}`

The model `{selected['name']}` was selected from Level 2 validation before `test.csv` was read. The held-out operating point is the frozen Pipeline `predict` rule. It was not tuned on test labels.

## Held-out metrics

| Model | Accuracy | Precision 1 | Recall 1 | F1 1 | Support 0 | Support 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `{selected['name']}` | {sm['accuracy']:.3f} | {sm['positive_class']['precision']:.3f} | {sm['positive_class']['recall']:.3f} | {sm['positive_class']['f1']:.3f} | {sm['class_support']['0']} | {sm['class_support']['1']} |
| `{baseline['name']}` | {bm['accuracy']:.3f} | {bm['positive_class']['precision']:.3f} | {bm['positive_class']['recall']:.3f} | {bm['positive_class']['f1']:.3f} | {bm['class_support']['0']} | {bm['class_support']['1']} |

Selected `[[TN, FP], [FN, TP]]`: `{sm['confusion_matrix']['matrix']}`

Baseline `[[TN, FP], [FN, TP]]`: `{bm['confusion_matrix']['matrix']}`

Undefined precision, recall, and F1 values use `zero_division=0.0`. A false positive flags recovery when the synthetic label is 0. A false negative clears a runner when it is 1. Accuracy alone can reward a majority predictor that misses the minority class.

## Reproducibility

- Test SHA-256: `{report['test_data']['sha256']}`
- Selected model SHA-256: `{selected['sha256']}`
- Selection lock SHA-256: `{report['selection_lock']['sha256']}`
- Consumed marker: `{report['holdout_consumed_marker']['path']}`
- Train-test runner overlap: `{report['group_safety']['runner_overlap_count']}`

## Limitations

{limitations}
"""


def run_evaluation(root: Path) -> dict[str, Any]:
    outputs = [REPORT_RELATIVE_PATH, MARKDOWN_RELATIVE_PATH, ERRORS_RELATIVE_PATH, HOLDOUT_MARKER_RELATIVE_PATH, LEVEL4_RELATIVE_PATH]
    existing = [(root / path).as_posix() for path in outputs if (root / path).exists()]
    if existing:
        raise FileExistsError(f"Refusing repeated held-out scoring: {existing}")
    contract, metadata, train, lock = verify_selection_lock(root)
    selected_name = lock["selected_model"]
    features, target, group = list(contract["feature_columns"]), contract["target_column"], contract["group_column"]
    test_path = root / contract["held_out_test_set"]["path"]
    test_sha256 = fingerprint(test_path)
    test = pd.read_csv(test_path)
    missing = sorted(set([*features, target, group]) - set(test.columns))
    if missing:
        raise ValueError(f"Held-out test is missing columns: {missing}")
    labels = {int(value) for value in test[target].unique()}
    if not labels.issubset({0, 1}):
        raise ValueError(f"Expected binary labels, found {labels}")
    overlap = sorted({str(value) for value in train[group].unique()}.intersection({str(value) for value in test[group].unique()}))
    if overlap:
        raise ValueError(f"Train and test runners overlap: {overlap}")
    selected_spec, baseline_spec = contract["models"][selected_name], contract["models"][BASELINE_NAME]
    selected_model = joblib.load(root / selected_spec["artifact_path"])
    baseline_model = joblib.load(root / baseline_spec["artifact_path"])
    selected_predictions = [int(value) for value in selected_model.predict(test[features])]
    baseline_predictions = [int(value) for value in baseline_model.predict(test[features])]
    y_true = test[target].astype(int)
    selected_metrics, baseline_metrics = metric_bundle(y_true, selected_predictions), metric_bundle(y_true, baseline_predictions)
    errors = build_error_rows(test, selected_predictions, features, target, group)
    evaluated_at_utc = datetime.now(timezone.utc).isoformat()
    report = {"report_version": "1.1.0", "experiment_id": "level-3-evaluate-v1", "evaluated_at_utc": evaluated_at_utc, "dataset_version": DATASET_VERSION, "holdout_consumed": True, "evaluation_status": "first and only planned held-out evaluation completed", "selection_basis": lock["selection_basis"], "selection_lock": {"path": LOCK_RELATIVE_PATH.as_posix(), "sha256": fingerprint(root / LOCK_RELATIVE_PATH)}, "operating_point": {"held_out_prediction_method": "Pipeline.predict", "held_out_threshold_role": "frozen default operating rule, not tuned on test labels", "exploratory_threshold_analysis_is_separate": True}, "metric_policy": {"labels": [0, 1], "positive_class": 1, "zero_division": 0.0, "confusion_matrix_layout": "[[TN, FP], [FN, TP]]"}, "test_data": {"path": paths_as_relative(test_path, root), "sha256": test_sha256, "row_count": int(len(test)), "runner_count": int(test[group].nunique()), "class_support": selected_metrics["class_support"]}, "group_safety": {"runner_overlap": overlap, "runner_overlap_count": len(overlap)}, "models": {"selected": {"name": selected_name, "path": selected_spec["artifact_path"], "sha256": fingerprint(root / selected_spec["artifact_path"]), "fit_calls_during_evaluation": 0, "metrics": selected_metrics}, "majority_baseline": {"name": BASELINE_NAME, "path": baseline_spec["artifact_path"], "sha256": fingerprint(root / baseline_spec["artifact_path"]), "fit_calls_during_evaluation": 0, "metrics": baseline_metrics}}, "comparison_to_majority_baseline": {"accuracy_delta": selected_metrics["accuracy"] - baseline_metrics["accuracy"], "positive_precision_delta": selected_metrics["positive_class"]["precision"] - baseline_metrics["positive_class"]["precision"], "positive_recall_delta": selected_metrics["positive_class"]["recall"] - baseline_metrics["positive_class"]["recall"], "positive_f1_delta": selected_metrics["positive_class"]["f1"] - baseline_metrics["positive_class"]["f1"]}, "error_analysis": {"path": ERRORS_RELATIVE_PATH.as_posix(), "row_count": int(len(errors)), "false_positive_count": selected_metrics["confusion_matrix"]["false_positives"], "false_negative_count": selected_metrics["confusion_matrix"]["false_negatives"]}, "holdout_consumed_marker": {"path": HOLDOUT_MARKER_RELATIVE_PATH.as_posix()}, "limitations": LIMITATIONS, "level4_contract_path": LEVEL4_RELATIVE_PATH.as_posix(), "level2_validation_results_preserved": metadata["results"]}
    errors_path = root / ERRORS_RELATIVE_PATH
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(errors_path, index=False, mode="x")
    write_json_exclusive(root / REPORT_RELATIVE_PATH, report)
    markdown_path = root / MARKDOWN_RELATIVE_PATH
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with markdown_path.open("x", encoding="utf-8") as handle:
        handle.write(markdown_report(report))
    consumed_marker = {"marker_version": "1.0.0", "holdout_consumed": True, "consumed_at_utc": evaluated_at_utc, "consumption_reason": "Level 3 first and only planned final evaluation", "locked_selection": {"model": selected_name, "selection_basis": lock["selection_basis"], "selection_lock_path": LOCK_RELATIVE_PATH.as_posix(), "selection_lock_sha256": fingerprint(root / LOCK_RELATIVE_PATH)}, "dataset": {"path": paths_as_relative(test_path, root), "sha256": test_sha256}, "selected_model": {"path": selected_spec["artifact_path"], "sha256": fingerprint(root / selected_spec["artifact_path"])}, "evaluation_report": {"path": REPORT_RELATIVE_PATH.as_posix(), "sha256": fingerprint(root / REPORT_RELATIVE_PATH)}, "forbidden_future_uses": FORBIDDEN_HOLDOUT_USES, "allowed_future_uses": ["historical_reporting", "audit"]}
    write_json_exclusive(root / HOLDOUT_MARKER_RELATIVE_PATH, consumed_marker)
    level4_contract = {"handoff_version": "1.1.0", "next_level": "Level 4: Features", "source_dataset_version": DATASET_VERSION, "recommended_next_dataset_version": "v2", "evaluated_at_utc": evaluated_at_utc, "feature_columns": features, "target_column": target, "group_column": group, "locked_selection": consumed_marker["locked_selection"], "selected_model": consumed_marker["selected_model"], "holdout_policy": {"holdout_consumed": True, "marker_path": HOLDOUT_MARKER_RELATIVE_PATH.as_posix(), "marker_sha256": fingerprint(root / HOLDOUT_MARKER_RELATIVE_PATH), "forbidden_future_uses": FORBIDDEN_HOLDOUT_USES, "required_level4_selection_data": "grouped validation predictions from training data only"}, "evaluation_evidence": {"json_report": {"path": REPORT_RELATIVE_PATH.as_posix(), "sha256": fingerprint(root / REPORT_RELATIVE_PATH)}, "markdown_report": {"path": MARKDOWN_RELATIVE_PATH.as_posix(), "sha256": fingerprint(root / MARKDOWN_RELATIVE_PATH)}, "error_analysis": {"path": ERRORS_RELATIVE_PATH.as_posix(), "sha256": fingerprint(root / ERRORS_RELATIVE_PATH)}}, "selected_test_metrics": selected_metrics, "majority_baseline_test_metrics": baseline_metrics, "limitations": LIMITATIONS}
    write_json_exclusive(root / LEVEL4_RELATIVE_PATH, level4_contract)
    return report


def verify_no_fit_calls_in_this_module() -> list[int]:
    tree = ast.parse(inspect.getsource(sys.modules[__name__]))
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform"}]


def readme_section(report: dict[str, Any]) -> str:
    selected, baseline = report["models"]["selected"], report["models"]["majority_baseline"]
    sm, bm = selected["metrics"], baseline["metrics"]
    return f"""<!-- LEVEL-3-EVALUATE:START -->
## Level 3: Evaluate

Evaluated at `{report['evaluated_at_utc']}` after `{selected['name']}` was locked from grouped validation evidence.

| Model | Accuracy | Precision 1 | Recall 1 | F1 1 |
| --- | ---: | ---: | ---: | ---: |
| `{selected['name']}` | {sm['accuracy']:.3f} | {sm['positive_class']['precision']:.3f} | {sm['positive_class']['recall']:.3f} | {sm['positive_class']['f1']:.3f} |
| `{baseline['name']}` | {bm['accuracy']:.3f} | {bm['positive_class']['precision']:.3f} | {bm['positive_class']['recall']:.3f} | {bm['positive_class']['f1']:.3f} |

Selected confusion matrix `[[TN, FP], [FN, TP]]`: `{sm['confusion_matrix']['matrix']}`

- [Evaluation report](reports/v1/level3_evaluation.md)
- [Machine-readable metrics](reports/v1/level3_evaluation.json)
- [Error analysis](reports/v1/level3_errors.csv)
- [Consumed-holdout marker](artifacts/v1/holdout_consumed.json)
- [Level 4 handoff](artifacts/v1/level4_features_contract.json)

Reproduce safeguards with `python -m unittest -v tests/test_data_pipeline.py tests/test_training.py tests/test_evaluation.py`. The holdout command is intentionally excluded because v1 test labels are consumed.
<!-- LEVEL-3-EVALUATE:END -->"""


def update_readme(root: Path) -> Path:
    report_path = root / REPORT_RELATIVE_PATH
    if not report_path.exists():
        raise FileNotFoundError("Run evaluation before documentation")
    readme = root / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
    start, end = "<!-- LEVEL-3-EVALUATE:START -->", "<!-- LEVEL-3-EVALUATE:END -->"
    section = readme_section(read_json(report_path))
    if start in existing:
        if end not in existing:
            raise ValueError("Incomplete Level 3 README marker block")
        before, after = existing.split(start, 1)[0].rstrip(), existing.split(end, 1)[1].lstrip()
        updated = f"{before}\n\n{section}\n" + (f"\n{after.rstrip()}\n" if after else "")
    else:
        prefix = existing.rstrip()
        updated = f"{prefix}\n\n{section}\n" if prefix else f"{section}\n"
    readme.write_text(updated, encoding="utf-8")
    return readme


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"lock", "evaluate", "document"}:
        raise SystemExit("Usage: python -m src.evaluate_models [lock|evaluate|document]")
    root = Path(__file__).resolve().parents[1]
    if sys.argv[1] == "lock":
        print(json.dumps(lock_selection(root), indent=2))
    elif sys.argv[1] == "evaluate":
        report = run_evaluation(root)
        print(json.dumps({"evaluated_at_utc": report["evaluated_at_utc"], "holdout_consumed": report["holdout_consumed"], "selected": report["models"]["selected"], "majority_baseline": report["models"]["majority_baseline"], "comparison": report["comparison_to_majority_baseline"]}, indent=2))
    else:
        print(update_readme(root).relative_to(root).as_posix())


if __name__ == "__main__":
    main()
