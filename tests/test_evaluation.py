import ast
import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

import src.evaluate_models as evaluate_models
from src.evaluate_models import ERRORS_RELATIVE_PATH, HOLDOUT_MARKER_RELATIVE_PATH, LEVEL4_RELATIVE_PATH, LOCK_RELATIVE_PATH, REPORT_RELATIVE_PATH, fingerprint, lock_selection, run_evaluation

class FixtureModel:
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def predict(self, features: pd.DataFrame) -> list[int]:
        if self.strategy == "majority":
            return [0] * len(features)
        return [int(value >= 0.5) for value in features["signal"]]


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build_fixture()

    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build_fixture(self) -> None:
        train_path = self.root / "data/v1/processed/train.csv"
        test_path = self.root / "data/v1/processed/test.csv"
        schema_path = self.root / "contracts/v1/schema_contract.json"
        preprocessing_path = self.root / "artifacts/v1/preprocessing.json"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train = pd.DataFrame({"signal": [0.1, 0.9, 0.2, 0.8], "needs_recovery_next_day": [0, 1, 0, 1], "runner_id": ["runner-a", "runner-b", "runner-c", "runner-d"]})
        test = pd.DataFrame({"signal": [0.2, 0.8, 0.7, 0.3], "needs_recovery_next_day": [0, 1, 0, 1], "runner_id": ["runner-e", "runner-e", "runner-f", "runner-f"]})
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        self.write_json(schema_path, {"dataset_version": "v1", "feature_columns": ["signal"], "target_column": "needs_recovery_next_day", "group_column": "runner_id"})
        self.write_json(preprocessing_path, {"dataset_version": "v1", "feature_columns": ["signal"], "fit_scope": "training only"})
        models = self.root / "models/v1"
        models.mkdir(parents=True, exist_ok=True)
        joblib.dump(FixtureModel("learned"), models / "logistic_regression.joblib")
        joblib.dump(FixtureModel("majority"), models / "majority_baseline.joblib")
        fingerprints = {"train": {"path": "data/v1/processed/train.csv", "sha256": fingerprint(train_path)}, "schema": {"path": "contracts/v1/schema_contract.json", "sha256": fingerprint(schema_path)}, "preprocessing": {"path": "artifacts/v1/preprocessing.json", "sha256": fingerprint(preprocessing_path)}}
        results = {"majority_baseline": {"aggregate": {"mean_validation_accuracy": 0.5}}, "logistic_regression": {"aggregate": {"mean_validation_accuracy": 0.75}}}
        self.write_json(self.root / "reports/v1/training_experiment.json", {"experiment_id": "level-2-train-v1", "dataset_version": "v1", "cross_validation": {"n_splits": 2, "fold_manifest": [{"runner_overlap": [], "validation_runners": ["runner-a", "runner-b"]}, {"runner_overlap": [], "validation_runners": ["runner-c", "runner-d"]}]}, "results": results, "selected_by_mean_validation_accuracy": "logistic_regression"})
        self.write_json(self.root / "artifacts/v1/level3_evaluation_contract.json", {"experiment_id": "level-2-train-v1", "dataset_version": "v1", "input_fingerprints": fingerprints, "held_out_test_set": {"path": "data/v1/processed/test.csv"}, "feature_columns": ["signal"], "target_column": "needs_recovery_next_day", "group_column": "runner_id", "selected_model": "logistic_regression", "models": {"majority_baseline": {"artifact_path": "models/v1/majority_baseline.joblib", "validation_metrics": results["majority_baseline"]["aggregate"]}, "logistic_regression": {"artifact_path": "models/v1/logistic_regression.joblib", "validation_metrics": results["logistic_regression"]["aggregate"]}}})

    def test_selection_lock_succeeds_without_test_csv(self) -> None:
        (self.root / "data/v1/processed/test.csv").unlink()
        lock = lock_selection(self.root)
        self.assertEqual(lock["selected_model"], "logistic_regression")
        self.assertEqual(lock["held_out_test_status"], "not read during selection lock")

    def test_selection_lock_uses_validation_winner_and_cannot_be_rewritten(self) -> None:
        lock = lock_selection(self.root)
        self.assertEqual(lock["selection_metric"], "mean_validation_accuracy")
        with self.assertRaises(FileExistsError):
            lock_selection(self.root)

    def test_level2_fingerprint_mutation_is_rejected(self) -> None:
        path = self.root / "data/v1/processed/train.csv"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            lock_selection(self.root)

    def test_evaluation_source_contains_no_fit_call(self) -> None:
        tree = ast.parse(inspect.getsource(evaluate_models))
        lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform"}]
        self.assertEqual(lines, [])
        self.assertEqual(evaluate_models.verify_no_fit_calls_in_this_module(), [])

    def test_evaluation_requires_a_selection_lock(self) -> None:
        self.assertFalse((self.root / LOCK_RELATIVE_PATH).exists())
        with self.assertRaises(FileNotFoundError):
            run_evaluation(self.root)

    def test_evaluation_is_once_only_and_writes_complete_evidence(self) -> None:
        lock_selection(self.root)
        report = run_evaluation(self.root)
        selected = report["models"]["selected"]["metrics"]
        baseline = report["models"]["majority_baseline"]["metrics"]
        self.assertEqual(selected["confusion_matrix"]["matrix"], [[1, 1], [1, 1]])
        self.assertEqual(selected["class_support"], {"0": 2, "1": 2})
        self.assertEqual(selected["positive_class"]["f1"], 0.5)
        self.assertEqual(baseline["positive_class"]["recall"], 0.0)
        self.assertEqual(report["metric_policy"]["zero_division"], 0.0)
        self.assertTrue(report["evaluated_at_utc"].endswith("+00:00"))
        marker = json.loads((self.root / HOLDOUT_MARKER_RELATIVE_PATH).read_text(encoding="utf-8"))
        self.assertTrue(marker["holdout_consumed"])
        self.assertEqual(set(marker["forbidden_future_uses"]), {"feature_selection", "threshold_tuning", "model_comparison", "model_selection"})
        self.assertEqual(len(pd.read_csv(self.root / ERRORS_RELATIVE_PATH)), 2)
        self.assertTrue((self.root / REPORT_RELATIVE_PATH).exists())
        self.assertTrue((self.root / LEVEL4_RELATIVE_PATH).exists())
        with self.assertRaises(FileExistsError):
            run_evaluation(self.root)


if __name__ == "__main__":
    unittest.main()