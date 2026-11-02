import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import src.feature_experiment as feature_experiment
from src.feature_experiment import (
    FORBIDDEN_HOLDOUT_USES,
    LEVEL5_HANDOFF_PATH,
    RAW_FEATURES,
    SCHEMA_PATH,
    V2_REPORT_PATH,
    audit_handoff,
    build_pipeline,
    fingerprint,
    read_json,
    resolve_feature_plan,
    run_experiment,
)


class FeatureExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.build_fixture(self.root)

    def write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def build_fixture(self, root: Path) -> None:
        rows = []
        for number in range(12):
            rows.append(
                {
                    "runner_id": f"runner-{number:02d}",
                    "weekly_km_to_date": 20.0 + number * 2.0,
                    "sleep_hours_last_night": 8.0 - (number % 4) * 0.5,
                    "resting_hr_morning": 45.0 + number % 5,
                    "soreness_score_evening": float(number % 10),
                    "prior_injury": int(number % 3 == 0),
                    "needs_recovery_next_day": int(number % 3 != 0),
                }
            )
        train_path = root / "data/v1/processed/train.csv"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(train_path, index=False)

        columns = [
            {"name": name, "role": "feature", "available_at_prediction": True}
            for name in RAW_FEATURES
        ]
        columns.extend(
            [
                {"name": "runner_id", "role": "group_identifier", "available_at_prediction": True},
                {"name": "needs_recovery_next_day", "role": "target", "available_at_prediction": False},
            ]
        )
        schema = {
            "schema_version": "1.0.0",
            "dataset_version": "v1",
            "feature_columns": list(RAW_FEATURES),
            "target_column": "needs_recovery_next_day",
            "group_column": "runner_id",
            "columns": columns,
        }
        schema_path = root / "contracts/v1/schema_contract.json"
        preprocessing_path = root / "artifacts/v1/preprocessing.json"
        self.write_json(schema_path, schema)
        self.write_json(
            preprocessing_path,
            {
                "dataset_version": "v1",
                "fit_scope": "training runners only",
                "feature_columns": list(RAW_FEATURES),
            },
        )
        level3_path = root / "artifacts/v1/level3_evaluation_contract.json"
        self.write_json(
            level3_path,
            {
                "input_fingerprints": {
                    "train": {"path": "data/v1/processed/train.csv", "sha256": fingerprint(train_path)},
                    "schema": {"path": "contracts/v1/schema_contract.json", "sha256": fingerprint(schema_path)},
                    "preprocessing": {"path": "artifacts/v1/preprocessing.json", "sha256": fingerprint(preprocessing_path)},
                }
            },
        )
        lock_path = root / "artifacts/v1/level3_selection_lock.json"
        self.write_json(lock_path, {"selected_model": "decision_tree"})
        marker_path = root / "artifacts/v1/holdout_consumed.json"
        self.write_json(
            marker_path,
            {
                "holdout_consumed": True,
                "forbidden_future_uses": sorted(FORBIDDEN_HOLDOUT_USES),
            },
        )
        evaluation_path = root / "reports/v1/level3_evaluation.json"
        self.write_json(evaluation_path, {"holdout_consumed": True})
        self.write_json(
            root / "artifacts/v1/level4_features_contract.json",
            {
                "locked_selection": {
                    "model": "decision_tree",
                    "selection_lock_sha256": fingerprint(lock_path),
                },
                "holdout_policy": {
                    "holdout_consumed": True,
                    "marker_sha256": fingerprint(marker_path),
                    "forbidden_future_uses": sorted(FORBIDDEN_HOLDOUT_USES),
                },
                "evaluation_evidence": {
                    "json_report": {"sha256": fingerprint(evaluation_path)}
                },
                "target_column": "needs_recovery_next_day",
                "group_column": "runner_id",
            },
        )

    def test_immutable_audit_passes_and_rejects_drift(self) -> None:
        self.assertEqual(audit_handoff(self.root)["audit_status"], "passed")
        train_path = self.root / "data/v1/processed/train.csv"
        train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            audit_handoff(self.root)

    def test_schema_gate_supports_three_and_skips_missing_source(self) -> None:
        plan = resolve_feature_plan(read_json(self.root / SCHEMA_PATH))
        self.assertEqual(len(plan["supported"]), 3)
        self.assertEqual(plan["skipped"][0]["name"], "prior_week_load_ratio")
        self.assertEqual(plan["skipped"][0]["missing_sources"], ["prior_week_km"])

    def test_pipeline_owns_imputation_selection_and_classifier(self) -> None:
        plan = resolve_feature_plan(read_json(self.root / SCHEMA_PATH))
        engineered = tuple(record["name"] for record in plan["supported"])
        pipeline = build_pipeline("engineered", engineered)
        self.assertEqual(
            list(pipeline.named_steps),
            ["features", "imputer", "selector", "classifier"],
        )

    def test_variants_reuse_identical_group_folds(self) -> None:
        report = run_experiment(self.root)
        raw = [fold["fold_signature"] for fold in report["results"]["raw"]["folds"]]
        engineered = [
            fold["fold_signature"]
            for fold in report["results"]["engineered"]["folds"]
        ]
        self.assertEqual(raw, engineered)
        self.assertTrue(
            all(
                fold["runner_overlap"] == []
                for variant in report["results"].values()
                for fold in variant["folds"]
            )
        )

    def test_report_and_level5_handoff_are_deterministic(self) -> None:
        first_report = run_experiment(self.root)
        with TemporaryDirectory() as second_directory:
            second_root = Path(second_directory)
            self.build_fixture(second_root)
            second_report = run_experiment(second_root)
            self.assertEqual(
                json.dumps(first_report, sort_keys=True),
                json.dumps(second_report, sort_keys=True),
            )
            self.assertEqual(
                (self.root / LEVEL5_HANDOFF_PATH).read_text(encoding="utf-8"),
                (second_root / LEVEL5_HANDOFF_PATH).read_text(encoding="utf-8"),
            )

    def test_source_and_handoff_preserve_consumed_holdout_boundary(self) -> None:
        source = inspect.getsource(feature_experiment)
        self.assertNotIn("test.csv", source)
        run_experiment(self.root)
        report = read_json(self.root / V2_REPORT_PATH)
        handoff = read_json(self.root / LEVEL5_HANDOFF_PATH)
        self.assertFalse(report["consumed_v1_holdout_read"])
        self.assertEqual(
            set(handoff["holdout_policy"]["forbidden_future_uses"]),
            FORBIDDEN_HOLDOUT_USES,
        )


if __name__ == "__main__":
    unittest.main()