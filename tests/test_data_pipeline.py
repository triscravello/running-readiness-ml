import unittest

from src.data_pipeline import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    TARGET_COLUMN,
    build_outputs,
    build_raw_dataset,
    clean_structural_issues,
    fit_preprocessing,
    split_by_runner,
)

class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = build_outputs()

    def test_schema_and_feature_time_contract_are_valid(self) -> None:
        contract = self.outputs["contract"]
        raw = self.outputs["raw"]
        self.assertTrue(self.outputs["report"]["schema_validation"]["valid"])
        self.assertNotIn(TARGET_COLUMN, FEATURE_COLUMNS)
        self.assertTrue(
            all(
                raw["prediction_timestamp"]
                < raw["target_available_timestamp"]
            )
        )
        feature_specs = [
            column
            for column in contract["columns"]
            if column["role"] == "feature"
        ]
        self.assertTrue(
            all(column["available_at_prediction"] for column in feature_specs)
        )

    def test_cleaning_removes_duplicates_and_invalid_values(self) -> None:
        cleaned_quality = self.outputs["report"]["cleaned_quality"]
        self.assertEqual(cleaned_quality["duplicate_runner_days"], 0)
        self.assertEqual(cleaned_quality["invalid_values"], 0)
        self.assertGreater(cleaned_quality["missing_feature_values"], 0)

    def test_processed_splits_have_no_missing_features(self) -> None:
        train = self.outputs["train"]
        test = self.outputs["test"]
        self.assertEqual(int(train[FEATURE_COLUMNS].isna().sum().sum()), 0)
        self.assertEqual(int(test[FEATURE_COLUMNS].isna().sum().sum()), 0)

    def test_runners_are_separated_between_train_and_test(self) -> None:
        train_runners = set(self.outputs["train"][GROUP_COLUMN])
        test_runners = set(self.outputs["test"][GROUP_COLUMN])
        self.assertEqual(train_runners.intersection(test_runners), set())
        self.assertEqual(
            self.outputs["report"]["split"]["runner_overlap"], []
        )

    def test_preprocessing_is_fitted_on_training_runners_only(self) -> None:
        cleaned = clean_structural_issues(build_raw_dataset())
        train_raw, _ = split_by_runner(cleaned)
        expected = fit_preprocessing(train_raw)
        actual = self.outputs["preprocessing"]["fill_values"]

        self.assertEqual(
            self.outputs["preprocessing"]["fit_scope"],
            "training runners only",
        )
        for feature in FEATURE_COLUMNS:
            self.assertAlmostEqual(actual[feature], float(expected[feature]))

    def test_leakage_experiment_detects_test_contamination(self) -> None:
        experiment = self.outputs["report"]["leakage_demo"]
        self.assertTrue(
            experiment["global_fit_changed_after_test_perturbation"]
        )
        self.assertFalse(
            experiment["train_fit_changed_after_test_perturbation"]
        )

    def test_outputs_are_deterministic(self) -> None:
        second = build_outputs()
        for name in ("raw", "cleaned", "train", "test"):
            self.assertTrue(self.outputs[name].equals(second[name]))
        for name in ("preprocessing", "report", "contract"):
            self.assertEqual(self.outputs[name], second[name])


if __name__ == "__main__":
    unittest.main()