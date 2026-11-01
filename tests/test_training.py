import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
from sklearn.pipeline import Pipeline

from src.train_models import (
    build_model_candidates,
    load_training_handoff,
    run_training,
)

class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_root = Path(__file__).resolve().parents[1]
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project_root = Path(self.temporary_directory.name)
        self.copy_handoff(self.project_root)
        self.metadata = run_training(self.project_root)

    def copy_handoff(self, destination: Path) -> None:
        sources = {
            "train": self.source_root
            / "data"
            / "v1"
            / "processed"
            / "train.csv",
            "schema": self.source_root
            / "contracts"
            / "v1"
            / "schema_contract.json",
            "preprocessing": self.source_root
            / "artifacts"
            / "v1"
            / "preprocessing.json",
        }
        destinations = {
            "train": destination
            / "data"
            / "v1"
            / "processed"
            / "train.csv",
            "schema": destination
            / "contracts"
            / "v1"
            / "schema_contract.json",
            "preprocessing": destination
            / "artifacts"
            / "v1"
            / "preprocessing.json",
        }
        for name, source in sources.items():
            destinations[name].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destinations[name])

    def test_group_folds_have_no_overlap_and_cover_all_runners(self) -> None:
        train, contract, _ = load_training_handoff(self.project_root)
        manifest = self.metadata["cross_validation"]["fold_manifest"]
        validation_runners = set()
        for fold in manifest:
            self.assertEqual(fold["runner_overlap"], [])
            validation_runners.update(fold["validation_runners"])
        self.assertEqual(
            validation_runners,
            set(str(value) for value in train[contract["group_column"]].unique()),
        )

    def test_majority_baseline_predicts_one_class_in_every_fold(self) -> None:
        folds = self.metadata["results"]["majority_baseline"]["folds"]
        self.assertTrue(
            all(len(fold["unique_validation_predictions"]) == 1 for fold in folds)
        )

    def test_saved_pipelines_reload_and_predict(self) -> None:
        train, contract, _ = load_training_handoff(self.project_root)
        for name in build_model_candidates():
            artifact = self.project_root / self.metadata["model_artifacts"][name]
            model = joblib.load(artifact)
            self.assertIsInstance(model, Pipeline)
            predictions = model.predict(train[contract["feature_columns"]])
            self.assertEqual(len(predictions), len(train))

    def test_training_does_not_require_the_held_out_test_file(self) -> None:
        test_path = self.project_root / "data" / "v1" / "processed" / "test.csv"
        self.assertFalse(test_path.exists())
        self.assertEqual(
            self.metadata["held_out_test_set"]["status"],
            "not loaded, fingerprinted, or scored in Level 2",
        )

    def test_level3_contract_is_complete(self) -> None:
        contract_path = (
            self.project_root
            / "artifacts"
            / "v1"
            / "level3_evaluation_contract.json"
        )
        level3 = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(level3["input_fingerprints"]),
            {"train", "schema", "preprocessing"},
        )
        self.assertEqual(set(level3["models"]), set(build_model_candidates()))
        self.assertIn(level3["selected_model"], level3["models"])
        self.assertEqual(level3["random_seeds"]["group_kfold"], 42)

    def test_experiment_and_contract_are_deterministic(self) -> None:
        first_contract_path = (
            self.project_root
            / "artifacts"
            / "v1"
            / "level3_evaluation_contract.json"
        )
        first_contract = json.loads(
            first_contract_path.read_text(encoding="utf-8")
        )
        with TemporaryDirectory() as second_directory:
            second_root = Path(second_directory)
            self.copy_handoff(second_root)
            second_metadata = run_training(second_root)
            second_contract = json.loads(
                (
                    second_root
                    / "artifacts"
                    / "v1"
                    / "level3_evaluation_contract.json"
                ).read_text(encoding="utf-8")
            )
        self.assertEqual(
            json.dumps(self.metadata, sort_keys=True),
            json.dumps(second_metadata, sort_keys=True),
        )
        self.assertEqual(
            json.dumps(first_contract, sort_keys=True),
            json.dumps(second_contract, sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()