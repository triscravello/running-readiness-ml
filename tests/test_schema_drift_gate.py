import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data_pipeline import run_pipeline
from src.schema_drift_gate import run_schema_drift_demo


class SchemaDriftGateTests(unittest.TestCase):
    def test_valid_batch_succeeds(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            run_pipeline(project_root)
            evidence = run_schema_drift_demo(project_root)
            self.assertTrue(evidence["valid_batch"]["accepted"])
            self.assertEqual(
                evidence["valid_batch"]["validation"]["missing"], []
            )
            self.assertEqual(
                evidence["valid_batch"]["validation"]["unexpected"], []
            )

    def test_malformed_batch_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            run_pipeline(project_root)
            evidence = run_schema_drift_demo(project_root)
            malformed = evidence["malformed_batch"]
            self.assertTrue(malformed["rejected"])
            self.assertEqual(
                malformed["validation"]["missing"],
                ["sleep_hours_last_night"],
            )
            self.assertEqual(
                malformed["validation"]["unexpected"],
                ["next_day_resting_hr"],
            )


if __name__ == "__main__":
    unittest.main()