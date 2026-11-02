import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.leakage_proxy_experiment import run_leakage_proxy_experiment
from tests.test_features import FeatureExperimentTests


class LeakageProxyTests(unittest.TestCase):
    def test_naive_proxy_succeeds_but_repaired_gate_rejects_it(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = FeatureExperimentTests(methodName="test_schema_gate_supports_three_and_skips_missing_source")
            fixture.build_fixture(root)
            report = run_leakage_proxy_experiment(root)
            self.assertEqual(report["naive_grouped_validation_accuracy"], 1.0)
            self.assertEqual(
                report["repaired_pipeline"]["status"],
                "rejected before fitting",
            )
            self.assertIn(
                "development_only_leakage_proxy",
                report["repaired_pipeline"]["reason"],
            )
            self.assertFalse(report["consumed_v1_holdout_read"])


if __name__ == "__main__":
    unittest.main()