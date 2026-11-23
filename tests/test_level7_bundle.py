"""Level 7 container-bundle isolation tests.

These assert properties of the *serving bundle* (the ``/app`` layout built by
``src/level7_package.py``): no historical data, no holdout, no consumed-holdout
marker, and the expected serving files present.

They are only meaningful inside the bundle. The Docker test stage runs them with
``LEVEL7_REQUIRE_BUNDLE_TESTS=1``; in that mode they can never be skipped. If the
bundle is wrong they fail, and if they are run in the repository layout with the
flag set they also fail (the repository contains the very files being excluded).

Without the flag, the tests are skipped *only* when the repository layout is
positively detected (``src/level7_service.py`` exists, which the bundle never
contains). The skip message says so. Anywhere else the tests run.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_LAYOUT = (ROOT / "src" / "level7_service.py").is_file()
REQUIRED = os.environ.get("LEVEL7_REQUIRE_BUNDLE_TESTS") == "1"

EXCLUDED_FILES = (
    "data/v1/processed/train.csv",
    "data/v1/processed/test.csv",
    "artifacts/v1/holdout_consumed.json",
    "reports/v1/level3_evaluation.json",
    "reports/v4/finetune_experiment.json",
)
EXCLUDED_DIRECTORIES = (
    "data",
    "models",
    "reports",
    "contracts",
    "evidence",
    "src",
    "artifacts/v1",
    "artifacts/v2",
    "artifacts/v3",
)
REQUIRED_FILES = (
    "level7_runtime.py",
    "level7_service.py",
    "serving-manifest.json",
    "lineage-attestation.json",
    "artifacts/v4/level6_selected_model.pt",
    "artifacts/v4/level6_preprocessor.json",
    "artifacts/v4/level7_deploy_handoff.json",
)


class Level7BundleTests(unittest.TestCase):
    def setUp(self) -> None:
        if REPO_LAYOUT and not REQUIRED:
            self.skipTest(
                "NOT APPLICABLE: repository layout detected. Bundle isolation is "
                "verified inside the container build (LEVEL7_REQUIRE_BUNDLE_TESTS=1)."
            )

    def test_bundle_excludes_historical_and_holdout_files(self):
        for relative in EXCLUDED_FILES:
            self.assertFalse((ROOT / relative).exists(), relative)
        self.assertTrue((ROOT / "lineage-attestation.json").is_file())
        self.assertTrue((ROOT / "serving-manifest.json").is_file())

    def test_bundle_excludes_historical_directories(self):
        for relative in EXCLUDED_DIRECTORIES:
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_bundle_contains_the_serving_files(self):
        for relative in REQUIRED_FILES:
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
