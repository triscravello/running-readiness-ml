"""Regression guard for the frozen lineage files.

Compares protected files against the committed inventory ``frozen-hashes.json``.
Standard library only, so it runs natively and in the canonical Docker images.

Holdout rule: this module never opens, parses, or hashes the consumed v1 holdout
(``data/v1/processed/test.csv``). Directory listings may mention its name, and the
recorded holdout metadata is checked through ``artifacts/v1/holdout_consumed.json``.

Gitignored lineage files are absent from a fresh clone. Those are reported as
UNAVAILABLE (a skip), never as a pass. Set ``FROZEN_REQUIRE_LOCAL=1`` to make
their absence a failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "frozen-hashes.json"
HOLDOUT = "data/v1/processed/test.csv"
MARKER = "artifacts/v1/holdout_consumed.json"
LEVEL4_CONTRACT = "artifacts/v1/level4_features_contract.json"
LEVEL3_REPORT = "reports/v1/level3_evaluation.json"
EXPECTED_ROOTS = ["artifacts", "contracts", "data", "evidence", "models", "reports"]
FORBIDDEN_USES = {
    "feature_selection",
    "model_comparison",
    "model_selection",
    "threshold_tuning",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRE_LOCAL = os.environ.get("FROZEN_REQUIRE_LOCAL") == "1"


def sha256_of(relative: str) -> str:
    """Hash a protected file. Refuses the holdout by construction."""
    if relative == HOLDOUT:
        raise AssertionError("the consumed holdout must never be read or hashed")
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def read_json(relative: str) -> dict:
    if relative == HOLDOUT:
        raise AssertionError("the consumed holdout must never be read or parsed")
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


class FrozenArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = load_inventory()
        cls.tracked = cls.inventory["files"]["tracked"]
        cls.local_only = cls.inventory["files"]["local_only"]
        cls.scope = cls.inventory["scope"]

    def unavailable(self, relative: str) -> None:
        message = f"UNAVAILABLE (gitignored lineage file not present): {relative}"
        if REQUIRE_LOCAL:
            self.fail(message + " [FROZEN_REQUIRE_LOCAL=1]")
        self.skipTest(message)

    # -- inventory self-consistency ---------------------------------------
    def test_inventory_is_well_formed(self) -> None:
        self.assertEqual(self.inventory["inventory_version"], "1.0.0")
        self.assertEqual(self.scope["protected_roots"], EXPECTED_ROOTS)
        roots = tuple(f"{name}/" for name in EXPECTED_ROOTS)
        allowed_root_files = set(self.scope["protected_root_files"])
        self.assertFalse(set(self.tracked) & set(self.local_only))
        for group in (self.tracked, self.local_only):
            for relative, digest in group.items():
                with self.subTest(path=relative):
                    self.assertRegex(digest, SHA256_PATTERN)
                    self.assertNotIn("..", Path(relative).parts)
                    self.assertFalse(Path(relative).is_absolute())
                    self.assertTrue(
                        relative.startswith(roots) or relative in allowed_root_files
                    )

    def test_holdout_is_excluded_from_inventory(self) -> None:
        self.assertNotIn(HOLDOUT, self.tracked)
        self.assertNotIn(HOLDOUT, self.local_only)
        excluded = self.inventory["excluded_holdout"]
        self.assertEqual(excluded["path"], HOLDOUT)
        self.assertIs(excluded["hashed_here"], False)
        self.assertEqual(excluded["metadata_source"], MARKER)

    def test_helpers_refuse_the_holdout(self) -> None:
        with self.assertRaises(AssertionError):
            sha256_of(HOLDOUT)
        with self.assertRaises(AssertionError):
            read_json(HOLDOUT)

    # -- modification and deletion ----------------------------------------
    def test_tracked_files_match_inventory(self) -> None:
        for relative, expected in self.tracked.items():
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file(), f"deleted: {relative}")
                self.assertEqual(sha256_of(relative), expected, f"modified: {relative}")

    def test_local_only_files_match_inventory_when_present(self) -> None:
        for relative, expected in self.local_only.items():
            with self.subTest(path=relative):
                if not (ROOT / relative).is_file():
                    self.unavailable(relative)
                self.assertEqual(sha256_of(relative), expected, f"modified: {relative}")

    # -- additions --------------------------------------------------------
    def test_no_unexpected_files_in_protected_paths(self) -> None:
        known = set(self.tracked) | set(self.local_only)
        ignored_names = set(self.scope["ignored_basenames"])
        ignored_dirs = set(self.scope["ignored_directory_names"])
        unexpected = []
        for name in self.scope["protected_roots"]:
            for directory, subdirectories, files in os.walk(ROOT / name):
                subdirectories[:] = [d for d in subdirectories if d not in ignored_dirs]
                for filename in files:
                    if filename in ignored_names:
                        continue
                    relative = (Path(directory) / filename).relative_to(ROOT).as_posix()
                    if relative == HOLDOUT:
                        continue  # excluded by name only; never opened
                    if relative not in known:
                        unexpected.append(relative)
        self.assertEqual(sorted(unexpected), [], "unexpected additions in protected paths")

    # -- holdout metadata (marker only) ------------------------------------
    def test_pinned_marker_hash_matches_level4_contract(self) -> None:
        contract = read_json(LEVEL4_CONTRACT)
        policy = contract["holdout_policy"]
        self.assertIs(policy["holdout_consumed"], True)
        self.assertEqual(policy["marker_path"], MARKER)
        self.assertEqual(set(policy["forbidden_future_uses"]), FORBIDDEN_USES)
        self.assertEqual(self.local_only[MARKER], policy["marker_sha256"])

    def test_recorded_holdout_metadata_without_opening_the_holdout(self) -> None:
        if not (ROOT / MARKER).is_file():
            self.unavailable(MARKER)
        marker = read_json(MARKER)
        self.assertIs(marker["holdout_consumed"], True)
        self.assertEqual(set(marker["forbidden_future_uses"]), FORBIDDEN_USES)
        self.assertEqual(marker["dataset"]["path"], HOLDOUT)
        self.assertRegex(marker["dataset"]["sha256"], SHA256_PATTERN)
        report = ROOT / LEVEL3_REPORT
        if report.is_file():
            test_data = read_json(LEVEL3_REPORT)["test_data"]
            self.assertEqual(test_data["path"], HOLDOUT)
            self.assertEqual(test_data["sha256"], marker["dataset"]["sha256"])


if __name__ == "__main__":
    unittest.main()
