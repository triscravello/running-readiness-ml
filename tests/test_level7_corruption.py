from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.level7_lineage import PROJECT_ROOT, VerificationError, sha256, verify_file
from src.level7_package import stage_context

SELECTED_RELATIVE = Path("artifacts/v4/level6_selected_model.pt")

class CorruptedArtifactTests(unittest.TestCase):
    def test_corruption_stops_packaging_before_docker(self):
        original = PROJECT_ROOT / SELECTED_RELATIVE
        handoff = json.loads(
            (PROJECT_ROOT / "artifacts/v4/level7_deploy_handoff.json").read_text(
                encoding="utf-8"
            )
        )
        expected = handoff["fingerprints"]["selected_checkpoint"]
        original_before = sha256(original)
        docker_calls: list[list[str]] = []

        def fake_runner(command, **kwargs):
            docker_calls.append(command)

        with tempfile.TemporaryDirectory() as directory:
            corrupted = Path(directory) / original.name
            shutil.copy2(original, corrupted)
            data = bytearray(corrupted.read_bytes())
            data[len(data) // 2] ^= 0x01
            corrupted.write_bytes(data)
            with self.assertRaises(VerificationError):
                stage_context(
                    PROJECT_ROOT,
                    overrides={SELECTED_RELATIVE: corrupted},
                    runner=fake_runner,
                )
            self.assertEqual(docker_calls, [])
            shutil.copy2(original, corrupted)
            verify_file(corrupted, expected)

        self.assertEqual(original_before, expected)
        self.assertEqual(sha256(original), original_before)


if __name__ == "__main__":
    unittest.main()