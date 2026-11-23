"""Level 7 service tests, portable between the repository and the ``/app`` bundle.

Two layouts are supported and detected positively:

* **Image layout** (``/app``): flat modules next to the real ``serving-manifest.json``.
  The service runs with its default root (``SERVING_ROOT`` unset).
* **Repository layout**: ``src/`` modules. A disposable serving root is built in a
  temporary directory from the verified artifacts and the existing manifest
  generator (``create_serving_manifest``), then selected with ``SERVING_ROOT``.
  Nothing tracked is written and the tracked ``serving-manifest.json`` is never
  touched.

Serving-root overrides and startup failures are tested in subprocesses so no
environment or imported-module state leaks between tests. Container bundle isolation
lives in ``tests/test_level7_bundle.py``.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
IMAGE_LAYOUT = not (ROOT / "src" / "level7_service.py").is_file()
SERVICE_MODULE = "level7_service" if IMAGE_LAYOUT else "src.level7_service"

# Bytes of the serving manifest shipped in running-readiness-level7:4.0.0, and
# reproduced by create_serving_manifest() from tracked handoffs (verified in F1).
VERIFIED_MANIFEST_SHA256 = (
    "a007e449a7728f632d63ba3709109997db88e0bfe52e35b7f4263d1fc4d2b12a"
)
# Golden regression reference for VALID_REQUEST. Observed bit-identical on
# torch 2.2.2 / NumPy 1.26.4 (native) and torch 2.15.0 / NumPy 2.5.4 (canonical
# Docker). The tolerance (absolute 1e-6) allows a few float32 ulps (one ulp near 0.5
# is about 6e-8) and is orders of magnitude below any meaningful change.
GOLDEN_PROBABILITY = 0.49793776869773865
GOLDEN_TOLERANCE = 1e-6
LOCKED_THRESHOLD = 0.5


def build_fixture_root(destination: Path) -> Path:
    """Disposable serving root: verified artifacts + the existing manifest generator."""
    from src.level7_package import create_serving_manifest
    from src.level7_runtime import CHECKPOINT_PATH, HANDOFF_PATH, PREPROCESSOR_PATH

    for relative in (CHECKPOINT_PATH, HANDOFF_PATH, PREPROCESSOR_PATH):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    manifest = create_serving_manifest(ROOT)
    (destination / "serving-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


if IMAGE_LAYOUT:
    SERVING_ROOT = ROOT
    from level7_runtime import (  # noqa: E402
        CHECKPOINT_PATH,
        HANDOFF_PATH,
        MANIFEST_PATH,
        PREPROCESSOR_PATH,
        IntegrityError,
        ModelRuntime,
    )
    import level7_service as service  # noqa: E402
else:
    _fixture_directory = tempfile.TemporaryDirectory(prefix="level7-fixture-")
    atexit.register(_fixture_directory.cleanup)
    SERVING_ROOT = build_fixture_root(Path(_fixture_directory.name))
    os.environ["SERVING_ROOT"] = str(SERVING_ROOT)
    try:
        from src.level7_runtime import (  # noqa: E402
            CHECKPOINT_PATH,
            HANDOFF_PATH,
            MANIFEST_PATH,
            PREPROCESSOR_PATH,
            IntegrityError,
            ModelRuntime,
        )
        import src.level7_service as service  # noqa: E402
    finally:
        os.environ.pop("SERVING_ROOT", None)

if service.BASE_DIR != SERVING_ROOT.resolve():
    raise ImportError(
        "level7_service was imported earlier with a different serving root: "
        f"{service.BASE_DIR} != {SERVING_ROOT.resolve()}"
    )
app = service.app

ARTIFACT_PATHS = (CHECKPOINT_PATH, HANDOFF_PATH, PREPROCESSOR_PATH)
VALID_REQUEST = {
    "weekly_km_to_date": 32.0,
    "sleep_hours_last_night": 7.0,
    "resting_hr_morning": 50.0,
    "soreness_score_evening": 4.0,
    "prior_injury": 0,
}
LOWER_BOUNDS = {
    "weekly_km_to_date": 0.0,
    "sleep_hours_last_night": 0.0,
    "resting_hr_morning": 30.0,
    "soreness_score_evening": 0.0,
    "prior_injury": 0,
}
UPPER_BOUNDS = {
    "weekly_km_to_date": 200.0,
    "sleep_hours_last_night": 14.0,
    "resting_hr_morning": 220.0,
    "soreness_score_evening": 10.0,
    "prior_injury": 1,
}
RESPONSE_FIELDS = {"probability", "decision", "threshold", "device", "selected_approach"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clone_serving_root(destination: Path) -> Path:
    """Copy the serving root under test (manifest + the three artifacts)."""
    for relative in (MANIFEST_PATH, *ARTIFACT_PATHS):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SERVING_ROOT / relative, target)
    return destination


def flip_middle_bit(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(data)


def import_service_in_subprocess(serving_root, cwd=ROOT):
    """Import the service in a fresh interpreter. ``None`` leaves SERVING_ROOT unset."""
    env = {key: value for key, value in os.environ.items() if key != "SERVING_ROOT"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Importability must not depend on the working directory (a relative override is
    # resolved against ``cwd``), so put the code root on the path explicitly.
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    if serving_root is not None:
        env["SERVING_ROOT"] = str(serving_root)
    code = f"import {SERVICE_MODULE} as s; print('BASE_DIR=' + str(s.BASE_DIR))"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


class ReferenceEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer2(torch.relu(self.layer1(inputs))))


class ReferenceTransferClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = ReferenceEncoder(input_dim)
        self.head = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs)).squeeze(1)


def reference_prediction(payload: dict) -> tuple[float, int, float]:
    """Independent reference. Always reads the originals under ROOT, never the fixture."""
    handoff = json.loads(
        (ROOT / "artifacts/v4/level7_deploy_handoff.json").read_text(encoding="utf-8")
    )
    preprocessor = json.loads(
        (ROOT / "artifacts/v4/level6_preprocessor.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        ROOT / CHECKPOINT_PATH, map_location=torch.device("cpu"), weights_only=True
    )
    values = np.asarray(
        [[float(payload[name]) for name in handoff["feature_names"]]], dtype=np.float64
    )
    transformed = (
        (
            np.where(
                np.isnan(values),
                np.asarray(preprocessor["imputer_statistics"], dtype=np.float64),
                values,
            )
            - np.asarray(preprocessor["scaler_mean"], dtype=np.float64)
        )
        / np.asarray(preprocessor["scaler_scale"], dtype=np.float64)
    ).astype(np.float32)
    model = ReferenceTransferClassifier(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(torch.from_numpy(transformed))).item())
    threshold = float(checkpoint["threshold"])
    return probability, int(probability >= threshold), threshold


class Level7ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def post_raw(self, text: str):
        return self.client.post(
            "/predict", content=text, headers={"content-type": "application/json"}
        )

    def test_health_documents_integrity_and_limitations(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["integrity_status"], "passed")
        self.assertEqual(body["device"], "cpu")
        self.assertEqual(body["selected_approach"], "partial")
        self.assertEqual(body["evidence_scope"], "development_only")
        self.assertIn("Synthetic educational", body["model_notice"])
        self.assertIn("not claim", body["inherited_v1_preprocessing_limitation"])

    def test_prediction_matches_independent_reference(self):
        probability, decision, threshold = reference_prediction(VALID_REQUEST)
        response = self.client.post("/predict", json=VALID_REQUEST)
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.json()["probability"], probability, places=7)
        self.assertEqual(response.json()["decision"], decision)
        self.assertEqual(response.json()["threshold"], threshold)

    def test_prediction_matches_reference_across_inputs(self):
        cases = {
            "sample": VALID_REQUEST,
            "lower bounds": LOWER_BOUNDS,
            "upper bounds": UPPER_BOUNDS,
            "integer for float field": dict(VALID_REQUEST, weekly_km_to_date=32),
            "prior injury": dict(VALID_REQUEST, prior_injury=1),
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                probability, decision, threshold = reference_prediction(payload)
                response = self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertAlmostEqual(body["probability"], probability, places=7)
                self.assertEqual(body["decision"], decision)
                self.assertEqual(body["threshold"], threshold)

    def test_golden_sample_probability(self):
        body = self.client.post("/predict", json=VALID_REQUEST).json()
        self.assertLessEqual(
            abs(body["probability"] - GOLDEN_PROBABILITY),
            GOLDEN_TOLERANCE,
            f"probability {body['probability']!r} drifted from {GOLDEN_PROBABILITY!r}",
        )

    def test_decision_and_threshold_are_locked(self):
        body = self.client.post("/predict", json=VALID_REQUEST).json()
        self.assertEqual(body["decision"], 0)
        self.assertEqual(body["threshold"], LOCKED_THRESHOLD)
        self.assertEqual(body["selected_approach"], "partial")
        self.assertEqual(body["device"], "cpu")
        self.assertEqual(set(body), RESPONSE_FIELDS)

    def test_inclusive_bounds_are_accepted(self):
        for label, payload in (("lower", LOWER_BOUNDS), ("upper", UPPER_BOUNDS)):
            with self.subTest(label=label):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 200)

    def test_invalid_requests_are_rejected(self):
        missing = dict(VALID_REQUEST)
        missing.pop("sleep_hours_last_night")
        cases = {
            "missing": missing,
            "extra": dict(VALID_REQUEST, needs_recovery_next_day=1),
            "wrong type": dict(VALID_REQUEST, resting_hr_morning="50.0"),
            "out of range": dict(VALID_REQUEST, soreness_score_evening=11.0),
            "float prior_injury": dict(VALID_REQUEST, prior_injury=0.0),
            "bool prior_injury": dict(VALID_REQUEST, prior_injury=True),
            "prior_injury 2": dict(VALID_REQUEST, prior_injury=2),
            "null prior_injury": dict(VALID_REQUEST, prior_injury=None),
            "weekly km above max": dict(VALID_REQUEST, weekly_km_to_date=200.0001),
            "weekly km negative": dict(VALID_REQUEST, weekly_km_to_date=-0.1),
            "sleep above max": dict(VALID_REQUEST, sleep_hours_last_night=14.0001),
            "resting hr below min": dict(VALID_REQUEST, resting_hr_morning=29.99),
            "resting hr above max": dict(VALID_REQUEST, resting_hr_morning=220.01),
            "empty object": {},
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)

    def test_nonfinite_values_are_rejected(self):
        template = json.dumps(VALID_REQUEST)
        for label, token in (("NaN", "NaN"), ("Infinity", "Infinity"), ("-Infinity", "-Infinity")):
            with self.subTest(label=label):
                response = self.post_raw(template.replace("32.0", token))
                self.assertEqual(response.status_code, 422)

    def test_malformed_bodies_and_wrong_methods(self):
        for label, text in (("not json", "not-json"), ("list body", "[]")):
            with self.subTest(label=label):
                self.assertEqual(self.post_raw(text).status_code, 422)
        self.assertEqual(self.client.get("/predict").status_code, 405)
        self.assertEqual(self.client.get("/no-such-route").status_code, 404)

    def test_validation_errors_do_not_echo_client_input(self):
        response = self.client.post("/predict", json=dict(VALID_REQUEST, prior_injury=7))
        self.assertEqual(response.status_code, 422)
        for entry in response.json()["detail"]:
            self.assertEqual(set(entry), {"type", "loc", "msg"})

    def test_startup_rejects_corrupted_temporary_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = clone_serving_root(Path(directory))
            flip_middle_bit(temporary_root / CHECKPOINT_PATH)
            with self.assertRaises(IntegrityError):
                ModelRuntime(temporary_root)


class ServingManifestFixtureTests(unittest.TestCase):
    def test_serving_manifest_matches_verified_bytes(self):
        self.assertEqual(sha256_file(SERVING_ROOT / MANIFEST_PATH), VERIFIED_MANIFEST_SHA256)

    @unittest.skipIf(IMAGE_LAYOUT, "NOT APPLICABLE: image layout has no repository sources")
    def test_fixture_pins_match_the_tracked_manifest(self):
        fixture = json.loads((SERVING_ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))
        tracked = json.loads((ROOT / "serving-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(fixture["artifacts"], tracked["artifacts"])

    @unittest.skipIf(IMAGE_LAYOUT, "NOT APPLICABLE: image layout has no repository sources")
    def test_fixture_artifacts_are_byte_identical_to_the_originals(self):
        for relative in ARTIFACT_PATHS:
            with self.subTest(path=relative.as_posix()):
                self.assertEqual(
                    sha256_file(SERVING_ROOT / relative), sha256_file(ROOT / relative)
                )

    @unittest.skipIf(IMAGE_LAYOUT, "NOT APPLICABLE: image layout has no repository sources")
    def test_fixture_lives_outside_the_repository(self):
        self.assertNotIn(ROOT.resolve(), SERVING_ROOT.resolve().parents)


class ArtifactIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = clone_serving_root(Path(self.directory.name))

    def test_unmodified_clone_loads(self):
        self.assertEqual(ModelRuntime(self.root).integrity_status, "passed")

    def test_missing_manifest_is_rejected(self):
        (self.root / MANIFEST_PATH).unlink()
        with self.assertRaisesRegex(IntegrityError, "Missing serving manifest"):
            ModelRuntime(self.root)

    def test_each_missing_artifact_is_rejected(self):
        for relative in ARTIFACT_PATHS:
            with self.subTest(path=relative.as_posix()):
                clone = clone_serving_root(Path(tempfile.mkdtemp(dir=self.directory.name)))
                (clone / relative).unlink()
                with self.assertRaisesRegex(IntegrityError, "Missing serving artifact"):
                    ModelRuntime(clone)

    def test_each_corrupted_artifact_is_rejected(self):
        for relative in ARTIFACT_PATHS:
            with self.subTest(path=relative.as_posix()):
                clone = clone_serving_root(Path(tempfile.mkdtemp(dir=self.directory.name)))
                flip_middle_bit(clone / relative)
                with self.assertRaisesRegex(IntegrityError, "Serving artifact mismatch"):
                    ModelRuntime(clone)

    def test_altered_manifest_hash_is_rejected(self):
        manifest_path = self.root / MANIFEST_PATH
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = PREPROCESSOR_PATH.as_posix()
        digest = manifest["artifacts"][key]
        manifest["artifacts"][key] = ("0" if digest[0] != "0" else "1") + digest[1:]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(IntegrityError, "Serving artifact mismatch"):
            ModelRuntime(self.root)

    def test_manifest_entry_for_an_absent_file_is_rejected(self):
        manifest_path = self.root / MANIFEST_PATH
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["artifacts/v4/not-shipped.bin"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(IntegrityError, "Missing serving artifact"):
            ModelRuntime(self.root)


class ServingRootTests(unittest.TestCase):
    """Default and overridden serving roots, each in a fresh interpreter."""

    def test_default_root_is_unchanged_when_unset(self):
        result = import_service_in_subprocess(None)
        if IMAGE_LAYOUT:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"BASE_DIR={ROOT.resolve()}", result.stdout)
        else:
            expected = ROOT.resolve() / "src" / "serving-manifest.json"
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f"Missing serving manifest: {expected}", result.stderr)

    def test_empty_override_is_treated_as_unset(self):
        unset = import_service_in_subprocess(None)
        empty = import_service_in_subprocess("")
        self.assertEqual(empty.returncode, unset.returncode)
        self.assertEqual(empty.stdout, unset.stdout)
        if not IMAGE_LAYOUT:
            self.assertIn("src", empty.stderr)
            self.assertIn("Missing serving manifest", empty.stderr)

    def test_override_selects_the_given_root(self):
        result = import_service_in_subprocess(SERVING_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"BASE_DIR={SERVING_ROOT.resolve()}", result.stdout)

    def test_relative_override_resolves_against_the_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            clone_serving_root(parent / "bundle")
            result = import_service_in_subprocess("bundle", cwd=parent)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"BASE_DIR={(parent / 'bundle').resolve()}", result.stdout)

    def test_override_to_a_missing_directory_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "does-not-exist"
            result = import_service_in_subprocess(missing)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing serving manifest", result.stderr)
            self.assertIn(str(missing.resolve()), result.stderr)

    def test_override_with_a_corrupted_artifact_fails_closed(self):
        for relative in ARTIFACT_PATHS:
            with self.subTest(path=relative.as_posix()):
                with tempfile.TemporaryDirectory() as directory:
                    root = clone_serving_root(Path(directory))
                    flip_middle_bit(root / relative)
                    result = import_service_in_subprocess(root)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Serving artifact mismatch", result.stderr)

    def test_override_with_a_missing_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = clone_serving_root(Path(directory))
            (root / CHECKPOINT_PATH).unlink()
            result = import_service_in_subprocess(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing serving artifact", result.stderr)

    def test_override_with_a_mismatched_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = clone_serving_root(Path(directory))
            manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
            key = HANDOFF_PATH.as_posix()
            manifest["artifacts"][key] = "f" * 64
            (root / MANIFEST_PATH).write_text(json.dumps(manifest), encoding="utf-8")
            result = import_service_in_subprocess(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Serving artifact mismatch", result.stderr)

    def test_override_does_not_leak_into_this_process(self):
        self.assertNotIn("SERVING_ROOT", os.environ)


@unittest.skipIf(IMAGE_LAYOUT, "NOT APPLICABLE: the image has no Dockerfile or sources")
class DockerfileContractTests(unittest.TestCase):
    """Static guard that the build keeps executing the bundle test without skips."""

    @staticmethod
    def instructions(text: str) -> list[str]:
        """Dockerfile instructions with backslash continuations joined."""
        return [line for line in text.replace("\\\n", " ").splitlines() if line.strip()]

    def setUp(self) -> None:
        self.text = (ROOT / "Dockerfile.level7").read_text(encoding="utf-8")
        self.lines = self.instructions(self.text)

    def test_test_stage_copies_and_runs_both_test_files_without_skips(self):
        copies = [line for line in self.lines if line.startswith("COPY test_level7_")]
        self.assertIn(
            "COPY test_level7_bundle.py /app/tests/test_level7_bundle.py", copies
        )
        runs = [line for line in self.lines if line.startswith("RUN") and "unittest" in line]
        self.assertEqual(len(runs), 1, runs)
        run = " ".join(runs[0].split())
        self.assertIn("LEVEL7_REQUIRE_BUNDLE_TESTS=1 python -m unittest", run)
        self.assertIn("tests/test_level7_service.py", run)
        self.assertIn("tests/test_level7_bundle.py", run)
        self.assertLess(run.index("unittest"), run.index("touch /tmp/level7-tests-passed"))
        self.assertIn("&&", run[run.index("unittest"): run.index("touch")])

    def test_runtime_stage_still_takes_only_the_marker_from_the_test_stage(self):
        copies = [line for line in self.lines if "--from=test" in line]
        self.assertEqual(
            copies, ["COPY --from=test /tmp/level7-tests-passed /app/level7-tests-passed"]
        )

    def test_packager_allowlist_ships_the_bundle_test(self):
        from src.level7_package import ALLOWLIST

        self.assertEqual(
            ALLOWLIST[Path("tests/test_level7_bundle.py")], Path("test_level7_bundle.py")
        )
        self.assertEqual(
            ALLOWLIST[Path("tests/test_level7_service.py")], Path("test_level7_service.py")
        )


if __name__ == "__main__":
    unittest.main()
