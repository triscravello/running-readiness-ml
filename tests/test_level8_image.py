"""Level 8 flat-image integration tests (L8-2D).

These run in the FLAT serving layout of a disposable image derived from the original Level 7 image
(committed F1 Level 7 service plus the committed Level 8 modules, by explicit allowlist). In the
repository layout they are NOT APPLICABLE and skip, unless LEVEL8_REQUIRE_IMAGE_TESTS=1, in which case
a repository layout is a failure (the sentinel makes them impossible to skip silently in an image build).

Golden values were produced by the repository-layout application and engine; they pin flat/repository
parity without sharing any code path with the flat image.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_LAYOUT = not (ROOT / "src" / "level8_service.py").is_file()
REQUIRED = os.environ.get("LEVEL8_REQUIRE_IMAGE_TESTS") == "1"

VALID = {"weekly_km_to_date": 32.0, "sleep_hours_last_night": 7.0, "resting_hr_morning": 50.0,
         "soreness_score_evening": 4.0, "prior_injury": 0}
FEATURES = ("weekly_km_to_date", "sleep_hours_last_night", "resting_hr_morning", "soreness_score_evening",
            "prior_injury")
GOLDEN_PROBABILITY = 0.49793776869773865
GOLDEN_LOGIT = -0.008249014616012573
GOLDEN_ATTRIBUTIONS = (-0.0034358110278844833, -0.3442489206790924, -0.09241339564323425,
                       -0.40909576416015625, -0.20982761681079865)
ALLOWED_FILES = {
    "level7_runtime.py", "level7_service.py", "level8_explain.py", "level8_worker.py", "level8_dispatch.py",
    "level8_schemas.py", "level8_service.py", "lineage-attestation.json", "serving-manifest.json",
    "level7-tests-passed", "artifacts/v4/level6_preprocessor.json", "artifacts/v4/level6_selected_model.pt",
    "artifacts/v4/level7_deploy_handoff.json",
}


def setUpModule():
    if not IMAGE_LAYOUT:
        if REQUIRED:
            raise AssertionError("LEVEL8_REQUIRE_IMAGE_TESTS=1 but the repository layout is in use")
        raise unittest.SkipTest("NOT APPLICABLE: repository layout (the flat-image tests run in the image)")


def wait_for(predicate, timeout=60.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def children_of(pid):
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                if int(Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()[1]) == pid:
                    found.append(int(entry))
            except (OSError, IndexError, ValueError):
                pass
    return found


def torch_globals(torch):
    return (torch.get_num_threads(), torch.get_num_interop_threads(), torch.are_deterministic_algorithms_enabled(),
            str(torch.get_default_dtype()), torch.is_grad_enabled())


class FlatImageContentTests(unittest.TestCase):
    def test_image_contains_only_the_allowlisted_serving_files(self):
        present = set()
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT).as_posix()
            if path.is_file() and not relative.startswith("tests/") and "__pycache__" not in relative:
                present.add(relative)
        self.assertEqual(present - ALLOWED_FILES, set(), "unexpected files in the image")
        self.assertTrue({"level8_service.py", "level7_service.py", "serving-manifest.json"} <= present)
        for forbidden in ("data", "src", "models", "build", "reports", ".git"):
            self.assertFalse((ROOT / forbidden).exists(), forbidden)
        self.assertEqual([p.name for p in ROOT.rglob("*.csv")], [])  # no dataset, no consumed holdout
        self.assertEqual(sorted(p.name for p in ROOT.rglob("*.pt")), ["level6_selected_model.pt"])
        self.assertFalse(list(ROOT.rglob("train*.py")))

    def test_committed_f1_level7_service_replaces_the_stale_original(self):
        import hashlib

        digest = hashlib.sha256((ROOT / "level7_service.py").read_bytes()).hexdigest()
        self.assertTrue(digest.startswith("022c275f"), digest)  # the committed F1 module, not 142a76a3...


class FlatImportAndRuntimeTests(unittest.TestCase):
    def run_python(self, code, env=None):
        merged = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env or {}))
        return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=merged, capture_output=True, text=True)

    def test_import_starts_nothing_and_resolves_the_serving_root(self):
        done = self.run_python(
            "import multiprocessing, threading, level7_service as l7, level8_service as s8;"
            "print(len(multiprocessing.active_children()), sorted(t.name for t in threading.enumerate()),"
            " l7.BASE_DIR, l7.runtime.integrity_status, s8.app.state.explanation_dispatcher)")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip().splitlines()[-1], f"0 ['MainThread'] {ROOT} passed None")

    def test_tampered_artifact_fails_the_parent_integrity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "app"
            shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns("tests", "__pycache__"))
            target = copy / "artifacts" / "v4" / "level6_preprocessor.json"
            target.write_bytes(target.read_bytes() + b" ")
            done = subprocess.run([sys.executable, "-c", "import level8_service"], cwd=copy, capture_output=True,
                                  text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("IntegrityError", done.stderr)


class FlatInProcessApplicationTests(unittest.TestCase):
    def test_real_worker_in_the_flat_layout(self):
        import torch
        from fastapi.testclient import TestClient

        import level7_service as l7
        import level8_explain
        import level8_service as s8

        before = torch_globals(torch)
        saved = torch.get_num_threads()
        torch.set_num_threads(1)  # the engine reference under the child's setting; restored right away
        try:
            reference = json.loads(json.dumps(level8_explain.ExplanationEngine(l7.runtime).explain(VALID).to_dict()))
        finally:
            torch.set_num_threads(saved)
        app = s8.build_app()
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            end, response = time.monotonic() + 90, None
            while time.monotonic() < end:
                response = client.post("/v1/explanations", json=VALID)
                if response.status_code != 503:
                    break
                time.sleep(0.2)
            self.assertEqual(response.status_code, 200, response.text)
            body = json.loads(response.text, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(c)))
            self.assertEqual(response.headers["cache-control"], "no-store")
            self.assertEqual(body["schema_version"], "explanation.v1")
            self.assertEqual([a["feature"] for a in body["attributions"]], list(FEATURES))
            self.assertEqual(body["attributions"], reference["attributions"])  # verified-engine parity, exact
            self.assertEqual(tuple(a["attribution"] for a in body["attributions"]), GOLDEN_ATTRIBUTIONS)
            self.assertEqual(body["prediction"]["probability"], GOLDEN_PROBABILITY)  # repository-layout parity
            self.assertEqual(body["prediction"]["logit"], GOLDEN_LOGIT)
            self.assertEqual(body["model"]["artifact_sha256"], dict(l7.runtime.manifest["artifacts"]))
            predict = client.post("/predict", json=VALID).json()
            self.assertEqual(predict["probability"], GOLDEN_PROBABILITY)
            self.assertLessEqual(abs(body["prediction"]["probability"] - predict["probability"]), 2e-6)
            info = app.state.explanation_dispatcher.worker_info()
            self.assertEqual((info["torch_threads"], info["torch_interop_threads"]), (1, 1))
            self.assertEqual(info["loaded_forbidden_modules"], [])  # no FastAPI/Uvicorn in the child
            self.assertEqual(torch_globals(torch), before)  # parent settings unchanged while serving
            # strict request and response contracts, sanitized operational errors
            with TestClient(l7.app) as legacy:
                for bad in ({}, dict(VALID, prior_injury=0.5), dict(VALID, weekly_km_to_date=201.0)):
                    new, old = client.post("/v1/explanations", json=bad), legacy.post("/predict", json=bad)
                    self.assertEqual((new.status_code, new.json()), (old.status_code, old.json()))
                self.assertEqual(client.get("/health").content, legacy.get("/health").content)
            domain = client.post("/v1/explanations", json=dict(VALID, weekly_km_to_date=19.0))
            self.assertEqual((domain.status_code, domain.json()["error"]["code"]), (422, "outside_explanation_domain"))
            self.assertNotIn("19.0", domain.text)
            self.assertEqual(domain.headers["cache-control"], "no-store")
        self.assertIs(app.state.explanation_cleanup_confirmed, True)
        self.assertEqual(multiprocessing.active_children(), [])
        self.assertFalse([t for t in threading.enumerate() if t.name == "level8-dispatcher"])
        self.assertEqual(torch_globals(torch), before)


class FlatUvicornTests(unittest.TestCase):
    def test_supported_launcher_serves_and_shuts_down_cleanly(self):
        import httpx

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        env = dict(os.environ, MPLCONFIGDIR=tempfile.mkdtemp(), PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "level8_service:app", "--host", "127.0.0.1",
                                 "--port", str(port), "--workers", "1", "--log-level", "info"],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.addCleanup(lambda: proc.poll() is None and (proc.kill(), proc.wait(5)))
        base = f"http://127.0.0.1:{port}"

        def healthy():
            try:
                return httpx.get(base + "/health", timeout=3).status_code == 200
            except httpx.HTTPError:
                return False

        self.assertTrue(wait_for(healthy, 90), Path(log.name).read_text()[-600:])
        response = None
        end = time.monotonic() + 90
        while time.monotonic() < end:
            response = httpx.post(base + "/v1/explanations", json=VALID, timeout=10)
            if response.status_code != 503:
                break
            time.sleep(0.2)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(tuple(a["attribution"] for a in response.json()["attributions"]), GOLDEN_ATTRIBUTIONS)
        self.assertEqual(httpx.post(base + "/predict", json=VALID, timeout=5).json()["probability"], GOLDEN_PROBABILITY)
        kids = children_of(proc.pid)
        cmdlines = {pid: Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace") for pid in kids}
        workers = [pid for pid, text in cmdlines.items() if "spawn_main" in text]
        self.assertEqual(len(workers), 1, cmdlines)  # one spawned worker, no recursive server
        self.assertNotIn("uvicorn", cmdlines[workers[0]])
        proc.send_signal(signal.SIGTERM)
        self.assertTrue(wait_for(lambda: proc.poll() is not None, 20))
        text = Path(log.name).read_text()
        self.assertIn(proc.returncode, (0, -signal.SIGTERM), text[-600:])
        self.assertIn("Application shutdown complete", text)
        self.assertNotIn("Traceback", text)
        self.assertTrue(wait_for(lambda: not any(alive(pid) for pid in kids), 10))


if __name__ == "__main__":
    unittest.main()
