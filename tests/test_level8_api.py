"""Level 8 L8-2C API integration tests (versioned FastAPI integration).

Layers: controlled dispatcher stubs (deterministic contract and error mapping), the real bounded
dispatcher with fake worker handles (deadlines, cancellation, disconnects, Level 7 availability
per worker state), real spawned workers in-process, and a real Uvicorn server (one worker, no
reload) in canonical Linux/amd64 Docker.

The Level 7 fixture serving root is reused from tests.test_level7_service, which also guarantees
the Level 7 module is imported with that root before the Level 8 app is composed.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import contextlib
import copy
import dataclasses
import json
import multiprocessing
import os
import select
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
from unittest import mock

import httpx
import torch
from fastapi.testclient import TestClient

from tests import test_level7_service as t7
from tests import level8_oracle as oracle
from tests import test_level8_dispatch as dt  # module reference only: its test classes are not imported here
from src import level8_compare as compare
from src import level8_dispatch as d
from src import level8_schemas as schemas
from src import level8_service as s8
from src import level8_worker as w
from src.level8_explain import ExplanationEngine, OutsideExplanationDomain

ROOT = Path(__file__).resolve().parents[1]
L7 = t7.service
VALID = dict(t7.VALID_REQUEST)
HASHES = dict(L7.runtime.manifest["artifacts"])
TIMINGS: dict = {}
_STATE: dict = {}


def setUpModule():
    saved = torch.get_num_threads()
    torch.set_num_threads(1)  # the worker child's setting; restored right after computing the references
    try:
        engine = ExplanationEngine(L7.runtime)
        _STATE["engine"] = engine
        _STATE["metadata"] = json.loads(json.dumps(engine.metadata()))
        _STATE["explanation"] = w._wire_form(engine.explain(VALID))
    finally:
        torch.set_num_threads(saved)


def tearDownModule():
    print("L8_API_TIMINGS", json.dumps(TIMINGS, sort_keys=True))


def wait_for(predicate, timeout=30.0, interval=0.005):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def torch_globals():
    return (torch.get_num_threads(), torch.get_num_interop_threads(), torch.are_deterministic_algorithms_enabled(),
            str(torch.get_default_dtype()), torch.is_grad_enabled())


def success_outcome():
    return d.Outcome("result", explanation=copy.deepcopy(_STATE["explanation"]))


def engine_outcome(payload):
    """A genuine, self-consistent engine result for exactly this input (distinct inputs -> distinct results)."""
    return d.Outcome("result", explanation=w._wire_form(_STATE["engine"].explain(payload)))


def error_outcome(code, violations=()):
    return d.Outcome("error", None, code, tuple(violations))


# --------------------------------------------------------------------------- stubs
class StubJob:
    def __init__(self, outcome, deadline_in=0.5):
        self.deadline = time.monotonic() + deadline_in
        self.future = concurrent.futures.Future()
        if outcome is not None:
            self.future.set_result(outcome)


class StubDispatcher:
    """Controlled dispatcher: submit() applies the committed domain validation, then returns a job
    whose outcome the test chose (or never resolves, when outcome is None)."""

    def __init__(self, outcome=None, submit_exc=None, metadata=None, outcome_fn=None, deadline=None):
        self.outcome, self.submit_exc = outcome, submit_exc
        self.outcome_fn, self.deadline = outcome_fn, deadline
        self._metadata = metadata if metadata is not None else _STATE["metadata"]
        self.started = self.stopped = False
        self.submitted, self.jobs, self.cancelled = [], [], []

    def start(self):
        self.started = True

    def shutdown(self, timeout=None):
        self.stopped = True

    def diagnostics(self):
        return {"state": "stopped"}

    def metadata(self):
        return self._metadata

    def submit(self, payload):
        ExplanationEngine.validate_payload(payload)  # same helper the real dispatcher applies first
        if self.submit_exc is not None:
            raise self.submit_exc
        self.submitted.append(payload)
        job = StubJob(self.outcome_fn(payload) if self.outcome_fn else self.outcome)
        if self.deadline is not None:
            job.deadline = self.deadline
        self.jobs.append(job)
        return job

    def cancel(self, job):
        self.cancelled.append(job)
        return job.future.cancel()


def stub_client(clock=None, **kwargs):
    stub = StubDispatcher(**kwargs)
    app = s8.build_app(lambda root, hashes: stub, clock=clock) if clock else s8.build_app(lambda root, hashes: stub)
    return TestClient(app), stub


def plain(response):
    return json.loads(response.text, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(c)))


def finite_numbers(node):
    if isinstance(node, bool):
        return True
    if isinstance(node, float):
        return node == node and node not in (float("inf"), float("-inf"))
    if isinstance(node, dict):
        return all(finite_numbers(v) for v in node.values())
    if isinstance(node, list):
        return all(finite_numbers(v) for v in node)
    return True


# --------------------------------------------------------------- response contract
class ResponseContractTests(unittest.TestCase):
    def test_success_response_schema_and_content(self):
        client, stub = stub_client(outcome=success_outcome())
        with client:
            response = client.post("/v1/explanations", json=VALID)
        self.assertEqual(response.status_code, 200)
        body = plain(response)
        parsed = schemas.ExplanationResponse.model_validate(body)
        self.assertEqual(body["schema_version"], "explanation.v1")
        self.assertEqual([a["feature"] for a in body["attributions"]], list(schemas.FEATURES))
        self.assertTrue(finite_numbers(body))
        reference = _STATE["explanation"]
        for got, want in zip(body["attributions"], reference["attributions"]):
            self.assertEqual(got, want)  # copied unchanged: not recomputed, not rounded
        self.assertEqual(body["completeness"]["target_difference"], reference["target_difference"])
        self.assertEqual(body["completeness"]["attribution_sum"], reference["attribution_sum"])
        self.assertEqual(body["completeness"]["residual"], reference["completeness_residual"])
        self.assertEqual(body["completeness"]["allowed_residual"], reference["allowed_residual"])
        self.assertLessEqual(abs(body["completeness"]["residual"]), body["completeness"]["allowed_residual"])
        self.assertEqual(body["method"], {
            "name": "integrated_gradients", "library": "captum", "library_version": "0.9.0",
            "integration": "riemann_middle", "n_steps": 800, "target": "pre_sigmoid_logit",
            "one_input_per_call": True})
        self.assertEqual(body["baseline"]["id"], "B0")
        self.assertEqual(body["baseline"]["reference"], "development_mean")
        self.assertEqual(list(body["baseline"]["standardized_values"].values()), [0.0] * 5)
        self.assertAlmostEqual(body["baseline"]["raw_values"]["prior_injury"], 1 / 6, places=9)
        self.assertEqual(body["model"]["artifact_sha256"], HASHES)
        self.assertIs(body["model"]["development_only"], True)
        self.assertEqual(body["model"]["handoff_version"], "4.0.0")
        self.assertEqual(body["prediction"]["threshold"], 0.5)
        self.assertAlmostEqual(body["prediction"]["decision_margin"], body["prediction"]["logit"], places=12)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(parsed.eligibility.status, "within_development_ranges")
        self.assertEqual(stub.submitted, [VALID])

    def test_notices_are_the_fixed_interpretation_notices(self):
        client, _ = stub_client(outcome=success_outcome())
        with client:
            notices = plain(client.post("/v1/explanations", json=VALID))["notices"]
        self.assertEqual(notices, list(schemas.INTERPRETATION_NOTICES))
        text = " ".join(notices)
        for phrase in ("needs_recovery_next_day", "not a general readiness score", "not causal", "development-mean",
                       "not a healthy-runner reference", "fractional prior_injury", "synthetic data",
                       "inherited preprocessing limitation", "not establish clinical", "in-distribution",
                       "real-world predictive validity", "not medical advice"):
            self.assertIn(phrase, text)

    def test_no_internal_details_in_success_responses(self):
        client, _ = stub_client(outcome=success_outcome())
        with client:
            text = client.post("/v1/explanations", json=VALID).text
        for leak in ("subtype", '"gen"', "request_id", '"pid"', "generation", "gate_failure", "worker", "ipc"):
            self.assertNotIn(leak, text)

    def test_malformed_worker_success_payloads_are_rejected_without_partial_results(self):
        def mutate(fn):
            outcome = success_outcome()
            fn(outcome.explanation)
            return outcome

        bad = {
            "four_attributions": mutate(lambda e: e["attributions"].pop()),
            "wrong_order": mutate(lambda e: e["attributions"].reverse()),
            "nan_attribution": mutate(lambda e: e["attributions"][0].update(attribution=float("nan"))),
            "inf_logit": mutate(lambda e: e.update(logit=float("inf"))),
            "extra_attribution_field": mutate(lambda e: e["attributions"][0].update(extra=1)),
            "missing_completeness": mutate(lambda e: e.pop("allowed_residual")),
            "threshold_zero": mutate(lambda e: e["prediction"].update(threshold=0.0)),
            "decision_out_of_range": mutate(lambda e: e["prediction"].update(decision=2)),
            "string_value": mutate(lambda e: e["attributions"][1].update(value="x")),
        }
        for label, outcome in bad.items():
            with self.subTest(case=label):
                client, _ = stub_client(outcome=outcome)
                with client:
                    response = client.post("/v1/explanations", json=VALID)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(plain(response)["error"]["code"], "explanation_unavailable")
                self.assertNotIn("attributions", response.text)

    def test_inconsistent_worker_metadata_is_rejected(self):
        for label, change in {
            "wrong_steps": lambda m: m["method"].update(n_steps=801),
            "wrong_integration": lambda m: m["method"].update(integration="gausslegendre"),
            "wrong_target": lambda m: m["method"].update(target="probability"),
            "other_artifacts": lambda m: m["model"].update(artifact_sha256={"x": "0" * 64}),
            "not_development_only": lambda m: m["model"].update(development_only=False),
            "baseline_order": lambda m: m["baseline"].update(
                raw_values=dict(reversed(list(m["baseline"]["raw_values"].items())))),
        }.items():
            with self.subTest(case=label):
                metadata = copy.deepcopy(_STATE["metadata"])
                change(metadata)
                client, _ = stub_client(outcome=success_outcome(), metadata=metadata)
                with client:
                    response = client.post("/v1/explanations", json=VALID)
                self.assertEqual((response.status_code, plain(response)["error"]["code"]),
                                 (503, "explanation_unavailable"))

    def test_openapi_documents_the_new_contract_and_preserves_level7(self):
        spec = s8.app.openapi()
        self.assertEqual((spec["info"]["title"], spec["info"]["version"]), ("Running Readiness ML API", "5.0.0"))
        self.assertEqual(set(spec["paths"]), {"/health", "/predict", "/v1/explanations", "/v1/comparisons"})
        legacy = L7.app.openapi()["paths"]
        self.assertEqual(spec["paths"]["/health"], legacy["/health"])
        self.assertEqual(spec["paths"]["/predict"], legacy["/predict"])
        operation = spec["paths"]["/v1/explanations"]["post"]
        self.assertEqual(set(operation["responses"]), {"200", "422", "503", "504"})
        self.assertIn("ExplanationResponse", json.dumps(operation["responses"]["200"]))
        self.assertIn("ErrorResponse", json.dumps(operation["responses"]["503"]))
        self.assertIn("PredictionRequest", json.dumps(operation["requestBody"]))
        schema_names = set(spec["components"]["schemas"])
        self.assertTrue({"ExplanationResponse", "ErrorResponse", "PredictionRequest"} <= schema_names)
        props = spec["components"]["schemas"]["PredictionRequest"]["properties"]
        self.assertEqual(set(props), set(schemas.FEATURES))  # no client-controlled baseline or steps
        self.assertEqual(spec["components"]["schemas"]["ExplanationResponse"]["properties"]["schema_version"]["const"],
                         "explanation.v1")


# ------------------------------------------------------------- request handling
class RequestHandlingTests(unittest.TestCase):
    def test_unresolved_job_is_answered_at_its_deadline_and_cancelled(self):
        client, stub = stub_client(outcome=None)  # the dispatcher never resolves the job
        original = StubJob.__init__

        def short(self, outcome, deadline_in=0.15):
            original(self, outcome, deadline_in)

        StubJob.__init__ = short
        self.addCleanup(setattr, StubJob, "__init__", original)
        with client:
            started = time.monotonic()
            response = client.post("/v1/explanations", json=VALID)
            elapsed = time.monotonic() - started
        self.assertEqual((response.status_code, plain(response)["error"]["code"]), (504, "explanation_deadline_exceeded"))
        self.assertLess(elapsed, 0.6)
        self.assertEqual(len(stub.cancelled), 1)  # HTTP asked the dispatcher to cancel; it released nothing itself
        TIMINGS["unresolved_job_http_s"] = round(elapsed, 4)

    def test_invalid_requests_match_level_7_exactly(self):
        invalid = [
            {}, dict(VALID, prior_injury=0.5), dict(VALID, prior_injury=True), dict(VALID, prior_injury="0"),
            dict(VALID, prior_injury=2), dict(VALID, weekly_km_to_date=201.0), dict(VALID, weekly_km_to_date="x"),
            dict(VALID, resting_hr_morning=29.9), dict(VALID, extra_field=1), dict(VALID, sleep_hours_last_night=None),
            {k: v for k, v in VALID.items() if k != "prior_injury"},
        ]
        client, stub = stub_client(outcome=success_outcome())
        with client, TestClient(L7.app) as legacy:
            for body in invalid:
                with self.subTest(body=str(body)[:60]):
                    new = client.post("/v1/explanations", json=body)
                    old = legacy.post("/predict", json=body)
                    self.assertEqual((new.status_code, new.json()), (old.status_code, old.json()))
            raw = client.post("/v1/explanations", content=b"{not json", headers={"content-type": "application/json"})
            old_raw = legacy.post("/predict", content=b"{not json", headers={"content-type": "application/json"})
            self.assertEqual((raw.status_code, raw.json()), (old_raw.status_code, old_raw.json()))
            nan = client.post("/v1/explanations", content=b'{"weekly_km_to_date": NaN}',
                              headers={"content-type": "application/json"})
            self.assertEqual(nan.status_code, 422)
        self.assertEqual(stub.submitted, [])  # invalid requests never reach the dispatcher

    def test_explanation_domain_boundaries_are_inclusive_and_checked_before_admission(self):
        client, stub = stub_client(outcome_fn=engine_outcome)
        ranges = {"weekly_km_to_date": (20.0, 45.0), "sleep_hours_last_night": (5.5, 8.0),
                  "resting_hr_morning": (45.0, 57.0), "soreness_score_evening": (0.0, 10.0), "prior_injury": (0, 1)}
        with client:
            for name, (lo, hi) in ranges.items():
                for value in (lo, hi):
                    self.assertEqual(client.post("/v1/explanations", json=dict(VALID, **{name: value})).status_code,
                                     200, (name, value))
            before = len(stub.submitted)
            for name, (lo, hi) in ranges.items():
                if name in ("soreness_score_evening", "prior_injury"):
                    continue
                for value, text in ((lo - 0.000001, "-0.000001"), (hi + 0.000001, "+0.000001")):
                    response = client.post("/v1/explanations", json=dict(VALID, **{name: value}))
                    body = plain(response)
                    self.assertEqual((response.status_code, body["error"]["code"]), (422, "outside_explanation_domain"))
                    self.assertEqual(body["error"]["violations"], [{"feature": name, "min": lo, "max": hi}])
                    self.assertNotIn(str(value), response.text)  # the submitted value is never echoed
            multi = client.post("/v1/explanations", json=dict(VALID, weekly_km_to_date=19.0, resting_hr_morning=58.0))
            self.assertEqual([v["feature"] for v in plain(multi)["error"]["violations"]],
                             ["weekly_km_to_date", "resting_hr_morning"])
            self.assertNotIn("19.0", multi.text)
            self.assertEqual(len(stub.submitted), before)  # domain rejections never reached admission
            schemas.ErrorResponse.model_validate(plain(multi))

    def test_unstarted_application_is_unavailable_but_level_7_works(self):
        app = s8.build_app(lambda root, hashes: StubDispatcher())
        client = TestClient(app)  # no lifespan: no dispatcher exists
        response = client.post("/v1/explanations", json=VALID)
        self.assertEqual((response.status_code, response.headers["retry-after"], plain(response)["error"]["code"]),
                         (503, "5", "explanation_unavailable"))
        self.assertEqual(client.post("/predict", json=VALID).status_code, 200)
        self.assertEqual(client.get("/health").status_code, 200)

    def test_operational_error_mappings_and_sanitization(self):
        domain = [{"feature": "weekly_km_to_date", "min": 20.0, "max": 45.0}]
        cases = [
            ({"submit_exc": d.AdmissionRejected("explanation_overloaded")}, 503, "explanation_overloaded", "1"),
            ({"submit_exc": d.AdmissionRejected("explanation_unavailable")}, 503, "explanation_unavailable", "5"),
            ({"outcome": error_outcome("explanation_numerical_check_failed")}, 503,
             "explanation_numerical_check_failed", None),
            ({"outcome": error_outcome("explanation_deadline_exceeded")}, 504, "explanation_deadline_exceeded", None),
            ({"outcome": error_outcome("explanation_unavailable")}, 503, "explanation_unavailable", "5"),
            ({"outcome": error_outcome("outside_explanation_domain", domain)}, 422, "outside_explanation_domain", None),
            ({"outcome": error_outcome("invalid_explanation_input")}, 503, "explanation_unavailable", "5"),
            ({"outcome": error_outcome("something_unknown")}, 503, "explanation_unavailable", "5"),
            ({"submit_exc": RuntimeError("secret path /Users/x and value 31.4159")}, 503, "explanation_unavailable", "5"),
        ]
        for kwargs, status, code, retry in cases:
            with self.subTest(code=code, status=status):
                client, _ = stub_client(**kwargs)
                with client:
                    response = client.post("/v1/explanations", json=VALID)
                self.assertEqual(response.status_code, status)
                body = plain(response)
                self.assertEqual(set(body), {"error"})
                self.assertEqual(body["error"]["code"], code)
                self.assertEqual(response.headers.get("retry-after"), retry)
                self.assertEqual(response.headers["cache-control"], "no-store")
                schemas.ErrorResponse.model_validate(body)
                for leak in ("secret", "/Users", "31.4159", "Traceback", "subtype", "attributions", "pid"):
                    self.assertNotIn(leak, response.text)
                self.assertEqual(body["error"]["message"], schemas.ERROR_MESSAGES[code])


# ------------------------------------------------------- Level 7 compatibility
class Level7CompatibilityTests(unittest.TestCase):
    def test_health_and_predict_responses_are_identical(self):
        client, _ = stub_client(outcome=success_outcome())
        with client, TestClient(L7.app) as legacy:
            for method, path, kwargs in [("get", "/health", {}), ("post", "/predict", {"json": VALID}),
                                         ("post", "/predict", {"json": dict(VALID, prior_injury=1)}),
                                         ("post", "/predict", {"json": dict(VALID, prior_injury=0.5)}),
                                         ("post", "/predict", {"json": {}}), ("post", "/predict", {"content": b"x"}),
                                         ("get", "/predict", {}), ("post", "/health", {"json": {}}),
                                         ("put", "/predict", {"json": VALID}), ("delete", "/health", {}),
                                         ("get", "/nope", {}), ("options", "/predict", {})]:
                with self.subTest(method=method, path=path, kw=str(kwargs)[:40]):
                    new, old = getattr(client, method)(path, **kwargs), getattr(legacy, method)(path, **kwargs)
                    self.assertEqual((new.status_code, new.content, new.headers.get("content-type"),
                                      new.headers.get("allow")),
                                     (old.status_code, old.content, old.headers.get("content-type"),
                                      old.headers.get("allow")))
            prediction = client.post("/predict", json=VALID).json()
            self.assertEqual(prediction["probability"], t7.GOLDEN_PROBABILITY)

    def test_level_7_application_is_unmodified_by_composition(self):
        before = (len(L7.app.routes), sorted(r.path for r in L7.app.routes), dict(L7.app.exception_handlers).keys(),
                  L7.app.title, L7.app.version)
        s8.build_app(lambda root, hashes: StubDispatcher())
        after = (len(L7.app.routes), sorted(r.path for r in L7.app.routes), dict(L7.app.exception_handlers).keys(),
                 L7.app.title, L7.app.version)
        self.assertEqual(before, after)
        self.assertNotIn("/v1/explanations", [r.path for r in L7.app.routes])
        self.assertEqual((L7.app.title, L7.app.version), ("Running Readiness Level 7", "4.0.0"))
        source = (ROOT / "src" / "level7_service.py").read_bytes()
        import hashlib

        self.assertTrue(hashlib.sha256(source).hexdigest().startswith("022c275f"))

    def test_artifact_integrity_failure_behaves_like_level_7(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            shutil.copytree(t7.SERVING_ROOT, root)
            target = root / "artifacts" / "v4" / "level6_preprocessor.json"
            target.write_bytes(target.read_bytes() + b" ")  # one byte changes its hash
            results = {}
            for module in ("src.level7_service", "src.level8_service"):
                env = dict(os.environ, SERVING_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1")
                done = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=ROOT, env=env,
                                      capture_output=True, text=True)
                results[module] = (done.returncode != 0, "IntegrityError" in done.stderr)
        self.assertEqual(results["src.level7_service"], (True, True))
        self.assertEqual(results["src.level8_service"], (True, True))

    def test_import_starts_no_process_or_thread(self):
        env = dict(os.environ, SERVING_ROOT=str(t7.SERVING_ROOT), PYTHONDONTWRITEBYTECODE="1")
        code = ("import multiprocessing, threading, src.level8_service as s;"
                "print(len(multiprocessing.active_children()), sorted(t.name for t in threading.enumerate()),"
                " s.app.state.explanation_dispatcher)")
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip().splitlines()[-1], "0 ['MainThread'] None")


class PredictionAvailabilityByWorkerStateTests(unittest.TestCase):
    """/predict and /health stay identical whatever the explanation worker is doing."""

    def setUp(self):
        self.handles, self.owners, self.log = [], set(), []

    def factory(self, **cfg):
        def make(root, hashes):
            def handle(gen):
                h = dt.BlockingFake(gen, d.RealClock(), self.log, self.owners)
                self.handles.append(h)
                return h

            self.disp = d.Dispatcher(str(root), d.DispatcherConfig(**cfg), expected_artifacts=hashes,
                                     handle_factory=handle)
            return self.disp

        return make

    def ready_message(self, gen):
        message = dt.ready_message(gen, artifacts=HASHES)
        message["metadata"] = _STATE["metadata"]
        return message

    def assert_level7_unchanged(self, client):
        self.assertEqual(client.get("/health").json(), TestClient(L7.app).get("/health").json())
        body = client.post("/predict", json=VALID)
        self.assertEqual((body.status_code, body.json()["probability"]), (200, t7.GOLDEN_PROBABILITY))

    def test_prediction_through_every_worker_state(self):
        app = s8.build_app(self.factory(max_restarts=0))
        with TestClient(app) as client:
            self.assertTrue(wait_for(lambda: self.handles and self.disp.diagnostics()["state"] == d.STARTING))
            self.assert_level7_unchanged(client)  # starting
            self.assertEqual(client.post("/v1/explanations", json=VALID).status_code, 503)
            self.handles[0].push(("message", self.ready_message(1)))
            self.assertTrue(wait_for(lambda: self.disp.diagnostics()["state"] == d.READY))
            self.assert_level7_unchanged(client)  # ready
            jobs = [self.disp.submit(VALID) for _ in range(9)]  # one active, eight waiting
            overloaded = client.post("/v1/explanations", json=VALID)
            self.assertEqual((overloaded.status_code, overloaded.headers["retry-after"]), (503, "1"))
            self.assertEqual(plain(overloaded)["error"]["code"], "explanation_overloaded")
            self.assert_level7_unchanged(client)  # overloaded
            self.handles[0].push(("closed", "exited"))  # recovery, with no restart budget left
            self.assertTrue(wait_for(lambda: self.disp.diagnostics()["state"] == d.OPEN))
            self.assert_level7_unchanged(client)  # recovering / unavailable
            self.assertEqual(client.post("/v1/explanations", json=VALID).status_code, 503)
            self.assertTrue(all(j.future.result(5).code == "explanation_unavailable" for j in jobs))
        self.assertIs(app.state.explanation_cleanup_confirmed, True)

    def test_prediction_while_replacement_is_starting(self):
        app = s8.build_app(self.factory())
        with TestClient(app) as client:
            self.assertTrue(wait_for(lambda: self.handles))
            self.handles[0].push(("message", self.ready_message(1)))
            self.assertTrue(wait_for(lambda: self.disp.diagnostics()["state"] == d.READY))
            self.handles[0].push(("closed", "exited"))
            self.assertTrue(wait_for(lambda: len(self.handles) == 2 and self.disp.diagnostics()["state"] == d.STARTING))
            self.assert_level7_unchanged(client)
            self.assertEqual(client.post("/v1/explanations", json=VALID).status_code, 503)


# -------------------------------------------- deadlines, cancellation, disconnects
class HttpDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.handles, self.owners, self.log = [], set(), []

    async def start(self, handle_cls=dt.BlockingFake, **cfg):
        def make(root, hashes):
            def handle(gen):
                h = handle_cls(gen, d.RealClock(), self.log, self.owners)
                self.handles.append(h)
                return h

            self.disp = d.Dispatcher(str(root), d.DispatcherConfig(**cfg), expected_artifacts=hashes,
                                     handle_factory=handle)
            return self.disp

        self.app = s8.build_app(make)
        stack = contextlib_async_stack()
        await stack.enter_async_context(self.app.router.lifespan_context(self.app))
        self.addAsyncCleanup(stack.aclose)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t")
        self.addAsyncCleanup(self.client.aclose)
        await self.until(lambda: self.handles and self.disp.diagnostics()["state"] == d.STARTING)
        self.push_ready(0)
        await self.until(lambda: self.disp.diagnostics()["state"] == d.READY)

    def push_ready(self, index):
        message = dt.ready_message(self.handles[index].generation, artifacts=HASHES)
        message["metadata"] = _STATE["metadata"]
        self.handles[index].push(("message", message))

    async def until(self, predicate, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            await asyncio.sleep(0.002)
        return predicate()

    def reply(self, handle, payload=VALID):
        explanation = engine_outcome(payload).explanation
        handle.push(("message", {"t": "result", "gen": handle.generation, "id": handle.sent[-1]["id"],
                                 "explanation": explanation}))

    def tasks(self):
        return {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}

    async def post_asgi(self, payload=VALID):
        """Raw ASGI call with a controllable disconnect, to observe real disconnect handling."""
        body = json.dumps(payload).encode()
        disconnect = asyncio.Event()
        sent, state = [], {"first": True}

        async def receive():
            if state["first"]:
                state["first"] = False
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "path": "/v1/explanations", "raw_path": b"/v1/explanations", "query_string": b"", "root_path": "",
                 "scheme": "http", "server": ("t", 80), "client": ("c", 1),
                 "headers": [(b"host", b"t"), (b"content-type", b"application/json"),
                             (b"content-length", str(len(body)).encode())]}
        task = asyncio.create_task(self.app(scope, receive, send))
        return task, sent, disconnect

    async def test_execution_stall_returns_504_promptly_and_recovery_is_separate(self):
        await self.start()
        baseline = self.tasks()
        started = time.monotonic()
        response = await self.client.post("/v1/explanations", json=VALID)
        http_done = time.monotonic() - started
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (504, "explanation_deadline_exceeded"))
        self.assertNotIn("attributions", response.text)
        await self.until(lambda: len(self.handles) == 2 and self.disp.diagnostics()["state"] == d.STARTING)
        replacement_started = time.monotonic() - started
        self.push_ready(1)
        await self.until(lambda: self.disp.diagnostics()["state"] == d.READY)
        ready_again = time.monotonic() - started
        self.assertLess(http_done, 0.5)  # bounded by the execution deadline here, never by recovery
        TIMINGS["stall"] = {"http_response_s": round(http_done, 4), "replacement_started_s": round(replacement_started, 4),
                            "replacement_ready_s": round(ready_again, 4)}
        self.assertEqual(self.tasks(), baseline)  # no leaked monitor tasks
        diag = self.disp.diagnostics()
        self.assertEqual((diag["active"], diag["queued"], diag["invariant_violations"]), (0, 0, 0))

    async def test_http_response_does_not_wait_for_slow_cleanup(self):
        class SlowTerminate(dt.BlockingFake):
            def terminate(self):
                time.sleep(1.0)  # a worker that takes a second to be reaped
                return super().terminate()

        await self.start(handle_cls=SlowTerminate)
        started = time.monotonic()
        response = await self.client.post("/v1/explanations", json=VALID)
        http_done = time.monotonic() - started
        self.assertEqual(response.status_code, 504)
        self.assertLess(http_done, 0.6)
        await self.until(lambda: len(self.handles) == 2, timeout=10)
        cleanup_visible = time.monotonic() - started
        self.assertGreaterEqual(cleanup_visible, 1.0)  # termination really was slow...
        TIMINGS["slow_cleanup"] = {"http_response_s": round(http_done, 4), "replacement_started_s": round(cleanup_visible, 4)}

    async def test_queued_request_is_bounded_and_invalidated_by_the_active_job_timeout(self):
        # The active job always has the earlier deadline, so its timeout (recovery) invalidates a
        # queued request an instant before that request's own deadline: unavailable, or, in a
        # narrow race, deadline-exceeded. Either way the HTTP wait is bounded by the request deadline.
        await self.start(exec_timeout=5.0)
        first = asyncio.create_task(self.client.post("/v1/explanations", json=VALID))
        await self.until(lambda: self.handles[0].sent)
        started = time.monotonic()
        second = await self.client.post("/v1/explanations", json=VALID)
        second_done = time.monotonic() - started
        self.assertIn((second.status_code, second.json()["error"]["code"]),
                      {(503, "explanation_unavailable"), (504, "explanation_deadline_exceeded")})
        self.assertLess(second_done, 0.6)
        first_response = await first
        self.assertEqual(first_response.status_code, 504)
        await self.until(lambda: self.disp.diagnostics()["state"] == d.STARTING)
        diag = self.disp.diagnostics()
        self.assertEqual((diag["active"], diag["queued"], diag["invariant_violations"]), (0, 0, 0))
        TIMINGS["queued_behind_stall"] = {"queued_http_response_s": round(second_done, 4),
                                          "code": second.json()["error"]["code"]}

    async def test_client_disconnect_keeps_the_active_slot_and_discards_the_late_result(self):
        await self.start(exec_timeout=5.0)
        baseline = self.tasks()
        task, sent, disconnect = await self.post_asgi()
        await self.until(lambda: self.handles[0].sent)
        started = time.monotonic()
        disconnect.set()
        await asyncio.wait_for(task, 2.0)
        reaction = time.monotonic() - started
        self.assertEqual(sent[0]["status"], 499)  # handler ended by the disconnect, not by a timeout
        self.assertLess(reaction, 0.2)
        diag = self.disp.diagnostics()
        self.assertEqual((diag["active"], diag["counters"]["cancelled_active"]), (1, 1))  # slot NOT released by HTTP
        self.assertEqual(self.tasks(), baseline)
        # fill to capacity: the abandoned job still occupies the active slot
        jobs = [self.disp.submit(VALID) for _ in range(8)]
        with self.assertRaises(d.AdmissionRejected):
            self.disp.submit(VALID)
        for job in jobs:  # remove the extras (queued cancellation releases capacity immediately)
            self.disp.cancel(job)
        self.reply(self.handles[0], VALID)  # the abandoned job's result arrives late
        await self.until(lambda: self.disp.diagnostics()["active"] == 0)
        other = dict(VALID, sleep_hours_last_night=6.5)
        second = asyncio.create_task(self.client.post("/v1/explanations", json=other))
        await self.until(lambda: len(self.handles[0].sent) == 2)
        self.reply(self.handles[0], other)
        response = await second
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["attributions"][1]["value"], 6.5)  # its own result, not the late one
        self.assertNotEqual(response.json()["attributions"][1]["value"], VALID["sleep_hours_last_night"])
        diag = self.disp.diagnostics()
        self.assertEqual((diag["active"], diag["queued"], diag["invariant_violations"]), (0, 0, 0))

    async def test_disconnect_racing_a_successful_completion_keeps_accounting_exact(self):
        await self.start(exec_timeout=5.0)
        baseline = self.tasks()
        statuses = []
        for i in range(6):
            payload = dict(VALID, sleep_hours_last_night=round(5.6 + 0.1 * i, 1))
            task, sent, disconnect = await self.post_asgi(payload)
            await self.until(lambda: len(self.handles[0].sent) == i + 1)
            self.reply(self.handles[0], payload)  # the result and the disconnect land together
            disconnect.set()
            await asyncio.wait_for(task, 2.0)
            statuses.append(sent[0]["status"])
            self.assertIn(sent[0]["status"], (200, 499))
            await self.until(lambda: self.disp.diagnostics()["active"] == 0)
        diag = self.disp.diagnostics()
        self.assertEqual((diag["active"], diag["queued"], diag["invariant_violations"]), (0, 0, 0))
        self.assertEqual(diag["counters"]["completed"], 6)  # every job completed exactly once
        self.assertLessEqual(diag["counters"].get("cancelled_active", 0), 6)
        self.assertEqual(self.tasks(), baseline)  # no monitor or handler task survives
        TIMINGS["disconnect_vs_success_statuses"] = sorted(statuses)

    async def test_handler_task_cancellation_cancels_the_job(self):
        await self.start(exec_timeout=5.0)
        baseline = self.tasks()
        task, sent, disconnect = await self.post_asgi()
        await self.until(lambda: self.handles[0].sent)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        diag = self.disp.diagnostics()
        self.assertEqual((diag["active"], diag["counters"]["cancelled_active"]), (1, 1))
        self.assertEqual(self.tasks() - baseline, set())
        self.reply(self.handles[0], VALID)
        await self.until(lambda: self.disp.diagnostics()["active"] == 0)

    async def test_successful_request_through_the_real_dispatcher_with_a_fake_worker(self):
        await self.start()
        baseline = self.tasks()
        pending = asyncio.create_task(self.client.post("/v1/explanations", json=VALID))
        await self.until(lambda: self.handles[0].sent)
        self.reply(self.handles[0], VALID)
        response = await pending
        self.assertEqual(response.status_code, 200)
        schemas.ExplanationResponse.model_validate(plain(response))
        self.assertEqual(self.tasks(), baseline)

    async def test_domain_rejection_never_reaches_admission_with_the_real_dispatcher(self):
        await self.start()
        response = await self.client.post("/v1/explanations", json=dict(VALID, weekly_km_to_date=19.0))
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (422, "outside_explanation_domain"))
        self.assertEqual(self.disp.diagnostics()["counters"].get("admitted", 0), 0)

    async def test_numerical_failure_hides_the_internal_subtype(self):
        await self.start()
        pending = asyncio.create_task(self.client.post("/v1/explanations", json=VALID))
        await self.until(lambda: self.handles[0].sent)
        self.handles[0].push(("message", {"t": "error", "gen": 1, "id": self.handles[0].sent[-1]["id"],
                                          "code": w.CODE_NUMERICAL, "subtype": w.SUBTYPE_GATE}))
        response = await pending
        self.assertEqual((response.status_code, response.json()["error"]["code"]), (503, "explanation_numerical_check_failed"))
        self.assertNotIn(w.SUBTYPE_GATE, response.text)
        self.assertNotIn("subtype", response.text)
        self.assertEqual(self.disp.diagnostics()["counters"]["numerical_subtype:gate_failure"], 1)


def contextlib_async_stack():
    import contextlib

    return contextlib.AsyncExitStack()


# --------------------------------------------------- real spawned worker, in process
class RealWorkerLifespanTests(unittest.TestCase):
    def wait_explanation(self, client, timeout=90):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            response = client.post("/v1/explanations", json=VALID)
            if response.status_code != 503:
                return response
            time.sleep(0.2)
        self.fail("explanation worker never became ready")

    def test_real_worker_end_to_end_parity_and_cleanup(self):
        before = torch_globals()
        for cycle in range(2):  # repeated startup/shutdown with fresh dispatchers
            app = s8.build_app()
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                response = self.wait_explanation(client)
                self.assertEqual(response.status_code, 200, response.text)
                body = plain(response)
                schemas.ExplanationResponse.model_validate(body)
                prediction = client.post("/predict", json=VALID).json()
                self.assertLessEqual(abs(body["prediction"]["probability"] - prediction["probability"]), 2e-6)
                self.assertEqual((body["prediction"]["decision"], body["prediction"]["threshold"]),
                                 (prediction["decision"], prediction["threshold"]))
                reference = _STATE["explanation"]
                self.assertEqual(body["attributions"], reference["attributions"])  # exact parity with the engine
                self.assertEqual(body["completeness"]["residual"], reference["completeness_residual"])
                self.assertEqual(body["model"]["artifact_sha256"], HASHES)
                self.assertEqual(torch_globals(), before)  # parent Torch settings unchanged while running
                self.assertTrue(finite_numbers(body))
            self.assertIs(app.state.explanation_cleanup_confirmed, True)
            self.assertEqual(multiprocessing.active_children(), [])
            self.assertFalse([t for t in threading.enumerate() if t.name == "level8-dispatcher"])
            self.assertEqual(torch_globals(), before)

    def test_readiness_message_size_is_within_the_ipc_limit(self):
        core = w.WorkerCore(L7.BASE_DIR, HASHES)
        size = len(w.encode_message(w.ready_message(core, 1), "to_parent"))
        TIMINGS["ready_message_bytes"] = size
        self.assertLess(size, w.MAX_MESSAGE_BYTES)
        self.assertLess(size, w.MAX_MESSAGE_BYTES // 4)


# ------------------------------------------------------------- real Uvicorn server
def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def children_of(pid):
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                fields = Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[1]) == pid:
                    found.append(int(entry))
            except (OSError, IndexError, ValueError):
                pass
    return found


def alive(pid):
    """Running process (a zombie awaiting a reaper that is not us does not count)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


class UvicornServer:
    def __init__(self, test, launcher, target=("src.level8_service:app",), extra_env=None):
        self.test, self.port = test, free_port()
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        env = dict(os.environ, SERVING_ROOT=str(t7.SERVING_ROOT), MPLCONFIGDIR=tempfile.mkdtemp(),
                   PYTHONDONTWRITEBYTECODE="1", **(extra_env or {}))
        self.proc = subprocess.Popen(
            [*launcher, *target, "--host", "127.0.0.1", "--port", str(self.port),
             "--workers", "1", "--log-level", "info"], cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.base = f"http://127.0.0.1:{self.port}"

    def get(self, path):
        return httpx.get(self.base + path, timeout=5)

    def post(self, path, payload):
        return httpx.post(self.base + path, json=payload, timeout=10)

    def wait_health(self, timeout=90):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                self.test.fail("uvicorn exited early: " + Path(self.log.name).read_text()[-800:])
            try:
                if self.get("/health").status_code == 200:
                    return True
            except httpx.HTTPError:
                time.sleep(0.1)
        return False

    def wait_explanations(self, timeout=90):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            response = self.post("/v1/explanations", VALID)
            if response.status_code != 503:
                return response
            time.sleep(0.2)
        return None

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        return self.proc.returncode


class RealUvicornTests(unittest.TestCase):
    def run_server_cycle(self, launcher, label):
        server = UvicornServer(self, launcher)
        self.addCleanup(server.stop)
        started = time.monotonic()
        self.assertTrue(server.wait_health(), Path(server.log.name).read_text()[-800:])
        health_after = time.monotonic() - started
        kids = children_of(server.proc.pid)
        explanation = server.wait_explanations()
        ready_after = time.monotonic() - started
        return server, kids, explanation, health_after, ready_after

    def test_one_worker_uvicorn_spawns_and_serves_with_clean_shutdown(self):
        for cycle in range(2):
            server, kids, response, health_after, ready_after = self.run_server_cycle(
                [sys.executable, "-m", "uvicorn"], "m")
            self.assertIsNotNone(response, Path(server.log.name).read_text()[-800:])
            self.assertEqual(response.status_code, 200, response.text)
            kids = children_of(server.proc.pid)
            cmdlines = {pid: Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace") for pid in kids}
            workers = [pid for pid, text in cmdlines.items() if "spawn_main" in text]
            tracker = [pid for pid, text in cmdlines.items() if "resource_tracker" in text]
            self.assertEqual(len(workers), 1, cmdlines)  # exactly one spawned explanation worker, no recursion
            self.assertEqual(sorted(workers + tracker), sorted(kids), cmdlines)  # plus multiprocessing's tracker
            self.assertNotIn("uvicorn", cmdlines[workers[0]])
            child = [workers[0]] + tracker
            body = response.json()
            schemas.ExplanationResponse.model_validate(body)
            prediction = server.post("/predict", VALID).json()
            self.assertLessEqual(abs(body["prediction"]["probability"] - prediction["probability"]), 2e-6)
            self.assertEqual(body["attributions"], _STATE["explanation"]["attributions"])
            self.assertEqual(server.get("/health").status_code, 200)
            self.assertEqual(server.post("/v1/explanations", dict(VALID, weekly_km_to_date=19.0)).status_code, 422)
            TIMINGS[f"uvicorn_cycle_{cycle}"] = {"health_s": round(health_after, 2), "explanations_ready_s": round(ready_after, 2)}
            code = server.stop()
            log = Path(server.log.name).read_text()
            # uvicorn re-raises the captured SIGTERM after a graceful shutdown, so -15 is the normal exit
            self.assertIn(code, (0, -signal.SIGTERM), log[-800:])
            self.assertIn("Application shutdown complete", log)  # the lifespan (dispatcher shutdown) ran
            self.assertTrue(wait_for(lambda: not any(alive(pid) for pid in child), 10))  # all gone
            self.assertNotIn("Traceback", log)

    def test_console_script_launch_is_observed(self):
        script = shutil.which("uvicorn")
        if script is None:
            self.skipTest("no uvicorn console script")
        server = UvicornServer(self, [script])
        self.addCleanup(server.stop)
        self.assertTrue(server.wait_health(), Path(server.log.name).read_text()[-800:])
        response = server.wait_explanations(timeout=30)
        TIMINGS["console_script_explanations"] = response.status_code if response is not None else "never_ready"
        self.assertEqual(server.post("/predict", VALID).status_code, 200)  # Level 7 is unaffected either way
        self.assertEqual(server.get("/health").status_code, 200)
        server.stop()


# ===================================================================== L8-2CR hardening
class ManualClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def wire(explanation_dict):
    return json.loads(json.dumps(explanation_dict))


class DeadlineEnforcementTests(unittest.TestCase):
    """A success is never accepted at or after its absolute job deadline (the job deadline is 0.5 here;
    the clock is controlled, so the boundary is exact and nothing depends on sleeping)."""

    def post(self, start, after_build=None, patch_target="build_response", outcome=None):
        clock = ManualClock(start)
        client, stub = stub_client(outcome=outcome or success_outcome(), clock=clock, deadline=0.5)
        original = getattr(schemas, patch_target)

        def advancing(*args, **kwargs):
            result = original(*args, **kwargs)
            if after_build is not None:
                clock.t = after_build
            return result

        with mock.patch.object(schemas, patch_target, advancing), client:
            response = client.post("/v1/explanations", json=VALID)
        return response, stub

    def assert_late(self, response, stub):
        self.assertEqual((response.status_code, plain(response)["error"]["code"]), (504, "explanation_deadline_exceeded"))
        self.assertEqual(set(plain(response)), {"error"})
        self.assertNotIn("attributions", response.text)  # no partial explanation
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(stub.cancelled, [])  # the job had completed: nothing to cancel, nothing cancelled directly
        self.assertTrue(stub.jobs[0].future.done() and not stub.jobs[0].future.cancelled())

    def test_success_before_the_deadline_is_accepted(self):
        for finish in (0.0, 0.25, 0.4999):
            with self.subTest(finish=finish):
                response, stub = self.post(0.0, after_build=finish)
                self.assertEqual(response.status_code, 200)
                schemas.ExplanationResponse.model_validate(plain(response))

    def test_success_exactly_at_the_deadline_is_rejected(self):
        self.assert_late(*self.post(0.0, after_build=0.5))

    def test_success_after_the_deadline_is_rejected(self):
        for finish in (0.5000001, 0.6, 30.0):
            with self.subTest(finish=finish):
                self.assert_late(*self.post(0.0, after_build=finish))

    def test_already_resolved_success_with_an_expired_deadline_is_rejected(self):
        self.assert_late(*self.post(5.0))

    def test_deadline_expiring_during_response_validation_is_rejected(self):
        self.assert_late(*self.post(0.0, after_build=0.6, patch_target="verify_consistency"))
        response, _ = self.post(0.0, after_build=0.4, patch_target="verify_consistency")
        self.assertEqual(response.status_code, 200)

    def test_deadline_expiring_during_response_construction_is_rejected(self):
        self.assert_late(*self.post(0.0, after_build=0.5001))

    def test_a_rejected_late_success_never_reaches_another_request(self):
        clock = ManualClock(0.0)
        client, stub = stub_client(outcome_fn=engine_outcome, clock=clock, deadline=0.5)
        original = schemas.build_response

        def late(*args, **kwargs):
            result = original(*args, **kwargs)
            clock.t = 0.9
            return result

        with client:
            with mock.patch.object(schemas, "build_response", late):
                first = client.post("/v1/explanations", json=dict(VALID, sleep_hours_last_night=6.0))
            self.assertEqual(first.status_code, 504)
            self.assertNotIn("6.0", first.text)
            clock.t = 0.0
            second = client.post("/v1/explanations", json=dict(VALID, sleep_hours_last_night=6.5))  # its own result
        self.assertEqual(second.status_code, 200)
        self.assertEqual(plain(second)["attributions"][1]["value"], 6.5)
        self.assertNotIn("6.0", [a["value"] for a in plain(second)["attributions"]])


class TrustedReferenceTests(unittest.TestCase):
    def test_reference_is_built_once_without_touching_parent_state(self):
        import hashlib

        def digest():
            h = hashlib.sha256()
            for key, tensor in sorted(L7.runtime.model.state_dict().items()):
                h.update(key.encode())
                h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        before = (torch_globals(), digest(), L7.runtime.model.training,
                  all(p.grad is None for p in L7.runtime.model.parameters()),
                  all(p.requires_grad for p in L7.runtime.model.parameters()))
        reference = schemas.make_reference(L7.runtime)
        after = (torch_globals(), digest(), L7.runtime.model.training,
                 all(p.grad is None for p in L7.runtime.model.parameters()),
                 all(p.requires_grad for p in L7.runtime.model.parameters()))
        self.assertEqual(before, after)
        self.assertEqual(reference.threshold, L7.runtime.threshold)
        self.assertEqual(reference.artifacts, HASHES)
        self.assertEqual((reference.selected_approach, reference.handoff_version, reference.development_only),
                         ("partial", "4.0.0", True))
        self.assertAlmostEqual(reference.baseline_logit, _STATE["metadata"]["baseline"]["logit"], delta=1e-5)
        self.assertAlmostEqual(reference.baseline_probability, schemas.stable_sigmoid(reference.baseline_logit), places=15)
        self.assertEqual(tuple(abs(a - b) < 1e-12 for a, b in zip(reference.means,
                         _STATE["metadata"]["baseline"]["raw_values"].values())), (True,) * 5)

    def test_reference_failure_disables_explanations_but_not_level_7(self):
        app = s8.build_app(lambda root, hashes: StubDispatcher(outcome=success_outcome()))
        with mock.patch.object(schemas, "make_reference", side_effect=RuntimeError("boom")):
            with TestClient(app) as client:
                response = client.post("/v1/explanations", json=VALID)
                self.assertEqual((response.status_code, plain(response)["error"]["code"]), (503, "explanation_unavailable"))
                self.assertNotIn("boom", response.text)
                self.assertEqual(client.post("/predict", json=VALID).status_code, 200)


def shift_baseline(explanation, metadata, delta):
    """Consistently move the worker's baseline logit (and everything derived from it) by delta."""
    metadata["baseline"]["logit"] += delta
    metadata["baseline"]["probability"] = schemas.stable_sigmoid(metadata["baseline"]["logit"])
    explanation["target_difference"] -= delta
    explanation["completeness_residual"] += delta
    explanation["allowed_residual"] = 0.002 + 0.005 * abs(explanation["target_difference"])


def shift_attribution(explanation, delta):
    """A consistent but genuinely wrong attribution: sum and residual follow the changed value."""
    explanation["attributions"][0]["attribution"] += delta
    explanation["attribution_sum"] += delta
    explanation["completeness_residual"] += delta


class SemanticValidationTests(unittest.TestCase):
    """Every structurally valid but semantically inconsistent worker payload is rejected with the sanitized
    unavailable response; ordinary floating-point differences are accepted."""

    def post(self, fix_e=None, fix_m=None, payload=None):
        explanation, metadata = copy.deepcopy(_STATE["explanation"]), copy.deepcopy(_STATE["metadata"])
        if fix_e:
            fix_e(explanation)
        if fix_m:
            fix_m(metadata, explanation)
        client, _ = stub_client(outcome=d.Outcome("result", explanation=explanation), metadata=metadata)
        with client:
            return client.post("/v1/explanations", json=payload or VALID)

    def rejected(self, label, fix_e=None, fix_m=None):
        with self.subTest(case=label):
            response = self.post(fix_e, fix_m)
            self.assertEqual((response.status_code, plain(response)["error"]["code"]), (503, "explanation_unavailable"), label)
            self.assertEqual((response.headers["retry-after"], response.headers["cache-control"]), ("5", "no-store"))
            self.assertNotIn("attributions", response.text)
            self.assertNotIn("31.4159", response.text)
            schemas.ErrorResponse.model_validate(plain(response))

    def accepted(self, label, fix_e=None, fix_m=None):
        with self.subTest(case=label):
            response = self.post(fix_e, fix_m)
            self.assertEqual(response.status_code, 200, (label, response.text[:200]))

    def test_inconsistent_prediction_blocks_are_rejected(self):
        self.rejected("decision_flipped", lambda e: e["prediction"].update(decision=1))
        self.rejected("decision_string", lambda e: e["prediction"].update(decision="0"))
        self.rejected("threshold_0.3", lambda e: e["prediction"].update(threshold=0.3))
        self.rejected("threshold_not_exact", lambda e: e["prediction"].update(threshold=0.5000001))
        self.rejected("probability_0.9", lambda e: e["prediction"].update(probability=0.9))
        self.rejected("probability_off_by_3e-6", lambda e: e["prediction"].update(
            probability=e["prediction"]["probability"] + 3e-6))
        self.rejected("logit_changed", lambda e: e.update(logit=e["logit"] + 0.1))
        self.rejected("prediction_approach", lambda e: e["prediction"].update(selected_approach="other"))
        self.rejected("prediction_device", lambda e: e["prediction"].update(device="cuda"))

    def test_inconsistent_completeness_is_rejected(self):
        self.rejected("attribution_sum_123", lambda e: e.update(attribution_sum=123.0))
        self.rejected("attribution_sum_off_1e-6", lambda e: e.update(attribution_sum=e["attribution_sum"] + 1e-6))
        self.rejected("target_difference_5", lambda e: e.update(target_difference=5.0))
        self.rejected("target_difference_off_1e-6", lambda e: e.update(target_difference=e["target_difference"] + 1e-6))
        self.rejected("residual_0.5", lambda e: e.update(completeness_residual=0.5))
        self.rejected("residual_off_1e-6", lambda e: e.update(completeness_residual=e["completeness_residual"] + 1e-6))
        self.rejected("residual_outside_tolerance_but_arithmetically_consistent", lambda e: shift_attribution(e, 0.1))
        self.rejected("allowed_relaxed_5", lambda e: e.update(allowed_residual=5.0))
        self.rejected("allowed_off_1e-6", lambda e: e.update(allowed_residual=e["allowed_residual"] + 1e-6))
        self.rejected("value_does_not_echo_request", lambda e: e["attributions"][0].update(value=99.0))
        self.rejected("standardized_value_wrong", lambda e: e["attributions"][2].update(
            standardized_value=e["attributions"][2]["standardized_value"] + 1e-3))
        self.rejected("order_swapped", lambda e: e["attributions"].reverse())

    def test_inconsistent_identity_and_baseline_metadata_is_rejected(self):
        fixes = {
            "model_approach": lambda m, e: m["model"].update(selected_approach="other"),
            "handoff_version": lambda m, e: m["model"].update(handoff_version="9.9.9"),
            "not_development_only": lambda m, e: m["model"].update(development_only=False),
            "integrity_not_passed": lambda m, e: m["model"].update(integrity_status="failed"),
            "metadata_threshold": lambda m, e: m["model"].update(threshold=0.3),
            "artifact_hashes": lambda m, e: m["model"].update(artifact_sha256={"x": "0" * 64}),
            "baseline_raw_value": lambda m, e: m["baseline"]["raw_values"].update(weekly_km_to_date=1.0),
            "baseline_prior_injury_not_fractional": lambda m, e: m["baseline"]["raw_values"].update(prior_injury=0.0),
            "baseline_standardized_nonzero": lambda m, e: m["baseline"]["standardized_values"].update(sleep_hours_last_night=1e-3),
            "baseline_logit_far": lambda m, e: shift_baseline(e, m, 1e-4),
            "baseline_logit_inconsistent_with_target": lambda m, e: m["baseline"].update(logit=m["baseline"]["logit"] + 1e-4),
            "baseline_probability": lambda m, e: m["baseline"].update(probability=0.5),
            "baseline_id": lambda m, e: m["baseline"].update(id="B1"),
            "baseline_kind": lambda m, e: m["baseline"].update(kind="other"),
            "method_version": lambda m, e: m["method"].update(library_version="0.8.0"),
            "method_name": lambda m, e: m["method"].update(name="gradient_shap"),
            "method_integration": lambda m, e: m["method"].update(integration="gausslegendre"),
            "method_steps": lambda m, e: m["method"].update(n_steps=801),
            "method_target": lambda m, e: m["method"].update(target="probability"),
            "units": lambda m, e: m.update(units="probability points"),
            "eligibility_ranges": lambda m, e: m["explanation_ranges"].update(weekly_km_to_date=[10.0, 45.0]),
            "completeness_tolerance": lambda m, e: m["completeness_tolerance"].update(absolute=5.0),
        }
        for label, fix in fixes.items():
            self.rejected(label, fix_m=fix)

    def test_ordinary_floating_point_differences_are_accepted(self):
        self.accepted("probability_off_by_1e-6", lambda e: e["prediction"].update(probability=e["prediction"]["probability"] + 1e-6))
        self.accepted("attribution_sum_off_1e-12", lambda e: e.update(attribution_sum=e["attribution_sum"] + 1e-12))
        self.accepted("target_difference_off_1e-12", lambda e: e.update(target_difference=e["target_difference"] + 1e-12))
        self.accepted("residual_off_1e-12", lambda e: e.update(completeness_residual=e["completeness_residual"] + 1e-12))
        self.accepted("allowed_off_1e-13", lambda e: e.update(allowed_residual=e["allowed_residual"] + 1e-13))
        self.accepted("baseline_logit_off_1e-6_consistently", None, lambda m, e: shift_baseline(e, m, 1e-6))
        self.accepted("baseline_raw_off_1e-13", None, lambda m, e: m["baseline"]["raw_values"].update(
            weekly_km_to_date=m["baseline"]["raw_values"]["weekly_km_to_date"] + 1e-13))
        self.accepted("standardized_off_1e-8", lambda e: e["attributions"][2].update(
            standardized_value=e["attributions"][2]["standardized_value"] + 1e-8))
        self.accepted("unmodified", None, None)

    def test_valid_real_engine_payloads_across_the_development_grid_are_accepted(self):
        grids = oracle.build_grids(L7.runtime, VALID)
        engine = _STATE["engine"]
        payloads = [{n: (int(v) if n == "prior_injury" else float(v)) for n, v in zip(schemas.FEATURES, row)}
                    for row in list(grids["dev_range"][:20]) + list(grids["straddle_threshold"][:5])]
        client, stub = stub_client(outcome_fn=lambda p: d.Outcome("result", explanation=w._wire_form(engine.explain(p))))
        with client:
            for payload in payloads:
                response = client.post("/v1/explanations", json=payload)
                self.assertEqual(response.status_code, 200, (payload, response.text[:200]))
                body = plain(response)
                self.assertEqual([a["value"] for a in body["attributions"]],
                                 [float(payload[n]) for n in schemas.FEATURES])
        self.assertEqual(len(stub.submitted), len(payloads))

    def test_semantic_checks_do_not_repair_values(self):
        explanation = copy.deepcopy(_STATE["explanation"])
        explanation["attributions"][1]["attribution"] += 1e-13  # tiny, still consistent within tolerance
        client, _ = stub_client(outcome=d.Outcome("result", explanation=explanation))
        with client:
            body = plain(client.post("/v1/explanations", json=VALID))
        self.assertEqual(body["attributions"][1]["attribution"], explanation["attributions"][1]["attribution"])


class TracedHandle(d.ProcessWorkerHandle):
    """Real spawned worker with monotonic instrumentation. When ``stall`` is set, the (real) child's reply
    is withheld by this parent-side handle after dispatch: the child demonstrably finishes executing (its
    reply becomes readable) and the dispatcher then sees a communication/execution timeout. This verifies
    timeout handling with a real spawned process; it is NOT a genuinely hung model computation."""

    def __init__(self, generation, root, hashes, trace, stall):
        super().__init__(generation, root, hashes)
        self.trace, self.stall, self.dispatched, self.observed = trace, stall, False, False

    def start(self):
        self.trace(f"start_g{self.generation}")
        super().start()

    def send(self, message, deadline):
        super().send(message, deadline)
        if message["t"] == "explain":
            self.dispatched = True
            self.trace("dispatch")

    def wait(self, deadline, waker):
        if self.stall and self.dispatched:
            reply_fd = self._conns[1].fileno()
            while True:
                fds = [waker.fileno()] + ([] if self.observed else [reply_fd])
                ready, _, _ = select.select(fds, [], [], 0.005)
                if reply_fd in ready and not self.observed:
                    self.observed = True
                    self.trace("child_finished_reply_withheld")
                if waker.is_set():
                    return ("wake",)
                if time.monotonic() >= deadline:
                    self.trace("exec_timeout")
                    return ("timeout",)
        event = super().wait(deadline, waker)
        if event[0] == "message" and event[1]["t"] == "ready":
            self.trace(f"ready_g{self.generation}")
        return event

    def terminate(self):
        self.trace(f"terminate_start_g{self.generation}")
        confirmed = super().terminate()
        self.trace(f"terminate_end_g{self.generation}")
        return confirmed


class RecordingDispatcher(d.Dispatcher):
    def submit(self, payload):
        job = super().submit(payload)
        index = len(self.jobs) + 1
        self.jobs.append(job)
        self.times[f"admission_{index}"] = job.admitted_at
        job.future.add_done_callback(lambda f, i=index: self.times.setdefault(f"future_done_{i}", time.monotonic()))
        return job


class RealWorkerHttpTimeoutTests(unittest.IsolatedAsyncioTestCase):
    def tasks(self):
        return {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}

    async def until(self, predicate, timeout=90.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            await asyncio.sleep(0.005)
        return predicate()

    async def test_withheld_reply_from_a_real_worker_gives_504_then_real_recovery(self):
        ts = {}

        def trace(name):
            ts.setdefault(name, time.monotonic())

        def make(root, hashes):
            disp = RecordingDispatcher(
                str(root), d.DispatcherConfig(), expected_artifacts=hashes,
                handle_factory=lambda gen: TracedHandle(gen, str(root), hashes, trace, stall=(gen == 1)))
            disp.jobs, disp.times = [], ts
            self.disp = disp
            return disp

        app = s8.build_app(make)
        globals_before, tasks_before = torch_globals(), self.tasks()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
                self.assertTrue(await self.until(lambda: self.disp.diagnostics()["state"] == d.READY))
                old_pid = self.disp.worker_info()["pid"]
                response = await client.post("/v1/explanations", json=VALID)
                ts["http_response_1"] = time.monotonic()
                self.assertEqual((response.status_code, response.json()["error"]["code"]),
                                 (504, "explanation_deadline_exceeded"))
                self.assertEqual(set(response.json()), {"error"})
                self.assertNotIn("attributions", response.text)
                recovering_state = self.disp.diagnostics()["state"]
                prediction = await client.post("/predict", json=VALID)  # Level 7 stays available during recovery
                self.assertEqual((prediction.status_code, prediction.json()["probability"]), (200, t7.GOLDEN_PROBABILITY))
                self.assertTrue(await self.until(
                    lambda: self.disp.diagnostics()["state"] == d.READY and self.disp.diagnostics()["generation"] == 2))
                second = await client.post("/v1/explanations", json=VALID)
                ts["http_response_2"] = time.monotonic()
                self.assertEqual(second.status_code, 200, second.text)
                self.assertEqual(second.json()["attributions"], _STATE["explanation"]["attributions"])
                self.assertEqual(second.headers["cache-control"], "no-store")
                self.assertEqual(self.disp.worker_info()["generation"], 2)
                self.assertNotEqual(self.disp.worker_info()["pid"], old_pid)
        # ordering: the child really executed before the stall was induced; HTTP never waited for recovery
        order = ["admission_1", "dispatch", "child_finished_reply_withheld", "exec_timeout", "future_done_1"]
        values = [ts[name] for name in order]
        self.assertEqual(values, sorted(values), {k: round(v - ts["admission_1"], 4) for k, v in ts.items()})
        self.assertLessEqual(ts["future_done_1"], ts["terminate_start_g1"])  # dispatcher answers before reaping
        self.assertLess(ts["terminate_end_g1"], ts["start_g2"])  # old worker gone before the replacement starts
        self.assertLess(ts["start_g2"], ts["ready_g2"])
        self.assertLess(ts["http_response_1"], ts["ready_g2"])  # the 504 did not wait for replacement readiness
        self.assertIn(recovering_state, (d.RESTARTING, d.STARTING))
        self.assertFalse(alive(old_pid))
        self.assertEqual(multiprocessing.active_children(), [])
        self.assertFalse([t for t in threading.enumerate() if t.name == "level8-dispatcher"])
        self.assertTrue(all(j.future.done() for j in self.disp.jobs))
        diag = self.disp.diagnostics()
        self.assertEqual((diag["active"], diag["queued"], diag["invariant_violations"]), (0, 0, 0))
        self.assertTrue(app.state.explanation_cleanup_confirmed)
        self.assertEqual(self.tasks(), tasks_before)
        self.assertEqual(torch_globals(), globals_before)
        base = ts["admission_1"]
        TIMINGS["real_stall"] = {name: round(value - base, 4) for name, value in sorted(ts.items(), key=lambda kv: kv[1])}


_UV: dict = {"sent": 0}


class HoldFirstReplyHandle(d.ProcessWorkerHandle):
    """Test-only parent instrumentation for the real loopback disconnect test: the first explanation
    is genuinely dispatched to the real child, but its reply is held back for HOLD seconds, so the
    request is observably 'executing' while the client disconnects."""

    HOLD = 0.2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold_until = None

    def send(self, message, deadline):
        super().send(message, deadline)
        if message["t"] == "explain":
            _UV["sent"] += 1
            if _UV["sent"] == 1:
                self.hold_until = time.monotonic() + self.HOLD

    def wait(self, deadline, waker):
        while self.hold_until is not None and time.monotonic() < self.hold_until:
            ready, _, _ = select.select([waker.fileno()], [], [], 0.005)
            if ready:
                return ("wake",)
        self.hold_until = None
        return super().wait(deadline, waker)


def uvicorn_test_app():
    """Uvicorn --factory target (test-only): the production service plus instrumentation that lives
    outside the production API (a /__test/diag route and a lifespan wrapper that records Torch state)."""

    def dispatcher_factory(root, hashes):
        disp = d.Dispatcher(str(root), expected_artifacts=hashes,
                            handle_factory=lambda gen: HoldFirstReplyHandle(gen, str(root), hashes))
        _UV["dispatcher"] = disp
        return disp

    app = s8.build_app(dispatcher_factory)
    inner = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def wrapped(application):
        _UV["torch_at_startup"] = torch_globals()
        async with inner(application):
            yield
        Path(os.environ["LEVEL8_TEST_TORCH_REPORT"]).write_text(json.dumps({
            "at_startup": list(_UV["torch_at_startup"]), "after_shutdown": list(torch_globals()),
            "cleanup_confirmed": application.state.explanation_cleanup_confirmed}))

    app.router.lifespan_context = wrapped

    @app.get("/__test/diag")
    async def diag():
        disp = _UV.get("dispatcher")
        monitors = [t for t in asyncio.all_tasks() if getattr(t.get_coro(), "__name__", "") == "_watch_disconnect"]
        return {"dispatcher": disp.diagnostics() if disp else None, "sent": _UV["sent"], "monitors": len(monitors),
                "torch": list(torch_globals()), "torch_at_startup": list(_UV.get("torch_at_startup") or ()),
                "worker": disp.worker_info() if disp else None}

    return app


class RealLoopbackDisconnectTests(unittest.TestCase):
    def diag(self, server):
        return server.get("/__test/diag").json()

    def poll(self, server, predicate, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            snapshot = self.diag(server)
            if predicate(snapshot):
                return snapshot
            time.sleep(0.005)
        self.fail("condition not reached: " + json.dumps(self.diag(server))[:600])

    def test_real_client_disconnect_while_executing_over_loopback(self):
        report = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        server = UvicornServer(self, [sys.executable, "-m", "uvicorn"],
                               target=("tests.test_level8_api:uvicorn_test_app", "--factory"),
                               extra_env={"LEVEL8_TEST_TORCH_REPORT": report})
        self.addCleanup(server.stop)
        self.assertTrue(server.wait_health(), Path(server.log.name).read_text()[-800:])
        ready = self.poll(server, lambda s: s["dispatcher"] and s["dispatcher"]["state"] == "ready", 90)
        # Uvicorn parent and child Torch settings
        self.assertEqual(ready["torch"], ready["torch_at_startup"])
        self.assertEqual((ready["worker"]["torch_threads"], ready["worker"]["torch_interop_threads"]), (1, 1))
        self.assertEqual(ready["worker"]["loaded_forbidden_modules"], [])
        self.assertEqual((ready["monitors"], ready["sent"]), (0, 0))
        # a real HTTP request, then an abrupt client disconnect while it is executing
        body = json.dumps(VALID).encode()
        sock = socket.create_connection(("127.0.0.1", server.port))
        sock.sendall(b"POST /v1/explanations HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        self.poll(server, lambda s: s["sent"] == 1, 10)  # dispatched to the real child
        executing = self.diag(server)
        self.assertEqual(executing["dispatcher"]["active"], 1)
        self.assertGreaterEqual(executing["monitors"], 1)  # the disconnect monitor is running
        sock.close()
        held = self.poll(server, lambda s: s["dispatcher"]["counters"].get("cancelled_active", 0) == 1, 5)
        self.assertEqual(held["dispatcher"]["active"], 1)  # the slot is still reserved after the disconnect
        self.assertEqual(held["dispatcher"]["invariant_violations"], 0)
        self.assertEqual(server.get("/health").status_code, 200)  # Level 7 stays available
        self.assertEqual(server.post("/predict", VALID).json()["probability"], t7.GOLDEN_PROBABILITY)
        self.poll(server, lambda s: s["monitors"] == 0, 5)  # the monitor task terminated
        # capacity under load while the abandoned job still holds the slot, with distinct payloads
        payloads = [dict(VALID, sleep_hours_last_night=round(5.6 + 0.1 * i, 1)) for i in range(12)]
        results = [None] * len(payloads)

        def fire(i):
            results[i] = server.post("/v1/explanations", payloads[i])

        threads = [threading.Thread(target=fire, args=(i,)) for i in range(len(payloads))]
        [t.start() for t in threads]
        [t.join(30) for t in threads]
        codes = [r.status_code for r in results]
        overloaded = sum(1 for r in results if r.status_code == 503 and r.json()["error"]["code"] == "explanation_overloaded")
        self.assertTrue(set(codes) <= {200, 503, 504}, codes)
        for payload, response in zip(payloads, results):
            if response.status_code == 200:  # each success carries its OWN input: no late/foreign result
                self.assertEqual(response.json()["attributions"][1]["value"], payload["sleep_hours_last_night"])
                self.assertNotEqual(response.json()["attributions"][1]["value"], VALID["sleep_hours_last_night"])
                self.assertEqual(response.headers["cache-control"], "no-store")
        final = self.poll(server, lambda s: s["dispatcher"]["active"] == 0 and s["dispatcher"]["queued"] == 0, 20)
        self.assertEqual(final["monitors"], 0)
        self.assertEqual(final["dispatcher"]["generation"], 1)  # no restart was needed
        self.assertEqual((final["dispatcher"]["invariant_violations"], final["dispatcher"]["counters"]["cancelled_active"]), (0, 1))
        self.assertLessEqual(final["dispatcher"]["high_water"]["admitted"], 9)
        self.assertEqual(final["torch"], final["torch_at_startup"])  # parent Torch settings unchanged while serving
        TIMINGS["loopback_disconnect"] = {"codes": sorted(codes), "overloaded": overloaded,
                                          "high_water": final["dispatcher"]["high_water"]}
        code = server.stop()
        log = Path(server.log.name).read_text()
        self.assertIn(code, (0, -signal.SIGTERM), log[-800:])
        self.assertIn("Application shutdown complete", log)
        torch_report = json.loads(Path(report).read_text())
        self.assertEqual(torch_report["after_shutdown"], torch_report["at_startup"])  # parent Torch after shutdown
        self.assertTrue(torch_report["cleanup_confirmed"])
        self.assertEqual([pid for pid in children_of(server.proc.pid) if alive(pid)], [])



# ===================================================================== L8-3C comparisons
BASE_SCENARIO = dict(VALID)
SLEEP_LOWER = dict(VALID, sleep_hours_last_night=6.0)
COMPARE_URL = "/v1/comparisons"


def pair_body(original=None, modified=None):
    return {"original": dict(original or BASE_SCENARIO), "modified": dict(modified or SLEEP_LOWER)}


def with_generation(outcome, generation=1):
    return dataclasses.replace(outcome, generation=generation) if outcome.kind == "result" else outcome


class ComparisonStub(StubDispatcher):
    """Controlled comparison admission. ``script(payload, index)`` chooses each job's outcome."""

    def __init__(self, script=None, **kwargs):
        super().__init__(**kwargs)
        self.script = script or (lambda payload, index: with_generation(engine_outcome(payload)))
        self.admissions = []

    def submit_comparison(self, original, modified):
        first, second = (ExplanationEngine.validate_payload(p) for p in (original, modified))
        if self.submit_exc is not None:
            raise self.submit_exc
        payloads = (first,) if first == second else (first, second)
        deadline = self.deadline if self.deadline is not None else time.monotonic() + 0.5
        jobs = []
        for index, payload in enumerate(payloads):
            job = StubJob(self.script(payload, index))
            job.deadline = deadline
            jobs.append(job)
        self.admissions.append(payloads)
        self.jobs += jobs
        return tuple(jobs)


def comparison_client(clock=None, **kwargs):
    stub = ComparisonStub(**kwargs)
    app = s8.build_app(lambda root, hashes: stub, clock=clock) if clock else s8.build_app(lambda root, hashes: stub)
    return TestClient(app), stub


class ComparisonAssertions:
    def assert_error(self, response, status, code, retry=None):
        body = plain(response)
        self.assertEqual((response.status_code, body["error"]["code"]), (status, code), response.text)
        self.assertEqual(set(body), {"error"})  # never a partial comparison
        self.assertEqual(body["error"]["message"], s8._COMPARISON_MESSAGES[code])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers.get("retry-after"), retry)
        self.assertEqual(response.headers["content-type"], "application/json")
        s8.ComparisonErrorResponse.model_validate(body)
        for forbidden in ('"features"', '"differences"', "Traceback", "/workspace", "/app/"):
            self.assertNotIn(forbidden, response.text)
        return body["error"]


class ComparisonResponseTests(ComparisonAssertions, unittest.TestCase):
    def test_distinct_scenarios_use_two_evaluations_with_exact_association(self):
        client, stub = comparison_client()
        with client:
            response = client.post(COMPARE_URL, json=pair_body())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((response.headers["cache-control"], response.headers["content-type"]),
                         ("no-store", "application/json"))
        result = compare.ComparisonResponse.model_validate_json(response.text)
        self.assertEqual([len(a) for a in stub.admissions], [2])
        self.assertEqual([p["sleep_hours_last_night"] for p in stub.admissions[0]], [7.0, 6.0])
        for row, name in zip(result.features, schemas.FEATURES):
            self.assertEqual((row.input.original, row.input.modified),
                             (float(BASE_SCENARIO[name]), float(SLEEP_LOWER[name])))
        self.assertEqual(result.differences.changed_features, ["sleep_hours_last_night"])
        self.assertTrue(finite_numbers(plain(response)))
        self.assertEqual(stub.cancelled, [])

    def test_identical_scenarios_use_one_evaluation(self):
        client, stub = comparison_client()
        with client:
            response = client.post(COMPARE_URL, json=pair_body(modified=dict(VALID, sleep_hours_last_night=7)))
        result = compare.ComparisonResponse.model_validate_json(response.text)
        self.assertEqual(([len(a) for a in stub.admissions], len(stub.jobs)), ([1], 1))
        self.assertEqual((result.differences.logit, result.differences.probability, result.differences.changed_features,
                          result.ranking), (0.0, 0.0, [], []))
        self.assertTrue(all(r.attribution.delta == 0.0 and r.input.delta == 0.0 for r in result.features))
        self.assertEqual(result.original, result.modified)

    def test_identical_scenarios_do_not_bypass_numerical_validation(self):
        broken = success_outcome().explanation
        broken["attributions"][0]["attribution"] += 1.0
        client, _ = comparison_client(script=lambda p, i: d.Outcome("result", explanation=broken, generation=1))
        with client:
            self.assert_error(client.post(COMPARE_URL, json=pair_body(modified=BASE_SCENARIO)), 503,
                              "explanation_unavailable", "5")

    def test_eligibility_of_both_scenarios_in_fixed_order_without_values(self):
        client, stub = comparison_client()
        with client:
            response = client.post(COMPARE_URL, json=pair_body(
                dict(VALID, resting_hr_morning=58.25, weekly_km_to_date=19.25),
                dict(VALID, sleep_hours_last_night=8.25)))
        error = self.assert_error(response, 422, "outside_explanation_domain")
        self.assertEqual(error["violations"], [
            {"scenario": "original", "feature": "weekly_km_to_date", "min": 20.0, "max": 45.0},
            {"scenario": "original", "feature": "resting_hr_morning", "min": 45.0, "max": 57.0},
            {"scenario": "modified", "feature": "sleep_hours_last_night", "min": 5.5, "max": 8.0}])
        for value in ("19.25", "58.25", "8.25"):
            self.assertNotIn(value, response.text)
        self.assertEqual((stub.admissions, stub.jobs), ([], []))

    def test_admission_refusals_keep_codes_and_retry_after(self):
        for code, retry in (("explanation_overloaded", "1"), ("explanation_unavailable", "5")):
            client, stub = comparison_client(submit_exc=d.AdmissionRejected(code))
            with client:
                self.assert_error(client.post(COMPARE_URL, json=pair_body()), 503, code, retry)
        client, _ = comparison_client(submit_exc=RuntimeError("SECRET-internal-detail"))
        with client:
            response = client.post(COMPARE_URL, json=pair_body())
        self.assert_error(response, 503, "explanation_unavailable", "5")
        self.assertNotIn("SECRET", response.text)

    def test_unavailable_without_a_reference(self):
        client, stub = comparison_client()
        with client:
            client.app.state.explanation_reference = None
            self.assert_error(client.post(COMPARE_URL, json=pair_body()), 503, "explanation_unavailable", "5")
        self.assertEqual(stub.admissions, [])

    def fails_on_side(self, make, status, code, retry=None):
        for side in (0, 1):
            with self.subTest(side=side, code=code):
                client, _ = comparison_client(
                    script=lambda p, i: make(p) if i == side else with_generation(engine_outcome(p)))
                with client:
                    self.assert_error(client.post(COMPARE_URL, json=pair_body()), status, code, retry)

    def test_operational_failure_on_either_side(self):
        self.fails_on_side(lambda p: error_outcome("explanation_numerical_check_failed"), 503,
                           "explanation_numerical_check_failed")
        self.fails_on_side(lambda p: error_outcome("explanation_deadline_exceeded"), 504,
                           "explanation_deadline_exceeded")
        self.fails_on_side(lambda p: error_outcome("explanation_unavailable"), 503, "explanation_unavailable", "5")
        self.fails_on_side(lambda p: error_outcome("invalid_explanation_input"), 503, "explanation_unavailable", "5")

    def test_child_domain_violation_is_labelled_with_its_scenario(self):
        violation = {"feature": "weekly_km_to_date", "min": 20.0, "max": 45.0}
        client, _ = comparison_client(script=lambda p, i: error_outcome("outside_explanation_domain", [violation])
                                      if i == 1 else with_generation(engine_outcome(p)))
        with client:
            error = self.assert_error(client.post(COMPARE_URL, json=pair_body()), 422, "outside_explanation_domain")
        self.assertEqual(error["violations"], [dict(violation, scenario="modified")])

    def test_malformed_or_misassociated_result_on_either_side(self):
        def corrupt(change):
            def make(payload):
                body = engine_outcome(payload).explanation
                change(body)
                return d.Outcome("result", explanation=body, generation=1)
            return make

        for change in (lambda b: b.pop("logit"), lambda b: b["attributions"].reverse(),
                       lambda b: b.update(completeness_residual=0.5),
                       lambda b: b["prediction"].update(decision=1 - b["prediction"]["decision"])):
            self.fails_on_side(corrupt(change), 503, "explanation_unavailable", "5")
        other = dict(VALID, sleep_hours_last_night=6.5)  # a genuine result for a different input
        self.fails_on_side(lambda p: with_generation(engine_outcome(other)), 503, "explanation_unavailable", "5")

    def test_swapped_results_are_rejected(self):
        inputs = (BASE_SCENARIO, SLEEP_LOWER)
        client, _ = comparison_client(script=lambda p, i: with_generation(engine_outcome(inputs[1 - i])))
        with client:
            self.assert_error(client.post(COMPARE_URL, json=pair_body()), 503, "explanation_unavailable", "5")

    def test_generation_mismatch_or_missing_provenance(self):
        for generations in ((1, 2), (2, 1), (None, None), (1, None), (None, 1), (True, True)):
            with self.subTest(generations=generations):
                client, _ = comparison_client(script=lambda p, i: dataclasses.replace(
                    engine_outcome(p), generation=generations[i]))
                with client:
                    self.assert_error(client.post(COMPARE_URL, json=pair_body()), 503,
                                      "explanation_unavailable", "5")
        client, _ = comparison_client(script=lambda p, i: engine_outcome(p))
        with client:
            self.assert_error(client.post(COMPARE_URL, json=pair_body(modified=BASE_SCENARIO)), 503,
                              "explanation_unavailable", "5")

    def test_job_count_must_match_whether_the_scenarios_differ(self):
        class Wrong(ComparisonStub):
            def submit_comparison(self, original, modified):
                return super().submit_comparison(original, original)

        stub = Wrong()
        with TestClient(s8.build_app(lambda root, hashes: stub)) as client:
            self.assert_error(client.post(COMPARE_URL, json=pair_body()), 503, "explanation_unavailable", "5")

    def test_shared_identity_mismatch_and_consistency_errors_are_not_exposed(self):
        original = schemas.build_response
        calls = []

        def other_scope(*args, **kwargs):
            response = original(*args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                response.model.evidence_scope = "SECRET-other-scope"
            return response

        client, _ = comparison_client()
        with client, mock.patch.object(schemas, "build_response", other_scope):
            response = client.post(COMPARE_URL, json=pair_body())
        self.assert_error(response, 503, "explanation_unavailable", "5")
        for text in ("SECRET", "shared_model"):
            self.assertNotIn(text, response.text)
        client, _ = comparison_client()
        with client, mock.patch.object(compare, "build_comparison",
                                       side_effect=compare.ComparisonConsistencyError("completeness_gate")):
            response = client.post(COMPARE_URL, json=pair_body())
        self.assert_error(response, 503, "explanation_unavailable", "5")
        self.assertNotIn("completeness_gate", response.text)

    def test_both_sides_pass_through_the_trusted_builders(self):
        client, _ = comparison_client()
        with client, mock.patch.object(schemas, "build_response", wraps=schemas.build_response) as explain, \
                mock.patch.object(compare, "build_comparison", wraps=compare.build_comparison) as build:
            self.assertEqual(client.post(COMPARE_URL, json=pair_body()).status_code, 200)
        self.assertEqual((explain.call_count, build.call_count), (2, 1))
        self.assertTrue(all(type(a) is schemas.ExplanationResponse for a in build.call_args.args))


class ComparisonPrecedenceTests(ComparisonAssertions, unittest.TestCase):
    """Failure selection follows scenario order, never which future completed first."""

    def run_script(self, original, modified, deadline_in=0.3, order=("modified", "original"), delay=0.03):
        """``None`` stays unresolved; others are resolved from a thread in the given order."""
        outcomes = {"original": original, "modified": modified}

        class Timed(ComparisonStub):
            def submit_comparison(stub, first, second):
                jobs = super().submit_comparison(first, second)

                def resolve():
                    for name in order:
                        time.sleep(delay)
                        job = jobs[("original", "modified").index(name)]
                        if outcomes[name] is not None and not job.future.done():
                            job.future.set_result(outcomes[name])

                threading.Thread(target=resolve, daemon=True).start()
                return jobs

        stub = Timed(script=lambda p, i: None)
        with TestClient(s8.build_app(lambda root, hashes: stub)) as client:
            original_submit = stub.submit_comparison

            def submit(first, second):
                stub.deadline = time.monotonic() + deadline_in
                return original_submit(first, second)

            stub.submit_comparison = submit
            response = client.post(COMPARE_URL, json=pair_body())
        return response, stub

    def good(self, payload):
        return with_generation(engine_outcome(payload))

    def test_primary_beats_collateral_regardless_of_completion_order(self):
        for order in (("modified", "original"), ("original", "modified")):
            with self.subTest(order=order):
                response, _ = self.run_script(error_outcome("explanation_deadline_exceeded"),
                                              error_outcome("explanation_unavailable"), order=order)
                self.assert_error(response, 504, "explanation_deadline_exceeded")

    def test_failed_modified_waits_for_the_original_to_classify(self):
        response, _ = self.run_script(self.good(BASE_SCENARIO), error_outcome("explanation_numerical_check_failed"))
        self.assert_error(response, 503, "explanation_numerical_check_failed")
        response, _ = self.run_script(error_outcome("explanation_numerical_check_failed"),
                                      error_outcome("explanation_unavailable"))
        self.assert_error(response, 503, "explanation_numerical_check_failed")

    def test_failed_modified_with_a_stuck_original_ends_at_the_shared_deadline(self):
        started = time.monotonic()
        response, stub = self.run_script(None, error_outcome("explanation_unavailable"), deadline_in=0.2)
        self.assert_error(response, 504, "explanation_deadline_exceeded")
        self.assertLess(time.monotonic() - started, 10.0)
        self.assertEqual(stub.cancelled, [stub.jobs[0]])

    def test_failed_original_decides_immediately_and_cancels_the_sibling(self):
        response, stub = self.run_script(error_outcome("explanation_numerical_check_failed"), None,
                                         deadline_in=5.0, order=("original",))
        self.assert_error(response, 503, "explanation_numerical_check_failed")
        self.assertEqual(stub.cancelled, [stub.jobs[1]])
        self.assertTrue(stub.jobs[1].future.cancelled())

    def test_deadline_cancels_the_later_job_first(self):
        response, stub = self.run_script(None, None, deadline_in=0.05)
        self.assert_error(response, 504, "explanation_deadline_exceeded")
        self.assertEqual(stub.cancelled, [stub.jobs[1], stub.jobs[0]])


class ComparisonDeadlineTests(ComparisonAssertions, unittest.TestCase):
    def post(self, start, finish=None, target=(None, None)):
        clock = ManualClock(start)
        client, stub = comparison_client(clock=clock, deadline=0.5)
        module, name = target
        if module is None:
            with client:
                return client.post(COMPARE_URL, json=pair_body()), stub
        real_function = getattr(module, name)

        def advancing(*args, **kwargs):
            result = real_function(*args, **kwargs)
            clock.t = finish
            return result

        with client, mock.patch.object(module, name, advancing):
            return client.post(COMPARE_URL, json=pair_body()), stub

    def test_resolved_results_at_or_after_the_deadline_are_late(self):
        for start in (0.5, 5.0):
            response, stub = self.post(start)
            self.assert_error(response, 504, "explanation_deadline_exceeded")
            self.assertEqual(stub.cancelled, [])

    def test_deadline_during_validation_construction_and_rendering(self):
        for target in ((schemas, "verify_consistency"), (schemas, "build_response"), (compare, "build_comparison"),
                       (s8, "JSONResponse")):
            with self.subTest(target=target[1]):
                if target[1] == "JSONResponse":
                    response, _ = self.post_rendering(0.5)
                    self.assert_error(response, 504, "explanation_deadline_exceeded")
                    self.assertEqual(self.post_rendering(0.4999)[0].status_code, 200)
                    continue
                self.assert_error(self.post(0.0, 0.5, target)[0], 504, "explanation_deadline_exceeded")
                self.assertEqual(self.post(0.0, 0.4999, target)[0].status_code, 200)

    def post_rendering(self, finish):
        clock = ManualClock(0.0)
        client, stub = comparison_client(clock=clock, deadline=0.5)
        real_dump = compare.ComparisonResponse.model_dump

        def advancing(model, *args, **kwargs):
            result = real_dump(model, *args, **kwargs)
            clock.t = finish
            return result

        with client, mock.patch.object(compare.ComparisonResponse, "model_dump", advancing):
            return client.post(COMPARE_URL, json=pair_body()), stub


class ComparisonRequestParsingTests(ComparisonAssertions, unittest.TestCase):
    JSON = {"content-type": "application/json"}

    def send(self, content, headers=None):
        client, stub = comparison_client()
        with client:
            response = client.post(COMPARE_URL, content=content, headers=self.JSON if headers is None else headers)
        return response, stub

    def refused(self, content, status, code, headers=None):
        response, stub = self.send(content, headers)
        error = self.assert_error(response, status, code)
        self.assertEqual((stub.admissions, stub.jobs, stub.submitted), ([], [], []), "no capacity used")
        return response, error

    def test_invalid_nested_fields_report_only_allowlisted_identifiers(self):
        def body(**sides):
            return json.dumps(dict(pair_body(), **sides))

        missing = {k: v for k, v in VALID.items() if k != "prior_injury"}
        cases = (
            (body(original=missing), [{"scenario": "original", "feature": "prior_injury", "problem": "missing_field"}]),
            (json.dumps({"original": VALID}), [{"scenario": "modified", "problem": "missing_field"}]),
            (body(modified=dict(VALID, extra_field=1, other=2)), [{"scenario": "modified", "problem": "unknown_field"}]),
            (body(extra=1), [{"problem": "unknown_field"}]),
            (body(original=None), [{"scenario": "original", "problem": "invalid_value"}]),
            (body(modified=dict(VALID, prior_injury=1.0, weekly_km_to_date=True)),
             [{"scenario": "modified", "feature": "weekly_km_to_date", "problem": "invalid_value"},
              {"scenario": "modified", "feature": "prior_injury", "problem": "invalid_value"}]),
            (body(original=dict(VALID, resting_hr_morning="50"), modified=dict(VALID, sleep_hours_last_night=None)),
             [{"scenario": "original", "feature": "resting_hr_morning", "problem": "invalid_value"},
              {"scenario": "modified", "feature": "sleep_hours_last_night", "problem": "invalid_value"}]),
            (body(original=dict(VALID, weekly_km_to_date=201.0)),
             [{"scenario": "original", "feature": "weekly_km_to_date", "problem": "invalid_value"}]),
            ("[]", [{"problem": "invalid_value"}]),
            ('"text"', [{"problem": "invalid_value"}]),
        )
        for content, violations in cases:
            with self.subTest(content=content[:60]):
                _, error = self.refused(content, 422, "invalid_comparison_request")
                self.assertEqual(error["violations"], violations)
        huge = json.dumps(pair_body()).replace("32.0", "1" + "0" * 400, 1)
        self.refused(huge, 422, "invalid_comparison_request")

    def test_duplicate_keys_at_every_depth(self):
        good = json.dumps(pair_body())
        side = json.dumps(VALID)
        for content in (
            good[:-1] + ',"original":' + side + "}",
            good.replace('"prior_injury": 0', '"prior_injury": 1, "prior_injury": 0', 1),
            '{"original":' + side + ',"modified":' + side[:-1] + ',"weekly_km_to_date":30}}',
            '{"original":{"a":{"b":1,"b":2}},"modified":{}}',
        ):
            with self.subTest(content=content[-60:]):
                self.refused(content, 422, "duplicate_json_key")

    def test_malformed_and_non_finite_json(self):
        good = json.dumps(pair_body())
        cases = [good[:-1], "", "{", good + "x", "\ufeff" + good, b"\xff\xfe{}", "[" * 4000]
        cases += [good.replace("32.0", token, 1) for token in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999")]
        for content in cases:
            with self.subTest(content=repr(content[:30])):
                self.refused(content, 422, "malformed_comparison_json")

    def test_deep_but_valid_nesting_within_the_limit_fails_schema_validation(self):
        # The size limit bounds the depth. The canonical parser accepts it, so it is a schema failure.
        for content in ("[" * 2000 + "]" * 2000, '{"original":' + "[" * 1900 + "]" * 1900 + ',"modified":{}}'):
            with self.subTest(content=content[:20]):
                _, error = self.refused(content, 422, "invalid_comparison_request")
                self.assertLessEqual(len(error["violations"]), 6)

    def test_body_limit_declared_and_streamed(self):
        self.assertEqual(s8.MAX_COMPARISON_BODY_BYTES, 4096)
        good = json.dumps(pair_body()).encode()
        response, _ = self.send(good + b" " * (4096 - len(good)))
        self.assertEqual(response.status_code, 200, response.text)
        self.refused(good + b" " * (4097 - len(good)), 413, "comparison_request_too_large")

        def chunks():
            for _ in range(10):
                yield b" " * 1000

        response, _ = self.refused(chunks(), 413, "comparison_request_too_large")
        self.assertNotIn("content-length", response.request.headers)
        pretty = json.dumps({side: {k: 1.2345678901234567e+1 for k in schemas.FEATURES}
                             for side in compare.SCENARIOS}, indent=4)
        self.assertLess(len(pretty), 1024)

    def test_media_type_contract(self):
        good = json.dumps(pair_body())
        for content_type in ("application/json", "APPLICATION/JSON", "application/json; charset=utf-8",
                             'application/json;charset="UTF-8"'):
            with self.subTest(accepted=content_type):
                self.assertEqual(self.send(good, {"content-type": content_type})[0].status_code, 200)
        for headers in ({}, {"content-type": "text/plain"}, {"content-type": "application/x-www-form-urlencoded"},
                        {"content-type": "application/problem+json"},
                        {"content-type": "application/json; charset=latin-1"},
                        {"content-type": "application/json; boundary=x"},
                        {"content-type": "application/json", "content-encoding": "gzip"}):
            with self.subTest(refused=headers):
                self.refused(good, 415, "unsupported_media_type", headers)

    def test_nothing_client_supplied_is_echoed(self):
        marks = ("SYNTHETIC-TOKEN-sk-000000", "synthetic.person@example.invalid", "/Users/synthetic/secret.txt",
                 "<script>synthetic</script>", "87.654321")
        token, mail, path, script, number = marks
        documents = (
            json.dumps(pair_body(modified=dict(VALID, **{token: mail}))),
            json.dumps(dict(pair_body(), **{path: script})),
            json.dumps(pair_body(modified=dict(VALID, weekly_km_to_date=token))),
            json.dumps(pair_body(modified=dict(VALID, weekly_km_to_date=float(number) + 200))),
            '{"%s": 1, "%s": 2}' % (token, token),
            '{"original": %s ' % script,
            json.dumps({token: {mail: path}}),
        )
        for content in documents:
            response, stub = self.send(content)
            self.assertIn(response.status_code, (422,), response.text)
            everything = response.text + json.dumps(dict(response.headers))
            for mark in marks + ("287.654321",):
                self.assertNotIn(mark, everything)
            self.assertEqual(stub.admissions, [])
        response, _ = self.send(token, {"content-type": mail})
        self.assertNotIn(mail, response.text + json.dumps(dict(response.headers)))

    def test_no_client_selectable_settings(self):
        for extra in ("baseline", "n_steps", "threshold", "model", "method"):
            self.refused(json.dumps(dict(pair_body(), **{extra: 1})), 422, "invalid_comparison_request")


class ComparisonOpenApiTests(unittest.TestCase):
    def test_request_response_and_error_contracts(self):
        spec = s8.app.openapi()
        path = spec["paths"]["/v1/comparisons"]
        self.assertEqual(set(path), {"post"})
        operation = path["post"]
        self.assertNotIn("parameters", operation)
        self.assertEqual(set(operation["responses"]), {"200", "413", "415", "422", "503", "504"})
        self.assertIn("ComparisonResponse", json.dumps(operation["responses"]["200"]))
        for status in ("413", "415", "422", "503", "504"):
            self.assertIn("ComparisonErrorResponse", json.dumps(operation["responses"][status]))
        self.assertTrue(operation["requestBody"]["required"])
        request = operation["requestBody"]["content"]["application/json"]["schema"]
        self.assertEqual((request["type"], request["additionalProperties"], sorted(request["required"]),
                          sorted(request["properties"])),
                         ("object", False, ["modified", "original"], ["modified", "original"]))
        for side in compare.SCENARIOS:
            self.assertEqual(request["properties"][side], {"$ref": "#/components/schemas/PredictionRequest"})
        components = spec["components"]["schemas"]
        self.assertFalse(components["PredictionRequest"]["additionalProperties"])
        self.assertEqual(components["ComparisonResponse"]["properties"]["schema_version"]["const"], "comparison.v1")

        def references(node):
            if isinstance(node, dict):
                if "$ref" in node:
                    yield node["$ref"]
                for value in node.values():
                    yield from references(value)
            elif isinstance(node, list):
                for value in node:
                    yield from references(value)

        for reference in references(spec):
            self.assertIn(reference.removeprefix("#/components/schemas/"), components, reference)

    def test_existing_operations_do_not_gain_comparison_content(self):
        spec = s8.app.openapi()
        for path in ("/health", "/predict", "/v1/explanations"):
            self.assertNotIn("Comparison", json.dumps(spec["paths"][path]))


class ComparisonHttpTests(ComparisonAssertions, unittest.IsolatedAsyncioTestCase):
    """The real dispatcher with controlled fake workers. Helpers are borrowed, not inherited."""

    asyncSetUp = HttpDeadlineTests.asyncSetUp
    start = HttpDeadlineTests.start
    push_ready = HttpDeadlineTests.push_ready
    until = HttpDeadlineTests.until
    tasks = HttpDeadlineTests.tasks
    LONG = {"exec_timeout": 20.0, "request_deadline": 20.0}

    def answer(self, handle):
        message = handle.sent[-1]
        handle.push(("message", {"t": "result", "gen": handle.generation, "id": message["id"],
                                 "explanation": engine_outcome(message["payload"]).explanation}))

    async def post_comparison(self, body):
        content = json.dumps(body).encode()
        disconnect, sent, state = asyncio.Event(), [], {"first": True}

        async def receive():
            if state["first"]:
                state["first"] = False
                return {"type": "http.request", "body": content, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "path": COMPARE_URL, "raw_path": COMPARE_URL.encode(), "query_string": b"", "root_path": "",
                 "scheme": "http", "server": ("t", 80), "client": ("c", 1),
                 "headers": [(b"host", b"t"), (b"content-type", b"application/json"),
                             (b"content-length", str(len(content)).encode())]}
        return asyncio.create_task(self.app(scope, receive, send)), sent, disconnect

    def counters(self):
        return collections.Counter(self.disp.diagnostics()["counters"])

    async def idle(self):
        await self.until(lambda: self.disp.diagnostics()["active"] == 0 and self.disp.diagnostics()["queued"] == 0)
        diag = self.disp.diagnostics()
        self.assertEqual((diag["active"], diag["queued"], diag["invariant_violations"]), (0, 0, 0))

    async def assert_disconnected(self, task, sent, baseline):
        await asyncio.wait_for(task, 10)
        start = next(m for m in sent if m["type"] == "http.response.start")
        self.assertEqual(start["status"], 499)
        self.assertIn((b"cache-control", b"no-store"), start["headers"])
        self.assertEqual(self.tasks() - baseline, set())

    async def test_success_through_the_real_dispatcher(self):
        await self.start(**self.LONG)
        baseline = self.tasks()
        pending = asyncio.create_task(self.client.post(COMPARE_URL, json=pair_body()))
        for count in (1, 2):
            await self.until(lambda: len(self.handles[0].sent) == count)
            self.answer(self.handles[0])
        response = await pending
        self.assertEqual(response.status_code, 200, response.text)
        compare.ComparisonResponse.model_validate_json(response.text)
        await self.idle()
        c = self.counters()
        self.assertEqual((c["admitted"], c["comparison_requests_admitted"], c["completed"]), (2, 1, 2))
        self.assertEqual(self.tasks(), baseline)

    async def test_pair_overload_with_one_slot_left(self):
        await self.start(**self.LONG)
        for _ in range(8):
            self.disp.submit(VALID)
        response = await self.client.post(COMPARE_URL, json=pair_body())
        self.assert_error(response, 503, "explanation_overloaded", "1")
        c = self.counters()
        self.assertEqual((c["admitted"], c["comparison_requests_rejected_overloaded"], c["rejected_overloaded"],
                          self.disp.diagnostics()["queued"]), (8, 1, 0, 7))

    async def test_worker_not_ready(self):
        await self.start(**self.LONG)
        self.handles[0].push(("closed", "eof"))
        await self.until(lambda: self.disp.diagnostics()["state"] != d.READY)
        self.assert_error(await self.client.post(COMPARE_URL, json=pair_body()), 503,
                          "explanation_unavailable", "5")
        self.assertEqual(self.counters()["comparison_requests_rejected_unavailable"], 1)

    async def test_disconnect_before_either_evaluation_is_dispatched(self):
        await self.start(**self.LONG)
        blocker = self.disp.submit(VALID)
        await self.until(lambda: self.handles[0].sent)
        baseline = self.tasks()
        task, sent, disconnect = await self.post_comparison(pair_body())
        await self.until(lambda: self.disp.diagnostics()["queued"] == 2)
        disconnect.set()
        await self.assert_disconnected(task, sent, baseline)
        self.assertEqual((self.disp.diagnostics()["queued"], self.counters()["cancelled_queued"]), (0, 2))
        self.answer(self.handles[0])
        await self.idle()
        self.assertEqual((len(self.handles[0].sent), blocker.future.result(0).kind), (1, "result"))

    async def test_disconnect_during_the_original(self):
        await self.start(**self.LONG)
        baseline = self.tasks()
        task, sent, disconnect = await self.post_comparison(pair_body())
        await self.until(lambda: self.handles[0].sent)
        disconnect.set()
        await self.assert_disconnected(task, sent, baseline)
        diag, c = self.disp.diagnostics(), self.counters()
        self.assertEqual((diag["active"], diag["queued"], c["cancelled_queued"], c["cancelled_active"]),
                         (1, 0, 1, 1))  # the active slot stays reserved
        self.answer(self.handles[0])
        await self.idle()
        self.assertEqual(len(self.handles[0].sent), 1)

    async def test_disconnect_during_the_modified(self):
        await self.start(**self.LONG)
        baseline = self.tasks()
        task, sent, disconnect = await self.post_comparison(pair_body())
        await self.until(lambda: self.handles[0].sent)
        self.answer(self.handles[0])
        await self.until(lambda: len(self.handles[0].sent) == 2)
        disconnect.set()
        await self.assert_disconnected(task, sent, baseline)
        self.assertEqual((self.disp.diagnostics()["active"], self.counters()["cancelled_active"]), (1, 1))
        self.answer(self.handles[0])
        await self.idle()

    async def test_disconnect_between_the_evaluations(self):
        """The modified job is held unsent: the dispatcher thread is gated inside its send."""
        gate, reached = threading.Event(), threading.Event()
        self.addCleanup(gate.set)

        class Gated(dt.BlockingFake):
            def send(handle, message, deadline):
                if handle.sent:
                    reached.set()
                    gate.wait(20)
                super().send(message, deadline)

        await self.start(handle_cls=Gated, **self.LONG)
        baseline = self.tasks()
        task, sent, disconnect = await self.post_comparison(pair_body())
        await self.until(lambda: self.handles[0].sent)
        self.answer(self.handles[0])
        await self.until(reached.is_set)
        self.assertEqual(len(self.handles[0].sent), 1)
        disconnect.set()
        await self.assert_disconnected(task, sent, baseline)
        gate.set()
        await self.until(lambda: len(self.handles[0].sent) == 2)
        self.answer(self.handles[0])
        await self.idle()
        self.assertEqual(self.counters()["cancelled_active"], 1)

    async def test_handler_task_cancellation_cancels_both(self):
        await self.start(**self.LONG)
        baseline = self.tasks()
        task, _, _ = await self.post_comparison(pair_body())
        await self.until(lambda: self.handles[0].sent)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        c = self.counters()
        self.assertEqual((c["cancelled_queued"], c["cancelled_active"]), (1, 1))
        self.assertEqual(self.tasks() - baseline, set())
        self.answer(self.handles[0])
        await self.idle()

    async def test_original_stall_is_a_504_not_the_collateral_unavailable(self):
        await self.start()  # committed limits: 250 ms execution inside the 500 ms admission deadline
        response = await self.client.post(COMPARE_URL, json=pair_body())
        self.assert_error(response, 504, "explanation_deadline_exceeded")
        await self.idle()
        self.assertEqual([len(h.sent) for h in self.handles][0], 1)
        self.assertTrue(all(not h.sent for h in self.handles[1:]))  # the sibling is never resubmitted

    async def test_worker_death_between_evaluations(self):
        await self.start(**self.LONG)
        pending = asyncio.create_task(self.client.post(COMPARE_URL, json=pair_body()))
        await self.until(lambda: self.handles[0].sent)
        self.handles[0].send_error = "closed"
        self.answer(self.handles[0])
        self.assert_error(await pending, 503, "explanation_unavailable", "5")
        await self.until(lambda: len(self.handles) == 2)
        await self.idle()
        self.assertEqual(self.handles[1].sent, [])


# ------------------------------------------------------- real spawned worker: frozen matrix
_MULTI = dict(VALID, weekly_km_to_date=40.0, sleep_hours_last_night=6.0, soreness_score_evening=7.0)
REAL_MATRIX = (
    ("identical", VALID, VALID),
    ("single_feature", VALID, SLEEP_LOWER),
    ("multiple_features", VALID, _MULTI),
    ("prior_injury", VALID, dict(VALID, prior_injury=1)),
    ("reversed_multiple", _MULTI, VALID),
)


class KillingHandle(d.ProcessWorkerHandle):
    """Real worker handle that, once armed, kills and joins its own process inside the send of the
    second explain frame: deterministically between the two evaluations of one comparison."""

    def __init__(self, *args, plan, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan = plan

    def send(self, message, deadline):
        if message.get("t") == "explain":
            self.plan["sent"].append((self.generation, message["id"]))
            if self.plan["armed"] is not None:
                self.plan["armed"] += 1
                if self.plan["armed"] == 2:
                    self.plan["armed"] = None
                    self.plan["killed"] = self.generation
                    os.kill(self.pid, signal.SIGKILL)
                    self._proc.join(5)
        super().send(message, deadline)


class RealWorkerComparisonTests(ComparisonAssertions, unittest.TestCase):
    def wait_ready(self, client):
        end = time.monotonic() + 90
        while time.monotonic() < end:
            if client.app.state.explanation_dispatcher.diagnostics()["state"] == d.READY:
                return
            time.sleep(0.05)
        self.fail("worker not ready")

    def explain(self, client, payload):
        response = client.post("/v1/explanations", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return schemas.ExplanationResponse.model_validate_json(response.text)

    def test_frozen_matrix_and_failure_between_evaluations(self):
        plan = {"armed": None, "sent": [], "killed": None}

        def make(root, hashes):
            return d.Dispatcher(str(root), expected_artifacts=hashes, handle_factory=lambda gen: KillingHandle(
                gen, str(root), hashes, plan=plan, clock=d.RealClock()))

        app, timings = s8.build_app(make), {}
        with TestClient(app) as client:
            self.wait_ready(client)
            disp = app.state.explanation_dispatcher
            jobs_seen, real_submit = [], disp.submit_comparison

            def spy(*args):
                jobs = real_submit(*args)
                jobs_seen.append(jobs)
                return jobs

            disp.submit_comparison = spy
            for name, original, modified in REAL_MATRIX:
                with self.subTest(name=name):
                    separate = [self.explain(client, original), self.explain(client, modified)]
                    before = collections.Counter(disp.diagnostics()["counters"])
                    started = time.perf_counter()
                    response = client.post(COMPARE_URL, json={"original": original, "modified": modified})
                    timings[name] = round((time.perf_counter() - started) * 1000, 1)
                    self.assertEqual(response.status_code, 200, response.text)
                    result = compare.ComparisonResponse.model_validate_json(response.text)
                    self.assertEqual(result, compare.build_comparison(*separate))  # exact, field for field
                    for side, alone in zip((result.original, result.modified), separate):
                        self.assertEqual((side.prediction.probability, side.prediction.logit,
                                          side.completeness.residual),
                                         (alone.prediction.probability, alone.prediction.logit,
                                          alone.completeness.residual))
                    for index, alone in enumerate(separate):
                        key = compare.SCENARIOS[index]
                        self.assertEqual([getattr(r.attribution, key) for r in result.features],
                                         [a.attribution for a in alone.attributions])
                    expected = 1 if name == "identical" else 2
                    jobs = jobs_seen[-1]
                    self.assertEqual(len(jobs), expected)
                    self.assertEqual({j.future.result(0).generation for j in jobs}, {1})
                    after = collections.Counter(disp.diagnostics()["counters"])
                    self.assertEqual({k: after[k] - before[k] for k in ("admitted", "completed",
                                      "comparison_requests_admitted", "comparison_evaluations_admitted")},
                                     {"admitted": expected, "completed": expected,
                                      "comparison_requests_admitted": 1, "comparison_evaluations_admitted": expected})
            # controlled failure between the two evaluations
            before_sent = len(plan["sent"])
            plan["armed"] = 0
            response = client.post(COMPARE_URL, json=pair_body())
            self.assert_error(response, 503, "explanation_unavailable", "5")
            first, second = jobs_seen[-1]
            self.assertEqual((first.future.result(0).kind, first.future.result(0).generation), ("result", 1))
            self.assertEqual((second.future.result(0).code, second.future.result(0).generation),
                             ("explanation_unavailable", None))
            self.assertEqual(plan["killed"], 1)
            self.wait_ready(client)
            recovered = client.post(COMPARE_URL, json=pair_body())
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertEqual({j.future.result(0).generation for j in jobs_seen[-1]}, {2})
            sent = plan["sent"][before_sent:]
            self.assertEqual([g for g, i in sent if i == second.req_id], [1])  # never sent to the replacement
            diag = disp.diagnostics()
            self.assertEqual((diag["restarts_total"], diag["active"], diag["queued"], diag["invariant_violations"]),
                             (1, 0, 0, 0))
            self.assertLessEqual(diag["high_water"]["admitted"], 9)
        self.assertIs(app.state.explanation_cleanup_confirmed, True)
        self.assertEqual(multiprocessing.active_children(), [])
        TIMINGS["comparison_real_matrix_ms_descriptive"] = timings


if __name__ == "__main__":
    unittest.main()
