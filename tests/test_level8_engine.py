"""Level 8 L8-1 engine regression tests (numerical verification only).

These tests verify that the Captum Integrated Gradients engine agrees with an independent
exact oracle and with ModelRuntime. They say nothing about model validity or production
performance. Run inside the canonical image with Captum installed, from the repository root.

Thread configuration is test-harness territory: setUpModule pins one intra-op thread (as
in the verified experiment) and tearDownModule restores the previous value. The engine
itself never touches process-global Torch settings.

Environment switches:
  LEVEL8_PERF=1              run the serial latency smoke test (needs 1 CPU for meaning)
  LEVEL8_EVIDENCE_OUT=<file> write the verification report JSON to <file>
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from src.level7_runtime import SAMPLE_REQUEST, ModelRuntime
from src import level8_explain as le
from tests import level8_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "verification" / "level8" / "numerical_verification.json"

# Verified acceptance criteria (F2B / L8-0). Never relax to make a test pass.
FEATURE_ERR_MAX = 0.005
L1_REL_MAX = 0.005
L1_FLOOR = 0.05
SIGN_MIN = 0.02
TIE_TOL = 0.005
PARITY = 2e-6
LATENCY_P95_MS = 25.0

BUNDLE_FILES = (
    "serving-manifest.json",
    "artifacts/v4/level6_selected_model.pt",
    "artifacts/v4/level6_preprocessor.json",
    "artifacts/v4/level7_deploy_handoff.json",
    "src/level7_runtime.py",
)

_STATE: dict = {}
_PERF: dict = {}


def setUpModule():
    _STATE["prev_threads"] = torch.get_num_threads()
    torch.set_num_threads(1)
    runtime = ModelRuntime(ROOT)
    _STATE["runtime"] = runtime
    _STATE["before"] = _snapshot(runtime)
    _STATE["globals_before"] = _torch_globals()
    _STATE["engine"] = le.ExplanationEngine(runtime)
    _STATE["grids"] = oracle.build_grids(runtime, SAMPLE_REQUEST)


def tearDownModule():
    torch.set_num_threads(_STATE["prev_threads"])


def _torch_globals() -> dict:
    return {
        "threads": torch.get_num_threads(),
        "interop": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
    }


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _snapshot(runtime) -> dict:
    return {
        "files": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in BUNDLE_FILES},
        "weights": {k: _tensor_digest(v) for k, v in runtime.model.state_dict().items()},
        "requires_grad": sorted({p.requires_grad for p in runtime.model.parameters()}),
        "training": runtime.model.training,
        "grads_none": all(p.grad is None for p in runtime.model.parameters()),
        "threshold": runtime.threshold,
    }


def _payload(row) -> dict:
    names = _STATE["runtime"].feature_names
    return {n: (int(v) if n == "prior_injury" else float(v)) for n, v in zip(names, row)}


def _study() -> dict:
    """Explain and oracle-check every input of the two verification grids, once."""
    if "study" in _STATE:
        return _STATE["study"]
    runtime, engine, grids = _STATE["runtime"], _STATE["engine"], _STATE["grids"]
    p64 = oracle.params64(runtime.model)
    b0 = np.zeros(5)
    out = {}
    for name in ("dev_range", "straddle_threshold"):
        rows = grids[name]
        n = len(rows)
        attrs, orc = np.zeros((n, 5)), np.zeros((n, 5))
        resid, allowed, diff, parity = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        names_ok = True
        for i, row in enumerate(rows):
            result = engine.explain(_payload(row))
            names_ok &= [a.feature for a in result.attributions] == runtime.feature_names
            attrs[i] = [a.attribution for a in result.attributions]
            z = np.array([a.standardized_value for a in result.attributions])
            orc[i], _ = oracle.oracle_ig(p64, z, b0)
            resid[i], allowed[i] = result.completeness_residual, result.allowed_residual
            diff[i] = result.target_difference
            parity[i] = abs(le._sigmoid(result.logit) - result.prediction["probability"])
        err = np.abs(attrs - orc)
        feat_err = err.max(1)
        rel_l1 = err.sum(1) / np.maximum(np.abs(orc).sum(1), L1_FLOOR)
        material = np.abs(orc) > SIGN_MIN
        sign_ok = np.where(material, np.sign(attrs) == np.sign(orc), True).all(1)
        top = np.abs(orc).max(1)
        tie_sets = np.abs(orc) >= (top[:, None] - TIE_TOL)
        top_ok = tie_sets[np.arange(n), np.abs(attrs).argmax(1)]
        strict = np.abs(attrs).argmax(1) == np.abs(orc).argmax(1)
        out[name] = {
            "n": n, "names_ok": bool(names_ok), "attrs": attrs, "oracle": orc,
            "finite": bool(np.isfinite(attrs).all() and attrs.shape == (n, 5)),
            "resid_ok": np.abs(resid) <= allowed, "resid": resid, "allowed": allowed, "diff": diff,
            "feat_err": feat_err, "rel_l1": rel_l1, "sign_ok": sign_ok, "material": int(material.sum()),
            "top_ok": top_ok, "strict_agree": float(strict.mean()),
            "ties": int(tie_sets.sum(1).__gt__(1).sum()), "parity": parity,
        }
    _STATE["study"] = out
    return out


class GridTests(unittest.TestCase):
    def test_grid_construction_is_the_specified_one(self):
        g = _STATE["grids"]
        self.assertEqual(g["dev_range"].shape, (500, 5))
        self.assertEqual(g["straddle_threshold"].shape, (20, 5))
        self.assertEqual(g["api_corners"].shape, (32, 5))
        self.assertEqual(g["api_wide"].shape, (100, 5))
        dev = g["dev_range"]
        self.assertTrue((dev >= oracle.DEV_LO).all() and (dev <= oracle.DEV_HI).all())
        self.assertTrue(set(np.unique(dev[:, 4])) <= {0.0, 1.0})
        # Every straddling point is explanation-eligible and near the decision boundary.
        for row in g["straddle_threshold"]:
            le.ExplanationEngine.validate_payload(_payload(row))
        runtime = _STATE["runtime"]
        with torch.no_grad():
            logits = runtime.model(oracle.standardize(runtime, g["straddle_threshold"])).numpy()
        self.assertLess(np.abs(logits).max(), 0.05)

    def test_grid_digests_match_committed_evidence(self):
        if not EVIDENCE_PATH.is_file():
            self.skipTest("evidence file not present yet")
        recorded = json.loads(EVIDENCE_PATH.read_text())["grid_construction"]["sha256_float64"]
        g = _STATE["grids"]
        for name in ("dev_range", "straddle_threshold", "api_corners", "api_wide"):
            self.assertEqual(oracle.grid_digest(g[name]), recorded[name], name)


class NumericalAcceptanceTests(unittest.TestCase):
    def test_gates_hold_on_both_grids(self):
        for name, s in _study().items():
            with self.subTest(grid=name):
                self.assertTrue(s["finite"])
                self.assertTrue(s["names_ok"], "canonical feature order")
                self.assertTrue(s["resid_ok"].all(), f"worst residual {np.abs(s['resid']).max()}")
                self.assertLessEqual(s["feat_err"].max(), FEATURE_ERR_MAX)
                self.assertLessEqual(s["rel_l1"].max(), L1_REL_MAX)
                self.assertTrue(s["sign_ok"].all())
                self.assertTrue(s["top_ok"].all())
                self.assertLessEqual(s["parity"].max(), PARITY)
                self.assertGreater(s["material"], 0)

    def test_oracle_self_consistency(self):
        runtime = _STATE["runtime"]
        p64 = oracle.params64(runtime.model)
        z = oracle.standardize(runtime, _STATE["grids"]["sample"]).double().numpy()[0]
        attr, pieces = oracle.oracle_ig(p64, z, np.zeros(5))
        delta = oracle.logit64(p64, z) - oracle.logit64(p64, np.zeros(5))
        self.assertAlmostEqual(float(attr.sum()), delta, places=9)
        self.assertGreaterEqual(pieces, 1)

    def test_baseline_equal_input_has_zero_attribution(self):
        r = _STATE["engine"]._explain_standardized(torch.zeros(1, 5, dtype=torch.float32))
        self.assertTrue(all(a == 0.0 for a in r.attributions))
        self.assertEqual(r.target_difference, 0.0)
        self.assertEqual(r.completeness_residual, 0.0)

    def test_single_feature_perturbation(self):
        engine = _STATE["engine"]
        for j in range(5):
            for value in (-1.5, 0.8):
                x = torch.zeros(1, 5, dtype=torch.float32)
                x[0, j] = value
                r = engine._explain_standardized(x)
                for i, a in enumerate(r.attributions):
                    if i != j:
                        self.assertEqual(a, 0.0, f"feature {i} moved by feature {j}")
                self.assertAlmostEqual(r.attributions[j], r.target_difference, delta=r.allowed_residual)

    def test_deterministic_repeatability(self):
        engine = _STATE["engine"]
        first = [a.attribution for a in engine.explain(SAMPLE_REQUEST).attributions]
        for _ in range(20):
            self.assertEqual([a.attribution for a in engine.explain(SAMPLE_REQUEST).attributions], first)
        fresh = le.ExplanationEngine(_STATE["runtime"])
        self.assertEqual([a.attribution for a in fresh.explain(SAMPLE_REQUEST).attributions], first)

    def test_prediction_is_the_runtime_prediction(self):
        r = _STATE["engine"].explain(SAMPLE_REQUEST)
        self.assertEqual(r.prediction, _STATE["runtime"].predict(SAMPLE_REQUEST))
        self.assertEqual(len(r.attributions), 5)
        self.assertEqual([a.feature for a in r.attributions], list(le.EXPLANATION_RANGES))


class FailClosedTests(unittest.TestCase):
    def test_numerical_failure_returns_no_partial_result(self):
        engine = le.ExplanationEngine(_STATE["runtime"])
        original = engine._ig.attribute

        def nan_attribute(*a, **k):
            out = original(*a, **k)
            out[0, 2] = float("nan")
            return out

        engine._ig.attribute = nan_attribute
        with self.assertRaises(le.ExplanationNumericalError):
            engine.explain(SAMPLE_REQUEST)

        def skewed(*a, **k):
            return original(*a, **k) + 0.1

        engine._ig.attribute = skewed
        with self.assertRaises(le.ExplanationNumericalError):
            engine.explain(SAMPLE_REQUEST)

    def test_parity_failure_is_detected(self):
        runtime = _STATE["runtime"]

        class Shifted:
            def __getattr__(self, name):
                return getattr(runtime, name)

            def predict(self, payload):
                out = runtime.predict(payload)
                out["probability"] += 1e-4
                return out

        engine = le.ExplanationEngine(Shifted())
        with self.assertRaises(le.ExplanationNumericalError):
            engine.explain(SAMPLE_REQUEST)

    def test_engine_failure_is_wrapped(self):
        engine = le.ExplanationEngine(_STATE["runtime"])

        def boom(*a, **k):
            raise RuntimeError("boom")

        engine._ig.attribute = boom
        with self.assertRaises(le.ExplanationNumericalError):
            engine.explain(SAMPLE_REQUEST)

    def test_bad_standardized_tensors(self):
        engine = _STATE["engine"]
        bad = [
            torch.zeros(5), torch.zeros(2, 5), torch.zeros(1, 4), torch.zeros(1, 5, dtype=torch.float64),
            torch.tensor([[0.0, float("nan"), 0.0, 0.0, 0.0]]), torch.tensor([[float("inf"), 0, 0, 0, 0]]),
        ]
        for tensor in bad:
            with self.assertRaises(le.InvalidExplanationInput):
                engine._explain_standardized(tensor)
        with self.assertRaises(le.InvalidExplanationInput):
            engine._explain_standardized([[0.0] * 5])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.engine = _STATE["engine"]

    def test_invalid_inputs_are_rejected(self):
        base = dict(SAMPLE_REQUEST)
        cases = {
            "missing": {k: v for k, v in base.items() if k != "prior_injury"},
            "extra": dict(base, baseline=0),
            "nan": dict(base, weekly_km_to_date=float("nan")),
            "inf": dict(base, resting_hr_morning=float("inf")),
            "string": dict(base, sleep_hours_last_night="7"),
            "none": dict(base, soreness_score_evening=None),
            "bool_float_field": dict(base, weekly_km_to_date=True),
            "bool_injury": dict(base, prior_injury=True),
            "float_injury": dict(base, prior_injury=0.0),
            "fractional_injury": dict(base, prior_injury=0.5),
            "baseline_fractional_injury": dict(base, prior_injury=1 / 6),
            "huge_int_km": dict(base, weekly_km_to_date=10**400),
            "huge_negative_int_hr": dict(base, resting_hr_morning=-(10**400)),
            "string_injury": dict(base, prior_injury="0"),
            "injury_2": dict(base, prior_injury=2),
            "injury_neg": dict(base, prior_injury=-1),
            "km_above_api": dict(base, weekly_km_to_date=200.5),
            "hr_below_api": dict(base, resting_hr_morning=29.9),
            "sleep_above_api": dict(base, sleep_hours_last_night=14.01),
            "not_mapping": [1, 2, 3],
        }
        for label, payload in cases.items():
            with self.subTest(case=label), self.assertRaises(le.InvalidExplanationInput):
                self.engine.explain(payload)

    def test_oversized_integers_map_to_invalid_input(self):
        for name in ("weekly_km_to_date", "sleep_hours_last_night", "resting_hr_morning", "soreness_score_evening"):
            for value in (10**400, -(10**400), 10**309, -(10**309)):
                with self.subTest(feature=name, value_digits=len(str(abs(value)))):
                    with self.assertRaises(le.InvalidExplanationInput):
                        self.engine.explain(dict(SAMPLE_REQUEST, **{name: value}))
            with self.assertRaises(le.InvalidExplanationInput):  # large but convertible: range check
                self.engine.explain(dict(SAMPLE_REQUEST, **{name: 10**30}))
        with self.assertRaises(le.InvalidExplanationInput):
            self.engine.explain(dict(SAMPLE_REQUEST, prior_injury=10**400))

    def test_ordinary_integers_are_accepted_for_continuous_features(self):
        payload = dict(SAMPLE_REQUEST, weekly_km_to_date=30, sleep_hours_last_night=7,
                       resting_hr_morning=50, soreness_score_evening=4)
        result = self.engine.explain(payload)
        self.assertEqual(result.attributions[0].value, 30.0)
        self.assertEqual(len(result.attributions), 5)

    def test_out_of_domain_is_a_distinct_error_without_echoing_values(self):
        payload = dict(SAMPLE_REQUEST, weekly_km_to_date=19.999, resting_hr_morning=57.001)
        with self.assertRaises(le.OutsideExplanationDomain) as ctx:
            self.engine.explain(payload)
        violations = ctx.exception.violations
        self.assertEqual([v["feature"] for v in violations], ["weekly_km_to_date", "resting_hr_morning"])
        self.assertEqual(violations[0], {"feature": "weekly_km_to_date", "min": 20.0, "max": 45.0})
        self.assertNotIn("19.999", repr(violations) + str(ctx.exception))
        self.assertEqual(ctx.exception.code, "outside_explanation_domain")

    def test_boundaries_are_inclusive(self):
        for name, (lo, hi) in le.EXPLANATION_RANGES.items():
            for value in (lo, hi):
                payload = dict(SAMPLE_REQUEST)
                payload[name] = int(value) if name == "prior_injury" else value
                with self.subTest(feature=name, value=value):
                    result = self.engine.explain(payload)
                    self.assertEqual(len(result.attributions), 5)

    def test_just_outside_each_boundary_is_out_of_domain(self):
        eps = 1e-6
        for name, (lo, hi) in le.EXPLANATION_RANGES.items():
            if name in ("soreness_score_evening", "prior_injury"):
                continue  # the explanation range equals the prediction range
            for value in (lo - eps, hi + eps):
                with self.subTest(feature=name, value=value), self.assertRaises(le.OutsideExplanationDomain):
                    self.engine.explain(dict(SAMPLE_REQUEST, **{name: value}))

    def test_prediction_ranges_mirror_level7_service(self):
        text = (ROOT / "src" / "level7_service.py").read_text()
        for name, (lo, hi) in le.PREDICTION_RANGES.items():
            pattern = rf"{name}: Annotated\[\w+, Field\(ge=([\d.]+), le=([\d.]+)\)\]"
            match = re.search(pattern, text)
            self.assertIsNotNone(match, name)
            self.assertEqual((float(match.group(1)), float(match.group(2))), (lo, hi), name)


class InterfaceAndSigmoidTests(unittest.TestCase):
    def test_public_interface_is_explain_only(self):
        engine = _STATE["engine"]
        self.assertFalse(hasattr(engine, "explain_standardized"))
        self.assertTrue(callable(engine._explain_standardized))

    def test_public_result_is_strict_json_serializable(self):
        result = _STATE["engine"].explain(SAMPLE_REQUEST)
        text = json.dumps(result.to_dict(), allow_nan=False)
        decoded = json.loads(text)
        self.assertEqual([a["feature"] for a in decoded["attributions"]], list(le.EXPLANATION_RANGES))
        for a in result.attributions:
            self.assertTrue(all(math.isfinite(v) for v in (a.value, a.standardized_value, a.attribution)))
        self.assertEqual(result.attributions[4].value, 0.0)

    def test_internal_result_has_no_raw_values_and_is_finite(self):
        engine = _STATE["engine"]
        for x in (torch.zeros(1, 5), torch.full((1, 5), 0.7)):
            r = engine._explain_standardized(x)
            self.assertIsInstance(r, le._Numerics)
            self.assertFalse(hasattr(r, "prediction"))
            json.dumps(dataclasses.asdict(r), allow_nan=False)

    def test_public_explain_rejects_fractional_injury_even_at_baseline_values(self):
        runtime = _STATE["runtime"]
        baseline_raw = dict(zip(runtime.feature_names, map(float, runtime.preprocessor["scaler_mean"])))
        with self.assertRaises(le.InvalidExplanationInput):
            _STATE["engine"].explain(baseline_raw)

    def test_stable_sigmoid(self):
        for x in (0.0, 1.0, -1.0, 3.7, -3.7, 30.0, -30.0, 709.0, -709.0, 745.0, -745.0, 1e6, -1e6, 1e300, -1e300):
            value = le._sigmoid(x)
            self.assertTrue(math.isfinite(value) and 0.0 <= value <= 1.0, x)
        self.assertEqual(le._sigmoid(0.0), 0.5)
        self.assertEqual(le._sigmoid(1e6), 1.0)
        self.assertEqual(le._sigmoid(-1e6), 0.0)
        for x in (-12.5, -3.0, -0.3, 0.3, 2.2, 9.0):
            reference = float(torch.sigmoid(torch.tensor(x, dtype=torch.float64)))
            self.assertAlmostEqual(le._sigmoid(x), reference, places=14)
        self.assertAlmostEqual(le._sigmoid(-2.0) + le._sigmoid(2.0), 1.0, places=15)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(le.ExplanationNumericalError):
                le._sigmoid(bad)


class MetadataTests(unittest.TestCase):
    def test_metadata_fields(self):
        runtime, engine = _STATE["runtime"], _STATE["engine"]
        m = engine.metadata()
        json.dumps(m)  # serializable
        self.assertTrue(m["model"]["development_only"])
        self.assertEqual(m["model"]["selected_approach"], runtime.handoff["selected_approach"])
        self.assertEqual(m["model"]["handoff_version"], runtime.handoff["handoff_version"])
        self.assertEqual(m["model"]["integrity_status"], "passed")
        self.assertEqual(m["model"]["artifact_sha256"], runtime.manifest["artifacts"])
        self.assertEqual(m["method"]["library_version"], "0.9.0")
        self.assertEqual((m["method"]["integration"], m["method"]["n_steps"]), ("riemann_middle", 800))
        self.assertEqual(m["method"]["target"], "pre_sigmoid_logit")
        b = m["baseline"]
        self.assertEqual(b["id"], "B0")
        self.assertEqual(list(b["standardized_values"].values()), [0.0] * 5)
        self.assertAlmostEqual(b["raw_values"]["prior_injury"], 1 / 6, places=9)
        self.assertAlmostEqual(b["probability"], le._sigmoid(b["logit"]), places=12)
        self.assertEqual(m["completeness_tolerance"], {"absolute": 0.002, "relative": 0.005})
        self.assertIn("logit", m["units"])
        text = " ".join(m["notices"])
        for phrase in ("not causal", "fractional", "in-distribution", "real-world"):
            self.assertIn(phrase, text)


class IsolationTests(unittest.TestCase):
    def test_module_import_has_no_side_effects(self):
        code = (
            "import sys; import src.level8_explain; "
            "bad=[m for m in ('captum','matplotlib') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)"
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr.decode())

    def test_oracle_is_not_a_production_dependency(self):
        source = (ROOT / "src" / "level8_explain.py").read_text()
        self.assertNotIn("level8_oracle", source)
        self.assertNotIn("import tests", source)

    def test_runtime_model_and_bundle_unchanged_and_globals_untouched(self):
        runtime, engine = _STATE["runtime"], _STATE["engine"]
        _study()  # explain 520 inputs first
        for _ in range(3):
            engine.explain(SAMPLE_REQUEST)
        self.assertEqual(_snapshot(runtime), _STATE["before"])
        self.assertEqual(_torch_globals(), _STATE["globals_before"])
        self.assertTrue(all(p.requires_grad for p in runtime.model.parameters()))
        private = engine._model.model
        self.assertIsNot(private, runtime.model)
        self.assertFalse(any(p.requires_grad for p in private.parameters()))
        for a, b in zip(private.parameters(), runtime.model.parameters()):
            self.assertIsNot(a, b)
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(runtime.predict(SAMPLE_REQUEST), copy.deepcopy(_STATE["runtime"]).predict(SAMPLE_REQUEST))

    def test_lock_file_shape(self):
        lock = (ROOT / "requirements-level8.lock").read_text()
        pins = re.findall(r"^([a-z0-9-]+)==([\d.]+) \\\n\s+--hash=sha256:([0-9a-f]{64})", lock, re.M)
        self.assertEqual(
            sorted((n, v) for n, v, _ in pins),
            sorted([
                ("captum", "0.9.0"), ("contourpy", "1.4.0"), ("cycler", "0.12.1"),
                ("fonttools", "4.68.0"), ("kiwisolver", "1.5.1"), ("matplotlib", "3.11.2"),
                ("packaging", "26.3"), ("pillow", "13.0.0"), ("pyparsing", "3.3.3"), ("tqdm", "4.70.1"),
            ]),
        )
        self.assertEqual(len({h for _, _, h in pins}), 10)


class ZSerialPerformanceAndEvidence(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LEVEL8_PERF") == "1", "set LEVEL8_PERF=1 for the latency smoke test")
    def test_90_serial_latency_smoke(self):
        engine = _STATE["engine"]
        self.assertEqual(torch.get_num_threads(), 1)
        rows = _STATE["grids"]["dev_range"][:100]
        payloads = [_payload(r) for r in rows]
        for p in payloads[:30]:
            engine.explain(p)  # warm-up
        samples = []
        for _ in range(2):
            for p in payloads:
                t0 = time.perf_counter()
                engine.explain(p)
                samples.append((time.perf_counter() - t0) * 1000)
        a = np.array(samples)
        _PERF.update({
            "calls": len(samples), "torch_intra_op_threads": 1,
            "p50_ms": float(np.quantile(a, .5)), "p95_ms": float(np.quantile(a, .95)),
            "p99_ms": float(np.quantile(a, .99)), "max_ms": float(a.max()),
            "includes": "validation, ModelRuntime.preprocess, ModelRuntime.predict, IG (800 steps), residual check",
            "conditions": "canonical Linux/amd64 image; run with --cpus=1 when generating the committed evidence; 100 development-range inputs x2 after 30 warm-up calls; non-gating beyond the 25 ms p95 target",
        })
        print("L8_PERF", json.dumps(_PERF))
        self.assertLessEqual(_PERF["p95_ms"], LATENCY_P95_MS)

    def test_95_committed_evidence_is_consistent(self):
        if not EVIDENCE_PATH.is_file():
            self.skipTest("evidence file not present yet")
        doc = json.loads(EVIDENCE_PATH.read_text())
        self.assertEqual(doc["acceptance_criteria"], _criteria())
        self.assertEqual(doc["model_identity"]["artifact_sha256"], _STATE["runtime"].manifest["artifacts"])
        text = json.dumps(doc)
        for forbidden in ("/Users/", "/private/", "/tmp/", "/app/", "/workspace"):
            self.assertNotIn(forbidden, text)
        fresh = _l8_results()
        self.assertIs(doc["grid_construction"]["cross_check"]["reproducible_from_repository"], False)
        self.assertIn("HISTORICAL", doc["grid_construction"]["cross_check"]["provenance"])
        self.assertIn("HISTORICAL", doc["historical_f2b_results"]["label"])
        for grid, summary in fresh.items():
            self.assertEqual(doc["l8_1_results"][grid]["all_gates_pass_per_input"], summary["all_gates_pass_per_input"])
            self.assertAlmostEqual(
                doc["l8_1_results"][grid]["max_abs_residual"], summary["max_abs_residual"], places=9
            )

    def test_99_write_evidence_if_requested(self):
        target = os.environ.get("LEVEL8_EVIDENCE_OUT")
        if not target:
            self.skipTest("LEVEL8_EVIDENCE_OUT not set")
        Path(target).write_text(json.dumps(build_report(), indent=2, sort_keys=False) + "\n")


def _criteria() -> dict:
    return {
        "feature_count": 5,
        "completeness": "abs(residual) <= 0.002 + 0.005 * abs(target_difference)",
        "max_single_feature_abs_error_vs_oracle": FEATURE_ERR_MAX,
        "relative_L1_error_vs_oracle": {"max": L1_REL_MAX, "denominator_floor": L1_FLOOR},
        "sign_agreement": f"where abs(oracle attribution) > {SIGN_MIN}",
        "top_feature": f"engine top-1 within oracle set with abs >= max abs - {TIE_TOL}",
        "prediction_logit_parity_tolerance": PARITY,
        "serial_engine_p95_latency_ms": LATENCY_P95_MS,
    }


def _l8_results() -> dict:
    out = {}
    for name, s in _study().items():
        allp = s["resid_ok"] & (s["feat_err"] <= FEATURE_ERR_MAX) & (s["rel_l1"] <= L1_REL_MAX) & s["sign_ok"] & s["top_ok"]
        out[name] = {
            "n_inputs": s["n"], "all_gates_pass_per_input": int(allp.sum()),
            "finite_five_attributions_in_canonical_order": bool(s["finite"] and s["names_ok"]),
            "max_abs_residual": float(np.abs(s["resid"]).max()),
            "max_residual_over_allowed": float((np.abs(s["resid"]) / s["allowed"]).max()),
            "max_feature_abs_error": float(s["feat_err"].max()),
            "p99_feature_abs_error": float(np.quantile(s["feat_err"], .99)),
            "max_relative_L1": float(s["rel_l1"].max()),
            "material_sign_comparisons": s["material"],
            "strict_top1_agreement": s["strict_agree"],
            "inputs_with_oracle_ties": s["ties"],
            "max_probability_parity_error": float(s["parity"].max()),
        }
    return out


def build_report() -> dict:
    runtime, g = _STATE["runtime"], _STATE["grids"]
    return {
        "schema_version": 1,
        "scope": (
            "Numerical verification of the Level 8 Integrated Gradients engine against an independent "
            "exact oracle. It is NOT evidence of model validity, of real-world predictive value, or of "
            "production performance."
        ),
        "provenance": {
            "historical_f2b": "Feasibility experiment F2B (scratch harness, not in this repository). Reported unchanged, for context only.",
            "l8_1": "Level 8 milestone L8-1: tests/test_level8_engine.py run against src/level8_explain.py in the canonical image.",
            "holdout_data_used": False,
            "inputs": "Synthetic grids only (see grid_construction).",
        },
        "model_identity": {
            "selected_approach": runtime.handoff["selected_approach"],
            "handoff_version": runtime.handoff["handoff_version"],
            "development_only": True,
            "artifact_sha256": dict(runtime.manifest["artifacts"]),
        },
        "numerical_settings": {
            "library": "captum", "library_version": "0.9.0", "method": "integrated_gradients",
            "integration": le.METHOD, "n_steps": le.N_STEPS, "target": le.TARGET,
            "baseline": "B0 (standardized zero vector)", "one_input_per_call": True,
            "torch_intra_op_threads_in_verification": 1,
            "oracle": "exact float64 piecewise-linear IG (tests/level8_oracle.py)",
        },
        "grid_construction": {
            "generator": "numpy.random.default_rng(42)",
            "draw_order": ["dev uniform", "dev prior_injury", "wide uniform", "wide prior_injury", "pool uniform", "pool prior_injury"],
            "dev_range": {"n": oracle.N_DEV, "lo": oracle.DEV_LO.tolist(), "hi": oracle.DEV_HI.tolist(),
                          "prior_injury": "Bernoulli(1/6)"},
            "straddle_threshold": {"n": oracle.N_STRADDLE, "pool": oracle.N_POOL,
                                   "selection": "smallest abs(logit) from a development-range pool"},
            "api_corners": "32 corners of the Level 7 API bounds (digest only; not explained)",
            "api_wide": "100 uniform API-bound points (digest only; not explained)",
            "cross_check": {
                "provenance": "HISTORICAL, MANUALLY RECORDED. Not produced by the test suite and not reproducible from this repository.",
                "what": "A one-time comparison, made during L8-1, of the four digests above against grids built by the original external F2B scratch harness (which is not part of this repository). They were bit-identical.",
                "regenerated_by_tests": False,
                "reproducible_from_repository": False,
                "what_the_repository_can_check": "The recorded digests are re-verified against the repository grid builder (tests/level8_oracle.py) by test_grid_digests_match_committed_evidence; that check does not involve the F2B harness.",
            },
            "sha256_float64": {n: oracle.grid_digest(g[n]) for n in ("dev_range", "straddle_threshold", "api_corners", "api_wide")},
        },
        "acceptance_criteria": _criteria(),
        "historical_f2b_results": {
            "label": "HISTORICAL. Scratch harness, same configuration, 1 torch thread.",
            "dev_range": {"n_inputs": 500, "all_gates_pass_per_input": 500, "max_abs_residual": 0.0016907751560211182,
                          "max_residual_over_allowed": 0.706366012562183, "max_feature_abs_error": 0.0009215216473620891,
                          "max_relative_L1": 0.0013642862162892668, "material_sign_comparisons": 2402,
                          "strict_top1_agreement": 1.0, "inputs_with_oracle_ties": 2},
            "straddle_threshold": {"n_inputs": 20, "all_gates_pass_per_input": 20, "max_abs_residual": 0.0009660078212618828,
                                   "max_residual_over_allowed": 0.1328605131802433, "max_feature_abs_error": 0.0009486701671095332,
                                   "max_relative_L1": 0.0006190904722865397, "material_sign_comparisons": 93,
                                   "strict_top1_agreement": 1.0, "inputs_with_oracle_ties": 0},
            "determinism": "Identical attributions across repeated calls, across 1/2/4/8 torch threads and 5 processes (F2).",
            "excluded": "Gauss-Legendre timings and runs with invalid latency measurements are deliberately not reproduced here.",
        },
        "l8_1_results": {
            "label": "CURRENT. Regenerated by the test suite against the corrected engine (L8-1 as amended by L8-1R).",
            **_l8_results(),
        },
        "l8_1r_corrections": {
            "label": "Corrections made after the L8-1 pre-commit review. Numerical settings and tolerances are unchanged.",
            "items": [
                "Oversized integers in continuous fields are rejected as invalid input instead of escaping as OverflowError.",
                "explain_standardized is now the private _explain_standardized and returns finite floats only; the public result is built in explain() from validated input and serializes with allow_nan=False.",
                "Probability is computed with a numerically stable sigmoid; non-finite logits fail closed.",
                "The grid cross-check is recorded explicitly as a historical, manually recorded comparison.",
            ],
            "accepted_limitations": [
                "numpy float32 scalars are rejected at the public Python interface.",
                "Mirrored Level 7 prediction bounds are synchronized by a regex test over the service source.",
                "The shared-engine concurrency collapse has no known root cause; serialized execution only mitigates it.",
                "Mixed-load performance is not verified.",
                "Image-layout (flat /app) overlay verification is deferred to integration.",
            ],
        },
        "historical_performance_reference": {
            "label": "HISTORICAL, machine-specific, non-gating. Not a production claim.",
            "800_steps_1_cpu_1_thread_ms": {"p50": 14.16, "p95": 17.99},
            "800_steps_4_threads_ms": {"p50": 14.60, "p95": 20.72},
            "800_steps_8_threads_ms": {"p50": 27.69, "p95": 48.69},
            "shared_engine_concurrency": "Sharing one engine across concurrent callers collapsed to ~750 ms per call; root cause unexplained. Serialized execution was verified to avoid it.",
            "serial_48_requests_p95_ms": 18.37,
        },
        "l8_1_performance_smoke": _PERF or "not run in the generating invocation",
        "notices": list(le.NOTICES),
    }


if __name__ == "__main__":
    unittest.main()
