"""Level 8 L8-3B: pure deterministic comparison and interpretation tests.

Two kinds of ``ExplanationResponse`` fixtures are used and kept apart:

* REAL: the committed engine's output for a synthetic input, assembled and checked by the committed
  trusted path (``level8_schemas.build_response`` against the parent runtime's reference).
* SYNTHETIC: hand-chosen numbers placed in a copy of a real response so that exact boundaries
  (materiality, ties, probability 0.5) can be constructed. Shared metadata, including B0, is the real
  reference-derived metadata. These are NOT model outputs.

All inputs are synthetic. No training or holdout data is read.

The Level 7 fixture serving root is reused from tests.test_level7_service.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from pydantic import ValidationError

from tests import test_level7_service as t7
from tests import level8_oracle as oracle
from tests import test_level8_engine as et  # module reference only: its test classes are not collected here
from src import level8_compare as c
from src import level8_schemas as schemas
from src import level8_worker as w
from src.level8_explain import ExplanationEngine, InvalidExplanationInput

ROOT = Path(__file__).resolve().parents[1]
L7 = t7.service
FEATURES = schemas.FEATURES
SAMPLE = dict(t7.VALID_REQUEST)
_STATE: dict = {}


def row(km, sleep, rhr, soreness, prior):
    return dict(zip(FEATURES, (float(km), float(sleep), float(rhr), float(soreness), int(prior))))


# --------------------------------------------------------------- frozen numerical fixture matrix
# Fixed before the numerical tests were first run. "T<i>" is row i of the committed, seeded,
# synthetic threshold-straddling grid (tests.level8_oracle.build_grids); rows are taken by index.
LOW = row(20.0, 5.5, 45.0, 0.0, 0)
HIGH = row(45.0, 8.0, 57.0, 10.0, 1)
MIXED_A = row(25.0, 5.8, 47.0, 2.0, 1)
MIXED_B = row(38.0, 6.5, 52.0, 6.0, 0)
IDENTICAL_PAIRS = (("identical_sample", "S", "S"), ("identical_threshold", "T0", "T0"))
DISTINCT_PAIRS = (
    ("single_km", "S", dict(SAMPLE, weekly_km_to_date=40.0)),
    ("single_sleep", "S", dict(SAMPLE, sleep_hours_last_night=6.0)),
    ("single_resting_hr", "S", dict(SAMPLE, resting_hr_morning=55.0)),
    ("single_soreness", "S", dict(SAMPLE, soreness_score_evening=8.0)),
    ("single_prior_injury", "S", dict(SAMPLE, prior_injury=1)),
    ("multi_from_sample", "S", row(40.0, 6.0, 54.0, 7.0, 0)),
    ("multi_all_with_prior", MIXED_A, row(42.0, 7.8, 56.0, 9.0, 0)),
    ("multi_prior_toggle_on", MIXED_B, row(30.0, 7.5, 48.0, 3.0, 1)),
    ("boundary_low_to_high", LOW, HIGH),
    ("boundary_sample_to_low", "S", LOW),
    ("near_threshold_a", "T0", "T1"),
    ("near_threshold_b", "T2", "T3"),
)
PAIR_BOUND = 2 * et.FEATURE_ERR_MAX  # 0.01: two individual 0.005 per-feature limits; never loosened


def setUpModule():
    saved = torch.get_num_threads()
    torch.set_num_threads(1)  # the worker child's setting; restored in tearDownModule
    _STATE["threads"] = saved
    engine = ExplanationEngine(L7.runtime)
    _STATE["engine"] = engine
    _STATE["metadata"] = json.loads(json.dumps(engine.metadata()))
    _STATE["reference"] = schemas.make_reference(L7.runtime)
    manifest = L7.runtime.manifest
    _STATE["parent"] = {k: manifest[k] for k in
                        ("evidence_scope", "model_notice", "inherited_v1_preprocessing_limitation")}
    _STATE["cache"] = {}


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])


def real(payload) -> schemas.ExplanationResponse:
    """REAL fixture through the committed trusted validation path."""
    key = json.dumps(payload, sort_keys=True)
    if key not in _STATE["cache"]:
        clean = ExplanationEngine.validate_payload(payload)
        wire = w._wire_form(_STATE["engine"].explain(clean))
        _STATE["cache"][key] = schemas.build_response(wire, _STATE["metadata"], _STATE["parent"],
                                                      _STATE["reference"], clean)
    return _STATE["cache"][key]


def synthetic(values=None, others=(0.0, 0.0, 0.0, 0.0), logit=0.25, residual=0.0, probability=None):
    """SYNTHETIC fixture (not a model output). ``others`` are the attributions of features 1..4;
    the first attribution is solved so that the sum equals logit - B0 + residual."""
    body = real(SAMPLE).model_dump(mode="python")
    values = dict(SAMPLE if values is None else values)
    reference = _STATE["reference"]
    b0 = body["baseline"]["logit"]
    difference = logit - b0
    attributions = [difference + residual - math.fsum(others), *others]
    total = math.fsum(attributions)
    for i, (item, name) in enumerate(zip(body["attributions"], FEATURES)):
        item["value"] = float(values[name])
        item["standardized_value"] = float(np.float32((float(values[name]) - reference.means[i]) / reference.scales[i]))
        item["attribution"] = attributions[i]
    probability = schemas.stable_sigmoid(logit) if probability is None else probability
    body["prediction"].update(probability=probability, decision=int(probability >= 0.5), logit=logit,
                              decision_margin=logit)
    body["completeness"].update(target_difference=difference, attribution_sum=total, residual=total - difference,
                                allowed_residual=0.002 + 0.005 * abs(difference))
    return schemas.ExplanationResponse.model_validate(body)


def mutated(response, path, value):
    """A structurally valid copy with one field replaced."""
    body = response.model_dump(mode="python")
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return schemas.ExplanationResponse.model_validate(body)


def feature_row(result, name):
    return result.features[FEATURES.index(name)]


def all_floats(node):
    if isinstance(node, float):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from all_floats(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from all_floats(value)


def is_negative_zero(value):
    return value == 0.0 and math.copysign(1.0, value) < 0


# ------------------------------------------------------------------------------ module purity
class PurityTests(unittest.TestCase):
    def test_import_starts_nothing_and_loads_no_service_worker_or_captum(self):
        code = (
            "import json, multiprocessing, sys, threading; import src.level8_compare;"
            "names=('captum','matplotlib','fastapi','src.level7_service','src.level7_runtime','src.level8_service',"
            "'src.level8_worker','src.level8_dispatch');"
            "print(json.dumps([[m for m in names if m in sys.modules], len(multiprocessing.active_children()),"
            "[t.name for t in threading.enumerate()]]))"
        )
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout.strip().splitlines()[-1]), [[], 0, ["MainThread"]])

    def test_builder_does_not_modify_its_arguments(self):
        a, b = real(SAMPLE), real(MIXED_B)
        before = (a.model_dump_json(), b.model_dump_json())
        c.build_comparison(a, b)
        self.assertEqual((a.model_dump_json(), b.model_dump_json()), before)


# ---------------------------------------------------------------------------------- request
class RequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Model = c.make_comparison_request_model(L7.PredictionRequest)

    def rejected(self, body):
        with self.assertRaises(ValidationError) as caught:
            self.Model.model_validate(body)
        return caught.exception.errors()

    def test_reuses_the_existing_strict_class_for_both_sides(self):
        self.assertEqual(tuple(self.Model.model_fields), ("original", "modified"))
        parsed = self.Model.model_validate({"original": SAMPLE, "modified": MIXED_B})
        self.assertIs(type(parsed.original), L7.PredictionRequest)
        self.assertIs(type(parsed.modified), L7.PredictionRequest)
        from_json = self.Model.model_validate_json(json.dumps({"original": SAMPLE, "modified": MIXED_B}))
        self.assertEqual(from_json, parsed)

    def test_identical_scenarios_are_valid(self):
        parsed = self.Model.model_validate({"original": SAMPLE, "modified": dict(SAMPLE)})
        self.assertEqual(parsed.original, parsed.modified)

    def test_missing_unknown_and_null_at_both_levels(self):
        good = {"original": SAMPLE, "modified": SAMPLE}
        self.rejected(dict(good, extra=1))
        for side in c.SCENARIOS:
            self.rejected({k: v for k, v in good.items() if k != side})
            for bad in (None, {}, [], "x", 1):
                with self.subTest(side=side, bad=bad):
                    self.rejected(dict(good, **{side: bad}))
            errors = self.rejected(dict(good, **{side: dict(SAMPLE, extra=1)}))
            self.assertEqual([(e["type"], e["loc"]) for e in errors], [("extra_forbidden", (side, "extra"))])
            for name in FEATURES:
                with self.subTest(side=side, name=name):
                    errors = self.rejected(dict(good, **{side: {k: v for k, v in SAMPLE.items() if k != name}}))
                    self.assertEqual([(e["type"], e["loc"]) for e in errors], [("missing", (side, name))])
                    self.rejected(dict(good, **{side: dict(SAMPLE, **{name: None})}))

    def test_nested_strictness_rejects_booleans_strings_and_non_finite(self):
        for side in c.SCENARIOS:
            for name in FEATURES:
                for bad in (True, False, "7", float("nan"), float("inf"), float("-inf")):
                    with self.subTest(side=side, name=name, bad=bad):
                        body = {"original": SAMPLE, "modified": SAMPLE, side: dict(SAMPLE, **{name: bad})}
                        self.rejected(body)
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            text = json.dumps({"original": SAMPLE, "modified": SAMPLE}).replace("32.0", token, 1)
            with self.subTest(token=token), self.assertRaises(ValidationError):
                self.Model.model_validate_json(text)

    def test_prior_injury_is_a_strict_integer_zero_or_one(self):
        for bad in (0.0, 1.0, 2, -1, "1", True):
            for side in c.SCENARIOS:
                with self.subTest(side=side, bad=bad):
                    self.rejected({"original": SAMPLE, "modified": SAMPLE, side: dict(SAMPLE, prior_injury=bad)})

    def test_equivalent_numeric_representations_are_unchanged(self):
        integers = dict(SAMPLE, weekly_km_to_date=32, sleep_hours_last_night=7, resting_hr_morning=50,
                        soreness_score_evening=4)
        parsed = self.Model.model_validate({"original": integers, "modified": SAMPLE})
        self.assertEqual(parsed.original.model_dump(), parsed.modified.model_dump())
        result = c.build_comparison(real(parsed.original.model_dump()), real(parsed.modified.model_dump()))
        self.assertEqual(result.differences.changed_features, [])

    def test_out_of_range_is_not_clamped(self):
        self.rejected({"original": dict(SAMPLE, weekly_km_to_date=200.0001), "modified": SAMPLE})
        parsed = self.Model.model_validate({"original": dict(SAMPLE, weekly_km_to_date=46.0), "modified": SAMPLE})
        self.assertEqual(parsed.original.weekly_km_to_date, 46.0)

    def test_factory_refuses_other_models(self):
        from pydantic import BaseModel, ConfigDict, create_model

        lax = create_model("Lax", **{n: (float, ...) for n in FEATURES})
        short = create_model("Short", __config__=ConfigDict(strict=True, extra="forbid", allow_inf_nan=False),
                             weekly_km_to_date=(float, ...))
        for bad in (lax, short, BaseModel, dict, None):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                c.make_comparison_request_model(bad)


class EligibilityTests(unittest.TestCase):
    def test_inclusive_boundaries_pass_and_just_outside_fails(self):
        for name, (low, high) in schemas.EXPLANATION_RANGES.items():
            if name == "prior_injury":
                continue
            for edge, beyond in ((low, -math.inf), (high, math.inf)):
                inside = dict(SAMPLE, **{name: edge})
                self.assertEqual(c.eligibility_violations(inside, inside), ())
                if name == "soreness_score_evening":
                    continue  # its explanation range equals the prediction range
                outside = dict(SAMPLE, **{name: math.nextafter(edge, beyond)})
                for side, pair in (("original", (outside, SAMPLE)), ("modified", (SAMPLE, outside))):
                    with self.subTest(name=name, edge=edge, side=side):
                        self.assertEqual(c.eligibility_violations(*pair),
                                         ({"scenario": side, "feature": name, "min": low, "max": high},))

    def test_both_scenarios_in_deterministic_order_without_values(self):
        original = dict(SAMPLE, resting_hr_morning=58.25, weekly_km_to_date=19.25)
        modified = dict(SAMPLE, sleep_hours_last_night=8.25)
        found = c.eligibility_violations(original, modified)
        self.assertEqual([(v["scenario"], v["feature"]) for v in found],
                         [("original", "weekly_km_to_date"), ("original", "resting_hr_morning"),
                          ("modified", "sleep_hours_last_night")])
        self.assertTrue(all(set(v) == {"scenario", "feature", "min", "max"} for v in found))
        for value in ("19.25", "58.25", "8.25"):
            self.assertNotIn(value, json.dumps(found))

    def test_invalid_prediction_input_raises_even_after_a_domain_violation(self):
        with self.assertRaises(InvalidExplanationInput):
            c.eligibility_violations(dict(SAMPLE, weekly_km_to_date=19.0), dict(SAMPLE, prior_injury=True))


# --------------------------------------------------------------------------- response contract
class ResponseContractTests(unittest.TestCase):
    TOP = ("schema_version", "shared", "original", "modified", "differences", "features", "ranking",
           "completeness_of_difference", "interpretation_policy", "notices")

    def setUp(self):
        self.result = c.build_comparison(real(SAMPLE), real(MIXED_B))
        self.body = self.result.model_dump(mode="json")

    def invalid(self, body):
        with self.assertRaises(ValidationError):
            c.ComparisonResponse.model_validate_json(json.dumps(body))

    def test_required_top_level_fields_and_round_trip(self):
        self.assertEqual(tuple(self.body), self.TOP)
        text = json.dumps(self.body, allow_nan=False)
        self.assertEqual(c.ComparisonResponse.model_validate_json(text), self.result)
        self.assertTrue(all(math.isfinite(v) for v in all_floats(self.body)))
        for key in self.TOP:
            with self.subTest(missing=key):
                self.invalid({k: v for k, v in self.body.items() if k != key})

    def test_unknown_fields_rejected_at_every_level(self):
        paths = ((), ("shared",), ("shared", "model"), ("original",), ("original", "prediction"),
                 ("modified", "completeness"), ("differences",), ("differences", "decision"), ("features", 0),
                 ("features", 0, "input"), ("features", 0, "attribution"), ("completeness_of_difference",),
                 ("interpretation_policy",))
        for path in paths:
            body = copy.deepcopy(self.body)
            node = body
            for key in path:
                node = node[key]
            node["probability_percentage_points"] = 1.0
            with self.subTest(path=path):
                self.invalid(body)

    def test_no_percentage_point_field_exists(self):
        self.assertNotIn("percent", json.dumps(c.ComparisonResponse.model_json_schema()).lower())

    def test_required_model_and_safety_metadata_is_preserved(self):
        source = real(SAMPLE).model_dump(mode="json")
        shared = self.body["shared"]
        for key in ("model", "method", "baseline", "units", "eligibility"):
            self.assertEqual(shared[key], source[key])
        self.assertEqual(shared["completeness_tolerance"], source["completeness"]["tolerance"])
        for key in ("threshold", "device", "selected_approach"):
            self.assertEqual(shared[key], source["prediction"][key])
        self.assertEqual(shared["baseline"]["logit"], _STATE["metadata"]["baseline"]["logit"])
        self.assertTrue(0.0 < shared["baseline"]["raw_values"]["prior_injury"] < 1.0)
        self.assertEqual(set(shared["baseline"]["standardized_values"].values()), {0.0})

    def test_values_are_copied_unrounded(self):
        a, b = real(SAMPLE), real(MIXED_B)
        for item_a, item_b, result_row in zip(a.attributions, b.attributions, self.result.features):
            self.assertEqual((result_row.attribution.original, result_row.attribution.modified),
                             (item_a.attribution, item_b.attribution))
            self.assertEqual(result_row.attribution.delta, item_b.attribution - item_a.attribution)
            self.assertEqual((result_row.standardized.original, result_row.standardized.modified),
                             (item_a.standardized_value, item_b.standardized_value))
        self.assertEqual(self.result.original.prediction.probability, a.prediction.probability)
        self.assertEqual(self.result.modified.prediction.decision_margin, b.prediction.decision_margin)

    def test_feature_order_missing_and_duplicate_rows_rejected(self):
        rows = self.body["features"]
        for bad in (rows[::-1], rows[:4], rows[:4] + [rows[0]], rows + [rows[0]]):
            self.invalid(dict(self.body, features=bad))

    def test_self_contradictory_payloads_rejected(self):
        def change(path, value):
            body = copy.deepcopy(self.body)
            node = body
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            return body

        cases = (
            (("differences", "probability"), 0.5), (("differences", "logit"), 9.0),
            (("differences", "changed_features"), []), (("differences", "decision", "changed"),
                                                        not self.result.differences.decision.changed),
            (("differences", "direction"), "original_minus_modified"),
            (("features", 1, "attribution", "delta"), 9.0), (("features", 1, "input", "delta"), 9.0),
            (("features", 1, "input", "changed"), False), (("features", 1, "delta_direction"), "bogus"),
            (("ranking",), ["prior_injury"] * 5), (("completeness_of_difference", "residual"), 0.5),
            (("completeness_of_difference", "residual_bound"), 9.0),
            (("interpretation_policy", "materiality_threshold_logit"), 0.01),
            (("interpretation_policy", "version"), "comparison-interpretation.v2"),
            (("schema_version",), "comparison.v2"), (("notices",), list(c.COMPARISON_NOTICES[:-1])),
            (("original", "prediction", "probability"), "0.5"), (("original", "prediction", "decision"), True),
        )
        for path, value in cases:
            with self.subTest(path=path):
                self.invalid(change(path, value))
        for token in ("NaN", "Infinity"):
            body = copy.deepcopy(self.body)
            body["differences"]["logit"] = "@"
            with self.subTest(token=token), self.assertRaises(ValidationError):
                c.ComparisonResponse.model_validate_json(json.dumps(body).replace('"@"', token))

    def test_policy_version_and_notices_are_frozen(self):
        self.assertEqual(c.POLICY_VERSION, "comparison-interpretation.v1")
        self.assertEqual(c.MATERIALITY_THRESHOLD_LOGIT, 0.02)
        self.assertEqual(tuple(self.result.notices), c.COMPARISON_NOTICES)
        self.assertEqual(c.COMPARISON_NOTICES[:9], schemas.INTERPRETATION_NOTICES)
        self.assertEqual(len(c.COMPARISON_NOTICES), 13)
        self.assertEqual(c.COMPARISON_ONLY_NOTICES, (
            "All comparison differences are the modified scenario minus the original scenario.",
            "Comparing hypothetical inputs describes this model only; it does not predict what would happen if "
            "a person changed them.",
            "Changing one input can change the attributions of other features, including features whose inputs "
            "did not change.",
            "The probability is not a validated measure of individual certainty, and a change in the model's "
            "decision has no demonstrated real-world meaning.",
        ))
        text = " ".join(c.COMPARISON_ONLY_NOTICES).lower()
        for word in ("safe", "recovered", "medically", "injury risk", "better", "worse", "should"):
            self.assertNotIn(word, text)


# ------------------------------------------------------------------------------- arithmetic
class ArithmeticTests(unittest.TestCase):
    def assert_all_zero(self, result):
        self.assertEqual((result.differences.probability, result.differences.logit), (0.0, 0.0))
        for r in result.features:
            self.assertEqual((r.input.delta, r.attribution.delta, r.input.changed, r.delta_direction),
                             (0.0, 0.0, False, "negligible"))
            self.assertFalse(is_negative_zero(r.input.delta) or is_negative_zero(r.attribution.delta))
        done = result.completeness_of_difference
        self.assertEqual((done.logit_difference, done.attribution_delta_sum, done.residual), (0.0, 0.0, 0.0))
        self.assertEqual((result.differences.changed_features, result.ranking), ([], []))
        self.assertFalse(result.differences.decision.changed)

    def test_same_validated_explanation_gives_exact_zeros(self):
        response = real(SAMPLE)
        self.assert_all_zero(c.build_comparison(response, response))

    def test_single_and_multiple_changes(self):
        one = c.build_comparison(real(SAMPLE), real(dict(SAMPLE, sleep_hours_last_night=6.0)))
        self.assertEqual(one.differences.changed_features, ["sleep_hours_last_night"])
        self.assertEqual(feature_row(one, "sleep_hours_last_night").input.delta, -1.0)
        many = c.build_comparison(real(MIXED_A), real(row(42.0, 7.8, 56.0, 9.0, 0)))
        self.assertEqual(many.differences.changed_features, list(FEATURES))
        for result in (one, many):
            self.assertEqual(result.differences.logit,
                             result.modified.prediction.logit - result.original.prediction.logit)
            self.assertEqual(result.differences.probability,
                             result.modified.prediction.probability - result.original.prediction.probability)

    def test_prior_injury_as_the_only_change(self):
        result = c.build_comparison(real(SAMPLE), real(dict(SAMPLE, prior_injury=1)))
        self.assertEqual(result.differences.changed_features, ["prior_injury"])
        self.assertEqual(feature_row(result, "prior_injury").input.delta, 1.0)

    def test_real_unchanged_input_can_have_a_changed_attribution(self):
        result = c.build_comparison(real(SAMPLE), real(dict(SAMPLE, sleep_hours_last_night=6.0)))
        moved = [r.feature for r in result.features if not r.input.changed and r.attribution.delta != 0.0]
        self.assertTrue(moved, "expected an interaction on at least one unchanged feature")

    def test_synthetic_unchanged_inputs_with_material_attribution_changes(self):
        result = c.build_comparison(synthetic(others=(0.1, 0.0, 0.0, 0.0)), synthetic(others=(0.4, 0.0, 0.0, 0.0)))
        self.assertEqual(result.differences.changed_features, [])
        self.assertIn("sleep_hours_last_night", result.ranking)

    def test_synthetic_changed_input_with_negligible_attribution_change(self):
        result = c.build_comparison(synthetic(), synthetic(values=dict(SAMPLE, sleep_hours_last_night=6.0),
                                                           others=(0.001, 0.0, 0.0, 0.0)))
        changed = feature_row(result, "sleep_hours_last_night")
        self.assertTrue(changed.input.changed)
        self.assertEqual((changed.attribution.delta, changed.delta_direction), (0.001, "negligible"))
        self.assertNotIn("sleep_hours_last_night", result.ranking)

    def test_reversed_direction_negates_every_difference(self):
        a, b = real(SAMPLE), real(MIXED_B)
        forward, backward = c.build_comparison(a, b), c.build_comparison(b, a)
        self.assertEqual(backward.differences.logit, -forward.differences.logit)
        self.assertEqual(backward.differences.probability, -forward.differences.probability)
        for f, g in zip(forward.features, backward.features):
            self.assertEqual((g.input.delta, g.attribution.delta), (-f.input.delta, -f.attribution.delta))
        self.assertEqual(backward.ranking, forward.ranking)
        self.assertEqual(backward.completeness_of_difference.residual, -forward.completeness_of_difference.residual)

    def test_all_four_decision_transitions(self):
        low, high = synthetic(logit=-0.4), synthetic(logit=0.4)
        for pair, name in (((low, low), "unchanged_0"), ((high, high), "unchanged_1"),
                           ((low, high), "0_to_1"), ((high, low), "1_to_0")):
            with self.subTest(name=name):
                decision = c.build_comparison(*pair).differences.decision
                self.assertEqual((decision.transition, decision.changed), (name, name in ("0_to_1", "1_to_0")))

    def test_probability_exactly_one_half_is_decision_one(self):
        result = c.build_comparison(synthetic(logit=-0.4), synthetic(logit=0.0))
        self.assertEqual(result.modified.prediction.probability, 0.5)
        self.assertEqual((result.modified.prediction.decision, result.differences.decision.transition), (1, "0_to_1"))

    def test_authoritative_decision_survives_a_logit_sign_disagreement_within_tolerance(self):
        edge = synthetic(logit=-1e-7, probability=0.5)  # authoritative 0.5 -> 1, private logit just below 0
        result = c.build_comparison(synthetic(logit=-0.4), edge)
        self.assertLess(result.modified.prediction.logit, 0.0)
        self.assertLess(result.modified.prediction.decision_margin, 0.0)
        self.assertEqual((result.modified.prediction.decision, result.differences.decision.transition), (1, "0_to_1"))

    def test_derived_negative_zero_is_normalized_and_copied_values_are_not_altered(self):
        original = synthetic(values=dict(SAMPLE, soreness_score_evening=0.0), others=(0.0, 0.0, 0.0, 0.0))
        modified = synthetic(values=dict(SAMPLE, soreness_score_evening=-0.0), others=(-0.0, 0.0, 0.0, 0.0))
        result = c.build_comparison(original, modified)
        soreness, sleep = feature_row(result, "soreness_score_evening"), feature_row(result, "sleep_hours_last_night")
        for value in (soreness.input.delta, sleep.attribution.delta, result.differences.logit,
                      result.completeness_of_difference.residual):
            self.assertFalse(is_negative_zero(value))
        self.assertTrue(is_negative_zero(sleep.attribution.modified))
        self.assertFalse(soreness.input.changed)

    def test_signed_residual_convention_matches_the_committed_engine(self):
        explanation = _STATE["engine"].explain(SAMPLE)
        self.assertEqual(explanation.completeness_residual,
                         explanation.attribution_sum - explanation.target_difference)

    def test_difference_residual_equals_difference_of_residuals(self):
        a = synthetic(logit=-0.2, residual=0.0003)
        b = synthetic(logit=0.3, residual=-0.0004, others=(0.25, 0.01, 0.2, 0.0))
        done = c.build_comparison(a, b).completeness_of_difference
        self.assertAlmostEqual(done.residual, -0.0007, delta=1e-12)
        self.assertEqual(done.residual_bound, a.completeness.allowed_residual + b.completeness.allowed_residual)
        self.assertNotEqual(done.attribution_delta_sum, done.logit_difference)


# --------------------------------------------------------------------------- interpretation
class InterpretationTests(unittest.TestCase):
    def test_boundaries_are_inclusive_negligible_on_both_sides(self):
        t = 0.02
        for value, expected in (
            (math.nextafter(t, 0.0), "negligible"), (t, "negligible"), (math.nextafter(t, 1.0), "toward_class_1"),
            (-math.nextafter(t, 0.0), "negligible"), (-t, "negligible"),
            (-math.nextafter(t, 1.0), "toward_class_0"), (0.0, "negligible"), (-0.0, "negligible"),
        ):
            with self.subTest(value=value):
                self.assertEqual(c.classify_contribution(value), expected)
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(c.ComparisonConsistencyError):
                c.classify_contribution(bad)

    def test_boundary_applies_to_contributions_and_differences_through_the_builder(self):
        above = math.nextafter(0.02, 1.0)
        result = c.build_comparison(synthetic(), synthetic(others=(0.02, above, -0.02, -above)))
        labels = [(r.modified_direction, r.delta_direction) for r in result.features[1:]]
        expected = ["negligible", "toward_class_1", "negligible", "toward_class_0"]
        self.assertEqual(labels, [(e, e) for e in expected])
        self.assertEqual([r.original_direction for r in result.features[1:]], ["negligible"] * 4)
        self.assertEqual([r.attribution.delta for r in result.features[1:]], [0.02, above, -0.02, -above])

    def test_threshold_is_not_applied_to_probability_logit_or_input(self):
        result = c.build_comparison(synthetic(logit=0.25), synthetic(
            logit=0.251, values=dict(SAMPLE, sleep_hours_last_night=7.001)))
        self.assertNotEqual(result.differences.logit, 0.0)
        self.assertNotEqual(result.differences.probability, 0.0)
        self.assertEqual(result.differences.changed_features, ["sleep_hours_last_night"])

    def test_exact_ties_and_opposite_signs_use_feature_order(self):
        result = c.build_comparison(synthetic(), synthetic(others=(-0.05, 0.05, 0.05, -0.05)))
        self.assertEqual(result.features[0].attribution.delta, 0.0)
        self.assertEqual(result.ranking, list(FEATURES[1:]))
        self.assertEqual(c.rank_material_differences([0.03, -0.5, 0.5, -0.03, 0.0]),
                         [FEATURES[1], FEATURES[2], FEATURES[0], FEATURES[3]])

    def test_near_ties_are_not_grouped(self):
        larger = math.nextafter(0.05, 1.0)
        self.assertEqual(c.rank_material_differences([0.05, -larger, 0.0, 0.0, 0.0]), [FEATURES[1], FEATURES[0]])

    def test_empty_ranking_keeps_exact_negligible_values(self):
        result = c.build_comparison(synthetic(), synthetic(others=(0.013, -0.011, 0.0, 0.0)))
        solved = result.features[0].attribution
        self.assertEqual(solved.delta, solved.modified - solved.original)
        self.assertAlmostEqual(solved.delta, -0.002, delta=1e-12)
        self.assertEqual(result.ranking, [])
        self.assertEqual([r.attribution.delta for r in result.features[1:3]], [0.013, -0.011])


# ----------------------------------------------------------------- consistency and failure
class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = real(SAMPLE), real(MIXED_B)

    def fails(self, check, modified, original=None):
        original = self.a if original is None else original
        for pair in ((original, modified), (modified, original)):
            outcome = None
            with self.assertRaises(c.ComparisonConsistencyError) as caught:
                outcome = c.build_comparison(*pair)
            self.assertIsNone(outcome, "no partial result")
            self.assertEqual(caught.exception.check, check)
            self.assertEqual(str(caught.exception), "comparison consistency check failed: " + check)
            self.assertIsNone(caught.exception.__cause__)

    def test_model_identity_artifacts_and_evidence_metadata(self):
        hashes = dict(self.b.model.artifact_sha256)
        hashes[next(iter(hashes))] = "0" * 64
        for path, value in ((("model", "artifact_sha256"), hashes), (("model", "handoff_version"), "9.9.9"),
                            (("model", "evidence_scope"), "other"), (("model", "model_notice"), "other"),
                            (("model", "inherited_v1_preprocessing_limitation"), "other")):
            with self.subTest(path=path):
                self.fails("shared_model", mutated(self.b, path, value))

    def test_approach_device_and_threshold(self):
        body = self.b.model_dump(mode="python")
        body["model"]["selected_approach"] = body["prediction"]["selected_approach"] = "other"
        self.fails("shared_model", schemas.ExplanationResponse.model_validate(body))
        self.fails("selected_approach", mutated(self.b, ("prediction", "selected_approach"), "other"))
        self.fails("shared_device", mutated(self.b, ("prediction", "device"), "cuda"))
        body = self.b.model_dump(mode="python")
        p = body["prediction"]
        p["threshold"] = 0.6
        p["decision"] = int(p["probability"] >= 0.6)
        p["decision_margin"] = p["logit"] - math.log(0.6 / 0.4)
        self.fails("shared_threshold", schemas.ExplanationResponse.model_validate(body))

    def test_integrity_development_only_and_notices_fail_structural_validation(self):
        for attribute, name, value in (("model", "integrity_status", "failed"), ("model", "development_only", False),
                                       ("method", "n_steps", 50), ("method", "integration", "riemann_left"),
                                       ("method", "name", "saliency"), ("method", "target", "probability")):
            broken = self.b.model_copy(deep=True)
            setattr(getattr(broken, attribute), name, value)
            with self.subTest(name=name):
                self.fails("invalid_explanation", broken)
        broken = self.b.model_copy(deep=True)
        broken.notices[0] = "changed"
        self.fails("invalid_explanation", broken)

    def test_method_version_mismatch(self):
        self.fails("shared_method", mutated(self.b, ("method", "library_version"), "0.9.1"))

    def test_baseline_mismatches_use_exact_equality(self):
        raw = dict(self.b.baseline.raw_values)
        raw["weekly_km_to_date"] = math.nextafter(raw["weekly_km_to_date"], 100.0)
        for path, value in ((("baseline", "raw_values"), raw),
                            (("baseline", "probability"), math.nextafter(self.b.baseline.probability, 1.0)),
                            (("baseline", "prior_injury_note"), "fractional, other"),
                            (("baseline", "logit"), math.nextafter(self.b.baseline.logit, 9.0))):
            with self.subTest(path=path):
                self.fails("shared_baseline", mutated(self.b, path, value))
        zeros = dict(self.b.baseline.standardized_values, prior_injury=0.5)
        self.fails("baseline_not_zero", mutated(self.b, ("baseline", "standardized_values"), zeros))

    def test_units_eligibility_and_tolerance(self):
        self.fails("units", mutated(self.b, ("units",), "percent"))
        ranges = dict(self.b.eligibility.ranges, weekly_km_to_date=(0.0, 200.0))
        self.fails("eligibility_ranges", mutated(self.b, ("eligibility", "ranges"), ranges))
        self.fails("completeness_tolerance", mutated(self.b, ("completeness", "tolerance", "absolute"), 0.02))
        outside = copy.deepcopy([a.model_dump() for a in self.b.attributions])
        outside[0]["value"] = 46.0
        self.fails("input_outside_eligibility", mutated(self.b, ("attributions",), outside))

    def test_same_input_with_different_standardized_value(self):
        items = [a.model_dump() for a in self.a.attributions]
        items[1]["standardized_value"] += 0.5
        with self.assertRaises(c.ComparisonConsistencyError) as caught:
            c.build_comparison(self.a, mutated(self.a, ("attributions",), items))
        self.assertEqual(caught.exception.check, "standardized_value")

    def test_prediction_inconsistencies(self):
        self.fails("decision_rule", mutated(self.b, ("prediction", "decision"), 1 - self.b.prediction.decision))
        self.fails("probability_logit", mutated(self.b, ("prediction", "logit"), self.b.prediction.logit + 0.01))
        self.fails("decision_margin", mutated(self.b, ("prediction", "decision_margin"), 9.0))

    def test_corrupted_completeness_records(self):
        k = self.b.completeness
        for name, value, check in (
            ("attribution_sum", k.attribution_sum + 1e-6, "attribution_sum"),
            ("target_difference", k.target_difference + 1e-6, "target_difference"),
            ("residual", -k.residual if k.residual else 1e-6, "completeness_residual"),
            ("allowed_residual", k.allowed_residual * 2, "allowed_residual"),
        ):
            with self.subTest(name=name):
                self.fails(check, mutated(self.b, ("completeness", name), value))
        items = [a.model_dump() for a in self.b.attributions]
        items[2]["attribution"] += 0.001
        self.fails("attribution_sum", mutated(self.b, ("attributions",), items))
        self.fails("completeness_gate", synthetic(residual=0.5))

    def test_inverted_residual_convention_is_rejected(self):
        response = synthetic(residual=0.001)
        self.fails("completeness_residual",
                   mutated(response, ("completeness", "residual"), -response.completeness.residual))

    def test_difference_relationship_and_combined_bound_helper(self):
        # Unreachable through the builder once both sides pass their own gates, so tested directly.
        helper = c._check_completeness_of_difference
        deltas = [0.1, 0.0, 0.0, 0.0, 0.0]
        self.assertEqual(helper(deltas, 0.1 - 0.001, 0.0, 0.001, 0.003, 0.003)[2], 0.006)
        for arguments, check in (
            ((deltas, 0.1 - 0.001, 0.001, 0.0, 0.003, 0.003), "difference_residual_relationship"),
            ((deltas, 0.1 - 0.001, 0.0, 0.0, 0.003, 0.003), "difference_residual_relationship"),
            ((deltas, 0.1 - 0.01, 0.0, 0.01, 0.003, 0.003), "difference_residual_bound"),
            ((deltas, 0.1 + 0.01, 0.01, 0.0, 0.003, 0.003), "difference_residual_bound"),
        ):
            with self.subTest(check=check), self.assertRaises(c.ComparisonConsistencyError) as caught:
                helper(*arguments)
            self.assertEqual(caught.exception.check, check)

    def test_order_missing_duplicate_and_non_finite_explanations(self):
        def broken(change):
            response = self.b.model_copy(deep=True)
            change(response)
            return response

        changes = {
            "reordered": lambda r: r.attributions.reverse(),
            "missing": lambda r: r.attributions.pop(),
            "duplicated": lambda r: r.attributions.__setitem__(4, r.attributions[0]),
            "nan_attribution": lambda r: setattr(r.attributions[0], "attribution", math.nan),
            "inf_logit": lambda r: setattr(r.prediction, "logit", math.inf),
            "nan_residual": lambda r: setattr(r.completeness, "residual", math.nan),
            "nan_baseline": lambda r: setattr(r.baseline, "logit", math.nan),
            "string_number": lambda r: setattr(r.prediction, "probability", "0.5"),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                self.fails("invalid_explanation", broken(change))

    def test_only_explanation_responses_are_accepted(self):
        for bad in (None, {}, self.b.model_dump()):
            with self.subTest(kind=type(bad).__name__):
                self.fails("not_an_explanation_response", bad)

    def test_final_validation_failure_yields_no_result(self):
        with mock.patch.object(c, "RANKING_RULE", "other"):
            with self.assertRaises(c.ComparisonConsistencyError) as caught:
                c.build_comparison(self.a, self.b)
        self.assertEqual(caught.exception.check, "response_validation")

    def test_structural_validity_is_not_provenance(self):
        # Documented limit: a self-consistent hand-built response is accepted. Trust comes from the
        # caller using the committed build_response path.
        c.build_comparison(synthetic(logit=-0.4), synthetic(logit=0.4))


# -------------------------------------------------------------------- independent oracle
class IndependentOracleTests(unittest.TestCase):
    def test_frozen_matrix_against_the_exact_piecewise_linear_oracle(self):
        runtime = L7.runtime
        grid = oracle.build_grids(runtime, SAMPLE)["straddle_threshold"]
        committed = json.loads(et.EVIDENCE_PATH.read_text())["grid_construction"]["sha256_float64"]
        self.assertEqual(oracle.grid_digest(grid), committed["straddle_threshold"])
        named = {"S": SAMPLE, **{f"T{i}": row(*grid[i]) for i in range(4)}}
        resolve = lambda x: named[x] if isinstance(x, str) else x
        pairs = [(n, resolve(a), resolve(b)) for n, a, b in IDENTICAL_PAIRS + DISTINCT_PAIRS]
        pairs += [("reversed_" + n, resolve(b), resolve(a)) for n, a, b in DISTINCT_PAIRS]
        self.assertEqual(len(pairs), 26)
        p64, b0 = oracle.params64(runtime.model), np.zeros(5)
        worst = dict.fromkeys(("feature_error", "relative_l1", "parity", "residual_over_allowed",
                               "pair_feature_error", "pair_residual_over_bound", "relationship_error"), 0.0)
        failed, transitions, exact = [], {}, {}

        def individual(response):
            z = np.array([a.standardized_value for a in response.attributions])
            truth, _ = oracle.oracle_ig(p64, z, b0)
            got = np.array([a.attribution for a in response.attributions])
            error = np.abs(got - truth)
            l1 = error.sum() / max(np.abs(truth).sum(), et.L1_FLOOR)
            parity = abs(schemas.stable_sigmoid(response.prediction.logit) - response.prediction.probability)
            ratio = abs(response.completeness.residual) / response.completeness.allowed_residual
            material = np.abs(truth) > et.SIGN_MIN
            sign_ok = bool(np.all(np.sign(got[material]) == np.sign(truth[material])))
            top_ok = bool(np.abs(truth)[np.abs(got).argmax()] >= np.abs(truth).max() - et.TIE_TOL)
            for key, value in (("feature_error", error.max()), ("relative_l1", l1), ("parity", parity),
                               ("residual_over_allowed", ratio)):
                worst[key] = max(worst[key], float(value))
            ok = (error.max() <= et.FEATURE_ERR_MAX and l1 <= et.L1_REL_MAX and parity <= et.PARITY
                  and ratio <= 1.0 and sign_ok and top_ok)
            return truth, got, ok

        for name, first, second in pairs:
            try:
                a, b = real(first), real(second)
                truth_a, got_a, ok_a = individual(a)
                truth_b, got_b, ok_b = individual(b)
                result = c.build_comparison(a, b)
                deltas = np.array([r.attribution.delta for r in result.features])
                pair_error = float(np.abs(deltas - (truth_b - truth_a)).max())
                done = result.completeness_of_difference
                relationship = abs(done.residual - (b.completeness.residual - a.completeness.residual))
                ratio = abs(done.residual) / done.residual_bound
                worst["pair_feature_error"] = max(worst["pair_feature_error"], pair_error)
                worst["relationship_error"] = max(worst["relationship_error"], relationship)
                worst["pair_residual_over_bound"] = max(worst["pair_residual_over_bound"], ratio)
                transitions[result.differences.decision.transition] = \
                    transitions.get(result.differences.decision.transition, 0) + 1
                if not (ok_a and ok_b and pair_error <= PAIR_BOUND and relationship <= 1e-9 and ratio <= 1.0
                        and result.shared.baseline == a.baseline == b.baseline
                        and result.shared.model == a.model == b.model):
                    failed.append(name)
                if first == second:
                    exact[name] = (not result.ranking and not result.differences.changed_features
                                   and result.differences.logit == 0.0 and result.differences.probability == 0.0
                                   and not deltas.any())
            except Exception as exc:
                failed.append(f"{name}:{type(exc).__name__}:{getattr(exc, 'check', '')}")
        print("L8_COMPARE_ORACLE " + json.dumps({"pairs": len(pairs), "pair_bound": PAIR_BOUND, "worst": worst,
                                                 "transitions": transitions, "failed": failed}, sort_keys=True))
        self.assertEqual(failed, [])
        self.assertEqual(exact, {"identical_sample": True, "identical_threshold": True})


if __name__ == "__main__":
    unittest.main()
