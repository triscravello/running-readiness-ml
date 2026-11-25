"""Level 8 deterministic scenario comparison and interpretation (``comparison.v1``).

A pure layer over two ``explanation.v1`` responses. It performs no inference, imports no Captum,
starts no process or thread, submits no job, serves no HTTP and stores nothing. Importing it loads
Torch only indirectly (through the committed explanation modules) and constructs no model.

Trust boundary
--------------
``build_comparison`` is meant to receive two responses produced by the committed trusted path
(``level8_schemas.build_response``, which checks a worker payload against the parent's verified
reference). This module re-checks structure, shared identity and the arithmetic that can be checked
from the responses themselves. It does NOT authenticate model execution: a self-consistent,
hand-built ``ExplanationResponse`` passes these checks. Neither matching artifact hashes nor
anything else here shows that two responses came from the same worker generation; that is an
execution requirement of the caller.

Every difference is ``modified - original``. All values are unrounded float64.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictBool, StrictInt, create_model, model_validator

try:
    from src import level8_schemas as schemas
    from src.level8_explain import (ATTRIBUTION_UNITS, EXPLANATION_RANGES, RESIDUAL_ABS, RESIDUAL_REL,
                                    ExplanationEngine, OutsideExplanationDomain)
except ModuleNotFoundError as _exc:  # flat (image) layout
    if _exc.name != "src":
        raise
    import level8_schemas as schemas  # type: ignore[no-redef]
    from level8_explain import (ATTRIBUTION_UNITS, EXPLANATION_RANGES, RESIDUAL_ABS,  # type: ignore[no-redef]
                                RESIDUAL_REL, ExplanationEngine, OutsideExplanationDomain)

__all__ = [
    "SCHEMA_VERSION", "POLICY_VERSION", "MATERIALITY_THRESHOLD_LOGIT", "DIRECTIONS", "TRANSITIONS",
    "COMPARISON_ONLY_NOTICES", "COMPARISON_NOTICES", "SCENARIOS",
    "ComparisonConsistencyError", "ComparisonResponse",
    "make_comparison_request_model", "eligibility_violations",
    "classify_contribution", "rank_material_differences", "build_comparison",
]

FEATURES = schemas.FEATURES
SCHEMA_VERSION = "comparison.v1"
POLICY_VERSION = "comparison-interpretation.v1"
# Display policy, not a significance claim: abs(value) <= threshold is "negligible".
MATERIALITY_THRESHOLD_LOGIT = 0.02
DIRECTIONS = ("toward_class_1", "toward_class_0", "negligible")
TRANSITIONS = {(0, 0): "unchanged_0", (1, 1): "unchanged_1", (0, 1): "0_to_1", (1, 0): "1_to_0"}
SCENARIOS = ("original", "modified")
APPLIES_TO = ("original_attribution", "modified_attribution", "attribution_difference")
RANKING_RULE = "material_attribution_differences_by_descending_magnitude_then_feature_order"

COMPARISON_ONLY_NOTICES = (
    "All comparison differences are the modified scenario minus the original scenario.",
    "Comparing hypothetical inputs describes this model only; it does not predict what would happen if "
    "a person changed them.",
    "Changing one input can change the attributions of other features, including features whose inputs "
    "did not change.",
    "The probability is not a validated measure of individual certainty, and a change in the model's "
    "decision has no demonstrated real-world meaning.",
)
COMPARISON_NOTICES = schemas.INTERPRETATION_NOTICES + COMPARISON_ONLY_NOTICES

_TOL = schemas.ARITHMETIC_TOLERANCE
_close = schemas._close  # the committed comparison rule, reused so that the two cannot diverge

Feature = Literal[*FEATURES]
Direction = Literal[*DIRECTIONS]
Probability = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
Decision = Annotated[StrictInt, Field(ge=0, le=1)]


class ComparisonConsistencyError(Exception):
    """The two explanations cannot be compared. ``check`` is a fixed identifier; the exception never
    carries submitted values, model outputs or underlying exception text."""

    def __init__(self, check: str):
        super().__init__("comparison consistency check failed: " + check)
        self.check = check


def _require(condition: bool, check: str) -> None:
    if not condition:
        raise ComparisonConsistencyError(check)


def _positive_zero(value: float) -> float:
    """Normalize a DERIVED negative zero. Values copied from an explanation are never passed here."""
    return 0.0 if value == 0.0 else value


def _difference(modified: float, original: float) -> float:
    return _positive_zero(modified - original)


def _total(values: Sequence[float]) -> float:
    return _positive_zero(math.fsum(values))


# ------------------------------------------------------------------------------ interpretation
def classify_contribution(value: float) -> str:
    """Direction label of one attribution or attribution difference, in logit units."""
    if not math.isfinite(value):
        raise ComparisonConsistencyError("non_finite_value")
    if value > MATERIALITY_THRESHOLD_LOGIT:
        return "toward_class_1"
    if value < -MATERIALITY_THRESHOLD_LOGIT:
        return "toward_class_0"
    return "negligible"


def rank_material_differences(deltas: Sequence[float]) -> list[str]:
    """Feature names of the non-negligible differences (given in canonical feature order), largest
    magnitude first; exact ties keep canonical order. Near-ties are not grouped."""
    if len(deltas) != len(FEATURES):
        raise ComparisonConsistencyError("feature_count")
    material = [i for i, v in enumerate(deltas) if classify_contribution(v) != "negligible"]
    return [FEATURES[i] for i in sorted(material, key=lambda i: (-abs(deltas[i]), i))]


# ------------------------------------------------------------------------------------- request
def make_comparison_request_model(prediction_request_model: type[BaseModel]) -> type[BaseModel]:
    """Build ``ComparisonRequest {original, modified}`` around the existing Level 7
    ``PredictionRequest``. The class is passed in because importing ``level7_service`` constructs a
    model runtime. Each side is validated by that class's own strict configuration. The wrapper is
    deliberately not strict itself; it only forbids unknown fields."""
    if not (isinstance(prediction_request_model, type) and issubclass(prediction_request_model, BaseModel)):
        raise TypeError("a Pydantic request model is required")
    config = prediction_request_model.model_config
    if (tuple(prediction_request_model.model_fields) != FEATURES or config.get("strict") is not True
            or config.get("extra") != "forbid" or config.get("allow_inf_nan") is not False
            or not all(f.is_required() for f in prediction_request_model.model_fields.values())):
        raise TypeError("the model is not the strict five-feature prediction request contract")
    return create_model("ComparisonRequest", __config__=ConfigDict(extra="forbid"),
                        original=(prediction_request_model, ...), modified=(prediction_request_model, ...))


def eligibility_violations(original: Mapping[str, Any], modified: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Explanation-domain violations of both scenarios (original first, canonical feature order),
    using the committed validator. Submitted values are not echoed. An input that breaks the
    prediction contract raises the validator's ``InvalidExplanationInput`` unchanged."""
    found: list[dict[str, Any]] = []
    for scenario, payload in zip(SCENARIOS, (original, modified)):
        try:
            ExplanationEngine.validate_payload(payload)
        except OutsideExplanationDomain as exc:
            found.extend({"scenario": scenario, **violation} for violation in exc.violations)
    return tuple(found)


# ------------------------------------------------------------------------------ response models
class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, protected_namespaces=())


class SharedInfo(_Response):
    """Metadata required to be identical in both explanations, stated once. The positive class is
    defined by the fixed notices (``explanation.v1`` has no separate field for it)."""

    model: schemas.ModelInfo
    method: schemas.MethodInfo
    baseline: schemas.BaselineInfo
    units: str
    threshold: Probability
    device: str
    selected_approach: str
    eligibility: schemas.Eligibility
    completeness_tolerance: schemas.Tolerance


class ScenarioPrediction(_Response):
    probability: Probability
    decision: Decision
    logit: FiniteFloat
    decision_margin: FiniteFloat


class ScenarioCompleteness(_Response):
    target_difference: FiniteFloat
    attribution_sum: FiniteFloat
    residual: FiniteFloat
    allowed_residual: FiniteFloat


class ScenarioResult(_Response):
    prediction: ScenarioPrediction
    completeness: ScenarioCompleteness


class DecisionChange(_Response):
    original: Decision
    modified: Decision
    changed: StrictBool
    transition: Literal["unchanged_0", "unchanged_1", "0_to_1", "1_to_0"]


class Differences(_Response):
    direction: Literal["modified_minus_original"]
    probability: FiniteFloat
    logit: FiniteFloat
    decision: DecisionChange
    changed_features: list[Feature]


class InputValues(_Response):
    original: FiniteFloat
    modified: FiniteFloat
    delta: FiniteFloat
    changed: StrictBool


class StandardizedValues(_Response):
    original: FiniteFloat
    modified: FiniteFloat


class AttributionValues(_Response):
    original: FiniteFloat
    modified: FiniteFloat
    delta: FiniteFloat


class FeatureRow(_Response):
    feature: Feature
    input: InputValues
    standardized: StandardizedValues
    attribution: AttributionValues
    original_direction: Direction
    modified_direction: Direction
    delta_direction: Direction


class CompletenessOfDifference(_Response):
    logit_difference: FiniteFloat
    attribution_delta_sum: FiniteFloat
    residual: FiniteFloat
    residual_bound: FiniteFloat


class InterpretationPolicy(_Response):
    version: Literal["comparison-interpretation.v1"]
    materiality_threshold_logit: FiniteFloat
    threshold_is_negligible: Literal[True]
    applies_to: list[str]
    ranking: Literal["material_attribution_differences_by_descending_magnitude_then_feature_order"]

    @model_validator(mode="after")
    def _fixed(self) -> "InterpretationPolicy":
        if self.materiality_threshold_logit != MATERIALITY_THRESHOLD_LOGIT or tuple(self.applies_to) != APPLIES_TO:
            raise ValueError("the interpretation policy is fixed")
        return self


def _check_completeness_of_difference(deltas: Sequence[float], logit_difference: float,
                                      original_residual: float, modified_residual: float,
                                      original_allowed: float, modified_allowed: float) -> tuple[float, float, float]:
    """With a shared baseline and ``residual = attribution_sum - target_difference``:
    sum(deltas) = logit_difference + modified_residual - original_residual.
    This is a consistency check; it says nothing about per-feature accuracy. Returns
    (delta_sum, residual, bound)."""
    delta_sum = _total(deltas)
    residual = _difference(delta_sum, logit_difference)
    bound = original_allowed + modified_allowed
    _require(_close(residual, modified_residual - original_residual, _TOL), "difference_residual_relationship")
    _require(abs(residual) <= bound + _TOL, "difference_residual_bound")
    return delta_sum, residual, bound


class ComparisonResponse(_Response):
    """All fields are required. Validation also rejects a payload that contradicts itself."""

    schema_version: Literal["comparison.v1"]
    shared: SharedInfo
    original: ScenarioResult
    modified: ScenarioResult
    differences: Differences
    features: Annotated[list[FeatureRow], Field(min_length=5, max_length=5)]
    ranking: list[Feature]
    completeness_of_difference: CompletenessOfDifference
    interpretation_policy: InterpretationPolicy
    notices: list[str]

    @model_validator(mode="after")
    def _internally_consistent(self) -> "ComparisonResponse":
        def check(condition: bool, what: str) -> None:
            if not condition:
                raise ValueError(what)

        check(tuple(row.feature for row in self.features) == FEATURES, "features must be in canonical order")
        for row in self.features:
            check(row.input.delta == _difference(row.input.modified, row.input.original), "input delta")
            check(row.input.changed == (row.input.modified != row.input.original), "input changed flag")
            check(row.attribution.delta == _difference(row.attribution.modified, row.attribution.original),
                  "attribution delta")
            check((row.original_direction, row.modified_direction, row.delta_direction) == tuple(
                classify_contribution(v) for v in (row.attribution.original, row.attribution.modified,
                                                   row.attribution.delta)), "direction labels")
        d, a, b = self.differences, self.original, self.modified
        check(d.changed_features == [r.feature for r in self.features if r.input.changed], "changed features")
        check(d.probability == _difference(b.prediction.probability, a.prediction.probability), "probability delta")
        check(d.logit == _difference(b.prediction.logit, a.prediction.logit), "logit delta")
        pair = (a.prediction.decision, b.prediction.decision)
        check((d.decision.original, d.decision.modified) == pair and d.decision.changed == (pair[0] != pair[1])
              and d.decision.transition == TRANSITIONS[pair], "decision change")
        deltas = [r.attribution.delta for r in self.features]
        check(self.ranking == rank_material_differences(deltas), "ranking")
        c = self.completeness_of_difference
        try:
            delta_sum, residual, bound = _check_completeness_of_difference(
                deltas, d.logit, a.completeness.residual, b.completeness.residual,
                a.completeness.allowed_residual, b.completeness.allowed_residual)
        except ComparisonConsistencyError as exc:
            raise ValueError(exc.check) from None
        check((c.logit_difference, c.attribution_delta_sum, c.residual, c.residual_bound)
              == (d.logit, delta_sum, residual, bound), "completeness of difference")
        check(tuple(self.notices) == COMPARISON_NOTICES, "notices must be the fixed comparison notices")
        return self


# ------------------------------------------------------------------------------------ builder
def _revalidated(explanation: Any) -> schemas.ExplanationResponse:
    """A fresh, strictly re-validated copy, so a mutated or unvalidated instance fails closed
    instead of being coerced. A non-finite number serializes as null and is rejected."""
    _require(type(explanation) is schemas.ExplanationResponse, "not_an_explanation_response")
    try:
        return schemas.ExplanationResponse.model_validate_json(
            explanation.model_dump_json(warnings="error"), strict=True)
    except Exception:
        raise ComparisonConsistencyError("invalid_explanation") from None


def _check_explanation(e: schemas.ExplanationResponse) -> None:
    """Re-check what one explanation can show about itself, with the committed definitions."""
    p, c = e.prediction, e.completeness
    _require(0.0 < p.threshold < 1.0, "threshold_range")
    # the authoritative decision rule is on the probability; the logit sign is never used for it
    _require(p.decision == int(p.probability >= p.threshold), "decision_rule")
    _require(abs(p.probability - schemas.stable_sigmoid(p.logit)) <= schemas.PROBABILITY_TOLERANCE,
             "probability_logit")
    _require(_close(p.decision_margin, p.logit - math.log(p.threshold / (1.0 - p.threshold)), _TOL),
             "decision_margin")
    _require(p.selected_approach == e.model.selected_approach, "selected_approach")
    _require(e.units == ATTRIBUTION_UNITS, "units")
    ranges = e.eligibility.ranges
    _require(all(tuple(ranges[n]) == EXPLANATION_RANGES[n] for n in FEATURES), "eligibility_ranges")
    for item in e.attributions:
        low, high = EXPLANATION_RANGES[item.feature]
        _require(low <= item.value <= high, "input_outside_eligibility")
        _require(item.feature != "prior_injury" or item.value in (0.0, 1.0), "prior_injury_value")
    _require(all(e.baseline.standardized_values[n] == 0.0 for n in FEATURES), "baseline_not_zero")
    _require((c.tolerance.absolute, c.tolerance.relative) == (RESIDUAL_ABS, RESIDUAL_REL), "completeness_tolerance")
    total = math.fsum(item.attribution for item in e.attributions)
    _require(_close(c.attribution_sum, total, _TOL), "attribution_sum")
    _require(_close(c.target_difference, p.logit - e.baseline.logit, _TOL), "target_difference")
    _require(_close(c.residual, total - c.target_difference, _TOL), "completeness_residual")
    _require(_close(c.allowed_residual, RESIDUAL_ABS + RESIDUAL_REL * abs(c.target_difference), _TOL),
             "allowed_residual")
    _require(abs(c.residual) <= c.allowed_residual, "completeness_gate")


def _shared_fields(e: schemas.ExplanationResponse) -> tuple[tuple[str, Any], ...]:
    """Contractually identical for one model and baseline: compared with exact equality."""
    return (
        ("model", e.model), ("method", e.method), ("baseline", e.baseline), ("units", e.units),
        ("eligibility", e.eligibility), ("completeness_tolerance", e.completeness.tolerance),
        ("notices", e.notices), ("threshold", e.prediction.threshold), ("device", e.prediction.device),
        ("prediction_selected_approach", e.prediction.selected_approach),
    )


def _scenario(e: schemas.ExplanationResponse) -> dict[str, Any]:
    p, c = e.prediction, e.completeness
    return {
        "prediction": ScenarioPrediction(probability=p.probability, decision=p.decision, logit=p.logit,
                                         decision_margin=p.decision_margin),
        "completeness": ScenarioCompleteness(target_difference=c.target_difference,
                                             attribution_sum=c.attribution_sum, residual=c.residual,
                                             allowed_residual=c.allowed_residual),
    }


def build_comparison(original: schemas.ExplanationResponse,
                     modified: schemas.ExplanationResponse) -> ComparisonResponse:
    """All-or-nothing: returns a complete, validated response or raises
    ``ComparisonConsistencyError``. Neither argument is modified. For identical scenarios the caller
    may pass the same object twice."""
    a, b = _revalidated(original), _revalidated(modified)
    _check_explanation(a)
    _check_explanation(b)
    for (name, left), (_, right) in zip(_shared_fields(a), _shared_fields(b)):
        _require(left == right, "shared_" + name)

    rows = []
    for x, y in zip(a.attributions, b.attributions):
        changed = x.value != y.value
        # same input and same preprocessor artifact must give the same standardized value
        _require(changed or x.standardized_value == y.standardized_value, "standardized_value")
        delta = _difference(y.attribution, x.attribution)
        rows.append(FeatureRow(
            feature=x.feature,
            input=InputValues(original=x.value, modified=y.value, delta=_difference(y.value, x.value),
                              changed=changed),
            standardized=StandardizedValues(original=x.standardized_value, modified=y.standardized_value),
            attribution=AttributionValues(original=x.attribution, modified=y.attribution, delta=delta),
            original_direction=classify_contribution(x.attribution),
            modified_direction=classify_contribution(y.attribution),
            delta_direction=classify_contribution(delta),
        ))
    deltas = [row.attribution.delta for row in rows]
    logit_difference = _difference(b.prediction.logit, a.prediction.logit)
    delta_sum, residual, bound = _check_completeness_of_difference(
        deltas, logit_difference, a.completeness.residual, b.completeness.residual,
        a.completeness.allowed_residual, b.completeness.allowed_residual)
    decisions = (a.prediction.decision, b.prediction.decision)
    try:
        return ComparisonResponse(
            schema_version=SCHEMA_VERSION,
            shared=SharedInfo(model=a.model, method=a.method, baseline=a.baseline, units=a.units,
                              threshold=a.prediction.threshold, device=a.prediction.device,
                              selected_approach=a.prediction.selected_approach, eligibility=a.eligibility,
                              completeness_tolerance=a.completeness.tolerance),
            original=ScenarioResult(**_scenario(a)),
            modified=ScenarioResult(**_scenario(b)),
            differences=Differences(
                direction="modified_minus_original",
                probability=_difference(b.prediction.probability, a.prediction.probability),
                logit=logit_difference,
                decision=DecisionChange(original=decisions[0], modified=decisions[1],
                                        changed=decisions[0] != decisions[1], transition=TRANSITIONS[decisions]),
                changed_features=[row.feature for row in rows if row.input.changed]),
            features=rows,
            ranking=rank_material_differences(deltas),
            completeness_of_difference=CompletenessOfDifference(
                logit_difference=logit_difference, attribution_delta_sum=delta_sum, residual=residual,
                residual_bound=bound),
            interpretation_policy=InterpretationPolicy(
                version=POLICY_VERSION, materiality_threshold_logit=MATERIALITY_THRESHOLD_LOGIT,
                threshold_is_negligible=True, applies_to=list(APPLIES_TO), ranking=RANKING_RULE),
            notices=list(COMPARISON_NOTICES),
        )
    except ComparisonConsistencyError:
        raise
    except Exception:
        raise ComparisonConsistencyError("response_validation") from None
