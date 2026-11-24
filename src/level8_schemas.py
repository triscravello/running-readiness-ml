"""Level 8 explanation API contracts (``explanation.v1``) and the pure response builder.

The success model is deliberately strict: unknown fields, non-finite numbers, wrong feature
order, a wrong integration setting or an incomplete payload all fail validation, and the
service then fails closed with a sanitized "unavailable" error (never a partial explanation).
Attribution values are copied from the engine's result unchanged: nothing is recomputed or
rounded. Worker-internal details (IPC identifiers, generations, numerical diagnostic
subtypes, process information) have no place in any model here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, field_validator, model_validator

FEATURES = (
    "weekly_km_to_date",
    "sleep_hours_last_night",
    "resting_hr_morning",
    "soreness_score_evening",
    "prior_injury",
)
SCHEMA_VERSION = "explanation.v1"
N_STEPS = 800

# Fixed interpretation notices (no free-form interpretation: that belongs to L8-3).
INTERPRETATION_NOTICES = (
    "The positive class represents needs_recovery_next_day, the dataset's target; it is not a general "
    "readiness score.",
    "Attributions describe how this model's output behaves relative to the baseline; they are not causal "
    "effects and do not show what would happen if a person changed a behaviour.",
    "The reference is a development-mean baseline (every feature at its training mean). It is not a "
    "healthy-runner reference.",
    "The baseline contains a fractional prior_injury value (about 0.167); it is not an actual runner "
    "observation, and that value is not a valid request input.",
    "The model was developed on synthetic data and is a development-only artifact.",
    "The inherited preprocessing limitation remains: the historical v1 process used development-wide mean "
    "imputation before splitting, and the full inherited history is not claimed to be leakage-free.",
    "Explanation eligibility is a check against development ranges. It does not establish clinical "
    "validity or that an input is statistically in-distribution.",
    "Numerical completeness (the attributions sum to the logit difference within tolerance) does not "
    "establish real-world predictive validity.",
    "This output is not medical advice.",
)

Finite = FiniteFloat

try:
    from src.level8_explain import ATTRIBUTION_UNITS, EXPLANATION_RANGES, RESIDUAL_ABS, RESIDUAL_REL
except ModuleNotFoundError as _exc:  # flat (image) layout
    if _exc.name != "src":
        raise
    from level8_explain import (  # type: ignore[no-redef]
        ATTRIBUTION_UNITS, EXPLANATION_RANGES, RESIDUAL_ABS, RESIDUAL_REL)

# Comparison tolerances for the semantic checks (the committed numerical gates are NOT relaxed):
#  * PROBABILITY_TOLERANCE: the approved probability/logit parity tolerance.
#  * ARITHMETIC_TOLERANCE: pure float64 re-additions/subtractions of numbers that crossed the pipe as
#    exactly round-tripping JSON doubles; it only guards summation order, far below any real error.
#  * BASELINE_LOGIT_TOLERANCE: the worker's float32 baseline forward pass (1 thread) against the parent's
#    (default threads); observed differences are ~1e-7, this allows 1e-5 and is 200x below the 2e-3 gate.
#  * STANDARDIZED_TOLERANCE: float32 standardization recomputed in float64 then cast, as the runtime does.
PROBABILITY_TOLERANCE = 2e-6
ARITHMETIC_TOLERANCE = 1e-9
BASELINE_LOGIT_TOLERANCE = 1e-5
STANDARDIZED_TOLERANCE = 1e-6
MEAN_TOLERANCE = 1e-12


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelInfo(_Strict):
    selected_approach: str
    handoff_version: str
    development_only: Literal[True]
    integrity_status: Literal["passed"]
    artifact_sha256: dict[str, str]
    evidence_scope: str
    model_notice: str
    inherited_v1_preprocessing_limitation: str

    @field_validator("artifact_sha256")
    @classmethod
    def _hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in value.values()):
            raise ValueError("artifact hashes must be sha256 hex digests")
        return value


class PredictionInfo(_Strict):
    probability: Annotated[Finite, Field(ge=0.0, le=1.0)]
    decision: Annotated[StrictInt, Field(ge=0, le=1)]
    threshold: Annotated[Finite, Field(ge=0.0, le=1.0)]
    logit: Finite
    decision_margin: Finite
    device: str
    selected_approach: str


class MethodInfo(_Strict):
    name: Literal["integrated_gradients"]
    library: Literal["captum"]
    library_version: str
    integration: Literal["riemann_middle"]
    n_steps: Literal[800]
    target: Literal["pre_sigmoid_logit"]
    one_input_per_call: Literal[True]


class BaselineInfo(_Strict):
    id: Literal["B0"]
    kind: Literal["standardized_zero_vector"]
    reference: Literal["development_mean"]
    raw_values: dict[str, Finite]
    standardized_values: dict[str, Finite]
    logit: Finite
    probability: Annotated[Finite, Field(ge=0.0, le=1.0)]
    prior_injury_note: str

    @field_validator("raw_values", "standardized_values")
    @classmethod
    def _canonical_keys(cls, value: dict[str, float]) -> dict[str, float]:
        if tuple(value) != FEATURES:
            raise ValueError("baseline values must be in canonical feature order")
        return value


class AttributionItem(_Strict):
    feature: Literal[*FEATURES]
    value: Finite
    standardized_value: Finite
    attribution: Finite


class Tolerance(_Strict):
    absolute: Finite
    relative: Finite


class Completeness(_Strict):
    target_difference: Finite
    attribution_sum: Finite
    residual: Finite
    allowed_residual: Finite
    tolerance: Tolerance


class Eligibility(_Strict):
    status: Literal["within_development_ranges"]
    ranges: dict[str, tuple[Finite, Finite]]

    @field_validator("ranges")
    @classmethod
    def _canonical_keys(cls, value):
        if tuple(value) != FEATURES:
            raise ValueError("ranges must be in canonical feature order")
        return value


class ExplanationResponse(_Strict):
    schema_version: Literal["explanation.v1"]
    model: ModelInfo
    prediction: PredictionInfo
    method: MethodInfo
    baseline: BaselineInfo
    units: str
    attributions: Annotated[list[AttributionItem], Field(min_length=5, max_length=5)]
    completeness: Completeness
    eligibility: Eligibility
    notices: list[str]

    @model_validator(mode="after")
    def _canonical_order_and_notices(self) -> "ExplanationResponse":
        if tuple(a.feature for a in self.attributions) != FEATURES:
            raise ValueError("attributions must be in canonical feature order")
        if tuple(self.notices) != INTERPRETATION_NOTICES:
            raise ValueError("notices must be the fixed interpretation notices")
        return self


# ------------------------------------------------------------------------------ errors
ERROR_CODES = (
    "outside_explanation_domain",
    "explanation_overloaded",
    "explanation_unavailable",
    "explanation_numerical_check_failed",
    "explanation_deadline_exceeded",
)
ERROR_MESSAGES = {
    "outside_explanation_domain": "The input is outside the explanation-eligibility ranges.",
    "explanation_overloaded": "The explanation service is at capacity. Retry shortly.",
    "explanation_unavailable": "The explanation service is currently unavailable.",
    "explanation_numerical_check_failed": "The explanation did not pass its numerical checks.",
    "explanation_deadline_exceeded": "The explanation was not completed within its deadline.",
}


class Violation(_Strict):
    feature: Literal[*FEATURES]
    min: Finite
    max: Finite


class ErrorBody(_Strict):
    code: Literal[*ERROR_CODES]
    message: str
    violations: list[Violation] | None = None


class ErrorResponse(_Strict):
    error: ErrorBody


# ------------------------------------------------------------------- trusted reference
def stable_sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Reference:
    """Authoritative parent-side facts every worker success payload is checked against."""

    artifacts: dict
    threshold: float
    selected_approach: str
    handoff_version: str
    development_only: bool
    device: str
    means: tuple
    scales: tuple
    baseline_logit: float
    baseline_probability: float


def make_reference(runtime: Any) -> Reference:
    """One-time (per application start) reference from the verified parent runtime. The baseline logit
    is a single forward pass of the zero vector (B0) through the parent's own model under
    ``torch.inference_mode``: it changes no Torch setting, no parameter and no module state, and is the
    same operation Level 7 performs on every /predict request. It is never done per request."""
    import torch

    if tuple(runtime.feature_names) != FEATURES:
        raise ValueError("unexpected feature order")
    with torch.inference_mode():
        baseline_logit = float(runtime.model(torch.zeros(1, len(FEATURES), dtype=torch.float32)).item())
    if not math.isfinite(baseline_logit):
        raise ValueError("non-finite baseline logit")
    return Reference(
        artifacts=dict(runtime.manifest["artifacts"]),
        threshold=float(runtime.threshold),
        selected_approach=str(runtime.handoff["selected_approach"]),
        handoff_version=str(runtime.handoff["handoff_version"]),
        development_only=runtime.handoff["evidence_scope"] == "development_only",
        device=str(runtime.device),
        means=tuple(float(v) for v in runtime.preprocessor["scaler_mean"]),
        scales=tuple(float(v) for v in runtime.preprocessor["scaler_scale"]),
        baseline_logit=baseline_logit,
        baseline_probability=stable_sigmoid(baseline_logit),
    )


def _require(condition: bool, what: str) -> None:
    if not condition:
        raise ValueError(what)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol + tol * max(abs(a), abs(b))


def verify_consistency(explanation: Mapping[str, Any], metadata: Mapping[str, Any], reference: Reference,
                       request: Mapping[str, Any]) -> None:
    """Semantic validation of a structurally valid worker payload. Raises ValueError on any
    inconsistency. Nothing is repaired or recomputed into the response: values are only compared."""
    model, method, baseline = metadata["model"], metadata["method"], metadata["baseline"]
    pred = explanation["prediction"]
    # --- model identity against the authoritative parent runtime
    _require(dict(model["artifact_sha256"]) == reference.artifacts, "artifact identity")
    _require(model["selected_approach"] == reference.selected_approach, "model approach")
    _require(model["handoff_version"] == reference.handoff_version, "handoff version")
    _require(model["development_only"] is True and reference.development_only, "development-only status")
    _require(model["integrity_status"] == "passed", "integrity status")
    _require(model.get("threshold", reference.threshold) == reference.threshold, "metadata threshold")
    # --- prediction
    threshold = reference.threshold
    _require(0.0 < threshold < 1.0, "threshold range")
    _require(pred["threshold"] == threshold, "threshold")
    probability, logit = float(pred["probability"]), float(explanation["logit"])
    _require(0.0 <= probability <= 1.0, "probability range")
    _require(pred["decision"] == int(probability >= threshold), "decision rule")
    _require(abs(probability - stable_sigmoid(logit)) <= PROBABILITY_TOLERANCE, "probability/logit")
    _require(pred["device"] == reference.device and pred["selected_approach"] == reference.selected_approach,
             "prediction identity")
    # --- attributions echo the request and its standardization
    items = explanation["attributions"]
    for index, (name, item) in enumerate(zip(FEATURES, items)):
        _require(item["feature"] == name, "attribution order")
        _require(item["value"] == float(request[name]), "attribution value echo")
        standardized = (float(request[name]) - reference.means[index]) / reference.scales[index]
        _require(abs(item["standardized_value"] - float(standardized)) <= STANDARDIZED_TOLERANCE
                 * max(1.0, abs(standardized)), "standardized value")
    # --- completeness (committed definition: residual = sum - (logit - baseline_logit))
    total = math.fsum(float(i["attribution"]) for i in items)
    difference = float(explanation["target_difference"])
    residual = float(explanation["completeness_residual"])
    allowed = float(explanation["allowed_residual"])
    _require(_close(float(explanation["attribution_sum"]), total, ARITHMETIC_TOLERANCE), "attribution sum")
    _require(_close(difference, logit - float(baseline["logit"]), ARITHMETIC_TOLERANCE), "target difference")
    _require(_close(residual, total - difference, ARITHMETIC_TOLERANCE), "residual")
    _require(_close(allowed, RESIDUAL_ABS + RESIDUAL_REL * abs(difference), ARITHMETIC_TOLERANCE), "allowed tolerance")
    _require(abs(residual) <= allowed, "completeness gate")
    tolerance = metadata["completeness_tolerance"]
    _require(tolerance["absolute"] == RESIDUAL_ABS and tolerance["relative"] == RESIDUAL_REL, "tolerance")
    # --- baseline B0 against the trusted reference
    _require(baseline["id"] == "B0" and baseline["kind"] == "standardized_zero_vector", "baseline identity")
    _require(tuple(baseline["raw_values"]) == FEATURES and tuple(baseline["standardized_values"]) == FEATURES,
             "baseline order")
    for index, name in enumerate(FEATURES):
        _require(abs(baseline["raw_values"][name] - reference.means[index]) <= MEAN_TOLERANCE
                 * max(1.0, abs(reference.means[index])), "baseline raw value")
        _require(baseline["standardized_values"][name] == 0.0, "baseline standardized value")
    _require(0.0 < baseline["raw_values"]["prior_injury"] < 1.0 and "fractional" in baseline["prior_injury_note"],
             "fractional prior injury")
    _require(abs(baseline["logit"] - reference.baseline_logit) <= BASELINE_LOGIT_TOLERANCE, "baseline logit")
    _require(abs(baseline["probability"] - stable_sigmoid(baseline["logit"])) <= PROBABILITY_TOLERANCE,
             "baseline probability")
    # sigmoid' <= 1/4, so a logit difference of up to BASELINE_LOGIT_TOLERANCE moves the probability by <= tol/4
    _require(abs(baseline["probability"] - reference.baseline_probability)
             <= PROBABILITY_TOLERANCE + BASELINE_LOGIT_TOLERANCE / 4, "baseline probability reference")
    # --- method, units, eligibility
    _require(method["name"] == "integrated_gradients" and method["library"] == "captum"
             and method["library_version"] == "0.9.0" and method["integration"] == "riemann_middle"
             and method["n_steps"] == N_STEPS and method["target"] == "pre_sigmoid_logit"
             and method["one_input_per_call"] is True, "method")
    _require(metadata["units"] == ATTRIBUTION_UNITS, "units")
    ranges = metadata["explanation_ranges"]
    _require(tuple(ranges) == FEATURES and all(tuple(ranges[n]) == EXPLANATION_RANGES[n] for n in FEATURES),
             "eligibility ranges")


# --------------------------------------------------------------------------- builder
def build_response(explanation: Mapping[str, Any], metadata: Mapping[str, Any], parent: Mapping[str, Any],
                   reference: Reference, request: Mapping[str, Any]) -> ExplanationResponse:
    """Assemble the public response from the worker's explanation dict, the worker's one-time engine
    metadata and the parent runtime's manifest fields, after semantic validation against ``reference``.
    Raises on any inconsistency; engine-provided values are copied unchanged."""
    verify_consistency(explanation, metadata, reference, request)
    model = metadata["model"]
    prediction = explanation["prediction"]
    logit = float(explanation["logit"])
    margin = logit - math.log(reference.threshold / (1.0 - reference.threshold))  # authoritative threshold
    method = metadata["method"]
    baseline = metadata["baseline"]
    tolerance = metadata["completeness_tolerance"]
    return ExplanationResponse.model_validate({
        "schema_version": SCHEMA_VERSION,
        "model": {
            "selected_approach": model["selected_approach"],
            "handoff_version": model["handoff_version"],
            "development_only": model["development_only"],
            "integrity_status": model["integrity_status"],
            "artifact_sha256": dict(model["artifact_sha256"]),
            "evidence_scope": parent["evidence_scope"],
            "model_notice": parent["model_notice"],
            "inherited_v1_preprocessing_limitation": parent["inherited_v1_preprocessing_limitation"],
        },
        "prediction": {
            "probability": prediction["probability"], "decision": prediction["decision"],
            "threshold": prediction["threshold"], "logit": logit, "decision_margin": margin,
            "device": prediction["device"], "selected_approach": prediction["selected_approach"],
        },
        "method": {
            "name": method["name"], "library": method["library"], "library_version": method["library_version"],
            "integration": method["integration"], "n_steps": method["n_steps"], "target": method["target"],
            "one_input_per_call": method["one_input_per_call"],
        },
        "baseline": {
            "id": baseline["id"], "kind": baseline["kind"], "reference": "development_mean",
            "raw_values": baseline["raw_values"], "standardized_values": baseline["standardized_values"],
            "logit": baseline["logit"], "probability": baseline["probability"],
            "prior_injury_note": baseline["prior_injury_note"],
        },
        "units": metadata["units"],
        "attributions": [dict(item) for item in explanation["attributions"]],
        "completeness": {
            "target_difference": explanation["target_difference"], "attribution_sum": explanation["attribution_sum"],
            "residual": explanation["completeness_residual"], "allowed_residual": explanation["allowed_residual"],
            "tolerance": {"absolute": tolerance["absolute"], "relative": tolerance["relative"]},
        },
        "eligibility": {"status": "within_development_ranges", "ranges": metadata["explanation_ranges"]},
        "notices": list(INTERPRETATION_NOTICES),
    })
