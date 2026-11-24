"""Level 8 explainability engine: Integrated Gradients over the verified Level 7 model.

Scope: a library only. No HTTP routes, no worker process, no scenario comparison.

What an explanation is: a decomposition of (logit(x) - logit(B0)) into five per-feature
terms for the frozen, development-only model. It is a statement about the model, not
about causes, and not about real-world validity. Explanation eligibility (the inclusive
ranges below) is a development-range check, not a statistical in-distribution test.

Design constraints honoured here:
* The engine explains a PRIVATE deep copy of the runtime model; the runtime's model,
  preprocessing, threshold and artifact verification are never altered.
* ModelRuntime stays the authoritative prediction source.
* Process-global Torch settings (thread counts, deterministic mode, default dtype) are
  never changed here; thread configuration belongs to the harness / future worker process.
* Nothing is loaded or written at import time (Captum is imported lazily on construction).
* The independent piecewise-linear oracle lives in tests only.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Mapping

import torch
from torch import nn

METHOD = "riemann_middle"
N_STEPS = 800
BASELINE_ID = "B0"
TARGET = "pre_sigmoid_logit"
ATTRIBUTION_UNITS = "logit units (log-odds contribution relative to baseline B0)"

# Runtime completeness gate: abs(residual) <= RESIDUAL_ABS + RESIDUAL_REL * abs(target_difference)
RESIDUAL_ABS = 0.002
RESIDUAL_REL = 0.005
# Prediction / logit parity between the private copy and ModelRuntime.predict
PARITY_TOLERANCE = 2e-6

# Inclusive explanation-eligibility ranges (development-data ranges).
EXPLANATION_RANGES: dict[str, tuple[float, float]] = {
    "weekly_km_to_date": (20.0, 45.0),
    "sleep_hours_last_night": (5.5, 8.0),
    "resting_hr_morning": (45.0, 57.0),
    "soreness_score_evening": (0.0, 10.0),
    "prior_injury": (0.0, 1.0),
}
# Level 7 /predict validation bounds, mirrored here (not imported: the service module
# builds a runtime at import time). Kept in sync by a test against the Level 7 source.
PREDICTION_RANGES: dict[str, tuple[float, float]] = {
    "weekly_km_to_date": (0.0, 200.0),
    "sleep_hours_last_night": (0.0, 14.0),
    "resting_hr_morning": (30.0, 220.0),
    "soreness_score_evening": (0.0, 10.0),
    "prior_injury": (0.0, 1.0),
}

NOTICES = (
    "Attributions describe how this frozen, development-only model's logit differs from "
    "its logit at baseline B0; they are not causal effects and do not show what would "
    "happen if a person changed a behaviour.",
    "Baseline B0 is the training mean of every feature, including a fractional "
    "prior_injury of about 0.167. No real person has that value; it is a reference "
    "point, not a typical or healthy athlete.",
    "Explanation eligibility is a development-range check. It does not establish "
    "statistical in-distribution status or real-world predictive validity.",
    "Verification of the numbers (agreement with an independent oracle) is separate "
    "from model validity and from production performance.",
)


class ExplanationError(Exception):
    """Base class for engine failures."""

    code = "explanation_error"


class InvalidExplanationInput(ExplanationError):
    """Payload is malformed or violates the Level 7 prediction contract."""

    code = "invalid_explanation_input"


class OutsideExplanationDomain(ExplanationError):
    """Valid prediction input outside the explanation-eligibility ranges."""

    code = "outside_explanation_domain"

    def __init__(self, violations: list[dict[str, Any]]):
        self.violations = violations
        super().__init__(
            "Input outside the explanation domain for: "
            + ", ".join(v["feature"] for v in violations)
        )


class ExplanationNumericalError(ExplanationError):
    """Non-finite output or a failed numerical gate. No partial attributions are returned."""

    code = "explanation_numerical_check_failed"


@dataclass(frozen=True)
class FeatureAttribution:
    feature: str
    value: float
    standardized_value: float
    attribution: float


@dataclass(frozen=True)
class Explanation:
    prediction: dict[str, Any]
    logit: float
    baseline_logit: float
    target_difference: float
    attribution_sum: float
    completeness_residual: float
    allowed_residual: float
    attributions: tuple[FeatureAttribution, ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form; every number is finite, so json.dumps(allow_nan=False) works."""
        return asdict(self)


class _LogitModel(nn.Module):
    """Private copy of the runtime model; forward returns pre-sigmoid logits, shape (N,)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def _sigmoid(logit: float) -> float:
    """Numerically stable logistic function. Non-finite input fails closed."""
    if not math.isfinite(logit):
        raise ExplanationNumericalError("Non-finite logit")
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    e = math.exp(logit)
    return e / (1.0 + e)


@dataclass(frozen=True)
class _Numerics:
    """Internal numerical result: no raw values, no prediction, finite floats only."""

    logit: float
    baseline_logit: float
    target_difference: float
    attribution_sum: float
    completeness_residual: float
    allowed_residual: float
    attributions: tuple[float, ...]


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class ExplanationEngine:
    """Serial Integrated Gradients explainer. One input per call; not thread-safe by design
    (callers must serialize, see the Level 8 worker plan)."""

    def __init__(self, runtime: Any):
        # Captum (and matplotlib behind it) is imported here, never at module import.
        from captum.attr import IntegratedGradients

        self._runtime = runtime
        self._features = list(runtime.feature_names)
        if self._features != list(EXPLANATION_RANGES):
            raise ExplanationNumericalError("Runtime feature order differs from the canonical order")
        private = copy.deepcopy(runtime.model).eval()
        for parameter in private.parameters():
            parameter.requires_grad_(False)
        self._model = _LogitModel(private).eval()
        self._ig = IntegratedGradients(self._model)
        self._baseline = torch.zeros(1, len(self._features), dtype=torch.float32)
        with torch.no_grad():
            self._baseline_logit = float(self._model(self._baseline).item())
        if not math.isfinite(self._baseline_logit):
            raise ExplanationNumericalError("Baseline logit is not finite")
        self._captum_version = metadata.version("captum")

    # ------------------------------------------------------------------ metadata
    def metadata(self) -> dict[str, Any]:
        rt = self._runtime
        means = [float(v) for v in rt.preprocessor["scaler_mean"]]
        baseline_probability = _sigmoid(self._baseline_logit)
        return {
            "model": {
                "selected_approach": rt.handoff["selected_approach"],
                "handoff_version": rt.handoff["handoff_version"],
                "threshold": rt.threshold,
                "evidence_scope": rt.handoff.get("evidence_scope"),
                "development_only": True,
                "integrity_status": rt.integrity_status,
                "artifact_sha256": dict(rt.manifest["artifacts"]),
            },
            "method": {
                "name": "integrated_gradients",
                "library": "captum",
                "library_version": self._captum_version,
                "integration": METHOD,
                "n_steps": N_STEPS,
                "target": TARGET,
                "one_input_per_call": True,
            },
            "baseline": {
                "id": BASELINE_ID,
                "kind": "standardized_zero_vector",
                "raw_values": dict(zip(self._features, means)),
                "standardized_values": dict(zip(self._features, [0.0] * len(self._features))),
                "logit": self._baseline_logit,
                "probability": baseline_probability,
                "prior_injury_note": (
                    "prior_injury is fractional in B0 (training mean); it is not a valid "
                    "request value."
                ),
            },
            "units": ATTRIBUTION_UNITS,
            "completeness_tolerance": {"absolute": RESIDUAL_ABS, "relative": RESIDUAL_REL},
            "explanation_ranges": {k: list(v) for k, v in EXPLANATION_RANGES.items()},
            "notices": list(NOTICES),
        }

    # --------------------------------------------------------------- validation
    @staticmethod
    def validate_payload(payload: Mapping[str, Any]) -> dict[str, float | int]:
        """Check the Level 7 prediction contract, then the explanation-eligibility ranges."""
        if not isinstance(payload, Mapping):
            raise InvalidExplanationInput("Payload must be a mapping")
        names = list(EXPLANATION_RANGES)
        missing = [n for n in names if n not in payload]
        extra = [n for n in payload if n not in EXPLANATION_RANGES]
        if missing or extra:
            raise InvalidExplanationInput(f"Payload fields must be exactly {names}")
        clean: dict[str, float | int] = {}
        for name in names:
            value = payload[name]
            lo, hi = PREDICTION_RANGES[name]
            if name == "prior_injury":
                if type(value) is not int:  # strict int: rejects bool, float, str
                    raise InvalidExplanationInput("prior_injury must be an integer 0 or 1")
                if value not in (0, 1):
                    raise InvalidExplanationInput("prior_injury must be 0 or 1")
                clean[name] = value
                continue
            if not _is_real_number(value):
                raise InvalidExplanationInput(f"{name} must be a number")
            try:
                number = float(value)
            except (OverflowError, ValueError, TypeError):  # e.g. integers beyond float range
                raise InvalidExplanationInput(f"{name} must be a finite number") from None
            if not math.isfinite(number):
                raise InvalidExplanationInput(f"{name} must be finite")
            if not lo <= number <= hi:
                raise InvalidExplanationInput(f"{name} is outside the prediction range")
            clean[name] = number
        violations = [
            {"feature": n, "min": EXPLANATION_RANGES[n][0], "max": EXPLANATION_RANGES[n][1]}
            for n in names
            if not EXPLANATION_RANGES[n][0] <= float(clean[n]) <= EXPLANATION_RANGES[n][1]
        ]
        if violations:  # bounds only; the submitted value is never echoed
            raise OutsideExplanationDomain(violations)
        return clean

    # --------------------------------------------------------------- explanation
    def explain(self, payload: Mapping[str, Any]) -> Explanation:
        clean = self.validate_payload(payload)
        standardized = torch.from_numpy(self._runtime.preprocess(clean))
        prediction = self._runtime.predict(clean)
        numerics = self._explain_standardized(standardized)
        # ModelRuntime stays authoritative: the private copy must reproduce its probability.
        if abs(_sigmoid(numerics.logit) - prediction["probability"]) > PARITY_TOLERANCE:
            raise ExplanationNumericalError("Private-model probability differs from ModelRuntime")
        z = [float(v) for v in standardized[0].tolist()]
        items = tuple(
            FeatureAttribution(name, float(clean[name]), zv, av)
            for name, zv, av in zip(self._features, z, numerics.attributions)
        )
        return Explanation(
            prediction=prediction,
            logit=numerics.logit,
            baseline_logit=numerics.baseline_logit,
            target_difference=numerics.target_difference,
            attribution_sum=numerics.attribution_sum,
            completeness_residual=numerics.completeness_residual,
            allowed_residual=numerics.allowed_residual,
            attributions=items,
        )

    def _explain_standardized(self, standardized: torch.Tensor) -> _Numerics:
        """Internal numerical core (not an application interface; use explain()). Attributes
        an already-standardized (1, 5) float32 tensor with no eligibility check, but still
        validates shape, dtype and finiteness. Returns finite floats only."""
        if not isinstance(standardized, torch.Tensor) or tuple(standardized.shape) != (1, 5):
            raise InvalidExplanationInput("Standardized input must be a tensor of shape (1, 5)")
        if standardized.dtype != torch.float32:
            raise InvalidExplanationInput("Standardized input must be float32")
        if not bool(torch.isfinite(standardized).all()):
            raise InvalidExplanationInput("Standardized input must be finite")
        x = standardized.detach().clone()
        try:
            with torch.inference_mode(False), torch.enable_grad():
                attribution = self._ig.attribute(
                    x,
                    baselines=self._baseline,
                    n_steps=N_STEPS,
                    method=METHOD,
                )
            with torch.no_grad():
                logit = float(self._model(x).item())
        except Exception as exc:  # fail closed on any engine fault
            raise ExplanationNumericalError(f"Integrated Gradients failed: {type(exc).__name__}") from exc
        if tuple(attribution.shape) != (1, 5):
            raise ExplanationNumericalError("Attribution shape is not (1, 5)")
        values = [float(v) for v in attribution[0].double().tolist()]
        if not (all(math.isfinite(v) for v in values) and math.isfinite(logit)):
            raise ExplanationNumericalError("Non-finite attribution or logit")
        difference = logit - self._baseline_logit
        total = math.fsum(values)
        residual = total - difference
        allowed = RESIDUAL_ABS + RESIDUAL_REL * abs(difference)
        if abs(residual) > allowed:
            raise ExplanationNumericalError("Completeness residual exceeds the allowed tolerance")
        return _Numerics(
            logit=logit,
            baseline_logit=self._baseline_logit,
            target_difference=difference,
            attribution_sum=total,
            completeness_residual=residual,
            allowed_residual=allowed,
            attributions=tuple(values),
        )
