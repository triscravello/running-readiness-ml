"""Level 8 FastAPI application: the Level 7 contracts plus ``POST /v1/explanations``.

Composition: a NEW FastAPI app is created. The Level 7 ``/health`` and ``/predict`` API routes
(and its request-validation handler) are re-registered from the imported Level 7 application
without modifying it; no prediction logic is duplicated and no second prediction runtime is
loaded in the parent. Importing this module starts no process or thread (it only triggers the
existing Level 7 import-time runtime construction).

Lifespan: the application owns exactly one bounded ``Dispatcher`` in BACKGROUND mode. Startup
passes the parent runtime's verified artifact hashes as ``expected_artifacts`` and the exact
serving root the parent runtime used; worker initialization proceeds without blocking Level 7
availability (explanations answer 503 until the worker is ready). Shutdown is explicit and runs
off the event loop.

Supported launch for this milestone (not a globally enforced guarantee): ONE worker process,
no reload, started with ``python -m uvicorn src.level8_service:app`` (flat layout:
``python -m uvicorn level8_service:app``). Do not use ``--workers N`` with N > 1 or ``--reload``:
each server process would own its own explanation worker. The ``uvicorn`` console script is not
a supported launcher here; see the L8-2C report.

HTTP waiting is bounded by the job's own 500 ms admission deadline. The HTTP layer never
releases dispatcher capacity itself; on timeout, cancellation or client disconnect it only asks
the dispatcher to cancel, and the dispatcher keeps an active slot until the worker is done.

``POST /v1/comparisons`` parses its own body (bounded, duplicate keys rejected, errors never
echo client text), admits both evaluations atomically under ONE deadline with ONE disconnect
watcher, and returns either a complete ``comparison.v1`` response or a fixed error.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute, APIRouter
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.requests import ClientDisconnect

try:
    from src import level7_service as l7
    from src import level8_compare as comparison
    from src import level8_dispatch as dispatch
    from src import level8_schemas as schemas
    from src.level8_explain import OutsideExplanationDomain
except ModuleNotFoundError as _exc:  # flat (image) layout
    if _exc.name != "src":
        raise
    import level7_service as l7  # type: ignore[no-redef]
    import level8_compare as comparison  # type: ignore[no-redef]
    import level8_dispatch as dispatch  # type: ignore[no-redef]
    import level8_schemas as schemas  # type: ignore[no-redef]
    from level8_explain import OutsideExplanationDomain  # type: ignore[no-redef]

logger = logging.getLogger("level8.service")

ComparisonRequest = comparison.make_comparison_request_model(l7.PredictionRequest)

APP_TITLE = "Running Readiness ML API"
APP_VERSION = "5.0.0"

_STATUS = {
    "outside_explanation_domain": 422,
    "explanation_overloaded": 503,
    "explanation_unavailable": 503,
    "explanation_numerical_check_failed": 503,
    "explanation_deadline_exceeded": 504,
}
_RETRY_AFTER = {"explanation_overloaded": "1", "explanation_unavailable": "5"}
_DISCONNECTED = object()
_TIMED_OUT = object()


def _error(code: str, violations: Any = None) -> JSONResponse:
    """Sanitized operational error: fixed code and message, no values, no internals."""
    if code not in _STATUS:
        code = "explanation_unavailable"
    body: dict[str, Any] = {"code": code, "message": schemas.ERROR_MESSAGES[code]}
    if code == "outside_explanation_domain" and violations is not None:
        body["violations"] = [
            {"feature": v["feature"], "min": v["min"], "max": v["max"]} for v in violations
        ]
    headers = {"Cache-Control": "no-store"}
    if code in _RETRY_AFTER:
        headers["Retry-After"] = _RETRY_AFTER[code]
    return JSONResponse(status_code=_STATUS[code], content={"error": body}, headers=headers)


async def _watch_disconnect(request: Request) -> None:
    """Completes when the client disconnects. Uvicorn does not cancel a handler on disconnect,
    so the receive channel is observed explicitly (the request body has already been read)."""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


async def _await_outcome(request: Request, dispatcher: Any, job: Any, clock: Callable[[], float]):
    """Bounded asynchronous wait for the dispatcher future. Never blocks the event loop, never
    waits past the job's own deadline, and always asks the dispatcher to cancel an unfinished job
    (queued: released at once; active: slot stays reserved until the worker finishes)."""
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future = loop.create_future()

    def relay(_future) -> None:  # may run on the dispatcher thread
        def deliver() -> None:
            if not waiter.done():
                waiter.set_result(None)

        try:
            loop.call_soon_threadsafe(deliver)
        except RuntimeError:  # the loop is already closed
            pass

    job.future.add_done_callback(relay)
    watcher = asyncio.create_task(_watch_disconnect(request))
    try:
        remaining = max(0.0, job.deadline - clock())
        done, _ = await asyncio.wait({waiter, watcher}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
        if waiter in done and job.future.done() and not job.future.cancelled():
            return job.future.result(0)
        if watcher in done:
            return _DISCONNECTED
        return _TIMED_OUT
    finally:
        watcher.cancel()
        waiter.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        if not job.future.done():
            dispatcher.cancel(job)


# --------------------------------------------------------------- comparison: errors
# The largest legitimate request (two five-field objects, full-precision numbers, indented) is
# under 1 KB, so 4 KB leaves ample headroom while bounding what is read and parsed.
MAX_COMPARISON_BODY_BYTES = 4096

_COMPARISON_STATUS = {
    **_STATUS,
    "unsupported_media_type": 415,
    "comparison_request_too_large": 413,
    "malformed_comparison_json": 422,
    "duplicate_json_key": 422,
    "invalid_comparison_request": 422,
}
_COMPARISON_MESSAGES = {
    **schemas.ERROR_MESSAGES,
    "unsupported_media_type": "The request must be uncompressed application/json in UTF-8.",
    "comparison_request_too_large": "The comparison request body is too large.",
    "malformed_comparison_json": "The request body is not strict JSON.",
    "duplicate_json_key": "The request body repeats a JSON object key.",
    "invalid_comparison_request": "The comparison request is not valid.",
}


class ComparisonViolation(BaseModel):
    """Only allowlisted identifiers and committed bounds; never client text or submitted values."""

    model_config = ConfigDict(extra="forbid")
    scenario: Literal[*comparison.SCENARIOS] | None = None
    feature: Literal[*schemas.FEATURES] | None = None
    problem: Literal["missing_field", "unknown_field", "invalid_value"] | None = None
    min: float | None = None
    max: float | None = None


class ComparisonErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: Literal[*_COMPARISON_STATUS]
    message: str
    violations: list[ComparisonViolation] | None = None


class ComparisonErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: ComparisonErrorBody


def _comparison_error(code: str, violations: list[dict[str, Any]] | None = None) -> JSONResponse:
    if code not in _COMPARISON_STATUS:
        code = "explanation_unavailable"
    body: dict[str, Any] = {"code": code, "message": _COMPARISON_MESSAGES[code]}
    if violations is not None:
        body["violations"] = violations
    headers = {"Cache-Control": "no-store"}
    if code in _RETRY_AFTER:
        headers["Retry-After"] = _RETRY_AFTER[code]
    return JSONResponse(status_code=_COMPARISON_STATUS[code], content={"error": body}, headers=headers)


def _client_disconnected() -> JSONResponse:
    return JSONResponse(status_code=499, content={"error": {"code": "client_disconnected"}},
                        headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------- comparison: request
class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey()
        out[key] = value
    return out


def _reject_constant(_name: str) -> Any:
    raise ValueError("non-finite constant")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def _is_json_media_type(request: Request) -> bool:
    types = request.headers.getlist("content-type")
    encodings = request.headers.getlist("content-encoding")
    if len(types) != 1 or any(e.strip().lower() != "identity" for e in encodings):
        return False
    media, *parameters = types[0].split(";")
    return media.strip().lower() == "application/json" and all(
        p.strip().lower() in ("charset=utf-8", 'charset="utf-8"') for p in parameters)


def _violations_from(errors: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Reduce validation errors to allowlisted identifiers, deduplicated in a fixed order."""
    problems = {"missing": "missing_field", "extra_forbidden": "unknown_field"}
    found = set()
    for error in errors:
        location = error.get("loc", ())
        problem = problems.get(error.get("type"), "invalid_value")
        unknown_top_level = problem == "unknown_field" and len(location) == 1
        scenario = location[0] if location and not unknown_top_level and location[0] in comparison.SCENARIOS else None
        feature = None  # the location of an unknown field is client text and is never copied
        if scenario and problem != "unknown_field" and len(location) > 1 and location[1] in schemas.FEATURES:
            feature = location[1]
        found.add((scenario, feature, problem))

    def order(item):
        scenario, feature, problem = item
        return (-1 if scenario is None else comparison.SCENARIOS.index(scenario),
                -1 if feature is None else schemas.FEATURES.index(feature), problem)

    return [{k: v for k, v in zip(("scenario", "feature", "problem"), item) if v is not None}
            for item in sorted(found, key=order)]


async def _read_comparison(request: Request) -> Any:
    """Bounded, strict parsing. Returns a validated ``ComparisonRequest`` or an error response."""
    if not _is_json_media_type(request):
        return _comparison_error("unsupported_media_type")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_COMPARISON_BODY_BYTES:
        return _comparison_error("comparison_request_too_large")
    body = bytearray()
    try:
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_COMPARISON_BODY_BYTES:
                return _comparison_error("comparison_request_too_large")
    except ClientDisconnect:
        return _client_disconnected()
    try:
        document = json.loads(bytes(body).decode("utf-8"), object_pairs_hook=_unique_object,
                              parse_constant=_reject_constant, parse_float=_finite_float)
    except _DuplicateKey:
        return _comparison_error("duplicate_json_key")
    except (ValueError, RecursionError):  # includes invalid UTF-8 and JSON syntax errors
        return _comparison_error("malformed_comparison_json")
    try:
        return ComparisonRequest.model_validate(document)
    except ValidationError as exc:
        return _comparison_error("invalid_comparison_request", _violations_from(exc.errors()))
    except Exception:
        return _comparison_error("invalid_comparison_request", [])


# --------------------------------------------------------------- comparison: waiting
def _settled(job: Any) -> Any:
    """The job's outcome if its future is resolved, else None."""
    if not job.future.done():
        return None
    try:
        return job.future.result(0)
    except (Exception, concurrent.futures.CancelledError):
        return dispatch.Outcome("error", None, "explanation_unavailable")


def _select(jobs: tuple) -> tuple[int | None, list[Any]] | None:
    """Decide in SCENARIO order, never completion order. Returns None while undecided,
    (None, outcomes) for success, or (index_of_failure, outcomes). A failed ``modified`` does not
    decide anything while ``original`` is pending: a worker failure resolves queued jobs before the
    active one, so the later job's error may be only collateral."""
    outcomes = [_settled(job) for job in jobs]
    if outcomes[0] is None:
        return None
    if outcomes[0].kind != "result":
        return (0, outcomes)
    if any(o is None for o in outcomes):
        return None
    for index, outcome in enumerate(outcomes):
        if outcome.kind != "result":
            return (index, outcomes)
    return (None, outcomes)


async def _await_comparison(request: Request, dispatcher: Any, jobs: tuple, clock: Callable[[], float]) -> Any:
    """One shared deadline and one disconnect watcher. Unfinished jobs are always cancelled,
    the later one first, before anything is awaited in cleanup."""
    watcher = None
    try:
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()

        def relay(_future) -> None:  # may run on the dispatcher thread
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:  # the loop is already closed
                pass

        for job in jobs:
            job.future.add_done_callback(relay)
        watcher = asyncio.create_task(_watch_disconnect(request))
        deadline = jobs[0].deadline
        while True:
            wake.clear()
            selected = _select(jobs)
            if selected is not None:
                return selected
            if watcher.done():
                return _DISCONNECTED
            woken = asyncio.create_task(wake.wait())
            try:
                done, _ = await asyncio.wait({woken, watcher}, timeout=max(0.0, deadline - clock()),
                                             return_when=asyncio.FIRST_COMPLETED)
            finally:
                woken.cancel()
            if not done:
                selected = _select(jobs)
                return selected if selected is not None else _TIMED_OUT
    finally:
        for job in reversed(jobs):
            if not job.future.done():
                dispatcher.cancel(job)
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)


def _default_dispatcher_factory(root: Any, artifacts: dict[str, str]) -> Any:
    return dispatch.Dispatcher(str(root), expected_artifacts=artifacts)


def build_app(dispatcher_factory: Callable[[Any, dict[str, str]], Any] | None = None,
              clock: Callable[[], float] = time.monotonic) -> FastAPI:
    """Create the application. ``dispatcher_factory(serving_root, expected_artifacts)`` and ``clock``
    (the monotonic time source for job deadlines; the dispatcher's own is ``time.monotonic``) are the only
    seams (tests inject controlled ones); production uses the real dispatcher and the real clock."""
    factory = dispatcher_factory or _default_dispatcher_factory
    manifest = l7.runtime.manifest
    expected_artifacts = dict(manifest["artifacts"])
    parent_info = {
        "evidence_scope": manifest["evidence_scope"],
        "model_notice": manifest["model_notice"],
        "inherited_v1_preprocessing_limitation": manifest["inherited_v1_preprocessing_limitation"],
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:  # one-time trusted reference for response validation; failure leaves Level 7 unaffected
            app.state.explanation_reference = await asyncio.to_thread(schemas.make_reference, l7.runtime)
        except Exception:
            app.state.explanation_reference = None
            logger.error("explanation reference could not be established; explanations are disabled")
        dispatcher = factory(l7.BASE_DIR, expected_artifacts)
        dispatcher.start()  # background mode only; the worker starts on the dispatcher thread
        app.state.explanation_dispatcher = dispatcher
        app.state.explanation_cleanup_confirmed = None
        try:
            yield
        finally:
            app.state.explanation_dispatcher = None  # stop routing new requests to it
            await asyncio.to_thread(dispatcher.shutdown)
            confirmed = dispatcher.diagnostics()["state"] == dispatch.STOPPED
            app.state.explanation_cleanup_confirmed = confirmed
            if not confirmed:
                logger.error("explanation worker termination could not be confirmed")

    app = FastAPI(title=APP_TITLE, version=APP_VERSION, description=l7.app.description, lifespan=lifespan)
    app.state.explanation_dispatcher = None
    app.state.explanation_reference = None
    app.state.explanation_cleanup_confirmed = None

    # Level 7 contracts: re-register its API routes and its validation handler (no mutation of l7.app).
    legacy = APIRouter()
    legacy.routes.extend(r for r in l7.app.router.routes if isinstance(r, APIRoute))
    app.include_router(legacy)
    app.add_exception_handler(RequestValidationError, l7.app.exception_handlers[RequestValidationError])

    error_schema = {"model": schemas.ErrorResponse}

    @app.post(
        "/v1/explanations",
        response_model=schemas.ExplanationResponse,
        responses={
            422: {"description": "Request validation error (Level 7 shape) or outside_explanation_domain.",
                  **error_schema},
            503: {"description": "explanation_overloaded (Retry-After: 1), explanation_unavailable "
                                 "(Retry-After: 5) or explanation_numerical_check_failed.", **error_schema},
            504: {"description": "explanation_deadline_exceeded.", **error_schema},
        },
    )
    async def explain(body: l7.PredictionRequest, request: Request):
        dispatcher = request.app.state.explanation_dispatcher
        reference = request.app.state.explanation_reference
        if dispatcher is None or reference is None:
            return _error("explanation_unavailable")
        try:
            payload = body.model_dump()
            job = dispatcher.submit(payload)  # validates the domain BEFORE taking capacity
        except OutsideExplanationDomain as exc:
            return _error("outside_explanation_domain", exc.violations)
        except dispatch.AdmissionRejected as exc:
            return _error(exc.code)
        except Exception:  # e.g. InvalidExplanationInput after Pydantic accepted the body: fail closed
            return _error("explanation_unavailable")
        outcome = await _await_outcome(request, dispatcher, job, clock)
        if outcome is _DISCONNECTED:
            return JSONResponse(status_code=499, content={"error": {"code": "client_disconnected"}})
        if outcome is _TIMED_OUT:
            return _error("explanation_deadline_exceeded")
        if outcome.kind == "result":
            try:
                response = schemas.build_response(outcome.explanation, dispatcher.metadata(), parent_info,
                                                  reference, payload)
                content = response.model_dump(mode="json")
            except Exception:  # malformed or inconsistent worker payload: no partial result
                return _error("explanation_unavailable")
            if clock() >= job.deadline:  # final decision: a success is never accepted at/after its deadline
                return _error("explanation_deadline_exceeded")
            return JSONResponse(content=content, headers={"Cache-Control": "no-store"})
        violations = outcome.violations if outcome.code == "outside_explanation_domain" else None
        return _error(outcome.code, violations)

    async def compare_once(request: Request) -> JSONResponse:
        parsed = await _read_comparison(request)
        if isinstance(parsed, JSONResponse):
            return parsed
        dispatcher = request.app.state.explanation_dispatcher
        reference = request.app.state.explanation_reference
        if dispatcher is None or reference is None:
            return _comparison_error("explanation_unavailable")
        payloads = (parsed.original.model_dump(), parsed.modified.model_dump())
        violations = comparison.eligibility_violations(*payloads)  # both scenarios, before any capacity
        if violations:
            return _comparison_error("outside_explanation_domain", [dict(v) for v in violations])
        try:
            jobs = dispatcher.submit_comparison(*payloads)
        except dispatch.AdmissionRejected as exc:
            return _comparison_error(exc.code)
        deadline = jobs[0].deadline
        selected = await _await_comparison(request, dispatcher, jobs, clock)
        if selected is _DISCONNECTED:
            return _client_disconnected()
        if selected is _TIMED_OUT:
            return _comparison_error("explanation_deadline_exceeded")
        failed, outcomes = selected
        if failed is not None:
            outcome = outcomes[failed]
            if outcome.code == "outside_explanation_domain":
                return _comparison_error(outcome.code, [
                    {"scenario": comparison.SCENARIOS[failed], "feature": v["feature"], "min": v["min"],
                     "max": v["max"]} for v in outcome.violations])
            return _comparison_error(outcome.code)
        if clock() >= deadline:
            return _comparison_error("explanation_deadline_exceeded")
        generations = [o.generation for o in outcomes]
        if any(type(g) is not int for g in generations) or len(set(generations)) != 1 \
                or (len(jobs) == 1) != (payloads[0] == payloads[1]):
            return _comparison_error("explanation_unavailable")
        try:  # the committed trusted path for each result, then only the verified comparison builder
            metadata = dispatcher.metadata()
            explanations = [schemas.build_response(o.explanation, metadata, parent_info, reference, payload)
                            for o, payload in zip(outcomes, payloads)]
            if clock() >= deadline:
                return _comparison_error("explanation_deadline_exceeded")
            result = comparison.build_comparison(explanations[0], explanations[-1])
            response = JSONResponse(content=result.model_dump(mode="json"), headers={"Cache-Control": "no-store"})
        except Exception:  # malformed or inconsistent results: no partial comparison
            return _comparison_error("explanation_unavailable")
        if clock() >= deadline:  # final decision, after the response has been rendered
            return _comparison_error("explanation_deadline_exceeded")
        return response

    comparison_error = {"model": ComparisonErrorResponse}
    request_schema = ComparisonRequest.model_json_schema(ref_template="#/components/schemas/{model}")
    request_schema.pop("$defs", None)  # PredictionRequest is already a component of the existing routes

    @app.post(
        "/v1/comparisons",
        response_model=comparison.ComparisonResponse,
        responses={
            413: {"description": "comparison_request_too_large.", **comparison_error},
            415: {"description": "unsupported_media_type.", **comparison_error},
            422: {"description": "malformed_comparison_json, duplicate_json_key, invalid_comparison_request or "
                                 "outside_explanation_domain.", **comparison_error},
            503: {"description": "explanation_overloaded (Retry-After: 1), explanation_unavailable "
                                 "(Retry-After: 5) or explanation_numerical_check_failed.", **comparison_error},
            504: {"description": "explanation_deadline_exceeded.", **comparison_error},
        },
        openapi_extra={"requestBody": {"required": True,
                                       "content": {"application/json": {"schema": request_schema}}}},
    )
    async def compare_scenarios(request: Request):
        try:
            return await compare_once(request)
        except Exception:  # fail closed with a fixed error; never a traceback or client text
            return _comparison_error("explanation_unavailable")

    return app


app = build_app()
