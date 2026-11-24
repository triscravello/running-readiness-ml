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
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute, APIRouter

try:
    from src import level7_service as l7
    from src import level8_dispatch as dispatch
    from src import level8_schemas as schemas
    from src.level8_explain import OutsideExplanationDomain
except ModuleNotFoundError as _exc:  # flat (image) layout
    if _exc.name != "src":
        raise
    import level7_service as l7  # type: ignore[no-redef]
    import level8_dispatch as dispatch  # type: ignore[no-redef]
    import level8_schemas as schemas  # type: ignore[no-redef]
    from level8_explain import OutsideExplanationDomain  # type: ignore[no-redef]

logger = logging.getLogger("level8.service")

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

    return app


app = build_app()
