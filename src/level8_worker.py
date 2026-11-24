"""Level 8 dedicated explanation worker: child entry point and IPC protocol.

This module is imported by BOTH the parent (protocol + framing helpers only) and the
spawned child (``worker_main``). It must stay cheap and side-effect free at import time:
only the standard library is imported at module level. Torch, ModelRuntime and the
ExplanationEngine are imported inside the child, after thread configuration, and the child
never imports the Level 7 service, the Level 8 service or FastAPI.

Transport: two one-way OS pipes carrying length-prefixed frames (4-byte big-endian length,
then a strict-JSON payload of at most MAX_MESSAGE_BYTES). Reads and writes are
non-blocking and deadline-driven (``FdChannel``), so a stalled or partial peer can never
block the caller beyond its deadline. Messages are validated against exact schemas;
unexpected fields, NaN/Infinity, duplicate keys and oversize frames are rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import select
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

MAX_MESSAGE_BYTES = 65_536
CHILD_SEND_TIMEOUT = 5.0
_HEADER = struct.Struct(">I")

FEATURES = (
    "weekly_km_to_date",
    "sleep_hours_last_night",
    "resting_hr_morning",
    "soreness_score_evening",
    "prior_injury",
)

CODE_OUTSIDE_DOMAIN = "outside_explanation_domain"
CODE_INVALID_INPUT = "invalid_explanation_input"
CODE_NUMERICAL = "explanation_numerical_check_failed"
CODE_INTERNAL = "internal_error"
CHILD_ERROR_CODES = frozenset({CODE_OUTSIDE_DOMAIN, CODE_INVALID_INPUT, CODE_NUMERICAL, CODE_INTERNAL})
# Internal diagnostic subtype of a numerical failure (never client-facing). Only
# ``gate_failure`` is claimed when the engine's own validation check is positively identified;
# anything that cannot be established stays ``unclassified_numerical_failure``.
SUBTYPE_GATE = "gate_failure"
SUBTYPE_ENGINE_FAULT = "engine_fault_probe_passed"
SUBTYPE_UNCLASSIFIED = "unclassified_numerical_failure"
NUMERICAL_SUBTYPES = frozenset({SUBTYPE_GATE, SUBTYPE_ENGINE_FAULT, SUBTYPE_UNCLASSIFIED})
# Messages the committed engine raises (without a wrapped cause) from its own numerical gates:
# residual, non-finite output, output shape, and probability parity. Matched by equality only
# to classify; the text is never transmitted. Tests pin each of them against the real engine.
ENGINE_GATE_MESSAGES = frozenset({
    "Completeness residual exceeds the allowed tolerance",
    "Non-finite attribution or logit",
    "Attribution shape is not (1, 5)",
    "Private-model probability differs from ModelRuntime",
})
FATAL_CODES = frozenset({
    "integrity_error", "artifact_identity_mismatch", "init_failed", "self_test_failed",
    "thread_config_failed", "forbidden_module_loaded",
})
# Modules that must never be loaded in the child (parent application / HTTP stack).
FORBIDDEN_MODULES = (
    "fastapi", "starlette", "uvicorn", "level7_service", "src.level7_service",
    "level8_service", "src.level8_service",
)


class ProtocolError(Exception):
    """Malformed, oversize or schema-violating message."""


class ChannelClosed(Exception):
    """Peer closed the channel (EOF / broken pipe)."""


class ChannelTimeout(Exception):
    """A send did not complete before its deadline."""


# --------------------------------------------------------------------------- codec
def _no_constant(name: str) -> Any:
    raise ProtocolError("non-finite JSON constant")


def _strict_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ProtocolError("non-finite JSON number")
    return value


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ProtocolError("duplicate key")
        out[key] = value
    return out


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_violations(items: Any) -> None:
    if not isinstance(items, list) or len(items) > len(FEATURES):
        raise ProtocolError("bad violations")
    for item in items:
        if not (isinstance(item, dict) and set(item) == {"feature", "min", "max"}
                and item["feature"] in FEATURES and _is_number(item["min"]) and _is_number(item["max"])):
            raise ProtocolError("bad violation entry")


_EXPLANATION_KEYS = {
    "prediction", "logit", "baseline_logit", "target_difference", "attribution_sum",
    "completeness_residual", "allowed_residual", "attributions",
}
_ATTRIBUTION_KEYS = {"feature", "value", "standardized_value", "attribution"}
_PREDICTION_KEYS = {"probability", "decision", "threshold", "device", "selected_approach"}


def validate_explanation(value: Any) -> None:
    """Shape check of Explanation.to_dict() output crossing the process boundary."""
    if not (isinstance(value, dict) and set(value) == _EXPLANATION_KEYS):
        raise ProtocolError("bad explanation")
    pred = value["prediction"]
    if not (isinstance(pred, dict) and set(pred) == _PREDICTION_KEYS):
        raise ProtocolError("bad prediction")
    for key in ("logit", "baseline_logit", "target_difference", "attribution_sum",
                "completeness_residual", "allowed_residual"):
        if not _is_number(value[key]):
            raise ProtocolError("bad number")
    items = value["attributions"]
    if not (isinstance(items, list) and len(items) == len(FEATURES)):
        raise ProtocolError("bad attributions")
    for name, item in zip(FEATURES, items):
        if not (isinstance(item, dict) and set(item) == _ATTRIBUTION_KEYS and item["feature"] == name
                and all(_is_number(item[k]) for k in ("value", "standardized_value", "attribution"))):
            raise ProtocolError("bad attribution item")


def validate_message(message: Any, direction: str) -> dict[str, Any]:
    """Exact-schema validation. direction is 'to_child' or 'to_parent'."""
    if not isinstance(message, dict) or not isinstance(message.get("t"), str):
        raise ProtocolError("bad message")
    kind = message["t"]
    keys = set(message)
    if direction == "to_child":
        if kind == "explain":
            if keys != {"t", "gen", "id", "payload"} or not _is_int(message["gen"]) or not _is_int(message["id"]):
                raise ProtocolError("bad explain")
            payload = message["payload"]
            if not (isinstance(payload, dict) and set(payload) == set(FEATURES)):
                raise ProtocolError("bad payload")
            for name, v in payload.items():
                ok = _is_int(v) if name == "prior_injury" else _is_number(v)
                if not ok:
                    raise ProtocolError("bad payload value")
        elif kind == "shutdown":
            if keys != {"t", "gen"} or not _is_int(message["gen"]):
                raise ProtocolError("bad shutdown")
        else:
            raise ProtocolError("unknown message type")
        return message
    if direction != "to_parent":
        raise ValueError("direction")
    if kind == "ready":
        if (keys != {"t", "gen", "pid", "torch_threads", "torch_interop_threads", "loaded_forbidden_modules",
                     "metadata", "self_test_ms"}
                or not all(_is_int(message[k]) for k in ("gen", "pid", "torch_threads", "torch_interop_threads"))
                or not isinstance(message["metadata"], dict)
                or not isinstance(message["loaded_forbidden_modules"], list)
                or not _is_number(message["self_test_ms"])):
            raise ProtocolError("bad ready")
    elif kind == "result":
        if keys != {"t", "gen", "id", "explanation"} or not _is_int(message["gen"]) or not _is_int(message["id"]):
            raise ProtocolError("bad result")
        validate_explanation(message["explanation"])
    elif kind == "error":
        if keys not in ({"t", "gen", "id", "code"}, {"t", "gen", "id", "code", "violations"},
                        {"t", "gen", "id", "code", "subtype"}):
            raise ProtocolError("bad error")
        if not (_is_int(message["gen"]) and _is_int(message["id"]) and message["code"] in CHILD_ERROR_CODES):
            raise ProtocolError("bad error fields")
        if (message["code"] == CODE_NUMERICAL) != ("subtype" in message):
            raise ProtocolError("subtype is required for, and only for, numerical errors")
        if "subtype" in message and message["subtype"] not in NUMERICAL_SUBTYPES:
            raise ProtocolError("bad subtype")
        if "violations" in message:
            if message["code"] != CODE_OUTSIDE_DOMAIN:
                raise ProtocolError("unexpected violations")
            _validate_violations(message["violations"])
    elif kind == "fatal":
        if keys != {"t", "gen", "code"} or not _is_int(message["gen"]) or message["code"] not in FATAL_CODES:
            raise ProtocolError("bad fatal")
    else:
        raise ProtocolError("unknown message type")
    return message


def encode_message(message: Mapping[str, Any], direction: str) -> bytes:
    validate_message(dict(message), direction)
    try:
        data = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("unencodable message") from exc
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolError("oversize message")
    return data


def decode_message(data: bytes, direction: str) -> dict[str, Any]:
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolError("oversize message")
    try:
        message = json.loads(
            data.decode("utf-8"), parse_constant=_no_constant, parse_float=_strict_float,
            object_pairs_hook=_no_duplicates,
        )
    except ProtocolError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ProtocolError("malformed message") from exc
    return validate_message(message, direction)


# ------------------------------------------------------------------------ framing
class FdChannel:
    """Deadline-driven framing over a pair of OS pipe file descriptors.

    Both descriptors are switched to non-blocking mode. ``poll`` assembles frames from
    partial reads (the buffer is bounded by one frame plus one read) and ``send_frame``
    writes with a deadline. Neither can block past its deadline."""

    def __init__(self, read_fd: int, write_fd: int):
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        self._r, self._w = read_fd, write_fd
        self._buffer = bytearray()

    def _extract(self) -> bytes | None:
        if len(self._buffer) < _HEADER.size:
            return None
        (length,) = _HEADER.unpack_from(self._buffer, 0)
        if length == 0 or length > MAX_MESSAGE_BYTES:
            raise ProtocolError("bad frame length")
        if len(self._buffer) < _HEADER.size + length:
            return None
        data = bytes(self._buffer[_HEADER.size:_HEADER.size + length])
        del self._buffer[:_HEADER.size + length]
        return data

    def poll(self, deadline: float | None, extra_fds: Sequence[int] = (), now=time.monotonic):
        """Wait for a frame. Returns ('frame', bytes) | ('closed',) | ('timeout',) | ('fd', fd).
        Complete data always has priority over extra descriptors. Raises ProtocolError on a
        bad frame header (before any oversize payload is read)."""
        while True:
            frame = self._extract()
            if frame is not None:
                return ("frame", frame)
            timeout = None if deadline is None else max(0.0, deadline - now())
            ready, _, _ = select.select([self._r, *extra_fds], [], [], timeout)
            if self._r in ready:
                try:
                    chunk = os.read(self._r, MAX_MESSAGE_BYTES + _HEADER.size)
                except BlockingIOError:
                    chunk = None
                except OSError:
                    return ("closed",)
                if chunk == b"":
                    return ("closed",)
                if chunk:
                    self._buffer += chunk
                    continue
            for fd in extra_fds:
                if fd in ready:
                    return ("fd", fd)
            if not ready:
                return ("timeout",)

    def send_frame(self, data: bytes, deadline: float | None, now=time.monotonic) -> None:
        if not 0 < len(data) <= MAX_MESSAGE_BYTES:
            raise ProtocolError("bad frame size")
        view = memoryview(_HEADER.pack(len(data)) + data)
        while len(view):
            try:
                written = os.write(self._w, view)
                view = view[written:]
                continue
            except BlockingIOError:
                pass
            except OSError as exc:
                raise ChannelClosed() from exc
            timeout = None if deadline is None else deadline - now()
            if timeout is not None and timeout <= 0:
                raise ChannelTimeout()
            select.select([], [self._w], [], timeout)


# ------------------------------------------------------------- child-side core logic
class InitFailure(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _import_modules():
    """Late imports (after thread configuration). Mirrors the Level 7 layout fallback."""
    try:
        from src import level7_runtime as runtime_module
        from src import level8_explain as explain_module
    except ModuleNotFoundError as exc:
        if exc.name != "src":
            raise
        import level7_runtime as runtime_module  # type: ignore[no-redef]
        import level8_explain as explain_module  # type: ignore[no-redef]
    return runtime_module, explain_module


def _wire_form(explanation) -> dict[str, Any]:
    """Explanation.to_dict() as it will look after crossing the pipe (tuples become lists)."""
    return json.loads(json.dumps(explanation.to_dict(), allow_nan=False))


def _weights_digest(model) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class WorkerCore:
    """Child-side state: verified runtime, engine, startup self-test and request handling.
    Does not touch process-global Torch settings (worker_main does, in the child only)."""

    def __init__(self, serving_root: str | Path, expected_artifacts: Mapping[str, str] | None = None):
        self._rt_mod, self._ex_mod = _import_modules()
        root = Path(serving_root)
        try:
            self.runtime = self._rt_mod.ModelRuntime(root)
        except self._rt_mod.IntegrityError:
            raise InitFailure("integrity_error") from None
        except Exception:
            raise InitFailure("init_failed") from None
        self.artifacts = dict(self.runtime.manifest["artifacts"])
        if expected_artifacts is not None and self.artifacts != dict(expected_artifacts):
            raise InitFailure("artifact_identity_mismatch")
        try:
            self.engine = self._ex_mod.ExplanationEngine(self.runtime)
        except Exception:
            raise InitFailure("init_failed") from None
        self._root = root
        self._weights = _weights_digest(self.runtime.model)
        started = time.perf_counter()
        try:
            result = self.engine.explain(self._rt_mod.SAMPLE_REQUEST)
            same = result.prediction == self.runtime.predict(self._rt_mod.SAMPLE_REQUEST)
            body = _wire_form(result)
            validate_explanation(body)
        except Exception:
            raise InitFailure("self_test_failed") from None
        if not same:
            raise InitFailure("self_test_failed")
        self.self_test_ms = (time.perf_counter() - started) * 1000.0
        self._baseline_result = body

    def metadata(self) -> dict[str, Any]:
        return self.engine.metadata()

    def _healthy(self) -> bool:
        """Post-failure probe: artifacts, weights and a fresh self-test must all still match."""
        try:
            self._rt_mod.verify_serving_artifacts(self._root)
            if _weights_digest(self.runtime.model) != self._weights:
                return False
            return _wire_form(self.engine.explain(self._rt_mod.SAMPLE_REQUEST)) == self._baseline_result
        except Exception:
            return False

    def handle_explain(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Returns the message body (without t/gen/id): kind 'result' or 'error'.

        Classification: out-of-domain / invalid input are request errors; an
        ExplanationNumericalError is reported as a numerical failure ONLY if the worker then
        proves itself healthy (artifacts, weights and the startup self-test unchanged);
        anything else is an internal error, which the parent treats as restart-worthy."""
        ex = self._ex_mod
        try:
            result = self.engine.explain(payload)
            body = _wire_form(result)
            validate_explanation(body)
            return {"kind": "result", "explanation": body}
        except ex.OutsideExplanationDomain as exc:
            return {"kind": "error", "code": CODE_OUTSIDE_DOMAIN, "violations": list(exc.violations)}
        except ex.InvalidExplanationInput:
            return {"kind": "error", "code": CODE_INVALID_INPUT}
        except ex.ExplanationNumericalError as exc:
            if not self._healthy():
                return {"kind": "error", "code": CODE_INTERNAL}
            # A passing probe shows the worker is still sound; it does not prove the original
            # failure was a pure numerical-gate failure, so the subtype is deliberately conservative.
            if exc.__cause__ is not None:
                subtype = SUBTYPE_ENGINE_FAULT  # the engine wraps faults raised inside Captum "from" them
            elif str(exc) in ENGINE_GATE_MESSAGES:
                subtype = SUBTYPE_GATE
            else:
                subtype = SUBTYPE_UNCLASSIFIED
            return {"kind": "error", "code": CODE_NUMERICAL, "subtype": subtype}
        except Exception:
            return {"kind": "error", "code": CODE_INTERNAL}


def _configure_threads() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("thread configuration failed")


def ready_message(core: WorkerCore, generation: int) -> dict[str, Any]:
    import torch

    return {
        "t": "ready", "gen": generation, "pid": os.getpid(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "loaded_forbidden_modules": sorted(set(FORBIDDEN_MODULES) & set(sys.modules)),
        "metadata": core.metadata(), "self_test_ms": core.self_test_ms,
    }


def serve(core: WorkerCore, channel: FdChannel, generation: int) -> None:
    """Request loop. Returns on EOF, shutdown, or any protocol/channel fault. Handled operational
    failures end the loop silently: no traceback, exception text, path or value is ever emitted."""

    def send(message: dict[str, Any]) -> None:
        channel.send_frame(encode_message(message, "to_parent"), time.monotonic() + CHILD_SEND_TIMEOUT)

    try:
        send(ready_message(core, generation))
        while True:
            event = channel.poll(None)
            if event[0] != "frame":
                return  # EOF: the parent is gone or closed the channel
            message = decode_message(event[1], "to_child")
            if message["t"] == "shutdown":
                return
            if message["gen"] != generation:
                return
            reply = core.handle_explain(message["payload"])
            if reply["kind"] == "result":
                send({"t": "result", "gen": generation, "id": message["id"], "explanation": reply["explanation"]})
            else:
                body = {"t": "error", "gen": generation, "id": message["id"], "code": reply["code"]}
                for extra in ("violations", "subtype"):
                    if extra in reply:
                        body[extra] = reply[extra]
                send(body)
    except (ProtocolError, ChannelClosed, ChannelTimeout):
        return
    except Exception:
        return


def worker_main(req_conn, resp_conn, serving_root: str, generation: int,
                expected_artifacts: Mapping[str, str] | None = None) -> None:
    """Child process entry. ``req_conn`` is read-only, ``resp_conn`` write-only.
    Exits (and thereby closes its pipe ends) on EOF, shutdown or any fault."""
    channel = FdChannel(req_conn.fileno(), resp_conn.fileno())

    def fatal(code: str) -> None:
        try:
            channel.send_frame(encode_message({"t": "fatal", "gen": generation, "code": code}, "to_parent"),
                               time.monotonic() + CHILD_SEND_TIMEOUT)
        except Exception:
            pass

    try:
        _configure_threads()
    except Exception:
        fatal("thread_config_failed")
        return
    try:
        core = WorkerCore(serving_root, expected_artifacts)
    except InitFailure as failure:
        fatal(failure.code)
        return
    except Exception:
        fatal("init_failed")
        return
    if set(FORBIDDEN_MODULES) & set(sys.modules):
        fatal("forbidden_module_loaded")
        return
    serve(core, channel, generation)
