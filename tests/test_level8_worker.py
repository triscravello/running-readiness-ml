"""Level 8 worker tests: IPC protocol, framing, child-side core logic, and REAL spawned processes.

Layers:
  * codec / FdChannel unit tests (no processes);
  * WorkerCore in-process tests (real committed engine, fault injection by patching);
  * real spawn tests with controlled fault-injecting child entry points defined in this module
    (test fixtures, not production switches) and with the real engine child.

This module is imported by spawned fault-injection children, so its top level must stay light:
heavy imports (torch, the engine, the dispatcher) happen lazily inside tests.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

from src import level8_worker as w

ROOT = Path(__file__).resolve().parents[1]
VALID = {
    "weekly_km_to_date": 32.0, "sleep_hours_last_night": 7.0, "resting_hr_morning": 50.0,
    "soreness_score_evening": 4.0, "prior_injury": 0,
}


def wait_until(predicate, timeout=90.0, interval=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def explanation():
    return {
        "prediction": {"probability": 0.5, "decision": 1, "threshold": 0.5, "device": "cpu", "selected_approach": "x"},
        "logit": 0.0, "baseline_logit": 1.0, "target_difference": -1.0, "attribution_sum": -1.0,
        "completeness_residual": 0.0, "allowed_residual": 0.007,
        "attributions": [{"feature": f, "value": 1.0, "standardized_value": 0.0, "attribution": -0.2}
                         for f in w.FEATURES],
    }


# ------------------------------------------------- fault-injecting child entry points
def _boot(req, resp, gen, expected):
    channel = w.FdChannel(req.fileno(), resp.fileno())
    ready = {
        "t": "ready", "gen": gen, "pid": os.getpid(), "torch_threads": 1, "torch_interop_threads": 1,
        "loaded_forbidden_modules": [], "self_test_ms": 1.0,
        "metadata": {"model": {"integrity_status": "passed", "artifact_sha256": dict(expected or {})}},
    }
    channel.send_frame(w.encode_message(ready, "to_parent"), time.monotonic() + 5)
    return channel


def _request(channel):
    event = channel.poll(None)
    return w.decode_message(event[1], "to_child") if event[0] == "frame" else None


def _next_explain(channel):
    """Next explain request, or None on EOF/shutdown (fixtures must exit quietly then)."""
    message = _request(channel)
    return None if message is None or message["t"] == "shutdown" else message


def _reply(channel, gen, request_id, **extra):
    msg = {"t": "result", "gen": gen, "id": request_id, "explanation": explanation()}
    msg.update(extra)
    channel.send_frame(w.encode_message(msg, "to_parent"), time.monotonic() + 5)


def entry_echo(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    while True:
        message = _request(channel)
        if message is None or message["t"] == "shutdown":
            return
        _reply(channel, gen, message["id"])


def entry_crash_on_explain(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    if _next_explain(channel) is None:
        return
    os._exit(3)


def entry_partial_frame_then_stall(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    if _next_explain(channel) is None:
        return
    os.write(resp.fileno(), struct.pack(">I", 100) + b"0123456789")  # header promises 100 bytes
    time.sleep(3600)


def entry_never_reads(req, resp, root, gen, expected=None):
    _boot(req, resp, gen, expected)
    time.sleep(3600)


def entry_garbage(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    if _next_explain(channel) is None:
        return
    channel.send_frame(b"this is not json", time.monotonic() + 5)
    time.sleep(3600)


def entry_oversize_header(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    if _next_explain(channel) is None:
        return
    os.write(resp.fileno(), struct.pack(">I", w.MAX_MESSAGE_BYTES + 1))
    time.sleep(3600)


def entry_nan_result(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    message = _next_explain(channel)
    if message is None:
        return
    raw = json.dumps({"t": "result", "gen": gen, "id": message["id"], "explanation": explanation()}).replace(
        '"logit": 0.0', '"logit": NaN')
    channel.send_frame(raw.encode(), time.monotonic() + 5)
    time.sleep(3600)


def entry_stale_generation_then_result(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    while True:
        message = _request(channel)
        if message is None or message["t"] == "shutdown":
            return
        _reply(channel, gen + 99, message["id"])
        _reply(channel, gen, message["id"])


def entry_wrong_request_id(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    message = _next_explain(channel)
    if message is None:
        return
    _reply(channel, gen, message["id"] + 1000)
    time.sleep(3600)


def entry_hold_after_request(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    if _next_explain(channel) is None:
        return
    time.sleep(3600)  # received the request, never answers


def entry_slow_echo(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    while True:
        message = _next_explain(channel)
        if message is None:
            return
        time.sleep(0.1)
        _reply(channel, gen, message["id"])


def entry_probe_stall(req, resp, root, gen, expected=None):
    """Real engine child whose first explanation fails numerically and whose health probe then
    hangs; it announces that it is inside the probe through a marker file (test-only fixture)."""
    w._configure_threads()
    core = w.WorkerCore(root, expected)
    original, state = core.engine._ig.attribute, {"n": 0}

    def skew_first(*a, **k):
        state["n"] += 1
        out = original(*a, **k)
        return out + 0.1 if state["n"] == 1 else out

    core.engine._ig.attribute = skew_first

    def stalled_probe():
        Path(os.environ["LEVEL8_TEST_PROBE_MARKER"]).write_text("inside-probe")
        time.sleep(3600)

    core._healthy = stalled_probe
    w.serve(core, w.FdChannel(req.fileno(), resp.fileno()), gen)


def entry_exit_on_eof(req, resp, root, gen, expected=None):
    channel = _boot(req, resp, gen, expected)
    while channel.poll(None)[0] == "frame":
        pass
    os._exit(0)  # reached only on EOF (parent closed its write end)


# ------------------------------------------------------------------------- codec tests
class CodecTests(unittest.TestCase):
    def test_roundtrip_all_message_types(self):
        explain = {"t": "explain", "gen": 1, "id": 7, "payload": dict(VALID)}
        self.assertEqual(w.decode_message(w.encode_message(explain, "to_child"), "to_child"), explain)
        shutdown = {"t": "shutdown", "gen": 3}
        self.assertEqual(w.decode_message(w.encode_message(shutdown, "to_child"), "to_child"), shutdown)
        result = {"t": "result", "gen": 1, "id": 7, "explanation": explanation()}
        self.assertEqual(w.decode_message(w.encode_message(result, "to_parent"), "to_parent"), result)
        error = {"t": "error", "gen": 1, "id": 7, "code": w.CODE_OUTSIDE_DOMAIN,
                 "violations": [{"feature": "weekly_km_to_date", "min": 20.0, "max": 45.0}]}
        self.assertEqual(w.decode_message(w.encode_message(error, "to_parent"), "to_parent"), error)
        for subtype in sorted(w.NUMERICAL_SUBTYPES):
            numerical = {"t": "error", "gen": 1, "id": 7, "code": w.CODE_NUMERICAL, "subtype": subtype}
            self.assertEqual(w.decode_message(w.encode_message(numerical, "to_parent"), "to_parent"), numerical)
        fatal = {"t": "fatal", "gen": 1, "code": "init_failed"}
        self.assertEqual(w.decode_message(w.encode_message(fatal, "to_parent"), "to_parent"), fatal)

    def test_rejections(self):
        good = {"t": "explain", "gen": 1, "id": 7, "payload": dict(VALID)}
        cases = {
            "nan": b'{"t":"shutdown","gen":NaN}', "infinity": b'{"t":"shutdown","gen":Infinity}',
            "overflow_float": b'{"t":"explain","gen":1,"id":1,"payload":{"weekly_km_to_date":1e999}}',
            "malformed": b'{"t":', "not_object": b"[1,2]", "not_utf8": b"\xff\xfe",
            "duplicate_key": b'{"t":"shutdown","gen":1,"gen":2}',
            "unknown_type": b'{"t":"nope","gen":1}',
            "extra_field": json.dumps(dict(good, extra=1)).encode(),
            "missing_field": json.dumps({"t": "explain", "gen": 1, "id": 7}).encode(),
            "bool_generation": json.dumps(dict(good, gen=True)).encode(),
            "string_id": json.dumps(dict(good, id="7")).encode(),
            "payload_extra_key": json.dumps(dict(good, payload=dict(VALID, extra=1))).encode(),
            "fractional_injury": json.dumps(dict(good, payload=dict(VALID, prior_injury=0.5))).encode(),
            "oversize": b" " * (w.MAX_MESSAGE_BYTES + 1),
        }
        for label, data in cases.items():
            with self.subTest(case=label), self.assertRaises(w.ProtocolError):
                w.decode_message(data, "to_child")
        for bad in (
            {"t": "result", "gen": 1, "id": 1, "explanation": dict(explanation(), logit=float("nan"))},
            {"t": "result", "gen": 1, "id": 1, "explanation": dict(explanation(), extra=1)},
            {"t": "error", "gen": 1, "id": 1, "code": "boom"},
            {"t": "error", "gen": 1, "id": 1, "code": w.CODE_NUMERICAL, "violations": []},
            {"t": "fatal", "gen": 1, "code": "arbitrary text /path"},
            {"t": "explain", "gen": 1, "id": 1, "payload": dict(VALID)},  # wrong direction
        ):
            with self.subTest(bad=str(bad)[:40]), self.assertRaises(w.ProtocolError):
                w.encode_message(bad, "to_parent")

    def test_numerical_subtype_is_required_and_fixed_vocabulary(self):
        base = {"t": "error", "gen": 1, "id": 7}
        bad = [
            dict(base, code=w.CODE_NUMERICAL),  # subtype missing
            dict(base, code=w.CODE_NUMERICAL, subtype="made_up"),
            dict(base, code=w.CODE_NUMERICAL, subtype="Completeness residual exceeds the allowed tolerance"),
            dict(base, code=w.CODE_INTERNAL, subtype=w.SUBTYPE_GATE),  # subtype on a non-numerical error
            dict(base, code=w.CODE_NUMERICAL, subtype=7),
        ]
        for message in bad:
            with self.subTest(message=str(message)), self.assertRaises(w.ProtocolError):
                w.encode_message(message, "to_parent")
            with self.assertRaises(w.ProtocolError):
                w.decode_message(json.dumps(message).encode(), "to_parent")

    def test_oversize_encode_is_rejected(self):
        message = {"t": "ready", "gen": 1, "pid": 1, "torch_threads": 1, "torch_interop_threads": 1,
                   "loaded_forbidden_modules": [], "self_test_ms": 1.0, "metadata": {"x": "y" * 70_000}}
        with self.assertRaises(w.ProtocolError):
            w.encode_message(message, "to_parent")

    def test_feature_order_matches_the_engine(self):
        from src.level8_explain import EXPLANATION_RANGES

        self.assertEqual(tuple(EXPLANATION_RANGES), w.FEATURES)


# ----------------------------------------------------------------------- framing tests
class FdChannelTests(unittest.TestCase):
    def pair(self):
        r, wr = os.pipe()
        self.addCleanup(lambda: [os.close(fd) for fd in (r, wr) if self._open(fd)])
        a = w.FdChannel(r, wr)  # reads from r, writes to wr (loopback)
        return a, r, wr

    @staticmethod
    def _open(fd):
        try:
            os.fstat(fd)
            return True
        except OSError:
            return False

    def test_loopback_frame_and_partial_assembly(self):
        ch, r, wr = self.pair()
        ch.send_frame(b'{"a":1}', time.monotonic() + 1)
        self.assertEqual(ch.poll(time.monotonic() + 1), ("frame", b'{"a":1}'))
        os.write(wr, struct.pack(">I", 5) + b"ab")  # partial frame
        self.assertEqual(ch.poll(time.monotonic() + 0.05), ("timeout",))  # bounded, buffer retained
        os.write(wr, b"cde")
        self.assertEqual(ch.poll(time.monotonic() + 1), ("frame", b"abcde"))

    def test_oversize_header_rejected_before_payload(self):
        ch, r, wr = self.pair()
        os.write(wr, struct.pack(">I", w.MAX_MESSAGE_BYTES + 1))
        with self.assertRaises(w.ProtocolError):
            ch.poll(time.monotonic() + 1)
        ch2, r2, wr2 = self.pair()
        os.write(wr2, struct.pack(">I", 0))
        with self.assertRaises(w.ProtocolError):
            ch2.poll(time.monotonic() + 1)

    def test_send_size_limits(self):
        ch, r, wr = self.pair()
        import fcntl

        if hasattr(fcntl, "F_SETPIPE_SZ"):  # a maximum frame (65,540 bytes) exceeds the default 64 KiB pipe
            fcntl.fcntl(wr, fcntl.F_SETPIPE_SZ, 1 << 20)
        for data in (b"", b"x" * (w.MAX_MESSAGE_BYTES + 1)):
            with self.assertRaises(w.ProtocolError):
                ch.send_frame(data, None)
        ch.send_frame(b"x" * w.MAX_MESSAGE_BYTES, time.monotonic() + 1)  # exactly the limit is allowed

    def test_stalled_peer_send_times_out_at_the_deadline(self):
        ch, r, wr = self.pair()
        filler = b"x" * w.MAX_MESSAGE_BYTES
        started = time.monotonic()
        with self.assertRaises(w.ChannelTimeout):
            for _ in range(64):  # nobody reads: the pipe fills and the send must give up
                ch.send_frame(filler, time.monotonic() + 0.2)
        self.assertLess(time.monotonic() - started, 5.0)

    def test_eof_and_broken_pipe(self):
        r1, w1 = os.pipe()
        r2, w2 = os.pipe()
        ch = w.FdChannel(r1, w2)
        os.close(w1)  # peer closes its write end: EOF for the reader
        self.assertEqual(ch.poll(time.monotonic() + 1), ("closed",))
        os.close(r2)  # peer closes its read end: broken pipe for the writer
        with self.assertRaises(w.ChannelClosed):
            ch.send_frame(b"x", time.monotonic() + 1)
        os.close(r1)
        os.close(w2)

    def test_extra_fd_is_reported_but_data_has_priority(self):
        ch, r, wr = self.pair()
        er, ew = os.pipe()
        self.addCleanup(lambda: [os.close(fd) for fd in (er, ew)])
        os.write(ew, b"x")
        self.assertEqual(ch.poll(time.monotonic() + 1, extra_fds=(er,)), ("fd", er))
        ch.send_frame(b"data", time.monotonic() + 1)
        self.assertEqual(ch.poll(time.monotonic() + 1, extra_fds=(er,)), ("frame", b"data"))


# ----------------------------------------------------------- in-process WorkerCore tests
class WorkerCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.before = (torch.get_num_threads(), torch.get_num_interop_threads())
        cls.artifacts = json.loads((ROOT / "serving-manifest.json").read_text())["artifacts"]

    def core(self):
        return w.WorkerCore(ROOT, self.artifacts)

    def test_initialisation_and_self_test(self):
        core = self.core()
        self.assertEqual(core.metadata()["model"]["integrity_status"], "passed")
        self.assertEqual(core.metadata()["model"]["artifact_sha256"], self.artifacts)
        self.assertGreater(core.self_test_ms, 0)
        self.assertEqual((self.torch.get_num_threads(), self.torch.get_num_interop_threads()), self.before)

    def test_init_failures_have_stable_codes(self):
        with self.assertRaises(w.InitFailure) as ctx:
            w.WorkerCore(ROOT, {"artifacts/v4/level6_selected_model.pt": "0" * 64})
        self.assertEqual(ctx.exception.code, "artifact_identity_mismatch")
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(w.InitFailure) as ctx:
            w.WorkerCore(empty)
        self.assertEqual(ctx.exception.code, "integrity_error")

    def test_request_results_and_request_errors(self):
        core = self.core()
        reply = core.handle_explain(VALID)
        self.assertEqual(reply["kind"], "result")
        self.assertEqual(reply["explanation"], w._wire_form(core.engine.explain(VALID)))
        json.dumps(reply["explanation"], allow_nan=False)
        out = core.handle_explain(dict(VALID, weekly_km_to_date=19.0))
        self.assertEqual((out["code"], out["violations"][0]["feature"]), (w.CODE_OUTSIDE_DOMAIN, "weekly_km_to_date"))
        bad = core.handle_explain(dict(VALID, prior_injury=0.5))
        self.assertEqual(bad["code"], w.CODE_INVALID_INPUT)

    def test_numerical_failure_classification_and_subtypes(self):
        def once(core, wrap):
            original = core.engine._ig.attribute
            state = {"n": 0}

            def patched(*a, **k):
                state["n"] += 1
                if state["n"] == 1:
                    return wrap(original, a, k)
                return original(*a, **k)

            core.engine._ig.attribute = patched
            return core

        # Engine gate failure (completeness residual): identified positively by the engine's own check.
        core = once(self.core(), lambda f, a, k: f(*a, **k) + 0.1)
        self.assertEqual(core.handle_explain(VALID), {"kind": "error", "code": w.CODE_NUMERICAL, "subtype": w.SUBTYPE_GATE})
        # Non-finite output: the engine's non-finite check.
        core = once(self.core(), lambda f, a, k: f(*a, **k) * float("nan"))
        self.assertEqual(core.handle_explain(VALID)["subtype"], w.SUBTYPE_GATE)
        # Probability parity gate (private model versus ModelRuntime).
        core = self.core()
        original_predict, calls = core.runtime.predict, {"n": 0}

        def shifted(payload):
            out = original_predict(payload)
            calls["n"] += 1
            if calls["n"] == 1:
                out["probability"] += 1e-4
            return out

        core.runtime.predict = shifted
        self.assertEqual(core.handle_explain(VALID)["subtype"], w.SUBTYPE_GATE)
        # A fault inside Captum, wrapped by the engine: the worker proves healthy afterwards.
        def boom(f, a, k):
            raise RuntimeError("transient")

        self.assertEqual(once(self.core(), boom).handle_explain(VALID)["subtype"], w.SUBTYPE_ENGINE_FAULT)
        # Ambiguous: a numerical error the committed contract cannot attribute -> conservative subtype.
        core = self.core()
        ex = core._ex_mod
        raised = {"n": 0}
        original_explain = core.engine.explain

        def odd(payload):
            raised["n"] += 1
            if raised["n"] == 1:
                raise ex.ExplanationNumericalError("something else entirely")
            return original_explain(payload)

        core.engine.explain = odd
        self.assertEqual(core.handle_explain(VALID)["subtype"], w.SUBTYPE_UNCLASSIFIED)
        # A persistent fault cannot pass the health probe: internal error, no subtype (restart-worthy).
        core = self.core()
        original2 = core.engine._ig.attribute
        core.engine._ig.attribute = lambda *a, **k: original2(*a, **k) + 0.1
        self.assertEqual(core.handle_explain(VALID), {"kind": "error", "code": w.CODE_INTERNAL})

    def test_gate_messages_are_pinned_to_the_committed_engine(self):
        # The classifier matches the engine's own gate messages by equality; if the engine text
        # changes, this fails loudly instead of silently degrading to "unclassified".
        source = (ROOT / "src" / "level8_explain.py").read_text()
        for message in w.ENGINE_GATE_MESSAGES:
            self.assertIn(f'ExplanationNumericalError("{message}")', source)

    def test_tampered_weights_and_unexpected_exceptions_are_internal_errors(self):
        core = self.core()
        with self.torch.no_grad():
            next(core.runtime.model.parameters()).add_(0.01)
        self.assertEqual(core.handle_explain(VALID)["code"], w.CODE_INTERNAL)
        core2 = self.core()

        def boom(payload):
            raise ValueError("unexpected")

        core2.engine.explain = boom
        self.assertEqual(core2.handle_explain(VALID)["code"], w.CODE_INTERNAL)


# ------------------------------------------------------------- real spawned-process tests
def _dispatch_module():
    from src import level8_dispatch

    return level8_dispatch


def _fd_count():
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


class RealProcessBase(unittest.TestCase):
    def make(self, entry=None, **cfg):
        d = _dispatch_module()
        if entry is None:
            disp = d.Dispatcher(str(ROOT), d.DispatcherConfig(**cfg), expected_artifacts=self.artifacts)
        else:
            disp = d.Dispatcher(
                "unused", d.DispatcherConfig(**cfg),
                handle_factory=lambda gen: d.ProcessWorkerHandle(gen, "unused", None, entry=entry))
        self.addCleanup(self._cleanup, disp)
        disp.start()
        return disp

    @staticmethod
    def _cleanup(disp):
        disp.shutdown()

    def ready(self, disp, generation=1, timeout=90):
        ok = wait_until(lambda: disp.diagnostics()["state"] == "ready"
                        and disp.diagnostics()["generation"] == generation, timeout)
        self.assertTrue(ok, disp.diagnostics())

    def run_job(self, disp, payload=VALID, timeout=30):
        return disp.submit(payload).future.result(timeout)

    def no_children(self):
        self.assertTrue(wait_until(lambda: multiprocessing.active_children() == [], 10),
                        multiprocessing.active_children())


class FaultInjectionProcessTests(RealProcessBase):
    """Controlled real children (no engine): IPC faults, timeouts, crashes, recovery."""

    def test_echo_worker_serves_serialized_requests(self):
        d = _dispatch_module()
        disp = self.make(entry_echo)
        self.ready(disp)
        outcomes = []

        def client():
            outcomes.append(self.run_job(disp))

        threads = [threading.Thread(target=client) for _ in range(5)]
        [t.start() for t in threads]
        [t.join(30) for t in threads]
        self.assertEqual([o.kind for o in outcomes], ["result"] * 5)
        self.assertEqual(disp.diagnostics()["invariant_violations"], 0)
        disp.shutdown()
        self.assertEqual(disp.diagnostics()["state"], d.STOPPED)
        self.no_children()

    def test_crash_during_request_recovers(self):
        d = _dispatch_module()
        disp = self.make(entry_crash_on_explain)
        self.ready(disp)
        outcome = self.run_job(disp)
        self.assertEqual(outcome.code, d.CODE_UNAVAILABLE)
        self.ready(disp, generation=2)
        self.assertEqual(disp.diagnostics()["restarts_total"], 1)

    def test_partial_frame_stall_recovers_by_deadline(self):
        d = _dispatch_module()
        disp = self.make(entry_partial_frame_then_stall)
        self.ready(disp)
        started = time.monotonic()
        outcome = self.run_job(disp)
        self.assertEqual(outcome.code, d.CODE_DEADLINE)
        self.assertLess(time.monotonic() - started, 2.0)  # bounded, not stuck on the half frame
        self.ready(disp, generation=2)

    def test_child_that_never_reads_recovers_by_deadline(self):
        d = _dispatch_module()
        disp = self.make(entry_never_reads)
        self.ready(disp)
        self.assertEqual(self.run_job(disp).code, d.CODE_DEADLINE)
        self.ready(disp, generation=2)

    def test_garbage_oversize_and_nan_frames_are_protocol_faults(self):
        d = _dispatch_module()
        for entry in (entry_garbage, entry_oversize_header, entry_nan_result):
            with self.subTest(entry=entry.__name__):
                disp = self.make(entry)
                self.ready(disp)
                self.assertEqual(self.run_job(disp).code, d.CODE_UNAVAILABLE)
                self.ready(disp, generation=2)
                disp.shutdown()

    def test_stale_generation_ignored_and_wrong_id_is_a_fault(self):
        d = _dispatch_module()
        disp = self.make(entry_stale_generation_then_result)
        self.ready(disp)
        self.assertEqual(self.run_job(disp).kind, "result")
        self.assertEqual(disp.diagnostics()["counters"]["stale_ignored"], 1)
        self.assertEqual(disp.diagnostics()["generation"], 1)  # no restart for the stale message
        disp2 = self.make(entry_wrong_request_id)
        self.ready(disp2)
        self.assertEqual(self.run_job(disp2).code, d.CODE_UNAVAILABLE)
        self.ready(disp2, generation=2)

    def test_child_exits_on_parent_eof(self):
        d = _dispatch_module()
        handle = d.ProcessWorkerHandle(1, "unused", None, entry=entry_exit_on_eof)
        handle.start()
        waker = d.Waker()
        self.addCleanup(waker.close)
        event = handle.wait(time.monotonic() + 30, waker)
        self.assertEqual(event[0], "message")
        proc = handle._proc
        handle._conns[0].close()  # parent closes its write end of the request pipe
        proc.join(10)
        self.assertEqual(proc.exitcode, 0)  # child saw EOF and exited by itself
        self.assertEqual(handle.wait(time.monotonic() + 5, waker)[0], "closed")  # EOF the other way
        handle.terminate()
        self.no_children()

    def test_real_startup_timeout(self):
        d = _dispatch_module()
        disp = d.Dispatcher("unused", d.DispatcherConfig(startup_deadline=1.0, max_restarts=0),
                            handle_factory=lambda gen: d.ProcessWorkerHandle(
                                gen, "unused", None, entry=entry_never_reads_not_ready))
        self.addCleanup(disp.shutdown)
        disp.start()
        self.assertTrue(wait_until(lambda: disp.diagnostics()["state"] == d.OPEN, 30), disp.diagnostics())
        self.assertEqual(disp.diagnostics()["counters"]["failure:startup_timeout"], 1)
        with self.assertRaises(d.AdmissionRejected):
            disp.submit(VALID)
        disp.shutdown()
        self.no_children()

    def _hold(self, entry, **cfg):
        d = _dispatch_module()
        disp = d.Dispatcher("unused", d.DispatcherConfig(**cfg), handle_factory=lambda gen: d.ProcessWorkerHandle(
            gen, "unused", None, entry=entry))
        self.addCleanup(disp.shutdown)
        disp.start()
        self.ready(disp)
        return d, disp

    def test_shutdown_with_real_active_work_respects_the_execution_deadline(self):
        d, disp = self._hold(entry_hold_after_request)
        job = disp.submit(VALID)
        self.assertTrue(wait_until(lambda: job.state == d.EXECUTING, 10))
        started = time.monotonic()
        disp.shutdown()
        elapsed = time.monotonic() - started
        self.assertEqual(job.future.result(0).code, d.CODE_DEADLINE)
        self.assertLess(elapsed, 1.5)  # bounded by the 250 ms execution deadline, not the 2 s grace
        self.assertEqual(disp.diagnostics()["state"], d.STOPPED)
        self.no_children()

    def test_shutdown_grace_shorter_than_the_execution_deadline_is_unavailable(self):
        d, disp = self._hold(entry_hold_after_request, shutdown_grace=0.05)
        job = disp.submit(VALID)
        self.assertTrue(wait_until(lambda: job.state == d.EXECUTING, 10))
        disp.shutdown()
        self.assertEqual(job.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertEqual(disp.diagnostics()["state"], d.STOPPED)
        self.no_children()

    def test_shutdown_lets_real_active_work_finish_within_its_deadline(self):
        d, disp = self._hold(entry_slow_echo)
        job = disp.submit(VALID)
        self.assertTrue(wait_until(lambda: job.state == d.EXECUTING, 10))
        queued = disp.submit(VALID)
        disp.shutdown()
        self.assertEqual(job.future.result(0).kind, "result")
        self.assertEqual(queued.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertEqual(disp.diagnostics()["state"], d.STOPPED)
        self.no_children()

    def test_sigkill_while_executing_invalidates_queued_jobs_across_recovery(self):
        d, disp = self._hold(entry_hold_after_request)
        active = disp.submit(VALID)
        queued = [disp.submit(VALID) for _ in range(3)]
        self.assertTrue(wait_until(lambda: active.state == d.EXECUTING, 10))
        pid = disp.worker_info()["pid"]
        os.kill(pid, signal.SIGKILL)
        for job in [active] + queued:
            self.assertEqual(job.future.result(30).code, d.CODE_UNAVAILABLE)  # unexpired: invalidated only
        self.ready(disp, generation=2)
        diag = disp.diagnostics()
        self.assertEqual((diag["restarts_total"], diag["active"], diag["queued"], diag["invariant_violations"]),
                         (1, 0, 0, 0))
        self.assertNotEqual(disp.worker_info()["pid"], pid)

    def test_dispatcher_level_blocked_send_is_bounded_and_recovers(self):
        import fcntl

        if not hasattr(fcntl, "F_SETPIPE_SZ"):
            self.skipTest("pipe resizing needs Linux")
        d = _dispatch_module()

        class FilledPipeHandle(d.ProcessWorkerHandle):
            """Test transport fault: the request pipe is shrunk to one page and filled, so the
            dispatcher's own send() genuinely blocks (the child never reads)."""

            def start(self):
                super().start()
                fd = self._conns[0].fileno()
                fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, 4096)
                try:
                    while True:
                        os.write(fd, b"x" * 4096)
                except BlockingIOError:
                    pass

        disp = d.Dispatcher("unused", handle_factory=lambda gen: FilledPipeHandle(
            gen, "unused", None, entry=entry_never_reads))
        self.addCleanup(disp.shutdown)
        disp.start()
        self.ready(disp)
        started = time.monotonic()
        outcome = self.run_job(disp)
        elapsed = time.monotonic() - started
        counters = disp.diagnostics()["counters"]
        self.assertEqual(outcome.code, d.CODE_DEADLINE)
        self.assertLess(elapsed, 1.5)
        self.assertEqual(counters["failure:send_timeout"], 1)  # the SEND blocked and hit its deadline...
        self.assertNotIn("failure:exec_timeout", counters)  # ...not the later wait for a reply
        self.ready(disp, generation=2)

    def test_start_stop_cycles_leak_nothing(self):
        d = _dispatch_module()
        before_fds, before_threads = _fd_count(), {t.name for t in threading.enumerate()}
        for _ in range(3):
            disp = d.Dispatcher("unused", handle_factory=lambda gen: d.ProcessWorkerHandle(
                gen, "unused", None, entry=entry_echo))
            disp.start()
            self.ready(disp)
            self.assertEqual(self.run_job(disp).kind, "result")
            disp.shutdown()
            self.assertEqual(disp.diagnostics()["state"], d.STOPPED)
        self.no_children()
        self.assertEqual({t.name for t in threading.enumerate()} - before_threads, set())
        if before_fds is not None:
            self.assertEqual(_fd_count(), before_fds)

    def test_shutdown_during_startup_terminates_the_child(self):
        d = _dispatch_module()

        def slow_boot(gen):
            return d.ProcessWorkerHandle(gen, "unused", None, entry=entry_never_reads_not_ready)

        disp = d.Dispatcher("unused", handle_factory=slow_boot)
        disp.start()
        self.assertTrue(wait_until(lambda: disp.diagnostics()["state"] == "starting", 30))
        disp.shutdown()
        self.assertEqual(disp.diagnostics()["state"], d.STOPPED)
        self.no_children()


def entry_never_reads_not_ready(req, resp, root, gen, expected=None):
    time.sleep(3600)  # never completes the readiness handshake


# --------------------------------------------------- real engine child (canonical process)
class RealEngineProcessTests(RealProcessBase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from src.level7_runtime import ModelRuntime
        from src.level8_explain import ExplanationEngine
        from tests import level8_oracle as oracle

        cls.torch, cls.np, cls.oracle = torch, np, oracle
        cls.artifacts = json.loads((ROOT / "serving-manifest.json").read_text())["artifacts"]
        cls.runtime = ModelRuntime(ROOT)
        grids = oracle.build_grids(cls.runtime, VALID)
        cls.payloads = [
            {n: (int(v) if n == "prior_injury" else float(v)) for n, v in zip(cls.runtime.feature_names, row)}
            for row in list(grids["dev_range"][:10]) + list(grids["straddle_threshold"][:2])
        ]
        saved = torch.get_num_threads()
        torch.set_num_threads(1)  # reference results under the child's thread setting (restored below)
        try:
            engine = ExplanationEngine(cls.runtime)
            cls.reference = [w._wire_form(engine.explain(p)) for p in cls.payloads]
        finally:
            torch.set_num_threads(saved)
        cls.p64 = oracle.params64(cls.runtime.model)

    def torch_globals(self):
        t = self.torch
        return (t.get_num_threads(), t.get_num_interop_threads(), t.are_deterministic_algorithms_enabled(),
                str(t.get_default_dtype()), t.is_grad_enabled())

    def test_readiness_parity_numerics_and_isolation(self):
        d = _dispatch_module()
        before = self.torch_globals()
        fds_before = _fd_count()
        disp = self.make()
        self.ready(disp)
        info = disp.worker_info()
        self.assertEqual((info["torch_threads"], info["torch_interop_threads"]), (1, 1))
        self.assertEqual(info["loaded_forbidden_modules"], [])
        self.assertNotEqual(info["pid"], os.getpid())
        meta = disp.metadata()
        self.assertEqual(meta["model"]["artifact_sha256"], self.artifacts)
        for name, digest in self.artifacts.items():  # independent verification of the bundle identity
            import hashlib

            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(self.torch_globals(), before)  # the parent's settings are untouched
        for payload, reference in zip(self.payloads, self.reference):
            outcome = self.run_job(disp, payload)
            self.assertEqual(outcome.kind, "result")
            self.assertEqual(outcome.explanation, reference)  # exact parity with the committed engine
            json.dumps(outcome.explanation, allow_nan=False)
        for payload, reference in list(zip(self.payloads, self.reference))[:5]:  # sampled oracle correctness
            z = self.np.array([a["standardized_value"] for a in reference["attributions"]])
            attrs = self.np.array([a["attribution"] for a in reference["attributions"]])
            oracle, _ = self.oracle.oracle_ig(self.p64, z, self.np.zeros(5))
            self.assertLessEqual(float(abs(attrs - oracle).max()), 0.005)
            self.assertLessEqual(abs(reference["completeness_residual"]), reference["allowed_residual"])
        out = disp.submit(VALID).future.result(30)
        self.assertEqual(out.explanation["prediction"], self.runtime.predict(VALID))  # authoritative source
        concurrent = [disp.submit(VALID) for _ in range(5)]
        self.assertTrue(all(j.future.result(30).kind == "result" for j in concurrent))
        diag = disp.diagnostics()
        self.assertEqual((diag["invariant_violations"], diag["restarts_total"], diag["generation"]), (0, 0, 1))
        disp.shutdown()
        self.assertEqual(diag["high_water"]["active"], 1)
        self.no_children()
        self.assertEqual(self.torch_globals(), before)
        if fds_before is not None:
            self.assertEqual(_fd_count(), fds_before)

    def test_request_errors_do_not_restart_the_worker(self):
        d = _dispatch_module()
        from src.level8_explain import InvalidExplanationInput, OutsideExplanationDomain

        disp = self.make()
        self.ready(disp)
        with self.assertRaises(OutsideExplanationDomain):
            disp.submit(dict(VALID, weekly_km_to_date=19.0))
        with self.assertRaises(InvalidExplanationInput):
            disp.submit(dict(VALID, prior_injury=0.5))
        self.assertEqual(self.run_job(disp).kind, "result")
        self.assertEqual(disp.diagnostics()["generation"], 1)

    def test_kill_nine_is_detected_and_replaced(self):
        disp = self.make()
        self.ready(disp)
        first_pid = disp.worker_info()["pid"]
        os.kill(first_pid, signal.SIGKILL)
        self.ready(disp, generation=2)
        second_pid = disp.worker_info()["pid"]
        self.assertNotEqual(first_pid, second_pid)
        with self.assertRaises(ProcessLookupError):  # old process was reaped, not left as a zombie
            os.kill(first_pid, 0)
        self.assertEqual(self.run_job(disp).kind, "result")
        self.assertEqual(disp.diagnostics()["restarts_total"], 1)

    def test_execution_timeout_recovers_with_the_real_engine(self):
        d = _dispatch_module()
        disp = self.make(exec_timeout=0.001)  # far below the ~15 ms an explanation takes
        self.ready(disp)
        self.assertEqual(self.run_job(disp).code, d.CODE_DEADLINE)
        self.ready(disp, generation=2)
        self.assertEqual(disp.diagnostics()["restarts_total"], 1)

    def test_health_probe_timeout_cannot_extend_the_execution_deadline(self):
        d = _dispatch_module()
        marker = Path(tempfile.mkdtemp()) / "probe"
        os.environ["LEVEL8_TEST_PROBE_MARKER"] = str(marker)
        self.addCleanup(os.environ.pop, "LEVEL8_TEST_PROBE_MARKER", None)
        disp = d.Dispatcher(str(ROOT), expected_artifacts=self.artifacts, handle_factory=lambda gen: d.ProcessWorkerHandle(
            gen, str(ROOT), self.artifacts, entry=entry_probe_stall))
        self.addCleanup(disp.shutdown)
        disp.start()
        self.ready(disp)
        started = time.monotonic()
        job = disp.submit(VALID)
        self.assertTrue(wait_until(marker.exists, 10))  # the child is now inside the (hanging) probe
        self.assertFalse(job.future.done())
        outcome = job.future.result(10)
        elapsed = time.monotonic() - started
        self.assertEqual(outcome.code, d.CODE_DEADLINE)
        self.assertEqual(outcome.kind, "error")
        self.assertIsNone(outcome.explanation)  # no partial explanation
        self.assertLess(elapsed, 0.25 + 1.0)  # execution deadline plus scheduling slack, never the probe's hang
        self.assertEqual(disp.diagnostics()["counters"]["failure:exec_timeout"], 1)
        self.ready(disp, generation=2)

    def test_wrong_serving_root_and_wrong_artifacts_fail_closed(self):
        d = _dispatch_module()
        with tempfile.TemporaryDirectory() as empty:
            disp = d.Dispatcher(empty, d.DispatcherConfig(max_restarts=0), expected_artifacts=self.artifacts)
            self.addCleanup(disp.shutdown)
            disp.start()
            self.assertTrue(wait_until(lambda: disp.diagnostics()["state"] == "open", 90), disp.diagnostics())
            self.assertEqual(disp.diagnostics()["counters"]["failure:startup_fatal"], 1)
            with self.assertRaises(d.AdmissionRejected):
                disp.submit(VALID)
        wrong = {k: "0" * 64 for k in self.artifacts}
        disp2 = d.Dispatcher(str(ROOT), d.DispatcherConfig(max_restarts=0), expected_artifacts=wrong)
        self.addCleanup(disp2.shutdown)
        disp2.start()
        self.assertTrue(wait_until(lambda: disp2.diagnostics()["state"] == "open", 90), disp2.diagnostics())

    def test_repeated_real_start_and_shutdown_is_clean(self):
        d = _dispatch_module()
        fds_before = _fd_count()
        for _ in range(2):
            disp = d.Dispatcher(str(ROOT), expected_artifacts=self.artifacts)
            disp.start()
            self.ready(disp)
            self.assertEqual(self.run_job(disp).kind, "result")
            disp.shutdown()
            self.assertEqual(disp.diagnostics()["state"], d.STOPPED)
        self.no_children()
        self.assertFalse([t for t in threading.enumerate() if t.name == "level8-dispatcher"])
        if fds_before is not None:
            self.assertEqual(_fd_count(), fds_before)


if __name__ == "__main__":
    unittest.main()
