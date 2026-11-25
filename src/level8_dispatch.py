"""Level 8 bounded dispatcher for the dedicated explanation worker process.

Parent-side library (no HTTP). One dispatcher thread owns the worker handle (process
start, all IPC, termination and joining); callers only call ``submit``/``cancel``/
``shutdown`` and never block on IPC or process management.

Admission invariants, all guarded by ONE lock and checked after every transition:
    active (reserved or executing) <= 1,  waiting <= max_waiting (8),  admitted <= 9.

Time is read from an injectable monotonic clock; every job transition is idempotent
(compare-and-set on the job state under the lock) so a future is resolved exactly once
and capacity is released exactly once.

Provisional configuration (see DispatcherConfig); these are implementation settings, not
verified production guarantees.
"""

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import multiprocessing
import os
import select
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

try:
    from src import level8_worker as worker_module
    from src.level8_explain import ExplanationEngine
except ModuleNotFoundError as _exc:  # flat (image) layout
    if _exc.name != "src":
        raise
    import level8_worker as worker_module  # type: ignore[no-redef]
    from level8_explain import ExplanationEngine  # type: ignore[no-redef]

# worker / breaker states
INIT, STARTING, READY, RESTARTING, OPEN, STOPPING, STOPPED = (
    "init", "starting", "ready", "restarting", "open", "stopping", "stopped")
# Termination could not be confirmed: the live process is retained (never replaced or forgotten).
CLEANUP_FAILED, STOPPED_UNCONFIRMED = "cleanup_failed", "stopped_unconfirmed"
TERMINAL_STATES = (STOPPED, STOPPED_UNCONFIRMED)
TERMINATE_ATTEMPTS = 3
# job states
QUEUED, RESERVED, EXECUTING, DONE = "queued", "reserved", "executing", "done"

CODE_OVERLOADED = "explanation_overloaded"
CODE_UNAVAILABLE = "explanation_unavailable"
CODE_NUMERICAL = worker_module.CODE_NUMERICAL
CODE_DEADLINE = "explanation_deadline_exceeded"
CODE_OUTSIDE_DOMAIN = worker_module.CODE_OUTSIDE_DOMAIN
CODE_INVALID_INPUT = worker_module.CODE_INVALID_INPUT


@dataclass(frozen=True)
class DispatcherConfig:
    max_waiting: int = 8
    request_deadline: float = 0.500
    exec_timeout: float = 0.250
    min_dispatch_margin: float = 0.050
    startup_deadline: float = 20.0
    shutdown_grace: float = 2.0
    max_restarts: int = 3
    restart_window: float = 60.0
    breaker_cooldown: float = 60.0


class AdmissionRejected(Exception):
    """submit() refused the request. ``code`` is overloaded or unavailable."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class WorkerIOError(Exception):
    """Handle-level failure; kind is 'timeout', 'closed' or 'protocol'."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


@dataclass(frozen=True)
class Outcome:
    kind: str  # 'result' | 'error'
    explanation: dict[str, Any] | None = None
    code: str | None = None
    violations: tuple[dict[str, Any], ...] = ()
    # Successes only: the generation of the handle that returned this result, recorded after the
    # message's generation and request id were verified. Never read from the current generation.
    generation: int | None = None


def _error(code: str, violations: tuple = ()) -> Outcome:
    return Outcome("error", None, code, tuple(violations))


# ------------------------------------------------------------------ clock and waker
class RealClock:
    now = staticmethod(time.monotonic)

    def sleep(self, timeout: float | None, waker: "Waker") -> None:
        select.select([waker.fileno()], [], [], timeout)


class Waker:
    """Bounded wake-up: at most one pending byte (coalesced), so it can never grow.
    After close() every operation is a no-op (descriptor numbers may be reused elsewhere)."""

    def __init__(self) -> None:
        self._r, self._w = os.pipe()
        os.set_blocking(self._r, False)
        os.set_blocking(self._w, False)
        self._lock = threading.Lock()
        self._set = False
        self._closed = False

    def set(self) -> None:
        with self._lock:
            if self._closed or self._set:
                return
            try:
                os.write(self._w, b"x")
            except BlockingIOError:
                pass
            self._set = True

    def clear(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                while os.read(self._r, 64):
                    pass
            except BlockingIOError:
                pass
            self._set = False

    def is_set(self) -> bool:
        return self._set

    def fileno(self) -> int:
        return self._r

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for fd in (self._r, self._w):
                try:
                    os.close(fd)
                except OSError:
                    pass


# ------------------------------------------------------------------ process handle
class ProcessWorkerHandle:
    """One spawned worker process and its two one-way pipes. Used only by the dispatcher
    thread. ``entry`` is injectable so tests can use controlled fault-injecting children."""

    def __init__(self, generation: int, serving_root: str, expected_artifacts: Mapping[str, str] | None,
                 *, entry: Callable | None = None, clock: Any = None):
        self.generation = generation
        self._root = str(serving_root)
        self._expected = dict(expected_artifacts) if expected_artifacts is not None else None
        self._entry = entry or worker_module.worker_main
        self._clock = clock or RealClock()
        self._ctx = multiprocessing.get_context("spawn")
        self._proc = None
        self._channel = None
        self._conns: list[Any] = []

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def start(self) -> None:
        req_r, req_w = self._ctx.Pipe(duplex=False)
        resp_r, resp_w = self._ctx.Pipe(duplex=False)
        self._conns = [req_r, req_w, resp_r, resp_w]
        try:
            self._proc = self._ctx.Process(
                target=self._entry, args=(req_r, resp_w, self._root, self.generation, self._expected),
                name=f"level8-explain-worker-g{self.generation}", daemon=True)
            self._proc.start()
        except BaseException:
            self._close_conns()
            raise
        req_r.close()  # child's ends: close ours so EOF propagates in both directions
        resp_w.close()
        self._conns = [req_w, resp_r]
        self._channel = worker_module.FdChannel(resp_r.fileno(), req_w.fileno())

    def send(self, message: Mapping[str, Any], deadline: float) -> None:
        try:
            data = worker_module.encode_message(message, "to_child")
            self._channel.send_frame(data, deadline, now=self._clock.now)
        except worker_module.ChannelTimeout:
            raise WorkerIOError("timeout") from None
        except worker_module.ChannelClosed:
            raise WorkerIOError("closed") from None
        except worker_module.ProtocolError:
            raise WorkerIOError("protocol") from None

    def wait(self, deadline: float | None, waker: Waker):
        sentinel = self._proc.sentinel
        try:
            event = self._channel.poll(deadline, extra_fds=(sentinel, waker.fileno()), now=self._clock.now)
            if event[0] == "fd" and event[1] == sentinel:  # exited: drain any final frame first
                event = self._channel.poll(self._clock.now(), extra_fds=())
                if event[0] in ("timeout", "closed"):
                    return ("closed", "exited")
        except worker_module.ProtocolError:
            return ("fault", "frame")
        kind = event[0]
        if kind == "frame":
            try:
                return ("message", worker_module.decode_message(event[1], "to_parent"))
            except worker_module.ProtocolError:
                return ("fault", "malformed")
        if kind == "timeout":
            return ("timeout",)
        if kind == "closed":
            return ("closed", "eof")
        return ("wake",)

    def terminate(self) -> bool:
        """Bounded terminate/kill/join. Returns True only if the process is confirmed gone (or never
        existed). If it is still alive the handle KEEPS it so the caller can retry; it is never
        discarded."""
        proc = self._proc
        confirmed = True
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(0.5)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(1.0)
                else:
                    proc.join(0.1)
                if proc.is_alive():
                    confirmed = False
                else:
                    proc.close()
                    self._proc = None
            except ValueError:
                confirmed = False
        self._close_conns()
        return confirmed

    def graceful_stop(self, deadline: float) -> bool:
        proc = self._proc
        if proc is not None and proc.is_alive():
            try:
                self.send({"t": "shutdown", "gen": self.generation}, min(deadline, self._clock.now() + 0.2))
                proc.join(max(0.0, deadline - self._clock.now()))
            except Exception:
                pass
        return self.terminate()

    def _close_conns(self) -> None:
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        self._conns = []
        self._channel = None


# ------------------------------------------------------------------------ dispatcher
@dataclass(eq=False)
class Job:
    payload: dict[str, Any]
    admitted_at: float
    deadline: float
    future: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)
    state: str = QUEUED
    abandoned: bool = False
    req_id: int = 0
    exec_deadline: float = 0.0


class Dispatcher:
    """Bounded dispatcher. Lifecycle responsibility: the owner MUST call ``shutdown()`` explicitly
    (there is deliberately no destructor-based cleanup). Execution is mutually exclusive:
    ``start()`` selects background mode (only the dispatcher thread may step), ``pump()`` selects
    manual mode (only the first caller, the owner, may step). The two modes cannot be mixed and
    ownership is never transferred."""

    def __init__(self, serving_root: str, config: DispatcherConfig | None = None, *,
                 expected_artifacts: Mapping[str, str] | None = None,
                 handle_factory: Callable[[int], Any] | None = None, clock: Any = None):
        self._cfg = config or DispatcherConfig()
        self._clock = clock or RealClock()
        self._root = str(serving_root)
        self._expected = dict(expected_artifacts) if expected_artifacts is not None else None
        self._factory = handle_factory or (
            lambda gen: ProcessWorkerHandle(gen, self._root, self._expected, clock=self._clock))
        self._lock = threading.Lock()
        self._waker = Waker()
        self._thread: threading.Thread | None = None
        self._blocking = False
        self._mode: str | None = None  # None | "manual" | "background"; set once, under the lock
        self._owner_ident: int | None = None
        # state guarded by self._lock
        self._state = INIT
        self._generation = 0
        self._queue: collections.deque[Job] = collections.deque()
        self._active: Job | None = None
        self._ids = itertools.count(1)
        self._restart_times: collections.deque[float] = collections.deque()
        self._half_open = False
        self._cooldown_until = 0.0
        self._shutdown_deadline = 0.0
        self._counters: collections.Counter = collections.Counter()
        self._hw = {"active": 0, "queued": 0, "admitted": 0}
        self._worker_info: dict[str, Any] | None = None
        self._metadata: dict[str, Any] | None = None
        self._startup_deadline = 0.0
        self._ready = False
        # dispatcher-thread-only state
        self._handle: Any = None
        self._shutdown_aborted = False

    # ------------------------------------------------------------------ public API
    def start(self) -> None:
        """Select background mode and start the dispatcher thread (idempotent). The worker process
        starts on that thread. Rejected once manual ownership exists."""
        with self._lock:
            if self._mode == "manual":
                raise RuntimeError("manual pump ownership already established")
            if self._thread is not None or self._state in (STOPPING, *TERMINAL_STATES):
                return
            self._mode = "background"
            self._blocking = True
            self._thread = threading.Thread(target=self._run, name="level8-dispatcher", daemon=True)
        self._thread.start()

    def submit(self, payload: Mapping[str, Any]) -> Job:
        """Validate, then atomically admit. Raises InvalidExplanationInput / OutsideExplanationDomain
        (no capacity used) or AdmissionRejected (overloaded / unavailable)."""
        clean = ExplanationEngine.validate_payload(payload)
        now = self._clock.now()
        with self._lock:
            if self._state != READY:
                self._counters["rejected_unavailable"] += 1
                raise AdmissionRejected(CODE_UNAVAILABLE)
            job = Job(clean, now, now + self._cfg.request_deadline)
            if self._active is None and not self._queue:
                job.state = RESERVED
                self._active = job
            elif len(self._queue) < self._cfg.max_waiting:
                self._queue.append(job)
            else:
                self._counters["rejected_overloaded"] += 1
                raise AdmissionRejected(CODE_OVERLOADED)
            self._counters["admitted"] += 1
            self._note_high_water()
        self._waker.set()
        return job

    def submit_comparison(self, original: Mapping[str, Any], modified: Mapping[str, Any]) -> tuple[Job, ...]:
        """Admit the evaluations of one comparison, all or none. Both inputs are validated first. Inputs
        whose validated values are exactly equal need one evaluation; otherwise two jobs are returned
        (original, modified). They share one admission time and deadline, are adjacent in FIFO order, and
        each counts as one evaluation against the capacity limits. Raises like ``submit``.

        Accounting: ``admitted`` stays per evaluation. A refused comparison counts once in
        ``comparison_requests_rejected_*`` and never in the single-evaluation ``rejected_*`` counters.

        Because a worker failure drains every queued job in the same step that ends the active one, two
        successes of one comparison cannot span a replacement; each success also records its generation."""
        first = ExplanationEngine.validate_payload(original)
        second = ExplanationEngine.validate_payload(modified)
        payloads = (first,) if first == second else (first, second)
        now = self._clock.now()
        with self._lock:
            if self._state != READY:
                self._counters["comparison_requests_rejected_unavailable"] += 1
                raise AdmissionRejected(CODE_UNAVAILABLE)
            take_active = self._active is None and not self._queue
            queued_after = len(self._queue) + len(payloads) - (1 if take_active else 0)
            admitted_after = queued_after + (1 if self._active is not None or take_active else 0)
            if queued_after > self._cfg.max_waiting or admitted_after > self._cfg.max_waiting + 1:
                self._counters["comparison_requests_rejected_overloaded"] += 1
                raise AdmissionRejected(CODE_OVERLOADED)
            # everything that can fail happens before any state changes
            jobs = tuple(Job(payload, now, now + self._cfg.request_deadline, req_id=next(self._ids))
                         for payload in payloads)
            waiting = jobs[1:] if take_active else jobs
            queued_before = len(self._queue)
            try:
                self._queue.extend(waiting)
            except BaseException:
                while len(self._queue) > queued_before:
                    self._queue.pop()
                raise
            if take_active:
                jobs[0].state = RESERVED
                self._active = jobs[0]
            self._counters["admitted"] += len(jobs)
            self._counters["comparison_requests_admitted"] += 1
            self._counters["comparison_evaluations_admitted"] += len(jobs)
            self._note_high_water()
        self._waker.set()
        return jobs

    def cancel(self, job: Job) -> bool:
        """Client cancellation. Queued: released immediately. Active: slot stays reserved until
        the worker finishes or is terminated (the result is discarded)."""
        with self._lock:
            if job.state == QUEUED:
                try:
                    self._queue.remove(job)
                except ValueError:
                    pass
                job.state = DONE
                job.abandoned = True
                self._counters["cancelled_queued"] += 1
            elif job.state in (RESERVED, EXECUTING) and not job.abandoned:
                job.abandoned = True
                self._counters["cancelled_active"] += 1
            else:
                return False
            self._check_invariants()
        job.future.cancel()
        self._waker.set()
        return True

    def shutdown(self, timeout: float | None = None) -> None:
        """Idempotent. Stops admission, suppresses restarts, fails queued work, gives active work
        its own deadline (never extended by the grace budget), then terminates and joins the worker
        and the dispatcher thread. In manual mode only the owner drives it to completion."""
        now = self._clock.now()
        with self._lock:
            first = self._state not in (STOPPING, *TERMINAL_STATES)
            if first:
                self._state = STOPPING
                self._shutdown_deadline = now + self._cfg.shutdown_grace
                queued = self._drain_queue_locked(now)
            else:
                queued = []
        for job, code in queued:
            self._resolve(job, _error(code))
        self._waker.set()
        thread = self._thread
        if thread is not None:
            if thread is not threading.current_thread():
                thread.join(timeout if timeout is not None else self._cfg.shutdown_grace + 10.0)
        else:
            with self._lock:
                if self._mode is None:
                    self._mode, self._owner_ident = "manual", threading.get_ident()
                mine = self._mode == "manual" and self._owner_ident == threading.get_ident()
            if mine:
                self._run_steps()
        with self._lock:
            done = self._state in TERMINAL_STATES
        if done:
            self._waker.close()

    def diagnostics(self) -> dict[str, Any]:
        """Operational snapshot only: no submitted values, no explanations."""
        with self._lock:
            now = self._clock.now()
            window = [t for t in self._restart_times if t > now - self._cfg.restart_window]
            return {
                "state": self._state,
                "generation": self._generation,
                "active": 1 if self._active is not None else 0,
                "queued": len(self._queue),
                "high_water": dict(self._hw),
                "restarts_total": self._counters["restarts"],
                "restarts_in_window": len(window),
                "breaker_open": self._state == OPEN,
                "cleanup_failed": self._state in (CLEANUP_FAILED, STOPPED_UNCONFIRMED),
                "counters": dict(self._counters),
                "invariant_violations": self._counters["invariant_violations"],
            }

    def worker_info(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._worker_info) if self._worker_info else None

    def metadata(self) -> dict[str, Any] | None:
        with self._lock:
            return self._metadata

    def pump(self, max_steps: int = 100_000) -> int:
        """Manual mode (tests / embedding): run steps on the calling thread until no progress.
        The first caller becomes the exclusive owner. Rejected, before any handle or IPC access, in
        background mode, from any other thread, and after the dispatcher has stopped."""
        with self._lock:
            if self._state in TERMINAL_STATES:
                raise RuntimeError("dispatcher is closed")
            if self._mode == "background":
                raise RuntimeError("background mode: only the dispatcher thread may step")
            ident = threading.get_ident()
            if self._mode is None:
                self._mode, self._owner_ident = "manual", ident
            elif self._owner_ident != ident:
                raise RuntimeError("pump() is restricted to the manual owner")
        return self._run_steps(max_steps)

    def _run_steps(self, max_steps: int = 100_000) -> int:
        steps = 0
        while steps < max_steps and self._step():
            steps += 1
        return steps

    # --------------------------------------------------------------- bookkeeping
    def _note_high_water(self) -> None:
        self._hw["active"] = max(self._hw["active"], 1 if self._active is not None else 0)
        self._hw["queued"] = max(self._hw["queued"], len(self._queue))
        total = len(self._queue) + (1 if self._active is not None else 0)
        self._hw["admitted"] = max(self._hw["admitted"], total)
        self._check_invariants()

    def _check_invariants(self) -> None:
        active = 1 if self._active is not None else 0
        if len(self._queue) > self._cfg.max_waiting or len(self._queue) + active > self._cfg.max_waiting + 1:
            self._counters["invariant_violations"] += 1

    @staticmethod
    def _unfinished_code(job: Job, now: float) -> str:
        """Why an unfinished job is being failed: deadline-exceeded if its own absolute deadline
        (execution deadline once dispatched, request deadline otherwise) has been reached,
        otherwise unavailable. Equality counts as expired. Never extends a deadline."""
        limit = job.exec_deadline if job.state == EXECUTING else job.deadline
        return CODE_DEADLINE if now >= limit else CODE_UNAVAILABLE

    def _drain_queue_locked(self, now: float) -> list[tuple[Job, str]]:
        jobs = [(job, self._unfinished_code(job, now)) for job in self._queue]
        self._queue.clear()
        for job, _ in jobs:
            job.state = DONE
        return jobs

    def _bump(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    def _resolve(self, job: Job, outcome: Outcome) -> None:
        self._counters["resolved"] += 1
        try:
            job.future.set_result(outcome)
        except concurrent.futures.InvalidStateError:
            pass  # cancelled by the client; the outcome is discarded

    def _finish(self, job: Job, outcome: Outcome, counter: str) -> bool:
        """Idempotent terminal transition: releases capacity once, resolves the future once."""
        with self._lock:
            if job.state == DONE:
                return False
            if job.state == QUEUED:
                try:
                    self._queue.remove(job)
                except ValueError:
                    pass
            job.state = DONE
            if self._active is job:
                self._active = None
            self._counters[counter] += 1
            self._check_invariants()
        self._resolve(job, outcome)
        return True

    # ------------------------------------------------------------------ the step
    def _run(self) -> None:
        with self._lock:
            self._owner_ident = threading.get_ident()
        try:
            while True:
                self._step()
                with self._lock:
                    if self._state in TERMINAL_STATES:
                        return
        except BaseException:  # fail closed: never leave callers waiting
            self._emergency_stop()

    def _emergency_stop(self) -> None:
        now = self._clock.now()
        with self._lock:
            self._state = STOPPING
            pending = self._drain_queue_locked(now)
            if self._active is not None:
                pending.append((self._active, CODE_UNAVAILABLE))
        for job, code in pending:
            self._finish(job, _error(code), "emergency_failed")
        handle, self._handle = self._handle, None
        confirmed = True
        if handle is not None:
            try:
                confirmed = self._terminate_bounded(handle)
            except Exception:
                confirmed = False
            if not confirmed:
                self._handle = handle
        with self._lock:
            self._state = STOPPED if confirmed else STOPPED_UNCONFIRMED

    def _terminate_bounded(self, handle: Any) -> bool:
        for _ in range(TERMINATE_ATTEMPTS):
            if handle.terminate():
                return True
        with self._lock:
            self._counters["cleanup_failed"] += 1
        return False

    def _step(self) -> bool:
        if threading.get_ident() != self._owner_ident:
            raise RuntimeError("dispatcher step attempted outside its owning thread")
        with self._lock:
            state = self._state
        if state == STOPPING:
            return self._do_shutdown()
        if state == INIT:
            self._spawn(replacement=False)
            return True
        if state == STARTING:
            return self._step_starting()
        if state == READY:
            return self._step_ready()
        if state == RESTARTING:
            return self._step_restarting()
        if state == OPEN:
            return self._step_open()
        if state == CLEANUP_FAILED:
            if not self._blocking:
                return False
            self._clock.sleep(None, self._waker)  # nothing may run; wait for shutdown
            self._waker.clear()
            return True
        return False

    # ----------------------------------------------------------------- lifecycle
    def _spawn(self, replacement: bool) -> None:
        now = self._clock.now()
        with self._lock:
            if self._state in (STOPPING, *TERMINAL_STATES):
                return
            generation = self._generation + 1
            if replacement:
                self._restart_times.append(now)
                self._counters["restarts"] += 1
        try:
            handle = self._factory(generation)
            self._handle = handle
            handle.start()
        except Exception:
            with self._lock:
                self._generation = generation
            self._fail_worker("spawn_failed", CODE_UNAVAILABLE)
            return
        with self._lock:
            if self._state in (STOPPING, *TERMINAL_STATES):
                return
            self._generation = generation
            self._state = STARTING
            self._ready = False
            self._startup_deadline = now + self._cfg.startup_deadline

    def _fail_worker(self, reason: str, active_code: str | None = None) -> None:
        """Restart-worthy failure: stop admission, fail active and queued jobs (deadline-exceeded only
        where that job's own absolute deadline has been reached), schedule replacement."""
        now = self._clock.now()
        with self._lock:
            if self._state in (STOPPING, *TERMINAL_STATES):
                return
            self._state = RESTARTING
            self._ready = False
            queued = self._drain_queue_locked(now)
            active = self._active
            active_out = CODE_UNAVAILABLE
            if active is not None:
                hit = active_code == CODE_DEADLINE or self._unfinished_code(active, now) == CODE_DEADLINE
                active_out = CODE_DEADLINE if hit else CODE_UNAVAILABLE
            self._counters["failure:" + reason] += 1
        for job, code in queued:
            self._resolve(job, _error(code))
        if active is not None:
            self._finish(active, _error(active_out), "failed_active")
        self._waker.set()

    def _step_restarting(self) -> bool:
        handle, self._handle = self._handle, None
        if handle is not None and not self._terminate_bounded(handle):
            self._handle = handle  # still alive: keep ownership, never start a replacement
            with self._lock:
                if self._state == RESTARTING:
                    self._state = CLEANUP_FAILED
            return True
        now = self._clock.now()
        with self._lock:
            if self._state != RESTARTING:
                return True
            while self._restart_times and self._restart_times[0] <= now - self._cfg.restart_window:
                self._restart_times.popleft()
            exhausted = len(self._restart_times) >= self._cfg.max_restarts
            reopen = self._half_open
            if exhausted or reopen:
                self._state = OPEN
                self._half_open = False
                self._cooldown_until = now + self._cfg.breaker_cooldown
                self._counters["breaker_opened"] += 1
                return True
        self._spawn(replacement=True)
        return True

    def _step_open(self) -> bool:
        now = self._clock.now()
        with self._lock:
            until = self._cooldown_until
        if now >= until:
            with self._lock:
                self._half_open = True
            self._spawn(replacement=True)
            return True
        if not self._blocking:
            return False
        self._clock.sleep(until - now, self._waker)
        self._waker.clear()
        return True

    def _validate_ready(self, message: Mapping[str, Any], handle: Any) -> bool:
        if message["gen"] != handle.generation or message["torch_threads"] != 1 \
                or message["torch_interop_threads"] != 1 or message["loaded_forbidden_modules"]:
            return False
        model = message["metadata"].get("model") if isinstance(message["metadata"], dict) else None
        if not isinstance(model, dict) or model.get("integrity_status") != "passed":
            return False
        if self._expected is not None and model.get("artifact_sha256") != self._expected:
            return False
        return True

    def _step_starting(self) -> bool:
        handle = self._handle
        event = handle.wait(self._startup_deadline, self._waker)
        kind = event[0]
        if kind == "pending":
            return False
        if kind == "wake":
            self._waker.clear()
            return True
        if kind == "timeout":
            if self._clock.now() < self._startup_deadline:
                return True
            self._fail_worker("startup_timeout", CODE_UNAVAILABLE)
        elif kind in ("closed", "fault"):
            self._fail_worker("startup_" + event[1], CODE_UNAVAILABLE)
        elif event[1]["t"] == "ready" and self._validate_ready(event[1], handle):
            msg = event[1]
            with self._lock:
                if self._state != STARTING:
                    return True
                self._state = READY
                self._ready = True
                self._half_open = False
                self._metadata = msg["metadata"]
                self._worker_info = {
                    "pid": msg["pid"], "generation": handle.generation,
                    "torch_threads": msg["torch_threads"], "torch_interop_threads": msg["torch_interop_threads"],
                    "loaded_forbidden_modules": list(msg["loaded_forbidden_modules"]),
                    "self_test_ms": msg["self_test_ms"],
                }
        else:
            self._fail_worker("startup_fatal" if event[1]["t"] == "fatal" else "startup_protocol", CODE_UNAVAILABLE)
        return True

    # ---------------------------------------------------------------- READY state
    def _promote_next(self) -> Job | None:
        """Atomically move the head of the queue into the active slot (never over-fills)."""
        expired: list[Job] = []
        now = self._clock.now()
        promoted = None
        with self._lock:
            if self._active is None:
                while self._queue:
                    job = self._queue.popleft()
                    if job.state != QUEUED:
                        continue
                    if now >= job.deadline:  # expired while waiting: never reaches the child
                        job.state = DONE
                        expired.append(job)
                        continue
                    job.state = RESERVED
                    self._active = job
                    promoted = job
                    break
            self._note_high_water()
            for job in expired:
                self._counters["expired_queued"] += 1
        for job in expired:
            self._resolve(job, _error(CODE_DEADLINE))
        return promoted

    def _step_ready(self) -> bool:
        handle = self._handle
        with self._lock:
            job = self._active
        if job is None:
            job = self._promote_next()
            if job is None:
                event = handle.wait(None, self._waker)
                kind = event[0]
                if kind == "pending":
                    return False
                if kind == "wake":
                    self._waker.clear()
                elif kind in ("closed", "fault"):
                    self._fail_worker("idle_" + event[1], CODE_UNAVAILABLE)
                elif kind == "message":
                    self._fail_worker("idle_unexpected_message", CODE_UNAVAILABLE)
                return True
        if job.state == RESERVED:
            return self._dispatch(job, handle)
        return self._await_active(job, handle)

    def _dispatch(self, job: Job, handle: Any) -> bool:
        now = self._clock.now()
        if job.abandoned:  # never sent: nothing runs in the worker, so releasing is safe
            self._finish(job, _error(CODE_UNAVAILABLE), "abandoned_before_dispatch")
            return True
        if job.deadline - now < self._cfg.min_dispatch_margin:
            self._finish(job, _error(CODE_DEADLINE), "expired_before_dispatch")
            return True
        exec_deadline = min(job.deadline, now + self._cfg.exec_timeout)
        with self._lock:
            if not job.req_id:  # comparison jobs received theirs at admission
                job.req_id = next(self._ids)
            job.exec_deadline = exec_deadline
            job.state = EXECUTING
        message = {"t": "explain", "gen": handle.generation, "id": job.req_id, "payload": job.payload}
        try:
            handle.send(message, exec_deadline)
        except WorkerIOError as exc:
            self._fail_worker("send_" + exc.kind, CODE_DEADLINE if exc.kind == "timeout" else CODE_UNAVAILABLE)
        return True

    def _await_active(self, job: Job, handle: Any) -> bool:
        event = handle.wait(job.exec_deadline, self._waker)
        kind = event[0]
        if kind == "pending":
            return False
        if kind == "wake":
            self._waker.clear()
            return True
        if kind == "timeout":
            if self._clock.now() >= job.exec_deadline:
                self._fail_worker("exec_timeout", CODE_DEADLINE)
            return True
        if kind in ("closed", "fault"):
            self._fail_worker("active_" + event[1], CODE_UNAVAILABLE)
            return True
        self._on_active_message(job, handle, event[1])
        return True

    def _on_active_message(self, job: Job, handle: Any, message: Mapping[str, Any]) -> bool:
        """Returns True when the job completed. Faults trigger worker replacement."""
        if message["gen"] != handle.generation:
            with self._lock:
                self._counters["stale_ignored"] += 1
            return False
        kind = message["t"]
        if kind not in ("result", "error") or message.get("id") != job.req_id:
            self._fail_worker("protocol_mismatch", CODE_UNAVAILABLE)
            return False
        if kind == "result":
            self._finish(job, Outcome("result", explanation=message["explanation"],
                                      generation=handle.generation), "completed")
            return True
        code = message["code"]
        if code == worker_module.CODE_INTERNAL:
            self._fail_worker("worker_internal_error", CODE_UNAVAILABLE)
            return False
        if code == CODE_NUMERICAL:
            self._bump("numerical_subtype:" + message["subtype"])
        self._finish(job, _error(code, tuple(message.get("violations", ()))), "completed_error:" + code)
        return True

    # ------------------------------------------------------------------- shutdown
    def _do_shutdown(self) -> bool:
        handle = self._handle
        with self._lock:
            job = self._active
            ready = self._ready
        if job is not None and job.state == EXECUTING and handle is not None and ready \
                and not self._shutdown_aborted:
            # the grace budget can only shorten the wait, never extend the execution deadline
            limit = min(job.exec_deadline, self._shutdown_deadline)
            event = handle.wait(limit, self._waker)
            kind = event[0]
            if kind == "pending":
                return False
            if kind == "wake":
                self._waker.clear()
                return True
            if kind == "message":
                self._on_active_message_shutdown(job, handle, event[1])
                return True
            if kind == "timeout" and self._clock.now() < limit:
                return True
            self._shutdown_aborted = True
            return True
        now = self._clock.now()
        with self._lock:
            leftover = [(self._active, self._unfinished_code(self._active, now))] if self._active is not None else []
            leftover += self._drain_queue_locked(now)
        for item, code in leftover:
            self._finish(item, _error(code), "shutdown_failed")
        confirmed = True
        if handle is not None:
            if ready and not self._shutdown_aborted:
                confirmed = handle.graceful_stop(self._shutdown_deadline)
            else:
                confirmed = handle.terminate()
            if not confirmed:
                confirmed = self._terminate_bounded(handle)
        self._handle = None if confirmed else handle
        with self._lock:
            self._state = STOPPED if confirmed else STOPPED_UNCONFIRMED
            self._ready = False
        return True

    def _on_active_message_shutdown(self, job: Job, handle: Any, message: Mapping[str, Any]) -> bool:
        if message["gen"] != handle.generation:
            return False
        if message["t"] == "result" and message.get("id") == job.req_id:
            self._finish(job, Outcome("result", explanation=message["explanation"],
                                      generation=handle.generation), "completed")
            return True
        if message["t"] == "error" and message.get("id") == job.req_id \
                and message["code"] != worker_module.CODE_INTERNAL:
            if message["code"] == CODE_NUMERICAL:
                self._bump("numerical_subtype:" + message["subtype"])
            self._finish(job, _error(message["code"], tuple(message.get("violations", ()))),
                         "completed_error:" + message["code"])
            return True
        self._shutdown_aborted = True
        return False
