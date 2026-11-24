"""Deterministic tests for the Level 8 bounded dispatcher (no real processes).

The dispatcher is driven synchronously with ``pump()`` against fake worker handles and a
fake monotonic clock, so timing-sensitive paths (deadlines, expiry, restart budgets,
breaker cooldown) are exact and need no sleeps. Threads are used only where the property
under test is concurrency itself (atomic admission, completion/cancellation races).
"""

from __future__ import annotations

import collections
import os
import random
import select
import threading
import unittest

from src import level8_dispatch as d
from src import level8_worker as w
from src.level8_explain import InvalidExplanationInput, OutsideExplanationDomain

VALID = {
    "weekly_km_to_date": 31.4159, "sleep_hours_last_night": 7.0, "resting_hr_morning": 50.0,
    "soreness_score_evening": 4.0, "prior_injury": 0,
}
ARTIFACTS = {"a": "1" * 64}


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def advance(self, dt):
        self.t += dt

    def sleep(self, timeout, waker):  # never blocks: pump mode only
        pass


def explanation():
    return {
        "prediction": {"probability": 0.5, "decision": 1, "threshold": 0.5, "device": "cpu", "selected_approach": "x"},
        "logit": 0.0, "baseline_logit": 1.0, "target_difference": -1.0, "attribution_sum": -1.0,
        "completeness_residual": 0.0, "allowed_residual": 0.007,
        "attributions": [
            {"feature": f, "value": 1.0, "standardized_value": 0.0, "attribution": -0.2} for f in w.FEATURES
        ],
    }


def ready_message(gen, threads=1, interop=1, forbidden=(), integrity="passed", artifacts=None):
    return {
        "t": "ready", "gen": gen, "pid": 1000 + gen, "torch_threads": threads, "torch_interop_threads": interop,
        "loaded_forbidden_modules": list(forbidden), "self_test_ms": 1.0,
        "metadata": {"model": {"integrity_status": integrity, "artifact_sha256": artifacts or ARTIFACTS}},
    }


class FakeHandle:
    def __init__(self, gen, clock, log):
        self.generation, self.clock, self.log = gen, clock, log
        self.events = collections.deque()
        self.sent = []
        self.outstanding = None
        self.terminated = False
        self.send_error = None
        self.start_error = None
        self.terminate_results = collections.deque()  # scripted confirmations; default True
        self.terminate_calls = 0

    def start(self):
        self.log.append(("start", self.generation))
        if self.start_error:
            raise OSError("boom")

    def send(self, message, deadline):
        assert self.outstanding is None, "two explanations outstanding in the worker"
        if self.send_error:
            raise d.WorkerIOError(self.send_error)
        self.sent.append(message)
        self.outstanding = message["id"]

    def wait(self, deadline, waker):
        if self.events:
            event = self.events.popleft()
            if event[0] == "message" and event[1]["t"] in ("result", "error"):
                self.outstanding = None
            return event
        if waker.is_set():
            return ("wake",)
        if deadline is not None and self.clock.now() >= deadline:
            return ("timeout",)
        return ("pending",)

    def terminate(self):
        self.log.append(("terminate", self.generation))
        self.terminate_calls += 1
        confirmed = self.terminate_results.popleft() if self.terminate_results else True
        self.terminated = confirmed
        return confirmed

    def graceful_stop(self, deadline):
        self.log.append(("graceful", self.generation))
        confirmed = self.terminate_results.popleft() if self.terminate_results else True
        self.terminated = confirmed
        return confirmed

    def push(self, event):
        self.events.append(event)


class Rig:
    def __init__(self, **overrides):
        self.clock = FakeClock()
        self.log = []
        self.handles = []
        cfg = d.DispatcherConfig(**overrides)
        self.d = d.Dispatcher("unused-root", cfg, expected_artifacts=ARTIFACTS, handle_factory=self._factory,
                              clock=self.clock)

    def _factory(self, gen):
        h = FakeHandle(gen, self.clock, self.log)
        self.handles.append(h)
        return h

    @property
    def h(self):
        return self.handles[-1]

    def pump(self):
        if self.d.diagnostics()["state"] in d.TERMINAL_STATES:
            return 0
        return self.d.pump()

    def become_ready(self):
        self.pump()
        assert self.d.diagnostics()["state"] == d.STARTING
        self.h.push(("message", ready_message(self.h.generation)))
        self.pump()
        assert self.d.diagnostics()["state"] == d.READY, self.d.diagnostics()
        return self

    def answer(self, handle=None, gen=None):
        h = handle or self.h
        msg = h.sent[-1]
        h.push(("message", {"t": "result", "gen": gen if gen is not None else h.generation,
                            "id": msg["id"], "explanation": explanation()}))

    def diag(self):
        return self.d.diagnostics()

    def close(self):
        self.d._waker.close()


def wait_for(predicate, timeout=20.0, interval=0.005):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class BlockingFake(FakeHandle):
    """Real-time blocking variant for background mode. Records every thread that touches it."""

    def __init__(self, gen, clock, log, owners):
        super().__init__(gen, clock, log)
        self.owners = owners

    def _touch(self):
        self.owners.add(threading.get_ident())

    def start(self):
        self._touch()
        super().start()

    def send(self, message, deadline):
        self._touch()
        super().send(message, deadline)

    def terminate(self):
        self._touch()
        return super().terminate()

    def graceful_stop(self, deadline):
        self._touch()
        return super().graceful_stop(deadline)

    def wait(self, deadline, waker):
        self._touch()
        while True:
            if self.events:
                return super().wait(deadline, waker)
            if waker.is_set():
                return ("wake",)
            if deadline is not None and self.clock.now() >= deadline:
                return ("timeout",)
            select.select([waker.fileno()], [], [], 0.005)


class RigTestCase(unittest.TestCase):
    def rig(self, **overrides):
        r = Rig(**overrides)
        self.addCleanup(r.close)
        return r


class AdmissionTests(RigTestCase):
    def test_no_process_work_at_construction_and_unavailable_before_ready(self):
        r = self.rig()
        self.assertEqual(r.log, [])
        with self.assertRaises(d.AdmissionRejected) as ctx:
            r.d.submit(VALID)
        self.assertEqual(ctx.exception.code, d.CODE_UNAVAILABLE)
        r.pump()
        self.assertEqual(r.diag()["state"], d.STARTING)
        with self.assertRaises(d.AdmissionRejected):
            r.d.submit(VALID)

    def test_one_active_eight_waiting_tenth_rejected(self):
        r = self.rig().become_ready()
        jobs = [r.d.submit(VALID) for _ in range(9)]
        diag = r.diag()
        self.assertEqual((diag["active"], diag["queued"]), (1, 8))
        with self.assertRaises(d.AdmissionRejected) as ctx:
            r.d.submit(VALID)
        self.assertEqual(ctx.exception.code, d.CODE_OVERLOADED)
        self.assertEqual(r.diag()["high_water"], {"active": 1, "queued": 8, "admitted": 9})
        self.assertEqual(jobs[0].state, d.RESERVED)
        self.assertEqual([j.state for j in jobs[1:]], [d.QUEUED] * 8)
        self.assertEqual(r.diag()["invariant_violations"], 0)

    def test_idle_worker_reserves_the_active_slot_directly(self):
        r = self.rig().become_ready()
        r.d.submit(VALID)
        diag = r.diag()
        self.assertEqual((diag["active"], diag["queued"]), (1, 0))
        self.assertEqual(diag["high_water"]["queued"], 0)  # never transits the waiting queue

    def test_validation_happens_before_capacity(self):
        r = self.rig().become_ready()
        before = r.diag()
        with self.assertRaises(InvalidExplanationInput):
            r.d.submit(dict(VALID, weekly_km_to_date="x"))
        with self.assertRaises(OutsideExplanationDomain):
            r.d.submit(dict(VALID, weekly_km_to_date=19.0))
        with self.assertRaises(InvalidExplanationInput):
            r.d.submit(dict(VALID, prior_injury=0.5))
        after = r.diag()
        self.assertEqual((before["active"], before["queued"]), (after["active"], after["queued"]))
        self.assertEqual(after["counters"].get("admitted", 0), 0)

    def test_concurrent_admission_is_atomic(self):
        r = self.rig().become_ready()
        results = []
        barrier = threading.Barrier(64)

        def attempt():
            barrier.wait()
            try:
                results.append(r.d.submit(VALID))
            except d.AdmissionRejected as exc:
                results.append(exc.code)

        threads = [threading.Thread(target=attempt) for _ in range(64)]
        [t.start() for t in threads]
        [t.join(10) for t in threads]
        accepted = [x for x in results if isinstance(x, d.Job)]
        self.assertEqual(len(accepted), 9)
        self.assertEqual(results.count(d.CODE_OVERLOADED), 55)
        diag = r.diag()
        self.assertEqual((diag["active"], diag["queued"]), (1, 8))
        self.assertEqual(diag["invariant_violations"], 0)

    def test_serialized_execution_and_promotion(self):
        r = self.rig().become_ready()
        jobs = [r.d.submit(VALID) for _ in range(3)]
        for i in range(3):
            r.pump()
            self.assertEqual(len(r.h.sent), i + 1)  # exactly one outstanding at a time
            r.answer()
            r.pump()
            self.assertEqual(jobs[i].future.result(0).kind, "result")
        self.assertEqual(r.diag()["active"], 0)
        self.assertEqual([m["id"] for m in r.h.sent], sorted({m["id"] for m in r.h.sent}))

    def test_threaded_submitters_with_a_single_owner_pumping(self):
        r = self.rig()
        stop = threading.Event()
        errors = []

        def owner():  # the pumper is the sole manual owner: it also performs the readiness handshake
            try:
                while not stop.is_set():
                    r.pump()
                    if r.diag()["state"] == d.STARTING:
                        r.h.push(("message", ready_message(r.h.generation)))
            except BaseException as exc:  # e.g. the fake's serialization assertion
                errors.append(exc)

        def responder():
            answered = 0
            while not stop.is_set():
                if answered < len(r.h.sent) and r.h.outstanding is not None:
                    r.answer()
                    answered += 1

        threads = [threading.Thread(target=owner), threading.Thread(target=responder)]
        [t.start() for t in threads]
        self.assertTrue(wait_for(lambda: r.diag()["state"] == d.READY))
        jobs, lock = [], threading.Lock()

        def client():
            for _ in range(20):
                try:
                    job = r.d.submit(VALID)
                    with lock:
                        jobs.append(job)
                    job.future.result(10)
                except d.AdmissionRejected:
                    pass

        clients = [threading.Thread(target=client) for _ in range(4)]
        [t.start() for t in clients]
        [t.join(30) for t in clients]
        stop.set()
        [t.join(10) for t in threads]
        self.assertEqual(errors, [])
        self.assertTrue(all(j.future.done() for j in jobs))
        self.assertTrue(all(j.future.result(0).kind == "result" for j in jobs))
        self.assertEqual(r.diag()["invariant_violations"], 0)
        self.assertLessEqual(r.diag()["high_water"]["admitted"], 9)
        self.assertEqual(len(r.h.sent), len(jobs))  # every admitted job executed exactly once


class DeadlineAndCancellationTests(RigTestCase):
    def test_queue_expiry_releases_capacity_without_reaching_the_child(self):
        r = self.rig().become_ready()
        a, b = r.d.submit(VALID), r.d.submit(VALID)
        r.pump()
        r.clock.advance(0.6)
        r.answer()  # A's reply is already there when the dispatcher looks
        r.pump()
        self.assertEqual(a.future.result(0).kind, "result")
        self.assertEqual(b.future.result(0).code, d.CODE_DEADLINE)
        self.assertEqual(len(r.h.sent), 1)  # B never reached the worker
        self.assertEqual((r.diag()["active"], r.diag()["queued"]), (0, 0))

    def test_minimum_remaining_time_before_dispatch(self):
        for finish_at, dispatched in ((0.44, True), (0.46, False)):
            with self.subTest(finish_at=finish_at):
                r = self.rig().become_ready()
                r.d.submit(VALID)
                b = r.d.submit(VALID)
                r.pump()
                r.clock.advance(finish_at)
                r.answer()
                r.pump()
                self.assertEqual(len(r.h.sent), 2 if dispatched else 1)
                if not dispatched:
                    self.assertEqual(b.future.result(0).code, d.CODE_DEADLINE)

    def test_effective_execution_deadline_is_the_earlier_one(self):
        r = self.rig().become_ready()
        a, b = r.d.submit(VALID), r.d.submit(VALID)
        r.pump()
        self.assertAlmostEqual(a.exec_deadline, 0.25)  # dispatch + 250 ms
        r.clock.advance(0.30)
        r.answer()
        r.pump()
        self.assertAlmostEqual(b.exec_deadline, 0.5)  # request deadline is earlier than 0.30 + 0.25

    def test_execution_timeout_restarts_and_fails_queued_jobs_consistently(self):
        r = self.rig().become_ready()
        a, b, c = r.d.submit(VALID), r.d.submit(VALID), r.d.submit(VALID)
        r.pump()
        r.clock.advance(0.26)
        r.pump()
        self.assertEqual(a.future.result(0).code, d.CODE_DEADLINE)
        self.assertEqual(b.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertEqual(c.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertLess(r.log.index(("terminate", 1)), r.log.index(("start", 2)))
        diag = r.diag()
        self.assertEqual((diag["state"], diag["generation"], diag["restarts_total"]), (d.STARTING, 2, 1))
        with self.assertRaises(d.AdmissionRejected):  # no admission while recovering
            r.d.submit(VALID)
        self.assertEqual((diag["active"], diag["queued"]), (0, 0))

    def test_queued_cancellation_releases_capacity_immediately(self):
        r = self.rig().become_ready()
        jobs = [r.d.submit(VALID) for _ in range(9)]
        with self.assertRaises(d.AdmissionRejected):
            r.d.submit(VALID)
        self.assertTrue(r.d.cancel(jobs[5]))
        self.assertTrue(jobs[5].future.cancelled())
        r.d.submit(VALID)  # capacity came back at once
        self.assertEqual(r.diag()["queued"], 8)
        self.assertFalse(r.d.cancel(jobs[5]))  # idempotent

    def test_active_cancellation_holds_the_slot_until_the_worker_finishes(self):
        r = self.rig().become_ready()
        jobs = [r.d.submit(VALID) for _ in range(9)]
        r.pump()
        r.d.cancel(jobs[0])
        self.assertTrue(jobs[0].future.cancelled())
        with self.assertRaises(d.AdmissionRejected):
            r.d.submit(VALID)  # still 9 admitted
        r.pump()
        self.assertEqual(len(r.h.sent), 1)  # no concurrent second execution
        r.answer()
        r.pump()
        self.assertEqual(len(r.h.sent), 2)  # the next job runs only now
        r.d.submit(VALID)
        self.assertEqual(r.diag()["invariant_violations"], 0)

    def test_cancelled_reserved_job_is_never_sent(self):
        r = self.rig().become_ready()
        job = r.d.submit(VALID)
        r.d.cancel(job)
        r.pump()
        self.assertEqual(r.h.sent, [])
        self.assertEqual(r.diag()["active"], 0)

    def test_completion_cancellation_and_deadline_races_resolve_once(self):
        r = self.rig().become_ready()
        job = r.d.submit(VALID)
        r.pump()
        r.answer()
        r.pump()
        self.assertEqual(job.future.result(0).kind, "result")
        self.assertFalse(r.d.cancel(job))  # cancel after completion is a no-op
        r.clock.advance(5)
        r.pump()
        self.assertEqual(r.diag()["counters"]["completed"], 1)
        self.assertEqual(r.diag()["counters"]["resolved"], 1)

    def test_threaded_cancel_versus_completion_race(self):
        for _ in range(100):
            r = self.rig().become_ready()
            job = r.d.submit(VALID)
            r.pump()
            r.answer()
            barrier = threading.Barrier(2)

            def cancel():
                barrier.wait()
                r.d.cancel(job)

            t = threading.Thread(target=cancel)
            t.start()
            barrier.wait()
            r.pump()
            t.join(5)
            diag = r.diag()
            self.assertEqual((diag["active"], diag["queued"]), (0, 0))
            self.assertEqual(job.state, d.DONE)
            self.assertTrue(job.future.done())
            self.assertEqual(diag["counters"]["completed"], 1)  # released exactly once
            self.assertEqual(diag["invariant_violations"], 0)


class FailureClassificationTests(RigTestCase):
    def active(self):
        r = self.rig().become_ready()
        job = r.d.submit(VALID)
        r.pump()
        return r, job

    def test_request_errors_do_not_restart(self):
        for code in (d.CODE_OUTSIDE_DOMAIN, d.CODE_INVALID_INPUT, d.CODE_NUMERICAL):
            with self.subTest(code=code):
                r, job = self.active()
                msg = {"t": "error", "gen": 1, "id": r.h.sent[-1]["id"], "code": code}
                if code == d.CODE_NUMERICAL:
                    msg["subtype"] = w.SUBTYPE_UNCLASSIFIED
                if code == d.CODE_OUTSIDE_DOMAIN:
                    msg["violations"] = [{"feature": "weekly_km_to_date", "min": 20.0, "max": 45.0}]
                r.h.push(("message", msg))
                r.pump()
                outcome = job.future.result(0)
                self.assertEqual((outcome.kind, outcome.code), ("error", code))
                self.assertEqual((r.diag()["state"], r.diag()["generation"]), (d.READY, 1))
                if code == d.CODE_OUTSIDE_DOMAIN:
                    self.assertEqual(outcome.violations[0]["feature"], "weekly_km_to_date")

    def test_internal_worker_error_is_restart_worthy(self):
        r, job = self.active()
        r.h.push(("message", {"t": "error", "gen": 1, "id": r.h.sent[-1]["id"], "code": w.CODE_INTERNAL}))
        r.pump()
        self.assertEqual(job.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertEqual((r.diag()["state"], r.diag()["generation"]), (d.STARTING, 2))

    def test_abrupt_death_active_and_idle(self):
        r, job = self.active()
        r.h.push(("closed", "exited"))
        r.pump()
        self.assertEqual(job.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertEqual(r.diag()["generation"], 2)
        r2 = self.rig().become_ready()
        r2.h.push(("closed", "exited"))
        r2.pump()
        self.assertEqual(r2.diag()["generation"], 2)

    def test_malformed_ipc_is_a_protocol_fault(self):
        r, job = self.active()
        r.h.push(("fault", "malformed"))
        r.pump()
        self.assertEqual(job.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertEqual(r.diag()["generation"], 2)

    def test_stale_generation_is_ignored_and_mismatched_id_is_a_fault(self):
        r, job = self.active()
        r.answer(gen=99)  # stale generation: ignored, job keeps waiting
        r.pump()
        self.assertFalse(job.future.done())
        self.assertEqual(r.diag()["counters"]["stale_ignored"], 1)
        r.h.push(("message", {"t": "result", "gen": 1, "id": 4242, "explanation": explanation()}))
        r.pump()
        self.assertEqual(job.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertEqual(r.diag()["generation"], 2)

    def test_unexpected_message_while_idle_is_a_fault(self):
        r = self.rig().become_ready()
        r.h.push(("message", {"t": "result", "gen": 1, "id": 1, "explanation": explanation()}))
        r.pump()
        self.assertEqual(r.diag()["generation"], 2)

    def test_send_failures(self):
        for kind, code in (("timeout", d.CODE_DEADLINE), ("closed", d.CODE_UNAVAILABLE)):
            with self.subTest(kind=kind):
                r = self.rig().become_ready()
                r.h.send_error = kind
                job = r.d.submit(VALID)
                r.pump()
                self.assertEqual(job.future.result(0).code, code)
                self.assertEqual(r.diag()["generation"], 2)

    def test_startup_failures(self):
        r = self.rig()
        r.pump()
        r.clock.advance(20.1)
        r.pump()
        self.assertEqual((r.diag()["generation"], r.diag()["restarts_total"]), (2, 1))  # timeout -> replacement
        r.h.push(("message", {"t": "fatal", "gen": 2, "code": "self_test_failed"}))
        r.pump()
        self.assertEqual(r.diag()["generation"], 3)
        r.h.push(("message", ready_message(3, threads=4)))  # wrong thread configuration
        r.pump()
        self.assertEqual(r.diag()["generation"], 4)
        r.pump()
        self.assertEqual((r.diag()["state"], r.diag()["generation"]), (d.STARTING, 4))  # third replacement
        r.h.push(("message", {"t": "fatal", "gen": 4, "code": "init_failed"}))
        r.pump()
        self.assertEqual(r.diag()["state"], d.OPEN)  # budget (3) exhausted

    def test_ready_validation(self):
        for bad in (dict(threads=2), dict(interop=2), dict(forbidden=["fastapi"]), dict(integrity="failed"),
                    dict(artifacts={"a": "2" * 64})):
            with self.subTest(bad=bad):
                r = self.rig()
                r.pump()
                r.h.push(("message", ready_message(1, **bad)))
                r.pump()
                self.assertEqual(r.diag()["generation"], 2)
        r = self.rig()
        r.pump()
        r.h.push(("message", ready_message(2)))  # wrong generation
        r.pump()
        self.assertEqual(r.diag()["generation"], 2)

    def test_spawn_failure(self):
        r = self.rig()
        r.d._factory = lambda gen: (_ for _ in ()).throw(OSError("no fork"))
        r.pump()
        self.assertEqual(r.diag()["counters"]["failure:spawn_failed"], 4)  # initial + 3 replacements
        self.assertEqual(r.diag()["state"], d.OPEN)


class CircuitBreakerTests(RigTestCase):
    def kill(self, r, at):
        r.clock.t = at
        r.h.push(("closed", "exited"))
        r.pump()
        if r.diag()["state"] == d.STARTING:
            r.h.push(("message", ready_message(r.h.generation)))
            r.pump()

    def test_budget_exhaustion_cooldown_and_half_open_recovery(self):
        r = self.rig().become_ready()
        for t in (0.0, 1.0, 2.0):
            self.kill(r, t)
            self.assertEqual(r.diag()["state"], d.READY)
        self.assertEqual(r.diag()["restarts_total"], 3)
        self.kill(r, 3.0)  # fourth failure inside the window: breaker opens
        diag = r.diag()
        self.assertEqual((diag["state"], diag["breaker_open"]), (d.OPEN, True))
        with self.assertRaises(d.AdmissionRejected) as ctx:
            r.d.submit(VALID)
        self.assertEqual(ctx.exception.code, d.CODE_UNAVAILABLE)
        r.clock.t = 62.9
        r.pump()
        self.assertEqual(r.diag()["state"], d.OPEN)
        starts = len([e for e in r.log if e[0] == "start"])
        r.clock.t = 63.0  # cooldown (60 s) elapsed: exactly one recovery attempt
        r.pump()
        self.assertEqual(len([e for e in r.log if e[0] == "start"]), starts + 1)
        r.h.push(("message", ready_message(r.h.generation)))
        r.pump()
        self.assertEqual(r.diag()["state"], d.READY)
        self.assertEqual(r.diag()["restarts_in_window"], 1)
        r.d.submit(VALID)

    def test_failed_recovery_attempt_reopens_the_circuit(self):
        r = self.rig(max_restarts=0).become_ready()
        self.kill(r, 5.0)
        self.assertEqual(r.diag()["state"], d.OPEN)
        r.clock.t = 65.0
        r.pump()
        self.assertEqual(r.diag()["state"], d.STARTING)
        r.h.push(("message", {"t": "fatal", "gen": r.h.generation, "code": "init_failed"}))
        r.pump()
        self.assertEqual(r.diag()["state"], d.OPEN)
        r.clock.t = 124.9
        r.pump()
        self.assertEqual(r.diag()["state"], d.OPEN)
        r.clock.t = 125.0
        r.pump()
        self.assertEqual(r.diag()["state"], d.STARTING)

    def test_rolling_window_boundary(self):
        for fail_at, expect_open in ((59.999, True), (60.0, False)):
            with self.subTest(fail_at=fail_at):
                r = self.rig().become_ready()
                for t in (0.0, 1.0, 2.0):
                    self.kill(r, t)
                self.kill(r, fail_at)
                self.assertEqual(r.diag()["state"] == d.OPEN, expect_open)

    def test_initial_startup_does_not_consume_the_budget(self):
        r = self.rig().become_ready()
        self.assertEqual((r.diag()["restarts_total"], r.diag()["restarts_in_window"]), (0, 0))


class ShutdownTests(RigTestCase):
    def test_shutdown_before_start_and_idempotent(self):
        r = self.rig()
        r.d.shutdown()
        r.d.shutdown()
        self.assertEqual(r.diag()["state"], d.STOPPED)
        self.assertEqual(r.log, [])

    def test_shutdown_during_startup(self):
        r = self.rig()
        r.pump()
        r.d.shutdown()
        self.assertEqual(r.diag()["state"], d.STOPPED)
        self.assertIn(("terminate", 1), r.log)
        self.assertNotIn(("graceful", 1), r.log)

    def test_shutdown_when_idle_is_graceful_and_blocks_admission(self):
        r = self.rig().become_ready()
        r.d.shutdown()
        self.assertEqual(r.diag()["state"], d.STOPPED)
        self.assertIn(("graceful", 1), r.log)
        with self.assertRaises(d.AdmissionRejected) as ctx:
            r.d.submit(VALID)
        self.assertEqual(ctx.exception.code, d.CODE_UNAVAILABLE)

    def test_shutdown_with_active_and_queued_work(self):
        r = self.rig().become_ready()
        a, b, c = r.d.submit(VALID), r.d.submit(VALID), r.d.submit(VALID)
        r.pump()
        r.d.shutdown()  # queued jobs fail immediately; the active job may finish
        self.assertEqual(b.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertEqual(c.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertFalse(a.future.done())
        r.answer()
        r.pump()
        self.assertEqual(a.future.result(0).kind, "result")
        self.assertEqual(r.diag()["state"], d.STOPPED)
        self.assertEqual(len(r.h.sent), 1)
        self.assertEqual(len(r.handles), 1)  # no restart during shutdown

    def test_shutdown_active_beyond_its_budget_terminates(self):
        r = self.rig().become_ready()
        a = r.d.submit(VALID)
        r.pump()
        r.d.shutdown()
        r.clock.advance(0.26)
        r.pump()
        self.assertEqual(a.future.result(0).code, d.CODE_DEADLINE)  # its own execution deadline had passed
        self.assertEqual(r.diag()["state"], d.STOPPED)
        self.assertIn(("terminate", 1), r.log)
        self.assertNotIn(("graceful", 1), r.log)

    def test_shutdown_during_recovery_and_open_suppresses_restarts(self):
        r = self.rig().become_ready()
        r.h.push(("closed", "exited"))
        r.d.pump(max_steps=1)  # failure noticed: RESTARTING
        self.assertEqual(r.diag()["state"], d.RESTARTING)
        r.d.shutdown()
        self.assertEqual(r.diag()["state"], d.STOPPED)
        self.assertEqual(len(r.handles), 1)
        r2 = self.rig(max_restarts=0).become_ready()
        r2.h.push(("closed", "exited"))
        r2.pump()
        self.assertEqual(r2.diag()["state"], d.OPEN)
        r2.d.shutdown()
        self.assertEqual(r2.diag()["state"], d.STOPPED)
        self.assertEqual(len(r2.handles), 1)

    def test_no_unresolved_futures_after_shutdown(self):
        r = self.rig().become_ready()
        jobs = [r.d.submit(VALID) for _ in range(9)]
        r.pump()
        r.d.shutdown()
        r.clock.advance(3)
        r.pump()
        self.assertTrue(all(j.future.done() for j in jobs))
        self.assertEqual((r.diag()["active"], r.diag()["queued"]), (0, 0))


class WakeAndDiagnosticsTests(RigTestCase):
    def test_waker_is_bounded_and_coalescing(self):
        waker = d.Waker()
        self.addCleanup(waker.close)
        for _ in range(100_000):
            waker.set()
        self.assertEqual(len(os.read(waker.fileno(), 1000)), 1)
        waker._set = True
        waker.clear()
        self.assertFalse(waker.is_set())

    def test_submit_cancel_and_shutdown_wake_the_dispatcher(self):
        r = self.rig().become_ready()
        r.d._waker.clear()
        job = r.d.submit(VALID)
        self.assertTrue(r.d._waker.is_set())
        r.pump()
        r.d._waker.clear()
        r.d.cancel(job)
        self.assertTrue(r.d._waker.is_set())
        r.d._waker.clear()
        r.d.shutdown()  # the abandoned active job holds the worker until its budget lapses
        r.clock.advance(0.3)
        r.pump()
        self.assertEqual(r.diag()["state"], d.STOPPED)

    def test_diagnostics_contain_only_operational_information(self):
        r = self.rig().become_ready()
        r.d.submit(VALID)
        r.pump()
        r.answer()
        r.pump()
        diag = r.diag()
        self.assertEqual(set(diag), {"state", "generation", "active", "queued", "high_water", "restarts_total",
                                     "restarts_in_window", "breaker_open", "cleanup_failed", "counters",
                                     "invariant_violations"})
        text = repr(diag)
        for secret in ("31.4159", "weekly_km", "attribution", "probability"):
            self.assertNotIn(secret, text)


class OwnershipTests(RigTestCase):
    """Exclusive dispatcher ownership: manual and background modes cannot be mixed."""

    def background(self, **cfg):
        owners, log, handles = set(), [], []

        def factory(gen):
            h = BlockingFake(gen, d.RealClock(), log, owners)
            handles.append(h)
            return h

        disp = d.Dispatcher("unused", d.DispatcherConfig(**cfg), expected_artifacts=ARTIFACTS, handle_factory=factory)
        self.addCleanup(disp._waker.close)
        self.addCleanup(disp.shutdown)
        return disp, handles, owners

    def test_external_pump_is_rejected_in_background_mode_before_any_handle_access(self):
        disp, handles, owners = self.background()
        disp.start()
        self.assertTrue(wait_for(lambda: handles and disp.diagnostics()["state"] == d.STARTING))
        handles[0].push(("message", ready_message(1)))
        self.assertTrue(wait_for(lambda: disp.diagnostics()["state"] == d.READY))
        before = set(owners)
        with self.assertRaises(RuntimeError):
            disp.pump()
        job = disp.submit(VALID)
        self.assertTrue(wait_for(lambda: handles[0].sent))
        with self.assertRaises(RuntimeError):
            disp.pump(max_steps=1)
        handles[0].push(("message", {"t": "result", "gen": 1, "id": handles[0].sent[-1]["id"],
                                     "explanation": explanation()}))
        self.assertEqual(job.future.result(10).kind, "result")
        disp.shutdown()
        self.assertEqual(len(owners), 1)  # every handle access came from the one dispatcher thread
        self.assertNotIn(threading.get_ident(), owners)
        self.assertEqual(before, owners)
        self.assertEqual(disp.diagnostics()["state"], d.STOPPED)

    def test_concurrent_manual_ownership_acquisition_has_one_winner(self):
        r = self.rig()
        barrier = threading.Barrier(16)
        outcomes = []

        def attempt():
            barrier.wait()
            try:
                r.d.pump(max_steps=1)
                outcomes.append("owner")
            except RuntimeError:
                outcomes.append("rejected")

        threads = [threading.Thread(target=attempt) for _ in range(16)]
        [t.start() for t in threads]
        [t.join(10) for t in threads]
        self.assertEqual((outcomes.count("owner"), outcomes.count("rejected")), (1, 15))
        self.assertEqual(len(r.handles), 1)  # the single owner spawned the single worker

    def test_non_owner_cannot_pump_and_owner_is_unaffected(self):
        r = self.rig().become_ready()
        errors = []
        t = threading.Thread(target=lambda: errors.append(self._try_pump(r)))
        t.start()
        t.join(10)
        self.assertEqual(errors, ["rejected"])
        job = r.d.submit(VALID)
        r.pump()
        r.answer()
        r.pump()
        self.assertEqual(job.future.result(0).kind, "result")
        self.assertEqual(len(r.h.sent), 1)  # no duplicate dispatch

    @staticmethod
    def _try_pump(r):
        try:
            r.d.pump()
            return "owner"
        except RuntimeError:
            return "rejected"

    def test_start_races_manual_ownership_and_exactly_one_wins(self):
        for _ in range(20):
            disp, handles, owners = self.background()
            barrier = threading.Barrier(2)
            results = {}

            def do_start():
                barrier.wait()
                try:
                    disp.start()
                    results["start"] = "ok"
                except RuntimeError:
                    results["start"] = "rejected"

            def do_pump():
                barrier.wait()
                try:
                    disp.pump(max_steps=1)
                    results["pump"] = "ok"
                except RuntimeError:
                    results["pump"] = "rejected"

            threads = [threading.Thread(target=do_start), threading.Thread(target=do_pump)]
            [t.start() for t in threads]
            [t.join(10) for t in threads]
            self.assertEqual(sorted(results.values()), ["ok", "rejected"], results)
            self.assertEqual(len(handles), 1 if "ok" in results.values() else 0)
            disp.shutdown()

    def test_start_after_manual_ownership_and_pump_after_shutdown_are_rejected(self):
        r = self.rig().become_ready()
        with self.assertRaises(RuntimeError):
            r.d.start()
        r.d.shutdown()
        self.assertEqual(r.diag()["state"], d.STOPPED)
        with self.assertRaises(RuntimeError):
            r.d.pump()
        r.d.shutdown()  # idempotent

    def test_non_owner_shutdown_in_manual_mode_only_flags_and_the_owner_finishes(self):
        r = self.rig().become_ready()
        t = threading.Thread(target=r.d.shutdown)
        t.start()
        t.join(10)
        self.assertEqual(r.diag()["state"], d.STOPPING)
        r.pump()
        self.assertEqual(r.diag()["state"], d.STOPPED)
        self.assertEqual(r.log.count(("graceful", 1)), 1)

    def test_internal_step_is_guarded_too(self):
        r = self.rig().become_ready()
        failures = []

        def poke():
            try:
                r.d._step()
            except RuntimeError:
                failures.append(True)

        t = threading.Thread(target=poke)
        t.start()
        t.join(10)
        self.assertEqual(failures, [True])

    def test_no_duplicate_dispatch_under_hostile_concurrent_pumping(self):
        r = self.rig().become_ready()
        stop = threading.Event()
        rejected = []

        def hostile():
            while not stop.is_set():
                try:
                    r.d.pump(max_steps=1)
                except RuntimeError:
                    rejected.append(1)

        threads = [threading.Thread(target=hostile) for _ in range(4)]
        [t.start() for t in threads]
        jobs = [r.d.submit(VALID) for _ in range(5)]
        for _ in range(5):
            r.pump()
            r.answer()
            r.pump()
        stop.set()
        [t.join(10) for t in threads]
        self.assertTrue(rejected)
        self.assertEqual([j.future.result(0).kind for j in jobs], ["result"] * 5)
        self.assertEqual(len(r.h.sent), 5)


class DeadlineMappingTests(RigTestCase):
    """Absolute-deadline-aware failure codes during recovery and shutdown."""

    def test_recovery_expired_vs_unexpired_queued_jobs(self):
        r = self.rig().become_ready()
        a, b = r.d.submit(VALID), r.d.submit(VALID)
        r.pump()
        r.clock.t = 0.3
        c = r.d.submit(VALID)  # admitted later: its own deadline is 0.8
        r.clock.t = 0.6
        r.h.push(("closed", "exited"))
        r.pump()
        self.assertEqual(a.future.result(0).code, d.CODE_DEADLINE)  # execution deadline (0.25) passed
        self.assertEqual(b.future.result(0).code, d.CODE_DEADLINE)  # its own request deadline (0.5) passed
        self.assertEqual(c.future.result(0).code, d.CODE_UNAVAILABLE)  # unexpired: merely invalidated

    def test_equality_boundaries_count_as_expired(self):
        for at, expected_queued, expected_active in ((0.5, d.CODE_DEADLINE, d.CODE_DEADLINE),
                                                     (0.4999, d.CODE_UNAVAILABLE, d.CODE_DEADLINE),
                                                     (0.25, d.CODE_UNAVAILABLE, d.CODE_DEADLINE),
                                                     (0.2499, d.CODE_UNAVAILABLE, d.CODE_UNAVAILABLE)):
            with self.subTest(at=at):
                r = self.rig().become_ready()
                a, b = r.d.submit(VALID), r.d.submit(VALID)
                r.pump()
                r.clock.t = at
                r.h.push(("closed", "exited"))
                r.pump()
                self.assertEqual(a.future.result(0).code, expected_active)
                self.assertEqual(b.future.result(0).code, expected_queued)

    def test_shutdown_expired_vs_unexpired_queued_jobs(self):
        r = self.rig().become_ready()
        a, b = r.d.submit(VALID), r.d.submit(VALID)
        r.pump()
        r.clock.t = 0.6
        c = r.d.submit(VALID)
        r.d.shutdown()
        self.assertEqual(b.future.result(0).code, d.CODE_DEADLINE)
        self.assertEqual(c.future.result(0).code, d.CODE_UNAVAILABLE)
        r.pump()
        self.assertEqual(a.future.result(0).code, d.CODE_DEADLINE)  # its execution deadline had passed too

    def test_active_execution_deadline_expiring_during_shutdown(self):
        r = self.rig().become_ready()
        a = r.d.submit(VALID)
        r.pump()
        r.d.shutdown()
        r.clock.advance(0.26)
        r.pump()
        self.assertEqual(a.future.result(0).code, d.CODE_DEADLINE)
        self.assertEqual(r.diag()["state"], d.STOPPED)

    def test_request_deadline_earlier_than_execution_timeout_during_shutdown(self):
        r = self.rig().become_ready()
        a, b = r.d.submit(VALID), r.d.submit(VALID)
        r.pump()
        r.clock.t = 0.30
        r.answer()
        r.pump()  # b dispatched at 0.30: exec deadline = min(0.5, 0.55) = 0.5
        self.assertAlmostEqual(b.exec_deadline, 0.5)
        r.d.shutdown()
        r.clock.t = 0.49
        r.pump()
        self.assertFalse(b.future.done())  # grace (2 s) does not extend, nor shorten, the 0.5 s deadline
        r.clock.t = 0.5
        r.pump()
        self.assertEqual(b.future.result(0).code, d.CODE_DEADLINE)

    def test_grace_expiring_before_the_execution_deadline_is_unavailable(self):
        r = self.rig(shutdown_grace=0.05).become_ready()
        a = r.d.submit(VALID)
        r.pump()
        r.d.shutdown()
        r.clock.advance(0.06)  # grace over, execution deadline (0.25) not reached
        r.pump()
        self.assertEqual(a.future.result(0).code, d.CODE_UNAVAILABLE)
        self.assertIn(("terminate", 1), r.log)

    def test_cancellation_racing_with_failure_resolution(self):
        r = self.rig().become_ready()
        a, b, c = r.d.submit(VALID), r.d.submit(VALID), r.d.submit(VALID)
        r.pump()
        r.d.cancel(b)  # queued cancel: released at once
        r.d.cancel(a)  # active cancel: slot held until the failure resolves it
        self.assertEqual((r.diag()["active"], r.diag()["queued"]), (1, 1))
        r.h.push(("closed", "exited"))
        r.pump()
        self.assertTrue(a.future.cancelled() and b.future.cancelled())
        self.assertEqual(c.future.result(0).code, d.CODE_UNAVAILABLE)
        diag = r.diag()
        self.assertEqual((diag["active"], diag["queued"], diag["invariant_violations"]), (0, 0, 0))


class NumericalSubtypeDiagnosticsTests(RigTestCase):
    def test_subtypes_are_counted_without_changing_the_client_code_or_restarting(self):
        for subtype in sorted(w.NUMERICAL_SUBTYPES):
            with self.subTest(subtype=subtype):
                r = self.rig().become_ready()
                job = r.d.submit(VALID)
                r.pump()
                r.h.push(("message", {"t": "error", "gen": 1, "id": r.h.sent[-1]["id"],
                                      "code": w.CODE_NUMERICAL, "subtype": subtype}))
                r.pump()
                outcome = job.future.result(0)
                self.assertEqual((outcome.kind, outcome.code, outcome.explanation), ("error", d.CODE_NUMERICAL, None))
                self.assertEqual(r.diag()["counters"]["numerical_subtype:" + subtype], 1)
                self.assertEqual((r.diag()["state"], r.diag()["generation"]), (d.READY, 1))


class CleanupSafetyTests(RigTestCase):
    def stuck(self):
        r = self.rig().become_ready()
        job = r.d.submit(VALID)
        r.pump()
        r.h.terminate_results.extend([False] * 20)  # an uncooperative process
        r.clock.advance(0.26)
        r.pump()
        return r, job

    def test_unconfirmed_termination_blocks_replacement_and_admission(self):
        r, job = self.stuck()
        self.assertEqual(job.future.result(0).code, d.CODE_DEADLINE)
        diag = r.diag()
        self.assertEqual((diag["state"], diag["cleanup_failed"]), (d.CLEANUP_FAILED, True))
        self.assertEqual(len(r.handles), 1)  # no replacement while the old process may be alive
        self.assertEqual(r.h.terminate_calls, d.TERMINATE_ATTEMPTS)  # bounded attempts
        self.assertEqual(diag["counters"]["cleanup_failed"], 1)
        with self.assertRaises(d.AdmissionRejected) as ctx:
            r.d.submit(VALID)
        self.assertEqual(ctx.exception.code, d.CODE_UNAVAILABLE)
        r.clock.advance(1000)
        r.pump()
        self.assertEqual(len(r.handles), 1)  # still nothing starts, even after the cooldown time

    def test_shutdown_does_not_claim_success_when_termination_is_unconfirmed(self):
        r, _ = self.stuck()
        r.d.shutdown()
        diag = r.diag()
        self.assertEqual((diag["state"], diag["cleanup_failed"]), (d.STOPPED_UNCONFIRMED, True))
        self.assertLessEqual(r.h.terminate_calls, d.TERMINATE_ATTEMPTS + 1 + d.TERMINATE_ATTEMPTS)  # restart + shutdown, bounded
        self.assertEqual(len(r.handles), 1)
        with self.assertRaises(RuntimeError):
            r.d.pump()

    def test_cleanup_failure_followed_by_confirmed_termination_at_shutdown(self):
        r, _ = self.stuck()
        r.h.terminate_results.clear()  # the process finally dies
        r.d.shutdown()
        self.assertEqual((r.diag()["state"], r.diag()["cleanup_failed"]), (d.STOPPED, False))

    def test_graceful_stop_failure_is_retried_boundedly(self):
        r = self.rig().become_ready()
        r.h.terminate_results.extend([False, False, False, False])
        r.d.shutdown()
        self.assertEqual(r.diag()["state"], d.STOPPED_UNCONFIRMED)
        self.assertEqual(r.h.terminate_calls, d.TERMINATE_ATTEMPTS)

    def test_normal_termination_path_is_unchanged(self):
        r = self.rig().become_ready()
        r.clock.advance(1)
        r.h.push(("closed", "exited"))
        r.pump()
        self.assertEqual((r.diag()["state"], r.diag()["generation"], r.diag()["cleanup_failed"]), (d.STARTING, 2, False))
        self.assertEqual(r.h.generation, 2)
        self.assertEqual(r.log.count(("terminate", 1)), 1)



class StormTests(RigTestCase):
    def test_randomized_operations_keep_every_invariant(self):
        rng = random.Random(8)
        r = self.rig().become_ready()
        jobs = []
        for _ in range(3000):
            op = rng.random()
            diag = r.diag()
            if op < 0.35:
                try:
                    jobs.append(r.d.submit(VALID))
                except d.AdmissionRejected:
                    pass
            elif op < 0.50 and jobs:
                r.d.cancel(rng.choice(jobs))
            elif op < 0.75 and r.h.outstanding is not None and diag["state"] == d.READY:
                r.answer()
            elif op < 0.85:
                r.clock.advance(rng.random() * 0.3)
            elif op < 0.88:
                r.h.push(("closed", "exited"))
            elif op < 0.90:
                r.clock.advance(rng.choice((20.5, 61.0)))
            r.pump()
            if r.diag()["state"] == d.STARTING:
                r.h.push(("message", ready_message(r.h.generation)))
                r.pump()
            diag = r.diag()
            self.assertLessEqual(diag["active"], 1)
            self.assertLessEqual(diag["queued"], 8)
            self.assertLessEqual(diag["active"] + diag["queued"], 9)
            self.assertEqual(diag["invariant_violations"], 0)
            self.assertLessEqual(len([j for j in jobs if j.state in (d.RESERVED, d.EXECUTING)]), 1)
        r.d.shutdown()
        r.clock.advance(3)
        r.pump()
        self.assertEqual(r.diag()["state"], d.STOPPED)
        self.assertTrue(all(j.future.done() for j in jobs))


if __name__ == "__main__":
    unittest.main()
