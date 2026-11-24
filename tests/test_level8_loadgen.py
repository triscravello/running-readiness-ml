"""Tests for the Level 8 mixed-load generator (summary schema 2).

Deterministic layers use an injectable clock, a fake time source for the open-loop scheduler and fake
``post`` functions, so accounting, capacity and deadline rules are proved without relying on real-time
sleeping. A separate class drives real asyncio HTTP traffic over loopback sockets against a controlled
in-test server (no production endpoint, no Level 8 service) to verify connection-pool contention, delayed
completion, the total client deadline, bounded outstanding work and cancellation with real sockets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

from tests import level8_loadgen as lg

WORKLOAD_FINGERPRINT = "2018255dbca75def54a926d78b250fd3209a9bd7d8fdc6d1f2d98b8b50e8edc6"
HISTORICAL_PROFILES = {
    "P0": dict(predict_rate=100.0, explain_callers=0, explain_rate=0.0, warmup=10.0, measure=30.0),
    "P1": dict(predict_rate=100.0, explain_callers=4, explain_rate=0.0, warmup=10.0, measure=30.0),
    "P2": dict(predict_rate=100.0, explain_callers=0, explain_rate=40.0, warmup=10.0, measure=30.0),
    "P3": dict(predict_rate=0.0, explain_callers=4, explain_rate=0.0, warmup=10.0, measure=30.0),
    "P4": dict(predict_rate=100.0, explain_callers=0, explain_rate=0.0, warmup=10.0, measure=10.0, burst=32, burst_at=1.0, bucket=1.0),
    "P5": dict(predict_rate=100.0, explain_callers=4, explain_rate=0.0, warmup=10.0, measure=300.0, bucket=30.0),
}


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self.headers = status, headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)


def predict_body(i):
    return {"probability": lg.EXPECTED_PROBABILITY[i], "decision": lg.EXPECTED_DECISION[i], "threshold": 0.5,
            "device": "cpu", "selected_approach": "partial"}


def explain_body(i):
    return {"schema_version": "explanation.v1", "notices": ["n"] * 9,
            "prediction": {"probability": lg.EXPECTED_PROBABILITY[i]},
            "attributions": [{"feature": f, "value": float(lg.INPUTS[i][f]), "attribution": 0.1} for f in lg.FEATURES]}


def ok_response(kind, i):
    if kind == "predict":
        return Resp(200, predict_body(i))
    return Resp(200, explain_body(i), {"cache-control": "no-store"})


def error_response(code, status, retry):
    headers = {"retry-after": retry} if retry else {}
    return Resp(status, {"error": {"code": code, "message": "m"}}, headers)


class FakeTime:
    """Scheduler time: sleep_until jumps the clock (no real waiting); arrivals already due never await."""

    def __init__(self):
        self.clock = FakeClock()
        self.jump_at = {}  # sleep call number -> time to jump to
        self.calls = 0

    async def sleep_until(self, when):
        self.calls += 1
        self.clock.t = max(self.clock.t, when, self.jump_at.get(self.calls, 0.0))
        await asyncio.sleep(0)


def make_ctx(post, clock, deadline_s=2.0, capacity=10):
    return lg.Context(post, clock, deadline_s, lg.Ledger(), lg.Admission(capacity))


async def instant_post(kind, index):
    return ok_response(kind, index)


async def run_stream(ctx, ft, rate, warm, meas, stream="predict_open_loop", kind="predict", stop=None):
    tasks = []
    await lg.run_open_loop(ctx, stream, kind, rate, warm, meas, 0.0, ft.sleep_until, asyncio.get_running_loop().create_task, tasks, stop)
    await asyncio.gather(*tasks, return_exceptions=True)
    return tasks


def entries_of(ctx, stream):
    return [e for e in ctx.ledger.entries.values() if e["stream"] == stream]


# =============================================================== arrival accounting
class ArrivalAccountingTests(unittest.IsolatedAsyncioTestCase):
    def test_exact_schedule_counts(self):
        self.assertEqual(lg.arrival_counts(100.0, 10.0, 30.0), (1000, 3000, 4000))
        self.assertEqual(lg.arrival_counts(40.0, 10.0, 30.0), (400, 1200, 1600))
        self.assertEqual(lg.arrival_counts(100.0, 0.0, 5.0), (0, 500, 500))
        self.assertEqual(lg.arrival_counts(7.5, 2.0, 4.0), (15, 30, 45))

    def test_invalid_configuration_is_rejected(self):
        for args in ((0.0, 10.0, 30.0), (-1.0, 10.0, 30.0), (100.0, -1.0, 30.0), (100.0, 10.0, 0.0), (100.0, 10.0, -5.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                lg.arrival_counts(*args)
        base = dict(profile="P0", predict_rate=None, explain_rate=None, explain_callers=None, warmup=None, measure=None,
                    burst=None, burst_at=None, bucket=None, timeout=2.0, max_backlog=100, max_connections=128)
        lg.build_config(SimpleNamespace(**base))
        for change in (dict(measure=0.0), dict(warmup=-1.0), dict(timeout=0.0), dict(max_backlog=0), dict(explain_callers=101)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                lg.build_config(SimpleNamespace(**{**base, **change}))
        with mock.patch.object(sys, "argv", ["x", "--base-url", "http://h", "--profile", "P0", "--measure", "0"]):
            with self.assertRaises(SystemExit) as raised:
                lg.main()
            self.assertEqual(raised.exception.code, 2)

    async def test_full_schedule_100_and_40_per_second_without_misses(self):
        for rate, warm_n, meas_n in ((100.0, 1000, 3000), (40.0, 400, 1200)):
            with self.subTest(rate=rate):
                ft = FakeTime()
                ctx = make_ctx(instant_post, ft.clock, capacity=100)
                await run_stream(ctx, ft, rate, 10.0, 30.0)
                es = entries_of(ctx, "predict_open_loop")
                ids = [e["id"] for e in es]
                self.assertEqual((len(es), len(set(ids))), (warm_n + meas_n, warm_n + meas_n))  # unique, none double counted
                self.assertEqual(sum(e["cohort"] == "warmup" for e in es), warm_n)
                self.assertEqual(sum(e["cohort"] == "measurement" for e in es), meas_n)
                self.assertTrue(all(e["state"] == "attempted" and e["outcome"] == "success" for e in es))
                self.assertAlmostEqual(es[-1]["sched"], (warm_n + meas_n - 1) / rate)

    async def test_half_open_boundary_belongs_to_the_next_cohort(self):
        ft = FakeTime()
        ctx = make_ctx(instant_post, ft.clock, capacity=100)
        await run_stream(ctx, ft, 10.0, 1.0, 1.0)
        es = {e["id"]: e for e in entries_of(ctx, "predict_open_loop")}
        self.assertEqual(es["predict_open_loop:9"]["cohort"], "warmup")        # t = 0.9
        self.assertEqual(es["predict_open_loop:10"]["cohort"], "measurement")  # t = 1.0 exactly
        self.assertEqual(es["predict_open_loop:19"]["cohort"], "measurement")  # t = 1.9; t = 2.0 does not exist
        self.assertNotIn("predict_open_loop:20", es)

    async def test_scheduler_stall_spanning_arrivals_keeps_the_schedule_and_records_lag(self):
        ft = FakeTime()
        ft.jump_at[500] = 8.0  # the scheduler resumes 3 s late, after ~300 arrivals were due
        ctx = make_ctx(instant_post, ft.clock, capacity=1000)
        await run_stream(ctx, ft, 100.0, 10.0, 30.0)
        es = entries_of(ctx, "predict_open_loop")
        self.assertEqual(len(es), 4000)
        self.assertEqual(sum(e["cohort"] == "measurement" for e in es), 3000)
        late = [e for e in es if e["start"] - e["sched"] > 1.0]
        self.assertGreater(len(late), 100)                  # issued late, back-to-back, in schedule order
        self.assertTrue(all(e["state"] == "attempted" for e in es))

    async def test_all_arrivals_missed_and_none_missed(self):
        ft = FakeTime()
        ctx = make_ctx(instant_post, ft.clock, capacity=3)
        for _ in range(3):
            ctx.admission.try_reserve("predict")             # saturate: nothing can be admitted
        await run_stream(ctx, ft, 100.0, 1.0, 2.0)
        s = lg.summarize_stream(entries_of(ctx, "predict_open_loop"), "predict", True, 1.0, 3.0)
        w = s["whole_run"]
        self.assertEqual((w["scheduled_arrivals"], w["attempted"], w["missed"]), (300, 0, 300))
        self.assertEqual(w["missed_by_reason"], {"capacity_exhausted": 300})
        self.assertIsNone(w["success_rate_of_attempted"]["value"])           # 0/0 is undefined, not 100%
        self.assertEqual(w["success_rate_of_scheduled"]["value"], 0.0)
        self.assertEqual(w["latency_attempted_all_terminal"]["n"], 0)
        ctx2 = make_ctx(instant_post, FakeTime().clock, capacity=10)
        ft2 = FakeTime(); ctx2.clock = ft2.clock
        await run_stream(ctx2, ft2, 100.0, 1.0, 2.0)
        w2 = lg.summarize_stream(entries_of(ctx2, "predict_open_loop"), "predict", True, 1.0, 3.0)["whole_run"]
        self.assertEqual((w2["scheduled_arrivals"], w2["attempted"], w2["missed"], w2["success_rate_of_scheduled"]["value"]), (300, 300, 0, 1.0))

    async def test_arrivals_not_reached_before_a_stop_are_recorded_as_missed(self):
        ft = FakeTime()
        stop = asyncio.Event()
        original = ft.sleep_until

        async def stopping(when):
            await original(when)
            if ft.calls == 50:
                stop.set()

        ft.sleep_until = stopping
        ctx = make_ctx(instant_post, ft.clock, capacity=100)
        await run_stream(ctx, ft, 100.0, 1.0, 2.0, stop=stop)
        es = entries_of(ctx, "predict_open_loop")
        self.assertEqual(len(es), 300)                        # every scheduled arrival is represented
        reasons = {e["miss_reason"] for e in es if e["state"] == "missed"}
        self.assertEqual(reasons, {"not_reached_before_stop"})
        self.assertTrue(all(e["sched"] is not None and e["detected"] is not None for e in es if e["state"] == "missed"))
        s = lg.summarize_stream(es, "predict", True, 1.0, 3.0)["whole_run"]
        self.assertEqual(s["scheduled_arrivals"], s["attempted"] + s["missed"])

    def test_duplicate_request_identities_are_rejected(self):
        ledger = lg.Ledger()
        ledger.add("x:1", "x", "predict", 0, 0.0, "warmup")
        with self.assertRaises(ValueError):
            ledger.add("x:1", "x", "predict", 0, 0.0, "warmup")

    def test_workload_and_profiles_are_unchanged(self):
        payload = json.dumps([lg.INPUTS, lg.EXPECTED_PROBABILITY, lg.EXPECTED_DECISION], sort_keys=True)
        self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(), WORKLOAD_FINGERPRINT)
        self.assertEqual(len(lg.INPUTS), 50)
        self.assertEqual(lg.PROFILES, HISTORICAL_PROFILES)


# ===================================================================== capacity
class CapacityTests(unittest.IsolatedAsyncioTestCase):
    async def test_reservation_precedes_task_execution_even_when_tasks_never_start(self):
        ft = FakeTime()
        gate = asyncio.Event()

        async def gated(kind, index):
            await gate.wait()
            return ok_response(kind, index)

        ctx = make_ctx(gated, ft.clock, capacity=10)
        ft.clock.t = 100.0                                        # every arrival is already due: no awaits, no task starts
        tasks = []
        await lg.run_open_loop(ctx, "predict_open_loop", "predict", 100.0, 0.0, 5.0, 0.0, ft.sleep_until,
                               asyncio.get_running_loop().create_task, tasks)
        es = entries_of(ctx, "predict_open_loop")
        self.assertEqual((len(es), len(tasks)), (500, 10))        # only 10 tasks exist although 500 arrivals were due
        self.assertEqual(sum(e["miss_reason"] == "capacity_exhausted" for e in es), 490)
        self.assertEqual(ctx.admission.max_outstanding["predict"], 10)
        self.assertEqual(ctx.admission.violations, 0)
        self.assertTrue(all(e["state"] == "scheduled" for e in es if e["state"] != "missed"))  # none started yet
        gate.set()
        await asyncio.gather(*tasks)
        self.assertEqual((ctx.admission.total_outstanding(), ctx.admission.reserved_total, ctx.admission.released_total), (0, 10, 10))
        self.assertEqual(ctx.admission.double_release_attempts, 0)

    async def test_concurrent_streams_respect_their_own_capacity(self):
        ft = FakeTime()
        gate = asyncio.Event()

        async def gated(kind, index):
            await gate.wait()
            return ok_response(kind, index)

        ctx = make_ctx(gated, ft.clock, capacity={"predict": 7, "explain": 3})
        ft.clock.t = 100.0
        tasks = []
        spawn = asyncio.get_running_loop().create_task
        await asyncio.gather(
            lg.run_open_loop(ctx, "predict_open_loop", "predict", 100.0, 0.0, 2.0, 0.0, ft.sleep_until, spawn, tasks),
            lg.run_open_loop(ctx, "explain_open_loop", "explain", 40.0, 0.0, 2.0, 0.0, ft.sleep_until, spawn, tasks))
        self.assertEqual(dict(ctx.admission.max_outstanding), {"predict": 7, "explain": 3})
        gate.set()
        await asyncio.gather(*tasks)
        self.assertEqual(ctx.admission.violations, 0)
        self.assertEqual(ctx.admission.total_outstanding(), 0)

    async def test_connection_pool_waiters_count_against_generator_capacity(self):
        ft = FakeTime()
        pool = asyncio.Semaphore(2)                  # a 2-connection pool: the rest wait inside post()
        gate = asyncio.Event()

        async def pooled(kind, index):
            async with pool:
                await gate.wait()
            return ok_response(kind, index)

        ctx = make_ctx(pooled, ft.clock, capacity=5)
        ft.clock.t = 100.0
        tasks = []
        await lg.run_open_loop(ctx, "predict_open_loop", "predict", 100.0, 0.0, 1.0, 0.0, ft.sleep_until,
                               asyncio.get_running_loop().create_task, tasks)
        await asyncio.sleep(0)
        self.assertEqual(ctx.admission.outstanding["predict"], 5)    # 2 executing + 3 waiting for the pool are all counted
        gate.set()
        await asyncio.gather(*tasks)
        self.assertEqual(ctx.admission.max_outstanding["predict"], 5)

    async def test_cancellation_before_and_during_execution_releases_exactly_once(self):
        ft = FakeTime()
        gate = asyncio.Event()

        async def gated(kind, index):
            await gate.wait()
            return ok_response(kind, index)

        ctx = make_ctx(gated, ft.clock, capacity=4)
        ft.clock.t = 100.0
        tasks = []
        await lg.run_open_loop(ctx, "predict_open_loop", "predict", 100.0, 0.0, 0.04, 0.0, ft.sleep_until,
                               asyncio.get_running_loop().create_task, tasks)
        self.assertEqual(len(tasks), 4)
        tasks[0].cancel()                                       # cancelled before its first step
        await asyncio.sleep(0)
        for t in tasks[1:]:                                     # now the rest are executing, blocked in post()
            self.assertFalse(t.done())
        tasks[1].cancel()                                       # cancelled during execution
        gate.set()                                              # the remaining two may now finish
        await asyncio.gather(*tasks, return_exceptions=True)
        es = {e["id"]: e for e in entries_of(ctx, "predict_open_loop")}
        self.assertEqual(es["predict_open_loop:0"]["miss_reason"], "cancelled_before_start")
        self.assertEqual(es["predict_open_loop:1"]["outcome"], "cancelled")
        self.assertEqual(sum(1 for e in es.values() if e["state"] == "attempted" and e["outcome"] == "success"), 2)
        self.assertEqual((ctx.admission.total_outstanding(), ctx.admission.double_release_attempts), (0, 0))
        self.assertTrue(all(t.done() for t in tasks))

    async def test_task_creation_failure_releases_the_reservation(self):
        ft = FakeTime()
        ctx = make_ctx(instant_post, ft.clock, capacity=5)
        ft.clock.t = 100.0

        def failing_spawn(coro):
            coro.close()
            raise RuntimeError("cannot create task")

        tasks = []
        await lg.run_open_loop(ctx, "predict_open_loop", "predict", 100.0, 0.0, 0.1, 0.0, ft.sleep_until, failing_spawn, tasks)
        es = entries_of(ctx, "predict_open_loop")
        self.assertEqual({e["miss_reason"] for e in es}, {"task_creation_failed"})
        self.assertEqual((len(es), len(tasks), ctx.admission.total_outstanding()), (10, 0, 0))

    def test_release_is_idempotent_and_counts_attempts(self):
        admission = lg.Admission(2)
        token = admission.try_reserve("predict")
        self.assertTrue(admission.release(token))
        self.assertFalse(admission.release(token))
        self.assertEqual((admission.outstanding["predict"], admission.double_release_attempts), (0, 1))

    def test_capacity_must_be_positive(self):
        with self.assertRaises(ValueError):
            lg.Admission(0)

    async def test_stress_with_delayed_starts_never_oversubscribes(self):
        ft = FakeTime()
        events = []

        async def sometimes_slow(kind, index):
            await asyncio.sleep(0.001 * (index % 5))
            return ok_response(kind, index)

        ctx = make_ctx(sometimes_slow, ft.clock, capacity=8)
        ft.jump_at.update({k: k * 0.01 + 5.0 for k in (50, 120, 200)})   # repeated catch-up bursts
        await run_stream(ctx, ft, 100.0, 1.0, 3.0)
        s = lg.summarize_stream(entries_of(ctx, "predict_open_loop"), "predict", True, 1.0, 4.0)
        self.assertLessEqual(ctx.admission.max_outstanding["predict"], 8)
        self.assertEqual((ctx.admission.violations, ctx.admission.total_outstanding()), (0, 0))
        self.assertTrue(s["whole_run"]["invariants"]["scheduled_equals_attempted_plus_missed"])
        self.assertEqual(s["whole_run"]["unterminated_attempts"], 0)


# ==================================================================== deadlines
class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def run_request(self, post, clock=None, deadline_s=2.0, kind="predict", index=3):
        fake = clock is None
        clock = clock or FakeClock()
        # with the fake clock the task starts at t=0; with the real clock the deadline starts "now"
        return await lg.execute_request(post, kind, index, clock, deadline_s, start=0.0 if fake else None), clock

    async def test_success_before_exactly_at_and_after_the_deadline(self):
        for finish, expected in ((0.0, "success"), (1.999, "success"), (2.0, "client_deadline"), (2.0001, "client_deadline"), (30.0, "client_deadline")):
            with self.subTest(finish=finish):
                clock = FakeClock()

                async def post(i, finish=finish):
                    clock.t = finish
                    return ok_response("predict", i)

                (outcome, status, code, reason, end), _ = await self.run_request(post, clock)
                self.assertEqual(outcome, expected)
                if expected == "client_deadline":
                    self.assertEqual(code, "late_completion")
                    self.assertEqual(status, 200)               # the response existed; it simply finished too late

    async def test_expiration_during_synchronous_validation_is_late(self):
        clock = FakeClock()
        original = lg.classify_response

        def slow_validation(*args, **kwargs):
            result = original(*args, **kwargs)
            clock.t = 5.0                                       # validation itself consumed the budget
            return result

        async def post(i):
            clock.t = 0.5
            return ok_response("predict", i)

        with mock.patch.object(lg, "classify_response", slow_validation):
            (outcome, _, code, reason, _), _ = await self.run_request(post, clock)
        self.assertEqual((outcome, code), ("client_deadline", "late_completion"))
        self.assertIn("success", reason)

    async def test_hung_connection_acquisition_hits_the_total_deadline(self):
        started = asyncio.Event()
        cancelled = []

        async def post(i):
            started.set()
            try:
                await asyncio.Event().wait()                    # never obtains a connection
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        (outcome, status, code, reason, _), _ = await self.run_request(post, clock=lg_monotonic(), deadline_s=0.05)
        self.assertEqual((outcome, status, code), ("client_deadline", None, "client_total_deadline"))
        self.assertEqual(cancelled, [True])                     # cancellation-safe: the in-flight work was cancelled

    async def test_several_short_phases_cannot_exceed_the_total_deadline(self):
        phases = []

        async def post(i):
            for _ in range(4):                                  # each phase is well below any per-phase timeout
                await asyncio.sleep(0.03)
                phases.append(1)
            return ok_response("predict", i)

        (outcome, _, code, _, end), _ = await self.run_request(post, clock=lg_monotonic(), deadline_s=0.08)
        self.assertEqual((outcome, code), ("client_deadline", "client_total_deadline"))
        self.assertLess(len(phases), 4)                         # stopped part-way: the budget was NOT reset between phases

    async def test_slow_body_and_connect_delay_share_one_budget(self):
        async def post(i):
            await asyncio.sleep(0.04)                           # "pool acquisition + connect"
            await asyncio.sleep(0.04)                           # "slow body"
            return ok_response("predict", i)

        (outcome, *_), _ = await self.run_request(post, clock=lg_monotonic(), deadline_s=0.07)
        self.assertEqual(outcome, "client_deadline")
        (outcome, *_), _ = await self.run_request(post, clock=lg_monotonic(), deadline_s=0.5)
        self.assertEqual(outcome, "success")

    async def test_server_504_is_an_http_outcome_not_a_client_deadline(self):
        async def post(i):
            return error_response("explanation_deadline_exceeded", 504, None)

        (outcome, status, code, _, _), _ = await self.run_request(post, kind="explain")
        self.assertEqual((outcome, status, code), ("http_error", 504, "explanation_deadline_exceeded"))

    async def test_phase_timeout_and_transport_errors_are_classified(self):
        async def phase(i):
            raise httpx.ReadTimeout("slow")

        async def refused(i):
            raise httpx.ConnectError("refused")

        (outcome, _, code, _, _), _ = await self.run_request(phase)
        self.assertEqual((outcome, code), ("client_deadline", "client_phase_timeout"))
        (outcome, _, code, _, _), _ = await self.run_request(refused)
        self.assertEqual((outcome, code), ("transport_failure", "connection_failure"))

    async def test_cancellation_during_an_attempt_is_recorded_and_cleans_up(self):
        ctx = make_ctx(None, FakeClock(), capacity=2)
        blocked = asyncio.Event()

        async def post(kind, index):
            blocked.set()
            await asyncio.Event().wait()

        ctx.post = post
        entry = ctx.ledger.add("x:0", "x", "predict", 0, None, "measurement")
        token = ctx.admission.try_reserve("predict")
        task = asyncio.create_task(lg.attempt(ctx, entry, token))
        await blocked.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((entry["outcome"], ctx.admission.total_outstanding(), ctx.admission.double_release_attempts), ("cancelled", 0, 0))


def lg_monotonic():
    import time
    return time.monotonic


# ================================================================ outcome accounting
class OutcomeAccountingTests(unittest.TestCase):
    def classify(self, kind, index, resp):
        return lg.classify_response(kind, index, resp.status_code, resp.headers, resp.text)[0]

    def test_every_terminal_category(self):
        i = 7
        self.assertEqual(self.classify("predict", i, ok_response("predict", i)), "success")
        self.assertEqual(self.classify("explain", i, ok_response("explain", i)), "success")
        self.assertEqual(self.classify("explain", i, error_response("explanation_overloaded", 503, "1")), "http_error")
        self.assertEqual(self.classify("explain", i, error_response("explanation_unavailable", 503, "5")), "http_error")
        self.assertEqual(self.classify("explain", i, Resp(422, {"detail": [{"msg": "x"}]})), "http_error")
        self.assertEqual(self.classify("predict", i, Resp(200, "{not json")), "invalid_json")
        self.assertEqual(self.classify("predict", i, Resp(200, '{"probability": NaN}')), "invalid_json")
        bad = dict(predict_body(i), extra=1)
        self.assertEqual(self.classify("predict", i, Resp(200, bad)), "contract_violation")
        self.assertEqual(self.classify("predict", i + 1, ok_response("predict", i)), "association_violation")   # another input's result
        self.assertEqual(self.classify("explain", i, Resp(200, explain_body(i), {})), "contract_violation")      # missing no-store
        self.assertEqual(self.classify("explain", i + 1, ok_response("explain", i)), "association_violation")
        self.assertEqual(self.classify("explain", i, error_response("explanation_overloaded", 503, None)), "contract_violation")  # missing Retry-After
        leaky = Resp(200, dict(explain_body(i), note="subtype"), {"cache-control": "no-store"})
        self.assertEqual(self.classify("explain", i, leaky), "contract_violation")

    def entries(self):
        ledger = lg.Ledger()
        out = []
        for n, (cohort, outcome, end) in enumerate([("measurement", "success", 5.0), ("measurement", "success", 5.5), ("measurement", "http_error", 6.0),
                                                     ("measurement", "client_deadline", 9.0), ("warmup", "success", 3.0)]):
            e = ledger.add(f"s:{n}", "s", "predict", 0, float(n), cohort)
            ledger.attempt(e, float(n) + 0.01)
            ledger.finish(e, end, outcome, 200 if outcome == "success" else 503, None)
            out.append(e)
        m = ledger.add("s:99", "s", "predict", 0, 4.0, "measurement")
        ledger.miss(m, "capacity_exhausted", 4.2)
        return out + [m]

    def test_outstanding_drain_and_cohort_assignment(self):
        s = lg.summarize_stream(self.entries(), "predict", True, 4.0, 8.0)
        meas = s["measurement"]
        self.assertEqual((meas["scheduled_arrivals"], meas["attempted"], meas["missed"]), (5, 4, 1))
        self.assertEqual(meas["outstanding_at_measurement_end"], 1)        # completed at 9.0, after the window end 8.0
        self.assertEqual(meas["drain_duration_s"], 1.0)
        self.assertEqual(s["warmup"]["attempted"], 1)                     # a warm-up request stays warm-up
        self.assertEqual(meas["outcomes"]["client_deadline"], 1)
        self.assertEqual(meas["completion_window_throughput_per_s"], 0.75)  # 3 completions in [4, 8)
        self.assertTrue(all(all(b["invariants"].values()) for b in (meas, s["warmup"], s["whole_run"])))

    def test_missed_arrivals_are_excluded_from_latency_but_included_in_offered_success(self):
        s = lg.summarize_stream(self.entries(), "predict", True, 4.0, 8.0)["measurement"]
        self.assertEqual(s["latency_attempted_all_terminal"]["n"], 4)
        self.assertEqual(s["latency_attempted_successful"]["n"], 2)
        self.assertEqual(s["success_rate_of_attempted"]["denominator"], 4)
        self.assertEqual(s["success_rate_of_scheduled"]["denominator"], 5)
        self.assertEqual(s["success_rate_of_scheduled"]["value"], 0.4)
        self.assertEqual(s["missed_detection_lag"]["n"], 1)

    def test_empty_workload_has_null_rates_and_samples(self):
        s = lg.summarize_stream([], "predict", True, 0.0, 1.0)["whole_run"]
        self.assertIsNone(s["success_rate_of_attempted"]["value"])
        self.assertIsNone(s["success_rate_of_scheduled"]["value"])
        self.assertEqual(s["latency_attempted_successful"], {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None, "mean_ms": None})

    def test_closed_loop_has_no_offered_arrival_denominator(self):
        ledger = lg.Ledger()
        e = ledger.add("c:0", "c", "explain", 0, None, "measurement")
        ledger.attempt(e, 5.0)
        ledger.finish(e, 5.1, "success", 200)
        block = lg.summarize_stream([e], "explain", False, 4.0, 8.0)["measurement"]
        self.assertIsNone(block["scheduled_arrivals"])
        self.assertIsNone(block["success_rate_of_scheduled"]["value"])
        self.assertEqual(block["success_rate_of_attempted"]["value"], 1.0)

    def test_percentiles_on_known_distributions(self):
        self.assertEqual(lg.percentile([1, 2, 3, 4, 5], .5), 3)
        self.assertAlmostEqual(lg.percentile([1, 2, 3, 4, 5], .95), 4.8)
        self.assertEqual(lg.percentile([7], .99), 7)
        self.assertIsNone(lg.percentile([], .5))
        stats = lg.latency_stats([0.001 * k for k in range(1, 101)])
        self.assertEqual((stats["n"], stats["p50_ms"], stats["max_ms"]), (100, 50.5, 100.0))
        self.assertEqual(lg.rate(0, 0, "x")["value"], None)
        self.assertEqual(lg.rate(1, 4, "x")["value"], 0.25)


class HistoricalFixtureTests(unittest.TestCase):
    """P2 repetition 3 arithmetic reproduced from synthetic records (the historical run is not repeated)."""

    def test_historical_p2_repetition_3_denominators(self):
        ledger = lg.Ledger()
        for k in range(1000):                                                   # warm-up: all attempted and successful
            e = ledger.add(f"p:{k}", "p", "predict", 0, k / 100, "warmup")
            ledger.attempt(e, k / 100)
            ledger.finish(e, k / 100 + 0.005, "success", 200)
        for k in range(1000, 4000):                                             # measurement: 2,642 successful, 358 missed
            e = ledger.add(f"p:{k}", "p", "predict", 0, k / 100, "measurement")
            if k < 1000 + 2642:
                ledger.attempt(e, k / 100)
                ledger.finish(e, k / 100 + 0.005, "success", 200)
            else:
                ledger.miss(e, "capacity_exhausted", k / 100)
        s = lg.summarize_stream(list(ledger.entries.values()), "predict", True, 10.0, 40.0)
        whole, meas = s["whole_run"], s["measurement"]
        self.assertEqual((whole["scheduled_arrivals"], whole["attempted"], whole["successful"], whole["missed"]), (4000, 3642, 3642, 358))
        self.assertEqual((meas["scheduled_arrivals"], meas["attempted"], meas["successful"], meas["missed"]), (3000, 2642, 2642, 358))
        self.assertAlmostEqual(whole["success_rate_of_scheduled"]["value"], 0.9105, places=6)
        self.assertAlmostEqual(meas["success_rate_of_scheduled"]["value"], 0.880667, places=6)
        self.assertEqual((whole["success_rate_of_scheduled"]["denominator"], meas["success_rate_of_scheduled"]["denominator"]), (4000, 3000))
        self.assertNotEqual(whole["success_rate_of_scheduled"]["denominator_is"], "")
        # the two rates live in different, separately labelled cohort blocks: the measurement rate cannot be read against 4,000
        self.assertIsNot(whole["success_rate_of_scheduled"], meas["success_rate_of_scheduled"])
        self.assertEqual((s["warmup"]["scheduled_arrivals"], s["warmup"]["success_rate_of_scheduled"]["value"]), (1000, 1.0))
        for block in (whole, meas, s["warmup"]):
            self.assertTrue(all(block["invariants"].values()))


# ============================================================ real sockets (loopback)
class LoopbackServer:
    """A controlled in-test HTTP/1.1 server (keep-alive). ``behavior(kind, idx, n)`` returns
    (delay_s, action, response_or_None); action in {'send', 'drop'}."""

    def __init__(self, behavior):
        self.behavior, self.server = behavior, None
        self.requests = self.open = self.max_open = 0
        self.disconnects_during_delay = 0

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self):
        self.server.close()
        await self.server.wait_closed()

    async def handle(self, reader, writer):
        self.open += 1
        self.max_open = max(self.max_open, self.open)
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode().split("\r\n")
                path = lines[0].split()[1]
                length = next(int(l.split(":")[1]) for l in lines if l.lower().startswith("content-length"))
                body = json.loads((await reader.readexactly(length)).decode())
                idx = next(i for i, v in enumerate(lg.INPUTS) if v == body)
                kind = "predict" if path == "/predict" else "explain"
                n = self.requests
                self.requests += 1
                delay, action, resp = self.behavior(kind, idx, n)
                if delay:
                    waiter = asyncio.ensure_future(reader.read(1))      # EOF while delaying = the client gave up
                    done, _ = await asyncio.wait({waiter}, timeout=delay)
                    if done:
                        self.disconnects_during_delay += 1
                        return
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)  # let the reader settle before the next request
                if action == "drop":
                    return
                data = resp.text.encode()
                headers = {"content-type": "application/json", "content-length": str(len(data)), **resp.headers}
                writer.write((f"HTTP/1.1 {resp.status_code} X\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n").encode() + data)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.open -= 1
            writer.close()


class SocketIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.servers = []

    async def asyncTearDown(self):
        for s in self.servers:
            await s.stop()

    async def serve(self, behavior):
        server = LoopbackServer(behavior)
        await server.start()
        self.servers.append(server)
        return server

    def client(self, server, max_connections=128, phase=5.0):
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        return httpx.AsyncClient(base_url=f"http://127.0.0.1:{server.port}", limits=limits, timeout=httpx.Timeout(phase))

    def context(self, client, deadline_s, capacity=100):
        async def post(kind, index):
            return await client.post("/predict" if kind == "predict" else "/v1/explanations", json=lg.INPUTS[index])

        return make_ctx(post, __import__("time").monotonic, deadline_s, capacity)

    async def attempt_all(self, ctx, kind, indices):
        tasks = []
        for n, i in enumerate(indices):
            entry = ctx.ledger.add(f"{kind}:{n}", kind, kind, i, float(n), "measurement")
            token = ctx.admission.try_reserve(kind)
            self.assertIsNotNone(token)
            tasks.append(asyncio.create_task(lg.attempt(ctx, entry, token)))
        await asyncio.gather(*tasks, return_exceptions=True)
        return entries_of(ctx, kind)

    async def test_mixed_outcomes_over_real_sockets(self):
        def behavior(kind, idx, n):
            r = idx % 5
            if r == 0:
                return 0, "send", error_response("explanation_overloaded", 503, "1")
            if r == 1:
                return 0, "send", error_response("explanation_deadline_exceeded", 504, None)
            if r == 2:
                return 1.5, "send", ok_response(kind, idx)               # slower than the client's total deadline
            if r == 3:
                return 0, "drop", None
            return 0, "send", ok_response(kind, idx)

        server = await self.serve(behavior)
        async with self.client(server) as client:
            ctx = self.context(client, 0.6)
            es = await self.attempt_all(ctx, "explain", range(25))
        counts = {o: sum(1 for e in es if e["outcome"] == o) for o in lg.OUTCOMES}
        self.assertEqual((counts["success"], counts["http_error"], counts["client_deadline"], counts["transport_failure"]), (5, 10, 5, 5))
        self.assertEqual({e["status"] for e in es if e["outcome"] == "http_error"}, {503, 504})   # 504 is an HTTP outcome
        self.assertEqual(ctx.admission.total_outstanding(), 0)
        block = lg.summarize_stream(es, "explain", False, 0.0, 1.0)["measurement"]
        self.assertTrue(all(block["invariants"].values()))

    async def test_total_deadline_is_enforced_and_the_server_sees_the_client_give_up(self):
        server = await self.serve(lambda kind, idx, n: (0.8, "send", ok_response(kind, idx)))
        async with self.client(server, phase=5.0) as client:       # phase timeouts are far looser than the total deadline
            ctx = self.context(client, 0.15)
            loop = asyncio.get_running_loop()
            started = loop.time()
            es = await self.attempt_all(ctx, "predict", [3])
            elapsed = loop.time() - started
        self.assertEqual((es[0]["outcome"], es[0]["code"]), ("client_deadline", "client_total_deadline"))
        self.assertLess(elapsed, 0.6)
        await asyncio.sleep(0.05)
        self.assertEqual(server.disconnects_during_delay, 1)

    async def test_pool_contention_counts_against_the_total_deadline(self):
        server = await self.serve(lambda kind, idx, n: (0.2, "send", ok_response(kind, idx)))
        async with self.client(server, max_connections=2) as client:
            ctx = self.context(client, 0.6)
            es = await self.attempt_all(ctx, "predict", range(6))     # 3 waves of 2 over a 2-connection pool: ~0.27/0.47/0.67 s
        outcomes = sorted(e["outcome"] for e in es)
        # exact split depends on host speed; the invariants below do not: a 2-connection pool cannot serve 6 requests of
        # 0.2 s within 0.6 s, so the queued ones must fail on the TOTAL deadline, and the first wave must succeed
        self.assertEqual(outcomes.count("success") + outcomes.count("client_deadline"), 6)
        self.assertGreaterEqual(outcomes.count("success"), 2)
        self.assertGreaterEqual(outcomes.count("client_deadline"), 1)
        self.assertTrue(all(e["code"] == "client_total_deadline" for e in es if e["outcome"] == "client_deadline"))
        self.assertLessEqual(server.max_open, 2)

    async def test_open_loop_over_sockets_is_bounded_and_fully_accounted(self):
        server = await self.serve(lambda kind, idx, n: (0.25, "send", ok_response(kind, idx)))
        async with self.client(server) as client:
            ctx = self.context(client, 1.0, capacity=5)
            tasks = []
            loop = asyncio.get_running_loop()

            async def sleep_until(when):
                await asyncio.sleep(max(0.0, when - loop.time()))

            ctx.clock = loop.time
            await lg.run_open_loop(ctx, "predict_open_loop", "predict", 200.0, 0.1, 0.4, loop.time(), sleep_until, loop.create_task, tasks)
            await asyncio.gather(*tasks)
        es = entries_of(ctx, "predict_open_loop")
        s = lg.summarize_stream(es, "predict", True, 0.0, 0.5)["whole_run"]
        self.assertEqual(s["scheduled_arrivals"], 100)
        self.assertEqual(s["scheduled_arrivals"], s["attempted"] + s["missed"])
        self.assertGreater(s["missed_by_reason"].get("capacity_exhausted", 0), 0)
        self.assertLessEqual(ctx.admission.max_outstanding["predict"], 5)
        self.assertEqual((ctx.admission.violations, ctx.admission.total_outstanding(), s["unterminated_attempts"]), (0, 0, 0))
        self.assertTrue(all(s["invariants"].values()))
        self.assertTrue(all(t.done() for t in tasks))

    async def test_cancelling_a_run_mid_flight_cleans_up_over_sockets(self):
        server = await self.serve(lambda kind, idx, n: (5.0, "send", ok_response(kind, idx)))
        baseline = {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}
        async with self.client(server) as client:
            ctx = self.context(client, 10.0, capacity=20)
            loop = asyncio.get_running_loop()
            ctx.clock = loop.time
            tasks = []

            async def sleep_until(when):
                await asyncio.sleep(max(0.0, when - loop.time()))

            runner = asyncio.create_task(lg.run_open_loop(ctx, "predict_open_loop", "predict", 50.0, 0.0, 2.0, loop.time(), sleep_until,
                                                           loop.create_task, tasks))
            await asyncio.sleep(0.15)
            runner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await runner
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(ctx.admission.total_outstanding(), 0)
        self.assertEqual(ctx.admission.double_release_attempts, 0)
        es = entries_of(ctx, "predict_open_loop")
        self.assertTrue(es and all(e["state"] == "attempted" and e["outcome"] == "cancelled" or e["state"] == "missed" for e in es))
        await asyncio.sleep(0.05)
        self.assertEqual({t for t in asyncio.all_tasks() if t is not asyncio.current_task()} - baseline, set())


if __name__ == "__main__":
    unittest.main()
