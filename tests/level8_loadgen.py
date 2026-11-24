#!/usr/bin/env python3
"""Level 8 mixed-load generator (test harness; not part of the service). Summary schema version 2.

Open-loop scheduled arrivals and closed-loop concurrent callers against /predict and
/v1/explanations, with the fixed, deterministic collection of 50 synthetic eligible inputs
(40 development-range draws and 10 near-threshold draws, numpy default_rng(42) via
tests/level8_oracle.build_grids; expected prediction results were computed once from the
repository-layout Level 7 runtime and are embedded below).

Measurement contract (version 2; the historical L8-2D harness differs, see
verification/level8/loadgen_verification.json):

* SCHEDULED ARRIVALS are counted from an independent deterministic schedule, arrival k at
  t_start + k/rate on a monotonic clock, never from issued requests. Cohorts use half-open
  intervals [0, warmup) and [warmup, warmup + measure); an arrival keeps the cohort of its SCHEDULED
  time wherever it completes. For 100/s with 10 s + 30 s the counts are exactly 1,000 + 3,000 = 4,000.
* LATE-ARRIVAL POLICY (unchanged in substance from the historical harness, now explicit): a late
  arrival is issued as soon as the scheduler resumes, in schedule order, and its lag is recorded. There
  is no lateness cutoff and no unbounded burst: every arrival must first reserve generator capacity
  synchronously. An arrival that cannot (capacity_exhausted), whose task cannot be created, that is
  cancelled before its task starts, or that the scheduler never reached before stopping becomes a MISSED
  arrival, recorded with identity, scheduled and detection times, cohort and a fixed reason code.
* GENERATOR CAPACITY (tasks reserved + not yet started + waiting for a pooled connection + executing,
  per traffic kind, --max-backlog) is distinct from HTTP connection-pool capacity (--max-connections)
  and from the server's dispatcher capacity.
* TOTAL CLIENT DEADLINE: each attempted request has one absolute deadline (--timeout, default 2 s) from
  actual task start covering pool acquisition, connection, transmission, response body, JSON parsing and
  validation. httpx phase timeouts equal the same value and are subordinate. Any response completed at or
  after the deadline is a client_deadline outcome. A server HTTP 504 received in time is an http_error.
  Synchronous parsing/validation cannot be interrupted; the deadline is checked after it. Under event-loop
  starvation this is an application-level deadline, not a wall-clock guarantee.
* TERMINAL OUTCOMES of an attempted request are mutually exclusive: success, http_error, transport_failure,
  client_deadline, invalid_json, contract_violation, association_violation, cancelled.
* Every rate states its denominator; undefined rates (zero denominator) are null.
Percentiles use linear interpolation between closest ranks (numpy's default); empty samples give n = 0 and
null statistics. Raw input values are never written to the summary.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import os
import sys
import time
from collections import Counter

import httpx

INPUTS = [{"weekly_km_to_date": 39.34890121389908, "sleep_hours_last_night": 6.597196099380131, "resting_hr_morning": 55.30317503893659, "soreness_score_evening": 6.973680290593639, "prior_injury": 1}, {"weekly_km_to_date": 44.3905587909189, "sleep_hours_last_night": 7.402849254975882, "resting_hr_morning": 54.43277166332345, "soreness_score_evening": 1.2811363267554587, "prior_injury": 0}, {"weekly_km_to_date": 29.26995060581453, "sleep_hours_last_night": 7.816912472121505, "resting_hr_morning": 52.72638144096798, "soreness_score_evening": 8.227616132708299, "prior_injury": 0}, {"weekly_km_to_date": 25.680968044619423, "sleep_hours_last_night": 6.886461967539587, "resting_hr_morning": 45.7658070732501, "soreness_score_evening": 8.27631171992582, "prior_injury": 0}, {"weekly_km_to_date": 38.95219350213435, "sleep_hours_last_night": 6.386314920324671, "resting_hr_morning": 56.64837629273884, "soreness_score_evening": 8.931211213221976, "prior_injury": 0}, {"weekly_km_to_date": 24.86596769629919, "sleep_hours_last_night": 6.666802509317586, "resting_hr_morning": 45.52564518944675, "soreness_score_evening": 1.5428949206754783, "prior_injury": 1}, {"weekly_km_to_date": 38.61905389769543, "sleep_hours_last_night": 7.918774331085525, "resting_hr_morning": 48.909904297657825, "soreness_score_evening": 3.704597060348689, "prior_injury": 0}, {"weekly_km_to_date": 24.736783977107144, "sleep_hours_last_night": 5.824803763338679, "resting_hr_morning": 50.708459114711204, "soreness_score_evening": 2.2690934905088413, "prior_injury": 0}, {"weekly_km_to_date": 30.92879797180827, "sleep_hours_last_night": 7.581695490144593, "resting_hr_morning": 53.40318122402699, "soreness_score_evening": 3.123666413820411, "prior_injury": 0}, {"weekly_km_to_date": 40.11910893742005, "sleep_hours_last_night": 6.468695947575436, "resting_hr_morning": 48.45993724716293, "soreness_score_evening": 6.824955039749755, "prior_injury": 0}, {"weekly_km_to_date": 24.99770506187771, "sleep_hours_last_night": 5.518405674377513, "resting_hr_morning": 54.44309253002566, "soreness_score_evening": 6.648508565920322, "prior_injury": 0}, {"weekly_km_to_date": 39.5182257755492, "sleep_hours_last_night": 6.64728943884585, "resting_hr_morning": 51.82489435143472, "soreness_score_evening": 1.3979699812765745, "prior_injury": 0}, {"weekly_km_to_date": 36.71007404476179, "sleep_hours_last_night": 6.677740515357831, "resting_hr_morning": 51.782833277774266, "soreness_score_evening": 7.649988574160256, "prior_injury": 1}, {"weekly_km_to_date": 33.8394850164499, "sleep_hours_last_night": 6.898017901863534, "resting_hr_morning": 48.647401176751345, "soreness_score_evening": 0.30817834567939406, "prior_injury": 1}, {"weekly_km_to_date": 25.36461682048823, "sleep_hours_last_night": 6.52132160931159, "resting_hr_morning": 55.240836879217994, "soreness_score_evening": 2.3393948586534075, "prior_injury": 0}, {"weekly_km_to_date": 27.034597300549912, "sleep_hours_last_night": 6.233984394416709, "resting_hr_morning": 52.94299817672274, "soreness_score_evening": 5.570321523412783, "prior_injury": 0}, {"weekly_km_to_date": 36.607838508184685, "sleep_hours_last_night": 6.515967153600176, "resting_hr_morning": 54.768244615992415, "soreness_score_evening": 1.669729199077039, "prior_injury": 1}, {"weekly_km_to_date": 22.251196519391044, "sleep_hours_last_night": 7.305898376491125, "resting_hr_morning": 50.542526763016646, "soreness_score_evening": 1.612717790336018, "prior_injury": 0}, {"weekly_km_to_date": 23.80780256782921, "sleep_hours_last_night": 7.24080093769434, "resting_hr_morning": 50.35387530688837, "soreness_score_evening": 3.810212260964825, "prior_injury": 0}, {"weekly_km_to_date": 35.757064827972215, "sleep_hours_last_night": 6.404531526383476, "resting_hr_morning": 46.05179903179321, "soreness_score_evening": 1.1800590212051532, "prior_injury": 0}, {"weekly_km_to_date": 42.71451726769018, "sleep_hours_last_night": 7.249267834526874, "resting_hr_morning": 48.19043953751424, "soreness_score_evening": 9.69176377347724, "prior_injury": 1}, {"weekly_km_to_date": 37.92225472897489, "sleep_hours_last_night": 6.623403755359472, "resting_hr_morning": 48.266898742141905, "soreness_score_evening": 0.9639096215349929, "prior_injury": 0}, {"weekly_km_to_date": 31.394407245840277, "sleep_hours_last_night": 6.0059084119880755, "resting_hr_morning": 48.67147948980783, "soreness_score_evening": 5.79219568941896, "prior_injury": 0}, {"weekly_km_to_date": 41.415357102309386, "sleep_hours_last_night": 7.396298824588025, "resting_hr_morning": 53.63355547141124, "soreness_score_evening": 4.320930397751037, "prior_injury": 0}, {"weekly_km_to_date": 34.60244922281839, "sleep_hours_last_night": 7.12461650388705, "resting_hr_morning": 46.01333185367867, "soreness_score_evening": 4.1580740217060965, "prior_injury": 0}, {"weekly_km_to_date": 32.34977048111297, "sleep_hours_last_night": 6.324653030831963, "resting_hr_morning": 46.73429026639256, "soreness_score_evening": 1.0340296772255164, "prior_injury": 1}, {"weekly_km_to_date": 24.264824213422152, "sleep_hours_last_night": 7.812800295941993, "resting_hr_morning": 51.972733676404744, "soreness_score_evening": 3.4686980453483707, "prior_injury": 0}, {"weekly_km_to_date": 20.57009677574244, "sleep_hours_last_night": 7.896398033103614, "resting_hr_morning": 50.7876412433148, "soreness_score_evening": 7.827352272502862, "prior_injury": 0}, {"weekly_km_to_date": 32.16645827095401, "sleep_hours_last_night": 6.7267674858863025, "resting_hr_morning": 56.25391745969979, "soreness_score_evening": 5.717280523760754, "prior_injury": 0}, {"weekly_km_to_date": 26.67439157729734, "sleep_hours_last_night": 6.32892249335638, "resting_hr_morning": 51.24806882965845, "soreness_score_evening": 4.389114603050467, "prior_injury": 0}, {"weekly_km_to_date": 40.657298104858945, "sleep_hours_last_night": 7.740401929599416, "resting_hr_morning": 46.68298906798333, "soreness_score_evening": 5.540361435390494, "prior_injury": 0}, {"weekly_km_to_date": 36.806002325995294, "sleep_hours_last_night": 6.2030844595975205, "resting_hr_morning": 52.91307161630282, "soreness_score_evening": 7.269946142868826, "prior_injury": 0}, {"weekly_km_to_date": 22.693523648897415, "sleep_hours_last_night": 7.79002961284402, "resting_hr_morning": 47.762567890738566, "soreness_score_evening": 0.3741255617617978, "prior_injury": 0}, {"weekly_km_to_date": 29.27305709656097, "sleep_hours_last_night": 7.574474357831033, "resting_hr_morning": 54.699017664771624, "soreness_score_evening": 3.1713889282271532, "prior_injury": 1}, {"weekly_km_to_date": 27.272945953502965, "sleep_hours_last_night": 6.7876428230792865, "resting_hr_morning": 48.071581086811236, "soreness_score_evening": 9.360435700489633, "prior_injury": 0}, {"weekly_km_to_date": 21.122765484808223, "sleep_hours_last_night": 6.587742650007595, "resting_hr_morning": 56.908506768670044, "soreness_score_evening": 8.916772662549139, "prior_injury": 0}, {"weekly_km_to_date": 42.26981227196312, "sleep_hours_last_night": 7.733616599244658, "resting_hr_morning": 51.22630032463739, "soreness_score_evening": 3.1592905183079303, "prior_injury": 0}, {"weekly_km_to_date": 36.54153157919403, "sleep_hours_last_night": 6.434144322184276, "resting_hr_morning": 46.133600016738185, "soreness_score_evening": 7.46789611349026, "prior_injury": 0}, {"weekly_km_to_date": 43.420328763344486, "sleep_hours_last_night": 6.102426437514212, "resting_hr_morning": 46.47309518893783, "soreness_score_evening": 8.311126721249062, "prior_injury": 1}, {"weekly_km_to_date": 24.481707703943478, "sleep_hours_last_night": 6.998456978802109, "resting_hr_morning": 55.49474449004957, "soreness_score_evening": 1.9643466571457324, "prior_injury": 0}, {"weekly_km_to_date": 22.810292151908182, "sleep_hours_last_night": 6.616996245208167, "resting_hr_morning": 47.162472910099595, "soreness_score_evening": 0.4377511394462086, "prior_injury": 0}, {"weekly_km_to_date": 23.60134398441407, "sleep_hours_last_night": 7.490170245560416, "resting_hr_morning": 53.644553781562024, "soreness_score_evening": 4.305065756034084, "prior_injury": 0}, {"weekly_km_to_date": 28.743393690509897, "sleep_hours_last_night": 7.8278801592588465, "resting_hr_morning": 49.82034302186699, "soreness_score_evening": 6.6742954804079115, "prior_injury": 0}, {"weekly_km_to_date": 24.5250707964297, "sleep_hours_last_night": 6.589542397186731, "resting_hr_morning": 46.08935581589583, "soreness_score_evening": 1.4038470334306197, "prior_injury": 0}, {"weekly_km_to_date": 37.21267389759654, "sleep_hours_last_night": 6.164478563136612, "resting_hr_morning": 49.209579863734376, "soreness_score_evening": 0.4365659740607897, "prior_injury": 0}, {"weekly_km_to_date": 32.37653327399095, "sleep_hours_last_night": 7.0995411020812575, "resting_hr_morning": 50.691594347893584, "soreness_score_evening": 4.124843968311625, "prior_injury": 0}, {"weekly_km_to_date": 30.693339347384715, "sleep_hours_last_night": 7.151428150705447, "resting_hr_morning": 48.45711960815063, "soreness_score_evening": 4.842737468731122, "prior_injury": 0}, {"weekly_km_to_date": 34.041570500666765, "sleep_hours_last_night": 7.318966409906522, "resting_hr_morning": 53.46787265962093, "soreness_score_evening": 1.4818059991052224, "prior_injury": 0}, {"weekly_km_to_date": 36.28232247796016, "sleep_hours_last_night": 6.986363457406426, "resting_hr_morning": 49.81051398954644, "soreness_score_evening": 3.5927381559991654, "prior_injury": 0}, {"weekly_km_to_date": 27.647742802312486, "sleep_hours_last_night": 6.504738861600465, "resting_hr_morning": 47.51851701812943, "soreness_score_evening": 1.4279966929963595, "prior_injury": 0}]
EXPECTED_PROBABILITY = [0.9816159009933472, 0.6466301679611206, 0.7094998955726624, 0.8194674253463745, 0.9685506820678711, 0.6138080358505249, 0.2796778380870819, 0.7853928208351135, 0.44625648856163025, 0.7948381900787354, 0.8955455422401428, 0.590545117855072, 0.9773354530334473, 0.37084758281707764, 0.7262565493583679, 0.7773712873458862, 0.8762093782424927, 0.4122971296310425, 0.5238415598869324, 0.2760106325149536, 0.9612578749656677, 0.33763587474823, 0.8151800036430359, 0.7858611345291138, 0.33266523480415344, 0.5786371231079102, 0.34359923005104065, 0.5661159157752991, 0.8547719717025757, 0.7328266501426697, 0.3431176245212555, 0.9133239984512329, 0.17729239165782928, 0.7881953120231628, 0.8630995750427246, 0.8990666270256042, 0.47743210196495056, 0.7973150014877319, 0.964302122592926, 0.6401840448379517, 0.500108003616333, 0.5001168847084045, 0.5001314282417297, 0.4998354911804199, 0.5002163648605347, 0.4997680187225342, 0.5002896785736084, 0.5003315806388855, 0.49964383244514465, 0.5004061460494995]
EXPECTED_DECISION = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
FEATURES = ("weekly_km_to_date", "sleep_hours_last_night", "resting_hr_morning",
            "soreness_score_evening", "prior_injury")
PROBABILITY_TOLERANCE = 2e-6
SUMMARY_SCHEMA = "level8-loadgen-summary/2"
PROFILES = {
    "P0": dict(predict_rate=100.0, explain_callers=0, explain_rate=0.0, warmup=10.0, measure=30.0),
    "P1": dict(predict_rate=100.0, explain_callers=4, explain_rate=0.0, warmup=10.0, measure=30.0),
    "P2": dict(predict_rate=100.0, explain_callers=0, explain_rate=40.0, warmup=10.0, measure=30.0),
    "P3": dict(predict_rate=0.0, explain_callers=4, explain_rate=0.0, warmup=10.0, measure=30.0),
    "P4": dict(predict_rate=100.0, explain_callers=0, explain_rate=0.0, warmup=10.0, measure=10.0,
               burst=32, burst_at=1.0, bucket=1.0),
    "P5": dict(predict_rate=100.0, explain_callers=4, explain_rate=0.0, warmup=10.0, measure=300.0, bucket=30.0),
}
OUTCOMES = ("success", "http_error", "transport_failure", "client_deadline", "invalid_json",
            "contract_violation", "association_violation", "cancelled")
MISS_REASONS = ("capacity_exhausted", "task_creation_failed", "cancelled_before_start", "not_reached_before_stop")
COHORTS = ("warmup", "measurement")


# ------------------------------------------------------------------------------- statistics
def percentile(sorted_values, q):
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def latency_stats(values_s):
    """Milliseconds. Empty sample: n = 0 and null statistics (never a fabricated zero)."""
    if not values_s:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None, "mean_ms": None}
    v = sorted(values_s)
    ms = lambda x: round(x * 1000.0, 3)
    return {"n": len(v), "p50_ms": ms(percentile(v, .5)), "p95_ms": ms(percentile(v, .95)),
            "p99_ms": ms(percentile(v, .99)), "max_ms": ms(v[-1]), "mean_ms": ms(sum(v) / len(v))}


def rate(numerator, denominator, label):
    """A rate that always names its denominator; null when the denominator is zero."""
    return {"value": (round(numerator / denominator, 6) if denominator else None), "numerator": numerator,
            "denominator": denominator, "denominator_is": label}


# ------------------------------------------------------------------------ arrival schedule
def arrival_counts(rate_per_s, warmup_s, measure_s):
    """Exact numbers of scheduled arrivals in the half-open warm-up and measurement intervals."""
    if not (rate_per_s > 0 and warmup_s >= 0 and measure_s > 0):
        raise ValueError("rate and measurement duration must be positive and warm-up non-negative")
    n = lambda t: int(math.ceil(round(t * rate_per_s, 9)))
    warm, total = n(warmup_s), n(warmup_s + measure_s)
    return warm, total - warm, total


class Admission:
    """Synchronous generator-capacity reservation, per traffic kind. A reservation is made BEFORE a task
    is created and released exactly once (success, error, deadline, cancellation, task-creation failure)."""

    class Token:
        __slots__ = ("kind", "released")

        def __init__(self, kind):
            self.kind, self.released = kind, False

    def __init__(self, capacity):
        if isinstance(capacity, int):
            capacity = {"predict": capacity, "explain": capacity}
        if any(v < 1 for v in capacity.values()):
            raise ValueError("capacity must be at least 1")
        self.capacity = dict(capacity)
        self.outstanding = Counter()
        self.max_outstanding = Counter()
        self.violations = 0
        self.double_release_attempts = 0
        self.reserved_total = 0
        self.released_total = 0

    def try_reserve(self, kind):
        if self.outstanding[kind] >= self.capacity[kind]:
            return None
        self.outstanding[kind] += 1
        self.reserved_total += 1
        self.max_outstanding[kind] = max(self.max_outstanding[kind], self.outstanding[kind])
        if self.outstanding[kind] > self.capacity[kind]:
            self.violations += 1
        return Admission.Token(kind)

    def release(self, token):
        if token.released:
            self.double_release_attempts += 1
            return False
        token.released = True
        self.outstanding[token.kind] -= 1
        self.released_total += 1
        return True

    def total_outstanding(self):
        return sum(self.outstanding.values())


class Ledger:
    """One entry per scheduled arrival (open-loop and burst) or per closed-loop iteration."""

    def __init__(self):
        self.entries = {}

    def add(self, rid, stream, kind, idx, sched, cohort):
        if rid in self.entries:
            raise ValueError(f"duplicate request identity {rid}")
        e = {"id": rid, "stream": stream, "kind": kind, "idx": idx, "sched": sched, "cohort": cohort,
             "state": "scheduled", "miss_reason": None, "detected": None, "start": None, "end": None,
             "outcome": None, "status": None, "code": None, "reason": None}
        self.entries[rid] = e
        return e

    def miss(self, e, reason, detected):
        assert reason in MISS_REASONS and e["state"] in ("scheduled", "attempted")
        e.update(state="missed", miss_reason=reason, detected=detected)

    def attempt(self, e, start):
        e.update(state="attempted", start=start, detected=e["detected"] if e["detected"] is not None else start)

    def finish(self, e, end, outcome, status=None, code=None, reason=None):
        assert outcome in OUTCOMES and e["state"] == "attempted" and e["outcome"] is None
        e.update(end=end, outcome=outcome, status=status, code=code, reason=reason)

    def finalize_open_loop(self, stream, kind, total, warm, sched_of, now):
        """Every scheduled arrival the scheduler never reached becomes a missed arrival."""
        for k in range(total):
            rid = f"{stream}:{k}"
            if rid not in self.entries:
                e = self.add(rid, stream, kind, k % len(INPUTS), sched_of(k), "warmup" if k < warm else "measurement")
                self.miss(e, "not_reached_before_stop", now)


# --------------------------------------------------------------------- response classification
def _contract_predict(index, status, body):
    ok = (isinstance(body, dict) and set(body) == {"probability", "decision", "threshold", "device", "selected_approach"}
          and body.get("threshold") == 0.5)
    if not ok:
        return "contract_violation"
    if not (abs(body["probability"] - EXPECTED_PROBABILITY[index]) <= PROBABILITY_TOLERANCE
            and body["decision"] == EXPECTED_DECISION[index]):
        return "association_violation"
    return None


def _contract_explain(index, status, body, headers):
    try:
        shape = (body["schema_version"] == "explanation.v1"
                 and [a["feature"] for a in body["attributions"]] == list(FEATURES)
                 and all(math.isfinite(a["attribution"]) for a in body["attributions"])
                 and len(body["notices"]) == 9 and "subtype" not in json.dumps(body))
    except Exception:
        shape = False
    if not shape or headers.get("cache-control") != "no-store":
        return "contract_violation"
    try:
        assoc = ([a["value"] for a in body["attributions"]] == [float(INPUTS[index][n]) for n in FEATURES]
                 and abs(body["prediction"]["probability"] - EXPECTED_PROBABILITY[index]) <= PROBABILITY_TOLERANCE)
    except Exception:
        assoc = False
    return None if assoc else "association_violation"


def classify_response(kind, index, status, headers, text):
    """-> (outcome, code, reason). Pure and synchronous."""
    try:
        body = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except ValueError:
        return "invalid_json", None, "non_strict_or_malformed_json"
    if status == 200:
        problem = _contract_predict(index, status, body) if kind == "predict" else _contract_explain(index, status, body, headers)
        return (problem or "success"), None, problem
    code = None
    try:
        error = body["error"]
        code = error["code"]
        ok = isinstance(error, dict) and set(error) <= {"code", "message", "violations"} and "attributions" not in text
        if kind == "explain" and status == 503 and code == "explanation_overloaded":
            ok = ok and headers.get("retry-after") == "1"
        if kind == "explain" and status == 503 and code == "explanation_unavailable":
            ok = ok and headers.get("retry-after") == "5"
    except Exception:
        ok = isinstance(body, dict) and "detail" in body  # the Level 7 validation shape
        code = "detail" if ok else None
    return ("http_error" if ok else "contract_violation"), code, None if ok else "error_contract"


async def execute_request(post, kind, index, clock, deadline_s, start=None):
    """One attempted request under ONE absolute deadline measured from the task start (``start``, default
    now). Returns (outcome, status, code, reason, end). CancelledError propagates to the caller."""
    deadline = (clock() if start is None else start) + deadline_s
    try:
        response = await asyncio.wait_for(post(index), timeout=max(0.0, deadline - clock()))
    except asyncio.TimeoutError:
        return "client_deadline", None, "client_total_deadline", "deadline_exceeded_in_flight", clock()
    except httpx.TimeoutException:
        return "client_deadline", None, "client_phase_timeout", "phase_timeout", clock()
    except httpx.HTTPError:
        return "transport_failure", None, "connection_failure", None, clock()
    outcome, code, reason = classify_response(kind, index, response.status_code, response.headers, response.text)
    end = clock()  # after the synchronous parse and validation
    if end >= deadline:
        return "client_deadline", response.status_code, "late_completion", f"completed_after_deadline_as_{outcome}", end
    return outcome, response.status_code, code, reason, end


async def attempt(ctx, entry, token):
    """Task body: runs one request, always releasing the reservation exactly once."""
    try:
        ctx.ledger.attempt(entry, ctx.clock())
        try:
            outcome, status, code, reason, end = await execute_request(
                lambda i: ctx.post(entry["kind"], i), entry["kind"], entry["idx"], ctx.clock, ctx.deadline_s,
                start=entry["start"])
        except asyncio.CancelledError:
            ctx.ledger.finish(entry, ctx.clock(), "cancelled", None, "cancelled", "task_cancelled")
            raise
        ctx.ledger.finish(entry, end, outcome, status, code, reason)
    finally:
        ctx.admission.release(token)


class Context:
    def __init__(self, post, clock, deadline_s, ledger, admission):
        self.post, self.clock, self.deadline_s, self.ledger, self.admission = post, clock, deadline_s, ledger, admission


def _done_callback(ctx, entry, token):
    def done(task):
        if entry["state"] == "attempted" and entry["outcome"] is None:   # cancelled mid-flight without a record
            ctx.ledger.finish(entry, ctx.clock(), "cancelled", None, "cancelled", "task_cancelled")
        if entry["state"] == "scheduled":                                 # cancelled before its first step
            ctx.ledger.miss(entry, "cancelled_before_start", ctx.clock())
        if not token.released:
            ctx.admission.release(token)  # safety net for a task that never ran its own cleanup
    return done


def reserve_and_spawn(ctx, entry, kind, spawn, tasks):
    """Reserve capacity synchronously, then create the task. No path creates a task without a reservation."""
    detected = ctx.clock()
    entry["detected"] = detected
    token = ctx.admission.try_reserve(kind)
    if token is None:
        ctx.ledger.miss(entry, "capacity_exhausted", detected)
        return None
    try:
        task = spawn(attempt(ctx, entry, token))
    except BaseException:
        ctx.admission.release(token)
        ctx.ledger.miss(entry, "task_creation_failed", detected)
        return None
    task.add_done_callback(_done_callback(ctx, entry, token))
    tasks.append(task)
    return task


async def run_open_loop(ctx, stream, kind, rate_per_s, warmup_s, measure_s, t_start, sleep_until, spawn, tasks, stop=None):
    warm, meas, total = arrival_counts(rate_per_s, warmup_s, measure_s)
    try:
        for k in range(total):
            if stop is not None and stop.is_set():
                break
            sched = t_start + k / rate_per_s
            if sched - ctx.clock() > 0:
                await sleep_until(sched)           # only on-time arrivals yield; late ones are issued back-to-back
            entry = ctx.ledger.add(f"{stream}:{k}", stream, kind, k % len(INPUTS), sched, "warmup" if k < warm else "measurement")
            reserve_and_spawn(ctx, entry, kind, spawn, tasks)
    finally:  # also on cancellation: every scheduled arrival the scheduler never reached is recorded as missed
        ctx.ledger.finalize_open_loop(stream, kind, total, warm, lambda k: t_start + k / rate_per_s, ctx.clock())
    return warm, meas, total


async def run_closed_loop(ctx, stream, kind, caller, t_meas0, t_end, spawn=None):
    n = 0
    while ctx.clock() < t_end:
        start = ctx.clock()
        entry = ctx.ledger.add(f"{stream}:c{caller}:{n}", stream, kind, (caller * 7 + n) % len(INPUTS), None,
                               "warmup" if start < t_meas0 else "measurement")
        token = ctx.admission.try_reserve(kind)
        if token is None:                     # callers never exceed capacity (validated); defensive only
            ctx.ledger.miss(entry, "capacity_exhausted", start)
            await asyncio.sleep(0.001)
            n += 1
            continue
        await attempt(ctx, entry, token)
        n += 1


async def run_burst(ctx, stream, kind, count, sched, spawn, tasks):
    """A synchronized burst: all arrivals are scheduled for the same instant and reserve capacity (and
    create their tasks) in one synchronous block, so none can start before the others exist."""
    entries = [ctx.ledger.add(f"{stream}:{i}", stream, kind, i % len(INPUTS), sched, "measurement") for i in range(count)]
    for entry in entries:
        reserve_and_spawn(ctx, entry, kind, spawn, tasks)
    return entries


# ----------------------------------------------------------------------------- summaries
def summarize_stream(entries, stream_kind, open_loop, t_meas0, t_meas1):
    """Per cohort and whole run, with explicit denominators and reconciliation invariants."""
    out = {"open_loop": open_loop}
    groups = {"warmup": [], "measurement": [], "whole_run": []}
    for e in entries:
        groups[e["cohort"]].append(e)
        groups["whole_run"].append(e)
    for name, es in groups.items():
        attempted = [e for e in es if e["state"] == "attempted"]
        missed = [e for e in es if e["state"] == "missed"]
        terminal = [e for e in attempted if e["outcome"] is not None]
        counts = Counter(e["outcome"] for e in terminal)
        succ = counts.get("success", 0)
        scheduled = len(es) if open_loop else None
        lat_all = [e["end"] - e["start"] for e in terminal]
        lat_ok = [e["end"] - e["start"] for e in terminal if e["outcome"] == "success"]
        failures = sum(v for o, v in counts.items() if o != "success")
        inv = {"all_attempts_terminal": len(attempted) == len(terminal),
               "attempted_equals_successful_plus_failed": len(attempted) == succ + failures + (len(attempted) - len(terminal))}
        if open_loop:
            inv["scheduled_equals_attempted_plus_missed"] = scheduled == len(attempted) + len(missed)
        block = {
            "scheduled_arrivals": scheduled, "attempted": len(attempted), "missed": len(missed) if open_loop else None,
            "missed_by_reason": dict(Counter(e["miss_reason"] for e in missed)) if open_loop else None,
            "unterminated_attempts": len(attempted) - len(terminal),
            "outcomes": {o: counts.get(o, 0) for o in OUTCOMES},
            "http_status_counts": dict(Counter(str(e["status"]) for e in terminal if e["status"] is not None)),
            "public_error_codes": dict(Counter(e["code"] for e in terminal if e["outcome"] == "http_error" and e["code"])),
            "successful": succ,
            "success_rate_of_attempted": rate(succ, len(attempted), "attempted_requests_in_this_cohort"),
            "success_rate_of_scheduled": rate(succ, scheduled, "scheduled_arrivals_in_this_cohort") if open_loop else
                                         {"value": None, "numerator": succ, "denominator": None,
                                          "denominator_is": "not defined for closed-loop traffic (no offered arrivals)"},
            "latency_attempted_successful": latency_stats(lat_ok),
            "latency_attempted_all_terminal": latency_stats(lat_all),
            "latency_scheduled_to_terminal_attempted_only": latency_stats(
                [e["end"] - e["sched"] for e in terminal if e["sched"] is not None]) if open_loop else None,
            "scheduling_lag_attempted_ms": latency_stats(
                [e["start"] - e["sched"] for e in attempted if e["sched"] is not None]) if open_loop else None,
            "missed_detection_lag": latency_stats([e["detected"] - e["sched"] for e in missed if e["sched"] is not None]) if open_loop else None,
            "invariants": inv,
            "latency_samples_exclude_missed_arrivals": True}
        if name == "measurement":
            block["outstanding_at_measurement_end"] = sum(1 for e in attempted if e["end"] is None or e["end"] > t_meas1)
            ends = [e["end"] for e in terminal]
            block["drain_duration_s"] = round(max(0.0, max(ends) - t_meas1), 3) if ends else 0.0
            block["completion_window_throughput_per_s"] = round(
                sum(1 for e in terminal if t_meas0 <= e["end"] < t_meas1) / (t_meas1 - t_meas0), 3)
        out[name] = block
    return out


# ------------------------------------------------------------------------------- runtime
async def loop_lag_monitor(lags, t_end):
    while time.monotonic() < t_end:
        before = time.monotonic()
        await asyncio.sleep(0.01)
        lags.append(time.monotonic() - before - 0.01)


async def diag_sampler(client, url, t_end, out):
    while time.monotonic() < t_end:
        try:
            response = await client.get(url, timeout=2.0)
            data = response.json()
            out.append({"t": time.monotonic(), **{k: data[k] for k in ("dispatcher", "monitors")}})
        except Exception:
            out.append({"t": time.monotonic(), "error": True})
        await asyncio.sleep(1.0)


def cpu_usage_usec():
    try:
        for line in open("/sys/fs/cgroup/cpu.stat"):
            if line.startswith("usage_usec"):
                return int(line.split()[1])
    except OSError:
        return None


def build_config(args):
    cfg = dict(PROFILES.get(args.profile, {})) if args.profile else {}
    for key in ("predict_rate", "explain_rate", "explain_callers", "warmup", "measure", "burst", "burst_at", "bucket"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg.setdefault("burst", 0)
    cfg.setdefault("burst_at", 1.0)
    cfg.setdefault("bucket", 0.0)
    for key in ("predict_rate", "explain_rate", "explain_callers"):
        cfg.setdefault(key, 0)
    if cfg.get("measure", 0) <= 0 or cfg.get("warmup", -1) < 0:
        raise ValueError("measurement duration must be positive and warm-up non-negative")
    if args.timeout <= 0 or args.max_backlog < 1 or args.max_connections < 1:
        raise ValueError("timeout, backlog and connection limits must be positive")
    if int(cfg["explain_callers"]) > args.max_backlog:
        raise ValueError("closed-loop callers must not exceed the generator capacity (--max-backlog)")
    return cfg


async def run(args):
    base = args.base_url.rstrip("/")
    limits = httpx.Limits(max_connections=args.max_connections, max_keepalive_connections=args.max_connections)
    phase = httpx.Timeout(args.timeout)  # subordinate safeguards; the total deadline is enforced above them
    async with httpx.AsyncClient(limits=limits, timeout=phase) as client:
        if args.mode == "ready":
            t_health = t_expl = None
            deadline = time.monotonic() + args.ready_timeout
            while time.monotonic() < deadline:
                try:
                    if t_health is None and (await client.get(base + "/health")).status_code == 200:
                        t_health = time.time()
                    if t_health is not None:
                        r = await client.post(base + "/v1/explanations", json=INPUTS[0])
                        if r.status_code == 200:
                            t_expl = time.time()
                            break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.05)
            print(json.dumps({"health_epoch": t_health, "explanation_epoch": t_expl}))
            return 0 if t_expl else 1
        cfg = build_config(args)

        async def post(kind, index):
            return await client.post(base + ("/predict" if kind == "predict" else "/v1/explanations"), json=INPUTS[index])

        ledger, admission = Ledger(), Admission(args.max_backlog)
        ctx = Context(post, time.monotonic, args.timeout, ledger, admission)
        loop = asyncio.get_running_loop()

        async def sleep_until(when):
            await asyncio.sleep(max(0.0, when - time.monotonic()))

        t_start = time.monotonic() + 0.5
        t_meas0 = t_start + cfg["warmup"]
        t_end = t_meas0 + cfg["measure"]
        cpu0, wall0 = cpu_usage_usec(), time.monotonic()
        lags, diag, tasks, streams = [], [], [], {}
        extra = [asyncio.create_task(loop_lag_monitor(lags, t_end + 3))]
        if args.diag_url:
            extra.append(asyncio.create_task(diag_sampler(client, base + args.diag_url, t_end, diag)))
        runners = []
        if cfg["predict_rate"]:
            streams["predict_open_loop"] = ("predict", True)
            runners.append(run_open_loop(ctx, "predict_open_loop", "predict", cfg["predict_rate"], cfg["warmup"], cfg["measure"],
                                         t_start, sleep_until, loop.create_task, tasks))
        if cfg["explain_rate"]:
            streams["explain_open_loop"] = ("explain", True)
            runners.append(run_open_loop(ctx, "explain_open_loop", "explain", cfg["explain_rate"], cfg["warmup"], cfg["measure"],
                                         t_start, sleep_until, loop.create_task, tasks))
        for caller in range(int(cfg["explain_callers"])):
            streams["explain_closed_loop"] = ("explain", False)
            runners.append(run_closed_loop(ctx, "explain_closed_loop", "explain", caller, t_meas0, t_end))
        if cfg["burst"]:
            streams["explain_burst"] = ("explain", True)

            async def burst():
                await asyncio.sleep(max(0.0, t_meas0 + cfg["burst_at"] - time.monotonic()))
                await run_burst(ctx, "explain_burst", "explain", int(cfg["burst"]), t_meas0 + cfg["burst_at"], loop.create_task, tasks)
            runners.append(burst())
        await asyncio.gather(*runners)
        await asyncio.gather(*tasks, return_exceptions=True)
        wall1, cpu1 = time.monotonic(), cpu_usage_usec()
        for t in extra:
            t.cancel()
        await asyncio.gather(*extra, return_exceptions=True)
        loop_lag = sorted(lags)
        results = {}
        for name, (kind, is_open) in streams.items():
            es = [e for e in ledger.entries.values() if e["stream"] == name]
            results[name] = summarize_stream(es, kind, is_open, t_meas0, t_end)
            if cfg["bucket"]:
                width, buckets = cfg["bucket"], {}
                for e in es:
                    if e["state"] == "attempted" and e["outcome"] is not None:
                        buckets.setdefault(int(((e["sched"] if e["sched"] is not None else e["start"]) - t_meas0) // width), []).append(e)
                results[name]["timeline"] = [
                    {"bucket_start_s": round(i * width, 2), "attempted": len(v), "outcomes": dict(Counter(e["outcome"] for e in v)),
                     "request_p95_ms": latency_stats([e["end"] - e["start"] for e in v]).get("p95_ms")}
                    for i, v in sorted(buckets.items()) if i >= 0]
        summary = {
            "summary_schema": SUMMARY_SCHEMA, "profile": args.profile, "config": {k: cfg[k] for k in sorted(cfg)},
            "client": {"total_deadline_s": args.timeout, "phase_timeouts_s": args.timeout, "max_connections": args.max_connections,
                       "generator_capacity_per_kind": args.max_backlog, "persistent_connections": True,
                       "percentile_method": "linear interpolation between closest ranks (numpy default)",
                       "generator_cpu_cores_used": (round((cpu1 - cpu0) / 1e6 / (wall1 - wall0), 3) if cpu0 is not None else None),
                       "event_loop_lag_ms": {"p99": round(1000 * (percentile(loop_lag, .99) or 0), 3),
                                             "max": round(1000 * (loop_lag[-1] if loop_lag else 0), 3)}},
            "generator_capacity": {"max_outstanding_by_kind": dict(admission.max_outstanding), "violations": admission.violations,
                                   "double_release_attempts": admission.double_release_attempts,
                                   "reserved_total": admission.reserved_total, "released_total": admission.released_total,
                                   "outstanding_at_end": admission.total_outstanding()},
            "inputs": {"count": len(INPUTS), "seed": 42, "near_threshold": 10},
            "windows_s": {"warmup": cfg["warmup"], "measurement": cfg["measure"]},
            "streams": results, "wall_epoch_end": time.time(),
        }
        if diag:
            ok = [d for d in diag if "dispatcher" in d and d["dispatcher"]]
            summary["dispatcher_samples"] = {
                "n": len(diag), "errors": sum(1 for d in diag if d.get("error")),
                "max_active": max((d["dispatcher"]["active"] for d in ok), default=None),
                "max_queued": max((d["dispatcher"]["queued"] for d in ok), default=None),
                "max_monitor_tasks": max((d["monitors"] for d in ok), default=None),
                "final": ok[-1]["dispatcher"] if ok else None}
        if args.raw:
            with gzip.open(args.raw, "wt") as handle:
                for e in ledger.entries.values():
                    handle.write(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items() if k != "idx"}) + "\n")
        with open(args.out, "w") as handle:
            json.dump(summary, handle, indent=1)
        print(json.dumps({"profile": args.profile, "done": True}))
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True)
    p.add_argument("--mode", choices=("run", "ready"), default="run")
    p.add_argument("--profile", choices=sorted(PROFILES))
    p.add_argument("--predict-rate", dest="predict_rate", type=float)
    p.add_argument("--explain-rate", dest="explain_rate", type=float)
    p.add_argument("--explain-callers", dest="explain_callers", type=int)
    p.add_argument("--warmup", type=float)
    p.add_argument("--measure", type=float)
    p.add_argument("--burst", type=int)
    p.add_argument("--burst-at", dest="burst_at", type=float)
    p.add_argument("--bucket", type=float)
    p.add_argument("--timeout", type=float, default=2.0, help="TOTAL client deadline per attempted request (seconds)")
    p.add_argument("--max-connections", dest="max_connections", type=int, default=128)
    p.add_argument("--max-backlog", dest="max_backlog", type=int, default=100, help="generator capacity per traffic kind")
    p.add_argument("--ready-timeout", dest="ready_timeout", type=float, default=90.0)
    p.add_argument("--diag-url", dest="diag_url")
    p.add_argument("--out", default="/dev/stdout")
    p.add_argument("--raw")
    args = p.parse_args()
    try:
        code = asyncio.run(run(args))
    except ValueError as exc:
        p.error(str(exc))
    sys.exit(code)


if __name__ == "__main__":
    main()
