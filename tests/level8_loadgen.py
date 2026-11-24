#!/usr/bin/env python3
"""Level 8 mixed-load generator (test harness; not part of the service).

Open-loop scheduled arrivals and closed-loop concurrent callers against /predict and
/v1/explanations, with a fixed, deterministic collection of 50 synthetic eligible inputs
(40 development-range draws and 10 near-threshold draws, numpy default_rng(42) via
tests/level8_oracle.build_grids; their expected prediction results were computed once from the
repository-layout Level 7 runtime and are embedded below, so no per-request oracle runs here).

Every response is checked for contract and request-result association: a prediction must equal the
expected probability and decision for THAT input, and an explanation must echo that input's values and
carry a probability consistent with the expected one. Percentiles use linear interpolation between
closest ranks (numpy's default method). Latencies: ``request`` is actual-start to completion;
``scheduled`` (open-loop only) is scheduled-arrival to completion. Open-loop arrivals are never
silently slowed: late arrivals are issued immediately and the lag is recorded; arrivals beyond the
bounded client backlog are counted as missed.

Raw input values are never written to the summary; per-request measurements go to an optional
gzip-JSONL file kept outside the repository.
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
PROFILES = {
    "P0": dict(predict_rate=100.0, explain_callers=0, explain_rate=0.0, warmup=10.0, measure=30.0),
    "P1": dict(predict_rate=100.0, explain_callers=4, explain_rate=0.0, warmup=10.0, measure=30.0),
    "P2": dict(predict_rate=100.0, explain_callers=0, explain_rate=40.0, warmup=10.0, measure=30.0),
    "P3": dict(predict_rate=0.0, explain_callers=4, explain_rate=0.0, warmup=10.0, measure=30.0),
    "P4": dict(predict_rate=100.0, explain_callers=0, explain_rate=0.0, warmup=10.0, measure=10.0,
               burst=32, burst_at=1.0, bucket=1.0),
    "P5": dict(predict_rate=100.0, explain_callers=4, explain_rate=0.0, warmup=10.0, measure=300.0, bucket=30.0),
}


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def latency_stats(values_s):
    if not values_s:
        return {"n": 0}
    v = sorted(values_s)
    ms = lambda x: round(x * 1000.0, 3)
    return {"n": len(v), "p50_ms": ms(percentile(v, .5)), "p95_ms": ms(percentile(v, .95)),
            "p99_ms": ms(percentile(v, .99)), "max_ms": ms(v[-1]), "mean_ms": ms(sum(v) / len(v))}


class Recorder:
    def __init__(self):
        self.rows = []  # dicts: kind, idx, sched, start, end, status, code, ok, problems, shape
        self.inflight = Counter()
        self.offered = Counter()
        self.missed = Counter()
        self.attempted = Counter()
        self.lag = []
        self.loop_lag = []
        self.max_inflight = Counter()


def check_predict(index, status, body, headers):
    problems = []
    if status != 200:
        return problems
    try:
        ok = (set(body) == {"probability", "decision", "threshold", "device", "selected_approach"}
              and abs(body["probability"] - EXPECTED_PROBABILITY[index]) <= PROBABILITY_TOLERANCE
              and body["decision"] == EXPECTED_DECISION[index] and body["threshold"] == 0.5)
    except Exception:
        ok = False
    if not ok:
        problems.append("predict_contract_or_association")
    return problems


def check_explain(index, status, body, headers):
    problems = []
    if status == 200:
        try:
            ok = (body["schema_version"] == "explanation.v1"
                  and [a["feature"] for a in body["attributions"]] == list(FEATURES)
                  and [a["value"] for a in body["attributions"]] == [float(INPUTS[index][n]) for n in FEATURES]
                  and abs(body["prediction"]["probability"] - EXPECTED_PROBABILITY[index]) <= PROBABILITY_TOLERANCE
                  and all(math.isfinite(a["attribution"]) for a in body["attributions"])
                  and len(body["notices"]) == 9)
        except Exception:
            ok = False
        if not ok:
            problems.append("explain_contract_or_association")
        if headers.get("cache-control") != "no-store":
            problems.append("explain_missing_no_store")
        if "subtype" in json.dumps(body):
            problems.append("explain_leaked_internal")
    else:
        try:
            error = body["error"]
            ok = set(error) <= {"code", "message", "violations"} and "attributions" not in json.dumps(body)
            if status == 503 and error["code"] == "explanation_overloaded":
                ok = ok and headers.get("retry-after") == "1"
            if status == 503 and error["code"] == "explanation_unavailable":
                ok = ok and headers.get("retry-after") == "5"
        except Exception:
            ok = False
        if not ok:
            problems.append("explain_error_contract")
    return problems


async def fire(client, rec, kind, index, sched, base):
    url = base + ("/predict" if kind == "predict" else "/v1/explanations")
    rec.inflight[kind] += 1
    rec.max_inflight[kind] = max(rec.max_inflight[kind], rec.inflight[kind])
    rec.attempted[kind] += 1
    start = time.monotonic()
    status, code, body, headers, problems = None, None, None, {}, []
    try:
        response = await client.post(url, json=INPUTS[index])
        status, headers = response.status_code, response.headers
        try:
            body = json.loads(response.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        except ValueError:
            body = None
            problems.append("non_strict_json")
        if status != 200 and isinstance(body, dict):
            code = (body.get("error") or {}).get("code") if isinstance(body.get("error"), dict) else "detail"
        problems += (check_predict if kind == "predict" else check_explain)(index, status, body or {}, headers)
        if kind == "predict" and status != 200 and not problems:
            problems.append("predict_non_200")
    except httpx.TimeoutException:
        status, code = -1, "client_timeout"
    except httpx.HTTPError:
        status, code = -2, "connection_failure"
    end = time.monotonic()
    rec.inflight[kind] -= 1
    rec.rows.append({"kind": kind, "idx": index, "sched": sched, "start": start, "end": end, "status": status,
                     "code": code, "problems": problems})


async def open_loop(client, rec, kind, rate, t0, t_end, base, max_backlog, tasks):
    interval, k = 1.0 / rate, 0
    while True:
        sched = t0 + k * interval
        if sched >= t_end:
            return
        delay = sched - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        rec.offered[kind] += 1
        if rec.inflight[kind] >= max_backlog:
            rec.missed[kind] += 1
        else:
            rec.lag.append((kind, sched, time.monotonic() - sched))
            tasks.append(asyncio.create_task(fire(client, rec, kind, k % len(INPUTS), sched, base)))
        k += 1


async def closed_loop(client, rec, kind, caller, t_end, base):
    n = 0
    while time.monotonic() < t_end:
        rec.offered[kind] += 1
        await fire(client, rec, kind, (caller * 7 + n) % len(INPUTS), None, base)
        n += 1


async def loop_lag_monitor(rec, t_end):
    while time.monotonic() < t_end:
        before = time.monotonic()
        await asyncio.sleep(0.01)
        rec.loop_lag.append(time.monotonic() - before - 0.01)


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


def summarize(rec, t_meas0, t_meas1, cfg):
    def measured(r):
        stamp = r["sched"] if r["sched"] is not None else r["start"]
        return t_meas0 <= stamp < t_meas1

    out = {"measurement_s": round(t_meas1 - t_meas0, 3)}
    for kind in ("predict", "explain"):
        rows = [r for r in rec.rows if r["kind"] == kind and measured(r)]
        if not rows and not rec.offered[kind]:
            continue
        done = [r for r in rows if r["status"] is not None]
        ok = [r for r in done if r["status"] == 200 and not r["problems"]]
        lag = sorted(l for (k, s, l) in rec.lag if k == kind and t_meas0 <= s < t_meas1)
        offered_win = sum(1 for (k, s, l) in rec.lag if k == kind and t_meas0 <= s < t_meas1)
        out[kind] = {
            "completed": len(done), "successful": len(ok),
            "success_fraction": round(len(ok) / len(done), 6) if done else None,
            "status_counts": dict(Counter(str(r["status"]) for r in done)),
            "error_code_counts": dict(Counter(r["code"] for r in done if r["code"])),
            "client_timeouts": sum(1 for r in done if r["status"] == -1),
            "connection_failures": sum(1 for r in done if r["status"] == -2),
            "contract_or_association_violations": sum(1 for r in done if r["problems"]),
            "violation_kinds": dict(Counter(p for r in done for p in r["problems"])),
            "latency_request_successful": latency_stats([r["end"] - r["start"] for r in ok]),
            "latency_request_all_completed": latency_stats([r["end"] - r["start"] for r in done]),
            "latency_scheduled_all_completed": latency_stats(
                [r["end"] - r["sched"] for r in done if r["sched"] is not None]),
            "latency_scheduled_successful": latency_stats(
                [r["end"] - r["sched"] for r in ok if r["sched"] is not None]),
            "throughput_completed_per_s": round(len(done) / (t_meas1 - t_meas0), 3),
            "throughput_successful_per_s": round(len(ok) / (t_meas1 - t_meas0), 3),
            "max_inflight": rec.max_inflight[kind],
        }
        if kind == "predict" and cfg["predict_rate"]:
            out[kind]["open_loop"] = {
                "scheduled_rate_per_s": cfg["predict_rate"], "scheduled_in_window": offered_win,
                "missed_arrivals_total": rec.missed[kind], "attempted_in_window": len(rows),
                "scheduling_lag_ms": {"p50": round(1000 * (lag[len(lag) // 2] if lag else 0), 3),
                                      "p99": round(1000 * (percentile(lag, .99) or 0), 3),
                                      "max": round(1000 * (lag[-1] if lag else 0), 3)}}
        if kind == "explain" and cfg["explain_rate"]:
            out[kind]["open_loop"] = {
                "scheduled_rate_per_s": cfg["explain_rate"], "scheduled_in_window": offered_win,
                "missed_arrivals_total": rec.missed[kind], "attempted_in_window": len(rows),
                "scheduling_lag_ms": {"p50": round(1000 * (lag[len(lag) // 2] if lag else 0), 3),
                                      "p99": round(1000 * (percentile(lag, .99) or 0), 3),
                                      "max": round(1000 * (lag[-1] if lag else 0), 3)}}
        if cfg.get("bucket"):
            width, buckets = cfg["bucket"], {}
            for r in rows:
                stamp = r["sched"] if r["sched"] is not None else r["start"]
                buckets.setdefault(int((stamp - t_meas0) // width), []).append(r)
            out[kind]["timeline"] = [
                {"bucket_start_s": round(i * width, 2), "completed": len(v),
                 "status_counts": dict(Counter(str(r["status"]) for r in v)),
                 "request_p95_ms": latency_stats([r["end"] - r["start"] for r in v if r["status"] is not None]).get("p95_ms")}
                for i, v in sorted(buckets.items())]
    return out


async def run(args):
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
    base = args.base_url.rstrip("/")
    limits = httpx.Limits(max_connections=args.max_connections, max_keepalive_connections=args.max_connections)
    async with httpx.AsyncClient(limits=limits, timeout=args.timeout) as client:
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
        rec = Recorder()
        tasks, extra = [], []
        t_start = time.monotonic() + 0.5
        t_meas0 = t_start + cfg["warmup"]
        t_end = t_meas0 + cfg["measure"]
        cpu0, wall0 = cpu_usage_usec(), time.monotonic()
        extra.append(asyncio.create_task(loop_lag_monitor(rec, t_end + 3)))
        diag = []
        if args.diag_url:
            extra.append(asyncio.create_task(diag_sampler(client, base + args.diag_url, t_end, diag)))
        request_tasks = []  # every open-loop request task, awaited at the end so the window drains cleanly
        if cfg["predict_rate"]:
            tasks.append(asyncio.create_task(open_loop(client, rec, "predict", cfg["predict_rate"], t_start, t_end,
                                                       base, args.max_backlog, request_tasks)))
        if cfg["explain_rate"]:
            tasks.append(asyncio.create_task(open_loop(client, rec, "explain", cfg["explain_rate"], t_start, t_end,
                                                       base, args.max_backlog, request_tasks)))
        for caller in range(int(cfg["explain_callers"])):
            tasks.append(asyncio.create_task(closed_loop(client, rec, "explain", caller, t_end, base)))
        if cfg["burst"]:
            async def burst():
                await asyncio.sleep(max(0.0, t_meas0 + cfg["burst_at"] - time.monotonic()))
                gate = asyncio.Event()

                async def one(i):
                    await gate.wait()
                    rec.offered["explain"] += 1
                    await fire(client, rec, "explain", i % len(INPUTS), t_meas0 + cfg["burst_at"], base)

                group = [asyncio.create_task(one(i)) for i in range(int(cfg["burst"]))]
                await asyncio.sleep(0)
                gate.set()
                await asyncio.gather(*group)
            tasks.append(asyncio.create_task(burst()))
        await asyncio.gather(*tasks)
        await asyncio.gather(*request_tasks, return_exceptions=True)
        wall1, cpu1 = time.monotonic(), cpu_usage_usec()
        for t in extra:
            t.cancel()
        await asyncio.gather(*extra, return_exceptions=True)
        loop_lag = sorted(rec.loop_lag)
        summary = {
            "profile": args.profile, "config": {k: cfg[k] for k in sorted(cfg)},
            "client": {"timeout_s": args.timeout, "max_connections": args.max_connections,
                       "max_backlog": args.max_backlog, "persistent_connections": True,
                       "percentile_method": "linear interpolation between closest ranks (numpy default)",
                       "generator_cpu_cores_used": (round((cpu1 - cpu0) / 1e6 / (wall1 - wall0), 3) if cpu0 is not None else None),
                       "event_loop_lag_ms": {"p99": round(1000 * (percentile(loop_lag, .99) or 0), 3),
                                             "max": round(1000 * (loop_lag[-1] if loop_lag else 0), 3)}},
            "inputs": {"count": len(INPUTS), "seed": 42, "near_threshold": 10},
            "results": summarize(rec, t_meas0, t_end, cfg),
            "wall_epoch_end": time.time(),
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
                for r in rec.rows:
                    handle.write(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}) + "\n")
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
    p.add_argument("--timeout", type=float, default=2.0)
    p.add_argument("--max-connections", dest="max_connections", type=int, default=128)
    p.add_argument("--max-backlog", dest="max_backlog", type=int, default=100)
    p.add_argument("--ready-timeout", dest="ready_timeout", type=float, default=90.0)
    p.add_argument("--diag-url", dest="diag_url")
    p.add_argument("--out", default="/dev/stdout")
    p.add_argument("--raw")
    args = p.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
