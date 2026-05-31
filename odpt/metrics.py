# metrics.py
# MetricsCollector: records per-request waiting/ride times, service rate,
# rejections, distance, and vehicle-hours. Writes summary.json.
# Not runnable — imported by main.py and benchmark.py.

from __future__ import annotations
import math
from dataclasses import dataclass, field
from statistics import mean, stdev, quantiles
from typing import Optional


@dataclass
class RequestRecord:
    req_id:       str
    request_time: float
    direct_time:  Optional[float] = None
    pickup_time:  Optional[float] = None
    dropoff_time: Optional[float] = None
    rejected:     bool            = False

    @property
    def wait_time(self):
        if self.pickup_time is not None:
            return self.pickup_time - self.request_time
        return None

    @property
    def ride_time(self):
        if self.pickup_time is not None and self.dropoff_time is not None:
            return self.dropoff_time - self.pickup_time
        return None

    @property
    def detour_ratio(self):
        if self.ride_time is not None and self.direct_time and self.direct_time > 0:
            return self.ride_time / self.direct_time
        return None


@dataclass
class ViolationRecord:
    kind:   str     # "wait" or "ride"
    req_id: str
    value:  float   # actual value that violated
    limit:  float   # the limit that was exceeded
    t:      float   # simulation time of violation

    @property
    def excess(self) -> float:
        return self.value - self.limit


class MetricsCollector:

    def __init__(self) -> None:
        self.records:            dict[str, RequestRecord] = {}
        self.violations:         list[ViolationRecord]    = []
        self.decision_latencies: list[float]              = []  # ms
        self.total_distance:     float                    = 0.0
        self.loaded_distance:    float                    = 0.0   # at least 1 passenger aboard
        self.empty_distance:     float                    = 0.0   # no passengers (deadheading)
        self.improvements:       int                      = 0

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def register(self, req_id, request_time, direct_time=None):
        self.records[req_id] = RequestRecord(req_id, request_time, direct_time)

    def mark_rejected(self, req_id):
        if req_id in self.records:
            self.records[req_id].rejected = True

    def mark_pickup(self, req_id, t):
        if req_id in self.records:
            self.records[req_id].pickup_time = t

    def mark_dropoff(self, req_id, t):
        if req_id in self.records:
            self.records[req_id].dropoff_time = t

    def log_decision_latency(self, elapsed_seconds: float):
        self.decision_latencies.append(elapsed_seconds * 1000.0)

    def log_distance(self, distance: float, passengers_onboard: int = 0):
        """
        Log distance travelled during one vehicle leg.

        Parameters
        ----------
        distance            : travel time proxy (minutes, same units as simulation)
        passengers_onboard  : number of passengers aboard BEFORE this leg starts.
                              Pass len(vehicle.onboard) from vehicle_process in main.py.
                              0 means the vehicle is running empty (deadheading).
        """
        self.total_distance += distance
        if passengers_onboard > 0:
            self.loaded_distance += distance
        else:
            self.empty_distance  += distance

    def log_improvement(self):
        self.improvements += 1

    def log_violation(self, kind: str, req_id: str,
                      value: float, limit: float, t: float):
        self.violations.append(ViolationRecord(kind, req_id, value, limit, t))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        served   = [r for r in self.records.values()
                    if not r.rejected and r.dropoff_time is not None]
        rejected = [r for r in self.records.values() if r.rejected]
        in_prog  = [r for r in self.records.values()
                    if not r.rejected and r.dropoff_time is None]
        total    = len(self.records)

        wait_times    = [r.wait_time    for r in served if r.wait_time    is not None]
        ride_times    = [r.ride_time    for r in served if r.ride_time    is not None]
        detour_ratios = [r.detour_ratio for r in served if r.detour_ratio is not None]

        def pct(lst, p):
            if len(lst) < 2:
                return lst[0] if lst else None
            return quantiles(lst, n=100)[p - 1]

        wait_viols = [v for v in self.violations if v.kind == "wait"]
        ride_viols = [v for v in self.violations if v.kind == "ride"]

        # Deadhead ratio — fraction of total distance driven without passengers.
        # 0.0 = perfectly loaded at all times; 1.0 = never carried anyone.
        dh_ratio = (
            self.empty_distance / self.total_distance
            if self.total_distance > 0 else None
        )

        # Within-run wait time standard deviation — measures equity across
        # passengers in a single episode (distinct from cross-seed robustness).
        wait_std = stdev(wait_times) if len(wait_times) >= 2 else 0.0

        return {
            # ---- Demand ----
            "total_requests":      total,
            "served":              len(served),
            "rejected":            len(rejected),
            "in_progress":         len(in_prog),
            "service_rate":        len(served) / total if total else 0.0,

            # ---- Wait time (passenger comfort) ----
            "mean_wait":           mean(wait_times)      if wait_times else None,
            "p50_wait":            pct(wait_times,  50)  if wait_times else None,
            "p95_wait":            pct(wait_times,  95)  if wait_times else None,
            "max_wait":            max(wait_times)       if wait_times else None,
            "min_wait":            min(wait_times)       if wait_times else None,
            "std_wait":            wait_std,              # within-run equity

            # ---- Ride time ----
            "mean_ride":           mean(ride_times)      if ride_times else None,
            "p95_ride":            pct(ride_times,  95)  if ride_times else None,

            # ---- Detour factor (actual_ride / direct) ----
            "mean_detour_ratio":   mean(detour_ratios)   if detour_ratios else None,
            "p95_detour_ratio":    pct(detour_ratios, 95)if detour_ratios else None,

            # ---- Fleet efficiency ----
            "total_distance":      self.total_distance,
            "loaded_distance":     self.loaded_distance,
            "empty_distance":      self.empty_distance,
            "deadhead_ratio":      dh_ratio,              # empty / total

            # ---- Algorithm ----
            "improvements":        self.improvements,
            "mean_latency_ms":     mean(self.decision_latencies)
                                   if self.decision_latencies else None,
            "p95_latency_ms":      pct(self.decision_latencies, 95)
                                   if self.decision_latencies else None,

            # ---- Constraint compliance ----
            "violations_total":    len(self.violations),
            "violations_wait":     len(wait_viols),
            "violations_ride":     len(ride_viols),
            "mean_wait_excess":    mean(v.excess for v in wait_viols)
                                   if wait_viols else None,
            "mean_ride_excess":    mean(v.excess for v in ride_viols)
                                   if ride_viols else None,
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "=" * 55)
        print("SIMULATION SUMMARY")
        print("=" * 55)
        parts = [f"{s['served']} served", f"{s['rejected']} rejected"]
        if s['in_progress'] > 0:
            parts.append(f"{s['in_progress']} in-progress")
        print(f"  Requests : {' / '.join(parts)} / "
              f"{s['total_requests']} total  ({s['service_rate']:.1%})")
        print(f"  Wait time: mean={_fmt(s['mean_wait'])}  "
              f"p50={_fmt(s['p50_wait'])}  "
              f"p95={_fmt(s['p95_wait'])}  "
              f"max={_fmt(s['max_wait'])}  "
              f"σ={_fmt(s['std_wait'])} min")
        print(f"  Ride time: mean={_fmt(s['mean_ride'])}  "
              f"p95={_fmt(s['p95_ride'])} min")
        print(f"  Detour   : mean={_fmt(s['mean_detour_ratio'],'.2f')}x  "
              f"p95={_fmt(s['p95_detour_ratio'],'.2f')}x")
        print(f"  Distance : total={s['total_distance']:.1f}  "
              f"loaded={s['loaded_distance']:.1f}  "
              f"empty={s['empty_distance']:.1f}  "
              f"deadhead={_fmt(s['deadhead_ratio'],'.1%')}")
        print(f"  Latency  : mean={_fmt(s['mean_latency_ms'])} ms  "
              f"p95={_fmt(s['p95_latency_ms'])} ms")
        print(f"  Improvements: {s['improvements']} successful route updates")
        print(f"  Violations: {s['violations_total']} total  "
              f"({s['violations_wait']} wait, {s['violations_ride']} ride)")
        if s['violations_wait']:
            print(f"    Mean wait excess : {_fmt(s['mean_wait_excess'])} min")
        if s['violations_ride']:
            print(f"    Mean ride excess : {_fmt(s['mean_ride_excess'])} min")
        print("=" * 55)


def _fmt(val, fmt=".1f") -> str:
    return f"{val:{fmt}}" if val is not None else "n/a"