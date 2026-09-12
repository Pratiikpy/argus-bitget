"""Measure this machine's real latency to Bitget, and write the recorded series.

:mod:`argus.execution.latency` can interpolate a recorded round-trip series, but a model with no
recording behind it is a guess with extra arithmetic. This module produces the recording.

It needs no credentials. Bitget's public market endpoints answer unauthenticated, and every response
envelope carries ``requestTime`` — the venue's own clock — which is exactly the ``exch_ts`` the
interpolation wants. So one public request yields a genuine three-timestamp row:

    req_ts   monotonic clock here, immediately before the send
    exch_ts  the venue's `requestTime`, converted to our clock by the offset measured below
    resp_ts  monotonic clock here, immediately after the body is read

**The clock-offset problem, and why it is handled rather than ignored.** ``requestTime`` comes from
the venue's wall clock and our timestamps from ours, and the two disagree by an unknown constant
plus our own clock error. Subtracting them raw produces an "entry latency" that is mostly clock
skew — and it can easily come out negative, which silently reads as hftbacktest's rejection marker.
So the offset is estimated first, by the standard NTP-style midpoint, and only then are the rows
built. Where the estimate is unusable, the row is recorded with the venue leg marked unknown rather
than fabricated.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.execution.latency import (
    NANOS_PER_MILLI,
    InterpolatedLatency,
    LatencyError,
    LatencyRow,
)

PUBLIC_TIME_URL = "https://api.bitget.com/api/v2/public/time"
PUBLIC_TICKER_URL = "https://api.bitget.com/api/v2/mix/market/ticker?symbol=BTCUSDT&productType=usdt-futures"
DEFAULT_SAMPLES = 30


@dataclass(frozen=True, slots=True)
class Sample:
    """One measured round trip, with both clocks kept apart until the offset is known."""

    req_ns: int
    resp_ns: int
    venue_ms: int | None
    ok: bool
    error: str = ""

    @property
    def round_trip_ns(self) -> int:
        return self.resp_ns - self.req_ns


def _one(url: str, *, timeout: float) -> Sample:
    req = urllib.request.Request(url, headers={"User-Agent": "argus-latency-probe/1"})
    req_ns = time.time_ns()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
        resp_ns = time.time_ns()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Sample(req_ns=req_ns, resp_ns=time.time_ns(), venue_ms=None, ok=False,
                      error=type(exc).__name__)

    try:
        payload: dict[str, Any] = json.loads(body)
        raw = payload.get("requestTime") or (payload.get("data") or {}).get("serverTime")
        venue_ms = int(raw) if raw is not None else None
    except (ValueError, TypeError, AttributeError):
        venue_ms = None
    return Sample(req_ns=req_ns, resp_ns=resp_ns, venue_ms=venue_ms, ok=True)


def estimate_clock_offset_ns(samples: list[Sample]) -> int | None:
    """Offset from the venue's clock to ours, by the NTP midpoint estimator.

    For each sample the venue's timestamp is assumed to have been taken somewhere inside our
    round trip, so its best local equivalent is the midpoint ``(req + resp) / 2``. The offset is
    then ``midpoint - venue``, and the **median** across samples is taken rather than the mean:
    one stalled request would drag a mean by its whole excess.

    Returns ``None`` when no sample carried a venue timestamp, because an offset that was never
    measured must not be silently taken as zero.
    """
    offsets = [
        (s.req_ns + s.resp_ns) // 2 - s.venue_ms * 1_000_000
        for s in samples
        if s.ok and s.venue_ms is not None
    ]
    if not offsets:
        return None
    return int(statistics.median(offsets))


def to_rows(samples: list[Sample], offset_ns: int) -> list[LatencyRow]:
    """Build interpolation rows, dropping any whose legs come out impossible.

    A negative leg after offset correction means the offset estimate did not hold for that sample —
    usually a request that stalled long enough to break the midpoint assumption. Keeping it would
    write a rejection marker (hftbacktest reads a negative entry latency as "never landed") for an
    order that in fact succeeded, so it is dropped and the count reported.
    """
    rows: list[LatencyRow] = []
    for s in samples:
        if not s.ok or s.venue_ms is None:
            continue
        exch_ns = s.venue_ms * 1_000_000 + offset_ns
        if not (s.req_ns < exch_ns < s.resp_ns):
            continue
        rows.append(LatencyRow(req_ts=s.req_ns, exch_ts=exch_ns, resp_ts=s.resp_ns))
    return rows


def probe(
    *, samples: int = DEFAULT_SAMPLES, url: str = PUBLIC_TIME_URL, timeout: float = 10.0,
    spacing_s: float = 0.2,
) -> dict[str, Any]:
    """Run the probe and return a report.

    Requests are spaced so the series samples a stretch of time rather than one burst — a burst
    measures the connection's best case and nothing about how it behaves under varying load, which
    is the only reason a recorded series beats a constant.
    """
    if samples < 2:
        raise LatencyError("a latency series needs at least two samples")

    measured: list[Sample] = []
    for i in range(samples):
        measured.append(_one(url, timeout=timeout))
        if i < samples - 1:
            time.sleep(spacing_s)

    ok = [s for s in measured if s.ok]
    report: dict[str, Any] = {
        "url": url,
        "samples_attempted": samples,
        "samples_ok": len(ok),
        "failures": [s.error for s in measured if not s.ok],
    }
    if not ok:
        report["verdict"] = "FAIL — the venue answered nothing; no latency was measured"
        return report

    trips_ms = sorted(s.round_trip_ns / NANOS_PER_MILLI for s in ok)
    report["round_trip_ms"] = {
        "min": round(trips_ms[0], 2),
        "median": round(statistics.median(trips_ms), 2),
        "p90": round(trips_ms[min(len(trips_ms) - 1, int(0.9 * (len(trips_ms) - 1) + 0.5))], 2),
        "max": round(trips_ms[-1], 2),
    }

    offset = estimate_clock_offset_ns(measured)
    if offset is None:
        report["clock_offset_ms"] = None
        report["verdict"] = (
            "PARTIAL — round trips measured, but no response carried a venue timestamp, so the "
            "legs cannot be split into entry and response"
        )
        return report

    report["clock_offset_ms"] = round(offset / NANOS_PER_MILLI, 2)
    rows = to_rows(measured, offset)
    report["rows_usable"] = len(rows)
    report["rows_dropped"] = len(ok) - len(rows)

    if len(rows) < 2:
        report["verdict"] = (
            "PARTIAL — fewer than two rows survived offset correction; not enough to interpolate"
        )
        return report

    model = InterpolatedLatency(rows)
    report["entry_ms"] = {
        "median": round(statistics.median(r.entry_ns for r in rows) / NANOS_PER_MILLI, 2),
    }
    report["response_ms"] = {
        "median": round(statistics.median(r.response_ns for r in rows) / NANOS_PER_MILLI, 2),
    }
    report["round_trip_p99_ms"] = round(model.percentile_ns(0.99) / NANOS_PER_MILLI, 2)
    report["rejection_rate"] = round(model.rejection_rate, 4)
    report["rows"] = [
        {"req_ts": r.req_ts, "exch_ts": r.exch_ts, "resp_ts": r.resp_ts} for r in rows
    ]
    report["verdict"] = "PASS — a real recorded series, usable by InterpolatedLatency"
    return report


def load_rows(path: Path) -> list[LatencyRow]:
    """Read a recorded series back off disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [LatencyRow(**row) for row in payload.get("rows", [])]


def main() -> int:
    out = Path(__file__).resolve().parents[3] / "data" / "latency_probe.json"
    report = probe()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(report["verdict"])
    if "round_trip_ms" in report:
        rt = report["round_trip_ms"]
        print(f"round trip ms — min {rt['min']} · median {rt['median']} · "
              f"p90 {rt['p90']} · max {rt['max']}")
    if "entry_ms" in report:
        print(f"entry median {report['entry_ms']['median']} ms · "
              f"response median {report['response_ms']['median']} ms · "
              f"{report['rows_usable']} rows ({report['rows_dropped']} dropped)")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Sample", "estimate_clock_offset_ns", "load_rows", "probe", "to_rows"]
