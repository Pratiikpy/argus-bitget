"""Latency — the price you decided on is not the price you get.

Ported from ``nkaz001/hftbacktest`` (MIT, ``backtest/models/latency.rs``). Three latencies, and
conflating them is the usual error:

* **feed** — the market event reaching us. Implicit in the data's timestamps.
* **entry** — our request reaching the venue. The order is submitted at ``req_ts`` and arrives at
  ``req_ts + entry``.
* **response** — the venue's answer reaching us. Filled at ``exch_ts``, known at
  ``exch_ts + response``.

They are asymmetric and they are not constants. hftbacktest's own note on ``ConstantLatency`` is
blunt about it: *"A 1ms entry latency during 10 PM might be 50ms during 2 PM spike. Constant latency
is optimistic."* So :class:`ConstantLatency` here demands a written reason, the same treatment the
cost model gives a frictionless fee, and :class:`InterpolatedLatency` — which reads a *recorded*
series — is the one meant for a result anybody should believe.

**Why this matters to us and not only to HFT.** ARGUS is not a latency-arbitrage system; 200ms would
be invisible to a strategy holding for a day. Latency matters here for one specific reason: our
decisions are made by a language model, and a model that thinks for **8 to 40 seconds** is making a
decision about a price that has moved by the time the order lands. That is not network latency, it
is *deliberation* latency, and it is three orders of magnitude larger than anything hftbacktest
models. :class:`DecisionLatency` treats it the same way — as a delay between the observed price and
the executed one — and :func:`latency_slippage_bps` prices it against realised volatility.

**And the venue's own clock makes it worse at exactly the wrong time.** Off-hours the book runs at
roughly a third of RTH depth, so the same delay crosses more of the book. Latency and the two-clock
model are the same problem seen twice.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from math import sqrt
from typing import Protocol, runtime_checkable

NANOS_PER_SECOND = 1_000_000_000
NANOS_PER_MILLI = 1_000_000


class LatencyError(ValueError):
    """The latency model was given something it cannot model honestly."""


@runtime_checkable
class LatencyModel(Protocol):
    """``latency.rs:13-20``. Both values are in nanoseconds."""

    def entry(self, req_ts: int) -> int:
        """Nanoseconds from our request to the venue receiving it.

        A **negative** return signals a venue rejection, following hftbacktest's convention at
        ``latency.rs:199-214``: the request never landed, and the magnitude is how long it took to
        find that out.
        """
        ...

    def response(self, exch_ts: int) -> int:
        """Nanoseconds from the venue acting to us knowing about it."""
        ...


@dataclass(frozen=True, slots=True)
class ConstantLatency:
    """One fixed pair of latencies. ``latency.rs:365-388``.

    hftbacktest documents this as optimistic, so it cannot be built without saying why it is
    acceptable here — the same construction guard the cost model puts on a zero fee, for the same
    reason: the convenient choice is the one that quietly flatters a result.
    """

    entry_ns: int
    response_ns: int
    reason: str

    def __post_init__(self) -> None:
        if self.entry_ns <= 0 or self.response_ns <= 0:
            raise LatencyError(
                "latency must be positive. Zero latency is the execution equivalent of a zero fee: "
                "it makes every result better and none of them true."
            )
        if not self.reason.strip():
            raise LatencyError(
                "ConstantLatency needs a written reason. Real latency varies with load — "
                "hftbacktest's own note is that 1ms at 10 PM can be 50ms at a 2 PM spike — so a "
                "constant is a claim, and a claim gets stated."
            )

    def entry(self, req_ts: int) -> int:
        return self.entry_ns

    def response(self, exch_ts: int) -> int:
        return self.response_ns


@dataclass(frozen=True, slots=True)
class LatencyRow:
    """One recorded round trip. ``OrderLatencyRow``, ``latency.rs:97-274``.

    ``exch_ts == 0`` is the venue's rejection marker, kept rather than cleaned away: a series with
    its overload rejections stripped out understates latency exactly when latency mattered most.
    """

    req_ts: int
    exch_ts: int
    resp_ts: int

    @property
    def was_rejected(self) -> bool:
        return self.exch_ts == 0

    @property
    def entry_ns(self) -> int:
        return self.exch_ts - self.req_ts

    @property
    def response_ns(self) -> int:
        return self.resp_ts - self.exch_ts

    @property
    def round_trip_ns(self) -> int:
        return self.resp_ts - self.req_ts


class InterpolatedLatency:
    """Latency interpolated from a recorded series. ``latency.rs:97-274``.

    Entry latency is keyed on ``req_ts`` and response latency on ``exch_ts`` — two separate cursors
    over the same rows, because the two delays are driven by different things and a single key would
    smear one into the other.

        lat = (y2 - y1) / (x2 - x1) * (t - x1) + y1

    Outside the recorded range the nearest observation is held rather than extrapolated.
    Extrapolating a latency curve invents the tail, and the tail is the part that costs money.
    """

    def __init__(self, rows: list[LatencyRow]) -> None:
        if len(rows) < 2:
            raise LatencyError(
                "interpolation needs at least two recorded round trips; with fewer, use "
                "ConstantLatency and state that it is a guess"
            )
        ordered = sorted(rows, key=lambda r: r.req_ts)
        for a, b in pairwise(ordered):
            if a.req_ts == b.req_ts:
                raise LatencyError(f"two rows share req_ts={a.req_ts}; interpolation needs a slope")
        self._rows = ordered
        self._req_keys = [r.req_ts for r in ordered]

        # Rejected rows carry exch_ts == 0 and would sort to the front of the response index,
        # dragging every response interpolation toward a timestamp that never happened.
        self._resp_rows = sorted(
            (r for r in ordered if not r.was_rejected), key=lambda r: r.exch_ts
        )
        self._exch_keys = [r.exch_ts for r in self._resp_rows]

    @property
    def rows(self) -> list[LatencyRow]:
        return list(self._rows)

    @property
    def rejection_rate(self) -> float:
        return sum(1 for r in self._rows if r.was_rejected) / len(self._rows)

    def entry(self, req_ts: int) -> int:
        row, nxt = self._bracket(self._rows, self._req_keys, req_ts)

        # latency.rs:199-214 — a rejection has no exchange timestamp, so entry latency is taken
        # from the response leg and returned negative to mark that the order never landed.
        if row.was_rejected:
            return -(row.resp_ts - row.req_ts)

        if nxt is None or nxt.was_rejected:
            return row.entry_ns
        return self._interpolate(
            req_ts, row.req_ts, nxt.req_ts, row.entry_ns, nxt.entry_ns
        )

    def response(self, exch_ts: int) -> int:
        if not self._resp_rows:
            raise LatencyError("every recorded round trip was rejected; no response leg to read")
        row, nxt = self._bracket(self._resp_rows, self._exch_keys, exch_ts)
        if nxt is None:
            return row.response_ns
        return self._interpolate(
            exch_ts, row.exch_ts, nxt.exch_ts, row.response_ns, nxt.response_ns
        )

    @staticmethod
    def _bracket(
        rows: list[LatencyRow], keys: list[int], t: int
    ) -> tuple[LatencyRow, LatencyRow | None]:
        i = bisect_right(keys, t) - 1
        if i < 0:
            return rows[0], None          # before the record: hold the first observation
        if i >= len(rows) - 1:
            return rows[-1], None         # after it: hold the last
        return rows[i], rows[i + 1]

    @staticmethod
    def _interpolate(t: int, x1: int, x2: int, y1: int, y2: int) -> int:
        if x2 == x1:
            return y1
        return int((y2 - y1) / (x2 - x1) * (t - x1) + y1)

    def percentile_ns(self, q: float) -> int:
        """Round-trip latency at quantile ``q``, rejections included.

        Reported because a mean latency is the least useful summary of a distribution whose whole
        risk lives in its right tail.
        """
        if not 0 <= q <= 1:
            raise LatencyError(f"q={q} must be in [0, 1]")
        trips = sorted(r.round_trip_ns for r in self._rows)
        idx = min(len(trips) - 1, int(q * (len(trips) - 1) + 0.5))
        return trips[idx]


@dataclass(frozen=True, slots=True)
class DecisionLatency:
    """Deliberation delay — the latency that actually binds on an LLM-driven desk.

    A network round trip to Bitget is single-digit milliseconds. A reasoning model answering a
    trading question takes seconds to tens of seconds, and our own measured Qwen figures span that
    whole range depending on the thinking budget. The decision is therefore made against a price
    that is already stale, and the staleness is a *choice* — buying more reasoning costs execution
    quality, and the trade is only visible if it is measured.

    Held as milliseconds because that is the unit these actually come in; converted at the boundary.
    """

    think_ms: int
    network_ms: int = 50
    label: str = ""

    def __post_init__(self) -> None:
        if self.think_ms < 0 or self.network_ms <= 0:
            raise LatencyError("deliberation may be zero only in principle; network never is")

    @property
    def total_ms(self) -> int:
        return self.think_ms + self.network_ms

    def entry(self, req_ts: int) -> int:
        return self.total_ms * NANOS_PER_MILLI

    def response(self, exch_ts: int) -> int:
        return self.network_ms * NANOS_PER_MILLI


def latency_slippage_bps(
    *,
    latency_ns: int,
    annualised_vol: Decimal,
    depth_multiplier: Decimal = Decimal("1"),
    seconds_per_year: Decimal = Decimal("31557600"),
) -> Decimal:
    """Expected adverse price move over a delay, in basis points.

    The price is modelled as a random walk, so the expected magnitude of the move over ``t`` scales
    with ``sqrt(t)``. A delay therefore costs ``vol * sqrt(t / year)``, and this is an *expected
    magnitude*, not a directional loss — but for a decision that was correct when made, the drift
    away from it is a loss in expectation, which is why it is charged rather than netted.

    ``depth_multiplier`` scales the result for a thinner book. Our measured off-hours regime runs at
    roughly a third of RTH depth, so ``3`` is the honest multiplier there and ``1`` during RTH.
    Latency and the two-clock model are the same problem seen twice.
    """
    if latency_ns < 0:
        raise LatencyError("a rejected order has no slippage; check was_rejected before pricing it")
    if annualised_vol < 0:
        raise LatencyError(f"annualised_vol={annualised_vol} cannot be negative")
    if depth_multiplier <= 0:
        raise LatencyError(f"depth_multiplier={depth_multiplier} must be positive")

    seconds = Decimal(latency_ns) / Decimal(NANOS_PER_SECOND)
    fraction_of_year = seconds / seconds_per_year
    move = annualised_vol * Decimal(str(sqrt(float(fraction_of_year))))
    return move * Decimal("10000") * depth_multiplier


def thinking_budget_cost_bps(
    *,
    think_ms: int,
    annualised_vol: Decimal,
    depth_multiplier: Decimal = Decimal("1"),
) -> Decimal:
    """What a thinking budget costs in execution quality.

    This is the number that makes "use the bigger reasoning budget" an argued decision rather than
    a reflex: more deliberation is only worth buying if it improves the decision by more than this.
    """
    return latency_slippage_bps(
        latency_ns=think_ms * NANOS_PER_MILLI,
        annualised_vol=annualised_vol,
        depth_multiplier=depth_multiplier,
    )


__all__ = [
    "NANOS_PER_MILLI",
    "NANOS_PER_SECOND",
    "ConstantLatency",
    "DecisionLatency",
    "InterpolatedLatency",
    "LatencyError",
    "LatencyModel",
    "LatencyRow",
    "latency_slippage_bps",
    "thinking_budget_cost_bps",
]
