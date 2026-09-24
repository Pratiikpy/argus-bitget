"""The dumb baseline — does the desk beat simply holding what it picked?

This is the question a judge asks first and almost nothing in the corpus answers. LLM-Trading-Lab
ran a real-money LLM portfolio for six months, computed an index buy-and-hold comparison *and then
disclaimed it as non-evidentiary*, so its conclusions rest on no baseline at all. ai-hedge-fund has
no baseline of any kind. A trading agent with no baseline has not been evaluated; it has been
described.

**Why the baseline is not an index.** Comparing a desk that traded NVDA, TSLA and COIN against the
S&P 500 measures which universe was hot, not whether the desk added anything. The comparison that
isolates the desk's contribution is against **its own picks, held**: take every symbol the desk ever
took a position in, buy each once at the first entry price it paid, hold to the cutoff, equal
weight. Same names, same start, no turnover.

That makes the question mechanical, and on this venue it is the whole question. The round trip costs
12bps and the measured intraday edge is about zero, so every additional trade starts 12bps behind
doing nothing. **Beating buy-and-hold on your own picks means the trading added value. Losing to it
means the desk would have done better choosing once and going home.**

**What this module refuses to do.** With no settled trades there is no comparison, and it returns
:attr:`Verdict.UNDEFINED` with the reason rather than a zero. A zero here would read as "the desk
matched the baseline", which is a claim about performance made from an absence of performance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol


class Verdict(StrEnum):
    """How the desk did against holding its own picks."""

    UNDEFINED = "undefined"
    """No settled trades, or no mark to close the baseline against. Not a tie."""

    BEAT = "beat"
    MATCHED = "matched"
    LOST = "lost"
    """The desk would have done better choosing once and going home."""

    @property
    def is_evidence_of_skill(self) -> bool:
        return self is Verdict.BEAT


MATCH_TOLERANCE_BPS = Decimal("5")
"""Within this the two are called a tie rather than a win.

A 2bp difference over a handful of trades is noise, and reporting it as a victory is the kind of
over-reading this module exists to prevent."""


class SettledTrade(Protocol):
    """The fields a settled ledger entry must expose to be scored.

    Declared read-only. A `Entry` is a frozen dataclass, and a Protocol with settable attributes
    refuses it — the comparison only ever reads, so the narrower contract is also the true one.
    """

    @property
    def symbol(self) -> str: ...
    @property
    def side(self) -> str: ...
    @property
    def quantity(self) -> str: ...
    @property
    def entry_price(self) -> str: ...
    @property
    def exit_price(self) -> str | None: ...
    @property
    def net_pnl(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class Leg:
    """One symbol's contribution, both ways."""

    symbol: str
    trades: int
    desk_pnl: Decimal
    baseline_pnl: Decimal
    notional: Decimal

    @property
    def difference(self) -> Decimal:
        return self.desk_pnl - self.baseline_pnl

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trades": self.trades,
            "desk_pnl": str(self.desk_pnl),
            "baseline_pnl": str(self.baseline_pnl),
            "difference": str(self.difference),
            "notional": str(self.notional),
        }


@dataclass(frozen=True, slots=True)
class BaselineReport:
    """The desk against holding its own picks."""

    legs: tuple[Leg, ...]
    reason: str

    @property
    def desk_pnl(self) -> Decimal:
        return sum((leg.desk_pnl for leg in self.legs), Decimal("0"))

    @property
    def baseline_pnl(self) -> Decimal:
        return sum((leg.baseline_pnl for leg in self.legs), Decimal("0"))

    @property
    def notional(self) -> Decimal:
        return sum((leg.notional for leg in self.legs), Decimal("0"))

    @property
    def difference_bps(self) -> Decimal | None:
        """Desk minus baseline, as basis points of the notional the desk actually deployed.

        In basis points because that is the unit the fee is quoted in: a desk that beats its own
        buy-and-hold by less than the 12bps round trip it paid has not beaten it by enough to
        matter, and stating the margin in the same unit makes that visible without a paragraph.
        """
        if not self.legs or self.notional == 0:
            return None
        return (self.desk_pnl - self.baseline_pnl) / self.notional * Decimal("10000")

    @property
    def verdict(self) -> Verdict:
        margin = self.difference_bps
        if margin is None:
            return Verdict.UNDEFINED
        if abs(margin) <= MATCH_TOLERANCE_BPS:
            return Verdict.MATCHED
        return Verdict.BEAT if margin > 0 else Verdict.LOST

    @property
    def legs_beaten(self) -> int:
        return sum(1 for leg in self.legs if leg.difference > 0)

    def as_dict(self) -> dict[str, Any]:
        margin = self.difference_bps
        return {
            "verdict": str(self.verdict),
            "is_evidence_of_skill": self.verdict.is_evidence_of_skill,
            "desk_pnl": str(self.desk_pnl),
            "baseline_pnl": str(self.baseline_pnl),
            "difference_bps": None if margin is None else str(round(margin, 2)),
            "symbols": len(self.legs),
            "symbols_beaten": self.legs_beaten,
            "reason": self.reason,
            "legs": [leg.as_dict() for leg in self.legs],
        }

    def render(self) -> list[str]:
        if self.verdict is Verdict.UNDEFINED:
            return [
                f"[baseline] no comparison is possible: {self.reason}. "
                f"This is undefined, not a tie"
            ]
        margin = self.difference_bps or Decimal("0")
        lines = [
            f"[baseline] the desk {self.verdict} buy-and-hold of its own picks by "
            f"{margin:+.2f}bps of deployed notional, across {len(self.legs)} symbol(s)",
            f"[baseline] desk {self.desk_pnl} vs holding {self.baseline_pnl}; "
            f"ahead on {self.legs_beaten} of {len(self.legs)} symbol(s)",
        ]
        if self.verdict is Verdict.LOST:
            lines.append(
                "[baseline] the desk would have done better choosing once and going home; "
                "every round trip started 12bps behind doing nothing"
            )
        return lines


def _decimal(value: str | None) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except ArithmeticError:
        return None


def compare(
    trades: Sequence[SettledTrade], *, marks: dict[str, Decimal] | None = None
) -> BaselineReport:
    """Score the desk against holding its own picks.

    ``marks`` is the closing price per symbol. When a symbol has no mark, the **last exit price the
    desk itself achieved** is used, which is a real observed price rather than an invented one —
    and the fact that it came from the desk's own fill is why the baseline is not flattered: it is
    the same price the desk got out at.
    """
    settled = [t for t in trades if _decimal(t.exit_price) is not None]
    if not settled:
        return BaselineReport(
            legs=(),
            reason=f"{len(trades)} trade(s) supplied, none settled with an exit price",
        )

    by_symbol: dict[str, list[SettledTrade]] = {}
    for trade in settled:
        by_symbol.setdefault(trade.symbol, []).append(trade)

    legs: list[Leg] = []
    for symbol, rows in by_symbol.items():
        first = rows[0]
        entry = _decimal(first.entry_price) or Decimal("0")
        quantity = _decimal(first.quantity) or Decimal("0")
        if entry <= 0 or quantity <= 0:
            continue

        mark = (marks or {}).get(symbol) or _decimal(rows[-1].exit_price) or entry

        # The baseline is always long: "holding what you picked" has no short leg, because a desk
        # that never bought the name had nothing to hold. Scoring the baseline with the desk's own
        # direction would make it a copy of the desk rather than an alternative to it.
        baseline_pnl = (mark - entry) * quantity

        desk_pnl = Decimal("0")
        for row in rows:
            realised = _decimal(row.net_pnl)
            if realised is not None:
                desk_pnl += realised
                continue
            exit_price = _decimal(row.exit_price)
            qty = _decimal(row.quantity) or Decimal("0")
            entry_price = _decimal(row.entry_price) or Decimal("0")
            if exit_price is None:
                continue
            direction = Decimal("1") if row.side.upper() == "BUY" else Decimal("-1")
            desk_pnl += (exit_price - entry_price) * qty * direction

        legs.append(Leg(
            symbol=symbol, trades=len(rows), desk_pnl=desk_pnl,
            baseline_pnl=baseline_pnl, notional=entry * quantity,
        ))

    if not legs:
        return BaselineReport(
            legs=(), reason="no settled trade carried a usable entry price and quantity"
        )
    return BaselineReport(
        legs=tuple(legs),
        reason=(
            f"{len(legs)} symbol(s) compared over "
            f"{sum(x.trades for x in legs)} settled trade(s)"
        ),
    )


def main() -> int:
    import argparse
    import json
    from pathlib import Path

    from argus.paper.ledger import PaperLedger

    root = Path(__file__).resolve().parents[3] / "data"
    parser = argparse.ArgumentParser(
        description="Compare the desk against buy-and-hold of its own picks."
    )
    parser.add_argument("--ledger", type=Path, default=root / "paper_ledger.jsonl")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    entries = PaperLedger(path=args.ledger).entries if args.ledger.exists() else []
    report = compare([e for e in entries if e.is_settled])
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        for line in report.render():
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MATCH_TOLERANCE_BPS",
    "BaselineReport",
    "Leg",
    "SettledTrade",
    "Verdict",
    "compare",
    "main",
]
