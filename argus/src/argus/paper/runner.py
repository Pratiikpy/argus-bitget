"""Paper-trading runner — drives the desk against live Bitget data and writes the log.

One invocation is one decision cycle: read the real market, build the session state and hedge
surface, run the analyst panel, let the Meta-PM decide, apply the Constitution, and append the
result to a hash-chained ledger. Run it on a schedule and the log accumulates.

**Settlement is the part that makes it honest.** Open entries are settled against the price at a
*later* fetch, never against the price in the same cycle. A runner that decided and settled in one
pass would be scoring itself on information it already had.

    python -m argus.paper.runner --once
    python -m argus.paper.runner --report
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from argus.agents.analysts import Evidence
from argus.agents.desk import ConstitutionPolicy, TradingDesk
from argus.cost.model import CostModel
from argus.llm.qwen import QwenClient, QwenError, TokenBudget
from argus.market.bitget import RTOKEN_SYMBOLS, Ticker, fetch_rtokens
from argus.paper.ledger import PaperLedger
from argus.risk.hedgeability import HedgeabilitySurface, shut_market_candidate
from argus.truth.clocks import DualClock

LEDGER_PATH = Path(__file__).resolve().parents[3] / "data" / "paper_ledger.jsonl"

# How long a paper position is held before it is settled against a later fetch.
HOLD_HOURS = 24


def _hedge_surface(session_asleep: bool, symbol: str) -> HedgeabilitySurface:
    """The anchor hedge, and whether it can actually be reached.

    Kept on the surface even when unreachable so the absence is visible rather than omitted — that
    distinction is the whole Sleeping-Anchor thesis.
    """
    if not session_asleep:
        return HedgeabilitySurface(())
    return HedgeabilitySurface(
        (shut_market_candidate(symbol.replace("USDT", ""), Decimal("0.95")),),
        session_note="anchor shut; no hedge placeable",
    )


def run_once(
    *,
    symbols: tuple[str, ...] = ("NVDAUSDT",),
    ledger_path: Path = LEDGER_PATH,
    budget: int = 150_000,
) -> dict[str, object]:
    """One decision cycle across the given symbols."""
    clock = DualClock()
    now = datetime.now(UTC)
    tickers = fetch_rtokens()
    ledger = PaperLedger(path=ledger_path, cost=CostModel.bitget_perp())

    settled = _settle_due(ledger, tickers, now)

    client = QwenClient(budget=TokenBudget(limit=budget))
    desk = TradingDesk(client)
    written: list[dict[str, object]] = []

    for symbol in symbols:
        if symbol not in tickers:
            continue
        ticker = tickers[symbol]
        session = clock.state(now, nav_age_seconds=(now - ticker.fetched_at).total_seconds())
        hedges = _hedge_surface(session.is_anchor_asleep, symbol)

        # Live market facts only. No synthetic evidence in the paper log — a log seeded with
        # invented events is not a record of anything.
        evidence = [
            Evidence(
                id=f"mkt-{symbol}",
                claim=(
                    f"{symbol} last {ticker.last}, 24h change {ticker.change_24h}, "
                    f"quoted spread {ticker.spread_bps:.2f}bps, "
                    f"24h base volume {ticker.base_volume}"
                ),
                source="news",
                available_at=ticker.fetched_at,
                credibility=1.0,
            ),
        ]

        run = desk.run(
            symbol=symbol,
            session=session,
            token_price=ticker.last,
            position=Decimal("0"),
            evidence=evidence,
            hedges=hedges,
            decision_id=f"paper-{symbol}-{int(now.timestamp())}",
            constitution=ConstitutionPolicy(),
        )

        final = run.proof.llm_revised_intent or run.proof.llm_original_intent
        entry = ledger.record(
            symbol=symbol,
            verdict=str(final.verdict),
            side=str(final.side).upper(),
            quantity=final.quantity,
            entry_price=ticker.last,
            stated_confidence=final.stated_confidence,
            thesis=final.thesis,
            invalidation=final.invalidation,
            market_state_hash=run.proof.market_state_hash,
            approved_intent_hash=run.proof.approved_intent_hash,
            session_phase=str(session.phase),
            hours_to_discovery=session.hours_to_next_discovery,
            decided_at=now,
            spread_bps=ticker.spread_bps,
        )
        written.append({
            "seq": entry.seq,
            "symbol": symbol,
            "verdict": entry.verdict,
            "quantity": entry.quantity,
            "confidence": entry.stated_confidence,
            "attests_llm_decided": run.proof.attests_llm_decided(),
        })

    return {
        "ran_at": now.isoformat(),
        "decisions_written": written,
        "settled_this_cycle": settled,
        "tokens_spent": client.budget.spent if client.budget else 0,
        "chain": ledger.verify(),
    }


def _settle_due(
    ledger: PaperLedger, tickers: dict[str, Ticker], now: datetime
) -> list[dict[str, object]]:
    """Settle open positions whose hold period has elapsed, at the current price.

    Only entries recorded in an *earlier* cycle are eligible, which is what keeps settlement from
    using information the decision already had.
    """
    out: list[dict[str, object]] = []
    for entry in ledger.entries:
        if entry.is_settled or entry.is_abstention:
            continue
        decided = datetime.fromisoformat(entry.decided_at)
        if now - decided < timedelta(hours=HOLD_HOURS):
            continue
        ticker = tickers.get(entry.symbol)
        if ticker is None:
            continue
        got = ledger.settle(entry.seq, exit_price=ticker.last, settled_at=now)
        out.append({
            "seq": got.seq,
            "symbol": got.symbol,
            "gross_pnl": got.gross_pnl,
            "net_pnl": got.net_pnl,
            "direction_correct": got.direction_correct,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="ARGUS paper-trading runner")
    parser.add_argument("--once", action="store_true", help="run one decision cycle")
    parser.add_argument("--report", action="store_true", help="print ledger performance")
    parser.add_argument("--verify", action="store_true", help="verify the hash chain only")
    parser.add_argument(
        "--symbols", default="NVDAUSDT",
        help=f"comma-separated; any of {','.join(RTOKEN_SYMBOLS)}",
    )
    args = parser.parse_args()

    if args.verify or args.report:
        ledger = PaperLedger(path=LEDGER_PATH)
        payload = ledger.verify() if args.verify else ledger.performance()
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.once:
        try:
            result = run_once(symbols=tuple(s.strip() for s in args.symbols.split(",")))
        except QwenError as exc:
            print(f"LLM unavailable: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, default=str))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
