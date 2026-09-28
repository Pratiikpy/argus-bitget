"""The console's own calls, recorded before the outcome and graded after it.

A research answer states things that the future settles: the book's open-session beta after the
trade, the share of the book's risk the added name will carry, and, from the long-run analogue, how
often a state like today's has gone up over the next 1, 5 and 20 trading days. Until 2026-09-27
none of it was kept, so no answer the console gave could be checked against what happened (audit
finding 24). This module keeps them.

**Recording.** Once a UTC day, each question in :data:`QUESTIONS` is run through the research task
exactly as ``/research`` runs it (`lui/task.research_task`, no model call), and the figures above
are appended to ``data/call_record.jsonl`` with the time. Each row carries the SHA-256 of the row
before it, so a row edited or dropped after the fact breaks the chain for anyone who checks it
(:func:`verify`). A day whose rows exist is not recorded again.

**Grading.** A call is graded only once its horizon has passed, on data that did not exist when it
was made:

* beta and risk share: :data:`HORIZON_DAYS` calendar days after the call, the book as it would be
  after the trade is valued over exactly those days by the same function the answer used
  (`desk/portfolio.copilot`), and the error is the absolute difference;
* direction: for each analogue horizon, whether the name's close that many trading days later was
  above its close on the day of the call. The stated up-rate is scored by its Brier score beside
  the Brier score of the plain base rate the same answer printed, so a conditional forecast that
  does no better than "every day" shows as no skill.

Nothing is dropped for being inconvenient: a call whose data cannot be read is counted as
ungraded and says why. The grades are written to ``data/call_grades.json``.

    python -m argus.eval.call_record --record --grade     # needs the network
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parents[3]
RECORD_PATH = PACKAGE / "data" / "call_record.jsonl"
GRADES_PATH = PACKAGE / "data" / "call_grades.json"

QUESTIONS: tuple[str, ...] = (
    "I hold 40% NVDA, 30% MSFT, 30% AAPL — should I add 15% TSLA?",
    "I hold 50% NVDA, 50% AAPL — should I add 20% COIN?",
    "I hold 60% QQQ, 40% MSFT — should I add 10% MSTR?",
    "I hold 40% TSLA, 30% META, 30% GOOGL — should I add 15% AMZN?",
)
"""Fixed on 2026-09-27 and never tuned: four books a trader of Bitget's stock perpetuals holds,
each asked to add a different name, so the record spans chip, crypto-linked and consumer names."""

HORIZON_DAYS = 28
"""Calendar days a beta or risk-share call is graded over: four weeks, the horizon the console's
beta figure is measured against elsewhere (`eval/copilot_rivals.py`)."""

GENESIS = "0" * 64


class CallRecordError(RuntimeError):
    """The record cannot be read honestly."""


def _digest(row: Mapping[str, Any]) -> str:
    body = {k: v for k, v in row.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def read(path: Path = RECORD_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify(rows: Sequence[Mapping[str, Any]]) -> None:
    """Raise unless every row hashes to its own ``hash`` and names the row before it."""
    previous = GENESIS
    for index, row in enumerate(rows):
        if row.get("prev") != previous:
            raise CallRecordError(f"row {index} does not follow the row before it")
        if row.get("hash") != _digest(row):
            raise CallRecordError(f"row {index} was changed after it was written")
        previous = str(row["hash"])


def predictions(task: Any) -> dict[str, Any]:
    """The figures of one research task that the future settles."""
    from argus.lui.task import IMPACT_TITLE

    by_title = {step.title: step for step in task.steps}
    out: dict[str, Any] = {"name": task.name}
    impact = by_title.get(IMPACT_TITLE)
    report = (impact.data.get("report") if impact is not None else None) or {}
    block = report.get("impact") or {}
    if block:
        out.update(
            symbol=report.get("symbol"), benchmark=report.get("benchmark"),
            weights_before=report.get("weights_before"),
            weights_after=report.get("weights_after"),
            beta_after=block.get("beta_after"), risk_share_after=block.get("risk_share_after"),
        )
    analogue = by_title.get("Has it been here before?")
    long_run = (analogue.data.get("long_run") if analogue is not None else None) or {}
    out["direction"] = [
        {"days": int(h["days"]), "up": float(h["up"]), "base_up": float(h["base_up"])}
        for h in long_run.get("horizons", [])
    ]
    return out


def record(now: datetime | None = None, *, path: Path = RECORD_PATH,
           run: Callable[[str], Any] | None = None) -> list[dict[str, Any]]:
    """Record today's calls for every question not yet recorded today; return the new rows."""
    from argus.lui.task import read_question, research_task

    now = now or datetime.now(UTC)
    rows = read(path)
    verify(rows)
    today = now.date().isoformat()
    done = {r["question"] for r in rows if str(r["recorded_at"])[:10] == today}

    def default(question: str) -> Any:
        reading = read_question(question)
        if isinstance(reading, str):
            raise CallRecordError(f"the question was not read: {reading}")
        return research_task(reading=reading, asked=question)

    ask = run or default
    added: list[dict[str, Any]] = []
    previous = str(rows[-1]["hash"]) if rows else GENESIS
    for question in QUESTIONS:
        if question in done:
            continue
        task = ask(question)
        verdict = task.verdict
        row: dict[str, Any] = {
            "recorded_at": now.isoformat(timespec="seconds"),
            "question": question,
            "verdict": None if verdict is None else verdict.call,
            "predictions": predictions(task),
            "prev": previous,
        }
        row["hash"] = _digest(row)
        previous = row["hash"]
        added.append(row)
    if added:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in added:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return added


# --- grading --------------------------------------------------------------------------------------

Loader = Callable[[Sequence[str], int], Mapping[str, Mapping[datetime, float]]]
Closes = Callable[[str], Mapping[date, float]]


def _default_loader(symbols: Sequence[str], days: int) -> Mapping[str, Mapping[datetime, float]]:
    from argus.lui.research import load

    data = load(symbols, days=days)
    if not data.live:
        raise CallRecordError("live candles did not arrive; a frozen series cannot grade a call")
    return data.raw


def _default_closes(symbol: str) -> Mapping[date, float]:
    from argus.lui.exposures import closes_for

    return closes_for(symbol)[0]


def grade_risk(row: Mapping[str, Any], now: datetime, loader: Loader) -> dict[str, Any] | None:
    """The beta and risk-share call against the book's realised figures over the horizon, or
    None while the horizon is still open."""
    from argus.desk.portfolio import copilot
    from argus.lui.research import BENCHMARK
    from argus.lui.research.session import anchor_is_open

    p = row["predictions"]
    if p.get("beta_after") is None:
        return {"ungraded": "the answer stated no beta"}
    start = datetime.fromisoformat(str(row["recorded_at"]))
    end = start + timedelta(days=HORIZON_DAYS)
    if now < end:
        return None
    before = {str(k): float(v) for k, v in (p.get("weights_before") or {}).items()}
    after = {str(k): float(v) for k, v in (p.get("weights_after") or {}).items()}
    add = str(p["symbol"])
    size = after.get(add, 0.0)  # rebalance() sets the added name to this share of the book
    try:
        raw = loader(sorted({*after, BENCHMARK}), (now - start).days + 2)
    except Exception as exc:
        return {"ungraded": f"candles unreadable ({type(exc).__name__})"}
    window = {s: {t: r for t, r in series.items() if start <= t < end}
              for s, series in raw.items()}
    try:
        report = copilot(add=add, before=before, size=size, raw=window,
                         benchmark=str(p.get("benchmark") or BENCHMARK), is_open=anchor_is_open())
    except Exception as exc:
        return {"ungraded": f"the window could not be valued ({type(exc).__name__})"}
    realised = report.as_dict()["impact"]
    out: dict[str, Any] = {"beta_after": realised["beta_after"],
                           "beta_error": abs(realised["beta_after"] - float(p["beta_after"]))}
    if p.get("risk_share_after") is not None and realised.get("risk_share_after") is not None:
        out["risk_share_after"] = realised["risk_share_after"]
        out["risk_share_error"] = abs(realised["risk_share_after"]
                                      - float(p["risk_share_after"]))
    return out


def grade_direction(row: Mapping[str, Any], closes: Closes) -> list[dict[str, Any]]:
    """Each analogue horizon whose close is in: whether it went up, and both Brier scores."""
    p = row["predictions"]
    if not p.get("direction"):
        return []
    day = datetime.fromisoformat(str(row["recorded_at"])).date()
    series = sorted(closes(str(p.get("symbol") or f"{p['name']}USDT")).items())
    after = [(d, c) for d, c in series if d > day]
    at = [c for d, c in series if d <= day]
    if not at:
        return []
    out = []
    for h in p["direction"]:
        days = int(h["days"])
        if len(after) < days:
            continue
        went_up = after[days - 1][1] > at[-1]
        outcome = 1.0 if went_up else 0.0
        out.append({"days": days, "up": went_up,
                    "brier": (float(h["up"]) - outcome) ** 2,
                    "brier_base": (float(h["base_up"]) - outcome) ** 2})
    return out


def grade(now: datetime | None = None, *, path: Path = RECORD_PATH,
          loader: Loader = _default_loader, closes: Closes = _default_closes) -> dict[str, Any]:
    """Every call graded that can be, and the totals."""
    now = now or datetime.now(UTC)
    rows = read(path)
    verify(rows)
    graded: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {"recorded_at": row["recorded_at"], "question": row["question"],
                                 "risk": grade_risk(row, now, loader), "direction": []}
        try:
            entry["direction"] = grade_direction(row, closes)
        except Exception as exc:  # counted as ungraded and named, never dropped
            entry["direction_ungraded"] = f"closes unreadable ({type(exc).__name__})"
        graded.append(entry)
    risk = [g["risk"] for g in graded if g["risk"] and "beta_error" in g["risk"]]
    shares = [r["risk_share_error"] for r in risk if "risk_share_error" in r]
    direction = [d for g in graded for d in g["direction"]]

    def mean(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "graded_at": now.isoformat(timespec="seconds"),
        "calls": len(rows),
        "first_call": rows[0]["recorded_at"] if rows else None,
        "risk_graded": len(risk),
        "mean_beta_error": mean([r["beta_error"] for r in risk]),
        "mean_risk_share_error": mean(shares),
        "direction_graded": len(direction),
        "brier": mean([d["brier"] for d in direction]),
        "brier_base_rate": mean([d["brier_base"] for d in direction]),
        "calls_graded": graded,
    }


# --- weekend bands --------------------------------------------------------------------------------

WEEKEND_PATH = PACKAGE / "data" / "weekend_calls.jsonl"
WEEKEND_TICKERS: tuple[str, ...] = ("NVDA", "TSLA", "AAPL", "MSFT", "COIN")
"""Fixed 2026-09-27: the stocks a weekend-hold question is most often about."""

DayLoader = Callable[[str], Sequence[Any]]


def _default_days(ticker: str) -> Sequence[Any]:
    from argus.market.equity_history import daily

    return daily(ticker)


def weekend_open(now: datetime) -> bool:
    """Whether the US market is shut for the weekend: Friday after 21:00 UTC (the close has
    printed in either season) through Sunday. The band is recorded then, before Monday's open
    that grades it."""
    return (now.weekday() == 4 and now.hour >= 21) or now.weekday() in (5, 6)


def record_weekend(now: datetime | None = None, *, path: Path = WEEKEND_PATH,
                   days_of: DayLoader = _default_days) -> list[dict[str, Any]]:
    """The weekend answer's 10/50/90 band (`equity_history.weekend_band`) for each of
    :data:`WEEKEND_TICKERS`, once per weekend, hash-chained like the daily calls. baserate keeps
    such a ledger of its own weekend forecasts, graded after the fact; until 2026-09-27 ARGUS
    printed the band and kept nothing (register row "A leveraged hold across the weekend")."""
    from argus.market.equity_history import weekend_band

    now = now or datetime.now(UTC)
    if not weekend_open(now):
        return []
    rows = read(path)
    verify(rows)
    done = {(r["ticker"], r["friday"]) for r in rows}
    previous = str(rows[-1]["hash"]) if rows else GENESIS
    added: list[dict[str, Any]] = []
    for ticker in WEEKEND_TICKERS:
        try:
            days = list(days_of(ticker))
        except Exception:
            continue  # nothing recorded for a stock that cannot be read; graded rows only
        if not days or days[-1].day.weekday() != 4:
            continue
        friday = days[-1].day.isoformat()
        if (ticker, friday) in done:
            continue
        band = weekend_band(days)
        if band is None:
            continue
        row: dict[str, Any] = {"recorded_at": now.isoformat(timespec="seconds"),
                               "ticker": ticker, "friday": friday,
                               "friday_close": days[-1].close, "band": band.as_dict(),
                               "prev": previous}
        row["hash"] = _digest(row)
        previous = row["hash"]
        added.append(row)
    if added:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in added:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return added


def grade_weekend(path: Path = WEEKEND_PATH, *,
                  days_of: DayLoader = _default_days) -> dict[str, Any]:
    """Each recorded band against the Monday (or first) open after its Friday: inside the 10-90
    band or not, and its pinball loss. A band whose reopen has not printed is pending."""
    from argus.market.equity_history import BAND_QUANTILES

    rows = read(path)
    verify(rows)
    history: dict[str, Sequence[Any]] = {}
    graded: list[dict[str, Any]] = []
    pending = 0
    for row in rows:
        ticker = str(row["ticker"])
        if ticker not in history:
            try:
                history[ticker] = list(days_of(ticker))
            except Exception as exc:
                history[ticker] = []
                graded.append({"ticker": ticker, "friday": row["friday"],
                               "ungraded": f"history unreadable ({type(exc).__name__})"})
                continue
        friday = date.fromisoformat(str(row["friday"]))
        reopen = next((d for d in history[ticker] if d.day > friday), None)
        if reopen is None:
            pending += 1
            continue
        move = reopen.open / float(row["friday_close"]) - 1
        band = row["band"]
        points = (band["p10"], band["p50"], band["p90"])
        loss = sum(max(q * (move - f), (q - 1) * (move - f))
                   for f, q in zip(points, BAND_QUANTILES, strict=True)) / 3
        graded.append({"ticker": ticker, "friday": row["friday"],
                       "reopened": reopen.day.isoformat(), "move": move,
                       "inside": band["p10"] <= move <= band["p90"],
                       "pinball_bps": 1e4 * loss})
    scored = [g for g in graded if "inside" in g]
    return {"recorded": len(rows), "graded": len(scored), "pending": pending,
            "covered": (sum(g["inside"] for g in scored) / len(scored)) if scored else None,
            "mean_pinball_bps": (sum(g["pinball_bps"] for g in scored) / len(scored)
                                 if scored else None),
            "bands": graded}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="record and grade the console's own calls")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--grade", action="store_true")
    args = parser.parse_args(argv)
    if args.record:
        added = record()
        print(f"recorded {len(added)} call(s)")
        weekend = record_weekend()
        print(f"recorded {len(weekend)} weekend band(s)")
    if args.grade:
        grades = grade()
        grades["weekend"] = grade_weekend()
        GRADES_PATH.write_text(json.dumps(grades, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8", newline="\n")
        print(f"{grades['calls']} calls, {grades['risk_graded']} risk-graded, "
              f"{grades['direction_graded']} direction-graded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
