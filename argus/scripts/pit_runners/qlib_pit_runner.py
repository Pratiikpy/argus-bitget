"""Run Qlib's real Point-in-Time database on a frozen SEC snapshot, in its own process.

Not imported by ARGUS. Qlib (microsoft/qlib, MIT) is run from its own clone at
``research/repos/qlib-upstream``, with its own venv (``research/_venvs/qlib-pit``, a plain venv
with the handful of pure-Python packages ``qlib.utils``/``qlib.config``/``scripts/dump_pit.py``
import — no Cython build; ``qlib``'s compiled extensions are never touched). Two real pieces of
Qlib's own code run here:

* **``scripts/dump_pit.py``'s ``DumpPitData``** builds Qlib's real on-disk PIT store (the
  ``<field>.data``/``<field>.index`` linked-list format documented in ``docs/advanced/PIT.rst``)
  from a CSV of ``date, period, value, field`` rows — Qlib's own documented input format. The CSV
  is prepared here from the snapshot: for each ticker, the alias tag ARGUS itself would pick for a
  period (``argus/market/fundamentals.py``'s ``CONCEPTS["revenue"]`` priority order, hardcoded
  below since this venv does not import ARGUS) supplies every revision of that period, dated by its
  **filed date** (Qlib's PIT store is day-granular by design — ``date`` is "the statement's
  date of publication", ``PIT.rst``) — this is the concept-alias cleanup any real Qlib PIT
  deployment does upstream of ``dump_pit.py``, which itself has no tag-disambiguation feature.
* **``qlib/utils/__init__.py::read_period_data(index_path, data_path, period, cur_date_int,
  quarterly)``** is Qlib's own low-level query: "at ``cur_date``, read the information at
  ``period``. Only the updating info before cur_date or at cur_date will be used." It answers one
  named period, not "the latest one" — so, exactly as a caller of this real, low-level function
  must, this runner tries every period known for the ticker from newest to oldest and keeps the
  first with a non-NaN value, which is the plain, undecorated way to use it for "the latest
  quarter's revenue as of X".

The probes come from :mod:`dump_pit_probes` (the exact 620 (ticker, accession, side) triples
``eval/pit_rivals.py`` scores everyone on), each with an ``as_of`` instant; Qlib's ``cur_date_int``
takes that instant's UTC calendar date (Qlib's own format carries no time of day, so this is the
only faithful reduction — this is the fairness caveat ``eval/pit_rivals.py``'s module docstring
names: a probe an hour *before* an acceptance on the filed day and Qlib's day-granular gate cannot
tell it from an hour *after*).

**Found by running it: Qlib's own dump pipeline loses precision on large values.**
``scripts/dump_pit.py::DumpPitData.get_source_data`` (this clone) does
``df[self.value_column_name] = df[self.value_column_name].astype("float32")`` before every value is
packed into the store — unconditionally, regardless of ``DATA_DTYPE``'s ``value`` field being
float64 (``qlib/config.py``'s ``pit_record_type``). Revenue in the billions has more significant
digits than float32 carries, so the stored figure comes back rounded (NVDA's 2,173,000,000 as
2,172,999,936.0). Checked against ``eval/pit_rivals.py``'s scoring: every one of this arm's
``stale_after_restatement`` outcomes (right quarter, "wrong" value) is within 1e-5 relative of the
true value — this rounding, not a real data or period error. The scoring keeps the same exact-match
"right" definition every other rival is held to (no per-rival tolerance), so the raw ``right`` count
understates Qlib's period-selection accuracy on its own; ``eval/pit_rivals.py``'s report and
``data/pit_rivals_report.md`` say so.

Usage::

    python qlib_pit_runner.py --clone DIR --snapshot DIR --probes probes.json --tickers NVDA \
        --out out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

# The same priority order as argus/market/fundamentals.py CONCEPTS["revenue"] — copied, not
# imported, because this venv does not (and should not) install ARGUS.
REVENUE_TAGS = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
QUARTER_DAYS = (80, 100)


def _quarterly_rows(companyfacts: dict[str, Any]) -> list[dict[str, Any]]:
    """Every quarterly USD row of the best-ranked revenue tag, per period end, across the whole
    history (not time-limited: the tag a period is reported under does not change with as_of)."""
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    by_end_tag_rank: dict[date, int] = {}
    raw_by_tag: dict[str, list[dict[str, Any]]] = {}
    for rank, tag in enumerate(REVENUE_TAGS):
        units = (gaap.get(tag) or {}).get("units", {}).get("USD") or []
        rows = []
        for row in units:
            try:
                end = date.fromisoformat(str(row["end"]))
                start = date.fromisoformat(str(row["start"]))
                filed = date.fromisoformat(str(row["filed"]))
                value = float(row["val"])
            except (KeyError, ValueError, TypeError):
                continue
            if not (QUARTER_DAYS[0] <= (end - start).days <= QUARTER_DAYS[1]):
                continue
            rows.append({"end": end, "filed": filed, "value": value, "form": row.get("form", "")})
            prior = by_end_tag_rank.get(end)
            if prior is None or rank < prior:
                by_end_tag_rank[end] = rank
        raw_by_tag[tag] = rows
    out = []
    for rank, tag in enumerate(REVENUE_TAGS):
        for row in raw_by_tag[tag]:
            if by_end_tag_rank[row["end"]] == rank:
                out.append(row)
    return out


def _period_of(end: date) -> int:
    """Qlib's ``<year><quarter-index>`` period id (``PIT.rst``): the calendar quarter ``end``
    falls in. A fiscal calendar drifting a few days around a quarter boundary can, in principle,
    put two distinct period-ends in one calendar quarter; callers of this function check for that
    collision and drop the ticker rather than silently mislabel a period."""
    quarter = (end.month - 1) // 3 + 1
    return end.year * 100 + quarter


def _build_store(rows: list[dict[str, Any]], qlib_dir: Path, symbol: str,
                 dump_module: Any) -> tuple[dict[int, date], list[str]]:
    """Write Qlib's real CSV input and call its real ``DumpPitData.dump`` to build the on-disk PIT
    store. Returns the period->end map and any collision notes.

    Qlib's period id is a calendar quarter (``PIT.rst``'s ``<year><quarter-index>``), coarser than
    a fiscal quarter-end date. A fiscal calendar that drifts across a quarter boundary can put two
    distinct period-ends in the same calendar quarter (found here: NVDA's fiscal Q2 end moved from
    2010-07-31 to 2010-08-01, both Q3 by calendar month). Real, not a harness artefact — Qlib's own
    period integer cannot represent both, so the earlier of the two is dropped and noted; the later
    one (closer to every probe this benchmark asks, all 2017+) is kept."""
    ends_by_period: dict[int, set[date]] = {}
    for row in rows:
        ends_by_period.setdefault(_period_of(row["end"]), set()).add(row["end"])
    period_to_end: dict[int, date] = {}
    notes: list[str] = []
    for period, ends in ends_by_period.items():
        chosen = max(ends)
        period_to_end[period] = chosen
        if len(ends) > 1:
            notes.append(f"period {period}: {sorted(ends)} collide; kept {chosen}")
    kept_ends = set(period_to_end.values())
    rows = [r for r in rows if r["end"] in kept_ends]
    csv_dir = qlib_dir / "source"
    csv_dir.mkdir(parents=True, exist_ok=True)
    lines = ["date,period,value,field"]
    for row in sorted(rows, key=lambda r: (r["filed"], r["end"])):
        lines.append(f"{row['filed'].isoformat()},{_period_of(row['end'])},{row['value']},revenue")
    (csv_dir / f"{symbol}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump_module.DumpPitData(
        csv_path=str(csv_dir), qlib_dir=str(qlib_dir), max_workers=1
    ).dump(interval="quarterly", overwrite=True)
    return period_to_end, notes


def _latest_as_of(read_period_data: Any, index_path: Path, data_path: Path,
                  periods_desc: list[int], cur_date_int: int) -> tuple[int, float] | None:
    """The plain use of ``read_period_data`` for "the latest quarter as of cur_date": it answers
    one named period, so try every known period, newest first, and keep the first non-NaN value."""
    for period in periods_desc:
        value, _ = read_period_data(str(index_path), str(data_path), period, cur_date_int, True)
        if not math.isnan(value):
            return period, value
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.clone)))
    sys.path.insert(0, str(Path(args.clone) / "scripts"))
    import dump_pit as dump_module  # Qlib's real scripts/dump_pit.py
    from qlib.utils import read_period_data

    snapshot = Path(args.snapshot)
    probes = json.loads(Path(args.probes).read_text(encoding="utf-8"))["tickers"]

    out: dict[str, Any] = {
        "runner": "qlib_pit_runner", "clone": Path(args.clone).resolve().name, "tickers": {}
    }
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        companyfacts = json.loads(
            (snapshot / f"{ticker}.companyfacts.json").read_text(encoding="utf-8"))
        rows = _quarterly_rows(companyfacts)
        ticker_probes = probes.get(ticker, [])
        if not rows or not ticker_probes:
            out["tickers"][ticker] = {"probes": [], "note": "no quarterly revenue rows"}
            continue
        with tempfile.TemporaryDirectory(prefix=f"qlib_pit_{ticker}_") as tmp:
            qlib_dir = Path(tmp)
            symbol = ticker.lower()
            period_to_end, notes = _build_store(rows, qlib_dir, symbol, dump_module)
            index_path = qlib_dir / "financial" / symbol / "revenue_q.index"
            data_path = qlib_dir / "financial" / symbol / "revenue_q.data"
            periods_desc = sorted(period_to_end, reverse=True)
            answered = []
            for probe in ticker_probes:
                as_of = datetime.fromisoformat(probe["as_of"])
                if as_of.tzinfo is None:
                    as_of = as_of.replace(tzinfo=UTC)
                cur_date_int = int(as_of.astimezone(UTC).date().strftime("%Y%m%d"))
                found = _latest_as_of(read_period_data, index_path, data_path, periods_desc,
                                      cur_date_int)
                answered.append({
                    "accn": probe["accn"], "side": probe["side"], "as_of": probe["as_of"],
                    "end": period_to_end[found[0]].isoformat() if found else None,
                    "value": found[1] if found else None,
                })
            out["tickers"][ticker] = {"probes": answered, "notes": notes}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
