"""The backtest critic against backtest-truth, on the same planted bugs and the same clean code.

``research/backtest_critic.py`` reads eight kinds of backtest error from a strategy's code. Its
closest rival, ``MrMaca11an/backtest-truth`` (MIT, cloned at ``research/repos-t2/backtest-truth``),
is a static linter for the first of them. Both are run here on one fixture set:

* one planted bug per check, in Python (and in Pine Script where the check applies there) — the
  question is whether each tool names it;
* clean code written the way a careful quant writes it, including the patterns a naive linter
  trips on: a Sharpe ratio from ``.mean()`` and ``.std()`` after the fact, ``train_test_split`` with
  ``shuffle=False``, a drawdown from ``cummax()``, ``merge_asof`` with its backward default, a
  signal lagged inside a function — the question is whether each tool stays quiet.

A bug counts as caught when the tool raises a finding of the right kind for that fixture (for
backtest-truth, any ERROR or WARNING code it maps to that kind; INFO notes such as "fit() detected,
confirm" count as a catch only on the look-ahead fixtures they are about, and as a false alarm on
clean code). backtest-truth has no rule for most of the eight kinds; a kind it has no rule for is
reported as not covered rather than as a miss, and its score is given on the kinds it covers too.

    python -m argus.eval.backtest_critic_comparison   # data/backtest_critic_comparison.json
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argus.research.backtest_critic import CHECKS, PRESENT, review
from argus.truth.paths import DATA_DIR

RIVAL = Path(__file__).resolve().parents[4] / "research" / "repos-t2" / "backtest-truth"
OUT = DATA_DIR / "backtest_critic_comparison.json"

RIVAL_KINDS = {"BT001": "look-ahead", "BT002": "look-ahead", "BT003": "look-ahead",
               "BT004": "look-ahead", "BT005": "look-ahead", "BT006": "fills"}
"""backtest-truth's static codes and the check each belongs to (`backtest_truth/rules.py`)."""


@dataclass(frozen=True)
class Fixture:
    name: str
    kind: str | None
    """The planted check, or None for clean code that should raise nothing."""
    code: str


_HEAD = "import pandas as pd\nimport numpy as np\n\n"
_RUN = ("\n\ndef backtest(df):\n    sig = signal(df)\n    ret = df['close'].pct_change()\n"
        "    fee = 0.0006\n    pnl = sig.shift(1) * ret - fee * sig.diff().abs()\n"
        "    return pnl['2019-01-01':'2025-06-30'].cumsum()\n")

FIXTURES: tuple[Fixture, ...] = (
    Fixture("negative shift", "look-ahead", _HEAD + "def signal(df):\n"
            "    tomorrow = df['close'].shift(-1)\n    return (tomorrow > df['close']).astype(int)"
            + _RUN),
    Fixture("centred window", "look-ahead", _HEAD + "def signal(df):\n"
            "    mid = df['close'].rolling(20, center=True).mean()\n"
            "    return (df['close'] > mid).astype(int)" + _RUN),
    Fixture("whole-series threshold", "look-ahead", _HEAD + "def signal(df):\n"
            "    level = df['close'].quantile(0.8)\n    return (df['close'] > level).astype(int)"
            + _RUN),
    Fixture("back-fill", "look-ahead", _HEAD + "def signal(df):\n"
            "    vol = df['volume'].bfill()\n"
            "    return (vol > vol.rolling(20).mean()).astype(int)" + _RUN),
    Fixture("shuffled split", "look-ahead", _HEAD + "from sklearn.model_selection import "
            "train_test_split\nfrom sklearn.linear_model import LogisticRegression\n\n"
            "def signal(df):\n    X, y = df[['rsi']], (df['close'].pct_change() > 0)\n"
            "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)\n"
            "    model = LogisticRegression().fit(X_train, y_train)\n"
            "    return pd.Series(model.predict(X), index=df.index)" + _RUN),
    Fixture("future index", "look-ahead", _HEAD + "def signal(df):\n    out = []\n"
            "    for i in range(len(df) - 1):\n"
            "        out.append(int(df['close'].iloc[i + 1] > df['close'].iloc[i]))\n"
            "    return pd.Series(out + [0], index=df.index)" + _RUN),
    Fixture("survivorship", "survivorship", _HEAD
            + "TICKERS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AVGO', 'LLY', "
              "'JPM', 'V', 'UNH', 'XOM', 'MA', 'COST', 'HD', 'PG', 'NFLX']\n\n"
              "def signal(df):\n    return (df['close'] > df['close'].rolling(50).mean())"
              ".astype(int)" + _RUN),
    Fixture("resampled join", "repainting", _HEAD + "def signal(df):\n"
            "    daily = df['close'].resample('1D').last()\n"
            "    trend = daily.reindex(df.index).ffill()\n"
            "    return (df['close'] > trend).astype(int)" + _RUN),
    Fixture("no costs", "costs", _HEAD + "def signal(df):\n"
            "    return (df['close'] > df['close'].rolling(50).mean()).astype(int)\n\n\n"
            "def backtest(df):\n    sig = signal(df).shift(1)\n"
            "    pnl = sig * df['close'].pct_change()\n"
            "    return pnl['2019-01-01':'2025-06-30'].cumsum()\n"),
    Fixture("same-bar fill", "fills", _HEAD + "def signal(df):\n"
            "    return (df['close'] > df['close'].rolling(50).mean()).astype(int)\n\n\n"
            "def backtest(df):\n    sig = signal(df)\n    fee = 0.0006\n"
            "    pnl = sig * df['close'].pct_change() - fee * sig.diff().abs()\n"
            "    return pnl['2019-01-01':'2025-06-30'].cumsum()\n"),
    Fixture("trade on close", "fills", _HEAD + "from backtesting import Backtest\n\n"
            "def run(data, Strat):\n"
            "    bt = Backtest(data, Strat, commission=0.0006, trade_on_close=True)\n"
            "    return bt.run()  # 2019-01-01 to 2025-06-30\n"),
    Fixture("grid search, no split", "parameter fitting", _HEAD + "import itertools\n\n"
            "def signal(df, fast, slow):\n"
            "    return (df['close'].rolling(fast).mean() > df['close'].rolling(slow).mean())"
            ".astype(int)\n\n\ndef best(df):\n    fee = 0.0006\n    scores = {}\n"
            "    for fast, slow in itertools.product(range(5, 50, 5), range(50, 200, 10)):\n"
            "        sig = signal(df, fast, slow).shift(1)\n"
            "        pnl = sig * df['close'].pct_change() - fee * sig.diff().abs()\n"
            "        scores[(fast, slow)] = pnl.sum()\n"
            "    return max(scores, key=scores.get)  # 2019-01-01 to 2025-06-30\n"),
    Fixture("one bull year", "regime coverage", _HEAD + "def signal(df):\n"
            "    return (df['close'] > df['close'].rolling(50).mean()).astype(int)\n\n\n"
            "def backtest(df):\n    sig = signal(df).shift(1)\n    fee = 0.0006\n"
            "    pnl = sig * df['close'].pct_change() - fee * sig.diff().abs()\n"
            "    return pnl['2023-01-01':'2023-12-31'].cumsum()\n"),
    Fixture("forward as-of join", "data alignment", _HEAD + "def signal(df, funding):\n"
            "    joined = pd.merge_asof(df, funding, on='time', direction='nearest')\n"
            "    return (joined['rate'] < 0).astype(int)" + _RUN.replace("signal(df)",
                                                                       "signal(df, df)")),
    Fixture("pine lookahead_on", "repainting", "//@version=5\nstrategy('htf', overlay=true, "
            "commission_type=strategy.commission.percent, commission_value=0.06)\n"
            "htf = request.security(syminfo.tickerid, 'D', close, "
            "lookahead=barmerge.lookahead_on)\nif close > htf\n    strategy.entry('L', "
            "strategy.long)\n"),
    Fixture("pine no commission", "costs", "//@version=5\nstrategy('ma')\n"
            "fast = ta.sma(close, 10)\nslow = ta.sma(close, 30)\n"
            "if ta.crossover(fast, slow)\n    strategy.entry('L', strategy.long)\n"),
    Fixture("pine fill on close", "fills", "//@version=5\nstrategy('ma', "
            "process_orders_on_close=true, commission_value=0.06)\nfast = ta.sma(close, 10)\n"
            "if ta.crossover(fast, ta.sma(close, 30))\n    strategy.entry('L', strategy.long)\n"),
    # clean code, with the patterns a naive linter flags
    Fixture("clean: lagged inside a function", None, _HEAD + "def signal(df):\n"
            "    raw = (df['close'] > df['close'].rolling(50).mean()).astype(int)\n"
            "    return raw.shift(1).fillna(0)\n\n\ndef backtest(df):\n    sig = signal(df)\n"
            "    fee = 0.0006\n    pnl = sig * df['close'].pct_change() - fee * sig.diff().abs()\n"
            "    pnl = pnl['2019-01-01':'2025-06-30']\n"
            "    sharpe = pnl.mean() / pnl.std() * np.sqrt(365)\n"
            "    equity = (1 + pnl).cumprod()\n    drawdown = equity / equity.cummax() - 1\n"
            "    return sharpe, drawdown.min()\n"),
    Fixture("clean: time-ordered split", None, _HEAD + "from sklearn.model_selection import "
            "train_test_split\nfrom sklearn.linear_model import LogisticRegression\n\n"
            "def signal(df):\n    X, y = df[['rsi']].shift(1), (df['close'].pct_change() > 0)\n"
            "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, "
            "shuffle=False)\n    model = LogisticRegression().fit(X_train, y_train)\n"
            "    return pd.Series(model.predict(X), index=df.index)" + _RUN),
    Fixture("clean: backward as-of join, walk-forward search", None, _HEAD + "import itertools\n\n"
            "def walk_forward(df, funding):\n"
            "    joined = pd.merge_asof(df, funding, on='time')\n    fee = 0.0006\n"
            "    for train, test in folds(joined):  # 2019-01-01 to 2025-06-30\n"
            "        best = max(itertools.product(range(5, 50, 5)),\n"
            "                   key=lambda p: score(train, p))\n"
            "        yield score(test, best) - fee\n"),
    Fixture("clean: pine", None, "//@version=5\nstrategy('htf', commission_type="
            "strategy.commission.percent, commission_value=0.06, slippage=2)\n"
            "htf = request.security(syminfo.tickerid, 'D', close[1], "
            "lookahead=barmerge.lookahead_off)\nif close > htf\n    strategy.entry('L', "
            "strategy.long)\n"),
)


def _rival_checker() -> Any:
    """backtest-truth's ``check_source``, imported from its clone (it is not installed)."""
    from importlib import import_module

    sys.path.insert(0, str(RIVAL))
    try:
        return import_module("backtest_truth.ast_checks").check_source
    finally:
        sys.path.remove(str(RIVAL))


def _rival(code: str) -> dict[str, list[str]] | None:
    """backtest-truth's static findings on ``code``, by check, with their severities."""
    if not RIVAL.exists():
        return None
    check_source = _rival_checker()
    found: dict[str, list[str]] = {}
    for finding in check_source(code):
        kind = RIVAL_KINDS.get(finding.code)
        if kind is not None:
            found.setdefault(kind, []).append(f"{finding.code}:{finding.severity.value}")
    return found


def run() -> dict[str, Any]:
    rows = []
    for fixture in FIXTURES:
        ours = review(fixture.code)
        flagged = [k for k, c in ours.checks.items() if c.state == PRESENT]
        theirs = _rival(fixture.code)
        if fixture.code.startswith("//@"):
            theirs = None  # backtest-truth parses Python only
        rival_flagged = sorted(theirs or {})
        rival_loud = sorted(k for k, v in (theirs or {}).items()
                            if any(not s.endswith(":info") for s in v))
        rows.append({
            "fixture": fixture.name, "planted": fixture.kind,
            "argus_flags": flagged,
            "argus_caught": fixture.kind in flagged if fixture.kind else None,
            "argus_false_alarms": list(flagged) if fixture.kind is None else [],
            "rival_flags": rival_flagged if theirs is not None else "not run (not Python)",
            "rival_covers": fixture.kind in set(RIVAL_KINDS.values()) if fixture.kind else None,
            "rival_caught": (fixture.kind in rival_flagged if fixture.kind and theirs is not None
                             else None),
            "rival_false_alarms": rival_loud if fixture.kind is None and theirs is not None
            else [],
        })
    planted = [r for r in rows if r["planted"]]
    clean = [r for r in rows if not r["planted"]]
    covered = [r for r in planted if r["rival_covers"] and r["rival_caught"] is not None]
    return {
        "fixtures": len(rows), "planted": len(planted), "clean": len(clean),
        "argus_caught": sum(bool(r["argus_caught"]) for r in planted),
        "argus_false_alarm_fixtures": sum(bool(r["argus_false_alarms"]) for r in clean),
        "rival_caught": sum(bool(r["rival_caught"]) for r in planted),
        "rival_kinds_covered": sorted(set(RIVAL_KINDS.values())),
        "rival_caught_on_kinds_it_covers": f"{sum(bool(r['rival_caught']) for r in covered)} of "
                                            f"{len(covered)}",
        "argus_caught_on_same": f"{sum(bool(r['argus_caught']) for r in covered)} of "
                                f"{len(covered)}",
        "rival_false_alarm_fixtures": sum(bool(r["rival_false_alarms"]) for r in clean),
        "rows": rows, "checks": list(CHECKS),
    }


INDEPENDENT = RIVAL.parent.parent / "repos-themed" / "marketcalls~vectorbt-backtesting-skills"
"""Backtest scripts neither tool was written against: marketcalls' vectorbt examples (four Python
files). Every flag either tool raises on them was read against the source by hand on 2026-10-04;
:func:`read_by_hand` records the two shapes they all took."""
def read_by_hand(line: str) -> str:
    """The verdict on one flagged line, from reading every flag either tool raised on the sixteen
    scripts on 2026-10-04: each was one of two shapes. A back-fill of a price series
    (``.reindex(...).ffill().bfill()``) fills its first rows from later closes — a real leak, if
    a small one, into a benchmark or an open-price series. A count or summary statistic
    (``entries.sum()`` in a print, ``mask.sum() == 0``, ``results_df['oos_return'].mean()``)
    decides nothing. A line of neither shape was not read and is said to be unread."""
    text = line.strip()
    if ".bfill()" in text or "method='bfill'" in text or 'method="bfill"' in text:
        return "true: a price series back-filled from later values"
    if re.search(r"\.(?:sum|mean|min|max|std)\(\)", text) and (
            text.startswith(("print(", "f\"", "\"", "if ")) or re.match(
                r"^\w+\s*=\s*[\w\[\]'\".()]+\.(?:sum|mean|min|max|std)\(\)", text)
            or "param_stable" in text or "no_action" in text):
        return "false: a count or summary statistic, deciding nothing"
    return "unread"




def independent() -> dict[str, Any] | None:
    """Both tools on code neither was written against, each flag scored by the hand reading."""
    if not INDEPENDENT.exists() or not RIVAL.exists():
        return None
    check_source = _rival_checker()
    rows: list[dict[str, Any]] = []
    for path in sorted(INDEPENDENT.rglob("*.py")):
        code = path.read_text(encoding="utf-8")
        at = code.splitlines()
        name = path.relative_to(INDEPENDENT).as_posix()
        ours = [(name, f.line) for c in review(code).checks.values() if c.state == PRESENT
                for f in c.findings if c.name != "regime coverage"]
        theirs = sorted({(name, int(f.location.rsplit(":", 1)[1]))
                         for f in check_source(code) if f.severity.value != "info"})
        rows.append({"file": name,
                     "argus": [[n, line, read_by_hand(at[line - 1]) if line else "unread"]
                               for n, line in ours],
                     "rival": [[n, line, read_by_hand(at[line - 1]) if line else "unread"]
                               for n, line in theirs]})

    def tally(side: str) -> dict[str, int]:
        verdicts = [v for r in rows for _, _, v in r[side]]
        return {"true": sum(v.startswith("true") for v in verdicts),
                "false": sum(v.startswith("false") for v in verdicts),
                "unread": sum(v == "unread" for v in verdicts)}

    return {"files": len(rows), "argus": tally("argus"), "rival": tally("rival"), "rows": rows}


def _rival_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(RIVAL), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:  # pragma: no cover - CLI
    result = run()
    result["independent"] = independent()
    # Both runs read only the fixtures and the clones, with nothing random or timed in the rows,
    # so a second run must reproduce the first exactly; it is run and compared, not assumed.
    again = {**run(), "independent": independent()}
    result["reference"] = {
        "rival": "MrMaca11an/backtest-truth", "licence": "MIT", "commit": _rival_commit(),
        "entry": "backtest_truth check_source, imported from the clone and run unmodified on "
                 "every fixture and independent script"}
    result["reproducibility"] = {
        "command": "python -m argus.eval.backtest_critic_comparison",
        "identical_on_rerun": again == {k: v for k, v in result.items()
                                        if k not in ("reference", "reproducibility")}}
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    for row in result["rows"]:
        print(f"{row['fixture']:<45} planted={row['planted']!s:<18} argus={row['argus_flags']} "
              f"rival={row['rival_flags']}")
    print({k: v for k, v in result.items() if k not in ("rows", "checks", "independent")})
    print("independent:", {k: v for k, v in (result["independent"] or {}).items() if k != "rows"})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
