"""A trader's claims about the tape, checked by ARGUS and by MirrorLine on the same sentences.

MirrorLine (PinnacleCryptNG, a Season 2 desk; no licence file, so nothing of its code is used here)
is built for one job: take what a trader asserts about a Bitget rToken — "rNVDA is up", "it rallied
because of earnings", "the US market is closed" — and mark each claim supported, challenged or
unsupported against a Bitget evidence pack, never letting a price confirm a stated cause. ARGUS's
premise check (`lui/research._claim_check`) does the same inside a research answer. This module
scores both on one set of sentences asked the same morning.

**How it was run** (`data/h2h_mirrorline/`): MirrorLine's own ``getInterpretationChallenge`` from
its clone, through a ten-line runner that passes each sentence as the thesis; ARGUS through
``server.handle_ask``, the console's own entry point. Bitget's perpetual and spot 24-hour changes
were captured before and after both runs (``tape.json``).

**Truth.** A direction claim is scored only where the perpetual and the rToken agree on the sign,
before and after the runs, and every one of those four readings is at least 0.30% from zero; the
rest are *flat* and reported apart, because either desk can defensibly call a 0.2% day either way.
A causal claim is right when its direction part is right and its cause is NOT confirmed. A session
claim is right when it matches the NYSE clock at the start of the run.

**The first run was a loss and is kept** (``argus_raw_first_run.json``): MirrorLine judged every
claim; ARGUS judged the direction claims and missed every past-tense causal claim ("META rallied
today because…") and both session claims. Both gaps were fixed the same morning and both desks
were run again; this module scores whichever run its artefacts hold and says which.

**What is scored for ARGUS (changed 2026-09-26).** Until then this module graded the console
answers recorded in ``argus_raw.json`` while `eval/standing.py` credited
`lui/research._claim_check`,
and the harness-validity canary (`data/harness_validity.json`) found that function never ran at
scoring time — a regression in it would have left this verdict standing. Now the harness calls
``_claim_check`` itself on every saved sentence (:func:`console_answers`): the contract is the one
the console's own resolver names (``research_symbols``, as ``research.run`` passes
``request.symbols[0]``), the 24-hour changes come from ``tape.json`` through the console's ticker
reader, and the clock is held at the moment the run started, which is what the session claims are
graded against. The recorded ``handle_ask`` answers are still read, and how far the replay agrees
with them is published (``replay``), so a difference between the function today and the console
that morning is on the record, not hidden. The replay skips the model planner that routed the
morning's questions (``classified_by``): the check credited is ``_claim_check``, and it receives
the contract the planner would have named.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator, Mapping
from datetime import datetime, tzinfo
from decimal import Decimal
from pathlib import Path
from typing import Any, Self

from argus.eval import artefact
from argus.eval.compare import ComparisonReport, finalise, sign_test, sign_test_outcome

DATA = Path(__file__).resolve().parents[3] / "data" / "h2h_mirrorline"
REPORT = Path(__file__).resolve().parents[3] / "data" / "claimcheck_comparison.json"
FLAT_PCT = 0.30


def _truth_direction(name: str, tape: dict[str, Any]) -> str:
    readings = [tape[k].get(f"{kind}:{sym}") for k in ("tape_before", "tape_after")
                for kind, sym in (("perp", f"{name}USDT"), ("spot", f"R{name}USDT"))]
    values = [100 * float(v) for v in readings if v is not None]
    if len(values) < 4 or any(abs(v) < FLAT_PCT for v in values):
        return "flat"
    if all(v > 0 for v in values):
        return "up"
    if all(v < 0 for v in values):
        return "down"
    return "flat"


def _claim_kind(claim: str) -> tuple[str, str | None]:
    """(kind, asserted) — kind is direction, causal or session; asserted is up/down/open/closed."""
    text = claim.lower()
    if "market" in text and ("open" in text or "closed" in text):
        return "session", "closed" if "closed" in text else "open"
    up = bool(re.search(r"\b(?:up|rallied)\b", text))
    kind = "causal" if "because" in text else "direction"
    return kind, "up" if up else "down"


def _argus_verdict(lines: list[str], kind: str) -> tuple[str | None, bool]:
    """(verdict on the asserted fact, whether a stated cause was left unconfirmed)."""
    premise = [line for line in lines if line.startswith("Your premise")]
    if kind == "session":
        premise = [line for line in premise if "US market" in line or "after hours" in line]
    elif premise:
        premise = [line for line in premise if "US market" not in line] or premise
    if not premise:
        return None, False
    line = premise[0]
    verdict = ("challenged" if ": does not hold" in line else
               "supported" if (": holds" in line or ": barely" in line) else None)
    return verdict, "not something a price can confirm" in line


def _mirror_verdict(result: dict[str, Any], kind: str) -> tuple[str | None, bool]:
    wanted = "session.us" if kind == "session" else "price.direction"
    statuses = {a.get("kind"): a.get("status") for a in result.get("assessments", [])}
    verdict = statuses.get(wanted)
    cause_held_back = statuses.get("causation") in ("unsupported", "unassessed", "challenged")
    return (verdict if verdict in ("supported", "challenged") else None), cause_held_back


@contextlib.contextmanager
def _replayed(tape_reading: Mapping[str, float], instant: datetime) -> Iterator[None]:
    """The console's live inputs, pinned to what was recorded: `market.bitget.fetch_tickers`
    returns one :class:`~argus.market.bitget.Ticker` per perpetual in the tape, carrying its
    recorded 24-hour change, and the clock `lui/research` reads (``research.datetime``) stands at
    ``instant``. The tape recorded changes only, so every price field is zero; no claim in this
    set reads a price (none names funding, a premium or the book), and a claim that did would be
    judged on a zero rather than on a live quote — the set is checked for that in the tests.
    Both are restored however the block exits."""
    from argus.lui import research
    from argus.market import bitget

    zero = Decimal(0)
    tickers = {key.split(":", 1)[1]: bitget.Ticker(
        symbol=key.split(":", 1)[1], last=zero, bid=zero, ask=zero, high_24h=zero,
        low_24h=zero, change_24h=Decimal(repr(change)), base_volume=zero, funding_rate=zero,
        fetched_at=instant) for key, change in tape_reading.items() if key.startswith("perp:")}

    class Frozen(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Self:
            return cls.fromtimestamp(instant.timestamp(), tz)

    def fetch_tickers(*_a: Any, **_k: Any) -> dict[str, bitget.Ticker]:
        return dict(tickers)

    # ``datetime`` is a module global of `lui/research` (``from datetime import datetime``), so
    # the frozen clock is set in the module's namespace for the block and put back after it.
    namespace = vars(research)
    real_tickers, real_clock = bitget.fetch_tickers, namespace["datetime"]
    bitget.fetch_tickers = fetch_tickers
    namespace["datetime"] = Frozen
    try:
        yield
    finally:
        bitget.fetch_tickers = real_tickers
        namespace["datetime"] = real_clock


def console_answers(claims: Mapping[str, list[str]], tape: Mapping[str, Any], *,
                    reading: str = "tape_before") -> dict[str, list[dict[str, Any]]]:
    """``research._claim_check`` on every saved sentence, in ``argus_raw.json``'s shape.

    ``reading`` picks the tape the console sees: ``tape_before`` (captured just before both desks
    ran; the default) or ``tape_after``. An exception from the console propagates."""
    from argus.lui import research

    started = datetime.fromisoformat(tape["started"])
    out: dict[str, list[dict[str, Any]]] = {}
    with _replayed(tape[reading], started):
        for name, sentences in claims.items():
            answers = []
            for sentence in sentences:
                symbols, _ = research.research_symbols(sentence)
                symbol = symbols[0] if symbols else ""
                checked = research._claim_check(sentence, symbol)
                answers.append({"claim": sentence, "symbol": symbol,
                                "lines": list(checked[0]) if checked else []})
            out[name] = answers
    return out


def _agreement(claims: Mapping[str, list[str]], a: Mapping[str, Any],
               b: Mapping[str, Any]) -> dict[str, Any]:
    """Claims on which two ARGUS answer sets give a different verdict or cause flag."""
    differ = []
    total = 0
    for name, sentences in claims.items():
        for i, sentence in enumerate(sentences):
            kind, _ = _claim_kind(sentence)
            total += 1
            left = _argus_verdict(a[name][i]["lines"], kind)
            right = _argus_verdict(b[name][i]["lines"], kind)
            if left != right:
                differ.append({"claim": sentence, "replay": list(left), "other": list(right)})
    return {"claims": total, "same_verdict": total - len(differ), "differ": differ}


def score() -> dict[str, Any]:
    """Replay the console's premise check on ``data/h2h_mirrorline/``, grade it beside
    MirrorLine's recorded run, and write the artefact."""
    claims: dict[str, list[str]] = json.loads((DATA / "claims.json").read_text("utf-8"))
    tape = json.loads((DATA / "tape.json").read_text("utf-8"))
    recorded = json.loads((DATA / "argus_raw.json").read_text("utf-8"))
    mirror = json.loads((DATA / "mirrorline_raw.json").read_text("utf-8"))
    argus = console_answers(claims, tape)
    after = console_answers(claims, tape, reading="tape_after")
    report = grade(claims, tape, argus, mirror, us_open=session_open_at_start(tape))
    report["replay"] = {
        "method": "argus.lui.research._claim_check called on each saved sentence with the "
                  "console's resolver's contract, tape_before's 24h changes and the clock at "
                  "run_started (console_answers); written to data/h2h_mirrorline/"
                  "argus_replay.json",
        "scored": "the replay (tape_before)",
        "against_the_recorded_console_run": _agreement(claims, argus, recorded),
        "against_a_replay_on_tape_after": _agreement(claims, argus, after),
    }
    artefact.write(DATA / "argus_replay.json", {"reading": "tape_before",
                                                "at": tape["started"], "answers": argus})
    artefact.write(REPORT, report)
    return report


def session_open_at_start(tape: dict[str, Any]) -> bool:
    """Whether the NYSE had price discovery at the moment the run started (the session truth)."""
    from argus.eval.baselines.lean_market_holidays_loader import load_usa_equity_holidays
    from argus.truth.clocks import DualClock

    clock = DualClock(holidays=load_usa_equity_holidays())
    return bool(clock.phase(datetime.fromisoformat(tape["started"])).has_price_discovery)


def grade(claims: dict[str, list[str]], tape: dict[str, Any], argus: dict[str, Any],
          mirror: dict[str, Any], *, us_open: bool) -> dict[str, Any]:
    """The scorer, separated from the file reads in 2026-09-25's spine migration so the
    harness-validity canary (`eval/harness_validity.py`) can hand it an oracle desk and check the
    oracle scores full marks. Output is key-for-key what :func:`score` wrote before."""
    rows: list[dict[str, Any]] = []
    for name, sentences in claims.items():
        truth_dir = _truth_direction(name, tape)
        for i, sentence in enumerate(sentences):
            kind, asserted = _claim_kind(sentence)
            if kind == "session":
                true_fact = (asserted == "open") == us_open
                graded = True
            else:
                graded = truth_dir != "flat"
                true_fact = truth_dir == asserted
            want = "supported" if true_fact else "challenged"
            a_verdict, a_cause = _argus_verdict(argus[name][i]["lines"], kind)
            m_verdict, m_cause = _mirror_verdict(mirror[name][i].get("result", {}), kind)
            a_ok = a_verdict == want and (kind != "causal" or a_cause)
            m_ok = m_verdict == want and (kind != "causal" or m_cause)
            rows.append({"claim": sentence, "kind": kind, "graded": graded,
                         "truth": want if graded else "flat", "argus": a_verdict,
                         "argus_cause_held_back": a_cause, "mirrorline": m_verdict,
                         "mirrorline_cause_held_back": m_cause,
                         "argus_correct": bool(a_ok) if graded else None,
                         "mirrorline_correct": bool(m_ok) if graded else None})
    graded_rows = [r for r in rows if r["graded"]]
    both = sum(1 for r in graded_rows if r["argus_correct"] and r["mirrorline_correct"])
    a_only = sum(1 for r in graded_rows if r["argus_correct"] and not r["mirrorline_correct"])
    m_only = sum(1 for r in graded_rows if r["mirrorline_correct"] and not r["argus_correct"])
    by_kind: dict[str, dict[str, int]] = {}
    for r in graded_rows:
        tally = by_kind.setdefault(r["kind"], {"n": 0, "argus": 0, "mirrorline": 0})
        tally["n"] += 1
        tally["argus"] += bool(r["argus_correct"])
        tally["mirrorline"] += bool(r["mirrorline_correct"])
    flat = [r for r in rows if not r["graded"]]
    report = {
        "comparison": "claims about the tape: ARGUS premise check vs MirrorLine challenge engine",
        "baseline": "PinnacleCryptNG/MirrorLine getInterpretationChallenge, run from its clone",
        "run_started": tape["started"],
        "us_session_open_at_start": us_open,
        "graded": len(graded_rows),
        "argus_correct": sum(bool(r["argus_correct"]) for r in graded_rows),
        "mirrorline_correct": sum(bool(r["mirrorline_correct"]) for r in graded_rows),
        "both_correct": both, "argus_only": a_only, "mirrorline_only": m_only,
        "by_kind": by_kind,
        "flat_not_graded": [{"claim": r["claim"], "argus": r["argus"],
                             "mirrorline": r["mirrorline"]} for r in flat],
        "first_run_loss": "argus_raw_first_run.json: ARGUS returned a verdict on 18 of 38 "
                          "claims (the direction claims) and none on the 18 past-tense causal "
                          "claims or the 2 session claims; MirrorLine returned one on all 38. "
                          "Fixed the same morning, then both desks run again.",
        "failure_cases": [r for r in graded_rows if not (r["argus_correct"]
                                                        and r["mirrorline_correct"])],
        "not_covered": "claim types only one desk reads — funding and perp premium (ARGUS), "
                       "40-level depth, liquidity and trade-action advice (MirrorLine) — are not "
                       "in this set; spot rToken and perpetual are different instruments, which "
                       "is why near-zero days are not graded",
        "rows": rows,
    }
    report["comparison_reports"] = [r.to_dict() for r in comparison_reports(report)]
    return report


def comparison_reports(report: dict[str, Any]) -> list[ComparisonReport]:
    """This harness's verdict in the spine's shape, read from its own report dict.

    Before the spine the verdict was two counts a reader subtracted; ``standing.py`` wrote "TIED"
    by hand. Here it is an exact sign test on the discordant claims — claims one desk judged right
    and the other wrong — which is the only information about *which desk is better* a paired
    right/wrong table holds. Claim kinds are groups, each with its own sign test, and
    ``every_group`` is set only when every kind produced a result.
    """
    graded = [r for r in report["rows"] if r["graded"]]

    def discordant(rows: list[dict[str, Any]]) -> tuple[int, int]:
        a_only = sum(1 for r in rows if r["argus_correct"] and not r["mirrorline_correct"])
        b_only = sum(1 for r in rows if r["mirrorline_correct"] and not r["argus_correct"])
        return a_only, b_only

    groups = {kind: sign_test_outcome(*discordant([r for r in graded if r["kind"] == kind])).value
              for kind in sorted({r["kind"] for r in graded})}
    a_only, b_only = discordant(graded)
    n = report["graded"]
    return [finalise(ComparisonReport(
        comparison="claimcheck", question="a trader's claims about the tape: which hold",
        argus="ARGUS premise check (lui/research._claim_check)",
        rival="MirrorLine getInterpretationChallenge", metric="share of gradable claims judged "
        "correctly (a causal claim also needs its stated cause held back)",
        lower_is_better=False, argus_score=report["argus_correct"] / n if n else None,
        rival_score=report["mirrorline_correct"] / n if n else None, n=n, unit="claim",
        outcome=sign_test_outcome(a_only, b_only),
        basis=f"exact two-sided sign test on discordant claims (ARGUS only {a_only}, MirrorLine "
              f"only {b_only}); LEVEL when there are none; ARGUS is _claim_check replayed on the "
              f"saved sentences and tape",
        p_value=sign_test(a_only, b_only), scored=n,
        total=n + len(report["flat_not_graded"]), groups=groups,
        artefact="data/claimcheck_comparison.json", created_at=report["run_started"]))]


def main() -> int:  # pragma: no cover - CLI
    report = score()
    print(f"graded {report['graded']}: ARGUS {report['argus_correct']}, "
          f"MirrorLine {report['mirrorline_correct']} (both {report['both_correct']}, "
          f"ARGUS only {report['argus_only']}, MirrorLine only {report['mirrorline_only']})")
    print(json.dumps(report["by_kind"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
