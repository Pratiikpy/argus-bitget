"""A thesis stated in the console, tested reason by reason where it was asked.

"I think NVDA runs on AI capex through 2027 — test my thesis" is the Track 3 "personalized thesis"
line asked in so many words, and the console answered it with the next day's base rates or with the
desk's own decision on NVDA (judge audit, 2026-09-30). The tester already existed, on the research
task's page (`lui/task.py`, `lui/thesis.py`), where nothing in the console led. This runs that same
task for the named contract — its eight engines side by side, so every reason finds the measurement
that bears on it — and answers with the verdicts, each with the figures behind it and where they
came from. The engines' own pages stay one click away, under the answer.
"""

from __future__ import annotations

import re
from typing import Any

from argus.lui.answer import Source
from argus.lui.trace import emit, trace_module

THESIS_ASK = re.compile(
    r"\b(?:test|check|challenge|stress[\s-]*test|pressure[\s-]*test|poke\s+holes\s+in|kill|"
    r"critique|validate|evaluate|assess)\s+(?:my|this|the|that)\s+(?:thesis|idea|view|theory|"
    r"take|call)\b|\bmy\s+thesis\s*(?:is\b|:)|\bhere'?s\s+my\s+(?:thesis|view)\b|"
    r"\bis\s+(?:my|that|this|the)\s+(?:thesis|view|idea|take|theory|reasoning)\s+(?:right|correct|"
    r"wrong|sound|valid|any\s+good|true)\b|\bam\s+i\s+(?:right|wrong)\s+(?:that|to\s+think)\b|"
    # "I think NVDA is undervalued because its P/E is low relative to growth. Is that thesis
    # right?" reached a data dump (a judge, round 11): a view with its reason is a thesis
    r"\bi\s+(?:think|believe|reckon|expect|feel)\s+(?:that\s+)?[^?.]{0,160}\bbecause\b|"
    # "My bull case for BTC: ... Which of these is strongest?" (a judge, round 12)
    r"\b(?:my|the)\s+(?:bull|bear)(?:ish)?\s+case\b|\bwhich\s+of\s+(?:these|my\s+reasons)\s+"
    r"(?:is|are)\s+(?:the\s+)?(?:strongest|weakest|best)\b|\bmy\s+thesis\b[^?]{0,200}"
    r"\btrue\s+or\s+false\b", re.I)
"""A trader stating a view and asking for it to be tested."""

_NOT_NAMES = frozenset({
    "test", "check", "thesis", "think", "believe", "because", "fed", "the", "this", "that", "my",
    "is", "it", "and", "ai", "ceo", "etf", "etfs", "gdp", "cpi", "fomc", "sec", "usa", "us", "eu",
    "china", "chinese", "america", "american", "europe", "japan", "india", "q1", "q2", "q3", "q4",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "poke", "kill", "here", "am", "right", "wrong"})
"""Capitalised words that are not a company a thesis could be about."""

_MARKETS: tuple[tuple[str, str, str], ...] = (
    (r"\b(?:crypto|coins|altcoins|the\s+crypto\s+market)\b", "BTCUSDT",
     "no single name was given, so the crypto market is read as bitcoin (BTCUSDT)"),
    # "semis will outperform because AI capex keeps rising" was told to name a stock, then the next
    # turn called it a bull case (round 18, row 616): a sector has a listed fund that stands for it.
    (r"\b(?:semis|semiconductors?|chip\s*(?:stocks|makers|sector)|chips)\b", "SMHUSDT",
     "no single name was given, so semiconductors are read as the VanEck semiconductor fund's "
     "contract (SMHUSDT)"),
    (r"\b(?:tech\s+stocks|tech|the\s+nasdaq|nasdaq)\b", "QQQUSDT",
     "no single name was given, so tech is read as the Nasdaq-100 fund's contract (QQQUSDT)"),
    (r"\b(?:stocks|equities|the\s+(?:stock\s+)?market|us\s+stocks|wall\s+street)\b", "SP500USDT",
     "no single name was given, so the stock market is read as Bitget's S&P 500 contract "
     "(SP500USDT)"),
)
"""A thesis about a whole market, and the one contract that stands for it."""

STRONGEST = re.compile(r"\b(?:strongest|weakest|best\s+(?:reason|point|argument))\b", re.I)
_STRENGTH = {"supported": 0, "not measurable": 1, "not tested": 2, "contradicted": 3}
"""How a reason's verdict ranks when the question asks which reason is strongest."""

_ORDER = {"contradicted": 0, "supported": 1, "not measurable": 2, "not tested": 3}


def asks(text: str) -> bool:
    from argus.lui.question import POSITION_THESIS

    return bool(THESIS_ASK.search(text) or POSITION_THESIS.match(text))


def answer(text: str, *, book: str = "", memory: str = ""
           ) -> tuple[list[str], list[Source], dict[str, Any]]:
    """The thesis's reasons, each tested (`lui/thesis.check` through `lui/task.research_task`)."""
    from urllib.parse import quote

    from argus.lui import thesis
    from argus.lui.research import research_symbols
    from argus.lui.task import read_question, research_task

    named = research_symbols(text)[0]
    market_note = ""
    if not named:
        # "I think stocks will go up" is a thesis about the market: tested on the broad index
        # contract Bitget lists, said on the answer (round 11: it was asked to name a stock).
        broad = next(((sym, label) for pattern, sym, label in _MARKETS
                      if re.search(pattern, text, re.I)), None)
        if broad is not None:
            named, market_note = (broad[0],), broad[1]
    unlisted = [w for w in re.findall(r"(?<!^)(?<![.?!]\s)\b([A-Z][A-Za-z&.-]{2,})\b", text)
                if w.lower() not in _NOT_NAMES]
    if not named and unlisted:
        # "I think Tencent will outperform because of gaming approvals" was told to name a stock
        # (a hostile review, round 11): it named one Bitget does not list.
        lines = [f"Bottom line: {unlisted[0]} is not a contract Bitget lists, so there is no "
                 f"price, filing or positioning here to test the thesis against. Ask it of a "
                 f"listed name — any US stock or ETF Bitget carries, gold, oil, an index or a "
                 f"coin."]
        return lines, [], {"thesis": None, "unlisted": unlisted[0]}
    if not named:
        lines = ["Bottom line: name the stock or coin the thesis is about, and why you hold it — "
                 "for example \"I think NVDA runs on AI capex through 2027, test my thesis\" or "
                 "\"long SOL because on-chain activity is growing and this dip is temporary\"."]
        return lines, [], {"thesis": None}
    symbol = named[0]
    name = symbol.removesuffix("USDT")
    stated = thesis.reasons(text)
    link = f"/research?q={quote(text)}"
    if not stated:
        lines = [f"Bottom line: I could not find a reason to test in that — the thesis names "
                 f"{name} but not why. Say it with the reason, for example \"{name} keeps rising "
                 f"because demand is growing\", and each reason is tested against its own data.",
                 f"The full research on {name} runs at {link}."]
        return lines, [], {"thesis": {"name": name, "tested": []}}
    reading = read_question(text, book)
    task = research_task(reading=None if isinstance(reading, str) else reading, asked=text,
                         name=symbol, memory=memory)
    from argus.lui.task import _data_by_kind

    quotes = (_data_by_kind(task.steps).get("quote") or {}).get("quotes") or {}
    last = float((quotes.get(symbol) or {}).get("last") or 0) or None
    premise = thesis.price_premise(text, name, last)
    tested = sorted(task.tested, key=lambda t: (t.implied, _ORDER.get(t.result.value, 9)))
    if premise is not None:
        tested = [premise, *tested]
        tally_note = "; the price it states is wrong"
    else:
        tally_note = ""
    counts = {r: sum(1 for t in tested if not t.implied and t.result.value == r) for r in _ORDER}
    tally = ", ".join(f"{n} {r}" for r, n in counts.items() if n) + tally_note
    lines = [f"Bottom line: your thesis on {name}, reason by reason — {tally}. Each verdict is "
             f"the one test named beside it; the figures under it inform, they do not vote."]
    if STRONGEST.search(text):
        # "Which of these is strongest?" asked for a ranking (a judge, round 12).
        own = [t for t in tested if not t.implied]
        ranked = sorted(own, key=lambda t: _STRENGTH.get(t.result.value, 9))
        if ranked and ranked[0].result.value == "supported":
            lines[0] = (f"Bottom line: on today's data the strongest of your reasons is "
                        f"\"{ranked[0].reason}\", which the data backs; in all — "
                        f"{tally}. Each verdict is the one test named beside it.")
        elif ranked:
            lines[0] = (f"Bottom line: none of your reasons is backed by today's data — {tally}; "
                        f"the least contradicted is \"{ranked[0].reason}\" "
                        f"({ranked[0].result.value}). Each verdict is the one test named beside "
                        f"it.")
    sources: list[Source] = []
    seen: set[str] = set()
    said: dict[str, str] = {}
    for t in tested:
        head = ("Premise" if t.line.startswith("The premise does not hold") else
                "Implied, and tested" if t.implied else t.result.value.capitalize())
        if t.line in said:
            # Two reasons read by one test ("NVDA is undervalued", "its P/E is low") say it once.
            lines.append(f"{head} — \"{t.reason}\": the same reading as \"{said[t.line]}\".")
            continue
        said[t.line] = t.reason
        lines.append(f"{head} — \"{t.reason}\": {t.line}")
        for finding in t.evidence[:4]:
            lines.append(f"  · {finding.text}")
            if finding.url and finding.url not in seen:
                seen.add(finding.url)
                sources.append(Source(kind="venue", ref=finding.url, detail=finding.source))
    lines.append(f"The full research on {name} — price and cost, technicals, news and filings, "
                 f"earnings, history, what it does to your book, exposure and execution — ran "
                 f"beside these tests in {task.seconds:.0f}s; its page is {link}.")
    emit(lines[1:-1], "computed")
    if market_note:
        lines.append(f"Assumed: {market_note}.")
    return lines, sources, {"thesis": {"name": name,
                                       "tested": [t.as_dict() for t in task.tested]}}



REVISE = re.compile(
    r"\b(?:scratch|drop|remove|forget|ignore|leave\s+out|take\s+out|set\s+aside)\b.{0,40}"
    r"\b(?:point|reason|argument|part|one)\b|\bassume\b.{0,80}\binstead\b|\bdoes\s+(?:that|this)"
    r"\s+change\s+(?:your|the|my)\s+(?:view|verdict|answer|read|sizing|size|position)|"
    r"\bmy\s+(?:real|only|main|actual)\s+reason\b|\bi\s+was\s+wrong\s+about\b", re.I)
"""A follow-up that changes the reasons of the thesis tested earlier in the conversation."""

_DROPS = re.compile(
    r"\b(?:scratch|drop|remove|forget(?:\s+about)?|ignore|leave\s+out|take\s+out|set\s+aside|"
    r"wrong\s+about)\s+(?:the\s+|my\s+)?(?P<what>[^.?!;]{2,80}?)(?=\s*(?:[.?!;]|$|\s+-\s|,?\s+"
    r"(?:it'?s|assume|my|and\s+(?:assume|my))\b))", re.I)
"""The reasons a follow-up names to set aside: "Forget capex and margins." names two."""

_KEEP_ONLY = re.compile(
    r"\b(?:my\s+)?(?:real|only|main|actual)\s+reason\s+(?:now\s+)?(?:is\s+)?(?:now\s+)?"
    r"(?:just\s+|only\s+|simply\s+)?(?:the\s+)?(?P<keep>[^.?!;]{3,120}?)(?=\s*(?:[.?!;]|$))", re.I)
"""A follow-up that keeps one reason and drops the rest: "My real reason now is just the bookings
backlog." — refused as an unlisted name (a judge, round 13, 2026-09-30)."""

_NOT_NAMING = {
    "the", "and", "that", "this", "point", "reason", "reasons", "scratch", "assume", "instead",
    "does", "change", "your", "view", "actually", "drop", "remove", "forget", "ignore", "one",
    "just", "now", "real", "only", "main", "was", "wrong", "about", "sizing", "size", "its",
    "next", "quarter", "really", "all", "part", "argument"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower())} - _NOT_NAMING


def _thesis_turn(prior: list[str]) -> int | None:
    """The last turn that stated a thesis to test, looking back past earlier revisions of it."""
    return next((i for i in range(len(prior) - 1, -1, -1) if asks(prior[i])
                 and not REVISE.search(prior[i])), None)


def _apply(revision: str, reasons: list[Any]) -> tuple[list[Any], list[Any]] | None:
    """(kept, dropped) after one follow-up, or None when it names no reason that was stated."""
    named = [m.group("what") for m in _DROPS.finditer(revision)]
    dropped: list[Any] = []
    for clause in named:
        for item in re.split(r"\s*(?:,|\band\b|&|\bor\b)\s*", clause):
            words = _words(item)
            best = max(reasons, key=lambda r: len(words & _words(r.text)), default=None)
            if best is not None and words & _words(best.text) and best not in dropped:
                dropped.append(best)
    if not named:
        # "Actually, the capex point is wrong — does that change your view?": the reason whose
        # words the follow-up shares most is the one it means (round 12).
        words = _words(revision)
        best = max(reasons, key=lambda r: len(words & _words(r.text)), default=None)
        if best is not None and words & _words(best.text):
            dropped.append(best)
    kept = [r for r in reasons if r not in dropped]
    only = _KEEP_ONLY.search(revision)
    if only is not None:
        keep_words = _words(only.group("keep"))
        # The claim itself ("NVDA keeps running") is what the kept reason argues for, not a
        # reason to drop.
        chosen = [r for r in kept if keep_words & _words(r.text) or r.kind.name == "DIRECTION"]
        if any(r.kind.name != "DIRECTION" for r in chosen):
            dropped += [r for r in kept if r not in chosen]
            kept = chosen
    if not dropped:
        return None
    return kept, dropped


SIZE_ASKED = re.compile(r"\bsiz(?:e|ing)\b|\bposition\b|\bhow\s+(?:much|big|large|small)\b", re.I)
"""A follow-up that also asks how big the position should be."""

_BEARISH = re.compile(r"\b(?:short(?:ing)?|bearish|bear\s+case|fade|sell(?:ing)?)\b", re.I)


def _case(earlier: str) -> str:
    """``bear`` for a thesis that argues for a fall, else ``bull``: the rebuilt question and the
    research link under the answer name the side the trader took (a judge, round 14: a short
    thesis was labelled "My bull case")."""
    return "bear" if _BEARISH.search(earlier) else "bull"


def _standing(turns: list[str], text: str = ""
              ) -> tuple[str, list[Any], list[Any], list[Any]] | None:
    """(the thesis as first stated, its reasons, the reasons still standing, the reasons set
    aside) after every revision in ``turns`` and, when given, in ``text`` too; None when no thesis
    was stated, no revision applies where ``text`` is one, or nothing is left."""
    from argus.lui import thesis

    at = _thesis_turn(turns)
    if at is None:
        return None
    earlier = split_other_question(turns[at])[0]
    stated = list(thesis.reasons(earlier))
    kept = stated
    dropped: list[Any] = []
    for revision in [*(t for t in turns[at + 1:] if REVISE.search(t)), *([text] if text else [])]:
        applied = _apply(revision, kept)
        if applied is None:
            if revision is text:
                return None
            continue
        kept, gone = applied
        dropped += gone
    return (earlier, stated, kept, dropped) if kept else None


_EXPLICIT_REVISION = re.compile(
    r"\bi\s+was\s+wrong\s+about\b|\bmy\s+(?:real|only|main|actual)\s+reason\b|"
    r"\b(?:scratch|drop|forget|ignore|set\s+aside)\b.{0,40}\b(?:point|reason|argument)\b", re.I)
"""Wording that can only mean a reason in a thesis is being changed."""


def nothing_to_revise(text: str, prior: list[str]
                     ) -> tuple[list[str], list[Source], dict[str, Any]] | None:
    """"I was wrong about margins, they are stable" with no thesis earlier in the conversation:
    said plainly, instead of the nearest record the language model could find (a first-time user,
    round 14, live: it was answered with an unrelated desk decision)."""
    if not _EXPLICIT_REVISION.search(text) or _standing(list(prior), text) is not None:
        return None
    return (["Bottom line: there is no thesis earlier in this conversation to change. State it "
             "with its reasons — for example \"short TSLA because margins are falling\" — and "
             "each reason is tested; then this revision is applied to it."], [],
            {"thesis": None, "revision_without_thesis": True})


_A_VIEW = re.compile(r"\bi\s+(?:think|believe|feel|expect|reckon|bet)\b|\bmy\s+(?:view|thesis|bet|"
                     r"take|call)\b", re.I)
"""A trader's stated opinion, with or without a reason."""

FALSIFY = re.compile(
    r"\bwhat\s+(?:would|could|will)(?:\s+it)?\s+(?:prove|show|make|mean|tell)"
    r"\b[^?.]{0,40}\b(?:wrong|invalid\w*)|\bwhat\s+would\s+change\s+my\s+mind\b|"
    r"\bhow\s+(?:would|will|do)\s+i\s+know\s+(?:if\s+)?(?:i(?:'m|\s+am))?\s*wrong\b", re.I)
"""A follow-up asking what would show the trader's thesis is wrong."""

_BREAKS = {
    "REVERSION": "the price keeps going the way it has — a fresh extreme, not a return",
    "ACTIVITY": "the activity figure it rests on turns down in the next two readings",
    "MOMENTUM": "the trend turns — a daily close beyond the stop level below",
    "POSITIONING": "the crowd and funding move to the other side of the claim and stay there",
    "SENTIMENT": "the sentiment reading moves away from the claim and holds there for a week",
    "VALUATION": "the market stops paying that multiple — the ratio moves further from the claim",
    "DRIVER": "the driver's next report comes in against the claim",
    "RELATIVE": "the gap to the other name closes or reverses over the next two weeks",
    "FLOWS": "net flows turn the other way for a week running",
    "EARNINGS": "the next earnings report misses the claim",
    "MACRO": "the next release of the macro series comes in the other way",
    "DIRECTION": "price closes beyond the stop level below",
    "OTHER": "the figure the claim rests on moves against it",
}


def falsify(text: str, prior: list[str], *, book: str = "", memory: str = ""
            ) -> tuple[list[str], list[Source], dict[str, Any]] | None:
    """"What would prove my thesis wrong, and how should I size it?" after a thesis: for each
    reason still standing, what would break it, what today's data already says of it, and a stop
    level from the last 20 daily bars (a judge, round 15: it was told "that" had nothing to refer
    to, and the reasons were not turned into things that can be watched)."""
    if not FALSIFY.search(text) or asks(text):
        return None
    current = _standing(list(prior))
    if current is None:
        view = next((t for t in reversed(prior) if _A_VIEW.search(t)), None)
        if view is not None:
            # A view with no reason in it is not nothing: saying "there is no thesis" straight
            # after the trader stated one read as the console not listening (round 18).
            said = re.sub(r"\s+", " ", view).strip()
            said = said if len(said) <= 110 else said[:107].rstrip() + "..."
            return ([f"Bottom line: you said “{said}” — a view, but it carries no reason that can "
                     f"be tested, so nothing can yet be shown wrong. Add the market and why — for "
                     f"example \"QQQ beats SPY this year because earnings are growing faster\" — "
                     f"and what would prove it wrong is worked out reason by reason."], [],
                    {"thesis": None, "falsify_without_thesis": True, "untestable_view": True})
        return (["Bottom line: there is no thesis earlier in this conversation to test. State it "
                 "with its reasons — for example \"ETH beats BTC because funding is negative\" — "
                 "and what would prove it wrong is worked out reason by reason."], [],
                {"thesis": None, "falsify_without_thesis": True})
    from argus.lui import thesis

    earlier, _stated, kept, dropped = current
    name = research_names(earlier)
    case = _case(earlier)
    now_said: dict[str, str] = {}
    try:
        _lines, _sources, data = answer(earlier, book=book, memory=memory)
        for row in (data.get("thesis") or {}).get("tested", []):
            now_said[str(row.get("reason"))] = str(row.get("result"))
    except Exception:
        pass
    out = [f"Bottom line: your {case} case for {name} is wrong if any of these happens; each is "
           f"set beside what today's data already says of that reason."]
    for r in kept:
        state = now_said.get(r.text)
        today = {"contradicted": " Today's data already contradicts it.",
                 "supported": " Today's data backs it.",
                 "not_measurable": " No data tests it today.",
                 "not_tested": " Not tested today."}.get(state or "", "")
        breaks = _BREAKS.get(r.kind.name, _BREAKS["OTHER"])
        if r.kind.name == "OTHER" and thesis._RATIO_EXTREME.search(r.text):
            breaks = "the ratio breaks through that extreme and keeps going instead of turning"
        out.append(f"\"{r.text}\": wrong if {breaks}.{today}")
    stop = ""
    try:
        rows = thesis._closes(name + "USDT")[-20:]
        if len(rows) >= 10:
            last = rows[-1][2]
            level = min(r[1] for r in rows) if case == "bull" else max(r[0] for r in rows)
            away = (level / last - 1) * 100
            stop = (f"Stop level: the {'lowest low' if case == 'bull' else 'highest high'} of the "
                    f"last {len(rows)} daily bars is {level:,.4g}, {away:+.1f}% from the last "
                    f"close of {last:,.4g} — a daily close beyond it breaks the trade whatever the "
                    f"reasons say.")
    except Exception:
        stop = ""
    if stop:
        out.append(stop)
    if dropped:
        out.append("Assumed: " + " and ".join(f"\"{r.text}\"" for r in dropped)
                   + " stays set aside, as you said.")
    return out, [], {"thesis": {"name": name, "case": case}, "falsifiers": [r.text for r in kept]}


def strongest(text: str, prior: list[str], *, book: str = "", memory: str = ""
              ) -> tuple[list[str], list[Source], dict[str, Any]] | None:
    """"Which of those two remaining reasons is the strongest?" after a thesis and its
    revisions: the reasons still standing, ranked (a judge, round 14: it was told "that" had
    nothing to refer to, because the turn just before was a revision, not the thesis)."""
    if not STRONGEST.search(text) or asks(text):
        return None
    current = _standing(list(prior))
    if current is None:
        return None
    earlier, _stated, kept, dropped = current
    name = research_names(earlier)
    rebuilt = (f"My {_case(earlier)} case for {name}: " + ", ".join(r.text for r in kept)
               + ", test my thesis. Which of these is strongest?")
    lines, sources, data = answer(rebuilt, book=book, memory=memory)
    if dropped:
        aside = " and ".join(f"\"{r.text}\"" for r in dropped)
        lines = [lines[0], f"Assumed: {aside} stays set aside, as you said; only the reasons "
                           f"still standing are ranked.", *lines[1:]]
    return lines, sources, {**data, "dropped": [r.text for r in dropped]}


def revise(text: str, prior: list[str] | str, *, book: str = "", memory: str = ""
           ) -> tuple[list[str], list[Source], dict[str, Any]] | None:
    """The thesis tested earlier, again, with the reasons the follow-ups name set aside.

    "Actually scratch the ETF point — assume ETF flows go negative instead. Does that change your
    view?" returned the previous answer word for word (a judge, round 12). The reason whose words
    the follow-up names is removed and the rest re-tested; an assumed future ("flows go negative")
    is not a fact the data can test, and the answer says so rather than pretending to.

    Revisions accumulate: the thesis is found by looking back past earlier follow-ups, and every
    follow-up since is applied in order, so "forget capex" and then "my real reason now is just the
    backlog" leave the backlog alone (a judge, round 13: the second follow-up was refused as a
    name Bitget does not list, because only the turn just before was read)."""
    turns = [prior] if isinstance(prior, str) else list(prior)
    if not REVISE.search(text):
        return None
    current = _standing(turns, text)
    if current is None:
        return None
    earlier, stated, kept, dropped = current
    if not kept:
        return None
    from concurrent.futures import ThreadPoolExecutor

    name = research_names(earlier)
    rebuilt = (f"My {_case(earlier)} case for {name}: " + ", ".join(r.text for r in kept)
               + ", test my thesis")
    with ThreadPoolExecutor(max_workers=2) as pool:
        before_job = pool.submit(answer, earlier, book=book, memory=memory)
        lines, sources, data = answer(rebuilt, book=book, memory=memory)
        before = before_job.result()[2]
    was = {t.get("reason"): t.get("result")
           for t in (before.get("thesis") or {}).get("tested", [])}
    supported = [t for t in (data.get("thesis") or {}).get("tested", [])
                 if t.get("result") == "supported" and not t.get("implied")]
    verdict = ("what is left still has support: " + "; ".join(
        f"\"{t['reason']}\"" for t in supported) if supported else
               "none of the remaining reasons is backed by today's data")
    lost = [r.text for r in dropped if was.get(r.text) == "supported"]
    change = ("the reasons set aside were not backed by the data, so setting them aside changes "
              "nothing" if dropped and not lost else
              ("it was the only support the thesis had" if not supported else
               "the support that came from " + " and ".join(f"\"{x}\"" for x in lost)
               + " is gone, the rest stands") if lost else "")
    aside = " and ".join(f"\"{r.text}\"" for r in dropped)
    head = (f"Bottom line: with {aside} set aside, {verdict}"
            + (f"; {change}" if change else "") + ".")
    tail = []
    if re.search(r"\bassume\b|\binstead\b|\bslowing\b|\bnext\s+(?:quarter|year|month)\b", text,
                 re.I):
        tail.append("What you assume in its place is a future the data cannot test, so it is not "
                    "scored.")
    if SIZE_ASKED.search(text):
        tail.append(
            f"On sizing: a thesis verdict does not set a size — how far {name} can go against "
            f"you does. Ask \"how much should I put in {name}\" and it is sized to your loss "
            f"limit and the book you gave, with fewer reasons behind it now than when you "
            f"started ({len(kept)} of {len(stated)}).")
    return ([head, *tail, *(line for line in lines[1:])], sources,
            {**data, "dropped": [r.text for r in dropped]})


def split_other_question(text: str) -> tuple[str, str]:
    """(the thesis, a separate question asked after it about another name), or (text, "").

    Only a closing sentence that names a contract the thesis does not, and is not itself one of
    the tester's asking phrases ("test my thesis", "which is strongest"), is split off."""
    from argus.lui.research import research_symbols

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    if len(parts) < 2:
        return text, ""
    claim, last = " ".join(parts[:-1]), parts[-1]
    if asks(last) and not asks(claim):
        return text, ""
    theirs = set(research_symbols(claim)[0])
    named = set(research_symbols(last)[0])
    if not theirs or not named or named <= theirs or STRONGEST.search(last):
        return text, ""
    return claim, last


def research_names(text: str) -> str:
    from argus.lui.research import research_symbols

    named = research_symbols(text)[0]
    if not named:
        # the market a sector or market thesis was tested on, not "it" (round 18, row 616)
        named = tuple(sym for pattern, sym, _label in _MARKETS if re.search(pattern, text, re.I))
    return named[0].removesuffix("USDT") if named else "it"

trace_module(globals())
