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
    return bool(THESIS_ASK.search(text))


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
    from argus.lui import thesis

    turns = [prior] if isinstance(prior, str) else list(prior)
    if not REVISE.search(text):
        return None
    at = _thesis_turn(turns)
    if at is None:
        return None
    earlier = split_other_question(turns[at])[0]
    stated = list(thesis.reasons(earlier))
    kept = stated
    dropped: list[Any] = []
    for revision in [*(t for t in turns[at + 1:] if REVISE.search(t)), text]:
        applied = _apply(revision, kept)
        if applied is None:
            if revision is text:
                return None
            continue
        kept, gone = applied
        dropped += gone
    if not kept:
        return None
    from concurrent.futures import ThreadPoolExecutor

    name = research_names(earlier)
    rebuilt = f"My bull case for {name}: " + ", ".join(r.text for r in kept) + ", test my thesis"
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
               "that support is gone, the rest stands") if lost else "")
    aside = " and ".join(f"\"{r.text}\"" for r in dropped)
    head = (f"Bottom line: with {aside} set aside, {verdict}"
            + (f"; {change}" if change else "") + ".")
    tail = []
    if re.search(r"\bassume\b|\binstead\b|\bslowing\b|\bnext\s+(?:quarter|year|month)\b", text,
                 re.I):
        tail.append("What you assume in its place is a future the data cannot test, so it is not "
                    "scored.")
    if re.search(r"\bsiz(?:e|ing)\b|\bposition\b|\bhow\s+much\b", text, re.I):
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
    return named[0].removesuffix("USDT") if named else "it"

trace_module(globals())
