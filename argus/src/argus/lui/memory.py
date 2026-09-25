"""What a trader has told the console, kept across sessions and used in every later answer.

Track 3 judges a *personalised thesis*. Until this module, the only thing the console remembered was
the book a trader typed into one field; "I can't lose more than 10%", "I hold for weeks", "I think
NVDA runs on AI capex" were read once and forgotten, so the next answer was the same for everyone.

**What was taken from mem0 (Apache-2.0; `mem0/memory/main.py:881-1210`, `:143-172`), and what was
not.** Taken: extraction on every turn, facts deduplicated so the same statement is stored once, a
newer fact of the same kind replacing the older one, the date each fact was stated anchoring it
(`prompts.py:524-540`), and identity scoping — a fact can only belong to the trader whose browser
holds it. Not taken: the LLM extraction call (mem0 spends one per turn; the Qwen key here is a
budget, and the facts a trading answer needs have fixed shapes a pattern reads exactly), the vector
store (a trader's memory is tens of facts, not millions), and the server-side store: the memory
lives in the trader's own browser and is sent with each question, so the hosted console keeps
nothing about anyone and a second visitor can never read the first one's facts — mem0's three
cross-tenant issues (#4490, #6277, #6655) cannot happen when there is no shared store.

**What memory changes.** A remembered risk budget, holding period, loss limit, style or capital is
applied where the question did not state its own — and every use is said on a ``Remembered:`` line,
so a trader sees which of their words shaped the answer. A remembered thesis on a name is shown
beside every later answer about that name with the move since it was stated: the expectation gap,
measured, not narrated.

Everything read from the client is untrusted: parsed, bounded, and never executed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

MAX_FACTS = 40
MAX_TEXT = 200

KINDS = ("budget", "max_loss", "horizon", "style", "capital", "thesis", "avoid")


@dataclass(frozen=True, slots=True)
class Fact:
    kind: str
    subject: str
    """A symbol for a thesis or an avoid; empty for a fact about the trader."""
    value: str
    text: str
    """The trader's own words, as stated."""
    at: str
    """The date it was stated, ISO."""
    price_at: float | None = None
    """For a thesis: the instrument's price when it was stated."""

    def key(self) -> tuple[str, str]:
        return self.kind, self.subject


_BUDGET = re.compile(r"\b(?:my\s+)?risk\s+budget\s+(?:is\s+|of\s+)?(\d{1,2}(?:\.\d+)?)\s*%|"
                     r"\bno\s+(?:single\s+)?(?:name|position|holding)\s+(?:above|over|more\s+than)\s+"
                     r"(\d{1,2}(?:\.\d+)?)\s*%\s+of\s+(?:my\s+)?risk", re.I)
_MAX_LOSS = re.compile(
    r"\b(?:i\s+)?(?:can'?t|cannot|can\s+not|don'?t\s+want\s+to|won'?t|never)\s+(?:afford\s+to\s+)?"
    r"lose\s+"
    r"(?:more\s+than\s+)?(\d{1,2}(?:\.\d+)?)\s*%|\bmax(?:imum)?\s+(?:loss|drawdown)\s+(?:is\s+|of\s+)?"
    r"(\d{1,2}(?:\.\d+)?)\s*%|\b(?:my\s+)?loss\s+limit\s+(?:is\s+)?(\d{1,2}(?:\.\d+)?)\s*%", re.I)
_HORIZON = re.compile(
    r"\bi(?:'?m|\s+am)\s+an?\s+(day|swing|position|long[\s-]term)\s+(?:trader|investor)|"
    r"\bi\s+(?:usually\s+|normally\s+|typically\s+)?hold\s+(?:for\s+)?(?:a\s+few\s+|several\s+)?"
    r"(hours?|days?|weeks?|months?|years?)\b", re.I)
_STYLE = re.compile(
    r"\bi(?:'?m|\s+am)\s+(?:a\s+|an\s+|pretty\s+|quite\s+|very\s+)?(conservative|aggressive|"
    r"risk[\s-]averse|cautious)\b|\bi\s+(?:mostly\s+|only\s+|usually\s+)?trade\s+(earnings|momentum|"
    r"mean[\s-]reversion|breakouts?|news|macro|crypto|stocks)\b", re.I)
_CAPITAL = re.compile(
    r"\bmy\s+(?:account|book|portfolio|capital)\s+is\s+(?:about\s+|around\s+)?\$?\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(k|m)?\b|\bi\s+have\s+(?:about\s+|around\s+)?\$\s*(\d[\d,]*(?:\.\d+)?)"
    r"\s*(k|m)?\s+(?:to\s+(?:trade|invest)|in\s+my\s+account)", re.I)
_THESIS = re.compile(
    r"\bi\s+(?:think|believe|expect|reckon|bet)\s+(?:that\s+)?(.{2,40}?)\s+(?:will|is\s+going\s+to|"
    r"gonna|should|can|to)\s+(.{3,120})", re.I)
_ASKING = re.compile(r"^\s*(?:what|how|should|is|are|does|do|can|could|why|when|which|will|would|"
                     r"where|who)\b|\?\s*$", re.I)
"""A question states nothing about the trader: "what if I can't lose more than 10%?" and "should
I trade earnings" wrote false memories on the first extraction run (`eval/memory_eval.py`)."""
_AVOID = re.compile(r"\bi\s+(?:don'?t|never|won'?t)\s+(?:trade|touch|buy|hold|want)\s+(?:any\s+)?"
                    r"(.{2,30}?)(?:[,.;!?]|$)", re.I)
_BULL = re.compile(r"\b(?:up|rise|rally|outperform|beat|higher|moon|rip|double|recover|bounce)\w*",
                   re.I)
_BEAR = re.compile(r"\b(?:down|fall|drop|crash|underperform|miss|lower|dump|tank|decline|lose)\w*",
                   re.I)


def extract(question: str, now: datetime | None = None,
            price_of: Any = None) -> list[Fact]:
    """The facts a question states about the trader. ``price_of(symbol)`` stamps a thesis with
    the price at the time it was stated; it may be None (no price, no gap measured later)."""
    from argus.lui.research import research_symbols

    day = (now or datetime.now(UTC)).date().isoformat()
    text = question.strip()[:600]
    facts: list[Fact] = []
    if _ASKING.search(text) and not re.match(r"\s*i\b", text, re.I):
        return facts

    def add(kind: str, subject: str, value: str, words: str,
            price: float | None = None) -> None:
        facts.append(Fact(kind=kind, subject=subject, value=value, text=words.strip()[:MAX_TEXT],
                          at=day, price_at=price))

    if (m := _BUDGET.search(text)) is not None:
        add("budget", "", str(float(m.group(1) or m.group(2)) / 100), m.group(0))
    if (m := _MAX_LOSS.search(text)) is not None:
        add("max_loss", "", str(float(m.group(1) or m.group(2) or m.group(3)) / 100), m.group(0))
    if (m := _HORIZON.search(text)) is not None:
        word = (m.group(1) or m.group(2) or "").lower()
        hours = {"day": 24, "swing": 24 * 7, "position": 24 * 30}.get(word) or (
            24 * 90 if word.startswith("long") else
            1 if word.startswith("hour") else 24 if word.startswith("day") else
            168 if word.startswith("week") else 720 if word.startswith("month") else 24 * 365)
        add("horizon", "", str(hours), m.group(0))
    if (m := _STYLE.search(text)) is not None:
        add("style", "", (m.group(1) or m.group(2) or "").lower(), m.group(0))
    if (m := _CAPITAL.search(text)) is not None:
        amount = float((m.group(1) or m.group(3) or "0").replace(",", ""))
        unit = (m.group(2) or m.group(4) or "").lower()
        amount *= 1_000 if unit == "k" else 1_000_000 if unit == "m" else 1
        if amount >= 100:
            add("capital", "", f"{amount:.0f}", m.group(0))
    if (m := _THESIS.search(text)) is not None:
        named = research_symbols(m.group(1))[0]
        if named:
            claim = m.group(2)
            lean = ("bull" if _BULL.search(claim) and not _BEAR.search(claim) else
                    "bear" if _BEAR.search(claim) and not _BULL.search(claim) else "view")
            price = None
            if price_of is not None:
                try:
                    price = float(price_of(named[0]))
                except Exception:
                    price = None
            add("thesis", named[0], lean, m.group(0), price)
    if (m := _AVOID.search(text)) is not None:
        named = research_symbols(m.group(1))[0]
        if named:
            add("avoid", named[0], "avoid", m.group(0))
    return facts


def parse(raw: str | None) -> list[Fact]:
    """The client's stored memory, validated. Anything malformed is dropped, never trusted."""
    if not raw:
        return []
    try:
        rows = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(rows, list):
        return []
    out: list[Fact] = []
    for row in rows[:MAX_FACTS]:
        if not isinstance(row, dict) or row.get("kind") not in KINDS:
            continue
        try:
            price = row.get("price_at")
            out.append(Fact(
                kind=str(row["kind"]), subject=str(row.get("subject", ""))[:24],
                value=str(row.get("value", ""))[:40], text=str(row.get("text", ""))[:MAX_TEXT],
                at=str(row.get("at", ""))[:10],
                price_at=float(price) if isinstance(price, (int, float)) else None))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def merge(old: list[Fact], new: list[Fact]) -> list[Fact]:
    """A newer fact of the same kind and subject replaces the older; the rest are kept, newest
    first, up to :data:`MAX_FACTS`."""
    replaced = {f.key() for f in new}
    kept = [*new, *(f for f in old if f.key() not in replaced)]
    return kept[:MAX_FACTS]


def dumps(facts: list[Fact]) -> str:
    return json.dumps([asdict(f) for f in facts], separators=(",", ":"))


def get(facts: list[Fact], kind: str, subject: str = "") -> Fact | None:
    return next((f for f in facts if f.kind == kind and f.subject == subject), None)


def remembered_line(fact: Fact, use: str) -> str:
    """How a remembered fact is shown when it shapes an answer."""
    return f"Remembered: you said “{fact.text}” on {fact.at} — {use}."


def thesis_line(fact: Fact, name: str, price_now: float | None) -> str:
    """A stated thesis beside today's price: the gap between what was expected and what happened
    so far, measured from the price when it was said."""
    base = f"Your thesis on {name} ({fact.at}): “{fact.text}”"
    if fact.price_at and price_now:
        move = price_now / fact.price_at - 1
        agrees = ((fact.value == "bull" and move > 0) or (fact.value == "bear" and move < 0))
        direction = ("flat so far" if abs(move) < 0.001 else
                     ("with it" if agrees else "against it") if fact.value in ("bull", "bear")
                     else "since")
        return (f"{base}. Since then {name} has moved {move:+.1%} ({fact.price_at:,.6g} to "
                f"{price_now:,.6g}) — {direction}"
                + ("." if fact.value not in ("bull", "bear") else
                   "; a thesis is judged at its horizon, not on the first move."))
    return base + "."


def apply(request: Any, facts: list[Fact], question: str) -> tuple[Any, list[str]]:
    """The request with what the trader told the console filled in where the question itself
    was silent, and one ``Remembered:`` line for each fact that shaped it."""
    from dataclasses import replace

    from argus.lui.research import ResearchKind

    used: list[str] = []
    budget = get(facts, "budget")
    if budget is not None and not request.budget_stated:
        request = replace(request, budget=float(budget.value), budget_stated=True)
        used.append(remembered_line(budget, f"your {float(budget.value):.0%} risk budget is "
                                            f"applied"))
    horizon = get(facts, "horizon")
    defaulted = any("no horizon was stated" in note for note in request.notes)
    if (horizon is not None and request.kind is ResearchKind.ANALOGUE and defaulted
            and request.horizon_hours is not None):
        hours = int(horizon.value)
        request = replace(request, horizon_hours=hours, notes=tuple(
            n for n in request.notes if "no horizon was stated" not in n))
        used.append(remembered_line(horizon, f"read over your {_span(hours)} holding period"))
    capital = get(facts, "capital")
    if (capital is not None and request.notional is None
            and request.kind in (ResearchKind.BOOK, ResearchKind.HEDGE)
            and not re.search(r"\$|\d+\s*k\b", question, re.I)):
        from decimal import Decimal

        request = replace(request, notional=Decimal(capital.value))
        used.append(remembered_line(capital, f"sized on your ${float(capital.value):,.0f}"))
    return request, used


def after(lines: list[str], request: Any, facts: list[Fact],
          price_now: Any = None) -> list[str]:
    """Lines an answer gains from memory once it is computed: a stated loss limit set against the
    loss the answer found, a thesis beside the name it is about, and a name the trader said they
    avoid."""
    from argus.lui.research import _t

    extra: list[str] = []
    limit = get(facts, "max_loss")
    if limit is not None:
        worst = [abs(float(m.group(1))) for line in lines
                 for m in re.finditer(r"(?:book|position)\s+moves\s+(?:about\s+)?(-\d+(?:\.\d+)?)%",
                                      line)]
        if worst:
            deepest = max(worst)
            cap = float(limit.value) * 100
            extra.append(remembered_line(limit, (
                f"the worst move above, -{deepest:.1f}%, is past that limit — size this book so "
                f"that scenario costs {cap:.0f}% or less" if deepest > cap else
                f"the worst move above, -{deepest:.1f}%, stays inside it")))
    for symbol in getattr(request, "symbols", ())[:3]:
        thesis = get(facts, "thesis", symbol)
        if thesis is not None:
            now = None
            if price_now is not None:
                try:
                    now = float(price_now(symbol))
                except Exception:
                    now = None
            extra.append(thesis_line(thesis, _t(symbol), now))
        avoid = get(facts, "avoid", symbol)
        if avoid is not None:
            extra.append(remembered_line(avoid, f"{_t(symbol)} is a name you said you stay out "
                                                f"of"))
    return extra


def _span(hours: int) -> str:
    for size, unit in ((24 * 30, "month"), (24 * 7, "week"), (24, "day"), (1, "hour")):
        if hours >= size and hours % size == 0:
            count = hours // size
            return f"{count}-{unit}"
    return f"{hours}-hour"


def acknowledgement(new: list[Fact]) -> list[str]:
    """The reply to a message that only tells the console something about the trader."""
    said = "; ".join(f"“{f.text}”" for f in new)
    return [f"Actionable: noted — {said}. It is kept in this browser only and shapes every later "
            f"answer it applies to, each time with a Remembered: line saying so; forget it from "
            f"the list under your book."]


__all__ = ["KINDS", "MAX_FACTS", "Fact", "acknowledgement", "after", "apply", "dumps", "extract",
           "get", "merge", "parse", "remembered_line", "thesis_line"]
