"""The facts a trader states about themselves, read by the language model and checked by code.

Measured against mem0 on a blind held-out set (`eval/memory_comparison.py`, held_out_2,
2026-09-30), the pattern reader in `lui/memory.py` recalled 2 of 25 stated facts and mem0 25 of 25:
people say "I cant go past 12%" and "I'd hold mine for like a year", and no fixed pattern list
keeps up with that. mem0 also stored 10 of 15 sentences that were not about the trader at all —
a friend's habit, a hypothetical — which the pattern reader never did.

So this takes mem0's approach to recall (a model reads the message and extracts every fact,
`mem0/memory/main.py` ADDITIVE_EXTRACTION_PROMPT, Apache-2.0: the idea, not the prompt) and keeps
ARGUS's approach to trust:

* **Only when a message could carry a fact** — a fact word, not a bare question, not opening on
  someone else — so a plain research question costs no extra call. The question has already been
  counted against the visitor's model allowance; this rides on it.
* **Every fact must quote its words from the message.** A fact whose quote is not in the message,
  or whose number is not in its quote, is dropped: the model can find a fact, it cannot invent one.
* **Code, not the model, rejects the sentences mem0 stored wrongly**: a quote that opens on a
  hypothetical ("if", "what if", "suppose") or on someone else ("my friend", "he", "they") is not
  about the trader.
* **The kinds are `lui/memory.py`'s**, so merge, replacement history and every use downstream are
  unchanged; a fact of no known kind is kept as a ``goal`` only when it says what the money is for.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from argus.lui.memory import KINDS, MAX_TEXT, Fact
from argus.lui.trace import trace_module

SELF_DISCLOSURE = re.compile(
    r"\b(?:i|i'?m|im|i'?ve|ive|i'?d|id|i'?ll|ill|my|me|mine|we|our)\b", re.I)
FACT_WORDS = re.compile(
    r"\d|%|\$|\b(?:lose|loss|risk|hold|horizon|years?|months?|weeks?|days?|account|capital|"
    r"savings?|saving|retire\w*|goal|budget|conservative|aggressive|cautious|swing|day\s+trad\w*|"
    r"long[\s-]term|avoid|never|won'?t|don'?t|think|believe|expect|bullish|bearish|max|limit|"
    r"stop|downside|hit|out|sit|hold\w*|chase|type|playing|touch\w*|clear|confident|breaks?|"
    r"under|over|rule|fund|loans?|debt|pay\s+off|wedding|house|home|car|college|kids?|"
    r"independence|earmarked|aside|risk[\s-]?(?:on|off)|gambl\w*|conviction)\b|\d+\s*k\b", re.I)
"""A message worth reading for facts: a word that could carry one.

First person is not required: "got about 220k sitting in this account" and "not touching FTT
again" are the trader speaking, typed without an "I" — held_out_3 of the mem0 comparison (round
12) lost 11 of 20 facts at this gate when it asked for one. Questions and statements about someone
else are still kept out, below."""

NOT_ABOUT_ME = re.compile(
    r"^\s*(?:if|what\s+if|suppose|imagine|let'?s\s+say|say|hypothetically|would|could|should|"
    r"my\s+(?:friend|brother|sister|dad|father|mum|mom|mother|wife|husband|partner|boss|colleague)s?|"
    r"(?:he|she|they|people|everyone|someone|a\s+friend)\b)", re.I)
"""A quote that opens on a hypothetical or on someone else states nothing about the trader."""

PROMPT = """You read one message a trader typed into a market research console and list every fact
it states about THIS trader themselves. Kinds (use only these):
- budget: the share of their risk or book one position may carry, or a stated "risk budget",
  as a percent number (e.g. 25)
- max_loss: the most they can lose, as a percent number (e.g. 10)
- horizon: how long they hold, as a number of hours (a day = 24, a week = 168, a month = 720,
  a year = 8760)
- style: one word or hyphenated phrase (conservative, aggressive, cautious, day-trader,
  swing-trader, long-term, momentum, earnings, news, macro)
- capital: their account or trading money in US dollars, as a number
- avoid: a ticker or asset they will not trade
- thesis: a view they hold on a ticker, value "bull", "bear" or "view", subject = the ticker
- goal: what the money is for (a house, retirement, a car, school fees), value = a few words
Rules: only facts the trader states about themselves now; never a question, a hypothetical, or
what someone else thinks or does. A statement with no "I" is still the trader's ("got 50k in
here", "not touching FTT again"). One sentence can carry several facts: "in and out same day" is
both a day-trader style and a horizon of 24 hours. For each fact, "quote" must be copied exactly,
character for character, from the message. Answer JSON only: {"facts": [{"kind": "...",
"value": "...", "subject": "", "quote": "..."}]} and {"facts": []} when it states none."""

STYLES = frozenset({
    "conservative", "aggressive", "cautious", "risk-averse", "day-trader", "swing-trader",
    "long-term", "position-trader", "momentum", "earnings", "news", "macro", "income", "value",
    "growth", "breakout", "mean-reversion"})
"""The styles a stated style may be read as."""

_HOURS = {"hour": 1, "day": 24, "week": 168, "month": 720, "year": 8760}


def reads_as_disclosure(text: str) -> bool:
    """Worth one model call: a fact word, not a bare question, and not opening on someone else
    or on a hypothetical."""
    stripped = text.strip()
    if not FACT_WORDS.search(stripped) or NOT_ABOUT_ME.search(stripped):
        return False
    return not (stripped.endswith("?") and not SELF_DISCLOSURE.search(stripped))


_A_HOLDING = re.compile(r"\b(?:i\s+(?:hold|own|have)|holding|i'?m\s+(?:long|in))\b", re.I)
_A_LIMIT = re.compile(r"\b(?:max(?:imum)?|no\s+more|at\s+most|up\s+to|cap|limit|budget|rule|never|"
                      r"above|over|under|past|per\s+(?:trade|position|name))\b", re.I)
_A_POSITION = re.compile(r"\W*\$?\d[\d,.]*[%k]?\s+(?:in\s+|of\s+)?\S+(?:\s+\S+){0,3}\W*$", re.I)
"""An amount then a name and a word or two ("50% ETH perp long"): what the trader holds, no view."""
_A_CLAIM = re.compile(
    r"\b(?:will|won'?t|is|are|was|going|gonna|should|could|can|keeps?|has|have|needs?|because|"
    r"since|to)\b|\b(?:up|rise|rall|outperform|beat|higher|moon|rip|double|recover|bounce|down|fall|"
    r"drop|crash|underperform|miss|lower|dump|tank|decline|lose|run|grow|slow)\w*", re.I)
"""A thesis quote states something about the name; a bare noun phrase is not a view."""
_AN_ORDER = re.compile(r"\s*(?:please\s+)?(?:go\s+)?(?:long|short|buy|sell|open|close)\b", re.I)


def _valid(kind: str, value: str, subject: str, quote: str, context: str = ""
           ) -> tuple[str, str] | None:
    """The fact's (value, subject) as `lui/memory.py` stores them, or None when the quote does not
    carry what the model claimed."""
    from argus.lui.research import research_symbols

    numbers = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", quote)]
    if kind in ("budget", "max_loss"):
        try:
            pct = float(value)
        except ValueError:
            return None
        if not 0 < pct <= 100 or not any(abs(n - pct) < 1e-9 for n in numbers):
            return None
        around = context or quote
        if kind == "budget" and _A_HOLDING.search(around) and not _A_LIMIT.search(around):
            # "I hold 40% NVDA" is what the book holds, not what one name may carry: it was kept
            # as a 40% risk budget (a judge, round 13, 2026-09-30). The book keeps it instead.
            return None
        if kind == "budget" and not _A_LIMIT.search(quote):
            # "50% ETH perp long" in a message that also names a limit elsewhere: the limit word
            # must sit in the quote itself, or the quote is a holding (round 14, 2026-09-30).
            return None
        return str(pct / 100), ""
    if kind == "capital":
        try:
            amount = float(value)
        except ValueError:
            return None
        scaled = {n * m for n in numbers for m in (1, 1_000, 1_000_000)}
        if amount < 100 or not any(abs(s - amount) < 1e-6 for s in scaled):
            return None
        return f"{amount:.0f}", ""
    if kind == "horizon":
        try:
            hours = int(float(value))
        except ValueError:
            return None
        word = re.search(r"\b(hour|day|week|month|year)s?\b", quote, re.I)
        styled = re.search(r"\b(day|swing|position|long[\s-]?term)\b", quote, re.I)
        if hours <= 0 or not (word or styled):
            return None
        return str(hours), ""
    if kind == "style":
        slug = re.sub(r"[\s_]+", "-", value.strip().lower())
        # A style is a reading of the quote ("I chase the big moves" is aggressive), so the word
        # need not appear in it; it must be one of the styles the console acts on.
        if slug not in STYLES:
            return None
        return slug, ""
    if kind in ("avoid", "thesis"):
        named = research_symbols(subject or quote)[0]
        if not named:
            # A ticker Bitget no longer lists is still one the trader named ("not touching FTT
            # again"): kept as written, in capitals, when the quote carries it.
            raw = re.sub(r"[^A-Za-z0-9]", "", subject).upper()
            if not raw or not re.search(rf"\b{re.escape(raw)}\b", quote, re.I):
                return None
            named = (raw,)
        if kind == "thesis" and (_AN_ORDER.match(quote) or _A_POSITION.match(quote)):
            # "Long rNVDA over the weekend at 3x" is an order to price, not a view to keep: it
            # replaced the trader's stated NVDA thesis (a judge, round 13, 2026-09-30).
            return None
        if kind == "thesis" and not _A_CLAIM.search(quote):
            # "SOL short leg", taken from a sizing question, is a noun phrase: it was kept as a
            # thesis and later shown as the one a real thesis replaced (round 14, live).
            return None
        lean = value.strip().lower() if kind == "thesis" else "avoid"
        if kind == "thesis" and lean not in ("bull", "bear", "view"):
            lean = "view"
        return lean, named[0]
    if kind == "goal":
        words = set(re.findall(r"[a-z]{3,}", value.lower())) - {"the", "for", "and", "pay", "off"}
        if not words or not words & set(re.findall(r"[a-z]{3,}", quote.lower())):
            return None
        return value.strip()[:80], ""
    return None


def extract(text: str, client: Any, now: datetime | None = None) -> list[Fact]:
    """Facts the model finds in ``text``, each checked against the message; [] when the message
    is not self-disclosure, no model is available, or nothing checks out."""
    if client is None or not reads_as_disclosure(text):
        return []
    from argus.lui.question import ORDER_VERB

    first_person = re.search(r"\b(?:i|i['\u2019](?:m|ve|d|ll)|my|me|mine)\b", text, re.I)
    if ORDER_VERB.match(text) and not first_person:
        # "buy $500 of BTC" is an order to price: the model read the amount as the trader's
        # capital and the answer became "noted — $500" (a first-user audit, round 18, row 631).
        return []
    try:
        from argus.llm.qwen import Thinking

        raw = client.complete_json(
            [{"role": "system", "content": PROMPT}, {"role": "user", "content": text[:600]}],
            required_keys=("facts",), max_tokens=400, thinking=Thinking.OFF)
    except Exception:
        return []  # the pattern reader's facts stand on their own
    day = (now or datetime.now(UTC)).date().isoformat()
    out: list[Fact] = []
    seen: set[tuple[str, str]] = set()
    for row in raw.get("facts") or []:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind", "")).strip().lower()
        quote = str(row.get("quote", "")).strip()
        if kind not in KINDS or kind == "check" or not quote or quote not in text:
            continue
        if NOT_ABOUT_ME.search(quote):
            continue
        # The sentence the quote sits in: "40% NVDA" is a holding only beside "I hold".
        context = next((part for part in re.split(r"(?<=[.!?;])\s+", text) if quote in part),
                       quote)
        checked = _valid(kind, str(row.get("value", "")), str(row.get("subject", "")), quote,
                         context)
        if checked is None or (kind, checked[1]) in seen:
            continue
        seen.add((kind, checked[1]))
        out.append(Fact(kind=kind, subject=checked[1], value=checked[0], text=quote[:MAX_TEXT],
                        at=day))
    return out


def combined(text: str, client: Any, now: datetime | None = None,
             price_of: Any = None) -> list[Fact]:
    """The pattern reader's facts, and the model's for every kind the patterns missed: the
    patterns are exact where they fire, the model reads what they cannot."""
    from argus.lui import memory

    found = memory.extract(text, now, price_of=price_of)
    have = {f.key() for f in found}
    return [*found, *(f for f in extract(text, client, now) if f.key() not in have)]


trace_module(globals())
