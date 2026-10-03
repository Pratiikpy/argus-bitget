"""Tripwires: a trader decides what they will do before the market tests them, and the console
holds them to it.

"If BTC drops below 80k I'll sell half" is a decision made while calm. The moment it matters is
the moment the trader is least able to make it. ARGUS keeps the sentence (in the browser's memory,
like every other fact the trader states — `lui/memory.py`), checks it against Bitget's own prices
on every later answer about that name, and when the level has traded, replays the decision: what
following it did from the first hour the level traded to now, said as a number, not a judgement.

Taken from Prequel, a Bitget S2 entry (its "tripwire" pre-commitment and replay,
``research/s2-field/youtube/bitget-s2/13_prequel_decision_stress_testing.md``); its implementation
was not published, so this is ARGUS's own. Different from ``lui/watch.py``, which is a price alert
that fires once in the Telegram bot: a tripwire carries the trader's own planned action, survives
in the console's memory, and is replayed against what happened rather than only announced.

What is measured is the market after the level traded, at Bitget's hourly closes: for a plan to
reduce or hedge, the move it would have stepped out of; for a plan to add, the move it would have
caught. No order is placed and no advice is given; the trader's words are quoted as said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from argus.lui.trace import trace_module

TRIPWIRE: Final = re.compile(
    r"\b(?:if|when|once|should)\s+(?P<sym>\$?[A-Za-z][A-Za-z0-9]{1,11})\s+"
    r"(?:price\s+)?(?:ever\s+)?"
    r"(?P<verb>drops?|falls?|dips?|slides?|sinks?|goes|gets|trades?|breaks?|closes?|is|moves?|"
    r"rises?|climbs?|tops?|hits?|reaches?|crosses?|crosses|pushes|rallies)?\s*"
    r"(?P<dir>below|under|beneath|above|over|past|through|to|back\s+to)?\s*"
    r"\$?(?P<lvl>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k\b)?"
    r"[^.?!]{0,12}?,?\s*(?:then\s+)?i(?:(?:'ll|\s+will|\s+would|\s+am\s+going\s+to|\s+plan\s+to|"
    r"\u2019ll)\s+(?P<act>[^.?!]{3,90})|"
    # "if ETH goes under 2500 i sell all", "if eth hits 3000 i take profit": the plain present
    # tense, accepted only when an action verb follows (a first-time user, round 27)
    r"\s+(?P<act2>(?:sell|buy|take|cut|trim|exit|close|add|hedge|short|reduce|dump|get\s+out|"
    r"go\s+long|go\s+short|lighten|flatten|load)\b[^.?!]{0,80}))",
    re.I,
)
""""If BTC drops below 80k I'll sell half", "when NVDA breaks 200, I will add 10%", "once ETH
hits 2,400 I'm going to hedge" — a condition on one name's price and the trader's own action."""

_BELOW_WORDS: Final = re.compile(r"below|under|beneath|drop|fall|dip|slide|sink", re.I)
_ABOVE_WORDS: Final = re.compile(r"above|over|past|rise|climb|top|rall|push", re.I)
_REDUCE: Final = re.compile(
    r"\b(?:sell|cut|trim|exit|close|dump|take\s+(?:some\s+)?profits?|reduce|get\s+out|short|"
    r"flatten|stop\s+out|lighten)\b",
    re.I,
)
_ADD: Final = re.compile(r"\b(?:buy|add|go\s+long|enter|accumulate|double|load|scale\s+in)\b",
                         re.I)
_HEDGE: Final = re.compile(r"\bhedge\b", re.I)

ASKED: Final = re.compile(
    r"\b(?:check|show|replay|review|status\s+of|list|where\s+(?:are|is))\s+(?:on\s+)?my\s+"
    r"(?:tripwires?|pre-?commit\w*|plans?|triggers?|if-then\s+rules?)\b|\bmy\s+tripwires?\b|"
    r"\bdid\s+(?:my\s+)?(?:tripwire|plan)\s+(?:fire|trigger|hit)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class Plan:
    symbol: str
    side: str
    """``below`` or ``above``: the side of the level that fires it."""
    level: float
    action: str
    """``reduce``, ``add``, ``hedge`` or ``other``."""
    words: str
    set_at: int = 0
    """When the plan was set, Unix seconds; a level that traded before this does not fire it."""


def read(text: str, price_now: float | None = None, *,
         now: datetime | None = None) -> Plan | None:
    """The tripwire a message states, or ``None``. ``price_now`` settles an unsided level ("if
    BTC hits 80k"): a level under the price fires on the way down, one over it on the way up."""
    from argus.lui.research import research_symbols

    m = TRIPWIRE.search(text)
    if m is None:
        return None
    symbols = research_symbols(m.group("sym"))[0]
    if not symbols:
        return None
    level = float(m.group("lvl").replace(",", "")) * (1000 if m.group("k") else 1)
    if level <= 0:
        return None
    said = f"{m.group('verb') or ''} {m.group('dir') or ''}"
    if _BELOW_WORDS.search(said):
        side = "below"
    elif _ABOVE_WORDS.search(said):
        side = "above"
    elif price_now:
        side = "below" if level < price_now else "above"
    else:
        return None
    act = m.group("act") or m.group("act2") or ""
    action = ("hedge" if _HEDGE.search(act) else "reduce" if _REDUCE.search(act)
              else "add" if _ADD.search(act) else "other")
    return Plan(symbol=symbols[0], side=side, level=level, action=action,
                words=m.group(0).strip(), set_at=int((now or datetime.now(UTC)).timestamp()))


def value_of(plan: Plan) -> str:
    """How a plan is stored in a memory fact's ``value`` (40 characters at most): side, level,
    action and the second it was set."""
    return f"{plan.side}|{plan.level:.10g}|{plan.action}|{plan.set_at}"


def from_value(symbol: str, value: str, words: str) -> Plan | None:
    try:
        side, level, action, *rest = value.split("|")
        if side not in ("below", "above"):
            return None
        return Plan(symbol=symbol, side=side, level=float(level), action=action, words=words,
                    set_at=int(rest[0]) if rest else 0)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Status:
    plan: Plan
    price_now: float
    fired_at: datetime | None
    """The first hourly candle whose range reached the level, since the plan was stated."""
    already_true: bool
    """The price was already past the level when the plan was stated."""


def _name(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def check(plan: Plan, stated_on: str, price_at: float | None, *,
          now: datetime | None = None) -> Status | None:
    """Whether the level has traded since the day the plan was stated, from Bitget's hourly
    candles (high and low, so a wick through the level counts, as a resting order would)."""
    from argus.market import history

    when = now or datetime.now(UTC)
    if plan.set_at:
        start = datetime.fromtimestamp(plan.set_at, UTC)
    else:
        try:
            start = datetime.fromisoformat(stated_on).replace(tzinfo=UTC)
        except ValueError:
            start = when - timedelta(days=1)
    # only candles that opened after the plan was set: the hour it was set in held prices from
    # before it, and a level that traded then is not the plan firing
    first_hour = start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    try:
        # from a few hours before, so a plan set this hour still has a latest close to read
        candles = history.fetch_window(plan.symbol, start=start - timedelta(hours=3), end=when,
                                       interval="1H")
    except Exception:
        return None
    if not candles:
        return None
    price_now = float(candles[-1].close)
    already = price_at is not None and (
        (plan.side == "below" and price_at <= plan.level)
        or (plan.side == "above" and price_at >= plan.level))
    fired = next((c.ts for c in candles if c.ts >= first_hour
                  and ((plan.side == "below" and float(c.low) <= plan.level)
                  or (plan.side == "above" and float(c.high) >= plan.level))), None)
    return Status(plan=plan, price_now=price_now, fired_at=fired, already_true=already)


def _plan_words(plan: Plan) -> str:
    return (f"if {_name(plan.symbol)} trades {plan.side} {plan.level:,.10g}, "
            + {"reduce": "you reduce", "add": "you add", "hedge": "you hedge"}.get(
                plan.action, "you act"))


def acknowledgement(plan: Plan, price_now: float | None) -> list[str]:
    """The reply to a message that sets a tripwire."""
    where = ""
    if price_now:
        gap = plan.level / price_now - 1
        where = (f" {_name(plan.symbol)} is {price_now:,.6g} now, "
                 f"{abs(gap):.1%} {'above' if gap < 0 else 'below'} that level"
                 + ("; it is already past it, so the plan is live now." if (
                     (plan.side == "below" and price_now <= plan.level)
                     or (plan.side == "above" and price_now >= plan.level)) else "."))
    return [f"Bottom line: tripwire set — {_plan_words(plan)}, in your words “{plan.words}”."
            + where,
            "It is kept in this browser with the rest of what you have told the console. Every "
            f"later answer about {_name(plan.symbol)} checks it against Bitget's hourly prices, "
            "and \"check my tripwires\" replays what acting on it did once the level trades. "
            "No order is placed; the decision stays yours."]


def status_line(status: Status) -> str:
    """One line for a tripwire, fired or not, with the replay when it has fired."""
    plan = status.plan
    name = _name(plan.symbol)
    head = f"Tripwire ({_plan_words(plan)}):"
    past = ((plan.side == "below" and status.price_now <= plan.level)
            or (plan.side == "above" and status.price_now >= plan.level))
    if status.fired_at is None and past:
        # "change it to 85k" with BTC at 84.7k: the level is already behind the price, so the
        # plan is live now — the moment the trader set it for (a first-time user, round 27)
        return (f"{head} live now — {name} is {status.price_now:,.6g}, already "
                f"{plan.side} {plan.level:,.10g}; the condition you set is met.")
    if status.fired_at is None:
        gap = plan.level / status.price_now - 1
        return (f"{head} not reached — {name} is {status.price_now:,.6g}, "
                f"{abs(gap):.1%} {'above' if gap < 0 else 'below'} the level.")
    move = status.price_now / plan.level - 1
    since = f"{status.fired_at:%d %b %H:%M} UTC"
    if plan.action in ("reduce", "hedge"):
        effect = (f"stepping out then avoided a {abs(move):.1%} further fall" if move < 0
                  else f"stepping out then missed a {move:.1%} recovery")
    elif plan.action == "add":
        effect = (f"adding then caught a {move:.1%} rise" if move > 0
                  else f"adding then took a {abs(move):.1%} further fall")
    else:
        effect = f"{name} has moved {move:+.1%} since"
    already = (" It was already past the level when you set it." if status.already_true
               else "")
    return (f"{head} fired — {name} first traded {plan.side} {plan.level:,.10g} at {since}; "
            f"it is {status.price_now:,.6g} now, so {effect} (from the level, before fees and "
            f"slippage).{already}")


def subject_of(plan: Plan) -> str:
    """A tripwire's memory subject: the symbol and its side, so a stop and a take-profit on the
    same name are two facts, not one replacing the other."""
    return f"{plan.symbol}|{plan.side}"


def symbol_of(subject: str) -> str:
    return subject.split("|", 1)[0]


def lines_for(facts: list[Any], symbols: tuple[str, ...] | None = None, *,
              now: datetime | None = None) -> list[str]:
    """Status lines for the trader's tripwires — every one, or only those on ``symbols``."""
    out: list[str] = []
    for fact in facts:
        if getattr(fact, "kind", "") != "tripwire":
            continue
        if symbols is not None and symbol_of(fact.subject) not in symbols:
            continue
        plan = from_value(symbol_of(fact.subject), fact.value, fact.text)
        if plan is None:
            continue
        status = check(plan, fact.at, fact.price_at, now=now)
        if status is not None:
            out.append(status_line(status) + f" Set on {fact.at}.")
    return out


FOLLOW_UP: Final = re.compile(
    r"\bhow\s+far\s+(?:are\s+we|is\s+it|am\s+i|away)\b|\bhow\s+close\s+(?:are\s+we|is\s+it)\b|"
    r"\bdid\s+it\s+(?:fire|trigger|hit|go\s+off)\b|\bhas\s+it\s+(?:fired|triggered|hit)\b",
    re.I,
)
"""A question about the tripwire just set or checked, by pronoun ("how far are we from it",
"ok did it fire now"): both were declined after 26 seconds (a first-time user, round 27)."""

CHANGE: Final = re.compile(
    r"\b(?:change|move|set|make|update|raise|lower)\s+(?:it|that|the\s+(?:level|tripwire|trigger))"
    r"\s+to\s+\$?(?P<lvl>\d[\d,]*(?:\.\d+)?)\s*(?P<k>k\b)?",
    re.I,
)
DELETE: Final = re.compile(
    r"\b(?:delete|remove|cancel|drop|clear|forget)\s+(?:the\s+|my\s+|that\s+)?(?:(?P<sym>[A-Za-z]"
    r"{2,10})\s+)?(?:one|tripwires?|plan|trigger)\b|\b(?:delete|remove|cancel|clear)\s+(?:all\s+)?"
    r"my\s+tripwires\b",
    re.I,
)


def about_tripwires(prior: list[str]) -> bool:
    """Whether the last two turns were about a tripwire."""
    return any(TRIPWIRE.search(q) or ASKED.search(q) or FOLLOW_UP.search(q) or CHANGE.search(q)
               for q in prior[-2:])


def latest(facts: list[Any]) -> Any | None:
    """The tripwire set most recently (memory keeps the newest first)."""
    return next((f for f in facts if getattr(f, "kind", "") == "tripwire"), None)


__all__ = [
    "ASKED",
    "CHANGE",
    "DELETE",
    "FOLLOW_UP",
    "TRIPWIRE",
    "Plan",
    "Status",
    "about_tripwires",
    "acknowledgement",
    "check",
    "from_value",
    "latest",
    "lines_for",
    "read",
    "status_line",
    "subject_of",
    "symbol_of",
    "value_of",
]



trace_module(globals())
