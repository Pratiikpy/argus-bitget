"""Which reading of a research question stands: the model's, the patterns', or neither.

Two readers build a :class:`~argus.lui.research.ResearchRequest` from a question — a planner (the
language model, or with none available the trained kind model, `lui/kindmodel.py`) and the
deterministic patterns (`lui/research/parse.py`) — and each is right where the other is wrong.
Until 2026-09-27 the rules deciding between them were written inline in the console's HTTP
handler, one ``if`` after another, where no test could reach one without driving a request through
the whole server (audit finding 158). They are the same rules here, in the same order, each a named
function with the incident that made it; :func:`arbitrate` applies them and returns a
:class:`Reading` that says which reader won and why. The server only turns that into a payload.

Neither reader ever writes a figure: both only fill in the request, and the engines compute.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from argus.lui.kindmodel import LocalPlanner, kind_model
from argus.lui.question import TRADED_SYMBOLS, Intent, Question
from argus.lui.research import (
    ResearchKind,
    ResearchRequest,
    about_the_record,
    pattern_reading_wins,
    plan_with_model,
    price_forecast_asked,
    research_symbols,
    with_book,
)
from argus.lui.research import detect as detect_research

UNRELATED_CONFIDENCE = 0.8
"""How sure the planner must be that a question is unrelated before its view overrules the n-gram
layer. High on purpose: withdrawing a guess that was right costs a real answer."""

BINDING_KIND_CONFIDENCE = 0.25
"""The kind model's "refuse" or "record" overrules a research reading the patterns found only at
or above this confidence. Below it, a pattern reading stands."""

LEVERAGE_WORDS = re.compile(r"\b\d+(?:\.\d+)?\s*x\b|\bleverag\w*|\bliquidat\w*|\bmargin\b|"
                            r"杠杆|爆仓|倍", re.I)
"""What makes a question about leverage: a multiple, the word, liquidation or margin, or the
Chinese for leverage, liquidation and "times"."""


@dataclass(frozen=True)
class Reading:
    """What the arbitration decided.

    ``via`` is ``"research-model"`` or ``"research-patterns"`` when ``request`` answers the
    question, ``"forecast-refusal"`` when it asks for a future price, and ``""`` when neither
    reader claimed it and it goes on to the ledger. ``audit`` is the routing record the answer
    publishes; ``kind_said`` is a reader's confident "not research" verdict, when there was one."""

    request: ResearchRequest | None
    audit: dict[str, Any]
    via: str
    kind_said: tuple[str, float] | None = None


def kind_verdict(why: str) -> tuple[str, float] | None:
    """("refuse" | "record", confidence) from the local planner's audit line, else None."""
    match = re.match(r"kind model: (refuse|record) at ([0-9.]+)", why)
    return None if match is None else (match.group(1), float(match.group(2)))


def not_research(audit: Any) -> tuple[str, float] | None:
    """Either reader's confident verdict that a question is not a research question.

    The kind model says so in its audit line; the language model says so as ``kind: none`` or
    ``kind: record`` with a confidence. Found on the live console (2026-09-25): Qwen read "what will
    gold price be exactly one year from now" as a price forecast at 0.95 — correctly — and the
    name-only fallback still answered it as a risk profile of gold."""
    view = (audit.get("model") or {}) if isinstance(audit, dict) else {}
    local = kind_verdict(str(view.get("why") or ""))
    if local is not None:
        return local
    kind = str(view.get("kind") or "").strip().lower()
    try:
        confidence = float(view.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if kind in ("none", "record") and confidence >= UNRELATED_CONFIDENCE:
        return ("refuse" if kind == "none" else "record"), confidence
    return None


Rule = Callable[[str, str, ResearchRequest | None, ResearchRequest | None, dict[str, Any]],
                tuple[ResearchRequest | None, dict[str, Any]]]


def leverage_needs_leverage(text: str, book: str, planned: ResearchRequest | None,
                            patterned: ResearchRequest | None,
                            audit: dict[str, Any]) -> tuple[ResearchRequest | None, dict[str, Any]]:
    """A leverage reading with no leverage named is dropped for the patterns'.

    The hosted model read "Long MSTR perp into earnings — funding looks cheap" as a 10x leverage
    question (live, 2026-09-25): "perp" is not leverage. With no multiple, margin or liquidation
    named, the leverage engine has nothing it was asked."""
    if (planned is not None and planned.kind is ResearchKind.LEVERAGE
            and not LEVERAGE_WORDS.search(text)):
        return ((with_book(patterned, book, text) if patterned is not None else None),
                {**audit, "detail": "a leverage reading with no leverage named was dropped"})
    return planned, audit


def the_add_is_not_a_holding(text: str, book: str, planned: ResearchRequest | None,
                             patterned: ResearchRequest | None,
                             audit: dict[str, Any]) -> tuple[ResearchRequest | None,
                                                             dict[str, Any]]:
    """When the model's add is a name already held and the patterns found one outside the book,
    theirs is the add that was asked about.

    The hosted model read "nvda 40%, msft 20%, cash rest — want to add 5k of coin" as adding
    bitcoin, named no candidate it could resolve, and the plan fell back to a name already held —
    so the answer was about adding NVDA (live, 2026-09-25)."""
    if (planned is not None and patterned is not None
            and planned.kind is ResearchKind.IMPACT and patterned.kind is ResearchKind.IMPACT
            and planned.symbols and patterned.symbols
            and planned.symbols[0] in planned.book
            and patterned.symbols[0] not in planned.book):
        return (with_book(patterned, book, text),
                {**audit, "detail": "the model's add was a holding; the patterns' add kept"})
    return planned, audit


def the_specific_reading_stands(text: str, book: str, planned: ResearchRequest | None,
                                patterned: ResearchRequest | None,
                                audit: dict[str, Any]) -> tuple[ResearchRequest | None,
                                                                dict[str, Any]]:
    """Where the patterns name the one engine that answers the question
    (`research.pattern_reading_wins`), their reading stands over any planner reading that differs
    in a field the answer turns on.

    The model read "order book depth on NVDA" as a quote and "who is selling NVDA" as a news
    question (2026-09-24). A spot rToken holding is a field the model's plan does not carry, so a
    model reading of the same kind still loses it: "I hold RNVDAUSDT, protect it over the weekend"
    came back as a hedge of nothing and was refused on the live console (2026-09-24)."""
    if patterned is None or not pattern_reading_wins(patterned, text):
        return planned, audit
    if (planned is None or planned.kind is not patterned.kind
            or (patterned.spot is not None and planned.spot != patterned.spot)
            or planned.horizon_hours != patterned.horizon_hours
            or planned.target != patterned.target
            or planned.resize_by != patterned.resize_by
            or planned.shock_pct != patterned.shock_pct
            or planned.shock_on != patterned.shock_on
            or planned.leverage != patterned.leverage
            or (patterned.notional is not None and planned.notional != patterned.notional)
            or (planned.kind is ResearchKind.IMPACT
                and planned.symbols[:1] != patterned.symbols[:1])):
        return (with_book(patterned, book, text),
                {**audit, "detail": "the patterns' specific reading kept over the model's"})
    return planned, audit


def a_price_shock_is_a_stress(text: str, book: str, planned: ResearchRequest | None,
                              patterned: ResearchRequest | None,
                              audit: dict[str, Any]) -> tuple[ResearchRequest | None,
                                                              dict[str, Any]]:
    """A stated price shock beside a rate move is a stress, not a macro question.

    "If NVDA falls 6% and the 10-year yield rises 30bps, what happens?" against a long/short book
    was answered with the rates backdrop and no figure for the book (a hostile review, round 22):
    when the patterns read a stress with a shock and the planner read macro, the stress stands and
    the rate move is named as not applied."""
    from argus.lui.research.parse import RATE_MOVE

    if planned is None or planned.kind is not ResearchKind.MACRO or not RATE_MOVE.search(text):
        return planned, audit
    priced = detect_research(RATE_MOVE.sub(" ", text))
    if priced is not None and priced.kind is ResearchKind.STRESS and priced.shock_pct is not None:
        return (with_book(priced, book, text),
                {**audit, "detail": "a stated price shock read as a stress over macro"})
    return planned, audit


def the_stated_amounts_are_the_book(text: str, book: str, planned: ResearchRequest | None,
                                    patterned: ResearchRequest | None,
                                    audit: dict[str, Any]) -> tuple[ResearchRequest | None,
                                                                    dict[str, Any]]:
    """A stress of holdings the question states in amounts uses exactly those holdings, each on
    its side, whichever reader planned it: "long $1m QQQ and short $1m TQQQ" was stressed as $2m
    long QQQ on the live console (a hostile review, round 23)."""
    from argus.lui.research.parse import with_stated_amounts

    if planned is None or planned.kind is not ResearchKind.STRESS:
        return planned, audit
    fixed = with_stated_amounts(planned, text)
    if fixed is planned or fixed is None:
        return planned, audit
    return fixed, {**audit, "detail": "the holdings stated in amounts kept as the book"}


RULES: tuple[Rule, ...] = (leverage_needs_leverage, the_add_is_not_a_holding,
                           the_specific_reading_stands, a_price_shock_is_a_stress,
                           the_stated_amounts_are_the_book)
"""Applied in this order to the planner's reading; each may replace it with the patterns'."""


def arbitrate(text: str, *, book: str, model: Any, instruction: bool, desk_first: bool,
              audit: dict[str, Any]) -> Reading:
    """Decide which reading of ``text`` answers it.

    ``model`` is the language-model planner for this visitor, or None; with none, and unless the
    question is an order or about the desk, the kind model reads it instead — on the 2026-09-25
    held-out set (200 questions, a writer who never saw this repository, twelve languages) it read
    88.5% correctly where the patterns alone read 63.5% (`eval/kindtrain.py`), and it costs no
    token. ``audit`` is the routing record so far."""
    if model is None and not instruction and not desk_first:
        local = kind_model()
        model = LocalPlanner(local) if local is not None else None
    if model is not None:
        planned, audit = plan_with_model(text, model)
        if planned is not None and price_forecast_asked(text):
            # A price asked for a future time is refused whatever engine the model picked; it
            # read "比特币明年这个时候准确价格是多少" as a portfolio question (held-out corpus).
            planned, audit = None, {**audit, "detail": "a price forecast; refused below"}
        if (planned is None and not isinstance(model, LocalPlanner)
                and str(audit.get("detail", "")).startswith("planner unavailable")):
            # **A failed language-model call falls back to the kind model, not to the patterns.**
            # Scoring the blind set with Qwen reading first (2026-09-25) read 58.8% against the
            # kind model's 81.7%: under load some Qwen calls errored, and each error dropped the
            # question to the patterns alone.
            local = kind_model()
            if local is not None:
                planned, local_audit = plan_with_model(text, LocalPlanner(local))
                audit = {**local_audit, "fallback_from": audit.get("detail")}
        planned = with_book(planned, book, text)
        patterned = detect_research(text)
        for rule in RULES:
            planned, audit = rule(text, book, planned, patterned, audit)
        if planned is not None:
            return Reading(planned, audit, "research-model")
    # **The kind model's "not research" is binding on the research patterns.** Told a question is
    # off-topic, a trade instruction or a price forecast ("what will gold be a year from now"),
    # the patterns still found a price word and quoted it; told it is about the desk's record
    # ("the rationale logged for skipping SPY"), they found a ticker and sized a position. Binding
    # only when the patterns found nothing, or when the model is sure: "could you tell me the
    # current price of silver" is refused by the model at 0.19 — it has learnt that price
    # questions are often forecasts — and the patterns' quote is the right answer. Where the
    # patterns name the one engine that answers exactly they still stand.
    kind_said = not_research(audit)
    patterned_only = None if desk_first else detect_research(text)
    request: ResearchRequest | None
    if kind_said is not None and (patterned_only is None
                                  or kind_said[1] >= BINDING_KIND_CONFIDENCE):
        request = (with_book(patterned_only, book, text)
                   if pattern_reading_wins(patterned_only, text) else None)
    else:
        kind_said = None
        request = with_book(patterned_only, book, text)
    if request is None and price_forecast_asked(text) and research_symbols(text)[0]:
        return Reading(None, audit, "forecast-refusal", kind_said)
    if (request is None and not about_the_record(text) and not desk_first and not instruction
            and kind_said is None):
        # A question that names a listed contract the desk does not trade, in words no research
        # kind recognises ("give me a thesis on Solana for a conservative investor"), used to fall
        # through to the ledger and come back as the latest decision on an unrelated rToken. The
        # ledger has nothing on such a name; its risk profile is the honest answer, said as such.
        named = [s for s in research_symbols(text)[0] if s not in TRADED_SYMBOLS]
        if named:
            request = ResearchRequest(
                kind=ResearchKind.IMPACT, symbols=(named[0],),
                notes=((
                    # "and ETH?" was told no question was recognised while it was answered
                    # (a first-time user, round 11): a bare name is a question about the name.
                    "only the name was given"
                    if re.fullmatch(r"\s*(?:(?:and|so|but|what\s+about|how\s+about)\s+)?"
                                    r"[\w$.&/-]+\s*[?.!]*\s*", text, re.I)
                    else "no specific research question was recognised")
                    + ", so this is the name's risk profile — ask for its technicals, news, "
                      "earnings or what it does to your book for more",))
    if request is not None:
        return Reading(request, {**audit, "detail": "recognised by the research patterns"},
                       "research-patterns", kind_said)
    return Reading(None, audit, "", kind_said)

_DOMAIN = re.compile(
    r"\b(?:trad\w*|decision\w*|decid\w*|desk|positions?|holding\w*|risk\w*|sharpe|sortino|"
    r"drawdown|pnl|p&l|profit\w*|loss\w*|lose|lost|losing|money|returns?|orders?|fills?|filled|"
    r"log|ledger|record|calibrat\w*|confiden\w*|abstain\w*|abstention|pass(?:ed|es)?|skip\w*|"
    r"nothing|evidence|sources?|thesis|signals?|strateg\w*|model|hash\w*|tamper\w*|anchor\w*|"
    r"block\w*|kernel|guard\w*|weekend|sessions?|hours|market\w*|prices?|stocks?|shares?|"
    r"crypto\w*|coins?|bitcoin|hedg\w*|portfolio|book|fees?|costs?|funding|win\s+rate|exposure|"
    r"leverage|long|short|buy\w*|sell\w*|bought|sold|calls?|bets?|accura\w*|wrong|right|"
    r"perform\w*|history|past|latest|recent|last\s+(?:call|decision|trade|week|month)|why|"
    r"rtokens?|perp\w*|futures|equit\w*|index|nasdaq|volatil\w*|beta|you|your|yours|"
    r"up\s+or\s+down|in\s+the\s+(?:red|green|black)|overall|certain\w*|outcomes?|"
    r"audit\w*|entries|logged|rewrit\w*|modifi\w*|retroactiv\w*|holiday|inaction|"
    r"sidelines?|informat\w*|data\s+points?|recap|activity|tickers?|authentic\w*|conviction|"
    r"overconfiden\w*|minutes?|pre[\s-]?market|after[\s-]?hours|inputs?|indicators?)\b|"
    r"交易|决策|决定|操作|仓位|持仓|风险|收益|盈亏|亏损|盈利|赚|亏|胜率|表现|订单|市场|价格|策略|对冲|"
    r"股票|币|夏普|回撤|记录|账本|日志|修改|篡改|验证|加密|置信|确信|依据|判断|消息|信息|观望|平仓|"
    r"开仓|做空|做多|收盘|开盘|休市|盘前|盘后|时段|把握|信心|自信|参考|指标|敞口|删改|手脚|"
    r"为什么|理由|证据|校准|你", re.I)
"""Words that make a question about markets or the desk. Deliberately wide — it exists to stop
the n-gram layer answering chit-chat, not to judge a trading question."""


def in_domain(text: str) -> bool:
    if research_symbols(text)[0]:
        return True
    hit = _DOMAIN.search(text)
    if hit is None:
        return False
    # "you" alone is not a desk word: "can you review my resume" is not about the record. It
    # counts only with a second domain word or when the question is addressed to the desk's
    # conduct ("why did you ...", "what did you ...").
    words = {m.group(0).lower() for m in _DOMAIN.finditer(text)}
    if words <= {"you", "your", "yours", "你"}:
        return bool(re.search(r"\b(?:why|what|when|how)\s+(?:did|do|have|were)\s+you\b", text,
                              re.I))
    return True


_DAY_QUESTION_WORDS = frozenset({
    "what", "happened", "happen", "did", "do", "does", "you", "we", "on", "the", "last", "this",
    "past", "yesterday", "today", "week", "weekend", "day", "and", "so", "then", "in", "over",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "mon", "tue",
    "tues", "wed", "thu", "thurs", "fri", "sat", "sun", "was", "were", "decided", "decide",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "sep", "sept", "oct", "nov", "dec", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug"})


def _names_no_subject(text: str) -> bool:
    """A day question with no subject of its own — "what happened last Friday" — which, asked of
    this console, is asked of its record. It was declined as off-topic whenever the model was
    unavailable (a hostile review, round 11), while "what did the Lakers do last Friday" names a
    subject and stays declined."""
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and all(w in _DAY_QUESTION_WORDS for w in words)


def gate_ledger_reading(question: Question, classified_by: str, text: str, *, prior: list[str],
                        audit: Any, desk_first: bool) -> tuple[Question, str]:
    """The n-gram layer's reading of a ledger question, after two gates.

    **The n-gram layer only knows wording, so it needs a topic gate.** It answers with a class for
    any text at all, and on the 2026-09-25 blind corpus "who won the lakers game last night"
    reached `decision_why` at 0.19 and was answered with a decision, as were a CV review and
    "explain quantum computing" (which the hand-written patterns claimed on the one word
    "explain"). A question with no market, trading or desk word in it is not a question about the
    record. A follow-up ("explain that") is exempt: it inherits the topic of the turn before it.
    An order is refused as an order whatever language it is in; "अभी 1 बिटकॉइन खरीद लो" was
    recognised as an order and then declined as off-topic (2026-09-25 audit, round 2).

    **A confident "unrelated" from the planner overrules the n-gram layer's guess.** "Tell me a
    joke about NVDA" reached `decision_why` and was answered with a decision's thesis. When the
    planner that already read the question said, with confidence, that it is neither research nor
    about the desk's record, the n-gram guess is withdrawn and the question goes to the refusal.
    The hand-written patterns are never overruled this way — only the statistical layer is. The
    kind model's refusal counts once it clears the model's own threshold: since 2026-09-27 the plan
    carries its real probability on the plan's scale (`kindmodel.plan_confidence`), which a
    language model's 0.8 bar would read differently (audit 159)."""
    if ((classified_by == "ngram" or (classified_by == "patterns" and not prior))
            and not in_domain(text) and question.intent is not Intent.ORDER and not desk_first
            # "上周五做了什么" (what was done last Friday) names a day and asks what was done: the
            # record's own question, declined as off-topic for want of a desk word (2026-09-30).
            # English names its subject ("the desk", "you") and is read by `in_domain`; "what did
            # the Lakers do last Friday" must still be declined.
            and not (classified_by == "patterns" and question.intent is Intent.DECISION_LIST
                     and question.window is not None and (not re.search(r"[A-Za-z]", text)
                                                          or _names_no_subject(text)))):
        question = replace(question, intent=Intent.UNKNOWN,
                           reason="nothing in the question is about markets or the desk's record")
        classified_by = "declined-off-topic"
    view = (audit.get("model") or {}) if isinstance(audit, dict) else {}
    try:
        confidence = float(view.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    local_refusal = (kind_verdict(str(view.get("why") or "")) or ("", 0.0))[0] == "refuse"
    if (classified_by == "ngram" and str(view.get("kind", "")).lower() == "none"
            and (confidence >= UNRELATED_CONFIDENCE or local_refusal)):
        question = replace(question, intent=Intent.UNKNOWN,
                           reason="the model read this as unrelated to research or the record")
        classified_by = "declined-by-model"
    return question, classified_by
