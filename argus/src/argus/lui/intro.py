"""What this console is, who it is for, and what to ask it: the first question a stranger asks.

"What is this site and who is it for" was answered '"that" has nothing to refer to yet', and "what
is ARGUS" and "what can you do" were declined (a first-time user, 2026-09-30). The product's own
description was on the README and nowhere a visitor could ask for it. The standing counts are read
from the register at the moment of asking, so this answer cannot drift from /proof.
"""

from __future__ import annotations

import re
from typing import Any

from argus.lui.provenance import explains
from argus.lui.trace import trace_module
from argus.truth.source import Source

_SUBJECT = (r"(?:argus|this(?:\s+(?:site|website|page|tool|app|console|product|thing|desk|"
            r"service|project))?)")

INTRO_Q = re.compile(
    # "hi, I'm new here. what is this and what can you do?" was declined (a first-time user,
    # round 19, row 636): a greeting and "I'm new" are a preamble, not the question
    # "hi" alone, and "hlo, total noob here. what even is this", were declined (round 23)
    rf"^\s*(?:hi+|hello|hey|hlo|hii+|yo|namaste|gm)\W*$|"
    rf"^\s*(?:(?:hi+|hello|hey|hlo|yo)(?:\s+there)?\W+)?(?:(?:total\s+|complete\s+)?(?:noob|newbie|"
    rf"beginner)\s+here\W+)?(?:i'?m\s+new(?:\s+(?:here|to\s+(?:this|trading|"
    rf"crypto|investing)))?\W+|first\s+time\s+here\W+)?"
    rf"(?:so\s+)?(?:what\s+(?:even\s+)?(?:is|'s)\s+{_SUBJECT}(?:\s+even)?|what\s+does\s+"
    rf"{_SUBJECT}\s+do|who\s+(?:is|'s)\s+(?:{_SUBJECT}|it)\s+for|who\s+(?:made|built)\s+{_SUBJECT}|"
    rf"what\s+can\s+(?:you|argus|this(?:\s+\w+)?)\s+do(?:\s+for\s+me)?|what\s+(?:can|should)\s+i\s+"
    rf"ask(?:\s+(?:you|it|argus|here))?|how\s+do\s+i\s+use\s+{_SUBJECT}|what\s+am\s+i\s+looking\s+at|"
    rf"help|getting\s+started|where\s+do\s+i\s+start|who\s+are\s+you)"
    rf"(?:\s*(?:,|and)\s*(?:who\s+(?:is|'s)\s+(?:{_SUBJECT}|it)\s+for|what\s+(?:does\s+{_SUBJECT}\s+do|"
    rf"can\s+(?:you|it)\s+do)))?\s*[?.!]*\s*$",
    re.I)
"""A question about the product itself, asked whole: "what is this site and who is it for"."""


ASK_Q = re.compile(
    r"^\s*(?:(?:hi|hello|hey)\W+)?(?:so\s+)?what\s+(?:can|should|could)\s+i\s+ask"
    r"(?:\s+(?:you|it|argus|here|this))?\s*[?.!]*\s*$|^\s*(?:give\s+me|show\s+me)\s+(?:some\s+)?"
    r"(?:example\s+questions|examples|ideas)\s*[?.!]*\s*$", re.I)
"""The question a visitor asks right after the introduction: "what can I ask" returned the same
paragraph as "what is this" (a first-time user, round 18, row 632)."""


NO_NAME_BUY_Q = re.compile(
    r"^\W*(?:(?:ok(?:ay)?|so|well|hmm|right|alright)\W+)*(?:what\s+should\s+i\s+(?:buy|invest\s+in|"
    r"get|trade|put\s+(?:my\s+)?money\s+in)|what\s+(?:do\s+you\s+recommend|would\s+you\s+buy)|"
    r"what'?s\s+a\s+good\s+(?:buy|investment|trade)|"
    # "whats a good thing to buy rn" was declined (a first-time user, round 21)
    r"what(?:'?s|\s+is)\s+(?:a\s+)?(?:good|smart|safe)\s+(?:thing|stock|coin|crypto|one)\s+to\s+"
    r"(?:buy|get|invest\s+in)|should\s+i(?:\s+(?:buy|invest|do\s+it|go\s+for\s+"
    r"it|buy\s+(?:it|now|something)))?)\s*(?:now|then|today|rn|right\s+now|atm)?\s*[?.!]*\s*$",
    re.I)
"""A buy question with nothing named: "what should I buy", "ok so should I?" (a first-time user,
round 19, row 636 — both were declined with the record reader's list)."""

THANKS_Q = re.compile(
    r"^\W*(?:ok(?:ay)?\W+)?(?:thanks?|thank\s+you|thx|ty|cheers|great|perfect|cool|nice|awesome|"
    r"got\s+it)\b(?![^?]*\b(?:what|how|why|when|where|which|should|can|could|do\s+i|is\s+it|"
    r"tell\s+me)\b)[^?]*$", re.I)
"""Thanks or an acknowledgement with no question in it: "ok thank you, what i do first step
today" carries one, and was answered with the thanks alone (a first-time user, round 30)."""

THAT_NUMBER_Q = re.compile(
    r"^\W*(?:and\s+|so\s+)?what\s+(?:does|do)\s+(?:that|this|those|these)\s+(?:number|figure|"
    r"numbers|figures|percentage|stat)s?\s+mean\b|^\W*(?:i\s+)?(?:don'?t|do\s+not)\s+understand"
    r"(?:\s+(?:that|this|the\s+numbers?))?\W*(?:lol|lmao|tbh|tho|bro|pls|please)?\W*$|"
    # "what does that mean lol" was told "that" had nothing to refer to (round 21)
    r"^\W*(?:ok(?:ay)?\W+|so\W+|um+\W+)?(?:wh?at|wut|wat)\s+(?:does|do|did)\s+(?:that|this|it|"
    r"all\s+that)\s+(?:even\s+)?mean\W*(?:lol|lmao|tbh|tho|bro|pls|please|for\s+me)?\W*$|"
    r"^\W*(?:explain|say)\s+(?:that|this|it)\s+(?:simpler|more\s+simply|in\s+plain\s+"
    r"(?:english|words)|like\s+i'?m\s+(?:5|five|new))\W*$|"
    # "explain simpler pls, im 5" got a desk decision, and "yaar thoda simple mein samjhao" was
    # declined (a first-time user, round 23)
    r"^\W*(?:can\s+you\s+|pls\s+|please\s+|ok\s+)?explain\s+(?:it\s+|that\s+|this\s+)?(?:a\s+bit\s+)?"
    r"(?:simpler|more\s+simply|simply|in\s+simple\s+(?:terms|words))\b[^?]{0,30}[?.!]*\s*$|"
    r"\b(?:thoda\s+)?(?:simple|aasan|asaan)\s+(?:mein|me|main|bhasha\s+mein)\s+samjha\w*|"
    r"^\W*(?:yaar\s+|bhai\s+)?samjhao\W*$|"
    # "explain like im 5" with no "that" printed a desk decision (a first-time user, round 24)
    r"^\W*(?:pls\s+|please\s+|can\s+you\s+)?explain\s+(?:it\s+|that\s+|this\s+)?(?:to\s+me\s+)?like\s+"
    r"i'?m\s+(?:5|five|a\s+kid|new|a\s+beginner|dumb|stupid)\W*(?:pls|please)?\W*$|\beli5\b|"
    # "can you say that again but simpler, I'm a total beginner" (a round-23 re-ask)
    r"^\W*(?:can\s+you\s+|could\s+you\s+|pls\s+|please\s+)?(?:say|explain|put)\s+(?:that|it|this)"
    r"\s+(?:again\s+)?(?:but\s+)?(?:simpler|more\s+simply|in\s+simpler\s+(?:words|terms)|"
    r"in\s+plain\s+(?:english|words))\b[^?]{0,60}[?.!]*\s*$",
    re.I)
"""Asking what the last answer's figures mean, which named nothing to look up (row 637)."""


def no_name_buy_answer(capital: float | None, earlier: str | None = None
                       ) -> tuple[list[str], list[Source], dict[str, Any]]:
    name = earlier.removesuffix("USDT") if earlier else "BTC"
    held = (f" With the ${capital:,.0f} you mentioned: \"how much could I lose with "
            f"${capital:,.0f} in {name} this week\"." if capital else "")
    about = (f" On {name}, the name you were just asking about: \"how much could I lose on "
             f"{name} this week\" or \"what would make {name} a bad buy now\"."
             if earlier else "")
    return ([
        "Bottom line: this console makes no buy call — it does not know your situation — but it "
        "will measure any choice you are weighing before you make it." + about + held,
        "Name what you are considering and ask what it would mean: \"how much could I lose on BTC "
        "this week\", \"is TSLA riskier than NVDA\", \"what does it cost to buy $500 of ETH\".",
        "If you have nothing in mind yet, say how much you have — \"I have $1,000, what should I "
        "do\" — and it shows what that sum went through in a year in three broad markets.",
    ], [], {"no_name_buy": True})


def thanks_answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    return (["Bottom line: glad it helped. Ask anything else about a market Bitget lists — "
             "a price, a risk, a cost, or a view you want tested."], [], {"thanks": True})


def that_number_answer(previous: str, *, answered: bool = True,
                       ) -> tuple[list[str], list[Source], dict[str, Any]]:
    from argus.lui.research import research_symbols

    if not answered:
        # "what does that mean" after a refusal was told "the last answer, on TSLA, gives each
        # figure..." when that answer had no figures (a first-time user, round 22)
        return ([
            f"Bottom line: the last question, “{previous[:120]}”, was not answered "
            f"— it was not read as a question this console can work out, so there are no "
            f"figures in it to explain.",
            "Try it in different words, or one of these: \"where should my stop go on TSLA\", "
            "\"what is liquidation\", \"how much could I lose on BTC in a bad week\".",
        ], [], {"that_number": None})
    named = research_symbols(previous)[0]
    subject = named[0].removesuffix("USDT") if named else "it"
    return ([
        f"Bottom line: the last answer, on {subject}, gives each figure with what it was "
        f"computed from; ask for the one you mean by its word and it is explained in plain "
        f"language.",
        "For example: \"what does 24h range mean\", \"what is funding\", \"what is the "
        "spread\", \"what is beta\", \"what is volatility\", \"what is a drawdown\".",
        f"Or ask it simpler: \"explain that simpler\" restates the last answer about {subject} "
        f"in plain words.",
    ], [], {"that_number": subject})

def ask_answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    lines = [
        "Bottom line: ask it anything about Bitget's markets in plain words — the questions below "
        "each work as written, and you can change the symbol or the number.",
        "A market: \"what is NVDA doing today\", \"compare gold and bitcoin this year\", "
        "\"how volatile is bitcoin\".",
        "A position: \"how much would I lose if the Nasdaq fell 10% and I hold $5k of TSLA\", "
        "\"size a trade: $10k account, 1% risk, stop 4% below entry\".",
        "A company: \"when does TSLA report and how big is the move usually\", "
        "\"is NVDA expensive versus MSFT\".",
        "Starting out: \"I have $1,000, where do I start\", \"what happens if I lose money\". "
        "Tell it your horizon or the most you can bear losing and it remembers.",
        "About this desk: \"how is your desk doing\", \"what did it get wrong\", \"how does ARGUS "
        "compare to Nautilus Trader\".",
    ]
    explains(*lines)
    return lines, [Source("computation", "argus.lui.intro", "example questions")], {}


def answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    from argus.eval.standing import REGISTER, State

    counts = {s: sum(1 for cap in REGISTER if cap.state is s) for s in State}
    total = len(REGISTER)
    lines = [
        "Bottom line: ARGUS is a research console for Bitget markets — tokenized US stocks, "
        "crypto, indices and commodities. Ask in plain words and it works the answer out from "
        "live data, showing where every number came from.",
        "Who it is for: anyone holding or weighing Bitget's tokenized US stocks or crypto, new "
        "to trading or not, who wants answers that show their work. It is not a signal service "
        "and not financial advice, and it never places an order for you.",
        "What to ask, for example: \"what is NVDA doing today\", \"how much would I lose if the "
        "Nasdaq fell 10% and I hold $5k of TSLA\", \"size a trade: $10k account, 1% risk, stop "
        "4% below entry\", \"compare gold and bitcoin this year\", \"when does TSLA report and "
        "how big is the move usually\", or \"I have $1,000, where do I start\".",
        f"How far to trust it: every capability is measured against a named rival on the same "
        f"input — of {total}, {counts[State.OWNED]} win, {counts[State.TIED]} tie and "
        f"{counts[State.LOST]} lose, all listed on /proof — and what it got wrong stays on "
        f"/wrong. Ask \"how is your desk doing\" for its own paper record; the team's separate "
        f"trading agent, which places orders on Bitget's demo venue, is on /agent.",
    ]
    explains(*lines)
    return lines, [Source("computation", "argus.eval.standing register",
                          "the capability counts, read now")], {
        "standing": {s.value: n for s, n in counts.items()}, "total": total}



STANDING_Q = re.compile(
    r"\bhow\s+many\s+(?:of\s+(?:your|the|its|argus'?s?)\s+)?(?:capabilities|features|things|"
    r"comparisons|rivals|competitors)\b|"
    r"\bwhat\s+(?:have\s+you|has\s+argus|did\s+you)\s+(?:beaten|beat|won|lost|tested)\b|"
    r"\bhow\s+(?:do\s+you|does\s+argus)\s+(?:compare|stack\s+up)\s+(?:to|with|against)\s+"
    r"(?:rivals|competitors|others|the\s+competition)\b|\bwhere\s+(?:do\s+you|does\s+argus)\s+"
    r"(?:win|lose)\b", re.I)
"""The scoreboard asked as a whole: "how many capabilities have you tested against rivals" was read
as a rival named "tested" and answered with one row (a hostile review, round 11)."""


def standing_answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    from argus.eval.standing import REGISTER, State

    counts = {s: sum(1 for cap in REGISTER if cap.state is s) for s in State}
    lost = [cap.name for cap in REGISTER if cap.state is State.LOST]
    lines = [
        f"Bottom line: {len(REGISTER)} capabilities have each been run against a named rival "
        f"on the same input: {counts[State.OWNED]} won (OWNED), {counts[State.TIED]} tied, "
        f"{counts[State.LOST]} lost" + (f", {counts[State.IMPLEMENTED]} built but not yet compared"
                                        if counts[State.IMPLEMENTED] else "") + ".",
        "OWNED needs all thirteen checks to pass — the rival run on the same input, costs "
        "included, out-of-sample, an ablation, an adversarial test and more; a loss is published "
        "the moment it is found.",
        "Lost, by name: " + "; ".join(lost) + "." if lost else "Nothing is graded lost.",
        "Every row, with its rival and the measurement, is on /proof; ask \"how does ARGUS "
        "compare to Nautilus Trader\" (or any rival) for the rows that name it.",
    ]
    explains(*lines[1:2], *lines[3:])
    return lines, [Source("computation", "argus.eval.standing register",
                          "every capability's state, read now")], {
        "standing": {s.value: n for s, n in counts.items()}, "total": len(REGISTER)}


TESTS_Q = re.compile(
    r"\bhow\s+many\s+tests\b|\b(?:are|do)\s+(?:all\s+)?(?:the\s+|your\s+|its\s+)?tests\s+"
    r"(?:all\s+)?(?:pass|passing|green)\b|\btest\s+(?:suite|count|coverage)\b", re.I)
"""The test suite asked about: "how many tests does ARGUS have and are they all passing" was
answered with the hash-chain status (a hostile review, round 16, 2026-10-01)."""


def tests_answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    import json

    from argus.truth.paths import DATA_DIR

    quoted = ""
    try:
        report = json.loads((DATA_DIR / "doc_claims.json").read_text(encoding="utf-8"))
        for found in report.get("findings", []):
            if found.get("claim") == "tests_passing" and found.get("quoted"):
                quoted = str(found["quoted"][0])
                break
    except (OSError, ValueError):
        quoted = ""
    count = f"{quoted} tests are collected by pytest" if quoted else "the suite is published"
    lines = [
        f"Bottom line: the repository's README states that {count}, with lint and strict type "
        "checks clean. That is a published figure, not something this console ran.",
        "This live console does not run the tests, so it cannot tell you they pass right now. "
        "To check yourself, clone the repository and run pytest in the argus folder; the "
        "last full run's result is written in the README.",
    ]
    explains(*lines)
    return lines, [Source("artefact", "README.md", "the published test count, as quoted")], {
        "tests_quoted": quoted or None}


VS_Q = re.compile(
    r"\b(?:what\s+(?:can|does)\s+(?:you|argus|this|it)\s+do\s+(?:that|which)\s+(?:chat\s*gpt|gpt|"
    r"claude|gemini|a\s+chatbot|an?\s+(?:ai|llm))\s+(?:can(?:'?t|not)|doesn'?t|does\s+not)|"
    r"how\s+(?:are|is)\s+(?:you|this|argus|it)\s+(?:different|better)\s+(?:from|than)\s+"
    r"(?:chat\s*gpt|gpt|claude|gemini|a\s+chatbot|just\s+asking\s+an?\s+(?:ai|llm))|"
    r"why\s+(?:use|not\s+just\s+use)\s+(?:you|this|argus)\s+(?:instead\s+of|over|rather\s+than)\s+"
    r"(?:chat\s*gpt|gpt|claude|gemini|a\s+chatbot))\b", re.I)
"""The product against a general chatbot: "what can you do that ChatGPT cannot" was declined with
"that has nothing to refer to" (round 17)."""


def vs_answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    lines = [
        "Bottom line: it does not write the numbers. The language model only reads your question "
        "and picks an engine; each figure is computed at that moment from Bitget's live data or "
        "SEC filings, and the receipt under the answer names the source.",
        "What that buys you: a price, a cost, a loss range or a position size you can trace to "
        "its inputs; \"I don't know\" and a stated reason when the data is missing, not a "
        "plausible guess; and a record of the paper desk's decisions that is hash-chained, so a "
        "past row cannot be quietly edited.",
        "What a general chatbot does better: write, code and talk about anything else. This "
        "console only answers questions about Bitget's markets, and it is not advice.",
    ]
    explains(*lines)
    return lines, [Source("computation", "argus.lui.arbiter", "which reader decides a question")
                   ], {}


HOW_Q = re.compile(
    r"^\s*(?:(?:and|so|but)\s+)?(?:how\s+do\s+you\s+know\s+(?:all\s+)?(?:this|that|these|it)|"
    r"where\s+(?:do|does)\s+(?:you|this|the\s+data|your\s+data|the\s+numbers)\s+(?:get|come\s+from)"
    r"(?:\s+(?:it|this|that|your\s+data|the\s+data|from))?|what\s+(?:model|llm|ai|language\s+model)"
    r"\s+(?:do\s+you|does\s+(?:this|argus|it))\s+(?:run\s+on|use)|which\s+(?:model|llm|ai)\s+"
    r"(?:is\s+this|do\s+you\s+use|are\s+you)|are\s+you\s+(?:an?\s+)?(?:ai|chatgpt|gpt|llm|bot|"
    r"robot))\s*[?.!]*\s*$", re.I)
"""How the console knows what it says, and which model it runs on: "how do you know all this" and
"what model do you run on" were answered with decision 879's evidence (round 11, 2026-09-30)."""


def how_answer(text: str = "") -> tuple[list[str], list[Source], dict[str, Any]]:
    lines = [
        "Bottom line: nothing here is remembered or made up — every figure is fetched or computed "
        "at the moment you ask, from the sources listed in the receipt under each answer.",
        "The sources: Bitget's live market data and its bitget-signal Skills and data service; "
        "SEC filings (EDGAR, XBRL) for companies; official calendars for CPI and the Fed; and a "
        "few public feeds (options quotes, news, prediction markets), each named where it is used.",
        "The language model: Qwen 3.8 Max, through Bitget's hackathon endpoint, reads your "
        "question and picks which engine answers it; it never writes a number. When it is "
        "unavailable, a small classifier trained here and the console's own patterns read the "
        "question instead. The model reads up to 40 questions an hour from one network address, "
        "counted on each server instance; past that the classifier and patterns answer, in "
        "English. Each line of an answer is marked with where it came from.",
    ]
    if re.search(r"\b(?:model|llm|ai|gpt|chatgpt|bot|robot)\b", text, re.I):
        # Asked which model, the model leads.
        lines = ["Bottom line: " + lines[2].removeprefix("The language model: ")[:1].upper()
                 + lines[2].removeprefix("The language model: ")[1:], *lines[:2]]
        lines[1] = lines[1].replace("Bottom line: ", "", 1)
        lines[1] = lines[1][:1].upper() + lines[1][1:]
    explains(*lines)
    return lines, [Source("computation", "argus.lui.arbiter", "which reader decides a question")
                   ], {}

_LISTED = (r"(?:(?:main|key|core|all)\s+)?(?:capabilities|features|functions|tools|skills|"
           r"data\s+sources)")
"""Bare "sources" is left out: "what are your sources" after an answer asks for that answer's."""
CAPABILITIES_Q = re.compile(
    rf"^\s*(?:(?:what\s+can\s+(?:you|argus|it)\s+do|who\s+are\s+you|what\s+is\s+{_SUBJECT})"
    rf"\s*[?.!,]+\s*)?(?:(?:please\s+)?(?:list|name|show(?:\s+me)?|tell\s+me|give\s+me)\s+"
    rf"(?:all\s+)?(?:of\s+)?(?:your|the|its|argus'?s?)?\s*{_LISTED}"
    rf"|what\s+(?:are|is)\s+(?:your|the|its|argus'?s?)\s+{_LISTED})"
    rf"(?:\s*(?:,|and|&)\s*(?:your\s+|the\s+)?{_LISTED})?\s*[?.!]*\s*$", re.I)
"""The product asked for its feature list: "What can you do? List your main capabilities and data
sources." was answered with decision 879's evidence (a judge, round 13, 2026-09-30) — the second
sentence stopped the intro pattern from matching, and the question fell through to the last
decision."""

SOURCES_ONLY_Q = re.compile(
    r"^\s*(?:what|which)\s+(?:data\s+)?sources\s+(?:do\s+you|does\s+(?:argus|it|this))\s+"
    r"(?:use|have|read|rely\s+on|pull\s+from)\s*[?.!]*\s*$", re.I)
"""Sources alone: answered by :func:`how_answer`, which names them and says the model writes no
number."""


def capabilities_answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    """What the console does, each line a thing to ask, then where the data comes from.

    Deliberately concrete: a judge reading this should be able to copy any example into the box
    and get that capability, so every example is one the router already answers."""
    from argus.eval.standing import REGISTER, State

    counts = {s: sum(1 for cap in REGISTER if cap.state is s) for s in State}
    lines = [
        "Bottom line: ARGUS turns a plain-words question about a Bitget market into a sourced "
        "answer; its main capabilities are below, each with a question to try.",
        "Full research task: \"research NVDA\" — price, fundamentals from SEC filings, news, "
        "options and prediction-market odds, factor exposures and the next report, in one "
        "report (also at /research).",
        "Thesis tester: \"I think NVDA goes higher because cloud capex keeps rising and it is "
        "cheap against its sector\" — each reason tested separately against data; drop or change "
        "a reason and it re-tests.",
        "Risk and sizing: \"how much would I lose if the Nasdaq fell 10% and I hold $5k of TSLA\", "
        "\"size a trade: $10k account, 1% risk, stop 4% below entry\".",
        "Events: \"when does TSLA report and how big is the move usually\" — the measured move "
        "after each past results release.",
        "Compare and explain: \"compare gold and bitcoin this year\", \"what is funding rate\".",
        "Personal memory: say your loss limit, horizon and holdings once (\"I can't lose more "
        "than 10%, horizon a few weeks\") and later answers size to them.",
        "Bitget Skills: \"what does bitget-signal say about BTC\" — the five bitget-signal Skills "
        "(macro, market intel, news, sentiment, technicals), each marked live or read earlier.",
        "Data sources: Bitget's live market API, bitget-signal and Bitget's MCP data service "
        "(US fundamentals, 13F holdings, analyst estimates, earnings calendar, corporate "
        "actions); SEC EDGAR filings; FRED for rates; official CPI and Fed calendars; and public "
        "news and prediction-market feeds — each named in the receipt under an answer.",
        f"How far to trust it: {len(REGISTER)} capabilities measured against named rivals on the "
        f"same input — {counts[State.OWNED]} win, {counts[State.TIED]} tie, "
        f"{counts[State.LOST]} lose (/proof). It never places an order for you.",
    ]
    explains(*lines)
    return lines, [Source("computation", "argus.eval.standing register",
                          "the capability counts, read now")], {
        "standing": {s.value: n for s, n in counts.items()}, "total": len(REGISTER)}


trace_module(globals())
