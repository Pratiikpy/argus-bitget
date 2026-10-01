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
    rf"^\s*(?:(?:hi|hello|hey)\W+)?(?:so\s+)?(?:what\s+(?:is|'s)\s+{_SUBJECT}|what\s+does\s+"
    rf"{_SUBJECT}\s+do|who\s+(?:is|'s)\s+(?:{_SUBJECT}|it)\s+for|who\s+(?:made|built)\s+{_SUBJECT}|"
    rf"what\s+can\s+(?:you|argus|this(?:\s+\w+)?)\s+do(?:\s+for\s+me)?|what\s+(?:can|should)\s+i\s+"
    rf"ask(?:\s+(?:you|it|argus|here))?|how\s+do\s+i\s+use\s+{_SUBJECT}|what\s+am\s+i\s+looking\s+at|"
    rf"help|getting\s+started|where\s+do\s+i\s+start|who\s+are\s+you)"
    rf"(?:\s*(?:,|and)\s*(?:who\s+(?:is|'s)\s+(?:{_SUBJECT}|it)\s+for|what\s+(?:does\s+{_SUBJECT}\s+do|"
    rf"can\s+(?:you|it)\s+do)))?\s*[?.!]*\s*$",
    re.I)
"""A question about the product itself, asked whole: "what is this site and who is it for"."""


def answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    from argus.eval.standing import REGISTER, State

    counts = {s: sum(1 for cap in REGISTER if cap.state is s) for s in State}
    total = len(REGISTER)
    lines = [
        "Bottom line: ARGUS is a research console for Bitget markets — tokenized US stocks, "
        "crypto, indices and commodities. Ask in plain words and it works the answer out from "
        "live data, showing where every number came from.",
        "Who it is for: a trader holding Bitget's tokenized US stocks who wants a second desk "
        "that shows its work. It is not a signal service and not financial advice, and it never "
        "places an order for you.",
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
        "question instead. The model reads up to 40 questions an hour from one network address; "
        "past that the classifier and patterns answer, in English. Each line of an answer is "
        "marked with where it came from.",
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
