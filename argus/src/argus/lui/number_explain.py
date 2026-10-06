"""One figure from the last answer, said in plain words.

"why is that? explain the first number in simpler words" re-printed the whole previous answer
prefixed "Said more simply:", with a thirteen-source receipt and no word on which number or which
answer it meant (round 44 visual audit, major 2). A follow-up that points at a figure is answered
about that figure: which answer it came from, the number, and what it means, from the words around
it in that answer. Nothing is recomputed, so nothing new can disagree with the answer it explains.
"""

from __future__ import annotations

import re
from typing import Final

from argus.lui.trace import trace_module

ASKED: Final = re.compile(
    r"\b(?:the\s+)?(?P<which>first|second|third|last|1st|2nd|3rd|that|this|main|big|top)\s+"
    r"(?:number|figure|percent(?:age)?|stat(?:istic)?|value)\b"
    r"|\bwhat\s+(?:does|do)\s+(?:the\s+)?(?P<pct>-?\d+(?:\.\d+)?\s*%|\$?\d[\d,]*(?:\.\d+)?)\s+"
    r"mean\b", re.I)
_ORDINAL: Final = {"first": 0, "1st": 0, "that": 0, "this": 0, "main": 0, "big": 0, "top": 0,
                   "second": 1, "2nd": 1, "third": 2, "3rd": 2, "last": -1}
_FIGURE: Final = re.compile(
    r"(?<![\w.])[-+\u2212]?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|bps|bp|x\b|hours?\b|days?\b|years?\b|bn\b|m\b|k\b)?")


def _figures(line: str) -> list[re.Match[str]]:
    """The figures in ``line``, leaving out dates and the numbering of a list."""
    out = []
    for m in _FIGURE.finditer(line):
        raw = m.group(0).strip()
        before = line[max(0, m.start() - 4):m.start()]
        after = line[m.end():m.end() + 4]
        if re.fullmatch(r"(?:19|20)\d\d", raw) or re.match(r"-\d\d", after) or re.search(
                r"\d-$", before) or re.match(r"\s?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|"
                                            r"dec)\w*", after, re.I) or re.search(
                r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s$", before, re.I):
            continue  # a date
        if m.start() == 0 and re.match(r"\.\s", line[m.end():m.end() + 2]):
            continue  # "1. " numbering
        out.append(m)
    return out


def _meaning(figure: str, around: str) -> str:
    """What a figure means, from the kind of figure it is and the words beside it."""
    lowered = around.lower()
    number = figure.strip()
    if "%" in number and re.search(r"\ba\s+year\b|\bannual", lowered) and re.search(
            r"swing|swung|volatil|vol\b", lowered):
        return (f"{number} a year is how much the price typically moves up or down over a year: "
                f"in an ordinary year expect it to end roughly {number.lstrip('+-')} above or "
                f"below where it started, and in about one year in three by more than that. A "
                f"bigger number means a bumpier ride, not a better or worse investment")
    if "%" in number and re.search(r"value\s+at\s+risk|\bvar\b|on\s+5%\s+of", lowered):
        return (f"{number} is the loss line for a bad day: on about one day in twenty the book "
                f"lost more than {number.lstrip('+-')} of its value, so expect a day like that "
                f"roughly once a month")
    if re.search(r"\bdays?\b", number) and re.search(r"report|earnings|results", lowered):
        return (f"{number} is how long until the company publishes its next quarterly results; "
                f"prices often jump on that day, so a trade held through it carries extra risk")
    if "%" in number and re.search(r"worst|fall|drawdown|lost|loss|down", lowered):
        return (f"{number} is a fall that already happened: the price dropped that much at the "
                f"worst point, which is a guide to what holding it can feel like, not a forecast")
    if "%" in number and re.search(r"funding", lowered):
        return (f"{number} is the small fee longs and shorts pay each other on a perpetual "
                f"contract every few hours; positive means longs pay")
    if "%" in number and re.search(r"over\s+(?:24\s*h|the\s+last|the\s+past)|today|this\s+week",
                                   lowered):
        return f"{number} is how much the price has changed over the period named in that sentence"
    window = re.fullmatch(r"(\d+)\s*(hours?|days?|years?)", number)
    if window is not None and re.search(rf"\b(?:worst|best|biggest)\s+{re.escape(number)}",
                                        lowered):
        return (f"{number} is the length of the stretch measured: the figure beside it is the "
                f"biggest move in any {number} in a row, start to end")
    if window is not None and re.search(rf"(?:over|last|past|in|worst)\s+(?:its\s+|"
                                        rf"the\s+)?(?:last\s+|"
                                        rf"past\s+)?{re.escape(number)}", lowered):
        return (f"{number} is the stretch of history the other figures in that sentence were "
                f"measured over — the most recent {number} of prices; a longer window would "
                f"give a steadier but slower-moving number")
    if re.search(r"\bbeta\b", lowered):
        return (f"{number} compares its moves with the market's: 1.2 means when the market moves "
                f"1% it has tended to move about 1.2%")
    if "$" in number:
        return f"{number} is an amount of money; the sentence it sits in says what it pays for"
    return f"{number} is the figure in this sentence"


def lines(text: str, prior: list[str], previous_lines: list[str]) -> list[str] | None:
    """The figure the follow-up points at, from the previous answer's lines, or None."""
    asked = ASKED.search(text)
    if asked is None or not prior or not previous_lines:
        return None
    body = [x for x in previous_lines if not x.startswith(("Data:", "Sources reached", "Terms:",
                                                            "Read as", "Prices:", "Quoted"))]
    if not body:
        return None
    lead = body[0].removeprefix("Bottom line: ")
    if re.match(r"your\s+question\s+has\s+\d+\s+parts", lead, re.I) and len(body) > 1:
        # a multi-part answer leads with its count of parts; the first figure is in part 1
        lead = re.sub(r"^\d+\.\s*", "", body[1])
    if asked.group("pct"):
        wanted = asked.group("pct").replace(" ", "")
        holder = next((x for x in body if wanted in x.replace(" ", "")), None)
        if holder is None:
            return [f"Bottom line: {wanted} is not in my last answer (to “{prior[-1]}”), "
                    "so I cannot say what it meant there; quote the sentence and I will."]
        found = next((m for m in _figures(holder) if wanted in m.group(0).replace(" ", "")), None)
        figure_line = holder
    else:
        figures = _figures(lead)
        index = _ORDINAL[asked.group("which").lower()]
        if not figures or (index >= 0 and index >= len(figures)):
            return None
        found = figures[index]
        figure_line = lead
    if found is None:
        return None
    figure = found.group(0).strip()
    sentence = re.split(r"(?<=[.;])\s+", figure_line[max(0, found.start() - 160):
                                                     found.end() + 160])
    around = next((s for s in sentence if figure in s), figure_line)
    return [f"Bottom line: {_meaning(figure, around)}.",
            f"It comes from my last answer, to “{prior[-1]}”, where it reads: "
            f"“{around.strip()}”",
            "Any other number or word, point at it the same way (\"what does the second number "
            "mean\")."]


__all__ = ["ASKED", "lines"]

trace_module(globals())
