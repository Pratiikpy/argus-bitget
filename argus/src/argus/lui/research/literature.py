"""What the published research says about a trading question: the papers themselves, found live.

ARGUS cited papers only in its own docstrings; a trader asking "is there any research on funding
rates predicting returns?" got an engine answer or a refusal, never the literature. The idea is
taken from alphaXiv's OpenResearch (MIT; ``src/commands/paper.rs:26-127``), which looks papers up
through keyless public APIs. OpenResearch resolves one known identifier to its metadata; here the
question is a topic, so the search is OpenAlex's ``/works?search=`` (keyless, about 250 million
works, https://docs.openalex.org) and the answer is the most relevant papers with how often each
has been cited, its venue, its link, and the opening of its own abstract.

What it does not do, and says so: it does not read the papers or state their findings as fact.
The abstract's opening is the authors' own claim. Preprint servers (arXiv, SSRN) are marked, since
a preprint has not been peer reviewed. Citation counts favour older papers, so the list is ordered
first by how much of the question each title carries, then by citations on a log scale.
"""

from __future__ import annotations

import math
import re
import urllib.parse
from typing import Any, Final

from argus.lui.trace import trace_module

OPENALEX: Final = "https://api.openalex.org/works"
ASKED: Final = re.compile(
    r"\bwhat\s+(?:does|do)\s+(?:the\s+)?(?:research|literature|academic\w*|studies|papers?|"
    r"science|evidence\s+from\s+studies)\s+say\b|\b(?:any|are\s+there(?:\s+any)?|find\s+(?:me\s+)?)"
    r"\s*(?:academic\s+|peer[\s-]reviewed\s+|published\s+)?(?:papers?|studies|research)\s+"
    r"(?:on|about|into|showing|that)\b|\bacademic\s+(?:evidence|research|papers?|literature)\b|"
    r"\bis\s+there\s+(?:academic\s+|published\s+)?(?:research|a\s+paper|a\s+study|literature)\b|"
    r"\bpeer[\s-]reviewed\b|\b(?:arxiv|ssrn)\s+papers?\b", re.I)
_CUE: Final = re.compile(
    r"^.*?\b(?:on|about|into|showing|that|say(?:s)?\s+about)\s+", re.I)
_PREPRINT: Final = ("arxiv", "ssrn", "biorxiv", "medrxiv", "research square", "preprints")


def topic(text: str) -> str:
    """The subject of the question, without the asking words."""
    rest = _CUE.sub("", text, count=1) if _CUE.search(text) else text
    rest = re.sub(r"\b(?:what\s+does\s+the\s+(?:research|literature)\s+say|any|academic|papers?|"
                  r"studies|research|peer[\s-]reviewed|published|please|is\s+there|evidence)\b",
                  " ", rest, flags=re.I)
    return " ".join(re.sub(r"[?!.,;:]+", " ", rest).split())[:160]


def _opening(inverted: dict[str, list[int]] | None, words: int = 40) -> str:
    """The first sentence or so of an abstract, rebuilt from OpenAlex's inverted index."""
    if not inverted:
        return ""
    placed = sorted((p, w) for w, spots in inverted.items() for p in spots)
    text = " ".join(w for _, w in placed[:words * 3])
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    cut = first.split()
    return " ".join(cut[:words]) + ("…" if len(cut) > words else "")


def _words(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z]{3,}", text.lower())
            if w not in ("the", "and", "for", "with", "from", "into", "are", "does")}


def search(subject: str, *, fetch: Any = None, limit: int = 5) -> list[dict[str, Any]]:
    """The works on ``subject`` that match it best, with their metadata.

    OpenAlex's relevance ranks a full-text match, which put a paper on heteroskedasticity first
    for "funding rates predicting crypto returns". Twenty-five candidates are read and ranked by
    how many of the question's words the title carries, then by citations (on a log scale, so a
    famous paper on the wrong subject does not win); a preprint and its published version are
    shown once, as the published one."""
    from argus.truth import http

    query = urllib.parse.urlencode({
        "search": subject, "per-page": "25",
        "filter": "type:article|preprint|review",
        "select": "display_name,publication_year,cited_by_count,doi,primary_location,"
                  "abstract_inverted_index,authorships",
    })
    payload = (fetch or http.fetch_json)(f"{OPENALEX}?{query}", timeout=12.0)
    found = []
    for work in (payload or {}).get("results") or []:
        source = ((work.get("primary_location") or {}).get("source") or {})
        venue = str(source.get("display_name") or "")
        authors = [str((a.get("author") or {}).get("display_name") or "")
                   for a in work.get("authorships") or []]
        found.append({
            "title": " ".join(str(work.get("display_name") or "").split()),
            "year": work.get("publication_year"),
            "cited": int(work.get("cited_by_count") or 0),
            "doi": work.get("doi"),
            "venue": venue,
            "preprint": any(p in venue.lower() for p in _PREPRINT),
            "authors": [a for a in authors if a][:3],
            "opening": _opening(work.get("abstract_inverted_index")),
        })
    wanted = _words(subject)
    seen: dict[str, dict[str, Any]] = {}
    for row in found:
        key = re.sub(r"[^a-z]", "", row["title"].lower())[:60]
        held = seen.get(key)
        if held is None or (held["preprint"] and not row["preprint"]):
            seen[key] = row
    ranked = sorted(seen.values(), key=lambda r: (len(wanted & _words(r["title"])),
                                                  math.log10(1 + r["cited"])), reverse=True)
    found_total = int(((payload or {}).get("meta") or {}).get("count") or 0)
    for row in ranked:
        row["total"] = found_total
    return ranked[:limit]


def lines(text: str, *, fetch: Any = None) -> list[str] | None:
    """The answer to a literature question, or None when the question is not one."""
    if not ASKED.search(text):
        return None
    subject = topic(text)
    if len(subject) < 4:
        return None
    try:
        works = search(subject, fetch=fetch)
    except Exception:
        return [f"Bottom line: the paper search (OpenAlex) did not answer just now, so no papers "
                f"on \"{subject}\" can be listed; nothing below stands in for them.",
                "Ask again in a minute, or ask the engines directly — for example \"has funding "
                "on BTC predicted its next day\" runs the test on Bitget's own data."]
    if not works:
        return [f"Bottom line: OpenAlex finds no papers matching \"{subject}\" — try the question "
                f"in the words a paper would use (\"funding rate\", \"perpetual futures\", "
                f"\"momentum\")."]
    total = works[0]["total"]
    out = [f"Bottom line: {total:,} published works match \"{subject}\" on OpenAlex; the "
           f"{len(works)} most relevant are below, each in its authors' own words — the console "
           f"lists them and does not vouch for their findings."]
    for row in works:
        who = ", ".join(row["authors"]) + (" et al." if len(row["authors"]) >= 3 else "")
        kind = "preprint, not peer reviewed" if row["preprint"] else (row["venue"] or "venue "
                                                                            "not given")
        link = row["doi"] or "no DOI"
        out.append(f"{row['title']} ({row['year']}; {who or 'authors not listed'}; {kind}; cited "
                   f"{row['cited']:,} times) — {link}"
                   + (f". Abstract opens: \"{row['opening']}\"" if row["opening"] else "."))
    out.append("A paper's result holds for its own data and period; to test the same effect on "
               "Bitget's own markets, ask the console directly — every engine runs on live data. "
               "Search: OpenAlex (openalex.org), keyless, read just now.")
    return out


__all__ = ["ASKED", "OPENALEX", "lines", "search", "topic"]

trace_module(globals())
