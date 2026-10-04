"""What each analyst was handed under channel routing, against kind routing (build-list 5.2).

Until 2026-10-05 the desk routed evidence by its channel (``source``), and channels stood in for
types. This counts, over every decision whose evidence the runner kept (``data/desk_notes.jsonl``),
what each analyst received under the old rule and would receive under the new one
(`agents/selection.reads`). It is a count of the record, not a performance claim: whether the
analysts' readings get better is a question for the decisions made after the change.

The kept lines carry no kind (it did not exist), so each line's kind is read from its id prefix,
which every producer sets and which names the producer: ``skill-<tool>-`` and ``mirror-<tool>-``
by `market/skills.TOOL_KINDS`, ``edgar-`` by its channel (an event filing on ``sec-edgar``, a
periodic report on ``filing``), and the rest one prefix to one producer. A prefix not in the table
is counted as unknown, never guessed.

    python -m argus.eval.evidence_routing      # writes data/evidence_routing.json
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Final

from argus.truth import artefact
from argus.truth.evidence import Kind
from argus.truth.paths import DATA_DIR

NOTES: Final = DATA_DIR / "desk_notes.jsonl"
OUT: Final = DATA_DIR / "evidence_routing.json"
_LINE: Final = re.compile(r"^\[(?P<id>[^\]]+)\] \((?P<source>[^,]+), credibility")

OLD: Final[dict[str, frozenset[str]]] = {
    "event": frozenset({"sec-edgar", "news", "macro"}),
    "sentiment": frozenset({"social"}),
    "earnings": frozenset({"filing", "transcript"}),
}
"""The channel map as it stood before 2026-10-05 (`agents/selection.py`, then ``SOURCES``)."""

_PREFIX: Final[dict[str, Kind]] = {
    "mkt": Kind.QUOTE, "feeds": Kind.COVERAGE, "twitter": Kind.SOCIAL, "reddit": Kind.SOCIAL,
    "rss": Kind.NEWS, "halt": Kind.NEWS, "form4": Kind.FILING, "xbrl": Kind.FUNDAMENTAL,
    "consensus": Kind.ESTIMATE, "ust": Kind.MACRO, "vix": Kind.MACRO, "fng": Kind.SENTIMENT_INDEX,
    "bitget": Kind.SENTIMENT_INDEX, "finra": Kind.POSITIONING, "darkpool": Kind.POSITIONING,
    "options": Kind.DERIVATIVES,
}


def kind_from_line(item_id: str, source: str) -> Kind | None:
    """A kept line's kind, from the producer its id names; ``None`` for an unknown producer."""
    from argus.market.skills import TOOL_KINDS

    head = item_id.split("-", 1)[0]
    if head in ("skill", "mirror"):
        rest = item_id.split("-", 1)[1] if "-" in item_id else ""
        tool = next((t for t in sorted(TOOL_KINDS, key=len, reverse=True)
                     if rest.startswith(t + "-")), None)
        return TOOL_KINDS[tool] if tool else None
    if head == "edgar":
        return Kind.FILING if source == "sec-edgar" else Kind.REPORT
    return _PREFIX.get(head)


def _lines(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str) and raw:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    return []


def count(path: Path = NOTES) -> dict[str, Any]:
    from argus.agents.selection import KINDS

    decisions = 0
    unknown: Counter[str] = Counter()
    per: dict[str, dict[str, Counter[str]]] = {
        a: {"old": Counter(), "new": Counter(), "lost": Counter(), "gained": Counter()}
        for a in KINDS}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        lines = _lines(json.loads(line).get("evidence"))
        if not lines:
            continue
        decisions += 1
        for text in lines:
            match = _LINE.match(text)
            if match is None:
                continue
            source, item_id = match.group("source"), match.group("id")
            kind = kind_from_line(item_id, source)
            if kind is None:
                unknown[item_id.split("-", 1)[0]] += 1
            label = kind.value if kind else "unknown"
            for analyst, kinds in KINDS.items():
                was = source in OLD[analyst]
                now = kind is not None and kind in kinds
                if was:
                    per[analyst]["old"][label] += 1
                if now:
                    per[analyst]["new"][label] += 1
                if was and not now:
                    per[analyst]["lost"][label] += 1
                if now and not was:
                    per[analyst]["gained"][label] += 1
    analysts = {}
    for analyst, c in per.items():
        old_total, wrong = sum(c["old"].values()), sum(c["lost"].values())
        analysts[analyst] = {
            "handed_before": old_total, "handed_now": sum(c["new"].values()),
            "before_not_its_kind": wrong,
            "before_not_its_kind_share": round(wrong / old_total, 4) if old_total else None,
            "before_by_kind": dict(c["old"].most_common()),
            "no_longer_handed": dict(c["lost"].most_common()),
            "newly_handed": dict(c["gained"].most_common()),
        }
    return {"decisions": decisions, "analysts": analysts,
            "unknown_producers": dict(unknown.most_common())}


def run(*, out: Path = OUT) -> dict[str, Any]:
    blob = count()
    artefact.write(out, blob)
    return blob


def main() -> int:  # pragma: no cover - CLI
    print(json.dumps(run(), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
