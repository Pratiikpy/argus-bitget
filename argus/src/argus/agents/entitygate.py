"""Entity grounding — every instrument a thesis names must appear in the evidence it cites.

`agents/grounding.py` already enforces this for **numbers**: a figure in a thesis resolves to the
fact that produced it, or it is reported as unattributable. This module is the missing sibling for
**instruments**. A model writing about NVDA can name TSLA in the same sentence, and until now
nothing checked whether TSLA appeared anywhere in what the model was actually reading.

**The mechanism is taken from `HKUSTDial/DeepEar` (MIT), read at source.** Its `fin_agent.py`
lets the analyst model emit an ``impact_tickers`` list and then *drops any ticker that does not
also appear in the signal's title, summary or sources* — its own comment calls this
"sanitize tickers to avoid low-quality hallucinated associations". That is the whole idea, and it
is a good one: the model may propose, but an instrument it names has to be traceable to text it
was given.

**Two things were read and deliberately not copied.** DeepEar's pipeline is credited elsewhere
with novelty, crowding and divergence-versus-price scoring; searching its source for any of the
three returns nothing, so that framing was not taken. And its filter also accepts a ticker that
appears only in the *researcher agent's own structured output* — which is another model's
assertion, not a source. Accepting it would make the gate pass on a two-model agreement rather
than on evidence, so this implementation requires the evidence itself.

**What "appears in" means here, and why it is not a substring test.** ``COIN`` occurs inside
``COINBASE``, inside ``BITCOIN``, and inside the word ``coincide``. A naive ``in`` check passes all
three and the gate becomes decorative. Matching is on token boundaries that work in both languages
this desk reads: Latin-alphabet boundaries for English, and no word-boundary assumption for
Chinese, where ``关于NVDA的看法`` has no space around the ticker. The same boundary bug was already
found twice in `lui/question.py`, and it is not being written a third time.

**This gate refuses; it does not repair.** An unsupported instrument is reported, never quietly
rewritten to a supported one — a gate that guesses what the model meant has invented a different
claim and hidden the fact that it did.

    python -m argus.agents.entitygate
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from argus.lui.question import TRADED_SYMBOLS

_TICKER_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,6}(?:USDT)?)(?![A-Za-z0-9])")
"""Instrument-shaped tokens, bounded by explicit lookarounds rather than ``\\b``.

Han characters are word characters to Python's ``re``, so ``\\bNVDA\\b`` matches nothing in
``关于NVDA的看法`` — half this desk's corpus is Chinese and a boundary assumption imported from
English silently switches the gate off there. `lui/question.py` and `eval/ngrambench.py` each
carried this exact bug before it was found by running them."""

_NOT_INSTRUMENTS: frozenset[str] = frozenset({
    # Units, measures and finance words that are upper-case but name no instrument.
    "USD", "USDT", "PNL", "EPS", "ROI", "ROE", "YTD", "MTD", "QTD", "NAV", "CAGR", "IRR",
    "EBIT", "EBITDA", "GAAP", "IPO", "ETF", "CEO", "CFO", "COO", "SEC", "FED", "FOMC",
    "GDP", "CPI", "PPI", "PMI", "ATH", "ATL", "OI", "AI", "API", "UTC", "RTH", "ADR",
    "BPS", "VWAP", "TWAP", "MOC", "LOC", "TIF", "GTC", "IOC", "FOK", "OK", "US", "UK", "EU",
    # This system's own vocabulary.
    "ARGUS", "NO", "YES", "BUY", "SELL", "HOLD", "LONG", "SHORT", "TRADE",
})
"""Upper-case tokens that are words, units or verbs rather than instruments.

Listed rather than inferred so a reader can disagree with a specific entry. A token wrongly listed
here weakens the gate silently, which is why the list is deliberately short and contains no
company-like strings."""


class EntityGateError(ValueError):
    """The gate cannot be applied honestly. Raised rather than passing everything."""


@dataclass(frozen=True, slots=True)
class Mention:
    """One instrument the thesis names, and whether the evidence supports it."""

    symbol: str
    supported_by: tuple[str, ...]
    """Evidence ids whose text carries this instrument. Empty means unsupported."""

    tradeable: bool
    """Is it in this desk's universe? An instrument can be evidenced and still untradeable —
    those are different failures and are reported separately."""

    @property
    def supported(self) -> bool:
        return bool(self.supported_by)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "supported": self.supported,
            "supported_by": list(self.supported_by),
            "tradeable": self.tradeable,
        }


@dataclass(frozen=True, slots=True)
class EntityReport:
    """Every instrument the thesis named, and what the evidence says about each."""

    mentions: tuple[Mention, ...]

    @property
    def unsupported(self) -> tuple[Mention, ...]:
        """Named, but present in no cited evidence. **These are the hallucination candidates.**"""
        return tuple(m for m in self.mentions if not m.supported)

    @property
    def untradeable(self) -> tuple[Mention, ...]:
        """Evidenced, but outside the desk's universe — a real instrument we cannot act on."""
        return tuple(m for m in self.mentions if m.supported and not m.tradeable)

    @property
    def grounded(self) -> bool:
        """**No mention is unsupported.** Deliberately not "most are": a thesis that names one
        instrument out of nowhere is exactly the failure this exists to catch, and a ratio would
        let it pass whenever the model also named four real ones."""
        return not self.unsupported

    @property
    def rate(self) -> float:
        if not self.mentions:
            return 1.0
        return sum(1 for m in self.mentions if m.supported) / len(self.mentions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mentions": [m.as_dict() for m in self.mentions],
            "grounded": self.grounded,
            "supported_rate": round(self.rate, 4),
            "unsupported": [m.symbol for m in self.unsupported],
            "untradeable": [m.symbol for m in self.untradeable],
        }

    def render(self) -> list[str]:
        lines = [
            f"ENTITY GATE — {len(self.mentions)} instrument(s) named, "
            f"{len(self.unsupported)} unsupported"
        ]
        for mention in self.mentions:
            if mention.supported:
                where = ", ".join(mention.supported_by[:3])
                flag = "" if mention.tradeable else "  [not in this desk's universe]"
                lines.append(f"  ok        {mention.symbol:10} <- {where}{flag}")
            else:
                lines.append(
                    f"  UNSUPPORTED {mention.symbol:8} named by the model, "
                    f"present in no cited evidence"
                )
        if not self.grounded:
            lines.append(
                "  An instrument the evidence does not mention is the model's own invention. "
                "It is reported, never silently replaced with one the evidence does support."
            )
        return lines


def extract(text: str) -> tuple[str, ...]:
    """Instrument-shaped tokens in ``text``, in order of first appearance, de-duplicated.

    ``NVDAUSDT`` and ``NVDA`` collapse to the venue symbol, because they are the same instrument
    and reporting them separately would make one of them look unsupported whenever the evidence
    happened to use the other spelling.
    """
    seen: dict[str, None] = {}
    for match in _TICKER_PATTERN.finditer(text):
        token = match.group(1).upper()
        if token in _NOT_INSTRUMENTS:
            continue
        canonical = token if token.endswith("USDT") else f"{token}USDT"
        seen.setdefault(canonical if canonical in TRADED_SYMBOLS else token, None)
    return tuple(seen)


def _mentions_symbol(haystack: str, symbol: str) -> bool:
    """Does this text name this instrument, by either spelling?

    Checked against both the venue symbol and the bare ticker, because evidence is written by
    news sources that say "NVDA" and by the venue which says "NVDAUSDT".
    """
    spellings = {symbol, symbol.removesuffix("USDT")}
    found = {m.group(1).upper() for m in _TICKER_PATTERN.finditer(haystack)}
    return bool(spellings & found)


def check(
    thesis: str,
    *,
    evidence: Sequence[Any] = (),
    extra_text: Mapping[str, str] | None = None,
) -> EntityReport:
    """Resolve every instrument in ``thesis`` against the text the desk actually had.

    ``evidence`` is a sequence of :class:`~argus.truth.evidence.Evidence`; each contributes its
    ``claim`` text under its ``id``. ``extra_text`` carries any other source the model was shown —
    a filing excerpt, a headline block — keyed by an id a reader can follow.

    An empty evidence set does not make everything pass. With nothing to check against, every
    named instrument is unsupported, which is the honest reading: a thesis citing no sources has
    grounded nothing.
    """
    sources: dict[str, str] = {}
    for item in evidence:
        ident = str(getattr(item, "id", "") or "")
        claim = str(getattr(item, "claim", "") or "")
        if ident:
            sources[ident] = claim
    for ident, text in (extra_text or {}).items():
        sources[str(ident)] = str(text)

    mentions: list[Mention] = []
    for symbol in extract(thesis):
        backing = tuple(
            ident for ident, text in sources.items() if _mentions_symbol(text, symbol)
        )
        mentions.append(Mention(
            symbol=symbol,
            supported_by=backing,
            tradeable=symbol in TRADED_SYMBOLS,
        ))
    return EntityReport(tuple(mentions))


def supported_only(
    proposed: Iterable[str], *, evidence: Sequence[Any] = (),
) -> tuple[str, ...]:
    """The subset of ``proposed`` the evidence actually mentions — DeepEar's filter, directly.

    Provided for the case where a model returns a *list* of instruments rather than prose. The
    dropped ones are not returned here; callers that need to report them should use :func:`check`,
    which names each one and why it failed.
    """
    text = " ".join(str(getattr(e, "claim", "") or "") for e in evidence)
    return tuple(s for s in proposed if _mentions_symbol(text, s))


def main() -> int:  # pragma: no cover - CLI
    """A worked example, using the failure this gate exists to catch."""
    import sys
    from datetime import UTC, datetime

    from argus.truth.evidence import Evidence

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    at = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    evidence = [
        Evidence(id="e1", claim="NVDA raised datacentre guidance for the coming quarter",
                 source="filing", available_at=at, credibility=0.9),
        Evidence(id="e2", claim="关于NVDA的分析师目标价上调", source="news",
                 available_at=at, credibility=0.7),
    ]
    thesis = (
        "NVDA guidance implies upside, and TSLA should follow on shared supply-chain exposure; "
        "AAPL is unaffected."
    )
    report = check(thesis, evidence=evidence)
    print(f'thesis: "{thesis}"\n')
    for line in report.render():
        print(line)
    print(
        "\nTSLA and AAPL are the model's own associations. They may even be right — the gate "
        "does not judge that. It reports that nothing the desk read mentions them."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EntityGateError", "EntityReport", "Mention", "check", "extract", "main", "supported_only",
]
