"""What the desk knows, and when it became knowable. The type both halves of the system share.

**This lived in `agents/analysts.py` and that was a real coupling defect, found by measuring the
import graph rather than by reading the code.** The market layer *produces* evidence and the agent
layer *consumes* it, so a producer importing its consumer's module inverted the dependency — and
because `agents/analysts.py` imports `argus.llm.base` and `argus.llm.qwen`, the consequence was
concrete and worse than untidy:

    >>> import argus.market.evidence      # a pure data fetcher
    # ... also loads argus.llm.qwen

Fetching an SEC filing pulled in the model client. ARGUS's central architectural claim is that the
deterministic layers never touch the model — verified for `truth`, `cost`, `risk`, `decision` and
`backtest`, and quietly false for `market`, which is the layer that *gathers the facts the claim is
about*.

`truth` is the right home rather than a new shared package: this module already owns
`clocks.py`, and the two types answer the same question. `SessionState` says whether a price can be
discovered at an instant; `Evidence` says what was knowable at one. Point-in-time discipline is a
`truth` concern, and `available_at` is the field the whole discipline turns on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class Kind(StrEnum):
    """What a piece of evidence *is*, as distinct from the channel it arrived on (build-list 5.2).

    ``source`` names a channel, and channels were being used as types: the Bitget quote line and
    the feed-coverage note arrived as ``news``, every bitget-signal Skill but macro arrived as
    ``social``, a dark-pool summary as ``news``, FINRA short volume as ``macro``. Routing read the
    channel, so the sentiment analyst summarised RSI, MACD and news headlines as crowd mood, and the
    event analyst was handed option-chain and dark-pool statistics as events (measured over the
    desk's record in ``eval/evidence_routing.py``). The type is the thing routing needs.

    Taken from quant-mind (``LLMQuant/quant-mind``, MIT, ``quantmind/knowledge/_base.py``): typed
    knowledge items with an ``item_type`` and a typed source rather than a bare string. Its item
    types (paper, news, factor, earnings, thesis) are research-library types; these are the types
    of evidence a trading decision reads, so the list is ARGUS's own."""

    QUOTE = "quote"
    """A venue's price and volume snapshot."""
    TECHNICAL = "technical"
    """An indicator computed on price bars (RSI, MACD, ATR, support and resistance)."""
    DERIVATIVES = "derivatives"
    """Options, funding, open interest: what the derivatives market is pricing."""
    POSITIONING = "positioning"
    """Who holds what: long/short ratios, taker flow, short volume, dark-pool share."""
    SENTIMENT_INDEX = "sentiment_index"
    """An aggregate mood gauge (Fear & Greed, a vendor sentiment index, trending lists)."""
    SOCIAL = "social"
    """Posts written by people (X, Reddit)."""
    NEWS = "news"
    """Published articles and headlines, and venue notices such as a trading halt."""
    FILING = "filing"
    """A regulatory filing as filed (8-K, 10-Q, Form 4)."""
    FUNDAMENTAL = "fundamental"
    """Reported company figures (XBRL facts, earnings surprise)."""
    ESTIMATE = "estimate"
    """Analysts' expectations (consensus EPS and revenue)."""
    TRANSCRIPT = "transcript"
    """What management said on a call."""
    MACRO = "macro"
    """Economy-wide prints: rates, the curve, volatility indices, releases."""
    ONCHAIN = "onchain"
    """Blockchain state: TVL, gas."""
    COVERAGE = "coverage"
    """A note about the evidence itself (which feeds answered), not about the market."""


@dataclass(frozen=True, slots=True)
class Evidence:
    """One piece of evidence with the provenance the independence graph needs."""

    id: str
    claim: str
    source: str
    """The channel it arrived on. Recorded per decision and counted by `eval/sourceaudit.py`, so
    it is kept as it was; what the item *is* lives in :attr:`kind`."""
    available_at: datetime
    """When this became knowable — never when it was fetched.

    The difference is the whole point-in-time model: stamping ingestion time would make every
    decision look prescient by exactly the fetch latency.
    """

    credibility: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)
    """Structured fields behind ``claim``, when the source has them.

    ``claim`` is prose, and prose cannot refute a thesis: a desk that writes "these sales are
    pre-arranged" about a filing whose ``aff10b5One`` is 0 has contradicted its own evidence, and
    nothing in the rendered sentence makes that checkable. Sources that carry structured attributes
    (Form 4, XBRL) put them here so :mod:`argus.agents.claims` can settle such a claim against the
    record rather than against another model's opinion. Empty for prose-only sources, and a claim
    with no field behind it is reported as unexamined rather than as passing."""

    kind: Kind | None = None
    """What the item is (:class:`Kind`). Every live producer sets it; ``None`` is read through
    :func:`kind_of`, which falls back to the channel exactly as routing did before kinds existed,
    so an older fixture routes as it always did."""

    def render(self) -> str:
        return (
            f"[{self.id}] ({self.source}, credibility {self.credibility:.2f}, "
            f"available {self.available_at.isoformat()}) {self.claim}"
        )


_BY_CHANNEL: dict[str, Kind] = {
    "sec-edgar": Kind.FILING, "news": Kind.NEWS, "macro": Kind.MACRO, "social": Kind.SOCIAL,
    "filing": Kind.FUNDAMENTAL, "transcript": Kind.TRANSCRIPT,
}
"""The old channel-as-type reading, used only for an item that carries no kind."""


def kind_of(item: Evidence) -> Kind | None:
    """``item.kind``, or for an untyped item the type its channel used to stand for; ``None`` for
    a channel nothing reads."""
    return item.kind if item.kind is not None else _BY_CHANNEL.get(item.source)


__all__ = ["Evidence", "Kind", "kind_of"]
