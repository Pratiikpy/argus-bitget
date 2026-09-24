# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/TauricResearch/TradingAgents
# Path:    tradingagents/agents/utils/rating.py
# Commit:  be952b8eccb49720509af544c6675233bc1f10d0 (2026-09-07)
# Licence: Apache License 2.0 — full text below. The repository's own LICENSE file carries the
#          unfilled Apache 2.0 template ("Copyright [yyyy] [name of copyright owner]", never
#          completed by the project) rather than a filled-in notice; attributed here to the
#          repository owner, TauricResearch, per the organisation GitHub records as the copyright
#          holder — the reasonable reading of an incomplete template, not an invented one.
#
# `parse_rating` is a dependency of the sibling vendored `tradingagents_memory.py`
# (`TradingMemoryLog.store_decision` calls it to tag each entry) — vendored alongside it so that
# file runs completely unmodified rather than needing this one name stubbed out.
#
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION (summarised terms retained from the
# upstream LICENSE file; the full canonical text is at http://www.apache.org/licenses/LICENSE-2.0)
#
# 1. Definitions — "License", "Licensor", "Legal Entity", "You", "Source"/"Object" form, "Work",
#    "Derivative Works", "Contribution", "Contributor" as defined by the Apache 2.0 license text.
# 2. Grant of Copyright License — perpetual, worldwide, non-exclusive, no-charge, royalty-free,
#    irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display,
#    publicly perform, sublicense, and distribute the Work and Derivative Works.
# 3. Grant of Patent License — a patent license as stated in the License, terminating upon
#    initiating patent litigation over the Work.
# 4. Redistribution — permitted in Source or Object form provided this License and all
#    copyright/patent/trademark/attribution notices are retained, and any modified files carry
#    prominent notices of the changes (none made here — this file is unmodified).
# 5. Submission of Contributions is under this License unless stated otherwise.
# 6. Trademarks — this License grants no permission to use the Licensor's trade names, trademarks,
#    or service marks.
# 7. Disclaimer of Warranty — the Work is provided "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF
#    ANY KIND, express or implied.
# 8. Limitation of Liability — no Contributor is liable for damages arising from use of the Work.
# 9. Accepting Warranty or Additional Liability — offered only on the offeror's own responsibility.
#
# Copyright TauricResearch (see attribution note above)
# Licensed under the Apache License, Version 2.0; you may not use this file except in compliance
# with the License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#
# ============================== VENDORED FROM HERE ==============================
"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.

``extract_rating`` returns ``None`` when no rating can be found, so the graph can
surface an explicit ``REVIEW`` signal instead of a fabricated ``Hold`` (#1170).
``parse_rating`` keeps the legacy silent-default behaviour for callers (e.g. the
memory log) that need a rating string regardless.
"""

from __future__ import annotations

import re
import unicodedata

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

# Signal emitted when the model's decision has no recognizable rating. It is not
# a tradeable position: it flags output that needs a human/re-run rather than
# silently degrading to Hold. Callers that map the signal onto the 5-tier enum
# (e.g. ``PortfolioRating(signal)``) should guard with ``is_review`` first.
RATING_REVIEW = "REVIEW"

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)

# Standalone 5-tier word anywhere (word boundaries so "Buyer"/"Holding" don't match).
_RATING_WORD_RE = re.compile(
    r"\b(" + "|".join(RATINGS_5_TIER) + r")\b", re.IGNORECASE
)


def extract_rating(text: str) -> str | None:
    """Extract a 5-tier rating from prose, or ``None`` if none is present.

    Two-pass strategy on the NFKC-normalized text (so fullwidth punctuation like
    ``Rating：Overweight`` is matched the same as ASCII):
    1. An explicit "Rating: X" label (tolerant of markdown bold).
    2. The first standalone 5-tier rating word found anywhere.
    """
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)

    for line in norm.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            return m.group(1).capitalize()

    m = _RATING_WORD_RE.search(norm)
    if m:
        return m.group(1).capitalize()

    return None


def parse_rating(text: str, default: str = "Hold") -> str:
    """Extract a 5-tier rating, falling back to ``default`` when none is found.

    Legacy convenience wrapper: it always returns a rating string, so an
    unparseable decision silently becomes ``default`` (``Hold``). Callers that
    must distinguish "no rating" from a real Hold should use
    :func:`extract_rating` (or the graph's REVIEW-surfacing signal) instead.
    """
    rating = extract_rating(text)
    return rating if rating is not None else default


def is_review(signal: str) -> bool:
    """Whether a signal is the non-tradeable REVIEW sentinel (#1170)."""
    return signal == RATING_REVIEW
