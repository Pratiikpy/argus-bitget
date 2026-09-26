"""What followed a decision: the move, where the number came from, and when it is no direction.

Moved from `eval/shadow.py` on 2026-09-27. The desk's own review (`desk/review.py`) grades leans
with the same arithmetic the shadow record uses, and imported it upward from the evaluation layer;
the definition of an outcome is decision vocabulary, so it lives here and both read it.
"""

from __future__ import annotations

from typing import Any

DEAD_ZONE_BPS = 5.0
"""Moves smaller than this are not scored either way.

A lean is a claim about direction, and a two-basis-point move is not a direction — it is the tape
being flat. Scoring those as wins or losses adds noise to the numerator and the denominator at once.
"""


def move_of(row: Any) -> tuple[float, str] | None:
    """The move that followed a decision, and where the number came from.

    Two recorded shapes, because the ledger settles the two verdict classes differently:

    * an **abstention** carries ``counterfactual_move_bps``, written by ``settle_abstention`` at
      settlement — the move as it happened, signed, not as a judgement;
    * a **trade** carries an ``exit_price`` instead, and the same move is
      ``(exit - entry) / entry * 10000``.

    The second is computed here rather than stored, and that is the identical arithmetic
    ``ledger.settle_abstention`` performs (`paper/ledger.py:383`) on fields of equal standing — both
    prices are settlement fields, neither is hashed, and neither is re-fetched from a price series
    that may since have been revised. Restricting the record to the first would have made it a
    measurement of abstentions wearing the name of the desk.

    Returns ``None`` for anything unsettled, which is not a gap in the lean but an outcome that does
    not exist yet.
    """
    raw = getattr(row, "counterfactual_move_bps", None)
    if raw is not None:
        try:
            return float(raw), "counterfactual"
        except (TypeError, ValueError):
            return None
    exit_price = getattr(row, "exit_price", None)
    entry_price = getattr(row, "entry_price", None)
    if exit_price is None or entry_price is None:
        return None
    try:
        entry = float(entry_price)
        if entry <= 0:
            return None
        return (float(exit_price) - entry) / entry * 10_000.0, "fill"
    except (TypeError, ValueError):
        return None
