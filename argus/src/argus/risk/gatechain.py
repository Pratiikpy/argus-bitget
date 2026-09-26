"""The Constitution's gate chain and the risk record's vocabulary.

Moved here from `eval/autopsy.py` on 2026-09-27. The chain is what the risk layer *is*: the
desk writes records in this vocabulary (`paper/runner.py`), the carry desk and the order
review read it, and the autopsy judges it. Kept in the evaluation layer, every one of them
imported upward from the harness that grades them, against the published layer order.
"""

from __future__ import annotations

from dataclasses import dataclass

FIRED = "FIRED"
"""The gate executed and changed or refused the intent."""

PASSED = "PASSED"
"""The gate executed and let the intent through. A real permission."""

UNREACHED = "UNREACHED"
"""The gate never executed, because an earlier one returned first.

Not the same as PASSED and never to be counted with it. This is the state that makes an unexercised
risk layer look like a permissive one.
"""


@dataclass(frozen=True, slots=True)
class Gate:
    """One rule in the Constitution's ordered chain, as it is written in the source.

    ``order`` is the position in `risk.constitution.ConstitutionPolicy.rule`. The chain is a
    sequence of early returns, so a gate is reachable on a given decision only if every gate before
    it declined to return. Recording the order here rather than inferring it keeps the funnel
    honest when the policy is reordered: a rule moved above the exposure check becomes reachable,
    and this file must be updated with it or the test below fails.
    """

    order: int
    name: str
    what_it_checks: str
    terminal: bool = True
    """Does this gate END the evaluation, or only contribute a ceiling?

    **The distinction was added on 2026-09-15 and it changes what UNREACHED means.** The chain used
    to be seven early returns, so any gate could leave every later one unexecuted. It was
    demonstrated that this let an order resized by `unhedgeable_gap` bypass `risk_budget` entirely —
    approved at 20,000 while the circuit breaker sized the book to zero.

    The Constitution now evaluates **every** ceiling gate and applies the minimum, so:

    * a **terminal** gate (`no_exposure`, `min_confidence`, `oracle_stale`) still ends the
      evaluation, and gates after it are genuinely UNREACHED;
    * a **ceiling** gate (`unhedgeable_gap`, `risk_budget`, `session_volatility`, `max_position`)
      always runs once the terminals pass. It can be UNREACHED only because a terminal fired —
      never because another ceiling bound first.

    Reporting a ceiling as UNREACHED merely because a different ceiling was tighter would understate
    how much of the layer is exercised, which is the opposite of this module's purpose.
    """


CHAIN: tuple[Gate, ...] = (
    Gate(1, "no_exposure", "verdict carries no quantity, or quantity <= 0", terminal=True),
    Gate(2, "min_confidence", "stated confidence below the 0.55 floor", terminal=True),
    Gate(3, "oracle_stale", "NAV stale while the anchor is shut", terminal=True),
    Gate(4, "unhedgeable_gap", "no hedge placeable and size above the unhedged cap",
         terminal=False),
    Gate(5, "gross_exposure", "whole-book gross notional above the configured cap",
         terminal=False),
    Gate(6, "signed_exposure", "whole-book net directional notional above the configured cap",
         terminal=False),
    Gate(7, "hedge_integrity", "REDUCE would leave a linked hedge position's partner exposed",
         terminal=False),
    Gate(8, "margin_usage", "venue-reported margin ratio at or above the policy cap",
         terminal=False),
    Gate(9, "factor_exposure", "measured book exposure to a named factor above its configured "
         "cap", terminal=False),
    Gate(10, "scenario_loss", "worst modelled benchmark shock already past the loss floor",
         terminal=False),
    Gate(11, "liquidation_cost", "worst position's forced-exit slippage proxy at or above the "
         "policy cap", terminal=False),
    Gate(12, "per_symbol_underperformance", "one symbol's windowed realized PnL below its "
         "configured floor — narrows that symbol only, not the whole order", terminal=False),
    Gate(13, "risk_budget", "circuit breaker's drawdown ladder sizes the book below the request",
         terminal=False),
    Gate(14, "session_volatility", "measured path volatility above the regular-hours baseline",
         terminal=False),
    Gate(15, "max_position", "size above the position cap", terminal=False),
)
"""The Constitution's chain, in source order (`risk/constitution.py`, ``ConstitutionPolicy.rule``).

Written out because the risk records carry only the constraint that *bound*, and a reader cannot
tell from ``binding_constraint: none`` which of the other four were even consulted.
"""


NOTHING_BOUND = "none"
"""Every gate ran and allowed. **Not** the same as gate 1 returning — see :data:`RECORD_SCHEMA`."""

RECORD_SCHEMA = 2
"""The risk-record vocabulary this module can read.

**Schema 1 wrote ``"none"`` for two different outcomes** — gate 1 returning (no exposure proposed)
*and* the terminal all-clear (passed all seven) — so a reader could not tell them apart, and this
module guessed wrong in the dangerous direction: it counted every ``"none"`` as gate 1 firing,
which makes gates 2-7 read UNREACHED on a decision that in fact reached and passed all of them.

Schema 2 gives gate 1 its own name (`agents/desk.py`), leaving ``"none"`` to mean only *nothing
bound*. Records are refused rather than reinterpreted, because a schema-1 file read under schema-2
rules would silently report the **opposite** of the truth on this project's single differentiator.
Run ``python -m argus.eval.migrate_risk_records`` to convert.
"""


__all__ = ["CHAIN", "FIRED", "NOTHING_BOUND", "PASSED", "RECORD_SCHEMA", "UNREACHED", "Gate"]
