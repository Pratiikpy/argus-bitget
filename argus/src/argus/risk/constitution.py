"""The Constitution: the desk's deterministic risk rules, applied in code and never in a prompt.

Moved from `agents/desk.py` on 2026-09-27. The rules depend on nothing above the risk layer, and
three callers below the agents needed them: the carry desk and the order review ran orders through
the real Constitution, and both imported it upward from the agent that proposes the orders it
judges. The agent now imports it from here like everyone else.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from decimal import Decimal
from typing import Any

from argus.decision.verdicts import (
    ConstitutionRuling,
    ConstitutionVerdict,
    Intent,
    PositionSide,
    Side,
    Verdict,
    apply_constraint,
)
from argus.risk.hedgeability import HedgeabilitySurface
from argus.truth.clocks import SessionState


@dataclass(frozen=True, slots=True)
class ConstitutionPolicy:
    """Deterministic risk rules, in versioned config rather than in a prompt.

    TradingAgents' own teardown records that its risk management is "via prompt guidance, not code
    enforcement". A rule a model can be talked out of is not a rule, so every limit here is applied
    in Python and the model is told the outcome rather than asked to respect it.
    """

    reference_price: Decimal | None = None
    """Price per unit, used to turn an intent's **quantity** into the **notional** every cap here
    is denominated in.

    **Added 2026-09-21, after a constructed adversary found that four gates were comparing a unit
    count against a dollar ceiling.** `max_position_notional` is 50,000 *dollars*; the gate read
    `intent.quantity > 50_000`, and quantity is *units*. At NVDAUSDT's ~$180 a unit that made the
    cap **180x looser than written**: a 400-unit order is $72,000 of notional, over the cap, and
    passed untouched, while the gate could only fire above 50,000 units — $9,000,000. The
    `gross_exposure` and `signed_exposure` gates were worse: they subtract a real dollar book total
    (`Book.total_gross_notional`, which does multiply by price) from a dollar cap and then compared
    the dollar headroom against a unit count, mixing the two inside one expression.

    447 live decisions never caught it because the desk has never proposed a position. A rogue
    agent built to attack the caps caught it on its first run (`eval/rogue.py`).

    ``None`` means no price was supplied, and then every notional gate is **skipped and said to be
    skipped** rather than silently evaluated in the wrong unit — the same discipline `book_state`
    and `session_risk` already follow. A cap that cannot be computed is absent, not satisfied."""

    max_position_notional: Decimal = Decimal("50000")
    max_unhedged_notional: Decimal = Decimal("20000")
    """Ceiling on exposure carried with an empty hedge menu — the Sleeping-Anchor constraint."""

    min_confidence_to_trade: float = 0.55

    book_state: Any = None
    """The book the risk budget is drawn against (:class:`argus.risk.circuit.BookState`), or
    ``None``.

    Injected on the same terms as ``session_risk`` below and for the same reason: the rulebook is
    deterministic and a rule that reaches for the network is a rule that can fail open. ``None``
    means the gate does not fire, which is honest when no book has been computed at all — and is
    not the same as a book reporting zero drawdown it never measured.
    """

    book: Any = None
    """The real portfolio (:class:`argus.desk.book.Book`), or ``None``.

    Distinct from ``book_state`` above: that one is the circuit breaker's realised-equity scalar,
    this one is actual position state — Foundation 5's safety contract names ``order size``,
    ``signed exposure``, ``gross exposure``, ``factor exposure``, ``margin usage``,
    ``liquidation cost``, ``scenario loss`` and ``hedge integrity`` as eight separate dimensions
    precisely because collapsing them loses information (a resized order proves quantity never
    rose, not that risk never rose). ``gross_exposure`` below is the first of the seven still
    building on top of ``max_position``'s order-size dimension. Same injection discipline as
    ``session_risk`` and ``book_state``: ``None`` means the gate does not fire, not that the book
    is empty.
    """

    max_gross_exposure_notional: Decimal = Decimal("150000")
    """Ceiling on total gross notional across the whole book, this order included.

    Three times ``max_position_notional`` — a stated, revisable default (this project's other two
    flat caps, ``max_position_notional`` and ``max_unhedged_notional``, carry no derivation either;
    this one at least states its reasoning): a book concurrently near its per-position cap on three
    names is a materially different risk from one order at that cap, and `desk.book.Book` had no
    caller to size this against before today. Revisit once real position sizing exists to reason
    from `argus.desk.book.VenueMarginSnapshot`'s ``account_equity`` instead of a flat number.
    """

    max_signed_exposure_notional: Decimal = Decimal("100000")
    """Ceiling on net directional notional across the whole book — long minus short, netted
    within a symbol, projected including this order.

    Twice ``max_position_notional``: the desk may be net-long or net-short by at most two
    full-sized positions' worth, overall. Distinct from ``gross_exposure`` on purpose — a book
    hedged flat can carry high gross with zero net, and a book that is all one direction can carry
    the same gross with maximal net; the two describe different risks and Foundation 5's contract
    lists them separately for exactly that reason (see `desk.book.Book.total_signed_notional`'s
    docstring). Revisit alongside ``max_gross_exposure_notional`` once real equity exists to size
    both against.
    """

    max_margin_usage_ratio: Decimal = Decimal("0.5")
    """Policy cap on the venue-reported margin ratio, in the fraction scale
    `desk.book.VenueMarginSnapshot`'s docstring evidences (1.0 == Bitget's own conceptual 100%
    liquidation trigger, confirmed against Bitget's own liquidation page; the exact numeric scale
    of this specific field is evidenced by analogy to a verified sibling field, not independently
    proven — stated in the gate's own reason string, not hidden).

    **Half of Bitget's own trigger — our own conservative policy choice, not a venue-defined
    number**, the same "stated, revisable default" honesty as ``max_gross_exposure_notional``.
    Chosen because the venue's own ratio is a cross-asset, proprietary computation this project
    cannot reproduce (module docstring) — a policy line drawn well short of the real one buys
    room for exactly that uncertainty, rather than gating at the edge of a number we cannot
    ourselves verify precisely.
    """

    factor_exposures: Any = None
    """Precomputed book exposure to each named systematic factor
    (``Sequence[argus.desk.portfolio.FactorExposure]``), or ``None``.

    Same injection discipline as ``session_risk``/``book_state``: computed by the caller from
    `desk.portfolio.factor_exposures` against the book's current weights and real market history
    — the Constitution does not fetch either (module docstring: "a rule that reaches for the
    network is a rule that can fail open"). ``None`` means the gate does not fire, honest for a
    desk that has never run the regression, not read as zero exposure.
    """

    factor_exposure_limits: Mapping[str, float] = field(default_factory=dict)
    """Per-factor cap on ``abs(FactorExposure.exposure)``, keyed by factor name.

    A factor absent from this mapping is uncapped, not zero-capped — matching every other
    ``None``-means-unmeasured convention here rather than silently thresholding at a made-up
    default for a factor nobody configured.
    """

    stress_outcomes: Any = None
    """Precomputed benchmark-shock results (``Sequence[argus.desk.portfolio.StressOutcome]``),
    or ``None``. Computed by the caller from `desk.portfolio.stress_by_beta` against the book's
    current weights and :data:`argus.desk.portfolio.STANDARD_SHOCKS` (or a caller-supplied shock
    set). ``None`` means the gate does not fire.
    """

    max_scenario_loss_pct: float | None = None
    """Floor on the worst :class:`~argus.desk.portfolio.StressOutcome`'s ``portfolio_move_pct``
    (most negative = worst). ``None`` means the gate does not fire — an unset floor is not the
    same claim as an infinite one, and this field exists so a caller must set it deliberately
    rather than the gate silently adopting some default severity nobody chose.
    """

    liquidation_cost_estimates: Any = None
    """Precomputed forced-exit slippage per open position
    (``Mapping[str, argus.market.depth.Sweep]``, keyed by symbol), or ``None``.

    **A liquidation-severity PROXY, not the literal cost of Bitget's own bankruptcy-price/ADL
    mechanism — stated here and in the gate's own reason string, not hidden.** Building the real
    thing needs Bitget's tiered maintenance-margin table, which has not been read (the earlier
    liquidation-page research covered the ``mgnRatio``/100% trigger, not the tier table a true
    bankruptcy price is derived from). This instead reads real, live order-book depth
    (`market.depth.OrderBook.sweep`, verified 2026-09-15 against the actual class) for "how much
    would it cost to exit this position by force, right now" — an honestly-measured number, just
    not Bitget's own exact figure at the moment of an actual liquidation. Computed by the caller,
    never fetched inside `rule()` (same discipline as every other injected field here). ``None``
    means no order-book fetch has happened this cycle, not zero cost.
    """

    max_liquidation_cost_bps: Decimal | None = None
    """Floor on the worst position's forced-exit ``slippage_bps``. ``None`` means the gate does
    not fire.

    **The default is ``None``: the gate is off.** The live cycle computes no forced-exit estimate,
    so a threshold here could not fire there (`data/risk_policy_live.json` records which gates can).
    500 bps (5%) is the value `eval/riskproof.py` exercises the gate at — until 2026-09-28 this
    docstring called it "our own conservative policy default", which the field never was. That
    value is roughly 40x this venue's measured round-trip taker cost (~12 bps, `cost/model.py`): a
    position that would cost forty times its normal exit to unwind by force is a severity signal
    worth blocking on, however exactly Bitget's own bankruptcy-price math would price it.
    """

    graded_predictions: Any = None
    """Settled (confidence, was-it-right) pairs — :class:`argus.eval.observatory.Prediction`.

    This is what decides whether stated confidence may scale a position. `risk/sizing.py` requires
    twenty graded outcomes and an expected calibration error under 0.15 before it will use
    confidence at all; below that it returns a fixed fraction and names the reason. Empty is the
    honest state today and produces the fixed fraction, which is **tighter** than the position cap
    that used to be the only limit — an unproven desk should not be sized like a proven one.
    """

    assumed_payoff: Decimal = Decimal("1")
    """Win/loss ratio fed to the Kelly arithmetic, used **only once calibration passes**.

    Stated as an explicit, neutral 1:1 rather than inferred, because with zero settled trades there
    is nothing to infer it from and a payoff pulled out of the air would be levered directly into
    position size the moment the calibration gate opened. A caller with a realised win/loss record
    should pass it; until one exists this is the assumption, and it is written down rather than
    buried in a default deep inside the sizing call.
    """

    session_risk: Any = None
    """Measured per-phase volatility for this instrument
    (:class:`argus.risk.session_risk.SessionRisk`), or ``None``.

    Supplied by the caller rather than fetched here: this class is the deterministic rulebook and a
    rule that reaches for the network is a rule that can fail open. ``None`` means the throttle
    below does nothing and says so, which is the same refusal `risk/effectiveness.py` makes — an
    unmeasured risk is not a risk of zero, but neither is it a licence to invent a cautious number.
    """

    session_horizon_bars: int = 24
    """How long a position is assumed to be held, for the volatility-path calculation.

    Twenty-four hours because that is what `paper/runner.py` actually does: decisions are settled
    against the price one day later. A different desk would pass a different number and get a
    different throttle, which is the point — the reopen jump matters enormously at two hours and not
    at all at twenty-four, and a constant multiplier could not express that.
    """

    min_symbol_realized_pnl: Decimal | None = None
    """Floor on one symbol's windowed realized PnL
    (`desk.book.Book.symbol_realized_pnl_since`). ``None`` means the gate does not fire.

    **The ARGUS-native counterpart to freqtrade's ``LowProfitPairs``**
    (`eval.freqtrade_baseline.freqtrade_low_profit_pairs`, ported and tested against freqtrade's
    own source at `low_profit_pairs.py:41-73`) — closing the real capability gap proven in
    `tests/test_freqtrade_baseline.py`'s ``TestLowProfitPairsIsGenuinelyPerSymbol``:
    `risk/circuit.py`'s Breaker/Activation is whole-book only
    (``inspect.getsource(circuit)`` contains zero occurrences of ``"symbol"``), so nothing before
    this gate could lock out one underperforming symbol while leaving the rest of the book
    tradeable — every other ceiling here narrows the whole order or nothing.

    Reads real `Book` fills directly via ``symbol_realized_pnl_since``, not a
    backtest-reconstructed `backtest.engine.SyntheticTrade` sequence: `extract_trades` exists so a
    *backtest comparison* can reconstruct discrete trades from a continuous-weight simulation, but
    live `Book` already carries real `Lot`s with real ``ts_filled`` — replaying those directly is
    the honest path, not a reason to round-trip live state through the backtest engine just to
    reuse the same arithmetic.
    """

    symbol_underperformance_window_minutes: int = 1440
    """Lookback window for ``min_symbol_realized_pnl``, in minutes.

    Twenty-four hours — the same horizon ``session_horizon_bars`` uses, and for the same reason:
    `paper/runner.py` settles every decision a day later, so a symbol's last 24 hours of realized
    results is the horizon this desk actually reasons in, not its lifetime record.
    """


    def _notional(self, quantity: Decimal) -> Decimal | None:
        """Quantity in units -> notional in dollars, or ``None`` when no price was supplied.

        Every caller must treat ``None`` as "this gate cannot be evaluated" and say so, never as
        "this gate passed". See :attr:`reference_price`."""
        if self.reference_price is None or self.reference_price <= 0:
            return None
        return quantity * self.reference_price

    def rule(
        self, intent: Intent, *, session: SessionState, hedges: HedgeabilitySurface
    ) -> ConstitutionRuling:
        if not intent.verdict.carries_quantity or intent.quantity <= 0:
            # **Gate 1 names itself, like every other gate.** It used to return "none", the same
            # token the terminal all-clear below returns — and `eval/autopsy.py` counted "none" as
            # gate 1 firing. So an intent that reached and cleared all seven gates was counted as
            # gate 1 *firing*, and gates 2-7 reported UNREACHED **on the very decision that proved
            # them reachable**. The funnel inverted exactly when it finally had something to say.
            #
            # Latent only because no decision has ever proposed exposure. `desk/carrydesk.py:112`
            # worked around it by reading `resulting_intent.quantity`; with the two return sites
            # distinguishable that workaround is no longer needed anywhere.
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.ALLOW,
                binding_constraint="no_exposure",
                reason="no exposure proposed; nothing to narrow",
            )

        if intent.stated_confidence < self.min_confidence_to_trade:
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.REJECT,
                binding_constraint="min_confidence",
                reason=(
                    f"stated confidence {intent.stated_confidence:.2f} is below the "
                    f"{self.min_confidence_to_trade} floor"
                ),
            )

        if session.nav_is_stale() and session.is_anchor_asleep:
            return apply_constraint(
                intent, verdict=ConstitutionVerdict.DELAY,
                binding_constraint="oracle_stale",
                reason="NAV is stale while the anchor is shut; no reliable reference price",
            )

        # --- CONSTRAINT PRODUCERS ------------------------------------------------------------
        #
        # **Every remaining gate contributes a ceiling, and the binding one is the MINIMUM of all
        # of them — not the first encountered.** This replaced a chain that returned on the first
        # gate to bind, and the difference is not cosmetic. Demonstrated on 2026-09-15:
        #
        #     book in deep drawdown (equity 30000 vs a 100000 peak, 3 consecutive losses)
        #     proposed 45000, no hedge placeable
        #       -> unhedgeable_gap capped it at 20000 and RETURNED
        #       -> risk_budget, which would have capped it at 0, never ran
        #       -> a 20000 order was approved while the circuit breaker said trade nothing,
        #          and the record showed `binding_constraint: unhedgeable_gap` — a gate that
        #          appeared to be working
        #
        # "A risk layer may only reduce what you choose" was true of each gate in isolation and
        # false of the chain, because the first gate to bind ended it. Reachability reporting is an
        # *observability* feature; it was standing in for *enforcement*, and those are different
        # properties. Collecting every ceiling restores the claim the Constitution actually makes.
        #
        # Latent rather than active: all 231 recorded decisions carry quantity 0, so the terminal
        # `no_exposure` check above short-circuits and the chain has never run past it on live
        # data. It would have fired on the first decision that proposed exposure.
        ceilings: list[tuple[str, Decimal, str]] = []

        # `price` is bound rather than read through `self` at each site so the type checker can
        # see that a notional gate and its ceiling share one non-None price. `notional` is derived
        # from it, so the two are None together by construction.
        price = self.reference_price if (
            self.reference_price is not None and self.reference_price > 0
        ) else None
        notional = intent.quantity * price if price is not None else None
        # Stated, not swallowed. Without a price the four notional gates cannot be evaluated, and
        # this suffix travels into the ruling's reason so a reader can tell "not checked" from
        # "checked and passed". An absent cap that reads like a satisfied one is how a risk layer
        # becomes decorative.
        unpriced = "" if notional is not None else (
            " | NOT EVALUATED: no reference_price, so the unhedged, gross, signed and "
            "max_position notional caps were skipped rather than passed"
        )
        if (hedges.is_empty and price is not None and notional is not None
                and notional > self.max_unhedged_notional):
            ceilings.append((
                "unhedgeable_gap",
                self.max_unhedged_notional / price,  # dollars -> units; see reference_price
                f"no hedge placeable for {session.hours_to_next_discovery:.1f}h; "
                f"unhedged exposure capped at {self.max_unhedged_notional}",
            ))

        # Gross exposure — the whole book, not just this order. "Smaller quantity is not safer"
        # applies here exactly as Foundation 5's safety contract says it does to margin: a book
        # already carrying $140,000 gross across other names and proposing $20,000 more is a
        # materially different risk from a first order of $20,000, and `max_position_notional`
        # above only ever sees this single order's own size. `self.book` is the first Foundation-3
        # consumer in the Constitution — see its field docstring for why it is a separate injection
        # from `self.book_state`.
        if self.book is not None:
            existing_gross = self.book.total_gross_notional()
            headroom = max(Decimal("0"), self.max_gross_exposure_notional - existing_gross)
            if price is not None and notional is not None and notional > headroom:
                ceilings.append((
                    "gross_exposure",
                    headroom / price,  # dollars -> units; see reference_price
                    f"book already carries {existing_gross} gross; the "
                    f"{self.max_gross_exposure_notional} cap leaves {headroom} of headroom for "
                    f"this order",
                ))

            # Net exposure — a book can be flat on gross (hedged) or maximal on gross (one-way)
            # at the same gross number; this reads the direction gross deliberately throws away.
            # ``direction`` matters: an order that trims an existing skew has *more* headroom, not
            # less, because it moves net exposure toward zero rather than away from it.
            direction = Decimal("1") if intent.side is Side.BUY else Decimal("-1")
            existing_signed = self.book.total_signed_notional()
            signed_headroom = max(
                Decimal("0"), self.max_signed_exposure_notional - direction * existing_signed
            )
            if price is not None and notional is not None and notional > signed_headroom:
                ceilings.append((
                    "signed_exposure",
                    signed_headroom / price,  # dollars -> units; see reference_price
                    f"book is net {existing_signed} (long positive); a {intent.side} at this size "
                    f"would push net exposure past the {self.max_signed_exposure_notional} cap "
                    f"({signed_headroom} of headroom in this direction)",
                ))

            # Hedge integrity — "smaller quantity is not safer" in its sharpest form. A REDUCE
            # narrows quantity, which every other ceiling here reads as strictly less risky; when
            # the position being reduced is declared as a hedge for another position that is
            # STILL OPEN, reducing it is the opposite. `Verdict.REDUCE` plus `intent.side` already
            # disambiguate which position this is without any change to `Intent`: a SELL reduces
            # a LONG, a BUY reduces a SHORT — Bitget's own `tradeSide` is a venue execution detail
            # (`execution/bitget_client.py`), not a Constitution-level ambiguity, and conflating
            # the two was a design error caught while building this, not assumed away.
            if intent.verdict is Verdict.REDUCE:
                being_reduced = (
                    intent.symbol,
                    PositionSide.LONG if intent.side is Side.SELL else PositionSide.SHORT,
                )
                position = self.book.positions.get(being_reduced)
                if position is not None and not position.is_flat:
                    cluster = self.book.hedge_cluster(*being_reduced)
                    still_hedging = [
                        other for other in cluster if other != being_reduced
                        and (p := self.book.positions.get(other)) is not None and not p.is_flat
                    ]
                    if still_hedging:
                        names = ", ".join(f"{s} {v}" for s, v in still_hedging)
                        ceilings.append((
                            "hedge_integrity",
                            Decimal("0"),
                            f"{intent.symbol} {being_reduced[1]} is linked as a hedge for "
                            f"{names}, which remains open; an automatic reduction is refused "
                            f"rather than assumed safe — a resize can raise the risk it is "
                            f"hedging even while lowering its own quantity",
                        ))

            # Margin usage — a binary block, not a headroom, and the reason string says exactly
            # why: Bitget's own margin engine is cross-asset and proprietary (module docstring),
            # so there is no way to compute how much of THIS order's quantity the venue would
            # still accept before its own ratio moved further — only whether the venue-reported
            # ratio is already past our policy cap. `mgn_ratio` is ``None``-guarded at the
            # `self.book.margin is not None` level: a book with no margin snapshot fetched yet
            # cannot honestly say anything about margin usage, and must not be read as zero usage.
            if (
                self.book.margin is not None
                and self.book.margin.mgn_ratio >= self.max_margin_usage_ratio
            ):
                ceilings.append((
                    "margin_usage",
                    Decimal("0"),
                    f"venue-reported margin ratio {self.book.margin.mgn_ratio} is at or above "
                    f"the {self.max_margin_usage_ratio} policy cap (Bitget's own liquidation "
                    f"trigger is conceptually 100%; this field's exact numeric scale is "
                    f"evidenced by analogy, not directly proven — see "
                    f"`desk.book.VenueMarginSnapshot`'s docstring); no further exposure until it "
                    f"eases",
                ))

        # Factor exposure — a binary block, same reasoning as margin_usage. `FactorExposure` is a
        # regression slope for the WHOLE book (`desk.portfolio.factor_exposures`), not decomposed
        # per symbol, so there is no per-symbol beta here to compute how much of THIS order's
        # quantity would move the aggregate — only whether the book's already-measured exposure is
        # past the cap. Injected, not fetched: computing it needs real market history, which this
        # deterministic rule must not reach for (see `session_risk`'s field docstring).
        if self.factor_exposures is not None:
            for exposure in self.factor_exposures:
                limit = self.factor_exposure_limits.get(exposure.factor)
                if limit is None or exposure.exposure is None:
                    continue
                if abs(exposure.exposure) >= limit:
                    ceilings.append((
                        "factor_exposure",
                        Decimal("0"),
                        f"book exposure to {exposure.factor!r} is {exposure.exposure:.3f}, at or "
                        f"above the {limit} cap; no further exposure until it is reduced",
                    ))
                    break

        # Scenario loss — same binary shape again. `stress_by_beta`'s output is the book's
        # market-driven move under a shock, not this order's marginal contribution to it, for the
        # same per-symbol-decomposition reason as factor exposure above.
        if self.stress_outcomes is not None and self.max_scenario_loss_pct is not None:
            scored = [o for o in self.stress_outcomes if o.portfolio_move_pct is not None]
            worst = min(scored, key=lambda o: o.portfolio_move_pct, default=None)
            if worst is not None and worst.portfolio_move_pct <= self.max_scenario_loss_pct:
                ceilings.append((
                    "scenario_loss",
                    Decimal("0"),
                    f"worst modelled shock ({worst.shock}) already moves the book "
                    f"{worst.portfolio_move_pct:+.2f}%, past the {self.max_scenario_loss_pct}% "
                    f"cap; no further exposure until stress is reduced",
                ))

        # Liquidation cost — a proxy, and the reason string says so every time it binds, not just
        # in the field's own docstring. Same binary shape as the three gates above: `Sweep` prices
        # exiting one position at today's book depth, not this specific order's own marginal
        # contribution to that cost.
        if (
            self.liquidation_cost_estimates is not None
            and self.max_liquidation_cost_bps is not None
        ):
            worst_symbol, worst_sweep = max(
                self.liquidation_cost_estimates.items(),
                key=lambda item: item[1].slippage_bps, default=(None, None),
            )
            if (
                worst_sweep is not None
                and worst_sweep.slippage_bps >= self.max_liquidation_cost_bps
            ):
                ceilings.append((
                    "liquidation_cost",
                    Decimal("0"),
                    f"forced-exit slippage proxy for {worst_symbol} is "
                    f"{worst_sweep.slippage_bps:.1f}bps (real order-book depth, not Bitget's own "
                    f"bankruptcy-price formula — see `liquidation_cost_estimates`'s docstring), "
                    f"at or above the {self.max_liquidation_cost_bps}bps cap; no further exposure "
                    f"until it eases",
                ))

        # Per-symbol underperformance — the ARGUS-native LowProfitPairs. Unlike every ceiling
        # above, this one narrows exposure to ONE symbol, not the order's whole size: an order on
        # a different, healthy symbol is untouched even while this symbol is locked out.
        if self.book is not None and self.min_symbol_realized_pnl is not None:
            cutoff = session.as_of - timedelta(minutes=self.symbol_underperformance_window_minutes)
            windowed_pnl = self.book.symbol_realized_pnl_since(intent.symbol, cutoff)
            if windowed_pnl < self.min_symbol_realized_pnl:
                ceilings.append((
                    "per_symbol_underperformance",
                    Decimal("0"),
                    f"{intent.symbol} realized {windowed_pnl} over the last "
                    f"{self.symbol_underperformance_window_minutes} minutes, below the "
                    f"{self.min_symbol_realized_pnl} floor; no further exposure to this symbol "
                    f"until it recovers",
                ))

        # The circuit breaker's de-risking ladder, applied to the size the desk asked for.
        #
        # `risk/circuit.py` and `risk/sizing.py` were both complete, both tested, and **neither was
        # ever called on a live decision**: `sizing.size()` had no caller anywhere in `src/argus`,
        # and `sizing.py:149` documented `risk_multiplier` as coming from `circuit.risk_multiplier`
        # while nothing connected them. A risk control that cannot change a decision is a comment.
        #
        # `book_state` is injected exactly as `session_risk` is, and absent it the gate does not
        # fire — a visible degradation rather than a silent one, and never a fabricated drawdown.
        if self.book_state is not None:
            from argus.risk.circuit import risk_multiplier as circuit_multiplier
            from argus.risk.sizing import size as size_position

            # The session throttle is deliberately passed as 1 here and applied by its own ceiling
            # below. `sizing.size` accepts both multipliers and would fold them into one fraction,
            # which would make it impossible to say which of the two bound.
            sizing = size_position(
                win_probability=intent.stated_confidence,
                payoff=self.assumed_payoff,
                predictions=self.graded_predictions or (),
                risk_multiplier=circuit_multiplier(self.book_state),
                session_multiplier=Decimal("1"),
            )
            budget = self.book_state.equity * sizing.fraction
            if budget < intent.quantity:
                ceilings.append((
                    "risk_budget",
                    budget,
                    f"risk budget {sizing.fraction:.1%} of {self.book_state.equity} = {budget} "
                    f"({sizing.basis}); drawdown from peak "
                    f"{self.book_state.total_drawdown:.1%}",
                ))

        # Measured session volatility — how violent the path is, as distinct from whether the
        # exposure can be covered at all.
        if self.session_risk is not None:
            from argus.risk.session_risk import throttle as session_throttle

            scaled = session_throttle(
                self.session_risk, start=session.as_of,
                horizon_bars=self.session_horizon_bars,
            )
            if scaled.multiplier < 1:
                narrowed = intent.quantity * scaled.multiplier
                if narrowed < intent.quantity:
                    ceilings.append(("session_volatility", narrowed, scaled.reason))

        # The hard cap, which must bind last so it is never diluted by a multiplier. As a ceiling
        # among ceilings that ordering is automatic: a minimum does not care what order it is
        # given its arguments, which is one fewer thing to get wrong.
        if price is not None and notional is not None and notional > self.max_position_notional:
            ceilings.append((
                "max_position",
                self.max_position_notional / price,  # dollars -> units; see reference_price
                f"position capped at {self.max_position_notional} notional",
            ))

        # --- FINAL VALIDATION ------------------------------------------------------------------
        if ceilings:
            name, permitted, reason = min(ceilings, key=lambda c: c[1])
            others = [c for c in ceilings if c[0] != name]
            trail = reason
            if others:
                # The explanation improves rather than degrades: naming every ceiling that was
                # computed is strictly more informative than naming the first one to bind.
                trail += " | also computed: " + "; ".join(
                    f"{other_name} at {other_cap}" for other_name, other_cap, _ in others
                )
            if permitted <= 0:
                return apply_constraint(
                    intent, verdict=ConstitutionVerdict.REJECT,
                    binding_constraint=name,
                    reason=trail + " — no size satisfies every constraint",
                )
            ruled = apply_constraint(
                intent, verdict=ConstitutionVerdict.RESIZE,
                binding_constraint=name, reason=trail, resized_quantity=permitted,
            )
            # Re-checked against every ceiling, not merely against the one that bound. A resize
            # that still violated another constraint is the defect this whole restructure exists
            # to make impossible, so it is asserted rather than assumed.
            final = ruled.resulting_intent.quantity
            breached = [n for n, cap, _ in ceilings if final > cap]
            if breached:  # pragma: no cover - unreachable while `min` is correct
                return apply_constraint(
                    intent, verdict=ConstitutionVerdict.REJECT,
                    binding_constraint=name,
                    reason=(
                        f"{trail} — the narrowed size {final} still breaches "
                        f"{', '.join(breached)}; refusing rather than approving an order that "
                        f"cannot be shown to satisfy every mandatory constraint"
                    ),
                )
            return ruled

        # "none" now means exactly what it says: **nothing bound.** Every gate that could run,
        # ran and allowed — and `unpriced` names any that could not, so an all-clear never
        # silently stands in for an unevaluated cap.
        return apply_constraint(
            intent, verdict=ConstitutionVerdict.ALLOW,
            binding_constraint="none",
            reason="within every configured limit" + unpriced,
        )


DIMENSIONS: tuple[str, ...] = (
    "order_size",
    "gross_exposure",
    "signed_exposure",
    "hedge_integrity",
    "margin_usage",
    "factor_exposure",
    "scenario_loss",
    "liquidation_cost",
    "per_symbol_underperformance",
)
"""Foundation 5's eight safety dimensions, plus the ninth (`per_symbol_underperformance`, added
2026-09-15 alongside `desk.book.Book.symbol_realized_pnl_since`) — the ARGUS-native counterpart to
freqtrade's ``LowProfitPairs`` — in the order named in the product plan / added to the chain."""


_HUGE = Decimal("1e30")
"""Effectively-infinite ceiling for ablating a notional/ratio cap. Not ``Decimal("Infinity")``:
some downstream arithmetic (``f"{x:.1f}"`` formatting in the gate's own reason strings) raises on
a literal infinity, and a cap ten orders of magnitude above anything `QUANTITIES` sweeps is
functionally unreachable without that fragility."""


def _book_without_hedge_links(book: Any) -> Any:
    """A copy of ``book`` with every :class:`~argus.desk.book.HedgeLink` removed, everything else
    (positions, balances, reservations, the margin snapshot) identical.

    `Book` is not a frozen dataclass, so `dataclasses.replace` does not apply — its state is
    plain mutable attributes, copied here rather than constructed through fills a second time,
    which would risk the copy silently drifting from the original as `Position.apply_fill`'s
    logic evolves. A shallow copy of the object itself (not a fresh ``Book()``) keeps this module
    free of the desk layer, so it can sit with the Constitution in the risk layer.
    """
    clone = copy.copy(book)
    clone.positions = dict(book.positions)
    clone.balances = dict(book.balances)
    clone.reservations = dict(book.reservations)
    clone.margin = book.margin
    clone.hedge_links = []
    return clone


def _ablate(dimension: str, baseline: ConstitutionPolicy) -> ConstitutionPolicy:
    """One policy identical to ``baseline`` except that ``dimension`` cannot bind.

    Raises on an unknown dimension rather than silently returning ``baseline`` unchanged — a typo
    here would otherwise report a gate as "inert" because it was never actually ablated, the exact
    false negative this module exists to prevent.
    """
    if dimension == "order_size":
        return replace(baseline, max_position_notional=_HUGE)
    if dimension == "gross_exposure":
        return replace(baseline, max_gross_exposure_notional=_HUGE)
    if dimension == "signed_exposure":
        return replace(baseline, max_signed_exposure_notional=_HUGE)
    if dimension == "hedge_integrity":
        book = baseline.book
        return replace(baseline, book=None if book is None else _book_without_hedge_links(book))
    if dimension == "margin_usage":
        return replace(baseline, max_margin_usage_ratio=_HUGE)
    if dimension == "factor_exposure":
        return replace(baseline, factor_exposure_limits={})
    if dimension == "scenario_loss":
        return replace(baseline, max_scenario_loss_pct=None)
    if dimension == "liquidation_cost":
        return replace(baseline, max_liquidation_cost_bps=None)
    if dimension == "per_symbol_underperformance":
        return replace(baseline, min_symbol_realized_pnl=None)
    raise ValueError(f"unknown dimension {dimension!r}; known: {', '.join(DIMENSIONS)}")


def ablated_variants(
    baseline: ConstitutionPolicy, *, dimensions: Sequence[str] = DIMENSIONS,
) -> dict[str, ConstitutionPolicy]:
    """One gate-ablated variant per dimension, keyed by name — the ``ablated_constitutions``
    argument :meth:`agents.desk.TradingDesk.run` expects."""
    return {dimension: _ablate(dimension, baseline) for dimension in dimensions}
