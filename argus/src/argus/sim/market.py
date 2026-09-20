"""Running a simulated session — and measuring what leaving a position actually costs.

This is the module that turns :mod:`argus.sim` from a matching engine into an answer. It populates
a book with the ABIDES agent zoo, lets it settle, and then asks the question a stress test actually
needs answered:

    **If I had to get out of this position right now, what would it cost?**

The field norm — and, until this module existed, our own Track-3 workbench — is to *assume* that
answer: take a round-trip fee and divide it by a liquidity multiplier. That produces a number that
moves in the right direction and is otherwise invented. Here the position is liquidated against a
real book, the fills are whatever the book actually had, and the cost is the measured difference
between the mid at the moment of the decision and the volume-weighted price actually achieved.

**Why this matters more on this venue than most.** Our measured off-hours regime runs near a third
of RTH depth for 65 hours a week. On a thin book a large exit does not pay a slightly worse price —
it walks several levels and pays a much worse one, and the relationship is not linear. An assumed
exit cost cannot express that. A simulated one finds it without being told.

Determinism is a hard requirement: every draw comes from a seeded generator, so a stress number can
be argued about rather than merely quoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from random import Random

from argus.sim.agents import (
    AdaptiveMarketMaker,
    Agent,
    LadderConfig,
    NoiseAgent,
    ValueAgent,
)
from argus.sim.book import BookError, Liquidity, Order, OrderBook, Side

_ZERO = Decimal("0")
_BPS = Decimal("10000")


class MarketError(ValueError):
    """The simulated session was asked for something it cannot answer honestly."""


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """How liquid the simulated session is.

    ``depth_multiplier`` is the one knob that matters: 1.0 is RTH, and our measured off-hours
    regime is roughly 0.33. It scales the market maker's ladder size and the number of noise
    traders, which together are what depth actually *is*.
    """

    reference_price: Decimal = Decimal("100.00")
    tick_size: Decimal = Decimal("0.01")
    depth_multiplier: Decimal = Decimal("1")
    noise_agents: int = 6
    ladder_levels: int = 8
    base_ladder_size: Decimal = Decimal("200")
    warmup_wakeups: int = 30
    seed: int = 20260912

    def __post_init__(self) -> None:
        if self.depth_multiplier <= 0:
            raise MarketError(f"depth_multiplier={self.depth_multiplier} must be positive")
        if self.reference_price <= 0:
            raise MarketError(f"reference_price={self.reference_price} must be positive")


@dataclass
class SimulatedSession:
    """A populated book plus the agents maintaining it."""

    book: OrderBook
    agents: list[Agent]
    config: SessionConfig
    rng: Random

    @property
    def depth(self) -> Decimal:
        """Total quantity across the top five levels of both sides."""
        return (self.book.total_depth(Side.BUY, levels=5)
                + self.book.total_depth(Side.SELL, levels=5))

    def step(self, wakeups: int = 1) -> None:
        """Wake every agent, submit what they produce, and **route the fills back to them**.

        The routing is not optional and its absence was a real bug here: without it no agent ever
        learns it was filled, so the market maker's position stays at zero, inventory skew can
        never engage, and `maker_fill_rate` reads 0 for every participant. A simulated market in
        which nobody knows their own position is not a market. ABIDES does this over a message
        bus; with no bus, the session does it directly.
        """
        by_id = {a.agent_id: a for a in self.agents}
        for _ in range(wakeups):
            for agent in self.agents:
                for order in agent.wakeup(self.book, self.rng):
                    # A wash trade or an off-tick price is refused by the book, as designed; an
                    # agent proposing one is a modelling artefact, not a market event.
                    try:
                        executions = self.book.submit(order)
                    except BookError:
                        continue
                    for execution in executions:
                        counterparty = by_id.get(execution.agent_id)
                        if counterparty is not None:
                            counterparty.on_fill(execution)


def build_session(config: SessionConfig | None = None) -> SimulatedSession:
    """Populate a book and let it settle into a two-sided market.

    The seeding order is deliberate: a single resting quote on each side first, because every agent
    in the zoo needs a midpoint before it will act, and a market maker asked to quote around
    nothing correctly does nothing.
    """
    cfg = config or SessionConfig()
    book = OrderBook("SIM", tick_size=cfg.tick_size)
    rng = Random(cfg.seed)

    # The reference price is snapped to the tick before anything is quoted around it.
    #
    # Found by running the empirical stress test: a shock derived from a real distribution is
    # -0.8271%, not -5%, so the shocked price arrives as 217.3058794275855069521973003 and the
    # book — which enforces tick size, correctly — refuses every seed quote. Every caller until
    # then had passed a round number, so this path had only ever been exercised with prices no
    # measurement produces. Snapping here rather than at the call site because the book built below
    # is what enforces the rule, so this is where a price becomes a *venue* price.
    reference = (cfg.reference_price / cfg.tick_size).to_integral_value() * cfg.tick_size
    if reference <= 0:
        raise MarketError(
            f"a reference price of {cfg.reference_price} rounds to {reference} at tick "
            f"{cfg.tick_size}; a book cannot be seeded around a non-positive price"
        )

    # Seed one quote per side so a mid exists.
    #
    # **The seed is scaled by `depth_multiplier`, and it was not until 2026-09-20.** A flat 10
    # units per side is not "enough to establish a mid" when the caller has asked for a book at 1%
    # of normal depth — it is a floor that silently refills the book the scenario was trying to
    # empty. Together with `max(ladder, 1)` and `max(1, noise_agents * multiplier)` below, a
    # `depth_multiplier` of 0.01 still left roughly 19% of full depth standing.
    #
    # That made `desk/stress.py`'s `venue_outage` scenario unable to fail. Its own stated
    # assumption reads *"near-zero depth stands for no orders accepted. The exit test should fail;
    # that failing is the finding, not a defect."* It did not fail — it exited at 12.22bps, the
    # CHEAPEST of all nine scenarios and cheaper than the 12.45bps normal-market baseline, so the
    # report told a reader that a total venue outage is the best moment to get out, and
    # `survives_all: true` was unearned. Found by an adversarial audit, not by a test.
    #
    # The seed still never rounds to zero, because a book with no quote has no mid and the
    # simulation cannot start at all — but at 0.01x it is now 1 unit a side rather than 10.
    half_spread = cfg.tick_size
    seed_size = max(
        (Decimal("10") * cfg.depth_multiplier).to_integral_value(), Decimal("1")
    )
    book.submit(Order(agent_id="seed", side=Side.BUY,
                      price=reference - half_spread, quantity=seed_size))
    book.submit(Order(agent_id="seed", side=Side.SELL,
                      price=reference + half_spread, quantity=seed_size))

    ladder = (cfg.base_ladder_size * cfg.depth_multiplier).to_integral_value()
    agents: list[Agent] = [
        AdaptiveMarketMaker("mm", config=LadderConfig(
            num_ticks=cfg.ladder_levels,
            min_order_size=max(ladder, Decimal("1")),
        )),
    ]
    # Fewer participants on a thinner book — depth is not only order size, it is how many people
    # are willing to be there at all.
    n_noise = max(1, int(cfg.noise_agents * float(cfg.depth_multiplier)))
    agents.extend(NoiseAgent(f"noise{i}", aggression=0.25) for i in range(n_noise))
    agents.append(ValueAgent("value", r_bar=reference, kappa=Decimal("0.2"),
                             sigma_n=Decimal("0.05"), percent_aggressive=0.15))

    session = SimulatedSession(book=book, agents=agents, config=cfg, rng=rng)
    session.step(cfg.warmup_wakeups)
    return session


@dataclass(frozen=True, slots=True)
class ExitMeasurement:
    """What liquidating actually cost, measured rather than assumed."""

    quantity_requested: Decimal
    quantity_filled: Decimal
    mid_before: Decimal
    vwap: Decimal
    slippage_bps: Decimal
    levels_walked: int
    depth_available: Decimal

    @property
    def fully_exited(self) -> bool:
        return self.quantity_filled >= self.quantity_requested

    @property
    def unfilled(self) -> Decimal:
        return self.quantity_requested - self.quantity_filled

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": str(self.quantity_requested),
            "filled": str(self.quantity_filled),
            "fully_exited": self.fully_exited,
            "mid_before": str(self.mid_before),
            "vwap": str(round(self.vwap, 6)),
            "slippage_bps": str(round(self.slippage_bps, 2)),
            "levels_walked": self.levels_walked,
            "depth_available": str(self.depth_available),
        }


def measure_exit(
    session: SimulatedSession, *, quantity: Decimal, side: Side = Side.SELL
) -> ExitMeasurement:
    """Liquidate ``quantity`` into the book and measure what it cost.

    ``side`` is the side of the *exit* — selling to close a long, buying to close a short.

    The cost reported is **slippage against the mid at the moment of the decision**, in basis
    points, and it deliberately excludes the exchange fee. The fee is a known constant that the
    cost model already charges; what a simulation can tell you and arithmetic cannot is how far
    the book moves underneath you, so that is what this returns.

    An exit that cannot be completed reports ``fully_exited == False`` with the unfilled remainder
    rather than pretending the rest left at the last price. That case is the whole reason to
    simulate: on a thin book a position can be *unexitable at any price*, and no multiplier on a
    fee will ever say so.
    """
    if quantity <= 0:
        raise MarketError(f"quantity={quantity} must be positive")

    mid = session.book.mid
    if mid is None:
        raise MarketError(
            "the book is one-sided, so there is no mid to measure slippage against. A stress "
            "number computed here would be against a price that does not exist."
        )

    opposing = Side.SELL if side is Side.BUY else Side.BUY
    depth = session.book.total_depth(opposing, levels=20)

    # A marketable limit far through the book: the standard way to express "take whatever is
    # there" without inventing a market-order type the engine does not have.
    through = session.book.tick_size * Decimal("100000")
    price = (mid + through) if side is Side.BUY else max(session.book.tick_size, mid - through)
    price = (price / session.book.tick_size).to_integral_value() * session.book.tick_size

    executions = session.book.submit(
        Order(agent_id="liquidator", side=side, price=price, quantity=quantity)
    )
    ours = [e for e in executions if e.agent_id == "liquidator"]
    filled = sum((e.quantity for e in ours), _ZERO)

    if filled <= 0:
        return ExitMeasurement(
            quantity_requested=quantity, quantity_filled=_ZERO, mid_before=mid, vwap=mid,
            slippage_bps=_ZERO, levels_walked=0, depth_available=depth,
        )

    notional = sum((e.price * e.quantity for e in ours), _ZERO)
    vwap = notional / filled
    # Selling below the mid and buying above it are both costs, so the sign is normalised.
    raw = (mid - vwap) if side is Side.SELL else (vwap - mid)
    slippage = (raw / mid) * _BPS

    return ExitMeasurement(
        quantity_requested=quantity,
        quantity_filled=filled,
        mid_before=mid,
        vwap=vwap,
        slippage_bps=slippage,
        levels_walked=len({e.price for e in ours}),
        depth_available=depth,
    )


def exit_cost_bps(
    *, quantity: Decimal, depth_multiplier: Decimal, reference_price: Decimal, seed: int = 20260912
) -> ExitMeasurement:
    """Convenience: build a session at this liquidity and measure the exit in one call.

    This is the entry point the Track-3 stress test uses, so that an exit cost in a stress table is
    a measurement of a simulated book rather than a fee divided by a multiplier.
    """
    session = build_session(SessionConfig(
        reference_price=reference_price, depth_multiplier=depth_multiplier, seed=seed,
    ))
    return measure_exit(session, quantity=quantity, side=Side.SELL)


def maker_taker_split(session: SimulatedSession) -> dict[str, int]:
    """How the session's volume divided between passive and aggressive fills.

    Reported because a simulated market in which nothing ever rests is not a market, and a warmup
    that produced no maker fills has not built a book — it has built a queue of crossing orders.
    """
    counts = {"maker": 0, "taker": 0}
    for agent in session.agents:
        for fill in agent.fills:
            counts["maker" if fill.liquidity is Liquidity.MAKER else "taker"] += 1
    return counts


__all__ = [
    "ExitMeasurement",
    "MarketError",
    "SessionConfig",
    "SimulatedSession",
    "build_session",
    "exit_cost_bps",
    "maker_taker_split",
    "measure_exit",
]
