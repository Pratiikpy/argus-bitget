"""The background-agent zoo — the participants that make a simulated book behave like a book.

Rebuilt from ``jpmorganchase/abides-jpmc-public`` (**BSD 3-Clause**, © 2021 J.P. Morgan Chase).
Three agents carry almost all of the realism, and each answers a different question:

============================  ====================================================================
Agent                         What it contributes
============================  ====================================================================
:class:`NoiseAgent`           Baseline order flow with no view. ``agents/noise_agent.py``
:class:`ValueAgent`           Informed flow that pulls price toward a fundamental.
                              ``agents/value_agent.py``
:class:`AdaptiveMarketMaker`  The depth everything else trades against. Chakraborty-Kearns ladder,
                              sized as a percentage of recent volume.
                              ``agents/market_makers/adaptive_market_maker_agent.py``
============================  ====================================================================

**Why a market maker is not optional.** Without one there is no depth, so every order walks an
empty book and the simulation produces impact numbers that are pure artefact. ABIDES's own note on
``AdaptiveMarketMakerAgent`` is "critically important for market depth", and that is the agent whose
parameters set whether a simulated session looks like RTH or like 3am.

**Determinism is a requirement, not a nicety.** Every agent draws from a seeded generator passed in
by the kernel. A stress test whose result changes between runs cannot be used to argue anything, and
ABIDES makes the same commitment (``4.5``).

**Where ours departs from theirs, and why.** ABIDES's NoiseAgent wakes exactly once
(``noise_agent.py:38``) and is then inert; a population of them is built by instantiating thousands
with staggered wake times. That is a poor fit for a simulator meant to run inside a test suite, so
ours wakes repeatedly with the same per-wakeup behaviour. The distribution of flow is equivalent;
the object count is three orders of magnitude smaller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from random import Random

from argus.sim.book import Execution, Liquidity, Order, OrderBook, Side

_ZERO = Decimal("0")


class Agent(ABC):
    """A participant. Wakes, looks at the book, and may act."""

    def __init__(self, agent_id: str, *, cash: Decimal = Decimal("1000000")) -> None:
        self.agent_id = agent_id
        self.cash = cash
        self.position = _ZERO
        self.fills: list[Execution] = []

    @abstractmethod
    def wakeup(self, book: OrderBook, rng: Random) -> list[Order]:
        """Return the orders to submit this wakeup. An empty list is a valid decision."""

    def on_fill(self, execution: Execution) -> None:
        """Book a fill. Fees are *not* applied here — the kernel settles them.

        Deliberate: an agent that nets its own fees can quietly use a different rate from the one
        the run is scored on, which is the divergence that makes a simulated maker look profitable.
        """
        self.fills.append(execution)
        signed = execution.quantity if execution.side is Side.BUY else -execution.quantity
        self.position += signed
        self.cash -= signed * execution.price

    def mark_to_market(self, price: Decimal) -> Decimal:
        return self.cash + self.position * price

    @property
    def maker_fill_rate(self) -> Decimal:
        """Share of this agent's fills that were passive.

        The number that says whether a strategy actually earned the maker rate it assumed. Our own
        standing rules record a sim that returned +90% on assumed limit fills against -0.45% in
        replay; this is where that assumption becomes visible.
        """
        if not self.fills:
            return _ZERO
        maker = sum(1 for f in self.fills if f.liquidity is Liquidity.MAKER)
        return Decimal(maker) / Decimal(len(self.fills))


class NoiseAgent(Agent):
    """Order flow with no view. ``agents/noise_agent.py``.

    Buys and sells with equal probability, crossing the spread often enough to generate trades.
    Contributes no information — which is the point: a book in which every participant is informed
    has no one to trade against.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        min_size: Decimal = Decimal("20"),
        max_size: Decimal = Decimal("50"),
        aggression: float = 0.5,
        cash: Decimal = Decimal("1000000"),
    ) -> None:
        super().__init__(agent_id, cash=cash)
        # 20-50 shares is ABIDES's own default when no size model is supplied
        # (`noise_agent.py:55`).
        self._min, self._max = min_size, max_size
        self._aggression = aggression

    def wakeup(self, book: OrderBook, rng: Random) -> list[Order]:
        mid = book.mid
        if mid is None:
            return []
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        size = Decimal(rng.randint(int(self._min), int(self._max)))

        if rng.random() < self._aggression:
            # Cross the spread: price far enough through to take the touch.
            touch = book.best_ask if side is Side.BUY else book.best_bid
            if touch is None:
                return []
            price = touch
        else:
            # Rest one tick behind the touch.
            touch = book.best_bid if side is Side.BUY else book.best_ask
            if touch is None:
                return []
            price = touch - book.tick_size if side is Side.BUY else touch + book.tick_size
        return [Order(agent_id=self.agent_id, side=side, price=price, quantity=size)]


class ValueAgent(Agent):
    """Informed flow, trading toward a fundamental. ``agents/value_agent.py``.

    ABIDES estimates the fundamental with a Kalman-style filter over a noisy observation of the
    mid, with a mean-reverting prior:

        estimate = (1 - kappa) * observed + kappa * r_bar

    ``kappa`` is the reversion strength (ABIDES default 0.05) and ``r_bar`` the prior on true value.
    The agent buys when the market is below its estimate and sells when above, resting
    ``depth_spread`` ticks off the mid and aggressing the spread with probability ``percent_aggr``
    (``value_agent.py:61``, default 0.1).

    **This agent is what makes price discovery exist in the simulation** — and therefore what lets
    us simulate its *absence*. Our measured finding is that discovery attenuates roughly 7x when
    the anchor market is shut, and the way to reproduce that here is to thin this population rather
    than to freeze the price, because the real book does not freeze.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        r_bar: Decimal,
        kappa: Decimal = Decimal("0.05"),
        sigma_n: Decimal = Decimal("0.5"),
        percent_aggressive: float = 0.1,
        depth_spread: int = 2,
        size: Decimal = Decimal("40"),
        cash: Decimal = Decimal("1000000"),
    ) -> None:
        super().__init__(agent_id, cash=cash)
        if not _ZERO <= kappa <= Decimal("1"):
            raise ValueError(f"kappa={kappa} must be in [0, 1]")
        self.r_bar = r_bar
        self._kappa = kappa
        self._sigma_n = sigma_n
        self._aggr = percent_aggressive
        self._depth = depth_spread
        self._size = size

    def estimate(self, observed_mid: Decimal, rng: Random) -> Decimal:
        """Noisy observation, shrunk toward the prior. ``value_agent.py``."""
        noise = Decimal(str(rng.gauss(0.0, float(self._sigma_n))))
        observed = observed_mid + noise
        return (Decimal("1") - self._kappa) * observed + self._kappa * self.r_bar

    def wakeup(self, book: OrderBook, rng: Random) -> list[Order]:
        mid = book.mid
        if mid is None:
            return []
        fair = self.estimate(mid, rng)
        if fair == mid:
            return []

        side = Side.BUY if fair > mid else Side.SELL
        aggressive = rng.random() < self._aggr

        if aggressive:
            touch = book.best_ask if side is Side.BUY else book.best_bid
            if touch is None:
                return []
            price = touch
        else:
            offset = book.tick_size * self._depth
            price = mid - offset if side is Side.BUY else mid + offset
            price = self._to_tick(price, book.tick_size)
        return [Order(agent_id=self.agent_id, side=side, price=price, quantity=self._size)]

    @staticmethod
    def _to_tick(price: Decimal, tick: Decimal) -> Decimal:
        return (price / tick).to_integral_value() * tick


@dataclass
class LadderConfig:
    """Chakraborty-Kearns ladder parameters, named as ABIDES names them."""

    pov: Decimal = Decimal("0.05")
    """Fraction of recent transacted volume placed at each level (`adaptive_...py:75`)."""

    min_order_size: Decimal = Decimal("20")
    num_ticks: int = 5
    """Price levels per side (`:88`)."""

    level_spacing: int = 1
    """Ticks between levels (`:89`)."""

    skew_beta: Decimal = _ZERO
    """Inventory skew (`:108`, default 0 = no skew)."""


class AdaptiveMarketMaker(Agent):
    """The ladder that provides the book's depth. ``adaptive_market_maker_agent.py``.

    Cancels and reposts a ladder of limit orders each wakeup, sized as a percentage of recent
    transacted volume. ABIDES calls this agent "critically important for market depth", and it is
    the one whose parameters decide whether a simulated session looks like RTH or like 3am — which
    makes it the lever for reproducing our measured off-hours regime.

    **Inventory skew is implemented and off by default**, matching ABIDES. With ``skew_beta > 0``
    the ladder shifts away from the side the agent is already long, which is the mechanism that
    stops a market maker accumulating an unbounded position — and its absence is why a naive
    simulated maker ends a long run with a position the size of the day's volume.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        config: LadderConfig | None = None,
        cash: Decimal = Decimal("10000000"),
    ) -> None:
        super().__init__(agent_id, cash=cash)
        self.config = config or LadderConfig()
        self._last_volume = _ZERO

    def wakeup(self, book: OrderBook, rng: Random) -> list[Order]:
        book.cancel_all(self.agent_id)

        reference = book.mid or book.last_trade
        if reference is None:
            return []

        cfg = self.config
        recent = book.traded_volume - self._last_volume
        self._last_volume = book.traded_volume
        size = max(cfg.min_order_size, (recent * cfg.pov).to_integral_value())

        # Inventory skew: shift the ladder against the position we already hold.
        skew = _ZERO
        if cfg.skew_beta > 0 and self.position != 0:
            skew = -(self.position * cfg.skew_beta * book.tick_size)

        centre = self._to_tick(reference + skew, book.tick_size)
        orders: list[Order] = []
        for i in range(1, cfg.num_ticks + 1):
            offset = book.tick_size * i * cfg.level_spacing
            orders.append(Order(agent_id=self.agent_id, side=Side.BUY,
                                price=centre - offset, quantity=size))
            orders.append(Order(agent_id=self.agent_id, side=Side.SELL,
                                price=centre + offset, quantity=size))
        return orders

    @staticmethod
    def _to_tick(price: Decimal, tick: Decimal) -> Decimal:
        return (price / tick).to_integral_value() * tick


@dataclass
class ScriptedAgent(Agent):
    """An agent that submits exactly what it is told, for testing a strategy against the book.

    This is how an ARGUS decision enters the simulation: the desk produces an intent, and this
    agent places it. It has no behaviour of its own, which is the point — anything it does is
    attributable to the strategy under test rather than to the simulator.
    """

    queue: list[Order] = field(default_factory=list)

    def __init__(self, agent_id: str, *, cash: Decimal = Decimal("1000000")) -> None:
        super().__init__(agent_id, cash=cash)
        self.queue = []

    def enqueue(self, order: Order) -> None:
        self.queue.append(order)

    def wakeup(self, book: OrderBook, rng: Random) -> list[Order]:
        pending, self.queue = self.queue, []
        return pending


__all__ = [
    "AdaptiveMarketMaker",
    "Agent",
    "LadderConfig",
    "NoiseAgent",
    "ScriptedAgent",
    "ValueAgent",
]
