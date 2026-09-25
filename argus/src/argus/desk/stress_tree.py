"""Second-order stress: a bounded, recursive tree of scenarios grown from the shock a trader asked.

**What the one-pass answer leaves out.** The console's stress answer (`lui/research.py`, the STRESS
branch of ``_run``) makes one pass over the book: the stated shock through each holding's beta, a
hedge sized to that beta, and the realised worst 24-hour window. Every line answers the question
that was asked. None of them asks the next question, which is the one a risk review asks of every
first answer: *and then what?* With the beta hedge on, does it hold through the day that actually
went worst? The name the shock hits hardest through beta: is its own worst day bigger than its beta
share? The stated shock is a round number: has it ever happened, and if not, what does the worst
move that did happen do? The holding that carries the tail: how far must it be trimmed before it
stops carrying it? Each of those is a scenario grown from a *result*, and a one-pass answer cannot
reach it because it has not computed the result yet.

**Taken, from gpt-researcher** (assafelovic/gpt-researcher, Apache-2.0; licence and notice in
``licenses/gpt_researcher-APACHE-2.0.txt``), `gpt_researcher/skills/deep_research.py`:

* The recursive breadth-by-depth tree, ``DeepResearchSkill.deep_research`` (``:380-578``): each
  level runs ``breadth`` branches, then recurses from every branch's result with one level less.
* Children grown from each branch's *result*, not from the root question. Upstream seeds the next
  level with the branch's research goal plus its follow-up questions (``:540-543``), so the tree
  narrows into what the level above found. Here each angle has a follow-up rule that reads the
  figures its node computed and names the scenarios those figures raise.
* The breadth schedule: the next level runs ``max(2, breadth // 2)`` branches per parent, one level
  shallower (``:535-536``).
* Bounded concurrency per level, a semaphore at ``:426`` with a default of two
  (``deep_research_concurrency``, ``:249``); here a pool of at most ``concurrency`` workers.
* Per-branch failure isolation: a branch that raises returns ``None`` and the level filters it
  (``:474-479``, ``:483``). Here a node that raises is recorded as ``failed`` with its reason, and
  its siblings are untouched.
* The two empty-state stops. Upstream stops descent when no query was generated (``:409-417``) and
  when every branch at a level failed (``:499-514``, their issue #1579, added after it looped on
  empty goals in production). Both are here from the start.

**Changed, each deliberately.**

* **The angles are a fixed catalogue, not model-written queries.** Upstream asks the model for
  ``breadth`` query-and-goal pairs at every level (``generate_search_queries``, ``:259-290``).
  A stress angle here has to be *computed* by one of the desk's engines, and an angle a model
  invents has no engine to compute it — its figure would be the model's, which the console never
  allows (`lui/research.py`'s docstring: the model never says a number). The teardown left this
  open (`research/mypr-teardowns/gpt_researcher.md`, open questions) and it is closed this way on
  purpose: the tree is auditable, every run of it on the same history is identical, and it costs
  no token.
* **No learnings.** An upstream branch returns model prose (``process_research_results``,
  ``:346-378``), which `_SYNTHESIS.md` §4 names as the thing not to copy. A node here returns the
  figures an engine computed and one sentence built from those figures.
* **A node budget.** Upstream bounds breadth and depth but not the total, and a model that keeps
  producing follow-ups grows the tree as breadth times the product of the halvings. Here
  ``max_nodes`` caps it, and a scenario past the cap is listed as not run with the reason — the
  overflow message open_deep_research returns for research units past its limit
  (`deep_researcher.py:316-321`, MIT) — never dropped silently.
* **A visited set** keyed by angle and subject, the analogue of upstream's ``visited_urls``
  (``:450-452``, ``:519``): one scenario is never computed twice in one tree.
* **The zero-children stop returns state that exists.** Upstream's guard at ``:409-417`` returns
  ``all_learnings`` before ``all_learnings`` is first assigned at ``:419``, so the one path the
  guard exists for raises ``UnboundLocalError`` instead of stopping. Read in the source on
  2026-09-25, not inferred; the tree's state here is initialised before any stop can return it.
* **Deterministic order.** Nodes are listed in catalogue order, never in completion order, so two
  runs of the same book print the same tree.

**Every figure comes from an engine already in the desk.** ``beta_shock``, ``shut_shock`` and
``history_shock`` are :func:`argus.desk.portfolio.stress_by_beta` (over open-session, shut-session
and open-session hours respectively); ``realised_window``, ``window_driver``, ``idiosyncratic``
and ``hedge_residual`` are :func:`~argus.desk.portfolio.worst_window`; ``tail``, ``tail_trim`` and
``hedged_tail`` are :func:`~argus.desk.portfolio.tail_contributions` (skfolio's CVaR definition
with its exact Euler split); ``session_gap`` is :func:`~argus.desk.portfolio.beta` per session;
``shock_frequency`` is :func:`argus.desk.stress.horizon_moves` and
:func:`~argus.desk.stress.shock_frequency` over the shocked instrument's own path. Nothing is
estimated here that those functions do not already estimate; the tree decides which of them to
run next, and on what.

**Concurrency, measured rather than assumed** (`eval/research_depth.py`). The nodes are arithmetic
over columns already in memory, so under the GIL a thread pool cannot run two of them at once.
Upstream's two-worker default was this module's first default too, and on the live stress
questions it was slower on most books, so the default is now one worker, which runs each level
inline with no pool. The bound stays: a caller whose nodes read the network (an order book, a
daily history) can raise it, and the measured cost of doing so today is recorded in
``data/research_depth_stress.json``.

    python -m argus.desk.stress_tree --book "NVDAUSDT=0.6,AAPLUSDT=0.4" --shock -10
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from argus.desk.portfolio import (
    Shock,
    WorstWindow,
    align,
    beta,
    stress_by_beta,
    tail_contributions,
    worst_window,
)
from argus.desk.stress import (
    StressError,
    closes_from_returns,
    horizon_moves,
    shock_frequency,
)

WINDOW_BARS = 24
"""Horizon of every realised-window and history scenario, in hourly bars: one day, the same window
`desk/portfolio.worst_window` uses for the one-pass answer, so the tree and that line agree."""

DEFAULT_BREADTH = 5
"""First-order angles run at the root. All five by default: each reads a different engine."""

DEFAULT_DEPTH = 3
"""Levels below the root. Three is where the catalogue's follow-up rules run out, so a deeper
setting grows nothing further."""

DEFAULT_MAX_NODES = 16
"""The node budget. The catalogue can produce at most twelve today; the headroom keeps the cap
from biting on a normal book while still bounding a future catalogue that grows faster."""

DEFAULT_CONCURRENCY = 1
"""Workers per level: one, which runs the level inline. Upstream's default is two
(`deep_research.py:249`), and two was this module's first default. Measured on the 61 live stress
questions of `eval/research_depth.py` (2026-09-25): a two-worker pool made the tree slower on 45
of 61 books, median 603ms against 187ms inline, because every node is arithmetic under the GIL and
a pool only adds thread start-up. The bound is kept, and a caller whose nodes read the network can
raise it."""

HEDGE_BETA_FLOOR = 0.2
"""The book beta below which no beta hedge is tested. The same threshold the one-pass answer uses
before it recommends a hedge against a non-benchmark instrument (`lui/research.py`, STRESS
branch), so the tree tests the hedge exactly when the answer would suggest one."""

LOSS_SHARE_MARGIN = 0.02
"""How far a holding's share of the tail must exceed its share of the weight before it is named as
carrying it — the same two-point margin the one-pass answer uses for its loss-share line."""

RISK_BUDGET = 0.25
"""The single-name share the trim scenario targets: `lui/research.RISK_BUDGET`, restated rather
than imported so the desk layer does not import the console."""

SESSION_GAP_MIN = 0.1
"""Absolute gap between open- and shut-session book beta below which the shut-session shock is not
grown: a difference inside estimation noise would read as a finding."""

HISTORY_GAP_PCT = 1.0
"""Percentage points between the stated shock and the most extreme observed move below which the
history-grounded shock adds nothing the stated one did not say."""

TRIM_STEPS = 50
"""Grid resolution of the trim search: the weight is scanned down in fiftieths of its current
value. A grid, not a bisection, because the tail share is not guaranteed monotone in the weight
(the set of worst hours changes as the weight does) and a bisection on a non-monotone function
returns a boundary that may not exist."""

HEDGE_COLUMN = "{symbol} (beta hedge)"
"""Name of the synthetic hedge leg added to the columns: the shocked instrument's own returns,
held short at the book's beta. A separate key so a book that already holds the instrument keeps
its own position distinct from the hedge."""


class StressTreeError(RuntimeError):
    """A tree that cannot be grown honestly from these inputs. Raised rather than approximated."""


def _ticker(symbol: str) -> str:
    """NVDA, not NVDAUSDT; CVX, not CVXSTOCKUSDT — `lui/research._t`, restated for the desk."""
    base = symbol.removesuffix("USDT")
    return base.removesuffix("STOCK") if base.endswith("STOCK") and len(base) > 5 else base


# =============================================================================================
# Inputs
# =============================================================================================


@dataclass(frozen=True)
class StressInputs:
    """Everything every node reads, fixed before the first one runs.

    ``columns`` are aligned on the timestamps every loaded series shares — the same
    :func:`~argus.desk.portfolio.align` over the same ``raw`` the one-pass answer uses, so a beta
    here is the beta there. ``weights`` sum to one less ``cash``: cash dilutes every figure and is
    never scaled away, the console's convention (`lui/research.split_cash`).
    """

    weights: Mapping[str, float]
    cash: float
    shocked: str
    shocked_label: str
    shock_pct: float
    stamps: tuple[datetime, ...]
    columns: Mapping[str, tuple[float, ...]]
    open_rows: tuple[int, ...]
    shocked_series: tuple[tuple[datetime, float], ...]
    """The shocked instrument's own full series, not cut to the book's alignment: the history
    check wants every window the instrument has, not only the ones the book shares."""

    horizon_bars: int = WINDOW_BARS

    @property
    def held(self) -> dict[str, float]:
        return {s: w for s, w in self.weights.items() if w != 0.0 and s in self.columns}

    def held_columns(self, extra: Mapping[str, Sequence[float]] | None = None
                     ) -> dict[str, list[float]]:
        out = {s: list(self.columns[s]) for s in self.held}
        out.update({k: list(v) for k, v in (extra or {}).items()})
        return out

    def session_columns(self, *, open_session: bool) -> dict[str, list[float]]:
        rows = set(self.open_rows)
        keep = [i for i in range(len(self.stamps)) if (i in rows) is open_session]
        return {k: [v[i] for i in keep] for k, v in self.columns.items()}

    @property
    def hedge_column(self) -> str:
        return HEDGE_COLUMN.format(symbol=_ticker(self.shocked))


def prepare(
    raw: Mapping[str, Mapping[datetime, float]],
    *,
    is_open: Callable[[datetime], bool],
    weights: Mapping[str, float],
    shocked: str,
    shock_pct: float,
    cash: float = 0.0,
    shocked_label: str | None = None,
    horizon_bars: int = WINDOW_BARS,
) -> StressInputs:
    """Align the loaded returns once and fix the inputs every node reads.

    Refuses a shock of zero (no direction to grow anything from), a shocked instrument with no
    returns, and a book none of whose holdings has any.
    """
    if shock_pct == 0:
        raise StressTreeError("a shock of zero has no direction to grow scenarios from")
    if shocked not in raw or not raw[shocked]:
        raise StressTreeError(f"no returns were loaded for the shocked instrument {shocked}")
    stamps, columns = align(raw)
    if not stamps:
        raise StressTreeError("the loaded series share no timestamp, so nothing aligns")
    held = {s: float(w) for s, w in weights.items() if float(w) != 0.0}
    if not any(s in columns for s in held):
        raise StressTreeError("none of the book's holdings has returns in the loaded history")
    return StressInputs(
        weights=held, cash=float(cash), shocked=shocked,
        shocked_label=shocked_label or _ticker(shocked), shock_pct=float(shock_pct),
        stamps=tuple(stamps), columns={k: tuple(v) for k, v in columns.items()},
        open_rows=tuple(i for i, t in enumerate(stamps) if is_open(t)),
        shocked_series=tuple(sorted(raw[shocked].items())), horizon_bars=horizon_bars,
    )


# =============================================================================================
# The catalogue: what each angle computes, and which scenarios its result raises
# =============================================================================================


@dataclass(frozen=True)
class Probe:
    """One scenario to compute: an angle, the name it is about, and the figures it inherits."""

    angle: str
    subject: str
    params: tuple[tuple[str, float], ...] = ()

    def param(self, name: str) -> float:
        for key, value in self.params:
            if key == name:
                return value
        raise KeyError(name)


class _Empty(Exception):
    """An engine returned no estimate (too few observations, nothing estimable). A signal that the
    engine had nothing to say, not a failure: the node is recorded ``empty`` with this reason."""


Figures = dict[str, float]
Compute = Callable[[StressInputs, Probe], tuple[str, Figures]]
FollowUps = Callable[[StressInputs, Probe, Figures], list[Probe]]


def _pct(value: float) -> str:
    return f"{value:+.2f}%"


def _book_beta(inputs: StressInputs, cols: Mapping[str, Sequence[float]]) -> float | None:
    bench = cols.get(inputs.shocked)
    if bench is None:
        return None
    total, seen = 0.0, False
    for symbol, weight in inputs.held.items():
        b = beta(cols.get(symbol, []), bench)
        if b is not None:
            total += weight * b
            seen = True
    return total if seen else None


def _shock_move(inputs: StressInputs, cols: Mapping[str, Sequence[float]], move: float,
                label: str) -> tuple[float, tuple[str, float] | None]:
    outcome = stress_by_beta(weights=inputs.held, columns=cols, benchmark=cols[inputs.shocked],
                             shocks=[Shock(label, move)])[0]
    if outcome.portfolio_move_pct is None:
        raise _Empty(outcome.reason or "no holding had an estimable beta")
    return outcome.portfolio_move_pct, outcome.worst_position


# --- beta_shock: the stated shock, the anchor every other scenario is read against -----------


def _beta_shock(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    cols = inputs.session_columns(open_session=True)
    move, worst = _shock_move(inputs, cols, inputs.shock_pct, "stated")
    book_beta = _book_beta(inputs, cols)
    figures: Figures = {"book_move_pct": move, "book_beta": book_beta or 0.0,
                        "open_hours": float(len(inputs.open_rows))}
    text = (f"Stated shock through beta: {inputs.shocked_label} {inputs.shock_pct:+g}% moves the "
            f"book {_pct(move)} (book beta {book_beta or 0.0:.2f}, open-session hours)")
    if worst is not None:
        figures["hardest_hit_move_pct"] = worst[1]
        text += f"; hardest hit {_ticker(worst[0])} {_pct(worst[1])}"
    return text + ".", figures


def _beta_shock_children(inputs: StressInputs, probe: Probe, figures: Figures) -> list[Probe]:
    kids: list[Probe] = []
    book_beta = figures.get("book_beta", 0.0)
    if abs(book_beta) >= HEDGE_BETA_FLOOR:
        kids.append(Probe("hedge_residual", inputs.shocked, (("hedge_weight", -book_beta),)))
    cols = inputs.session_columns(open_session=True)
    bench = cols[inputs.shocked]
    shares: list[tuple[float, str, float]] = []
    for symbol, weight in inputs.held.items():
        if symbol == inputs.shocked:
            continue  # the shocked instrument has no move of its own apart from the shock
        b = beta(cols.get(symbol, []), bench)
        if b is not None:
            shares.append((weight * b * inputs.shock_pct, symbol, b))
    if shares:
        implied, symbol, b = min(shares)  # the holding the stated shock costs the book most
        if implied < 0:
            kids.append(Probe("idiosyncratic", symbol, (("beta", b), ("implied_pct", implied))))
    return kids


# --- hedge_residual / hedged_tail: does the hedge the answer suggests survive what happened ----


def _hedged(inputs: StressInputs, hedge_weight: float
            ) -> tuple[dict[str, float], dict[str, list[float]]]:
    weights = dict(inputs.held)
    weights[inputs.hedge_column] = hedge_weight
    cols = inputs.held_columns({inputs.hedge_column: inputs.columns[inputs.shocked]})
    return weights, cols


def _hedge_residual(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    hedge_weight = probe.param("hedge_weight")
    bare = worst_window(weights=inputs.held, columns=inputs.held_columns(),
                        bars=inputs.horizon_bars)
    weights, cols = _hedged(inputs, hedge_weight)
    hedged = worst_window(weights=weights, columns=cols, bars=inputs.horizon_bars)
    if bare.move_pct is None or hedged.move_pct is None:
        raise _Empty(bare.reason or hedged.reason)
    side = "short" if hedge_weight < 0 else "long"
    text = (f"With the beta hedge on ({side} {inputs.shocked_label} worth "
            f"{abs(hedge_weight):.0%} of the book, sized to the open-session beta), the worst "
            f"realised {inputs.horizon_bars}-hour window goes from {_pct(bare.move_pct)} to "
            f"{_pct(hedged.move_pct)}")
    if hedged.move_pct < bare.move_pct:
        text += " — the hedge would have made the worst day worse, not better"
    elif hedged.move_pct > bare.move_pct:
        text += "; what remains is what the holdings did on their own"
    figures = {"unhedged_pct": bare.move_pct, "hedged_pct": hedged.move_pct,
               "hedge_weight": hedge_weight}
    return text + ".", figures


def _hedge_residual_children(inputs: StressInputs, probe: Probe, figures: Figures
                             ) -> list[Probe]:
    return [Probe("hedged_tail", inputs.shocked, (("hedge_weight", figures["hedge_weight"]),))]


def _hedged_tail(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    hedge_weight = probe.param("hedge_weight")
    bare = tail_contributions(inputs.held, inputs.held_columns())
    weights, cols = _hedged(inputs, hedge_weight)
    hedged = tail_contributions(weights, cols)
    if bare is None or hedged is None or bare.cvar <= 0:
        raise _Empty("too few aligned hours for a tail estimate")
    cut = 1.0 - hedged.cvar / bare.cvar
    text = (f"In the worst 5% of hours the hedged book loses {hedged.cvar:.2%} an hour on "
            f"average, against {bare.cvar:.2%} unhedged — "
            + (f"the hedge takes out {cut:.0%} of the tail." if cut > 0 else
               "the hedge does not reduce the tail at all."))
    return text, {"unhedged_cvar": bare.cvar, "hedged_cvar": hedged.cvar,
                  "tail_cut_share": cut}


# --- idiosyncratic: the hardest-hit name, on its own worst day --------------------------------


def _idiosyncratic(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    symbol = probe.subject
    weight = inputs.held[symbol]
    own = worst_window(weights={symbol: weight}, columns={symbol: list(inputs.columns[symbol])},
                       bars=inputs.horizon_bars)
    if own.move_pct is None or not own.contributors:
        raise _Empty(own.reason)
    implied = probe.param("implied_pct")
    name_move = own.contributors[0][1]
    text = (f"{_ticker(symbol)} on its own: its worst realised {inputs.horizon_bars} hours cost "
            f"the book {_pct(own.move_pct)} ({_ticker(symbol)} itself {_pct(name_move)}), "
            f"against {_pct(implied)} that its beta share of the stated shock implies")
    if implied < 0 and own.move_pct < implied:
        text += (f" — its own worst day was {own.move_pct / implied:.1f}x its beta share, risk "
                 f"no index hedge touches")
    return text + ".", {"own_worst_book_pct": own.move_pct, "own_worst_name_pct": name_move,
                        "beta_implied_book_pct": implied}


# --- realised_window / window_driver: what actually happened, and what would have helped ------


def _realised_window(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    ww = worst_window(weights=inputs.held, columns=inputs.held_columns(),
                      bars=inputs.horizon_bars)
    if ww.move_pct is None or ww.start_index is None:
        raise _Empty(ww.reason)
    start = ww.start_index
    end_at = inputs.stamps[start + inputs.horizon_bars - 1]
    rows = set(inputs.open_rows)
    open_hours = sum(1 for i in range(start, start + inputs.horizon_bars) if i in rows)
    drivers = ", ".join(f"{_ticker(s)} {_pct(v)}" for s, v in ww.contributors[:3])
    days = max(1, round((inputs.stamps[-1] - inputs.stamps[0]).total_seconds() / 86400))
    text = (f"Worst realised {inputs.horizon_bars} hours for this book in the {days}-day "
            f"history: {_pct(ww.move_pct)}, ending {end_at:%d %b %H:%M} UTC, {open_hours} of "
            f"{inputs.horizon_bars} hours with the US market open; driven by {drivers}.")
    return text, {"move_pct": ww.move_pct, "start_index": float(start),
                  "open_hours": float(open_hours)}


def _realised_window_children(inputs: StressInputs, probe: Probe, figures: Figures
                              ) -> list[Probe]:
    if len(inputs.held) < 2:
        return []  # halving the only holding halves the loss exactly: arithmetic, not a finding
    start = int(figures["start_index"])
    losses = []
    for symbol, weight in inputs.held.items():
        growth = 1.0
        for value in inputs.columns[symbol][start:start + inputs.horizon_bars]:
            growth *= 1.0 + value
        losses.append((weight * (growth - 1.0), symbol))
    worst_loss, symbol = min(losses)
    if worst_loss >= 0:
        return []
    return [Probe("window_driver", symbol, (("was_pct", figures["move_pct"]),))]


def _window_driver(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    symbol = probe.subject
    weights = dict(inputs.held)
    weights[symbol] = weights[symbol] / 2.0
    ww = worst_window(weights=weights, columns=inputs.held_columns(), bars=inputs.horizon_bars)
    if ww.move_pct is None:
        raise _Empty(ww.reason)
    was = probe.param("was_pct")
    text = (f"Halving {_ticker(symbol)} (to {weights[symbol]:.0%} of the book, the rest to cash) "
            f"would have made the worst {inputs.horizon_bars} hours {_pct(ww.move_pct)} instead "
            f"of {_pct(was)}.")
    return text, {"halved_move_pct": ww.move_pct, "was_pct": was,
                  "halved_weight": weights[symbol]}


# --- tail / tail_trim: who carries the worst hours, and how far to cut them --------------------


def _shares(weights: Mapping[str, float], cols: Mapping[str, Sequence[float]]
            ) -> tuple[float, dict[str, float]] | None:
    tail = tail_contributions(weights, cols)
    if tail is None or tail.cvar <= 0:
        return None
    return tail.cvar, {s: v / tail.cvar for s, v in tail.contributions}


def _tail_top(inputs: StressInputs) -> tuple[float, dict[str, float], str, float] | None:
    """CVaR, every holding's share of it, the holding whose share most exceeds its share of the
    invested weight, and that weight share. None when the tail cannot be estimated."""
    read = _shares(inputs.held, inputs.held_columns())
    if read is None:
        return None
    cvar, shares = read
    gross = sum(abs(w) for w in inputs.held.values())
    excess = {s: shares[s] - abs(w) / gross for s, w in inputs.held.items() if s in shares}
    top = max(excess, key=lambda s: excess[s])
    return cvar, shares, top, abs(inputs.held[top]) / gross


def _tail(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    read = _tail_top(inputs)
    if read is None:
        raise _Empty("too few aligned hours for a tail estimate, or no loss in its worst hours")
    cvar, shares, top, weight_share = read
    text = (f"Tail: in the worst 5% of hours the book lost {cvar:.2%} an hour on average; "
            f"{_ticker(top)} carries {shares[top]:.0%} of that on {weight_share:.0%} of the "
            f"invested weight.")
    return text, {"cvar": cvar, "top_share": shares[top], "top_weight_share": weight_share}


def _tail_children(inputs: StressInputs, probe: Probe, figures: Figures) -> list[Probe]:
    if len(inputs.held) < 2 or figures["top_share"] <= RISK_BUDGET:
        return []
    if figures["top_share"] - figures["top_weight_share"] < LOSS_SHARE_MARGIN:
        return []
    read = _tail_top(inputs)
    if read is None or inputs.held[read[2]] <= 0:
        return []  # a short carrying the tail is cut by covering it, which is not a trim
    return [Probe("tail_trim", read[2])]


def _tail_trim(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    symbol = probe.subject
    current = inputs.held[symbol]
    cols = inputs.held_columns()
    for step in range(TRIM_STEPS + 1):
        target = current * (1.0 - step / TRIM_STEPS)
        weights = dict(inputs.held)
        weights[symbol] = target
        read = _shares({s: w for s, w in weights.items() if w != 0.0}, cols)
        if read is None:
            continue
        cvar, shares = read
        share = shares.get(symbol, 0.0)
        if share <= RISK_BUDGET:
            text = (f"Trimming {_ticker(symbol)} from {current:.0%} to about {target:.0%} of the "
                    f"book (the difference to cash) brings its share of the tail to {share:.0%}, "
                    f"inside the {RISK_BUDGET:.0%} single-name budget; the book's tail loss "
                    f"becomes {cvar:.2%} an hour.")
            return text, {"from_weight": current, "to_weight": target, "share_after": share,
                          "cvar_after": cvar}
    raise _Empty(f"no weight of {_ticker(symbol)} on the scanned grid brought its tail share "
                 f"under {RISK_BUDGET:.0%}")


# --- session_gap / shut_shock: the same shock, landing while the anchor is shut ----------------


def _session_gap(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    open_beta = _book_beta(inputs, inputs.session_columns(open_session=True))
    shut_beta = _book_beta(inputs, inputs.session_columns(open_session=False))
    if open_beta is None or shut_beta is None:
        raise _Empty("the book's beta could not be estimated in both sessions")
    n_open = len(inputs.open_rows)
    n_shut = len(inputs.stamps) - n_open
    text = (f"Session: the book's beta to {inputs.shocked_label} is {open_beta:.2f} with the US "
            f"market open ({n_open} hours) and {shut_beta:.2f} while it is shut ({n_shut} "
            f"hours).")
    return text, {"open_beta": open_beta, "shut_beta": shut_beta}


def _session_gap_children(inputs: StressInputs, probe: Probe, figures: Figures) -> list[Probe]:
    gap = abs(figures["shut_beta"] - figures["open_beta"])
    if gap < max(SESSION_GAP_MIN, 0.25 * abs(figures["open_beta"])):
        return []
    return [Probe("shut_shock", inputs.shocked)]


def _shut_shock(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    shut, _ = _shock_move(inputs, inputs.session_columns(open_session=False), inputs.shock_pct,
                          "stated, shut session")
    opened, _ = _shock_move(inputs, inputs.session_columns(open_session=True), inputs.shock_pct,
                            "stated, open session")
    text = (f"If the {inputs.shocked_label} {inputs.shock_pct:+g}% move lands while the US market "
            f"is shut, the book moves {_pct(shut)} through its shut-session betas, against "
            f"{_pct(opened)} in the open session.")
    return text, {"shut_move_pct": shut, "open_move_pct": opened}


# --- shock_frequency / history_shock: has the stated shock ever happened -----------------------


def _shock_frequency(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    closes = closes_from_returns(inputs.shocked_series)
    moves = horizon_moves(closes, bars=inputs.horizon_bars)
    try:
        read = shock_frequency(moves, shock_pct=Decimal(str(inputs.shock_pct)),
                               horizon_bars=inputs.horizon_bars)
    except StressError as exc:
        raise _Empty(str(exc)) from exc
    series = inputs.shocked_series
    days = max(1, round((series[-1][0] - series[0][0]).total_seconds() / 86400))
    worse = "worse" if inputs.shock_pct < 0 else "more"
    text = (f"History check: {inputs.shocked_label} moved {inputs.shock_pct:+g}% or {worse} over "
            f"{inputs.horizon_bars} hours in {read.occurrences} of {read.observations} windows in "
            f"the last {days} days (its most extreme: {_pct(float(read.extreme_pct))})")
    if read.beyond_history:
        text += " — the stated shock is beyond everything in this history"
    elif read.last_seen is not None:
        text += f", most recently {read.last_seen:%d %b}"
    return text + ".", {"occurrences": float(read.occurrences),
                        "observations": float(read.observations),
                        "extreme_pct": float(read.extreme_pct)}


def _shock_frequency_children(inputs: StressInputs, probe: Probe, figures: Figures
                              ) -> list[Probe]:
    extreme = figures["extreme_pct"]
    if abs(extreme - inputs.shock_pct) < HISTORY_GAP_PCT or extreme == 0:
        return []
    return [Probe("history_shock", inputs.shocked, (("extreme_pct", extreme),))]


def _history_shock(inputs: StressInputs, probe: Probe) -> tuple[str, Figures]:
    extreme = probe.param("extreme_pct")
    cols = inputs.session_columns(open_session=True)
    at_extreme, _ = _shock_move(inputs, cols, extreme, "worst observed")
    stated, _ = _shock_move(inputs, cols, inputs.shock_pct, "stated")
    text = (f"At the most extreme {inputs.horizon_bars}-hour move {inputs.shocked_label} actually "
            f"made in this history ({_pct(extreme)}), the book moves {_pct(at_extreme)} through "
            f"beta — against {_pct(stated)} at the stated {inputs.shock_pct:+g}%.")
    return text, {"extreme_pct": extreme, "book_move_at_extreme_pct": at_extreme,
                  "book_move_at_stated_pct": stated}


def _none(inputs: StressInputs, probe: Probe, figures: Figures) -> list[Probe]:
    return []


@dataclass(frozen=True)
class Angle:
    """One kind of scenario: what computes it, what it raises next, and which engine answers."""

    title: str
    compute: Compute
    follow_ups: FollowUps
    engine: str


CATALOGUE: dict[str, Angle] = {
    "beta_shock": Angle("the stated shock through beta", _beta_shock, _beta_shock_children,
                        "argus.desk.portfolio.stress_by_beta"),
    "realised_window": Angle("the worst realised window", _realised_window,
                             _realised_window_children, "argus.desk.portfolio.worst_window"),
    "shock_frequency": Angle("how often the stated shock has happened", _shock_frequency,
                             _shock_frequency_children,
                             "argus.desk.stress.horizon_moves + shock_frequency"),
    "tail": Angle("who carries the worst hours", _tail, _tail_children,
                  "argus.desk.portfolio.tail_contributions"),
    "session_gap": Angle("open against shut-session beta", _session_gap, _session_gap_children,
                         "argus.desk.portfolio.beta (per session)"),
    "hedge_residual": Angle("the beta hedge through the worst realised window", _hedge_residual,
                            _hedge_residual_children, "argus.desk.portfolio.worst_window"),
    "hedged_tail": Angle("the beta hedge through the worst hours", _hedged_tail, _none,
                         "argus.desk.portfolio.tail_contributions"),
    "idiosyncratic": Angle("the holding the shock costs most, on its own worst day",
                           _idiosyncratic, _none,
                           "argus.desk.portfolio.worst_window"),
    "window_driver": Angle("halving the name that drove the worst window", _window_driver, _none,
                           "argus.desk.portfolio.worst_window"),
    "tail_trim": Angle("the trim that brings the tail inside budget", _tail_trim, _none,
                       "argus.desk.portfolio.tail_contributions"),
    "shut_shock": Angle("the stated shock in the shut session", _shut_shock, _none,
                        "argus.desk.portfolio.stress_by_beta"),
    "history_shock": Angle("the most extreme move that actually happened", _history_shock,
                           _none, "argus.desk.portfolio.stress_by_beta"),
}
"""Every angle the tree can grow. Adding one means adding its engine call here — never a figure a
model supplies."""

ROOT_ANGLES: tuple[str, ...] = (
    "beta_shock", "realised_window", "shock_frequency", "tail", "session_gap",
)
"""The first-order angles, in the order a narrower breadth drops them from the end: the stated
shock and what actually happened first, because every later scenario is read against them."""


# =============================================================================================
# The tree
# =============================================================================================


@dataclass
class Scenario:
    """One node: what was computed, from which parent, and whether it answered."""

    id: str
    angle: str
    subject: str
    depth: int
    parent: str | None
    status: str
    """``answered`` · ``empty`` (the engine had no estimate, with the reason) · ``failed`` (it
    raised; the reason names the exception) · ``skipped`` (the node budget was spent)."""

    text: str = ""
    figures: Figures = field(default_factory=dict)
    reason: str = ""
    engine: str = ""
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "angle": self.angle, "subject": self.subject,
                "depth": self.depth, "parent": self.parent, "status": self.status,
                "text": self.text, "figures": dict(self.figures), "reason": self.reason,
                "engine": self.engine, "elapsed_ms": round(self.elapsed_ms, 3)}


@dataclass
class StressTree:
    inputs: StressInputs
    nodes: list[Scenario]
    stops: list[str]
    breadth: int
    depth: int
    max_nodes: int
    concurrency: int
    elapsed_ms: float

    @property
    def answered(self) -> list[Scenario]:
        return [n for n in self.nodes if n.status == "answered"]

    @property
    def levels(self) -> int:
        return max((n.depth for n in self.answered), default=0)

    @property
    def angles(self) -> list[str]:
        """Distinct angles answered with a figure: the tree's scenario coverage."""
        return sorted({n.angle for n in self.answered})

    def render(self) -> list[str]:
        """The tree as answer lines, depth first, each child indented under its parent."""
        second = sum(1 for n in self.answered if n.depth > 1)
        head = (f"Scenario tree: {len(self.answered)} scenarios grown from the stated "
                f"{self.inputs.shocked_label} {self.inputs.shock_pct:+g}%, "
                f"{len(self.answered) - second} first-order and {second} grown from what those "
                f"found, {self.levels} level{'s' if self.levels != 1 else ''} deep — every "
                f"figure from the desk's own engines over the same history.")
        lines = [head]
        for node in sorted(self.nodes, key=lambda n: tuple(int(p) for p in n.id.split("."))):
            indent = "  " * (node.depth - 1)
            if node.status == "answered":
                lines.append(f"{indent}{node.id}. {node.text}")
            else:
                title = CATALOGUE[node.angle].title
                lines.append(f"{indent}{node.id}. Not computed — {title}: {node.reason}.")
        lines.extend(f"Tree stopped: {stop}." for stop in self.stops)
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "shocked": self.inputs.shocked, "shock_pct": self.inputs.shock_pct,
            "weights": dict(self.inputs.weights), "cash": self.inputs.cash,
            "breadth": self.breadth, "depth": self.depth, "max_nodes": self.max_nodes,
            "concurrency": self.concurrency, "elapsed_ms": round(self.elapsed_ms, 3),
            "answered": len(self.answered), "levels": self.levels, "angles": self.angles,
            "nodes": [n.as_dict() for n in self.nodes], "stops": list(self.stops),
        }


def _run_node(inputs: StressInputs, node_id: str, parent: str | None, depth: int,
              probe: Probe) -> Scenario:
    angle = CATALOGUE[probe.angle]
    started = time.perf_counter()
    node = Scenario(id=node_id, angle=probe.angle, subject=probe.subject, depth=depth,
                    parent=parent, status="answered", engine=angle.engine)
    try:
        node.text, node.figures = angle.compute(inputs, probe)
    except _Empty as empty:
        node.status, node.reason = "empty", str(empty) or "the engine returned no estimate"
    except Exception as exc:  # one node's failure is recorded, never propagated to its siblings
        node.status, node.reason = "failed", f"{type(exc).__name__}: {exc}"
    node.elapsed_ms = (time.perf_counter() - started) * 1000
    return node


def grow(
    inputs: StressInputs,
    *,
    breadth: int = DEFAULT_BREADTH,
    depth: int = DEFAULT_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> StressTree:
    """Grow the scenario tree for one book and one stated shock.

    ``breadth`` first-order angles run at the root; each answered node raises at most
    ``max(2, breadth // 2)`` children one level down, as upstream halves its breadth
    (`deep_research.py:535`); nothing is grown below ``depth`` levels or past ``max_nodes``.
    """
    if breadth < 1 or depth < 1 or max_nodes < 1 or concurrency < 1:
        raise StressTreeError("breadth, depth, the node budget and concurrency must be positive")
    from concurrent.futures import ThreadPoolExecutor

    started = time.perf_counter()
    # State first, then any stop that returns it (the order upstream's guard got wrong).
    nodes: list[Scenario] = []
    stops: list[str] = []
    visited: set[tuple[str, str]] = set()

    def level(frontier: list[tuple[str | None, str, Probe]], left: int, width: int,
              number: int) -> None:
        if not frontier:
            stops.append(f"level {number} had no scenario to grow")
            return
        runnable: list[tuple[str | None, str, Probe]] = []
        for parent, node_id, probe in frontier:
            key = (probe.angle, probe.subject)
            if key in visited:
                continue
            if len(nodes) + len(runnable) >= max_nodes:
                nodes.append(Scenario(id=node_id, angle=probe.angle, subject=probe.subject,
                                      depth=number, parent=parent, status="skipped",
                                      reason=f"the {max_nodes}-scenario budget was spent",
                                      engine=CATALOGUE[probe.angle].engine))
                continue
            visited.add(key)
            runnable.append((parent, node_id, probe))
        if not runnable:
            stops.append(f"level {number}: every scenario was already computed or over budget")
            return
        workers = max(1, min(concurrency, len(runnable)))
        if workers == 1:
            # No pool at all: on this workload a pool's thread start-up was measured to cost more
            # than the nodes themselves (`eval/research_depth.py`, see DEFAULT_CONCURRENCY).
            done = [_run_node(inputs, node_id, parent, number, probe)
                    for parent, node_id, probe in runnable]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                done = list(pool.map(lambda item: _run_node(inputs, item[1], item[0], number,
                                                            item[2]), runnable))
        nodes.extend(done)
        answered = [(node, item[2]) for node, item in zip(done, runnable, strict=True)
                    if node.status == "answered"]
        if not answered:
            stops.append(f"no scenario at level {number} answered, so nothing below it can be "
                         f"grown (gpt-researcher's #1579 stop)")
            return
        if left <= 1:
            return
        width_next = max(2, width // 2)
        children: list[tuple[str | None, str, Probe]] = []
        for node, probe in answered:
            kids = CATALOGUE[probe.angle].follow_ups(inputs, probe, node.figures)[:width_next]
            children.extend((node.id, f"{node.id}.{i}", kid)
                            for i, kid in enumerate(kids, start=1))
        if not children:
            return  # a leaf level: nothing the answers found raised a further scenario
        level(children, left - 1, width_next, number + 1)

    roots: list[tuple[str | None, str, Probe]] = [
        (None, str(i), Probe(angle, inputs.shocked))
        for i, angle in enumerate(ROOT_ANGLES[:breadth], start=1)]
    level(roots, depth, breadth, 1)
    return StressTree(inputs=inputs, nodes=nodes, stops=stops, breadth=breadth, depth=depth,
                      max_nodes=max_nodes, concurrency=concurrency,
                      elapsed_ms=(time.perf_counter() - started) * 1000)


def book_worst_window(
    raw: Mapping[str, Mapping[datetime, float]],
    book: Mapping[str, float],
    *,
    bars: int = WINDOW_BARS,
) -> WorstWindow:
    """The worst realised window of the book *as stated*, for the one-pass answer's line.

    Found by the recomputation in `eval/research_depth.py` (2026-09-25): the STRESS branch of
    `lui/research._run` takes its "What actually happened" figure from ``copilot(add=<first name>,
    before=<the rest>, size=<its weight>)``, and `desk/portfolio.rebalance` scales ``before`` by
    ``1 - size`` — so a 50/50 QQQ and TSLA book is replayed as QQQ 50%, TSLA 25%. On the 61 live
    stress questions that made 28 of the 28 multi-name realised-window figures wrong (one book
    stated -2.01% where the book as asked lost -4.01%), and every one of the 28 matched the
    rebalanced weights exactly. This is the same :func:`~argus.desk.portfolio.worst_window` over
    the same :func:`~argus.desk.portfolio.align`, on the weights the trader gave; the tree's
    ``realised_window`` node reads it the same way, so the two lines of one answer agree.
    """
    _, columns = align(raw)
    held = {s: float(w) for s, w in book.items() if float(w) != 0.0 and s in columns}
    return worst_window(weights=held, columns={s: columns[s] for s in held}, bars=bars)


def research_lines(
    *,
    raw: Mapping[str, Mapping[datetime, float]],
    is_open: Callable[[datetime], bool],
    book: Mapping[str, float],
    shocked: str,
    shock_pct: float | None,
    cash: float = 0.0,
    shocked_label: str | None = None,
    provenance: str = "",
) -> tuple[list[str], list[Any], dict[str, Any]]:
    """The console's call: the tree for a stress question, as answer lines, sources and payload.

    ``raw``, ``is_open``, the book and the shocked instrument are exactly what the STRESS branch of
    `lui/research._run` already holds. A question with no stated shock is grown from -10%, the
    larger of the two standard shocks that branch shows, and the head line says which shock it
    grew from. Returns empty lines (and says why in the payload) when no tree can be grown, so the
    one-pass answer stands alone rather than failing.
    """
    from argus.lui.answer import Source

    stated = shock_pct if shock_pct not in (None, 0) else -10.0
    try:
        inputs = prepare(raw, is_open=is_open, weights=book, shocked=shocked,
                         shock_pct=float(stated), cash=cash, shocked_label=shocked_label)
    except StressTreeError as exc:
        return [], [], {"grown": False, "reason": str(exc)}
    tree = grow(inputs)
    if not tree.answered:
        return [], [], {"grown": False, "reason": "; ".join(tree.stops) or "nothing answered",
                        **tree.as_dict()}
    engines = sorted({n.engine for n in tree.answered})
    source = Source(kind="computation", ref="argus.desk.stress_tree",
                    detail=f"{len(tree.answered)} scenarios, {tree.levels} levels; "
                           + (f"{provenance}; " if provenance else "") + "; ".join(engines))
    return tree.render(), [source], {"grown": True, **tree.as_dict()}


def _parse_book(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for pair in text.split(","):
        name, _, weight = pair.partition("=")
        if name.strip() and weight.strip():
            out[name.strip().upper()] = float(weight)
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI over live data
    parser = argparse.ArgumentParser(description="ARGUS second-order stress tree")
    parser.add_argument("--book", required=True, help='e.g. "NVDAUSDT=0.6,AAPLUSDT=0.4"')
    parser.add_argument("--shock", type=float, default=-10.0)
    parser.add_argument("--on", default="QQQUSDT", help="the instrument the shock hits")
    parser.add_argument("--cash", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from argus.lui import research

    book = _parse_book(args.book)
    data = research.load((*book, args.on))
    lines, _sources, payload = research_lines(
        raw=data.raw, is_open=research._is_open(), book=book, shocked=args.on,
        shock_pct=args.shock, cash=args.cash, provenance=data.provenance)
    for line in lines or [f"no tree: {payload.get('reason')}"]:
        print(line)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    return 0


__all__ = [
    "CATALOGUE",
    "DEFAULT_BREADTH",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_DEPTH",
    "DEFAULT_MAX_NODES",
    "ROOT_ANGLES",
    "WINDOW_BARS",
    "Angle",
    "Probe",
    "Scenario",
    "StressInputs",
    "StressTree",
    "StressTreeError",
    "book_worst_window",
    "grow",
    "prepare",
    "research_lines",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
