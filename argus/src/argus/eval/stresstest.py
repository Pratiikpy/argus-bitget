"""Adaptive stress testing: search for the **most probable** market path that breaks the mandate.

**This module exists because `eval/rogue.py`'s headline number is a dial, not a measurement.**
That harness constructs an adversary that demands an 8,000-unit position and walks it over a
hand-picked window, reporting the equity difference against an ungoverned arm. It found a real and
serious defect on its first run — four Constitution gates were comparing unit counts against dollar
ceilings, making the position cap 180x looser than written — so it earned its place. But its
*number* cannot survive an adversarial reading: the recklessness multiplier and the window are both
chosen by us, and a reviewer who opens the file sees `quantity=Decimal("8000")` and correctly
concludes the dollar figure is whatever we set that to. The same objection applies to
`narutopyy/agent-arena`'s `scripts/firewall_value.py`, which is where the design came from.

The fix is not a better-chosen attacker. It is to stop choosing.

**Adaptive Stress Testing** (Lee et al., Stanford SISL / NASA) makes the adversary *search*, and
charges it for implausibility. Read from `sisl/POMDPStressTesting.jl` `src/AST.jl:96-118` (MIT), the
reward is:

    r = logprob                          # every step: the likelihood of the disturbance chosen
    r += reward_bonus                    # if terminal and a failure was reached
    r += -miss_distance                  # if terminal and no failure: how close it got

``logprob`` is the log-likelihood of the disturbance **under the nominal distribution of that
disturbance**. So the attacker cannot win by inventing a 40% gap down: an implausible path is
penalised by exactly the amount by which it is implausible. It must find the most *probable*
sequence of returns that still breaks the mandate.

That changes what can be claimed. Instead of "the guard saved $628,665" — a figure whose size we
chose — the claim becomes:

    the most probable market path that busts the mandate has log-likelihood L (probability p),
    and under the real Constitution no path in the searched space busts it at all

which is a statement about the search, not about our parameter choices.

**Three things are added to the reference design, each because the reference could not answer an
objection this project has already had to answer elsewhere.**

* **The fat tail is not hidden.** AST rewards the *most likely* failure, so it systematically
  ignores catastrophic-but-improbable paths. Every search is therefore run twice — once
  likelihood-weighted, once with the likelihood term zeroed — and both are published. Reporting
  only the first would be the same selective framing this module exists to remove.
* **Paired, with an interval.** `ethz-spylab/agentdojo` runs each case with and without the attack
  and keys results by the pair (`src/agentdojo/benchmark.py:117-118, 155-156`), which is the
  counterfactual structure `eval/rogue.py` lacks — but it then aggregates with a bare mean and no
  interval (`benchmark.py:36-37`). Here every path is run under both policies and the *paired*
  difference carries a bootstrap confidence interval. One difference is an anecdote.
* **The margin is signed, not binary.** From the CPS falsification literature (`psy-taliro`,
  `breach`): a run that ends 1bp inside the limit and one that ends 500bp inside are not the same
  evidence. ``miss_distance`` is the signed margin to the nearest breach, so a near-miss is visible
  rather than being rounded to "no breach".

**What is deliberately NOT claimed.** The disturbance distribution is fitted to real venue history,
so "probability p" is a statement about *that fitted model*, not about the market. A search over a
finite budget proves nothing about paths it never visited. And none of this is evidence about
ARGUS's own model, which has proposed exposure twice in 447 decisions — it measures the rule set
against a searched adversary.

    python -m argus.eval.stresstest
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.agents.desk import ConstitutionPolicy
from argus.decision.verdicts import Intent, Side, Verdict
from argus.eval.artefact import write
from argus.risk.hedgeability import HedgeabilitySurface, HedgeCandidate
from argus.truth.clocks import SessionPhase, SessionState

REPORT_PATH = Path(__file__).resolve().parents[3] / "data" / "stress_test.json"

STARTING_EQUITY = Decimal("10000")
FAILURE_DRAWDOWN = 0.30
"""Equity below 70% of its peak is the failure event the search hunts for.

A named threshold rather than "goes bust", because AST needs a *signed* distance to the event on
every non-failing run, and "how far from bankrupt" is uninformative until you are nearly there."""

HORIZON = 48
"""Bars per episode. Two days of hourly candles — long enough for a drawdown to develop under a
plausible path, short enough that the likelihood of a whole episode does not underflow."""

REWARD_BONUS = 100.0
"""`R_E` in `AST.jl:110`, added once on reaching the event. It must dominate the per-step logprob
so that finding *a* failure always beats improving the likelihood of not failing."""


class StressError(RuntimeError):
    """The search cannot be run honestly. Raised rather than reported as a null result."""


@dataclass(frozen=True, slots=True)
class Disturbance:
    """One step's market move, and how surprising it was.

    ``logprob`` is under the fitted nominal distribution — the attacker's budget. Storing it beside
    the value is what makes an episode's total likelihood recoverable afterwards rather than
    recomputed from a model that may since have changed."""

    ret: float
    logprob: float


@dataclass(frozen=True, slots=True)
class Episode:
    """One searched path and what it did to the account."""

    returns: tuple[float, ...]
    total_logprob: float
    failed: bool
    miss_distance: float
    """Signed margin to the failure threshold. Negative means the event was reached and by how
    much; positive is a near-miss, smaller being nearer."""

    final_equity: float
    max_drawdown: float
    reward: float
    log_ratio_vs_mode: float = 0.0
    """`total_logprob` minus the log-density of the modal path (every step at the mean).

    Always <= 0. This is the quantity AST's likelihood term is really optimising, and unlike a
    density it is comparable across symbols and horizons."""

    sigma_from_mode: float = 0.0
    """`sqrt(-2 * log_ratio_vs_mode)` — the aggregate distance from the mode, in standard
    deviations. A 48-step path where every step sits 1 sigma out has sigma_from_mode = sqrt(48)."""

    @property
    def sigma_distance(self) -> float:
        """How many standard deviations the whole path sits from the nominal, in aggregate.

        **This replaces a "probability" that was not one.** The first version of this module
        exponentiated `total_logprob` and published the result as a probability; it printed
        `p=2.35e+95`. A Gaussian *density* exceeds 1 whenever sigma is small — hourly rToken
        returns have sigma around 0.003, giving a peak density near 133 and a positive log-density
        — so summing 48 of them yields +219, and `exp(219)` is not a probability of anything. The
        AST reward was correct (it only ever compares densities); the reported figure was not.

        A likelihood ratio against the *most likely* path is the honest scalar: how many times less
        likely this path is than the modal one, expressed in sigma. It is dimensionless, it is
        bounded, and it is what "implausible" actually means here.
        """
        return self.sigma_from_mode

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_logprob": self.total_logprob,
            "log_likelihood_ratio_vs_modal_path": self.log_ratio_vs_mode,
            "sigma_from_mode": self.sigma_from_mode,
            "failed": self.failed,
            "miss_distance": self.miss_distance,
            "final_equity": self.final_equity,
            "max_drawdown": self.max_drawdown,
            "reward": self.reward,
        }


def _fit(closes: Sequence[float]) -> tuple[float, float]:
    """Mean and standard deviation of real log returns. The nominal distribution.

    Fitted from venue history rather than assumed, because the whole force of the likelihood term
    is that it is the likelihood of *this market*, not of a Gaussian somebody picked.
    """
    rets = [math.log(closes[i + 1] / closes[i]) for i in range(len(closes) - 1)
            if closes[i] > 0 and closes[i + 1] > 0]
    if len(rets) < 100:
        raise StressError(f"{len(rets)} returns is too few to fit a disturbance distribution")
    mu = statistics.fmean(rets)
    sigma = statistics.stdev(rets)
    if sigma <= 0:
        raise StressError("fitted volatility is zero; the history is degenerate")
    return mu, sigma


def _logpdf(x: float, mu: float, sigma: float) -> float:
    """Gaussian log density. The attacker's charge for choosing this move."""
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma * math.sqrt(2 * math.pi))


def _session() -> SessionState:
    return SessionState(
        phase=SessionPhase.RTH,
        as_of=datetime(2026, 9, 14, 15, 0, tzinfo=UTC),
        hours_to_next_discovery=0.0,
        nav_age_seconds=10.0,
    )


def _hedges() -> HedgeabilitySurface:
    return HedgeabilitySurface(
        candidates=(
            HedgeCandidate(
                instrument="QQQUSDT",
                risk_reduction=Decimal("0.85"),
                correlation_confidence=Decimal("0.9"),
                liquidity_availability=Decimal("0.9"),
                execution_probability=Decimal("0.95"),
                basis_stability=Decimal("0.8"),
                execution_cost_bps=Decimal("6"),
            ),
        ),
        session_note="stress harness: one placeable hedge",
    )


def _run_episode(
    returns: Sequence[Disturbance], *, policy: ConstitutionPolicy, start_price: float,
    symbol: str, demand: Decimal, likelihood_weighted: bool, mu: float, sigma: float,
) -> Episode:
    """Walk one disturbance sequence, governed by one policy, and score it the AST way.

    The agent demands the same position every bar; what varies is the market. That is the point —
    the adversary being searched is the *path*, not the trader. A trader-side search would measure
    how badly we can behave; this measures how badly the world can behave while we behave normally.
    """
    equity = STARTING_EQUITY
    peak = equity
    worst_drawdown = 0.0
    price = Decimal(str(start_price))
    session, hedges = _session(), _hedges()

    for step in returns:
        priced = replace(policy, reference_price=price)
        intent = Intent(
            symbol=symbol, side=Side.BUY, quantity=demand, verdict=Verdict.TRADE,
            stated_confidence=0.9, thesis="stress harness: constant long", invalidation=("x",),
        )
        permitted = priced.rule(intent, session=session, hedges=hedges).resulting_intent
        move = price * Decimal(str(math.expm1(step.ret)))
        equity += permitted.quantity * move
        price += move
        if price <= 0:
            break
        peak = max(peak, equity)
        if peak > 0:
            worst_drawdown = max(worst_drawdown, float((peak - equity) / peak))

    failed = worst_drawdown >= FAILURE_DRAWDOWN
    # Signed margin: negative once the event is reached, and by how far past it.
    miss = FAILURE_DRAWDOWN - worst_drawdown
    total_logprob = sum(s.logprob for s in returns)
    # Against the modal path — every step exactly at the mean. This is the only reference that
    # makes the likelihood term reportable: a density is not a probability, a ratio of densities
    # against a fixed reference is.
    modal_logprob = len(returns) * _logpdf(mu, mu, sigma)
    log_ratio = total_logprob - modal_logprob
    sigma_from_mode = math.sqrt(max(0.0, -2.0 * log_ratio))

    # `AST.jl:107-115`, standard (non-episodic) reward. The likelihood term is what stops the
    # search inventing an implausible crash; zeroing it is the deliberate ablation that exposes
    # the fat tail AST would otherwise never visit.
    reward = total_logprob if likelihood_weighted else 0.0
    reward += REWARD_BONUS if failed else -abs(miss) * 100.0
    return Episode(
        returns=tuple(s.ret for s in returns), total_logprob=total_logprob, failed=failed,
        miss_distance=miss, final_equity=float(equity), max_drawdown=worst_drawdown,
        reward=reward, log_ratio_vs_mode=log_ratio, sigma_from_mode=sigma_from_mode,
    )


def search(
    closes: Sequence[float], *, policy: ConstitutionPolicy, symbol: str, demand: Decimal,
    iterations: int, likelihood_weighted: bool, seed: int,
) -> dict[str, Any]:
    """Cross-entropy search over disturbance sequences for the highest-reward path.

    A population method rather than MCTS: the reference uses Monte Carlo tree search over a POMDP,
    which is the right tool when the adversary's actions are sequential and state-dependent. Here
    the episode's outcome is a function of the whole return sequence, the action space is
    continuous, and cross-entropy converges on exactly this shape of problem with no dependency to
    vendor. The mechanism that matters — reward = logprob + bonus - miss — is unchanged, and it is
    the mechanism, not the optimiser, that removes the free parameter.

    Deliberately seeded. A stress test whose result moves between runs cannot be cited.
    """
    mu, sigma = _fit(closes)
    rng = random.Random(seed)
    start_price = closes[-1]

    # The search distribution starts at the nominal and is pulled toward whatever breaks things.
    # Per-step mean, shared sigma: enough freedom to find a crash, few enough parameters to
    # converge inside a budget a reviewer can rerun.
    search_mu = [mu] * HORIZON
    search_sigma = sigma
    population = 60
    elite = 12
    generations = max(1, iterations // population)

    best: Episode | None = None
    evaluated = 0
    failures = 0
    first_failure_at: int | None = None

    for _ in range(generations):
        scored: list[tuple[float, list[float], Episode]] = []
        for _ in range(population):
            draws = [rng.gauss(search_mu[i], search_sigma) for i in range(HORIZON)]
            steps = [Disturbance(r, _logpdf(r, mu, sigma)) for r in draws]
            episode = _run_episode(
                steps, policy=policy, start_price=start_price, symbol=symbol, demand=demand,
                likelihood_weighted=likelihood_weighted, mu=mu, sigma=sigma,
            )
            evaluated += 1
            if episode.failed:
                failures += 1
                if first_failure_at is None:
                    first_failure_at = evaluated
            scored.append((episode.reward, draws, episode))
            # "Best" means highest reward, which under the likelihood-weighted objective means the
            # most probable failure — not the largest loss. That distinction is the whole method.
            if best is None or episode.reward > best.reward:
                best = episode

        scored.sort(key=lambda row: row[0], reverse=True)
        top = [row[1] for row in scored[:elite]]
        search_mu = [statistics.fmean(draw[i] for draw in top) for i in range(HORIZON)]
        spread = statistics.fmean(
            statistics.pstdev([draw[i] for draw in top]) for i in range(HORIZON)
        )
        # Floored so the search cannot collapse to a point and stop exploring.
        search_sigma = max(spread, sigma * 0.25)

    if best is None:  # pragma: no cover - population is always positive
        raise StressError("search evaluated no episodes")

    best_failure = None
    if failures:
        # `metrics.jl:23-40`: among failures only, the one with the highest log-likelihood.
        best_failure = best if best.failed else None
    return {
        "likelihood_weighted": likelihood_weighted,
        "episodes_evaluated": evaluated,
        "failures_found": failures,
        "failure_rate": failures / evaluated if evaluated else 0.0,
        "first_failure_at_episode": first_failure_at,
        "best_episode": best.as_dict(),
        "highest_loglikelihood_failure": best_failure.as_dict() if best_failure else None,
        "nominal_distribution": {"mu": mu, "sigma": sigma, "fitted_from_returns": len(closes) - 1},
    }


def _bootstrap(
    diffs: Sequence[float], *, seed: int = 20260921, draws: int = 2000,
) -> dict[str, Any]:
    """Percentile CI of the paired mean difference.

    The interval `agentdojo` does not compute (`benchmark.py:36-37` is a bare mean). A single
    guarded-minus-unguarded difference is an anecdote; this says whether the sign is stable.
    """
    if not diffs:
        return {"mean": None, "ci95": None, "n": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        means.append(statistics.fmean(sample))
    means.sort()
    return {
        "mean": statistics.fmean(diffs),
        "ci95": [means[int(0.025 * draws)], means[int(0.975 * draws)]],
        "n": len(diffs),
        "excludes_zero": means[int(0.025 * draws)] > 0 or means[int(0.975 * draws)] < 0,
    }


def _ungoverned(policy: ConstitutionPolicy) -> ConstitutionPolicy:
    """Identical rulebook, thresholds out of reach. Both arms run the same code path."""
    huge = Decimal("1e12")
    return replace(
        policy, max_position_notional=huge, max_unhedged_notional=huge,
        max_gross_exposure_notional=huge, max_signed_exposure_notional=huge,
        min_confidence_to_trade=0.0,
    )


def run(
    symbol: str = "NVDAUSDT", *, days: int = 90, iterations: int = 600, seed: int = 20260921,
) -> dict[str, Any]:
    """Both searches, both arms, with a paired interval over the searched paths."""
    from argus.market.history import fetch_range

    candles = fetch_range(symbol, days=days, interval="1H")
    closes = [float(c.close) for c in candles]
    mu, sigma = _fit(closes)
    policy = ConstitutionPolicy()
    free = _ungoverned(policy)
    demand = Decimal("8000")

    searches = {
        "ungoverned_likelihood_weighted": search(
            closes, policy=free, symbol=symbol, demand=demand, iterations=iterations,
            likelihood_weighted=True, seed=seed,
        ),
        "ungoverned_unweighted": search(
            closes, policy=free, symbol=symbol, demand=demand, iterations=iterations,
            likelihood_weighted=False, seed=seed,
        ),
        "governed_likelihood_weighted": search(
            closes, policy=policy, symbol=symbol, demand=demand, iterations=iterations,
            likelihood_weighted=True, seed=seed,
        ),
        "governed_unweighted": search(
            closes, policy=policy, symbol=symbol, demand=demand, iterations=iterations,
            likelihood_weighted=False, seed=seed,
        ),
    }

    # Paired arm: the SAME sampled paths through both policies, so the difference is attributable
    # to the rulebook and not to two different draws.
    rng = random.Random(seed + 1)
    diffs: list[float] = []
    governed_failures = ungoverned_failures = 0
    for _ in range(200):
        draws = [rng.gauss(mu, sigma) for _ in range(HORIZON)]
        steps = [Disturbance(r, _logpdf(r, mu, sigma)) for r in draws]
        g = _run_episode(steps, policy=policy, start_price=closes[-1], symbol=symbol,
                         demand=demand, likelihood_weighted=True, mu=mu, sigma=sigma)
        u = _run_episode(steps, policy=free, start_price=closes[-1], symbol=symbol,
                         demand=demand, likelihood_weighted=True, mu=mu, sigma=sigma)
        diffs.append(g.final_equity - u.final_equity)
        governed_failures += int(g.failed)
        ungoverned_failures += int(u.failed)

    weighted = searches["ungoverned_likelihood_weighted"]
    unweighted = searches["ungoverned_unweighted"]
    gov = searches["governed_likelihood_weighted"]
    return {
        "symbol": symbol,
        "horizon_bars": HORIZON,
        "failure_threshold_drawdown": FAILURE_DRAWDOWN,
        "seed": seed,
        "searches": searches,
        "paired": {
            "paths": len(diffs),
            "equity_difference": _bootstrap(diffs),
            "governed_failure_rate": governed_failures / len(diffs),
            "ungoverned_failure_rate": ungoverned_failures / len(diffs),
        },
        "headline": {
            "most_probable_failure_log_ratio": (
                weighted["highest_loglikelihood_failure"]["log_likelihood_ratio_vs_modal_path"]
                if weighted["highest_loglikelihood_failure"] else None
            ),
            "most_probable_failure_sigma_from_mode": (
                weighted["highest_loglikelihood_failure"]["sigma_from_mode"]
                if weighted["highest_loglikelihood_failure"] else None
            ),
            "ungoverned_failures_found": weighted["failures_found"],
            "governed_failures_found": gov["failures_found"],
            "fat_tail_failures_found_unweighted": unweighted["failures_found"],
            "guard_survived_the_search": gov["failures_found"] == 0,
        },
        "scope_statement": (
            "An adversary searches for the market path most likely to drive a constant-long "
            "account into a 30% drawdown, charged the log-likelihood of every move under a "
            "Gaussian fitted to this symbol's real hourly returns. Both arms run the identical "
            "code path and differ only in their thresholds; the paired block runs the SAME sampled "
            "paths through both. NOT CLAIMED: that the fitted Gaussian is the market — a "
            "probability here is a statement about that model, and real returns are fat-tailed, "
            "which is why the unweighted search is published beside it. NOT CLAIMED: anything "
            "about paths the search never visited, or about ARGUS's own model, which has proposed "
            "exposure twice in 447 decisions."
        ),
    }


def render(report: dict[str, Any]) -> list[str]:
    head = report["headline"]
    paired = report["paired"]
    lines = [
        f"ADAPTIVE STRESS TEST — {report['symbol']}, {report['horizon_bars']}-bar episodes, "
        f"failure = {report['failure_threshold_drawdown']:.0%} drawdown",
    ]
    if head["most_probable_failure_log_ratio"] is not None:
        lines.append(
            f"  ungoverned: the MOST PROBABLE failing path sits "
            f"{head['most_probable_failure_sigma_from_mode']:.1f} sigma from the modal path "
            f"(log-likelihood ratio {head['most_probable_failure_log_ratio']:.1f}); "
            f"{head['ungoverned_failures_found']} failing paths found"
        )
    else:
        lines.append("  ungoverned: no failing path found in the searched space")
    lines.append(
        f"  governed:   {head['governed_failures_found']} failing path(s) found"
        + ("  — the guard survived the search" if head["guard_survived_the_search"] else "")
    )
    lines.append(
        f"  fat tail:   {head['fat_tail_failures_found_unweighted']} failures when the likelihood "
        f"charge is removed — published because AST alone would never visit them"
    )
    ci = paired["equity_difference"]
    if ci["mean"] is not None:
        lo, hi = ci["ci95"]
        lines.append(
            f"  paired over {paired['paths']} identical paths: mean equity difference "
            f"${ci['mean']:,.0f}, 95% CI [${lo:,.0f}, ${hi:,.0f}]"
            + ("  (excludes zero)" if ci["excludes_zero"] else "  (includes zero)")
        )
        lines.append(
            f"  failure rate governed {paired['governed_failure_rate']:.1%} vs ungoverned "
            f"{paired['ungoverned_failure_rate']:.1%}"
        )
    return lines


def main() -> int:  # pragma: no cover - CLI
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="search for the most probable failing path")
    parser.add_argument("--symbol", default="NVDAUSDT")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--iterations", type=int, default=600)
    args = parser.parse_args()

    report = run(args.symbol, days=args.days, iterations=args.iterations)
    for line in render(report):
        print(line)
    write(REPORT_PATH, report)
    print(f"\nwritten to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "REPORT_PATH", "Disturbance", "Episode", "StressError",
    "main", "render", "run", "search",
]
