"""The adversary must search for the failing path, not be handed one.

**This file exists because `eval/rogue.py`'s number is a dial.** That harness picks the
recklessness (8,000 units) and the window, so its dollar figure is whatever those were set to — the
same objection that applies to `narutopyy/agent-arena`'s `firewall_value.py`, which is where the
design came from. It still earned its place: it found four Constitution gates comparing unit counts
against dollar ceilings on its first run.

Adaptive Stress Testing removes the dial by charging the attacker the log-likelihood of every move
it chooses (`sisl/POMDPStressTesting.jl` `src/AST.jl:96-118`). These tests assert the properties
that make the resulting number citable: the likelihood term actually binds, the fat tail is
published beside it, the arms are paired, and the reported figure is not a probability greater
than one — which the first version printed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from argus.eval.stresstest import FAILURE_DRAWDOWN, HORIZON, REWARD_BONUS

ARTEFACT = Path(__file__).resolve().parents[1] / "data" / "stress_test.json"


@pytest.fixture(scope="module")
def report() -> dict:
    if not ARTEFACT.exists():
        pytest.skip("run `python -m argus.eval.stresstest` to produce the artefact")
    return json.loads(ARTEFACT.read_text(encoding="utf-8"))


class TestTheSearchIsNotADial:
    def test_the_likelihood_term_binds(self, report: dict) -> None:
        """The whole method. Zeroing the charge must change what the search finds — if it does
        not, the attacker was never constrained and the number is chosen after all."""
        weighted = report["searches"]["ungoverned_likelihood_weighted"]
        unweighted = report["searches"]["ungoverned_unweighted"]
        assert weighted["failures_found"] != unweighted["failures_found"] or (
            weighted["best_episode"]["total_logprob"]
            != unweighted["best_episode"]["total_logprob"]
        )

    def test_the_fat_tail_is_published_not_hidden(self, report: dict) -> None:
        """AST rewards the *most likely* failure, so it systematically never visits the
        catastrophic-but-improbable. Reporting only the weighted search would hide exactly the
        paths a risk layer exists for."""
        assert "fat_tail_failures_found_unweighted" in report["headline"]
        assert report["searches"]["ungoverned_unweighted"]["likelihood_weighted"] is False

    def test_the_search_is_seeded(self, report: dict) -> None:
        """A stress test whose result moves between runs cannot be cited."""
        assert isinstance(report["seed"], int)


class TestTheReportedNumbersAreCoherent:
    def test_no_probability_above_one_is_published(self, report: dict) -> None:
        """**The defect this test was written for.** The first run printed `p=2.35e+95`. A Gaussian
        *density* exceeds 1 when sigma is small — hourly rToken sigma is ~0.003, giving a peak
        density near 133 — so summing 48 log-densities yields +219, and exp(219) is not a
        probability of anything. The search was right; the reporting was not."""
        blob = json.dumps(report)
        assert "probability" not in blob or all(
            v <= 1.0
            for v in _numbers_under_key(report, "probability")
        )

    def test_the_headline_is_a_likelihood_ratio_not_a_density(self, report: dict) -> None:
        """A ratio against the modal path is dimensionless and bounded above by zero in logs."""
        ratio = report["headline"]["most_probable_failure_log_ratio"]
        if ratio is not None:
            assert ratio <= 0.0, "a path cannot be more likely than the modal one"

    def test_sigma_from_mode_is_a_real_distance(self, report: dict) -> None:
        sigma = report["headline"]["most_probable_failure_sigma_from_mode"]
        if sigma is not None:
            assert sigma >= 0.0 and math.isfinite(sigma)


class TestThePairedArmIsPaired:
    def test_both_policies_saw_the_same_paths(self, report: dict) -> None:
        """`agentdojo` keys results by (task, attack) so differences are attributable; its own
        aggregation is a bare mean with no interval (`benchmark.py:36-37`). This keeps the pairing
        and adds the interval."""
        paired = report["paired"]
        assert paired["paths"] > 0
        assert paired["equity_difference"]["n"] == paired["paths"]

    def test_the_difference_carries_an_interval(self, report: dict) -> None:
        """One difference is an anecdote."""
        ci = report["paired"]["equity_difference"]["ci95"]
        assert ci is not None and len(ci) == 2 and ci[0] <= ci[1]

    def test_the_guard_reduces_the_failure_rate(self, report: dict) -> None:
        """The claim the whole module is for, and the one number that is not a dial."""
        paired = report["paired"]
        assert paired["governed_failure_rate"] < paired["ungoverned_failure_rate"]


class TestItStatesWhatItCannotClaim:
    def test_the_scope_statement_names_the_fitted_model_limit(self, report: dict) -> None:
        """"Probability" here is a statement about a fitted Gaussian, not about the market, and
        real returns are fat-tailed. Saying so is the difference between a measurement and a
        marketing number."""
        scope = report["scope_statement"]
        assert "NOT CLAIMED" in scope
        assert "fat-tailed" in scope

    def test_the_constants_are_named_not_magic(self) -> None:
        assert 0.0 < FAILURE_DRAWDOWN < 1.0
        assert HORIZON > 0
        assert REWARD_BONUS > 0


def _numbers_under_key(blob: object, key: str) -> list[float]:
    found: list[float] = []
    if isinstance(blob, dict):
        for k, v in blob.items():
            if k == key and isinstance(v, int | float):
                found.append(float(v))
            else:
                found.extend(_numbers_under_key(v, key))
    elif isinstance(blob, list):
        for item in blob:
            found.extend(_numbers_under_key(item, key))
    return found
