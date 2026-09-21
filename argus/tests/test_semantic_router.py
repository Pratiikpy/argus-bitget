"""The console must route on meaning, and must refuse what it does not understand.

**Both halves are asserted, and the second matters more.** The deterministic pattern layer scores
9.5% on the held-out corpus (`eval/obliquebench.py`) and says "unknown" to everything it misses —
that is the judge-facing state, because the deployed demo ships no model key. A semantic router
lifts in-scope accuracy, but a router that answers *everything* would answer "what is the weather
in Tokyo" with a trading intent, which is worse than silence.

So these tests pin the trade-off rather than the flattering half: in-scope accuracy **with**
out-of-scope recall, measured on corpora the router never saw.
"""

from __future__ import annotations

import pytest

from argus.eval.obliquebench import HELDOUT, TUNED
from argus.lui.semantic import ABSTAIN_THRESHOLD, Routed, SemanticRouter, available

pytestmark = pytest.mark.skipif(not available(), reason="model2vec is not installed")

OUT_OF_SCOPE = (
    "what is the weather in tokyo", "write me a poem about rain", "who won the world cup",
    "what is 17 times 23", "how do i make pasta", "tell me a joke", "recommend a movie",
)


@pytest.fixture(scope="module")
def router() -> SemanticRouter:
    """Built from TUNED only. HELDOUT is never seen during construction — that is the whole
    reason `obliquebench` keeps two corpora."""
    return SemanticRouter([(c.ask, str(c.expect)) for c in TUNED])


class TestItRoutesOnMeaning:
    def test_it_beats_the_pattern_layer_on_unseen_phrasings(self, router: SemanticRouter) -> None:
        """The measurement that justifies the module existing at all. 9.5% is the incumbent."""
        correct = sum(
            1 for c in HELDOUT
            if (r := router.route(c.ask)).confident and r.intent == str(c.expect)
        )
        accuracy = correct / len(HELDOUT)
        assert accuracy > 0.40, f"{accuracy:.1%} is not a material gain over the pattern layer"

    def test_paraphrases_of_one_question_route_together(self, router: SemanticRouter) -> None:
        """CheckList INV. Regexes fail this by construction; meaning-based routing should not."""
        a = router.route("how did we do")
        b = router.route("how did we perform")
        assert a.intent == b.intent

    def test_it_needs_no_network_at_query_time(self, router: SemanticRouter) -> None:
        """Static embeddings are a table lookup. The point of the whole design: the deployed demo
        has no key, so anything requiring one is not the thing a judge meets."""
        import socket

        real = socket.socket

        def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError("the router opened a socket while answering a question")

        socket.socket = refuse  # type: ignore[assignment,misc]
        try:
            assert router.route("how did we do") is not None
        finally:
            socket.socket = real  # type: ignore[misc]


class TestItRefusesWhatItDoesNotUnderstand:
    def test_out_of_scope_questions_are_declined(self, router: SemanticRouter) -> None:
        """**The hard half.** CLINC150's own finding is that BERT reaches ~97% in-scope accuracy
        and only 40-66% out-of-scope recall — a missed refusal is a confident wrong answer."""
        answered = [q for q in OUT_OF_SCOPE if router.route(q).confident]
        recall = 1 - len(answered) / len(OUT_OF_SCOPE)
        assert recall >= 0.80, f"answered out-of-scope questions: {answered}"

    def test_a_declined_question_still_reports_how_close_it_came(
        self, router: SemanticRouter
    ) -> None:
        """"Declined at 0.31" and "declined at 0.04" are different failures. A console that reports
        both as `unknown` cannot be debugged by the person reading it."""
        routed = router.route("what is the weather in tokyo")
        assert not routed.confident
        assert routed.similarity > 0.0

    def test_it_offers_the_nearest_intents_instead_of_a_dead_end(
        self, router: SemanticRouter
    ) -> None:
        """Replaces answering "unknown": say what you *can* be asked."""
        nearest = router.nearest("what is the weather in tokyo")
        assert len(nearest) == 3
        assert all(isinstance(label, str) for label, _ in nearest)

    def test_an_empty_question_is_declined_rather_than_matched(
        self, router: SemanticRouter
    ) -> None:
        assert not router.route("   ").confident


class TestTheThresholdIsAReportedChoice:
    def test_lowering_it_buys_accuracy_and_costs_refusal(self) -> None:
        """The trade-off is real and is published in both directions. A module that shipped only
        the answer-everything number would be choosing the flattering row."""
        strict = SemanticRouter([(c.ask, str(c.expect)) for c in TUNED])
        permissive = SemanticRouter(
            [(c.ask, str(c.expect)) for c in TUNED], threshold=0.0,
        )
        strict_correct = sum(
            1 for c in HELDOUT
            if (r := strict.route(c.ask)).confident and r.intent == str(c.expect)
        )
        loose_correct = sum(
            1 for c in HELDOUT
            if (r := permissive.route(c.ask)).confident and r.intent == str(c.expect)
        )
        assert loose_correct >= strict_correct
        # ...and the permissive router answers every out-of-scope probe, which is the cost.
        assert all(permissive.route(q).confident for q in OUT_OF_SCOPE)

    def test_the_default_threshold_is_the_fitted_one(self) -> None:
        """Pinned so a future edit that loosens it has to change this line and say why."""
        assert ABSTAIN_THRESHOLD == 0.32


class TestItRefusesToBeBuiltWrong:
    def test_no_examples_is_an_error_not_an_empty_router(self) -> None:
        with pytest.raises(ValueError, match="cannot route"):
            SemanticRouter([])

    def test_a_routed_result_reports_its_margin(self) -> None:
        got = Routed("performance", 0.8, runner_up="position", runner_up_similarity=0.3)
        assert abs(got.margin - 0.5) < 1e-9
