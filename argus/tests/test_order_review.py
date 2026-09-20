"""Pre-trade review tests — including the one payload that is the whole argument.

`agent-sdk/src/tools/safety.ts:78-107` is Bitget's own safety layer and it is three checks: dryRun,
readOnly-blocks-writes, and confirm-on-high-risk. **None of the three looks at the order.** They are
about the session and the operation class, by design — that layer exists to stop an *accidental*
write, not an ill-judged one.

So the test that matters is a payload `executeWithSafety` would pass untouched — a write, not
high-risk, no dryRun, readOnly off — that this reviewer narrows or refuses with the binding gate
named. That single case is the claim in miniature, and it is
`test_an_order_bitgets_own_layer_would_pass_is_narrowed_here`.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from argus.execution.review import (
    ALLOW,
    MALFORMED,
    NARROW,
    REFUSE,
    REQUIRED,
    UNEXERCISED_NOTE,
    ReviewVerdict,
    parse,
    review,
)


def _order(**overrides: object) -> dict[str, object]:
    """A payload that satisfies `/api/v3/trade/place-order` exactly."""
    body: dict[str, object] = {
        "category": "USDT-FUTURES",
        "symbol": "NVDAUSDT",
        "qty": "1",
        "side": "buy",
        "orderType": "market",
    }
    body.update(overrides)
    return body


class TestTheContractIsTheOneBitgetPublishes:
    def test_every_required_field_is_required(self) -> None:
        for name in REQUIRED:
            body = _order()
            del body[name]
            got = review(body)
            assert got.verdict == MALFORMED, name
            assert any(name in p for p in got.problems), name

    def test_a_limit_order_without_a_price_is_malformed(self) -> None:
        """The contract's own conditional rule: *price is required when the order type is a limit
        order*."""
        got = review(_order(orderType="limit"))
        assert got.verdict == MALFORMED
        assert any("price" in p and "limit" in p for p in got.problems)

    def test_a_market_order_with_a_price_is_malformed(self) -> None:
        """*This field is not applicable when the order type is a market order.*"""
        got = review(_order(orderType="market", price="100"))
        assert got.verdict == MALFORMED
        assert any("not applicable" in p for p in got.problems)

    def test_an_unknown_category_is_named_with_the_legal_set(self) -> None:
        got = review(_order(category="FUTURES"))
        assert got.verdict == MALFORMED
        assert any("USDT-FUTURES" in p for p in got.problems)

    def test_every_problem_is_reported_not_just_the_first(self) -> None:
        """A caller fixing one field at a time learns nothing about the second."""
        got = review({"category": "NOPE", "symbol": "X", "qty": "-1", "side": "up",
                      "orderType": "stop"})
        assert len(got.problems) >= 4

    def test_a_negative_quantity_is_not_a_size(self) -> None:
        assert any("positive" in p for p in parse(_order(qty="-5")).problems)

    def test_malformed_is_not_refused(self) -> None:
        """A refusal is a judgement about a trade. This is the absence of one to judge, and
        conflating them would let a caller read a typo as a risk opinion."""
        got = review(_order(qty="abc"))
        assert got.verdict == MALFORMED
        assert got.verdict != REFUSE


class TestItBeatsTheBaselineItNames:
    def test_an_order_bitgets_own_layer_would_pass_is_narrowed_here(self) -> None:
        """**The claim in miniature.**

        `executeWithSafety` sees: not a dryRun, readOnly off, riskLevel not high. It sends this
        order untouched. It never looks at the size, because size is not what that layer is for.
        """
        got = review(_order(qty="1000000"))
        assert got.verdict in (NARROW, REFUSE)
        assert got.permitted_qty is not None
        assert got.permitted_qty < Decimal("1000000")
        assert got.binding_constraint not in ("none", "no_exposure")

    def test_the_binding_constraint_is_named_not_scored(self) -> None:
        """`research/overfit.py` refuses a 0-100 score for a stated reason, and a 73/100 here would
        invite a caller to ship an order that failed a hard limit because it passed soft ones."""
        got = review(_order(qty="1000000"))
        assert isinstance(got.binding_constraint, str) and got.binding_constraint
        assert not hasattr(got, "score")

    def test_the_whole_chain_is_reported_not_only_the_gate_that_fired(self) -> None:
        got = review(_order(qty="1000000"))
        statuses = [row["status"] for row in got.chain]
        from argus.eval.autopsy import CHAIN

        assert len(got.chain) == len(CHAIN)
        assert "FIRED" in statuses
        assert statuses.count("FIRED") == 1, "exactly one rule binds"

    def test_gates_after_a_terminal_binder_read_unreached_not_passed(self) -> None:
        """The distinction the whole project exists for: a gate that never ran has not allowed
        anything. A TERMINAL binder (here, `min_confidence`) still ends evaluation outright."""
        got = review(_order(qty="1000000"), confidence=0.1)
        assert got.binding_constraint == "min_confidence"
        fired = next(i for i, row in enumerate(got.chain) if row["status"] == "FIRED")
        assert all(row["status"] == "UNREACHED" for row in got.chain[fired + 1:])
        assert all(row["status"] == "PASSED" for row in got.chain[:fired])

    def test_gates_after_a_ceiling_binder_read_passed_not_unreached(self) -> None:
        """The Constitution evaluates every ceiling and takes the minimum — a ceiling gate
        (here, `unhedgeable_gap`) does not short-circuit the ones after it in source order, and
        this chain report must not claim it does. Same defect this module's own tests once carried
        as `desk.carrydesk.walk_chain`, independently, before the Constitution's restructure."""
        got = review(_order(qty="1000000"))
        assert got.binding_constraint == "unhedgeable_gap"
        fired = next(i for i, row in enumerate(got.chain) if row["status"] == "FIRED")
        assert all(row["status"] == "PASSED" for row in got.chain[fired + 1:])
        assert all(row["status"] == "PASSED" for row in got.chain[:fired])


class TestItAdmitsWhatItCannotPromise:
    def test_every_verdict_carries_the_unexercised_caveat(self) -> None:
        """A safety belt marketed as tested and never shown to hold is worse than no belt, because
        someone installs it and stops looking."""
        for body in (_order(), _order(qty="1000000"), _order(qty="abc")):
            assert UNEXERCISED_NOTE in review(body).caveats

    def test_the_caveat_names_the_real_numbers(self) -> None:
        assert "gate 1" in UNEXERCISED_NOTE
        assert "swept domain" in UNEXERCISED_NOTE

    def test_a_narrow_is_not_permission_to_send(self) -> None:
        """A NARROW is not a yes. It is a different order, and a caller that reads it as approval
        sends the size that was refused."""
        got = review(_order(qty="1000000"))
        if got.verdict == NARROW:
            assert got.may_send is False

    def test_only_allow_permits_sending(self) -> None:
        for verdict in (NARROW, REFUSE, MALFORMED):
            assert not ReviewVerdict(verdict, "x", "y").may_send
        assert ReviewVerdict(ALLOW, "none", "y").may_send


class TestItDoesNotSubstituteOurJudgementForTheCallers:
    def test_confidence_defaults_to_full_so_gate_two_does_not_bind(self) -> None:
        """Gate 2 is the desk's hurdle about *its own* edge. Applying it to somebody else's thesis
        would be refusing their trade for not being our trade."""
        got = review(_order(qty="1"))
        assert got.binding_constraint != "min_confidence"

    def test_a_caller_may_still_supply_their_own_confidence(self) -> None:
        got = review(_order(qty="1"), confidence=0.1)
        assert got.binding_constraint == "min_confidence"
        assert got.verdict == REFUSE


class TestTheOutputIsMachineReadable:
    def test_as_dict_round_trips_through_json(self) -> None:
        blob = json.loads(json.dumps(review(_order(qty="1000000")).as_dict()))
        assert set(blob) >= {
            "reviewed_at", "verdict", "may_send_as_submitted", "binding_constraint",
            "reason", "submitted_qty", "permitted_qty", "chain", "caveats", "problems",
        }

    def test_the_rendered_form_shows_the_narrowing(self) -> None:
        got = review(_order(qty="1000000"))
        if got.verdict == NARROW:
            assert "->" in got.render()

    def test_may_send_matches_the_verdict_in_the_payload(self) -> None:
        blob = review(_order(qty="1000000")).as_dict()
        assert blob["may_send_as_submitted"] == (blob["verdict"] == ALLOW)


class TestTheBaselineIsQuotedNotRemembered:
    def test_the_docstring_cites_the_file_and_lines_it_beats(self) -> None:
        """Standing Rule #3: cite `file:line` for anything measured against a source."""
        from argus.execution import review as module

        assert "safety.ts:78-107" in (module.__doc__ or "")

    def test_the_three_baseline_checks_are_named(self) -> None:
        from argus.execution import review as module

        doc = module.__doc__ or ""
        for token in ("dryRun", "readOnly", "confirm"):
            assert token in doc, token

    @pytest.mark.parametrize("path", ["agent-sdk/src/tools/safety.ts"])
    def test_the_cited_source_still_says_what_we_say_it_says(self, path: str) -> None:
        """If Bitget's safety layer grows a fourth check, our comparison is out of date and this
        test is how we find out rather than a judge."""
        from pathlib import Path

        source = Path(__file__).resolve().parents[2] / path
        if not source.exists():
            pytest.skip("the Bitget SDK is not cloned on this machine")
        text = source.read_text(encoding="utf-8", errors="replace")
        body = text[text.index("export async function executeWithSafety"):][:1400]
        assert "if (dryRun)" in body
        assert "readOnly && op.isWrite" in body
        assert 'riskLevel === "high" && !confirm' in body
