"""Personalisation tests.

`TraderProfile` states its own acceptance test in its docstring — "two profiles get different
verdicts on identical market state" — and nothing demonstrated it. These tests are that
demonstration, and they are shaped so the harness can fail:

* **a proposal on which every profile agrees is reported as "did not bind"**, not as a success. A
  harness that can only confirm is not evidence;
* **a horizon breach is a refusal, never a resize.** A thesis needing a month does not become
  suitable for a two-day trader at half the size;
* **a loss-tolerance breach is also a refusal**, because resizing does not change a percentage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from argus.desk.book import Book, Lot, PositionSide
from argus.desk.personalisation import (
    Outcome,
    Proposal,
    audit,
    diverge,
    judge,
    standard_profiles,
)
from argus.desk.workbench import TraderProfile

_AT = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def _book_with_open_positions(n: int) -> Book:
    book = Book()
    for i in range(n):
        book.apply_fill(
            symbol=f"SYM{i}USDT", side=PositionSide.LONG,
            lot=Lot(order_id=f"o{i}", approved_intent_hash=f"h{i}", side="buy",
                    quantity=Decimal("1"), price=Decimal("100"), commission=Decimal("0"),
                    ts_filled=_AT),
        )
    return book

CONSERVATIVE, AGGRESSIVE = standard_profiles()


def _p(
    notional: str = "4000", horizon: float = 24.0, loss: str = "2", sector: str = "technology"
) -> Proposal:
    return Proposal("NVDAUSDT", sector, Decimal(notional), horizon, Decimal(loss))


class TestTheProofTheDocstringDemanded:
    def test_two_profiles_reach_different_verdicts_on_identical_state(self) -> None:
        """The acceptance test, run. 8% bad case: conservative refuses, aggressive takes it."""
        case = diverge(_p(notional="20000", loss="8"), standard_profiles())
        assert case.diverged
        assert case.verdicts[0].outcome is Outcome.REFUSED
        assert case.verdicts[1].outcome is Outcome.TAKEN

    def test_the_divergence_is_named_in_the_record(self) -> None:
        case = diverge(_p(notional="20000", loss="8"), standard_profiles())
        assert any("personalisation bound here" in line for line in case.render())

    def test_one_genuine_divergence_is_evidence(self) -> None:
        report = audit([_p(notional="20000", loss="8")], standard_profiles())
        assert report.is_evidence


class TestTheHarnessCanFail:
    def test_agreement_is_reported_as_not_binding(self) -> None:
        """Three of four shipped proposals do not diverge, and the module says so."""
        case = diverge(_p(), standard_profiles())
        assert not case.diverged
        assert any("did not bind here" in line for line in case.render())

    def test_agreement_is_not_dressed_up_as_success(self) -> None:
        case = diverge(_p(), standard_profiles())
        text = " ".join(case.render())
        assert "not evidence that profiles work" in text

    def test_a_set_that_never_separates_is_not_evidence(self) -> None:
        report = audit([_p(), _p(notional="1000")], standard_profiles())
        assert not report.is_evidence
        assert any("not demonstrated" in line for line in report.render())

    def test_no_proposals_at_all_is_untested(self) -> None:
        report = audit([], standard_profiles())
        assert report.rate is None
        assert "untested" in " ".join(report.render())

    def test_the_rate_counts_only_genuine_divergence(self) -> None:
        report = audit(
            [_p(notional="20000", loss="8"), _p(), _p(), _p()], standard_profiles()
        )
        assert report.rate == pytest.approx(0.25)


class TestAHorizonBreachIsARefusal:
    def test_a_long_thesis_is_refused_by_a_short_mandate(self) -> None:
        got = judge(AGGRESSIVE, _p(horizon=2000.0))
        assert got.outcome is Outcome.REFUSED

    def test_it_is_never_resized_into_suitability(self) -> None:
        """Halving the size does not shorten the thesis."""
        got = judge(AGGRESSIVE, _p(notional="1", horizon=2000.0))
        assert got.outcome is Outcome.REFUSED
        assert got.permitted_notional == Decimal("0")

    def test_the_reason_says_it_is_a_different_trader_s_trade(self) -> None:
        got = judge(AGGRESSIVE, _p(horizon=2000.0))
        assert "different trader's trade" in got.reasons[0]

    def test_a_horizon_inside_the_mandate_slack_is_permitted(self) -> None:
        """The mandate carries slack; a thesis just over the nominal horizon still fits."""
        got = judge(AGGRESSIVE, _p(horizon=float(AGGRESSIVE.holding_horizon_hours) + 1))
        assert got.outcome is not Outcome.REFUSED


class TestALossBreachIsAlsoARefusal:
    def test_a_bad_case_beyond_tolerance_is_refused(self) -> None:
        assert judge(CONSERVATIVE, _p(loss="8")).outcome is Outcome.REFUSED

    def test_resizing_does_not_change_a_percentage(self) -> None:
        tiny = judge(CONSERVATIVE, _p(notional="10", loss="8"))
        assert tiny.outcome is Outcome.REFUSED
        assert "does not change a percentage" in tiny.reasons[0]

    def test_a_bad_case_inside_tolerance_is_fine(self) -> None:
        assert judge(CONSERVATIVE, _p(loss="2")).outcome is Outcome.TAKEN

    def test_exactly_at_tolerance_is_permitted(self) -> None:
        loss = str(CONSERVATIVE.loss_tolerance_pct)
        assert judge(CONSERVATIVE, _p(loss=loss)).outcome is Outcome.TAKEN


class TestSizeIsAResizeBecauseTheIdeaSurvives:
    def test_an_oversized_ticket_is_cut_to_the_limit(self) -> None:
        got = judge(CONSERVATIVE, _p(notional="50000"))
        assert got.outcome is Outcome.RESIZED
        assert got.permitted_notional == CONSERVATIVE.capital * CONSERVATIVE.max_position_pct / 100

    def test_the_reason_names_the_limit_and_the_capital(self) -> None:
        got = judge(CONSERVATIVE, _p(notional="50000"))
        assert "position limit" in got.reasons[0] and "capital" in got.reasons[0]

    def test_a_resize_still_carries_risk(self) -> None:
        assert Outcome.RESIZED.carries_risk

    def test_a_refusal_carries_none(self) -> None:
        assert not Outcome.REFUSED.carries_risk

    def test_sizes_can_diverge_even_when_outcomes_match(self) -> None:
        case = diverge(_p(notional="50000", loss="2"), standard_profiles())
        assert case.sizes_diverged

    def test_that_case_is_reported_as_binding_on_size(self) -> None:
        case = diverge(_p(notional="50000", loss="2"), standard_profiles())
        if not case.diverged:
            assert any("bound on size" in line for line in case.render())


class TestTheSeverityOrderMatters:
    def test_a_horizon_breach_beats_a_size_breach(self) -> None:
        """Both wrong: the refusal must win, because a resize would imply the trade could fit."""
        got = judge(CONSERVATIVE, _p(notional="999999", horizon=5000.0))
        assert got.outcome is Outcome.REFUSED
        assert "different trader's trade" in got.reasons[0]

    def test_a_loss_breach_beats_a_size_breach(self) -> None:
        got = judge(CONSERVATIVE, _p(notional="999999", loss="50"))
        assert got.outcome is Outcome.REFUSED
        assert "percentage" in got.reasons[0]


class TestTheRecordIsUsable:
    def test_a_verdict_serialises(self) -> None:
        payload = judge(CONSERVATIVE, _p()).as_dict()
        for key in ("profile", "outcome", "permitted_notional", "reasons"):
            assert key in payload

    def test_a_divergence_serialises_every_profile(self) -> None:
        payload = diverge(_p(), standard_profiles()).as_dict()
        assert len(payload["verdicts"]) == 2

    def test_the_report_states_both_numbers(self) -> None:
        payload = audit([_p(), _p(notional="20000", loss="8")], standard_profiles()).as_dict()
        assert payload["proposals"] == 2 and payload["diverged"] == 1

    def test_every_verdict_carries_a_reason(self) -> None:
        for profile in standard_profiles():
            assert judge(profile, _p()).reasons

    def test_a_custom_profile_works_as_well_as_the_presets(self) -> None:
        custom = TraderProfile(
            name="tiny", capital=Decimal("1000"), max_position_pct=Decimal("1"),
            max_sector_pct=Decimal("5"), holding_horizon_hours=4,
            loss_tolerance_pct=Decimal("1"),
        )
        assert judge(custom, _p()).outcome is not Outcome.TAKEN


class TestTheMandateBindsInTheLiveDesk:
    """`agents/mandate.py` and `TraderProfile` existed and were wired to nothing.

    Track 3 scores "personalised thesis", so a profile that cannot change a live decision is a
    claim rather than a feature. These check the limits actually differ and that the narrowing is
    asymmetric — the mandate may reduce or refuse and can never enlarge.
    """

    def test_two_profiles_place_different_ceilings_on_the_same_capital(self) -> None:
        from argus.agents.mandate import Mandate

        ceilings = {p.name: Mandate(profile=p).max_position_notional for p in standard_profiles()}
        assert len(set(ceilings.values())) == 2

    def test_the_same_notional_breaches_one_mandate_and_not_the_other(self) -> None:
        from argus.agents.mandate import Mandate

        conservative, aggressive = standard_profiles()
        notional = Decimal("20000")
        assert Mandate(profile=conservative).out_of_mandate(horizon_hours=40.0, notional=notional)
        assert not Mandate(profile=aggressive).out_of_mandate(horizon_hours=40.0, notional=notional)

    def test_the_desk_accepts_a_profile(self) -> None:
        """Wired, not merely importable: the parameter exists on the live entry point."""
        import inspect

        from argus.agents.desk import TradingDesk

        assert "profile" in inspect.signature(TradingDesk.run).parameters

    def test_a_permitted_size_is_never_larger_than_what_was_proposed(self) -> None:
        """The asymmetry. A mandate that could enlarge would be a second decision-maker."""
        for profile in standard_profiles():
            for notional in ("100", "5000", "50000"):
                got = judge(profile, _p(notional=notional))
                assert got.permitted_notional <= Decimal(notional)


class TestTheMandateRespectsTheBook:
    """`TraderProfile.max_concurrent_positions`'s own docstring: "a cap the book must respect" —
    and until this book existed (Foundation 3, 2026-09-15), nothing here could respect it. Same
    defect class already found and fixed once for `excluded_symbols`: a declared profile field the
    judge silently never read."""

    def test_no_book_leaves_the_cap_inert(self) -> None:
        """`book=None` is "not supplied", not "zero positions open" — the cap must not fire."""
        conservative, _aggressive = standard_profiles()
        got = judge(conservative, _p(notional="4000", loss="2"))
        assert got.outcome is Outcome.TAKEN

    def test_a_book_under_the_cap_does_not_refuse(self) -> None:
        conservative, _aggressive = standard_profiles()
        book = _book_with_open_positions(2)  # conservative's cap is 3
        got = judge(conservative, _p(notional="4000", loss="2"), book=book)
        assert got.outcome is Outcome.TAKEN

    def test_a_book_at_the_cap_refuses_rather_than_resizes(self) -> None:
        conservative, _aggressive = standard_profiles()
        book = _book_with_open_positions(3)  # at the cap
        got = judge(conservative, _p(notional="4000", loss="2"), book=book)
        assert got.outcome is Outcome.REFUSED
        assert got.permitted_notional == Decimal("0")
        assert any("already open" in r for r in got.reasons)

    def test_a_profile_with_no_concurrency_cap_is_never_stopped_by_the_book(self) -> None:
        """aggressive's max_concurrent_positions=0 means unlimited, not zero-allowed."""
        _conservative, aggressive = standard_profiles()
        book = _book_with_open_positions(50)
        got = judge(aggressive, _p(notional="4000", loss="2"), book=book)
        assert got.outcome is Outcome.TAKEN

    def test_a_full_book_makes_two_profiles_diverge_on_identical_state(self) -> None:
        """The module's own acceptance test, driven by book state rather than proposal content:
        identical proposal, identical book, different verdicts."""
        book = _book_with_open_positions(3)
        case = diverge(_p(notional="4000", loss="2"), standard_profiles(), book=book)
        assert case.diverged
        conservative_verdict, aggressive_verdict = case.verdicts
        assert conservative_verdict.outcome is Outcome.REFUSED
        assert aggressive_verdict.outcome is Outcome.TAKEN

    def test_audit_threads_the_book_through_every_proposal(self) -> None:
        book = _book_with_open_positions(3)
        conservative, _aggressive = standard_profiles()
        report = audit([_p(notional="4000", loss="2")], (conservative,), book=book)
        assert report.cases[0].verdicts[0].outcome is Outcome.REFUSED


class TestTheDemonstrationLeavesEvidence:
    """**It printed its result and wrote nothing, and "personalized thesis" is a judged criterion.**

    Every other capability in this project lands an artefact that `eval/docclaims.py` can read and
    a document can cite. This one produced a divergence rate that existed only in whichever
    terminal last ran it — which is an assertion, the exact thing the module's own docstring says
    it was built to replace.
    """

    def test_running_it_writes_an_artefact(self, tmp_path, monkeypatch) -> None:
        import json

        import argus.desk.personalisation as mod

        target = tmp_path / "personalisation.json"
        monkeypatch.setattr(mod, "REPORT_PATH", target)
        assert mod.main() == 0
        written = json.loads(target.read_text(encoding="utf-8"))
        assert "divergence_rate" in written
        assert written.get("cases")

    def test_the_artefact_names_what_it_does_not_claim(self, tmp_path, monkeypatch) -> None:
        """A divergence rate with no scope statement invites the reading that higher is better.

        It is not: a profile set that disagrees about everything is as useless as one that agrees
        about everything.
        """
        import json

        import argus.desk.personalisation as mod

        target = tmp_path / "personalisation.json"
        monkeypatch.setattr(mod, "REPORT_PATH", target)
        mod.main()
        scope = json.loads(target.read_text(encoding="utf-8"))["scope_statement"]
        assert "NOT CLAIMED" in scope

    def test_cases_where_personalisation_did_not_bind_are_kept(
        self, tmp_path, monkeypatch
    ) -> None:
        """Dropping the non-diverging cases would turn a measurement into a highlight reel."""
        import json

        import argus.desk.personalisation as mod

        target = tmp_path / "personalisation.json"
        monkeypatch.setattr(mod, "REPORT_PATH", target)
        mod.main()
        cases = json.loads(target.read_text(encoding="utf-8"))["cases"]
        assert any(not c["diverged"] for c in cases), (
            "every case diverged, so either the fixtures changed or the non-diverging ones "
            "are being filtered out of the artefact"
        )
