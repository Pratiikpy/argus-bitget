"""The entity gate: an instrument a thesis names must appear in the evidence it cites.

`agents/grounding.py` enforces this for numbers. This is the same property for instruments, and
the tests below are shaped around the three ways a gate like this is normally useless:

* it passes a substring (``COIN`` inside ``BITCOIN``),
* it switches itself off in Chinese, because ``\\b`` finds no boundary between Han and Latin,
* it passes a thesis that cites nothing, by having nothing to disagree with.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from argus.agents.entitygate import check, extract, supported_only
from argus.truth.evidence import Evidence

AT = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def _ev(ident: str, claim: str) -> Evidence:
    return Evidence(id=ident, claim=claim, source="test", available_at=AT, credibility=0.9)


class TestTheGateCatchesTheInvention:
    def test_an_instrument_absent_from_every_source_is_unsupported(self) -> None:
        """The whole property in one assertion."""
        report = check(
            "NVDA is strong and TSLA will follow",
            evidence=[_ev("e1", "NVDA raised guidance")],
        )
        assert [m.symbol for m in report.unsupported] == ["TSLAUSDT"]
        assert not report.grounded

    def test_an_instrument_the_evidence_names_is_supported_and_cites_it(self) -> None:
        report = check("NVDA is strong", evidence=[_ev("e1", "NVDA raised guidance")])
        assert report.grounded
        assert report.mentions[0].supported_by == ("e1",)

    def test_every_backing_source_is_named_not_just_the_first(self) -> None:
        """A reader checking the gate needs to see all of them, not a sample."""
        report = check(
            "NVDA",
            evidence=[_ev("e1", "NVDA up"), _ev("e2", "NVDA guidance"), _ev("e3", "AAPL flat")],
        )
        assert report.mentions[0].supported_by == ("e1", "e2")


class TestTheThreeWaysThisGoesWrong:
    def test_a_substring_does_not_count_as_a_mention(self) -> None:
        """``COIN`` sits inside ``BITCOIN`` and inside ``coincide``.

        A naive ``in`` check passes both and the gate becomes decoration.
        """
        report = check(
            "COIN looks cheap", evidence=[_ev("e1", "BITCOIN rallied; the two coincide")]
        )
        assert not report.grounded, "BITCOIN must not support COIN"

    def test_it_still_works_when_the_evidence_is_chinese(self) -> None:
        """Han characters are word characters, so ``\\bNVDA\\b`` matches nothing in 关于NVDA的看法.

        Half this desk's corpus is Chinese. `lui/question.py` and `eval/ngrambench.py` each
        shipped this bug before it was found by running them.
        """
        report = check("NVDA", evidence=[_ev("e1", "关于NVDA的分析师目标价上调")])
        assert report.grounded
        assert report.mentions[0].supported_by == ("e1",)

    def test_a_thesis_citing_nothing_grounds_nothing(self) -> None:
        """An empty evidence set is the easiest way to accidentally pass everything."""
        report = check("NVDA and TSLA both look strong", evidence=[])
        assert not report.grounded
        assert len(report.unsupported) == 2


class TestItDoesNotOverreach:
    @pytest.mark.parametrize("word", ["EPS", "ETF", "USD", "GDP", "CEO", "BUY", "SELL", "TWAP"])
    def test_units_and_verbs_are_not_instruments(self, word: str) -> None:
        """Flagging these would bury the real findings under noise."""
        assert extract(f"{word} moved") == ()

    def test_both_spellings_of_one_instrument_collapse(self) -> None:
        """NVDA and NVDAUSDT are the same thing; reporting them apart would make one of them
        look unsupported whenever the evidence used the other spelling."""
        assert extract("NVDA and NVDAUSDT") == ("NVDAUSDT",)

    def test_evidence_using_the_venue_spelling_supports_the_bare_ticker(self) -> None:
        report = check("NVDA", evidence=[_ev("e1", "NVDAUSDT printed a new high")])
        assert report.grounded

    def test_an_evidenced_instrument_outside_the_universe_is_a_separate_finding(self) -> None:
        """Evidenced-but-untradeable is not the same failure as invented, and the report keeps
        them apart rather than collapsing both into 'rejected'."""
        report = check("BOND looks rich", evidence=[_ev("e1", "BOND yields rose")])
        assert report.grounded, "it is evidenced"
        assert [m.symbol for m in report.untradeable] == ["BOND"]


class TestTheRateIsNotTheVerdict:
    def test_one_invention_among_four_real_names_still_fails(self) -> None:
        """A ratio would let exactly this through, which is why `grounded` is all-or-nothing."""
        evidence = [_ev("e1", "NVDA AAPL MSFT all rose")]
        report = check("NVDA, AAPL and MSFT are up, so TSLA follows", evidence=evidence)
        assert report.rate == pytest.approx(0.75)
        assert not report.grounded

    def test_a_thesis_naming_no_instrument_is_vacuously_grounded(self) -> None:
        report = check("the hurdle exceeds the observed move", evidence=[])
        assert report.grounded
        assert report.rate == 1.0


class TestTheListFilter:
    def test_it_drops_the_unevidenced_ones(self) -> None:
        """DeepEar's filter shape, for models that return a list instead of prose."""
        kept = supported_only(
            ["NVDAUSDT", "TSLAUSDT"], evidence=[_ev("e1", "NVDA raised guidance")]
        )
        assert kept == ("NVDAUSDT",)

    def test_it_keeps_nothing_when_there_is_no_evidence(self) -> None:
        assert supported_only(["NVDAUSDT"], evidence=[]) == ()


class TestTheReportIsReadable:
    def test_it_says_what_it_did_not_do(self) -> None:
        """The gate reports an unsupported instrument; it must never quietly swap in a supported
        one, because that invents a different claim and hides that it did."""
        report = check("TSLA", evidence=[_ev("e1", "NVDA rose")])
        text = "\n".join(report.render())
        assert "UNSUPPORTED" in text
        assert "never silently replaced" in text

    def test_the_dict_carries_both_failure_kinds(self) -> None:
        report = check("TSLA and BOND", evidence=[_ev("e1", "BOND yields rose")])
        payload = report.as_dict()
        assert payload["unsupported"] == ["TSLAUSDT"]
        assert payload["untradeable"] == ["BOND"]
