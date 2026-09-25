"""S3: one wrong figure in the feed, one wrong step in an upstream argument, against noise."""

from __future__ import annotations

from decimal import Decimal
from random import Random
from typing import Any

import pytest
from test_perturbations import outcome, pm_answer, seat, snapshot

from argus.eval import feedbugged as fb
from argus.eval import perturbations as pt


def frame() -> pt.Snapshot:
    return snapshot("NVDAUSDT-f")


def test_every_bug_kind_changes_the_value_and_never_returns_the_truth() -> None:
    for kind in fb.BUG_KINDS:
        for seed in range(20):
            wrong = fb.corrupt("219.37", kind, Random(seed))
            assert wrong != "219.37"
    assert fb.corrupt("219.37", "explicit_error", Random(0)) == "NaN"
    assert fb.corrupt("219.37", "incorrect_type", Random(0)) in fb.INCORRECT_TYPE_WORDS
    assert fb.corrupt("0.0043", "sign_flip", Random(0)) == "-0.0043"
    assert fb.corrupt("+2.84", "sign_flip", Random(0)) == "-2.84"
    assert fb.corrupt("-0.5", "sign_flip", Random(0)) == "0.5"
    assert fb.corrupt("0", "sign_flip", Random(0)) in {"1", "-1"}
    assert fb.corrupt("15.11", "small_offset", Random(0)) in {"14.11", "16.11"}
    assert fb.corrupt("15.11", "large_offset", Random(0)) in {"5.11", "25.11"}
    # upstream random_output: an integer of the same order of magnitude, either sign
    assert 10 <= abs(int(fb.corrupt("15.11", "random_output", Random(0)))) <= 99
    with pytest.raises(ValueError):
        fb.corrupt("15.11", "gremlins", Random(0))


def test_only_claims_are_searched_never_headers() -> None:
    found = fb.features(frame())
    names = [f.name for f in found]
    assert names == ["price", "change_24h", "spread_bps", "vix", "fear_greed"]
    for f in found:
        line = frame().evidence[f.line]
        assert line[f.start:f.end] == f.value
        assert f.start > line.index(")")


def test_a_planted_price_is_cross_checkable_and_a_planted_vix_is_not() -> None:
    price = fb.plant(frame(), kind="small_offset", feature="price")
    assert price is not None and price.cross_checkable
    assert "last 219.37" not in price.snapshot.evidence[0]
    assert "token price: 219.37" in price.snapshot.frame().to_prompt_block()
    vix = fb.plant(frame(), kind="sign_flip", feature="vix")
    assert vix is not None and not vix.cross_checkable
    assert "VIX -16.2" in vix.snapshot.evidence[1]
    # everything outside the planted figure is untouched
    assert vix.snapshot.evidence[0] == frame().evidence[0]
    assert fb.plant(snapshot("empty", ("[x] (news, credibility 1.00, available t) nothing",)),
                    kind="sign_flip") is None


def test_correct_and_incorrect_reasoning_share_the_correct_steps() -> None:
    correct = fb.reasoning_line(frame(), wrong=False)
    wrong = fb.reasoning_line(frame(), wrong=True)
    assert correct is not None and wrong is not None
    shared = "NVDAUSDT moved 0.0043 over 24h, which is 43bps. The total hurdle is 13.93bps"
    assert shared in correct and shared in wrong
    assert correct.startswith("[panel] quant: neutral 0bps")
    # the wrong step is the 100x unit error, and it argues for the move's own direction
    assert "at 4300bps the move is" in wrong and wrong.startswith("[panel] quant: bullish 2150bps")
    down = snapshot("d", (frame().evidence[0].replace("0.0043", "-0.0043"),))
    assert (fb.reasoning_line(down, wrong=True) or "").startswith("[panel] quant: bearish")
    assert fb.reasoning_line(snapshot("n", ("no change here",)), wrong=True) is None


def test_precision_recall_fscore_matches_upstream_conventions() -> None:
    assert fb.precision_recall_fscore([(True, True), (False, True), (True, False), (False, False)])\
        == (1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5)
    # upstream: a class never predicted scores 0, not undefined
    assert fb.precision_recall_fscore([(False, True), (False, False)])[5:] == (0.0, 0.0, 0.0)


def test_detection_reads_the_models_own_words() -> None:
    flagged = pt.Outcome(verdict="data_insufficient", side="buy", quantity="0", lean="none",
                         confidence=0.4, thesis="The VIX reading of -16.2 is implausible.",
                         reasoning="")
    assert fb.predicted_bug(flagged)
    assert not fb.predicted_bug(outcome())

    def said(text: str) -> bool:
        return fb.predicted_bug(pt.Outcome(verdict="no_trade", side="buy", quantity="0",
                                           lean="none", confidence=0.5, thesis=text, reasoning=""))

    # the defect the first live run exposed: "nan" inside "financial" is not a detection
    assert not said("Strong financial results and a finance-led rally; no terror, no typology.")
    assert said("The price reads NaN.") and said("A data error in the feed.")
    assert said("Two errors in the VIX line.") and said("This looks anomalous.")


def test_a_gullible_desk_and_a_careful_one_are_told_apart() -> None:
    snaps = [snapshot("A"), snapshot("B")]

    def gullible(prompt: str) -> dict[str, Any]:
        # Trades any bullish panel claim and never questions a figure.
        return pm_answer("trade", "BUY", 1) if "quant: bullish" in prompt else pm_answer()

    s = seat(gullible)
    base = pt.collect_baseline(snaps, s, runs=3)
    cases = fb.run_cases(snaps, s)
    blob = fb.report(base, cases, requests=s.requests(), failures=s.failures)
    ir = blob["conditions"]["incorrect_reasoning"]
    assert ir["moved"] == 2 and ir["adopted_wrong_direction"] == 2
    assert blob["conditions"]["correct_reasoning"]["moved"] == 0
    assert blob["error_recovery"] == {**blob["error_recovery"], "NR_unchanged_rate": 1.0,
                                      "CR_unchanged_rate": 1.0, "IR_unchanged_rate": 0.0}
    assert blob["detection"]["tp"] == 0 and blob["detection"]["recall"] == 0.0
    assert blob["qwen_requests"] == 6 + 2 + 2 + 2

    def careful(prompt: str) -> dict[str, Any]:
        answer = pm_answer("data_insufficient")
        known = ("last 219.37", "24h change 0.0043", "VIX 16.2")
        if any(k not in prompt for k in known):
            answer["thesis"] = "A feed value is inconsistent with the rest of the frame."
        return answer

    s = seat(careful)
    base = pt.collect_baseline(snaps, s, runs=3)
    cases = fb.run_cases(snaps, s)
    blob = fb.report(base, cases, requests=s.requests(), failures=s.failures)
    assert blob["detection"]["tp"] == 2 and blob["detection"]["fp"] == 0
    assert blob["detection"]["f1"] == 1.0
    assert blob["conditions"]["incorrect_reasoning"]["adopted_wrong_direction"] == 0
    assert set(blob["by_feature"]) == {"price", "change_24h"}
    assert blob["cross_checkable_cases"] == 1


def test_cases_deal_every_kind_and_feature_once() -> None:
    snaps = [snapshot(f"S{i}") for i in range(6)]
    s = seat(lambda _: pm_answer())
    cases = [c for c in fb.run_cases(snaps, s) if c.cell.condition == "bugged"]
    assert sorted(c.detail["kind"] for c in cases) == sorted(fb.BUG_KINDS)
    # the test frame carries five of the six priority features; the sixth falls back
    assert {c.detail["feature"] for c in cases} >= {"price", "change_24h", "vix", "fear_greed",
                                                    "spread_bps"}
    for case in cases:
        assert case.detail["wrong"] != case.detail["true"]
        Decimal(case.detail["true"])  # the planted figure was a number before it was corrupted
