"""The VADER baseline: the README's thresholds, a clear refusal when the clone is missing, and the
real analyzer from the local clone when it is on this machine."""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.eval.baselines import vader_loader


def test_the_thresholds_are_the_readme_s() -> None:
    assert vader_loader.label(0.05) == "positive"
    assert vader_loader.label(-0.05) == "negative"
    assert vader_loader.label(0.049) == "neutral" and vader_loader.label(-0.049) == "neutral"


def test_a_missing_clone_is_refused_with_the_fix(tmp_path: Path) -> None:
    with pytest.raises(vader_loader.VaderLoadError, match="git clone"):
        vader_loader.load_analyzer(clone=tmp_path / "absent")


def test_the_real_analyzer_scores_written_text() -> None:
    try:
        analyzer = vader_loader.load_analyzer()
    except vader_loader.VaderLoadError:
        pytest.skip("VADER clone not on this machine")
    good = analyzer.polarity_scores("Great quarter, strong guidance, love this stock")
    bad = analyzer.polarity_scores("Terrible quarter, awful guidance, hate this stock")
    assert vader_loader.label(good["compound"]) == "positive"
    assert vader_loader.label(bad["compound"]) == "negative"
    assert vader_loader.provenance()["thresholds"].startswith("compound >= 0.05")
