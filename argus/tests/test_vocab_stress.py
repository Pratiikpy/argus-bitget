"""S2: the company renamed to nonsense, every figure held fixed, scored against resampling."""

from __future__ import annotations

from random import Random
from typing import Any

import pytest
from test_perturbations import pm_answer, seat, snapshot

from argus.eval import perturbations as pt
from argus.eval import vocab_stress as vs
from argus.market.bitget import RTOKEN_SYMBOLS

EVIDENCE = (
    "[mkt-NVDAUSDT] (news, credibility 1.00, available 2026-09-25T13:59:58+00:00) NVDAUSDT last "
    "219.37, 24h change 0.0043, quoted spread 0.46bps",
    "[sec-1] (sec-edgar, credibility 0.95, available 2026-09-24T20:05:00+00:00) NVIDIA "
    "Corporation filed an 8-K (Item 2.02) for NVDA; Nvidia guided revenue to 54.0B",
    "[rss-2] (news, credibility 0.60, available 2026-09-25T11:00:00+00:00) an apple a day; "
    "VIX 16.2 as Treasury 10y yield holds 4.12%",
)


def nvda() -> pt.Snapshot:
    return snapshot("NVDAUSDT-x", EVIDENCE)


def test_random_mapping_draws_without_replacement_like_planbench() -> None:
    mapping = vs.random_mapping(["a", "b", "c", "a"], ["x", "y", "z"], Random(1))
    assert sorted(mapping) == ["a", "b", "c"]
    assert sorted(mapping.values()) == ["x", "y", "z"]
    with pytest.raises(ValueError, match="not enough words"):
        vs.random_mapping(["a", "b"], ["x"], Random(1))


def test_entities_are_renamed_and_every_figure_survives() -> None:
    ob = vs.obfuscate(nvda(), layer="entities", seed=0)
    text = "\n".join(ob.snapshot.evidence)
    for real in ("NVDAUSDT", "NVDA", "NVIDIA", "Nvidia"):
        assert real not in text
    assert ob.numbers_fixed
    assert ob.snapshot.symbol == ob.mapping["NVDAUSDT"]
    assert ob.snapshot.symbol.endswith("USDT")
    # "Nvidia" and "NVIDIA" are one company, so they must become one invented name.
    assert ob.mapping["Nvidia"] == ob.mapping["NVIDIA"]
    # Ordinary English is not a brand, and labels are untouched in this layer.
    assert "an apple a day" in text and "8-K" in text and "VIX" in text
    assert "219.37" in text and "4.12%" in text


def test_the_label_layer_also_renames_events_and_indicators() -> None:
    ob = vs.obfuscate(nvda(), layer="entities+labels", seed=0)
    text = "\n".join(ob.snapshot.evidence)
    for label in ("8-K", "VIX", "Treasury"):
        assert label not in text
    assert ob.numbers_fixed
    assert "sec-edgar" in text  # a source tag in lower case is not the SEC label


def test_obfuscation_is_seeded_per_snapshot() -> None:
    a = vs.obfuscate(nvda(), layer="entities", seed=0)
    assert a.mapping == vs.obfuscate(nvda(), layer="entities", seed=0).mapping
    drawn = {vs.obfuscate(nvda(), layer="entities", seed=s).mapping["NVDAUSDT"] for s in range(8)}
    assert len(drawn) > 1
    with pytest.raises(ValueError):
        vs.obfuscate(nvda(), layer="everything", seed=0)


def test_no_replacement_ticker_can_be_a_real_one() -> None:
    real = set(RTOKEN_SYMBOLS) | {s.removesuffix("USDT") for s in RTOKEN_SYMBOLS}
    for word in vs.TICKER_WORDS:
        assert len(word) == 6 and word.isalpha() and word not in real
    assert len(set(vs.TICKER_WORDS)) == len(vs.TICKER_WORDS)


def test_every_symbol_in_the_universe_can_be_obfuscated() -> None:
    for symbol in RTOKEN_SYMBOLS:
        line = f"[mkt-{symbol}] (news, credibility 1.00, available 2026-09-25T13:59:58+00:00) " \
               f"{symbol} last 10.5"
        snap = pt.Snapshot.from_dict({**nvda().as_dict(), "id": f"{symbol}-t", "symbol": symbol,
                                      "evidence": [line]})
        for layer in vs.LAYERS:
            ob = vs.obfuscate(snap, layer=layer)
            assert symbol not in ob.snapshot.evidence[0] and ob.numbers_fixed


def test_a_brand_biased_desk_is_caught_and_a_blind_one_is_not() -> None:
    snaps = [nvda()]

    def brand_prior(prompt: str) -> dict[str, Any]:
        return pm_answer("trade", "BUY", 1) if "NVIDIA" in prompt else pm_answer()

    biased = seat(brand_prior)
    base = pt.collect_baseline(snaps, biased, runs=3)
    cells, made = vs.obfuscated_cells(snaps, biased)
    blob = vs.report(base, cells, made, requests=biased.requests(), failures=biased.failures)
    entities = blob["layers"]["entities"]
    assert entities["unchanged"] == 0 and entities["decisions"] == 1
    assert entities["score"]["flips_beyond_noise"]
    assert blob["numbers_fixed"] is True
    assert blob["qwen_requests"] == 5

    blind = seat(lambda _: pm_answer())
    base = pt.collect_baseline(snaps, blind, runs=3)
    cells, made = vs.obfuscated_cells(snaps, blind)
    blob = vs.report(base, cells, made, requests=blind.requests(), failures=[])
    assert blob["layers"]["entities"]["unchanged_share"] == 1.0
    assert blob["layers"]["entities+labels"]["requirement"].startswith("dependence")
