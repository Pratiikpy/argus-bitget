"""The compare answer's lead names a riskier name only when the betas differ
(`lui/research/dispatch._compare_lead`; stranger QA of 2026-09-29)."""

from __future__ import annotations

from argus.lui.research.dispatch import _compare_lead


def test_betas_inside_the_tie_band_name_no_winner() -> None:
    rows = [{"symbol": "XAUUSDT", "beta_open": 0.79, "realised_vol": 0.23},
            {"symbol": "BTCUSDT", "beta_open": 0.78, "realised_vol": 0.34}]
    lead = _compare_lead(rows)
    assert "carry about the same market risk per dollar (beta 0.79 and 0.78)" in lead
    assert "but BTC swings the most on its own" in lead
    rows[1]["beta_open"] = 0.40
    assert "XAU carries the most market risk per dollar of the 2" in _compare_lead(rows)
