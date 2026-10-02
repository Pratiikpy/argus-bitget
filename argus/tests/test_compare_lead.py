"""The compare answer's lead names a riskier name only when the betas differ
(`lui/research/dispatch._compare_lead`; stranger QA of 2026-09-29)."""

from __future__ import annotations

from argus.lui.research.dispatch import _compare_lead


def test_betas_inside_the_tie_band_name_no_winner() -> None:
    rows = [{"symbol": "XAUUSDT", "beta_open": 0.79, "realised_vol": 0.23},
            {"symbol": "BTCUSDT", "beta_open": 0.78, "realised_vol": 0.34}]
    lead = _compare_lead(rows)
    # with the betas tied, the name that swings further is the riskier one, said first (row 641)
    assert lead.startswith("Bottom line: BTC is the riskier — it swings about 34% a year against "
                           "XAU's 23%")
    assert "market risk per dollar is about the same (beta 0.79 and 0.78)" in lead
    rows[1]["realised_vol"] = 0.25
    same = _compare_lead(rows)
    assert "carry about the same market risk per dollar (beta 0.79 and 0.78)" in same
    assert "but BTC swings the most on its own" in same
    rows[1]["beta_open"] = 0.40
    assert "XAU carries the most market risk per dollar of the 2" in _compare_lead(rows)
