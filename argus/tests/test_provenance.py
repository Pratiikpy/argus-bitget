"""Per-line provenance: each line says whether it was read live, computed, measured, logged,
assumed or missing — and a line no rule knows says nothing rather than a guess."""

from __future__ import annotations

from typing import Any

import pytest

from argus.lui import provenance
from argus.lui.provenance import label


@pytest.mark.parametrize(("line", "expected"), [
    ("NVDA last 225.82 USDT on Bitget (+1.12% over 24h); bid 225.82 / ask 225.83", "live"),
    ("Open interest: 778,416.78 ETH held open on Bitget, about $2,092,462,146.", "live"),
    ("fed funds: 3.88% on 2026-09-23 (+25bp over the last 44 readings, about six weeks).",
     "live"),
    ("Crypto fear & greed: 71 (Greed); over the last 8 days it ranged 56 to 78.", "live"),
    ("Implied open: NVDA's perpetual has moved +0.49% since the Thu 20:00 UTC close; the open "
     "was missed by 30bps on average", "computed"),
    ("Your premise that NVDA is up: holds — NVDA is +1.18% over 24 hours", "computed"),
    ("If QQQ falls 10%, the book moves about -10.5% through beta alone.", "computed"),
    ("How far to trust that beta: on 1,800 test books, it missed the next four weeks' realised "
     "beta by 0.24 on average.", "record"),
    ("Tested on nights it had not seen: over the last 50 of 165 nights", "record"),
    ("seq 664 · NVDAUSDT · no_trade · decided 2026-09-24T19:41:11+00:00", "desk"),
    ("Evidence screen: 66 evidence item(s) screened, none withheld.", "desk"),
    ("Assumed: no size was given, so NVDA is assessed at 20%.", "assumed"),
    ("Sized on a $100,000 book — say your book's value for its own sizes.", "assumed"),
    ("Your premise that the contract trades at a premium: not checkable — the stock's own quote "
     "did not arrive.", "missing"),
    ("sharpe: not available — the window is 13 day(s)", "missing"),
    ("Data: Bitget live ticker. This is analysis, not advice — you make the call.", None),
    ("Sources reached: 4 of 4 answered.", None),
    ("A sentence no rule was written for.", None),
])
def test_each_line_family_gets_its_label(line: str, expected: str | None) -> None:
    assert label(line) == expected


def test_missing_wins_over_everything_a_line_also_says() -> None:
    assert label("Actionable: the quote did not arrive, so there is no gap to measure") \
        == "missing"


def test_the_reviewed_sample_still_reads_as_reviewed() -> None:
    from argus.eval.provenance_audit import audit

    report = audit()
    assert report["coverage"] == 1.0 and report["agreement"] == 1.0


def test_every_surface_carries_the_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus.lui import mcp_server, telegram_bot

    payload: dict[str, Any] = {"lines": ["Assumed: 20%.", "Data: Bitget."],
                               "sources": [{"ref": "bitget tickers"}]}
    text = telegram_bot.format_answer(payload, "q", "")
    assert "Assumed: 20%. <i>· assumed</i>" in text and "Data: Bitget.\n" in text + "\n"
    assert mcp_server._answer_text(payload).startswith("[assumed] Assumed: 20%.\nData: Bitget.")
    assert provenance.LABELS == ("live", "computed", "record", "desk", "assumed", "missing",
                                 "memory")
    # every label has its meaning on the page, so a chip is never unexplained
    from argus.lui.server import PAGE

    assert all(f"{name}:" in PAGE for name in provenance.LABELS)
