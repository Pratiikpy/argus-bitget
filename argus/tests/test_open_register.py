"""The register's opening batch (`register/open_register.py`), offline.

`build_batch` turns raw hourly candles into three mechanical claims per instrument; its only real
computation is the trailing-median volatility threshold, so that is pinned on hand-built candle
series where the median is known by construction, including a boundary case (exactly 24 hourly
moves, the minimum) and a zero-close bar that must be skipped rather than divide by zero.
`open_register` commits and anchors whatever `build_batch` returns; those tests replace it with a
stub batch and a stub Bitcoin anchor so the commit-and-chain machinery is exercised without ever
reaching the network or a real `data/register.jsonl`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import argus.market.history as history_mod
from argus.market.history import Candle
from argus.register import claims as claims_mod
from argus.register import open_register as reg
from argus.register.claims import Predicate

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


# --- build_batch: hand-built candle series --------------------------------------------------


def _candle(ts: datetime, close: Any) -> Candle:
    value = Decimal(str(close))
    return Candle(ts=ts, open=value, high=value, low=value, close=value, volume=Decimal("0"))


def _clean_series(start: Decimal, bps_steps: list[int], t0: datetime) -> list[Candle]:
    """``len(bps_steps) + 1`` hourly bars whose consecutive closes move by exactly the given
    number of basis points each hour, in the given order."""
    bars = [_candle(t0, start)]
    value = start
    for i, bps in enumerate(bps_steps, start=1):
        value = value * (Decimal(1) + Decimal(bps) / Decimal(10000))
        bars.append(_candle(t0 + timedelta(hours=i), value))
    return bars


def _ddd_bars(t0: datetime) -> list[Candle]:
    """26 bars: 12 clean ~5bps moves, a bar whose close is zero, then 11 more ~5bps moves.

    The move *into* the zero bar is a real, finite 10,000bps outlier (previous close is nonzero);
    the move *out of* it divides by zero and must be dropped by the `a.close > 0` guard rather
    than raising. 23 clean moves plus the one outlier is 24 moves total — exactly the minimum.
    """
    first = _clean_series(Decimal("10000"), [5] * 12, t0)
    zero = _candle(t0 + timedelta(hours=13), Decimal("0"))
    rest = _clean_series(Decimal("10000"), [5] * 11, t0 + timedelta(hours=14))
    return [*first, zero, *rest]


def test_build_batch_computes_the_volatility_threshold_from_the_trailing_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 25 bars, 24 moves of 1..24 bps: sorted, the median (index 12 of 24) is exactly 13 bps.
    # threshold = round(13 * sqrt(24), 2) = 63.69, worked out independently of the source code.
    bars = _clean_series(Decimal("10000"), list(range(1, 25)), datetime(2026, 1, 1, tzinfo=UTC))
    assert len(bars) == 25

    def fake_fetch_range(symbol: str, **kwargs: Any) -> list[Candle]:
        assert symbol == "AAAUSDT"
        assert kwargs["candle_type"] is history_mod.CandleType.MARKET
        assert kwargs["interval"] == "1H"
        return bars

    monkeypatch.setattr(history_mod, "fetch_range", fake_fetch_range)

    vol, up, down = reg.build_batch(("AAAUSDT",), now=NOW)

    assert vol.subject == up.subject == down.subject == "AAAUSDT"
    assert vol.predicate is Predicate.ABS_MOVE_ABOVE_BPS
    assert vol.threshold == "63.69"
    assert vol.confidence == pytest.approx(0.60)
    assert up.predicate is Predicate.RETURN_OVER_ABOVE_BPS
    assert down.predicate is Predicate.RETURN_OVER_BELOW_BPS
    assert up.threshold == "0" and down.threshold == "0"
    assert up.confidence == pytest.approx(0.50) and down.confidence == pytest.approx(0.50)
    for claim in (vol, up, down):
        assert claim.claimant == "argus"
        assert claim.source == "bitget:1H:market"
        assert claim.registered_at == NOW.isoformat()
        assert claim.resolves_at == (NOW + timedelta(hours=24)).isoformat()
        assert "trading halted" in claim.invalidation


def test_build_batch_skips_a_symbol_with_fewer_than_24_hourly_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _clean_series(Decimal("10000"), [5] * 23, datetime(2026, 1, 1, tzinfo=UTC))
    assert len(bars) == 24  # 23 moves: one short of the 24-move minimum

    monkeypatch.setattr(history_mod, "fetch_range", lambda symbol, **kw: bars)

    assert reg.build_batch(("CCCUSDT",), now=NOW) == []


def test_build_batch_ignores_a_zero_close_bar_without_dividing_by_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _ddd_bars(datetime(2026, 1, 1, tzinfo=UTC))
    assert len(bars) == 26
    monkeypatch.setattr(history_mod, "fetch_range", lambda symbol, **kw: bars)

    vol, _up, _down = reg.build_batch(("DDDUSDT",), now=NOW)

    # median of {23 x ~5bps, 1 x 10,000bps outlier} is still ~5bps: round(5*sqrt(24), 2) = 24.49.
    assert vol.threshold == "24.49"


def test_build_batch_skips_a_symbol_whose_history_fetch_fails_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = _clean_series(Decimal("10000"), list(range(1, 25)), datetime(2026, 1, 1, tzinfo=UTC))

    def fake_fetch_range(symbol: str, **kw: Any) -> list[Candle]:
        if symbol == "BBBUSDT":
            raise history_mod.HistoryError("venue has no history for BBBUSDT")
        return good

    monkeypatch.setattr(history_mod, "fetch_range", fake_fetch_range)

    claims = reg.build_batch(("AAAUSDT", "BBBUSDT"), now=NOW)

    assert [c.subject for c in claims] == ["AAAUSDT", "AAAUSDT", "AAAUSDT"]


# --- open_register: commit and anchor, with build_batch and the Bitcoin calendars stubbed ----


def _claim(subject: str, now: datetime = NOW) -> claims_mod.Claim:
    return claims_mod.make_claim(
        claimant="argus", subject=subject, predicate=Predicate.RETURN_OVER_ABOVE_BPS,
        threshold="0", resolves_at=now + timedelta(hours=24), source="bitget:1H:market",
        confidence=0.5, now=now,
    )


def test_open_register_commits_the_batch_without_anchoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    batch = [_claim("AAAUSDT"), _claim("BBBUSDT")]
    monkeypatch.setattr(reg, "build_batch", lambda symbols: batch)
    path = tmp_path / "register.jsonl"

    result = reg.open_register(symbols=("AAAUSDT", "BBBUSDT"), anchor_batch=False, path=path)

    assert result["claims"] == 2
    assert result["anchored"] is False
    assert result["calendars"] == []
    assert result["anchor_failures"] == []
    intact, _note = claims_mod.verify_chain(path)
    assert intact is True
    rows = claims_mod._read(path)
    assert len(rows) == 2
    assert result["head"] == rows[-1].entry_hash


def test_open_register_reports_which_calendars_accepted_the_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(reg, "build_batch", lambda symbols: [_claim("AAAUSDT")])
    cal1, cal2 = SimpleNamespace(calendar="https://cal1"), SimpleNamespace(calendar="https://cal2")
    proof = SimpleNamespace(
        anchored=True, receipts=[cal1, cal2], failures=("https://cal3: TimeoutError",),
    )
    monkeypatch.setattr("argus.paper.anchor.anchor", lambda digest, **kw: proof)

    result = reg.open_register(
        symbols=("AAAUSDT",), anchor_batch=True, path=tmp_path / "register.jsonl",
    )

    assert result["anchored"] is True
    assert result["calendars"] == ["https://cal1", "https://cal2"]
    assert result["anchor_failures"] == ["https://cal3: TimeoutError"]


def test_open_register_reports_a_failed_anchor_honestly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(reg, "build_batch", lambda symbols: [_claim("AAAUSDT")])
    proof = SimpleNamespace(
        anchored=False, receipts=[], failures=("cal1: TimeoutError", "cal2: ConnectionError"),
    )
    monkeypatch.setattr("argus.paper.anchor.anchor", lambda digest, **kw: proof)

    result = reg.open_register(
        symbols=("AAAUSDT",), anchor_batch=True, path=tmp_path / "register.jsonl",
    )

    assert result["anchored"] is False
    assert result["calendars"] == []
    assert result["anchor_failures"] == ["cal1: TimeoutError", "cal2: ConnectionError"]


def test_open_register_refuses_to_register_an_empty_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(reg, "build_batch", lambda symbols: [])

    with pytest.raises(RuntimeError, match="no claims could be built"):
        reg.open_register(symbols=("ZZZUSDT",), anchor_batch=False, path=tmp_path / "r.jsonl")


def test_open_register_chains_a_second_batch_onto_the_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path = tmp_path / "register.jsonl"
    batches = iter([
        [_claim("AAAUSDT"), _claim("BBBUSDT")],
        [_claim("CCCUSDT"), _claim("DDDUSDT"), _claim("EEEUSDT")],
    ])
    monkeypatch.setattr(reg, "build_batch", lambda symbols: next(batches))

    first = reg.open_register(symbols=("x",), anchor_batch=False, path=path)
    second = reg.open_register(symbols=("y",), anchor_batch=False, path=path)

    assert first["claims"] == 2
    assert second["claims"] == 3
    assert first["head"] != second["head"]
    intact, note = claims_mod.verify_chain(path)
    assert intact is True
    assert "5 claim(s)" in note
    rows = claims_mod._read(path)
    assert len(rows) == 5
    assert rows[2].prev_hash == rows[1].entry_hash  # the second batch chains onto the first
