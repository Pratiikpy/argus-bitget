"""A trader's own exposure, leverage and daily-order limits (`desk/mandate.py`, audit 98)."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from argus.desk.mandate import Mandate
from argus.desk.workbench import TraderProfile

BASE = replace(TraderProfile.aggressive(), capital=Decimal("100000"),
               max_position_pct=Decimal("100"), holding_horizon_hours=24)


def _reasons(profile: TraderProfile, **kwargs: object) -> tuple[str, ...]:
    args: dict[str, object] = {"horizon_hours": 4.0, "notional": Decimal("10000"),
                               "symbol": "NVDAUSDT"}
    args.update(kwargs)
    return Mandate(profile=profile).out_of_mandate(**args)  # type: ignore[arg-type]


def test_gross_exposure_nets_the_order_into_its_own_symbol() -> None:
    capped = replace(BASE, max_gross_exposure=Decimal("25000"))
    book = {"NVDAUSDT": Decimal("-20000"), "TSLAUSDT": Decimal("10000")}
    # a buy of 10,000 against a 20,000 NVDA short leaves 10,000 + 10,000 gross: inside
    assert _reasons(capped, positions=book) == ()
    over = _reasons(capped, positions=book, side="sell")
    assert over and over[0].startswith("gross exposure after this order would be 40000")


def test_gross_leverage_is_exposure_over_capital() -> None:
    capped = replace(BASE, max_gross_leverage=Decimal("0.15"))
    assert _reasons(capped, positions={}) == ()
    over = _reasons(capped, positions={"TSLAUSDT": Decimal("10000")})
    assert over == ("gross leverage after this order would be 0.20x, above this mandate's 0.15x",)


def test_the_daily_order_limit_counts_this_order() -> None:
    capped = replace(BASE, max_trades_per_day=3)
    assert _reasons(capped, trades_today=2) == ()
    assert _reasons(capped, trades_today=3) == (
        "this would be order 4 today, over this mandate's 3 a day",)


def test_unstated_limits_and_unsupplied_books_never_refuse() -> None:
    assert _reasons(BASE, positions={"TSLAUSDT": Decimal("1e9")}, trades_today=10_000) == ()
    capped = replace(BASE, max_gross_exposure=Decimal("1"), max_gross_leverage=Decimal("0.001"),
                     max_trades_per_day=1)
    # no book and no count supplied: a limit that cannot be computed is not a breach
    assert _reasons(capped) == ()
