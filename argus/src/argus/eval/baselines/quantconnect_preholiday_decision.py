# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/QuantConnect/Tutorials
# Path:    "04 Strategy Library/83 Pre-Holiday Effect/02 Method.html", the second real code
#          block (the position-decision logic; the calendar lookup that produces `holidays` is
#          the tutorial's real first code block, not vendored here since it calls Lean's own
#          `self.TradingCalendar.GetDaysByType()`, unrunnable outside a full Lean engine — see
#          `eval/baselines/lean_market_holidays_loader.py` for the real, same-source calendar
#          data this comparison uses to supply `holidays` instead)
# Commit:  4a341890296f7e79e095508f06170c72ccaa629c (2025-07-28)
# Licence: Apache License 2.0 (Copyright QuantConnect Corporation) -- full text already vendored
#          in this directory (lean_pairs_ranking.py, quantconnect_sue.py); not repeated for this
#          four-line excerpt.
#
# The real tutorial page publishes this snippet at column zero, as free-standing statements
# reading `self`/`holidays` from an enclosing scope it never shows -- not as an indented method
# body. Rather than re-indent it (which would mean editing whitespace inside the "verbatim"
# region), `eval/baselines/quantconnect_preholiday_loader.py` execs this file's real, unedited
# text directly, injecting real `self`/`holidays` objects into the exec namespace first, the
# same technique `eval/baselines/qlib_loader.py` already uses in this directory for `re`/`abc`.
#
# One real, published inconsistency, found by reading this snippet closely rather than assumed:
# the tutorial's own real first code block computes `public_holidays = list(set(holidays) -
# set(weekends))` -- specifically to separate genuine calendar holidays from ordinary weekend
# closures, per its own prose ("As PublicHoliday includes weekends, we use the type Weekend to
# subtract weekend holidays") -- and then this real second block never uses `public_holidays` at
# all, checking the un-subtracted `holidays` count instead. Whether that is deliberate or a
# copy-paste mismatch between the tutorial's own published fragments cannot be determined from
# the page alone; this vendoring runs the real, literal, published code exactly as it reads.
#
# Apache License 2.0
#
# ============================== VENDORED FROM HERE ==============================
if not self.Portfolio.Invested and len(holidays)>0:
    self.SetHoldings("SPY", 1)
elif self.Portfolio.Invested and len(holidays)==0:
    self.Liquidate()
