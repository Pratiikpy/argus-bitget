# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/TauricResearch/TradingAgents
# Path:    tradingagents/dataflows/errors.py
# Commit:  be952b8eccb49720509af544c6675233bc1f10d0 (2026-09-07)
# Licence: Apache License 2.0 — see `tradingagents_feedlist_interface.py`'s header for the full
#          attribution reasoning; identical here, same repository, same commit.
#
# The vendor-error taxonomy `route_to_vendor()` (in the sibling vendored `interface.py`) actually
# catches: `VendorError`, `NoMarketDataError`, `VendorRateLimitError`, `VendorNotConfiguredError`.
# Vendored unedited so the comparison's synthetic vendor stand-ins raise the REAL exception
# classes the real router dispatches on, not lookalikes.
#
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION (summarised terms retained from the
# upstream LICENSE file; the full canonical text is at http://www.apache.org/licenses/LICENSE-2.0)
#
# 1. Definitions — "License", "Licensor", "Legal Entity", "You", "Source"/"Object" form, "Work",
#    "Derivative Works", "Contribution", "Contributor" as defined by the Apache 2.0 license text.
# 2. Grant of Copyright License — perpetual, worldwide, non-exclusive, no-charge, royalty-free,
#    irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display,
#    publicly perform, sublicense, and distribute the Work and Derivative Works.
# 3. Grant of Patent License — a patent license as stated in the License, terminating upon
#    initiating patent litigation over the Work.
# 4. Redistribution — permitted in Source or Object form provided this License and all
#    copyright/patent/trademark/attribution notices are retained, and any modified files carry
#    prominent notices of the changes (none made here — this file is unmodified).
# 5. Submission of Contributions is under this License unless stated otherwise.
# 6. Trademarks — this License grants no permission to use the Licensor's trade names, trademarks,
#    or service marks.
# 7. Disclaimer of Warranty — the Work is provided "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF
#    ANY KIND, express or implied.
# 8. Limitation of Liability — no Contributor is liable for damages arising from use of the Work.
# 9. Accepting Warranty or Additional Liability — offered only on the offeror's own responsibility.
#
# Copyright TauricResearch (see attribution note above)
# Licensed under the Apache License, Version 2.0; you may not use this file except in compliance
# with the License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#
# ============================== VENDORED FROM HERE ==============================
"""Vendor data-error taxonomy.

A single hierarchy so the routing layer reacts by *behavior*, not by vendor:
every condition where a vendor cannot return usable data derives from
``VendorError``, and the router catches the base types. A new vendor raises
these (or a thin vendor-named subclass) and needs no new ``except`` clause.

    VendorError
    ├── NoMarketDataError          no usable rows (empty result OR stale data)
    ├── VendorRateLimitError       transient throttle -> skip to next vendor
    └── VendorNotConfiguredError   missing API key/config -> vendor unavailable

The number of types is the number of distinct router reactions, not the number
of human-describable causes: empty and stale data get identical handling, so
they share ``NoMarketDataError`` and differ only in the free-text ``detail``.
"""

from __future__ import annotations


class VendorError(Exception):
    """Base for any condition where a vendor could not return usable data."""


class NoMarketDataError(VendorError):
    """A vendor returned no usable rows for a symbol (empty result or stale data).

    Carries both the symbol the user requested and the canonical symbol the
    vendor was actually queried with, plus a free-text ``detail``, so callers
    can build a clear message instead of emitting a vendor-specific empty
    string into the data channel.
    """

    def __init__(self, symbol: str, canonical: str | None = None, detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class VendorRateLimitError(VendorError):
    """A vendor throttled the request; the router skips to the next vendor."""


class VendorNotConfiguredError(VendorError, ValueError):
    """A vendor was selected but its API key/configuration is missing.

    Also a ``ValueError`` so existing callers that catch ``ValueError`` keep
    working while the routing layer can treat it as "vendor unavailable".
    """
