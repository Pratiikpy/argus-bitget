"""The hosts ARGUS reads from, named once (audit finding 165).

Bitget's public API host was written out in twelve modules. A host is configuration, not a detail
of each reader: pointing the whole product at a regional mirror or a recorded replay server should
be one setting, and was twelve edits. ``ARGUS_BITGET_API`` overrides it; unset, it is Bitget's
public host. Timeouts and page sizes stay beside the call that needs them, because they are a
property of that endpoint (a funding history pages by 100, candles by 200, a Treasury CSV is slow)
rather than of the product.
"""

from __future__ import annotations

import os

BITGET_API = os.environ.get("ARGUS_BITGET_API", "https://api.bitget.com").rstrip("/")
"""Bitget's public REST host, without a trailing slash."""

__all__ = ["BITGET_API"]
