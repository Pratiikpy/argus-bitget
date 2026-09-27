"""The bounded mapping the console's user-keyed caches share (`truth/bounded.py`, audit 166)."""

from __future__ import annotations

import pytest

from argus.lui import exposures, research, server, translate
from argus.market import equity_history
from argus.truth.bounded import BoundedDict


def test_the_oldest_write_goes_first_and_a_rewrite_counts_as_new() -> None:
    cache: BoundedDict[str, int] = BoundedDict(2)
    cache["a"] = 1
    cache["b"] = 2
    cache["a"] = 3
    cache["c"] = 4
    assert list(cache) == ["a", "c"]
    assert cache.get("b") is None and cache["a"] == 3


def test_a_bound_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BoundedDict(0)


@pytest.mark.parametrize("cache", [translate._CACHE, server._FILINGS, equity_history._cache,
                                   research.data._SERIES_CACHE, exposures._BITGET_CACHE])
def test_every_cache_keyed_by_what_a_visitor_types_is_bounded(cache: object) -> None:
    assert isinstance(cache, BoundedDict) and cache.maxsize <= 512

