"""A dictionary that never grows past a fixed number of entries.

The console is a long-running server, and several of its caches were keyed by what a visitor typed
(a ticker, a question's text): an unbounded dict keyed by user input is a memory leak anyone can
drive (audit finding 166). This is the one bounded mapping they share: at :attr:`maxsize` the
least recently written entry is dropped. Reads do not reorder, so a lookup stays a plain dict
lookup; each cache already expires its own entries by age, and the bound is only the ceiling.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


class BoundedDict(OrderedDict[K, V]):
    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        super().__init__()
        self.maxsize = maxsize
        self._lock = threading.Lock()

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            super().__setitem__(key, value)
            self.move_to_end(key)
            while len(self) > self.maxsize:
                self.popitem(last=False)


__all__ = ["BoundedDict"]
