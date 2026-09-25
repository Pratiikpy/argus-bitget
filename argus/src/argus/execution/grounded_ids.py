"""Show a model short ids for orders and positions, and accept back only the ids it was shown.

**The failure this prevents.** When a model is shown a list of open orders or positions and asked
which to act on, the natural implementation passes the real identifiers — a 30-character
``client_order_id``, a symbol — and looks up whatever the model writes back. Two things go wrong.
The model copies a long identifier with a changed character and the lookup misses, or, worse, it
names something that *exists* but that it was never shown: an order filtered out of its view, one
from an earlier cycle, one belonging to another symbol. A lookup against the whole book accepts that
id, and the action lands on an order the model had no information about.

The fix is to close the action space to what was shown. The model sees ``[1] BUY 2 NVDAUSDT ...``,
``[2] ...``; it answers with ``1`` or ``[2]``; and anything else — an id not in this list, a real
``client_order_id`` even of a listed order, a malformed token — is refused before any action is
taken, naming every bad token at once.

What was taken, from which file, under which licence
----------------------------------------------------

WebArena (``web-arena-x/webarena``, Apache-2.0; local clone
``mypr/06_agent_evaluation_and_benchmarks/webarena``; licence and "Used in" record at
``argus/licenses/webarena-APACHE-2.0.txt``):

- **Compact state keyed by short ids, with a side table back to the real objects.**
  ``browser_env/processors.py:474-558`` (``parse_accessibility_tree``) renders every retained node
  as ``[id] role 'name' properties`` and records ``obs_nodes_info[id]`` mapping the short id back to
  the real DOM node; actions (``click [1234]``) name only those ids, and
  ``browser_env/actions.py:1124-1130`` resolves them through that table. Adapted from
  ``processors.py:474-558`` in :func:`show`, changed: the ids are dense integers from 1 in display
  order rather than accessibility-tree node ids, and the side table maps to our identifiers.
- **What is deliberately not copied: the unguarded lookup.** WebArena resolves an id with
  ``self.obs_nodes_info[element_id]`` (``processors.py:641-642``), so an id the model was not shown
  surfaces as a bare ``KeyError`` deep inside action execution. Here resolution is a separate,
  typed step that runs before anything acts and raises :class:`UnshownId` naming the bad tokens.

mem0 (``mem0ai/mem0``, Apache-2.0, Copyright 2023 Taranjeet Singh; licence and "Used in" record at
``argus/licenses/mem0-APACHE-2.0.txt``):

- **Small sequential integers instead of real ids, as an anti-hallucination device.**
  ``mem0/memory/main.py:935-940`` ("Map UUIDs to integers (anti-hallucination)") shows the
  extraction model ``str(idx)`` for each existing memory and keeps ``uuid_mapping[str(idx)]`` for
  the way back. **A finding from reading that source:** in this snapshot ``uuid_mapping`` is
  assigned at ``main.py:939`` and ``:2622`` and never read anywhere in the file, so the mapping back
  — and with it any refusal of an integer the model invented — does not happen. The half that makes
  the device a guard is the half built here: :meth:`ShownList.resolve`.

**A limit, stated.** Short ids are per rendering. A model that answers against an older list whose
``[2]`` meant something else cannot be caught by the id alone; the caller must resolve against the
same :class:`ShownList` it rendered into the prompt, which :attr:`ShownList.digest` makes checkable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, TypeVar

from argus.execution.orders import Order

T = TypeVar("T")

_TOKEN = re.compile(r"^\[?\s*(\d{1,4})\s*\]?$")
"""``3`` or ``[3]``, optionally padded. Nothing else is an id: not ``#3``, not ``order 3``."""


class UnshownId(ValueError):
    """The model named something it was not shown. Every bad token is listed; nothing acts."""

    def __init__(self, bad: Mapping[str, str], shown: Iterable[str]) -> None:
        self.bad = dict(bad)
        listed = ", ".join(f"[{s}]" for s in shown) or "nothing"
        detail = "; ".join(f"{token!s}: {why}" for token, why in self.bad.items())
        super().__init__(
            f"refused, the answer names ids it was not shown ({detail}); shown: {listed}"
        )


@dataclass(frozen=True, slots=True)
class ShownRow:
    short_id: str
    target: str
    """The real identifier this short id stands for. Never shown to the model."""

    line: str


@dataclass(frozen=True, slots=True)
class ShownList:
    """One rendering: what the model saw, and the only ids it may answer with."""

    kind: str
    rows: tuple[ShownRow, ...]

    @property
    def digest(self) -> str:
        """Identifies this exact rendering, so a caller can prove it resolved against what it
        showed rather than against a later list with different ``[n]`` meanings."""
        blob = "\n".join(f"{r.short_id}\t{r.target}\t{r.line}" for r in self.rows)
        return hashlib.sha256(f"{self.kind}\n{blob}".encode()).hexdigest()[:16]

    def render(self) -> str:
        head = f"{self.kind.upper()} (answer with the [n] id only)"
        if not self.rows:
            return f"{head}\n  (none)"
        return "\n".join([head, *(f"  [{r.short_id}] {r.line}" for r in self.rows)])

    def _why_not(self, token: object) -> tuple[str | None, str]:
        """``(short id, "")`` when ``token`` is a shown id, else ``(None, reason)``."""
        if isinstance(token, bool) or not isinstance(token, int | str):
            return None, f"{type(token).__name__} is not an id"
        match = _TOKEN.match(str(token).strip())
        if match is None:
            if any(str(token).strip() == r.target for r in self.rows):
                return None, "that is a real identifier; answer with its [n] id"
            return None, "not of the form n or [n]"
        short = str(int(match.group(1)))
        if not any(r.short_id == short for r in self.rows):
            return None, "no such id in the list that was shown"
        return short, ""

    def resolve(self, token: object) -> str:
        """The real identifier behind one shown id, or :class:`UnshownId`."""
        short, why = self._why_not(token)
        if short is None:
            raise UnshownId({str(token): why}, (r.short_id for r in self.rows))
        return next(r.target for r in self.rows if r.short_id == short)

    def resolve_all(self, tokens: Iterable[object]) -> tuple[str, ...]:
        """All-or-nothing: every token must be a shown id, and none may repeat.

        A partial action is refused whole. A model that named one thing it was not shown is working
        from a picture of the book that is wrong somewhere, and the parts it got right were chosen
        in light of the part it got wrong.
        """
        bad: dict[str, str] = {}
        chosen: list[str] = []
        for token in tokens:
            short, why = self._why_not(token)
            if short is None:
                bad[str(token)] = why
            elif short in chosen:
                bad[str(token)] = "named twice"
            else:
                chosen.append(short)
        if bad:
            raise UnshownId(bad, (r.short_id for r in self.rows))
        return tuple(next(r.target for r in self.rows if r.short_id == s) for s in chosen)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "digest": self.digest,
            "rows": [{"id": r.short_id, "line": r.line} for r in self.rows],
        }


def show(
    kind: str, items: Iterable[T], *, target: Callable[[T], str], describe: Callable[[T], str],
) -> ShownList:
    """Number ``items`` from 1 in the order given. Two items with one real id are refused."""
    rows: list[ShownRow] = []
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        real = target(item)
        if real in seen:
            raise ValueError(
                f"{real!r} appears twice in one {kind} list; the ids would be ambiguous"
            )
        seen.add(real)
        rows.append(ShownRow(short_id=str(index), target=real, line=describe(item)))
    return ShownList(kind=kind, rows=tuple(rows))


def show_orders(orders: Iterable[Order]) -> ShownList:
    """Open or in-flight orders as the model should see them: side, size, fill, state. No ids."""
    return show(
        "orders", orders,
        target=lambda o: o.client_order_id,
        describe=lambda o: (
            f"{o.side.upper()} {o.quantity} {o.symbol}, filled {o.filled_quantity}, "
            f"{o.state.value}"
        ),
    )


def show_positions(positions: Mapping[str, Decimal]) -> ShownList:
    """Net positions by symbol, flat ones omitted. The real id of a position is its symbol."""
    held = [(symbol, qty) for symbol, qty in sorted(positions.items()) if qty != 0]
    return show(
        "positions", held,
        target=lambda row: row[0],
        describe=lambda row: f"{'LONG' if row[1] > 0 else 'SHORT'} {abs(row[1])} {row[0]}",
    )


__all__ = [
    "ShownList",
    "ShownRow",
    "UnshownId",
    "show",
    "show_orders",
    "show_positions",
]
