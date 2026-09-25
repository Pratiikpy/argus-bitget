"""Identifier remapping — a model can only hand back an id it was shown.

**The failure this closes.** ARGUS shows models long identifiers and asks for some of them back:
analysts return the ``source_ids`` their view rests on (`agents/analysts.py`, ``SCHEMA_NOTE``), and
those ids are what `Panel.distinct_sources` counts and what the independence discount divides by.
The ids are long and numeric — ``twitter-2103455950178832631``, ``edgar-0001045810-26-000078`` — and
nothing checks that an echoed id was one the model saw. A dropped digit is a new "distinct source";
an invented id raises the panel's independence without adding a single piece of evidence. The
record cannot say how often that happened, because the analysts' raw ``source_ids`` are not
persisted (`eval/decision_primitives.py` reports this as unmeasurable rather than as zero). What it
can say is the exposure: the recorded thesis-quality frames show 133 evidence ids (95 distinct),
averaging 24 characters, 37 of the distinct ones carrying a run of ten or more digits; as handles
the same 3,234 characters would be 376.

**Adapted from mem0** (``mem0ai/mem0``, Apache-2.0; licence text and "Used in" record at
``argus/licenses/mem0-APACHE-2.0.txt``). Its extraction step shows the model existing memories
under sequential string handles instead of their UUIDs, ``uuid_mapping[str(idx)] = mem.id`` with
``{"id": str(idx), ...}`` in the prompt (``mem0/memory/main.py:935-940``; the async twin at
``:2620-2622``), so a small model is never asked to reproduce a 36-character UUID. Changed:

* **The round trip is completed.** In the checkout this was read from, ``uuid_mapping`` is built and
  then never read again — a grep of ``main.py`` finds it only where it is filled — while the prompt
  still says ``linked_memory_ids`` "uses the UUIDs from this list" (``mem0/configs/prompts.py:513``)
  of a list that no longer shows any. Here :meth:`IdMap.resolve_all` is the only way back, and it is
  tested end to end.
* **An unshown handle is rejected by name, never repaired.** mem0 has no rejection path at all.
  Guessing the nearest shown id would turn a hallucination into a citation.
* **Handles carry a prefix** (``E0``, ``E1`` …). mem0 can use bare integers because they sit in a
  JSON field; ARGUS handles also appear in prose, where a bare "3" is indistinguishable from a
  figure — and `agents/grounding.py` would count it as one. A digit preceded by a letter is not
  extracted as a figure there (its lookbehind refuses a word character), so a handle can never be
  mistaken for a claim about the market.
* **An identity mode** for ids that are already short (ledger seqs): no renaming, the same refusal
  of anything not shown. `agents/meta_pm.py`'s forced reflection uses it to check that the prior
  decision a model says it is evaluating is one its memory block actually listed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

HANDLE_PREFIX = "E"
"""Default handle prefix. Chosen so a handle reads as a reference, never as a number."""


@dataclass(frozen=True, slots=True)
class Resolved:
    """What came back: the real ids a model's answer names, and what it named that was not shown."""

    ids: tuple[str, ...]
    """Real ids, first-mention order, each once."""

    rejected: tuple[str, ...]
    """Handles or ids the model returned that were never shown to it, verbatim."""

    duplicates: int = 0
    """Repeats of an id already resolved. Counted, because a repeated source is not a second one."""

    @property
    def clean(self) -> bool:
        return not self.rejected

    def as_dict(self) -> dict[str, Any]:
        return {"ids": list(self.ids), "rejected": list(self.rejected),
                "duplicates": self.duplicates}


class IdMap:
    """Real ids ↔ the handles a model is shown, built per prompt from exactly what it shows."""

    def __init__(self, ids: Iterable[str], *, prefix: str = HANDLE_PREFIX,
                 identity: bool = False) -> None:
        if not identity and (not prefix or not prefix[0].isalpha()):
            raise ValueError("a handle prefix must start with a letter, or handles read as figures")
        self.prefix = prefix
        self.identity = identity
        self._to_handle: dict[str, str] = {}
        self._to_real: dict[str, str] = {}
        for real in ids:
            real = str(real)
            if real in self._to_handle:
                continue  # one handle per id, however often the caller lists it
            handle = real if identity else f"{prefix}{len(self._to_handle)}"
            self._to_handle[real] = handle
            self._to_real[handle] = real

    @classmethod
    def identities(cls, ids: Iterable[str]) -> IdMap:
        """Ids shown as themselves; only the refusal of unshown ones applies."""
        return cls(ids, identity=True)

    def __len__(self) -> int:
        return len(self._to_handle)

    def __contains__(self, handle: object) -> bool:
        return isinstance(handle, str) and handle.strip() in self._to_real

    @property
    def shown(self) -> tuple[str, ...]:
        """Every handle, in the order the ids were given."""
        return tuple(self._to_handle.values())

    def handle(self, real: str) -> str:
        """The handle for an id that was registered. A lookup of anything else is a caller bug."""
        try:
            return self._to_handle[real]
        except KeyError:
            raise KeyError(f"{real!r} was not registered in this IdMap") from None

    def resolve(self, handle: str) -> str | None:
        """The real id behind a handle, or ``None`` if it was never shown.

        Whitespace is forgiven; case and digits are not — "e3" or "E03" was not shown, and treating
        it as "E3" would be a guess.
        """
        return self._to_real.get(str(handle).strip())

    def resolve_all(self, returned: Sequence[Any] | Any) -> Resolved:
        """Every handle a model returned, split into real ids and what was never shown.

        ``returned`` may be a list (a ``source_ids`` field) or a single value; anything that is not
        a list is treated as one element, and an empty or blank element is ignored rather than
        rejected — "no id" is an answer, a wrong id is not.
        """
        items = list(returned) if isinstance(returned, list | tuple) else [returned]
        ids: list[str] = []
        rejected: list[str] = []
        duplicates = 0
        for item in items:
            text = str(item).strip() if item is not None else ""
            if not text:
                continue
            real = self.resolve(text)
            if real is None:
                rejected.append(text)
            elif real in ids:
                duplicates += 1
            else:
                ids.append(real)
        return Resolved(ids=tuple(ids), rejected=tuple(rejected), duplicates=duplicates)

    def expand(self, text: str) -> str:
        """Rewrite every shown handle in free text as its real id, for the record a person reads.

        Only whole handles are replaced (``E1`` inside ``E12`` is not), and an unshown handle-shaped
        token is left exactly as written, so prose never acquires a citation the model did not make.
        """
        if self.identity or not self._to_real:
            return text
        pattern = re.compile(rf"(?<![\w-]){re.escape(self.prefix)}\d+(?![\w-])")
        return pattern.sub(lambda m: self._to_real.get(m.group(0), m.group(0)), text)

    def as_dict(self) -> dict[str, str]:
        """Handle → real id, for persisting beside a prompt so a reader can audit the mapping."""
        return dict(self._to_real)


def restrict_to_shown(returned: Sequence[Any] | Any, shown: Iterable[str]) -> Resolved:
    """The same refusal without renaming: keep the returned ids that were shown, name the rest.

    The one-line integration for a caller whose prompt already shows real ids — the analysts' parse
    step can call this with the evidence ids it rendered, and pass ``Resolved.ids`` on as
    ``source_ids``.
    """
    return IdMap.identities(shown).resolve_all(returned)


__all__ = ["HANDLE_PREFIX", "IdMap", "Resolved", "restrict_to_shown"]
