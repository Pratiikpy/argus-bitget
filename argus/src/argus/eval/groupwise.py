"""Groupwise — no headline travels without its breakdown by symbol, date and regime.

This project has recorded the same failure twice, and both times the aggregate was right and the
reading of it was wrong. A single-name trend looked strong across the universe because one
instrument produced the whole return from four trades. Cross-sectional momentum looked excellent on
the full sample and went negative in both halves once the sample was split. Neither was a bug in a
formula. Each was an average that hid which part of the population produced it, and each was caught
only because someone remembered to look. This module makes the looking mechanical: it takes the
per-item results a comparison already has, with the keys that group them, and returns the
breakdown alongside the headline, with a named flag whenever the breakdown contradicts it.

**What was taken, from where.** Mind2Web's evaluation loop (OSU-NLP-Group/Mind2Web,
``src/action_prediction/metric.py:236-259``, **MIT**, read at source) never reports a step accuracy
without three companions computed in the same pass: the **macro average** over tasks, not steps
(``marco_step_acc``, ``:256-258`` — so a task with forty steps does not outvote one with four), an
**error-count histogram per task** bucketed ``0 / 1 / 2 / 3 / >3`` wrong steps (``error_ratio``,
``:246-254``), and **accuracy per website** as ``(mean, count)`` (``acc_per_website``,
``:247-255``) — so a single easy or hard site cannot hide inside the aggregate. Adapted here, and
changed as follows:

* The group is any key a comparison's rows carry (symbol, night, origin, regime, claim kind), not a
  website, and one call breaks the same rows down by several keys at once.
* The histogram counts, per group, the items that *oppose* the headline's direction rather than
  wrong steps, because a comparison's items are signed contributions (ARGUS's error minus the
  rival's, a trade's net return), not right/wrong flags. The buckets are Mind2Web's.
* Mind2Web reports its breakdown and leaves the reading to a person. This module reads it, with
  four named flags, because the two recorded failures were both failures to read:

  ``carried_by_one_group``  Removing the group that contributes most in the headline's direction
                            leaves a total that no longer points that way — one instrument produced
                            the whole return. (Leave-one-group-out, applied to the one group that
                            could carry the result.)
  ``single_group``          A key the comparison declares as a dimension it generalises over has
                            exactly one value: the headline *is* one symbol, or one day.
  ``macro_disagrees``       The macro average (every group weighted equally, Mind2Web's
                            ``marco_*``) points the other way from the item average: the headline
                            depends on which groups happen to have the most items.
  ``flips_across_halves``   Split chronologically, the two halves disagree in sign, or agree with
                            each other and not with the full sample — the momentum result.

**Rejected.** A concentration flag on the largest group's *share* alone (say, one symbol supplying
more than half the total). A share threshold punishes a result for the volatility of whichever
instrument supplied it: when every symbol points the same way, the one with the largest moves still
supplies the largest share, and nothing about the result is carried. Leave-one-group-out asks the
question the recorded failure needed — would the headline survive without that group — and the
share is still reported, as a number, for a reader to judge.

**Rejected, second.** A random split for the halves check. A random split grades a result on items
drawn from before and after the ones it was fitted on; the recorded momentum failure was a
chronological one, and only a chronological split reproduces it. Items with the same order value (a
date shared by every symbol) always land in the same half, so the split never cuts through one
day's cross-section.

Dated finding, 2026-09-25: run over every comparison artefact the capability register cites (see
:mod:`argus.eval.groupwise_audit` and ``data/groupwise_audit.json``); what it flagged is recorded
there, not restated here, because a count written into this docstring would outlive the run that
produced it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SIGN_TOLERANCE = 1e-9
"""A total within this fraction of the summed magnitudes is treated as having no direction.

Relative rather than absolute because the values arrive in basis points, percentage points,
probabilities and counts; floating-point noise on a sum of opposite contributions must not be read
as a direction.
"""

MIN_HALF_ITEMS = 3
"""Fewest items either chronological half may hold before the split is run at all.

Below this a half's mean is one or two observations, and a sign flip between two single numbers is
not evidence of anything. The check then reports that it did not run and why, rather than a flip.
"""

HISTOGRAM_BUCKETS = ("0", "1", "2", "3", ">3")
"""Mind2Web's own ``error_ratio`` buckets (``metric.py:250-254``)."""

FLAGS = ("carried_by_one_group", "single_group", "macro_disagrees", "flips_across_halves")
"""The four readings that contradict a headline. Every one fails a groupwise check."""


class GroupwiseError(ValueError):
    """Rows that cannot be broken down: empty, or missing a key the comparison declared."""


def _sign(value: float, scale: float) -> int:
    if not math.isfinite(value):
        raise GroupwiseError(f"a non-finite value ({value}) cannot be broken down")
    if abs(value) <= SIGN_TOLERANCE * max(scale, 1e-300):
        return 0
    return 1 if value > 0 else -1


@dataclass(frozen=True, slots=True)
class Item:
    """One per-item result: its signed contribution and the groups it belongs to.

    ``value`` is oriented by the caller so that the headline is the (weighted) mean of the values
    — ARGUS's error minus the rival's for a lower-is-better metric, a trade's net return, ``+1`` for
    a case where ARGUS's claimed property holds and ``-1`` where it fails. The flags read only
    signs relative to the headline's own sign, so the orientation changes the words in the report
    and never the verdict.

    ``weight`` lets a row stand for several underlying items when an artefact kept only a group's
    mean and its count (per-stock means over 113 nights). The histogram then counts rows, not the
    items inside them; that is stated wherever such rows are used.

    ``order`` is a sortable string (an ISO date, or a zero-padded sequence number) for the
    chronological halves check; ``None`` when the rows have no order.
    """

    value: float
    groups: Mapping[str, str]
    order: str | None = None
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class GroupRow:
    """One group's share of the headline — Mind2Web's ``acc_per_website`` entry, extended."""

    group: str
    rows: int
    weight: float
    total: float
    mean: float
    share_of_total: float | None
    opposing: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group, "rows": self.rows, "weight": round(self.weight, 6),
            "total": _round(self.total), "mean": _round(self.mean),
            "share_of_total": None if self.share_of_total is None else round(
                self.share_of_total, 4),
            "opposing": self.opposing,
        }


@dataclass(frozen=True, slots=True)
class GroupTable:
    """The rows broken down by one key, and what the breakdown says about the headline."""

    key: str
    rows: tuple[GroupRow, ...]
    micro_mean: float
    macro_mean: float
    direction: int
    largest: str | None
    largest_share: float | None
    total_without_largest: float | None
    loss_histogram: Mapping[str, float]
    carried: bool
    single_group: bool
    macro_disagrees: bool
    reading: str

    @property
    def groups(self) -> int:
        return len(self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "groups": self.groups,
            "micro_mean": _round(self.micro_mean), "macro_mean": _round(self.macro_mean),
            "direction": self.direction, "largest_contributor": self.largest,
            "largest_share_of_total": None if self.largest_share is None else round(
                self.largest_share, 4),
            "total_without_largest": (None if self.total_without_largest is None
                                      else _round(self.total_without_largest)),
            "loss_histogram": {k: round(v, 4) for k, v in self.loss_histogram.items()},
            "carried_by_one_group": self.carried, "single_group": self.single_group,
            "macro_disagrees": self.macro_disagrees, "reading": self.reading,
            "per_group": [r.as_dict() for r in self.rows],
        }


@dataclass(frozen=True, slots=True)
class SplitHalf:
    """The headline on each chronological half, and whether the halves agree with it."""

    label: str
    cut: str
    first_items: int
    second_items: int
    first_mean: float
    second_mean: float
    full_mean: float
    flips: bool
    contradicts_full: bool
    reading: str

    @property
    def flagged(self) -> bool:
        return self.flips or self.contradicts_full

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "cut": self.cut, "first_items": self.first_items,
            "second_items": self.second_items, "first_mean": _round(self.first_mean),
            "second_mean": _round(self.second_mean), "full_mean": _round(self.full_mean),
            "flips_across_halves": self.flips, "contradicts_full_sample": self.contradicts_full,
            "reading": self.reading,
        }


@dataclass(frozen=True)
class GroupwiseReport:
    """A headline with its breakdown by every declared key, its halves, and the flags."""

    name: str
    headline: str
    orientation: str
    items: int
    micro_mean: float
    total: float
    tables: tuple[GroupTable, ...]
    split: SplitHalf | None
    split_note: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def flags(self) -> tuple[str, ...]:
        """Every contradiction, named with the key and group it came from."""
        out: list[str] = []
        for table in self.tables:
            if table.carried:
                out.append(f"carried_by_one_group:{table.key}={table.largest}")
            if table.single_group:
                only = table.rows[0].group if table.rows else "?"
                out.append(f"single_group:{table.key}={only}")
            if table.macro_disagrees:
                out.append(f"macro_disagrees:{table.key}")
        if self.split is not None and self.split.flagged:
            out.append(f"flips_across_halves:{self.split.label}")
        return tuple(out)

    @property
    def flagged(self) -> bool:
        return bool(self.flags)

    @property
    def broken_down(self) -> bool:
        """Did at least one key actually split the rows into two or more groups?"""
        return any(t.groups >= 2 for t in self.tables)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "headline": self.headline, "orientation": self.orientation,
            "items": self.items, "micro_mean": _round(self.micro_mean),
            "total": _round(self.total), "broken_down": self.broken_down,
            "flags": list(self.flags), "flagged": self.flagged,
            "tables": [t.as_dict() for t in self.tables],
            "split_half": self.split.as_dict() if self.split is not None else None,
            "split_half_note": self.split_note, "notes": list(self.notes),
        }


def _round(value: float) -> float:
    return round(value, 6) if math.isfinite(value) else value


def _scale(items: Iterable[Item]) -> float:
    return sum(abs(i.value * i.weight) for i in items)


def breakdown(items: Sequence[Item], key: str) -> GroupTable:
    """The rows grouped by ``key``: per-group table, macro average, histogram, and flags.

    The per-group ``(mean, weight)`` pair is Mind2Web's ``acc_per_website``; the macro average is
    its ``marco_*`` (the mean of the group means); the histogram is its ``error_ratio`` with
    "wrong step" read as "an item opposing the headline". ``carried`` is the leave-one-group-out
    reading described in the module docstring, applied to the group that contributes most in the
    headline's direction — the only group whose removal could reverse it.
    """
    if not items:
        raise GroupwiseError("no items: an empty comparison has no headline to break down")
    missing = [i for i in items if key not in i.groups]
    if missing:
        raise GroupwiseError(
            f"{len(missing)} of {len(items)} item(s) carry no {key!r}; a breakdown that silently "
            f"drops rows is the aggregate this module exists to refuse")
    scale = _scale(items)
    weight = sum(i.weight for i in items)
    if weight <= 0:
        raise GroupwiseError("the items carry no weight")
    total = sum(i.value * i.weight for i in items)
    micro = total / weight
    direction = _sign(total, scale)

    by_group: dict[str, list[Item]] = {}
    for item in items:
        by_group.setdefault(str(item.groups[key]), []).append(item)
    rows: list[GroupRow] = []
    for name, members in by_group.items():
        g_weight = sum(m.weight for m in members)
        g_total = sum(m.value * m.weight for m in members)
        opposing = sum(1 for m in members
                       if direction != 0 and _sign(m.value, scale) == -direction)
        rows.append(GroupRow(
            group=name, rows=len(members), weight=g_weight, total=g_total,
            mean=g_total / g_weight if g_weight else math.nan,
            share_of_total=(g_total / total) if direction != 0 else None,
            opposing=opposing))
    rows.sort(key=lambda r: (-(r.total * direction) if direction else -abs(r.total), r.group))
    macro = sum(r.mean for r in rows) / len(rows)

    histogram = dict.fromkeys(HISTOGRAM_BUCKETS, 0.0)
    for r in rows:
        histogram[str(r.opposing) if r.opposing <= 3 else ">3"] += 1.0 / len(rows)

    single = len(rows) == 1
    largest: str | None = None
    share: float | None = None
    rest: float | None = None
    carried = False
    if direction != 0 and len(rows) >= 2:
        top = rows[0]
        largest, share = top.group, top.share_of_total
        rest = total - top.total
        carried = _sign(rest, scale) != direction
    macro_sign = _sign(macro, max(abs(r.mean) for r in rows) * len(rows))
    macro_disagrees = direction != 0 and len(rows) >= 2 and macro_sign != direction

    if direction == 0:
        reading = "the headline has no direction, so no group can carry it"
    elif single:
        reading = f"the whole headline is one {key} ({rows[0].group})"
    elif carried:
        reading = (f"{largest} carries the headline: without it the other {len(rows) - 1} "
                   f"{key} group(s) total {_round(rest or 0.0)}, which does not point the same way")
    else:
        reading = (f"no single {key} carries the headline: without the largest contributor "
                   f"({largest}, {round((share or 0.0) * 100, 1)}% of the total) the rest still "
                   f"point the same way")
    if macro_disagrees:
        reading += (f"; the macro average over {key} groups ({_round(macro)}) points the other "
                    f"way from the item average ({_round(micro)})")
    return GroupTable(
        key=key, rows=tuple(rows), micro_mean=micro, macro_mean=macro, direction=direction,
        largest=largest, largest_share=share, total_without_largest=rest,
        loss_histogram=histogram, carried=carried, single_group=single,
        macro_disagrees=macro_disagrees, reading=reading)


def _half_reading(label: str, first: float, second: float, full: float, flips: bool,
                  contradicts: bool) -> str:
    if flips:
        return (f"{label}: the halves disagree — first {_round(first)}, second {_round(second)} "
                f"(full {_round(full)})")
    if contradicts:
        return (f"{label}: both halves point the other way from the full sample — first "
                f"{_round(first)}, second {_round(second)}, full {_round(full)}")
    return (f"{label}: both halves point the same way as the full sample — first {_round(first)}, "
            f"second {_round(second)}, full {_round(full)}")


def halves_from(first_mean: float, second_mean: float, *, full_mean: float | None = None,
                first_items: int, second_items: int, label: str,
                cut: str = "") -> SplitHalf:
    """A split an artefact already computed (in-sample against out-of-sample, two windows).

    Many artefacts recorded their own chronological split without keeping the rows. Their two
    halves are checked here with the same rule as :func:`split_half`, including the case the
    recorded momentum failure needed — both halves the same sign, the full sample the other —
    which an average of the halves can never show but a Sharpe ratio, a hit rate against a moving
    base or an IC can.
    """
    full = full_mean if full_mean is not None else (
        (first_mean * first_items + second_mean * second_items)
        / max(first_items + second_items, 1))
    scale = max(abs(first_mean), abs(second_mean), abs(full))
    s1, s2, sf = _sign(first_mean, scale), _sign(second_mean, scale), _sign(full, scale)
    flips = s1 != s2
    contradicts = not flips and s1 != 0 and sf != 0 and s1 != sf
    return SplitHalf(
        label=label, cut=cut, first_items=first_items, second_items=second_items,
        first_mean=first_mean, second_mean=second_mean, full_mean=full, flips=flips,
        contradicts_full=contradicts,
        reading=_half_reading(label, first_mean, second_mean, full, flips, contradicts))


def split_half(items: Sequence[Item], *, label: str = "chronological halves",
               ) -> tuple[SplitHalf | None, str]:
    """Cut the rows at the order value that balances the two halves; compare their means.

    Returns ``(None, why)`` when the check cannot run: rows without an order, fewer than two
    distinct order values, or a half below :data:`MIN_HALF_ITEMS`. The cut is chosen among order
    values, never inside one, so a date's whole cross-section stays on one side.
    """
    ordered = [i for i in items if i.order is not None]
    if not ordered:
        return None, "the rows carry no order (no date or sequence), so no chronological split"
    if len(ordered) != len(items):
        return None, (f"{len(items) - len(ordered)} of {len(items)} row(s) carry no order; a "
                      f"split over the rest would silently drop them")
    distinct = sorted({str(i.order) for i in ordered})
    if len(distinct) < 2:
        return None, f"every row has the same order value ({distinct[0]}): nothing to split"
    counts: dict[str, int] = {}
    for i in ordered:
        counts[str(i.order)] = counts.get(str(i.order), 0) + 1
    best_cut, best_gap, running = 1, math.inf, 0
    for idx, value in enumerate(distinct[:-1], start=1):
        running += counts[value]
        gap = abs(len(ordered) - 2 * running)
        if gap < best_gap:
            best_cut, best_gap = idx, gap
    cut = distinct[best_cut]
    first = [i for i in ordered if str(i.order) < cut]
    second = [i for i in ordered if str(i.order) >= cut]
    if len(first) < MIN_HALF_ITEMS or len(second) < MIN_HALF_ITEMS:
        return None, (f"a half would hold fewer than {MIN_HALF_ITEMS} rows "
                      f"({len(first)} and {len(second)}): too few for a sign to mean anything")

    def mean(rows: Sequence[Item]) -> float:
        return sum(r.value * r.weight for r in rows) / sum(r.weight for r in rows)

    split = halves_from(mean(first), mean(second), full_mean=mean(ordered),
                        first_items=len(first), second_items=len(second), label=label, cut=cut)
    return split, ""


def audit(name: str, items: Sequence[Item], keys: Sequence[str], *, headline: str,
          orientation: str, halves: SplitHalf | None = None,
          notes: Sequence[str] = ()) -> GroupwiseReport:
    """The headline and its breakdown by every key, with the chronological halves.

    ``keys`` are the dimensions the comparison generalises over — symbol, date, regime, the kind of
    claim — and every item must carry every one of them. A key with one value is flagged
    ``single_group``: declaring it means the headline claims to generalise over it, and one value
    cannot show that. ``halves`` supplies a split the artefact computed itself; otherwise the rows'
    own ``order`` is split.
    """
    if not items:
        raise GroupwiseError(f"{name}: no items to break down")
    if not keys:
        raise GroupwiseError(f"{name}: no grouping key declared; a breakdown needs at least one")
    tables = tuple(breakdown(items, key) for key in keys)
    split: SplitHalf | None
    if halves is not None:
        split, note = halves, "halves as the artefact recorded them"
    else:
        split, note = split_half(items)
    weight = sum(i.weight for i in items)
    total = sum(i.value * i.weight for i in items)
    return GroupwiseReport(
        name=name, headline=headline, orientation=orientation, items=len(items),
        micro_mean=total / weight, total=total, tables=tables, split=split, split_note=note,
        notes=tuple(notes))


__all__ = [
    "FLAGS",
    "HISTOGRAM_BUCKETS",
    "MIN_HALF_ITEMS",
    "GroupRow",
    "GroupTable",
    "GroupwiseError",
    "GroupwiseReport",
    "Item",
    "SplitHalf",
    "audit",
    "breakdown",
    "halves_from",
    "split_half",
]
