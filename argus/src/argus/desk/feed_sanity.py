"""The feed-sanity gate: a deterministic check on every figure before the Meta-PM reads it.

**Why this exists — measured, 2026-09-25.** `eval/feedbugged.py` planted one wrong value per frame
(openai/evals' ``bugged_tools`` families) and one wrong step per upstream argument
(``error_recovery`` IR), then asked the Meta-PM to decide. The desk's detection F1 was **0.0**
(``data/feedbugged.json``: tp 0, fn 6). It read a 24-hour change of ``9.99346`` — a **+999%**
day — as an ordinary move; it took a ``NaN`` price without comment; and an injected quant line
claiming a **6895bps** expected edge turned the NVDA ``no_trade`` into **TRADE BUY 50**. A model
cannot be relied on to notice a wrong number it has been handed as a fact, and the prompt is the
wrong place to ask it to: a decision-maker told to audit its inputs is a different decision-maker
(feedbugged.py's own point 1).
So the check is code, it runs before the model, and a figure that fails never reaches the prompt.

**What is checked, and the evidence for each threshold.** Every rule is a contradiction or an
impossibility, never a judgement that a value is merely unusual:

* ``non_numeric`` — a figure slot that does not parse as a finite number (``NaN``, ``inf``, a word).
  The slots are located by the exact templates the live feeds render: the market line
  (`paper/runner.py:675-679`), VIX (`market/volatility.py:158-172`), Crypto Fear & Greed
  (`market/macro.py:285`), and SUE (`market/fundamentals.py:253-265`).
* ``price_mismatch`` — the market line's ``last`` against the token price the desk was handed. In
  production both are ``ticker.last`` from the same fetch (`paper/runner.py:677,782`), so any gap is
  a feed fault; the tolerance, :data:`PRICE_TOLERANCE` (0.5%), is half the VIX-implied one-day
  equity move (~95bps on 2026-09-25) and far below a one-digit error.
* ``implausible_change`` — ``|24h change| > 0.5``. The feed states the change as a *fraction*
  (``0.01379`` is +1.379%). A US large-cap tracker moving more than 50% in a day is not a market
  event the desk should act on unexamined; it is the shape of a unit or offset error.
* ``negative_spread`` — a quoted spread below zero is a crossed book or a sign error, not a cost.
* ``vix_inconsistent`` — the VIX line states both the level and the one-day move it implies, which
  `market/volatility.py:141` computes as ``level / sqrt(252)``. When the two disagree by more than
  :data:`VIX_TOLERANCE_BPS` (rendering rounds the move to whole bps, so honest lines differ by at
  most ~0.53bps), one of them is wrong and the line cannot say which.
* ``sue_direction`` — `fundamentals.py:253` derives the word *above*/*below* from the sign of the
  SUE. A positive value described as *below* (or the reverse) contradicts itself.
* ``ratio_inconsistent`` — two lines state a figure *and* the parts it is computed from: the FINRA
  short share is ``short / total`` (`market/microstructure.py:143-148`, rendered to 0.1%), and the
  Treasury curve's ``10y-2y`` spread is ``(10y - 2y) x 100`` bps (`market/macro.py:82-131`, rendered
  to whole bps). A stated figure that disagrees with its own parts beyond rendering precision is a
  corruption of one of them, and the line is withheld whole. The same templates, and every
  ``quarter on quarter N%`` filing line (`market/fundamentals.py`), are also checked for
  ``non_numeric`` slots.
* ``absurd_edge`` — an analyst or quant line claiming ``|expected edge| >= 200bps``
  (:data:`ABSURD_EDGE_BPS`). The desk's hurdle is ~14bps and the VIX-implied one-sigma day is
  ~95bps; an *expected* edge over twice a one-sigma day is a per-trade Sharpe above 2, which no
  documented desk sustains, and it is exactly the magnitude a 100x unit error produces.
* ``unit_error`` — a bps figure in an analyst line that is 100x the feed's own 24h move in bps (to
  within 1%): the fraction read as a percent, the single most common real mistake on a fractional
  change (`eval/feedbugged.py` ``WRONG_STEP_FACTOR``). This is what catches the injected META step,
  whose claimed 45bps edge is below the absurd threshold but whose ``90bps the move`` is 100x the
  feed's ``-0.9bps``.

**What happens to a failing figure.** It is withheld, in the shape `agents/quarantine.py` settled
on: the evidence item keeps its id, source and timestamp, and only the offending figure's text is
replaced by ``[withheld by feed-sanity: <rule>]`` — so the other, sound figures on the same line
(the spread beside a ``NaN`` price) still reach the model, and no count downstream changes. Where a
rule is a contradiction between two figures and the line cannot say which is wrong (VIX, SUE), the
whole claim is withheld. A flagged analyst view is removed from the panel before consensus and
before the frame is built. Every run reports what it checked, even when nothing fired — a screen
that reports only its hits is indistinguishable from one that never ran (quarantine's rule).

**What this does not do.** A plausible wrong value that contradicts nothing — a VIX of 15.11 read
as 15.61 *with* a matching implied move, a 24h change of +0.4% flipped to -0.4% — is not detectable
by any check that has only the frame. Cross-source verification (a second quote for the same
figure) is the only remedy, and no second source for those figures is wired into the desk; that
limit is stated rather than papered over with a heuristic that would fire on real data.

**Sources.** The corruption families and the scoring convention are openai/evals' (MIT,
``elsuite/bugged_tools/bugged_tools.py:1-146``, ``utils.py:8-61``), already ported in
`eval/feedbugged.py`; nothing is copied here — the gate is ARGUS's own, built against those cases.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from argus.truth.evidence import Evidence

if TYPE_CHECKING:
    from argus.agents.analysts import AnalystView

RULES: tuple[str, ...] = (
    "non_numeric", "price_mismatch", "implausible_change", "negative_spread",
    "vix_inconsistent", "sue_direction", "ratio_inconsistent", "absurd_edge", "unit_error",
)

PRICE_TOLERANCE = Decimal("0.005")
"""Relative gap between the market line's ``last`` and the token price: a fault, not noise."""

MAX_ABS_CHANGE_24H = Decimal("0.5")
"""A fractional 24h change beyond this (±50%) is withheld as implausible."""

VIX_TOLERANCE_BPS = 1.0
"""Stated vs recomputed VIX-implied one-day move. Rounding alone accounts for at most ~0.53bps."""

ABSURD_EDGE_BPS = Decimal("200")
"""An expected edge at or above this magnitude is withheld: over twice a one-sigma equity day."""

UNIT_ERROR_FACTOR = Decimal("100")
UNIT_ERROR_TOLERANCE = Decimal("0.01")
MIN_MOVE_BPS_FOR_UNIT_CHECK = Decimal("0.5")
"""Below half a basis point the 100x image of the move is itself tiny and would collide with honest
small figures, so the unit check is not applied."""

WITHHELD = "[withheld by feed-sanity: {rule}]"

_MARKET = re.compile(
    r"\blast (?P<last>\S+?), 24h change (?P<change>\S+?), quoted spread (?P<spread>\S+?)bps"
)
_VIX = re.compile(
    r"\bVIX (?P<level>\S+) \([^)]*\).*?one-day move of about (?P<move>\S+?)bps", re.DOTALL
)
_FNG = re.compile(r"Fear & Greed (?P<value>\S+?)/100")
_SUE = re.compile(
    r"Standardized Unexpected Earnings[^:]*: (?P<value>\S+) standard deviations "
    r"(?P<direction>above|below|in line with)"
)
_QOQ = re.compile(r"quarter on quarter (?P<value>\S+?)% \(")
_SHORT = re.compile(
    r"short volume (?P<short>[\d,]+) of (?P<total>[\d,]+) total on \S+ \S+ "
    r"(?P<share>\S+?)% of the session's prints were short"
)
_CURVE = re.compile(r"^US Treasury par curve [^:]*: (?P<body>.*?); 10y-2y (?P<spread>\S+?)bps")
_TENOR = re.compile(r"(?P<tenor>\d+[my]) (?P<value>\S+?)%")
SHORT_SHARE_TOLERANCE = Decimal("0.06")
"""Percentage points. The share is rendered to 0.1%, so an honest line is within 0.05."""
CURVE_TOLERANCE_BPS = Decimal("1.01")
"""Whole-bps spread from yields rendered to 0.01%: honest lines agree within 1bps."""
_PANEL = re.compile(r"^\[panel\] (?P<analyst>[\w.-]+): (?P<signal>[\w-]+) (?P<edge>\S+?)bps")
_BPS = re.compile(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)bps")


@dataclass(frozen=True, slots=True)
class Finding:
    """One figure that failed, where it is, and why."""

    item_id: str
    rule: str
    figure: str
    detail: str
    span: tuple[int, int] | None = None
    """The figure's position in the claim. ``None`` means the whole item is withheld."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id, "rule": self.rule, "figure": self.figure,
            "detail": self.detail, "whole_item": self.span is None,
        }


@dataclass(frozen=True)
class SanityReport:
    """What the gate checked and what it withheld. Rendered into the decision's notes."""

    checked: int
    findings: tuple[Finding, ...] = ()
    scope: str = "evidence"

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def withheld(self) -> tuple[str, ...]:
        return tuple(sorted({f.item_id for f in self.findings}))

    @property
    def rules_fired(self) -> tuple[str, ...]:
        return tuple(sorted({f.rule for f in self.findings}))

    def render(self) -> list[str]:
        if self.clean:
            return [
                f"[feed-sanity] {self.checked} {self.scope} item(s) checked for impossible or "
                f"self-contradicting figures; none withheld"
            ]
        lines = [
            f"[feed-sanity] {self.checked} {self.scope} item(s) checked; "
            f"{len(self.findings)} figure(s) in {len(self.withheld)} item(s) withheld before the "
            f"decision-maker read them ({', '.join(self.rules_fired)})"
        ]
        lines.extend(
            f"[feed-sanity] {f.item_id}: {f.rule} — {f.detail}" for f in self.findings
        )
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "checked": self.checked,
            "withheld": list(self.withheld),
            "rules_fired": list(self.rules_fired),
            "findings": [f.as_dict() for f in self.findings],
        }


def _number(text: str) -> Decimal | None:
    """A finite decimal, or ``None`` for anything that is not one (``NaN``, ``inf``, a word)."""
    try:
        value = Decimal(text.rstrip(",;"))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def market_move(claims: Sequence[str]) -> Decimal | None:
    """The feed's own fractional 24h change, if a market line carries a sound one."""
    for claim in claims:
        m = _MARKET.search(claim)
        if m is None:
            continue
        change = _number(m["change"])
        if change is not None and abs(change) <= MAX_ABS_CHANGE_24H:
            return change
    return None


def _check_market(
    item_id: str, claim: str, token_price: Decimal | None, out: list[Finding]
) -> None:
    m = _MARKET.search(claim)
    if m is None:
        return
    last, change, spread = _number(m["last"]), _number(m["change"]), _number(m["spread"])
    for name, value in (("last", last), ("change", change), ("spread", spread)):
        if value is None:
            out.append(Finding(
                item_id, "non_numeric", m[name], f"the {name} slot reads {m[name]!r}, not a number",
                (m.start(name), m.end(name)),
            ))
    if last is not None:
        if last <= 0:
            out.append(Finding(
                item_id, "price_mismatch", m["last"], f"a non-positive price {m['last']}",
                (m.start("last"), m.end("last")),
            ))
        elif token_price is not None and token_price > 0:
            gap = abs(last / token_price - 1)
            if gap > PRICE_TOLERANCE:
                out.append(Finding(
                    item_id, "price_mismatch", m["last"],
                    f"last {m['last']} disagrees with the token price {token_price} by "
                    f"{gap:.2%} (tolerance {PRICE_TOLERANCE:.1%})",
                    (m.start("last"), m.end("last")),
                ))
    if change is not None and abs(change) > MAX_ABS_CHANGE_24H:
        out.append(Finding(
            item_id, "implausible_change", m["change"],
            f"24h change {m['change']} is {change:+.2%}, beyond ±{MAX_ABS_CHANGE_24H:.0%}",
            (m.start("change"), m.end("change")),
        ))
    if spread is not None and spread < 0:
        out.append(Finding(
            item_id, "negative_spread", m["spread"], f"quoted spread {m['spread']}bps is negative",
            (m.start("spread"), m.end("spread")),
        ))


def _check_macro(item_id: str, claim: str, out: list[Finding]) -> None:
    vix = _VIX.search(claim)
    if vix is not None:
        level, move = _number(vix["level"]), _number(vix["move"])
        if level is None or level <= 0:
            out.append(Finding(
                item_id, "non_numeric", vix["level"], f"VIX level reads {vix['level']!r}",
                (vix.start("level"), vix.end("level")),
            ))
        elif move is None:
            out.append(Finding(
                item_id, "non_numeric", vix["move"], f"implied move reads {vix['move']!r}",
                (vix.start("move"), vix.end("move")),
            ))
        else:
            implied = float(level) / math.sqrt(252) * 100.0
            if abs(implied - float(move)) > VIX_TOLERANCE_BPS:
                out.append(Finding(
                    item_id, "vix_inconsistent", vix["level"],
                    f"VIX {vix['level']} implies {implied:.1f}bps a day but the line states "
                    f"{vix['move']}bps; one of the two is wrong",
                ))
    fng = _FNG.search(claim)
    if fng is not None:
        value = _number(fng["value"])
        if value is None or value != value.to_integral_value() or not 0 <= value <= 100:
            out.append(Finding(
                item_id, "non_numeric", fng["value"],
                f"Fear & Greed reads {fng['value']!r}, not an integer in 0-100",
                (fng.start("value"), fng.end("value")),
            ))
    sue = _SUE.search(claim)
    if sue is not None:
        value = _number(sue["value"])
        if value is None:
            out.append(Finding(
                item_id, "non_numeric", sue["value"], f"SUE reads {sue['value']!r}",
                (sue.start("value"), sue.end("value")),
            ))
        else:
            direction = sue["direction"]
            wrong = (value > 0 and direction == "below") or (value < 0 and direction == "above")
            if wrong:
                out.append(Finding(
                    item_id, "sue_direction", sue["value"],
                    f"SUE {sue['value']} is described as '{direction}' the trailing volatility; "
                    f"the sign and the word contradict each other",
                ))


def _check_ratios(item_id: str, claim: str, out: list[Finding]) -> None:
    qoq = _QOQ.search(claim)
    if qoq is not None and _number(qoq["value"]) is None:
        out.append(Finding(
            item_id, "non_numeric", qoq["value"],
            f"quarter-on-quarter change reads {qoq['value']!r}",
            (qoq.start("value"), qoq.end("value")),
        ))
    short = _SHORT.search(claim)
    if short is not None:
        share = _number(short["share"])
        parts = _number(short["short"].replace(",", "")), _number(short["total"].replace(",", ""))
        if share is None:
            out.append(Finding(
                item_id, "non_numeric", short["share"], f"short share reads {short['share']!r}",
                (short.start("share"), short.end("share")),
            ))
        elif parts[0] is not None and parts[1] is not None and parts[1] > 0:
            implied = parts[0] / parts[1] * 100
            if abs(implied - share) > SHORT_SHARE_TOLERANCE:
                out.append(Finding(
                    item_id, "ratio_inconsistent", short["share"],
                    f"short share {short['share']}% but {short['short']} of {short['total']} is "
                    f"{implied:.1f}%; one of the three figures is wrong",
                ))
    curve = _CURVE.search(claim)
    if curve is not None:
        tenors: dict[str, Decimal] = {}
        for m in _TENOR.finditer(curve["body"]):
            value = _number(m["value"])
            if value is None:
                start = curve.start("body") + m.start("value")
                out.append(Finding(
                    item_id, "non_numeric", m["value"],
                    f"the {m['tenor']} yield reads {m['value']!r}",
                    (start, start + len(m["value"])),
                ))
            else:
                tenors[m["tenor"]] = value
        spread = _number(curve["spread"])
        if spread is None:
            out.append(Finding(
                item_id, "non_numeric", curve["spread"], f"10y-2y reads {curve['spread']!r}",
                (curve.start("spread"), curve.end("spread")),
            ))
        elif "10y" in tenors and "2y" in tenors:
            implied = (tenors["10y"] - tenors["2y"]) * 100
            if abs(implied - spread) > CURVE_TOLERANCE_BPS:
                out.append(Finding(
                    item_id, "ratio_inconsistent", curve["spread"],
                    f"10y-2y stated {curve['spread']}bps but 10y {tenors['10y']}% - 2y "
                    f"{tenors['2y']}% is {implied:+.0f}bps; a tenor or the spread is wrong",
                ))


def check_panel_line(
    item_id: str, text: str, *, move_24h: Decimal | None
) -> list[Finding]:
    """The analyst/quant rules for one ``[panel] <analyst>: <signal> <N>bps — ...`` line."""
    out: list[Finding] = []
    m = _PANEL.search(text)
    if m is not None:
        edge = _number(m["edge"])
        if edge is None:
            out.append(Finding(item_id, "non_numeric", m["edge"],
                               f"{m['analyst']} edge reads {m['edge']!r}"))
        elif abs(edge) >= ABSURD_EDGE_BPS:
            out.append(Finding(
                item_id, "absurd_edge", m["edge"],
                f"{m['analyst']} claims a {edge}bps expected edge, at or beyond "
                f"{ABSURD_EDGE_BPS}bps — over twice a one-sigma equity day",
            ))
    if move_24h is not None:
        move_bps = abs(move_24h) * 10_000
        if move_bps >= MIN_MOVE_BPS_FOR_UNIT_CHECK:
            image = move_bps * UNIT_ERROR_FACTOR
            for fig in _BPS.finditer(text):
                value = _number(fig.group(1))
                if value is not None and abs(abs(value) - image) <= image * UNIT_ERROR_TOLERANCE:
                    out.append(Finding(
                        item_id, "unit_error", fig.group(1),
                        f"{fig.group(1)}bps is 100x the feed's own 24h move of "
                        f"{move_bps.normalize():f}bps: a fraction read as a percent",
                    ))
                    break
    return out


def check_claim(
    item_id: str, claim: str, *, token_price: Decimal | None, move_24h: Decimal | None = None
) -> list[Finding]:
    """Every rule that applies to one claim. Pure; the same input always gives the same findings."""
    if claim.startswith("[panel]"):
        return check_panel_line(item_id, claim, move_24h=move_24h)
    out: list[Finding] = []
    _check_market(item_id, claim, token_price, out)
    _check_macro(item_id, claim, out)
    _check_ratios(item_id, claim, out)
    return out


def screen_claims(
    items: Sequence[tuple[str, str]], *, token_price: Decimal | None, scope: str = "evidence"
) -> SanityReport:
    """Check ``(item_id, claim)`` pairs. The market move for the unit check is read from them."""
    move = market_move([c for _, c in items])
    findings: list[Finding] = []
    for item_id, claim in items:
        findings.extend(check_claim(item_id, claim, token_price=token_price, move_24h=move))
    return SanityReport(checked=len(items), findings=tuple(findings), scope=scope)


def redact(claim: str, findings: Sequence[Finding]) -> str:
    """The claim with each failing figure replaced, or the whole claim when a rule demands it."""
    if any(f.span is None for f in findings):
        rules = ", ".join(sorted({f.rule for f in findings}))
        return WITHHELD.format(rule=rules) + " the figures on this line contradict each other"
    out = claim
    for f in sorted(findings, key=lambda f: f.span[0] if f.span else 0, reverse=True):
        assert f.span is not None
        out = out[: f.span[0]] + WITHHELD.format(rule=f.rule) + out[f.span[1]:]
    return out


def screen_evidence(
    evidence: Sequence[Evidence], *, token_price: Decimal | None
) -> tuple[list[Evidence], SanityReport]:
    """Withhold every failing figure before anything reasons over the evidence.

    Identity is kept — id, source, credibility and availability are unchanged — so selection,
    provenance and every count downstream see the same items; only the untrustworthy text is gone.
    """
    report = screen_claims([(e.id, e.claim) for e in evidence], token_price=token_price)
    if report.clean:
        return list(evidence), report
    by_id: dict[str, list[Finding]] = {}
    for f in report.findings:
        by_id.setdefault(f.item_id, []).append(f)
    screened = [
        replace(e, claim=redact(e.claim, by_id[e.id])) if e.id in by_id else e for e in evidence
    ]
    return screened, report


def view_line(view: AnalystView) -> str:
    """An analyst view's text as the frame renders it (`agents/desk.py`), plus its counter-case."""
    return (
        f"[panel] {view.analyst}: {view.signal} {view.magnitude_bps}bps "
        f"(confidence {view.confidence:.2f}) — {view.reasoning} {view.counter_case}"
    )


def screen_views(
    views: Sequence[AnalystView], *, move_24h: Decimal | None
) -> tuple[list[AnalystView], SanityReport]:
    """Drop every analyst view whose line fails a panel rule, before consensus is taken."""
    findings: list[Finding] = []
    kept: list[AnalystView] = []
    for view in views:
        got = check_panel_line(f"panel:{view.analyst}", view_line(view), move_24h=move_24h)
        findings.extend(got)
        if not got:
            kept.append(view)
    return kept, SanityReport(checked=len(views), findings=tuple(findings), scope="analyst")


__all__ = [
    "ABSURD_EDGE_BPS",
    "MAX_ABS_CHANGE_24H",
    "PRICE_TOLERANCE",
    "RULES",
    "VIX_TOLERANCE_BPS",
    "Finding",
    "SanityReport",
    "check_claim",
    "check_panel_line",
    "market_move",
    "redact",
    "screen_claims",
    "screen_evidence",
    "screen_views",
    "view_line",
]
