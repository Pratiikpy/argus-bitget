"""Groupwise audit — every artefact the capability register cites, broken down by its groups.

:mod:`argus.eval.groupwise` is the check; this module runs it over the evidence. For each artefact
the register names (``standing.capability_artefacts`` and the scope of every
statistical-validity and out-of-sample proof, ``standing.proof_scope``), plus the result artefacts
the project's recorded failures came from (the weekend study, the shadow record, the PEAD and
overfitting grids, the search bake-off), it answers one question: **is there a per-item record
behind the headline, and when it is broken down by symbol, date or regime, does it still say what
the headline says?**

Every artefact ends in exactly one of four states, and the reason is written beside it:

``checked``                 Per-item rows were found (or regenerated, see below) and at least one
                            headline was broken down into two or more groups, or split into
                            chronological halves.
``designed_cases``          The evidence is cases an author designed, or a parameter sweep —
                            a demonstration that a property holds on chosen inputs. There is no
                            population to break down, so no groupwise check can run. This is not a
                            criticism of the demonstration; it is a statement that it is not a
                            measurement over symbols, dates or random draws.
``aggregate_without_rows``  The headline *is* an aggregate over a population (queries, books,
                            windows, sealed prompts) but the artefact kept only the aggregate. The
                            check could run and cannot, because the rows are gone. The harness
                            that wrote it is named, so the rows can be recorded.
``no_rows``                 A qualitative summary (one sentence through two desks, three theses):
                            nothing numeric per item.

Each checked headline carries a **role**, because an artefact often holds more than one number and
only some of them are the capability's claim:

``argus_vs_rival``  ARGUS against the named baseline, per item, oriented so a positive value means
                    ARGUS did better.
``argus_result``    ARGUS's own measured property per item (a gate's verdict, a parity with the
                    reference, a divergence rate) with no rival arm in the row.
``context``         A number in the artefact that is not the capability's claim — the rival's
                    own P&L, the strategies a gate judged, the desk's abstention value. Its flags
                    are listed (they are real findings) but never gate a capability.

**Regenerated rows.** Three harnesses kept only pooled figures although their offline pipelines
produce the rows from saved inputs: the overnight gap (``overnight_comparison.predict`` over
``h2h_gloaming/inputs.json``), nocturne's weekend walk-forward (``void_comparison.predictions``)
and the risk layer's combined book (``risk_layer_comparison._combined_book_checkpoints`` over its
frozen candle fixture). Their rows are regenerated here, and each regeneration is **parity-checked
against the artefact's own published aggregate before it is used**; a mismatch raises instead of
auditing numbers that are not the published ones. No other harness is re-run: where rows are
missing, the artefact is marked ``aggregate_without_rows`` and the missing field is named.

No model is called; nothing here spends Qwen. TweetEval and other unlicensed datasets are not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval import artefact
from argus.eval.groupwise import GroupwiseReport, Item, audit, halves_from

PACKAGE = Path(__file__).resolve().parents[3]
DATA = PACKAGE / "data"
REPORT_PATH = DATA / "groupwise_audit.json"

CHECKED = "checked"
DESIGNED = "designed_cases"
WITHOUT_ROWS = "aggregate_without_rows"
NO_ROWS = "no_rows"
STATUSES = (CHECKED, DESIGNED, WITHOUT_ROWS, NO_ROWS)

VS_RIVAL = "argus_vs_rival"
RESULT = "argus_result"
CONTEXT = "context"
ROLES = (VS_RIVAL, RESULT, CONTEXT)
GATING_ROLES = frozenset({VS_RIVAL, RESULT})
"""The roles whose flags a capability's statistical-validity and out-of-sample proofs answer to."""

PARITY_TOLERANCE = 0.011
"""How far a regenerated aggregate may sit from the published one: the artefacts round to two
decimals, so anything beyond a rounding step means the rows are not the published rows."""


class AuditError(RuntimeError):
    """A regenerated record that does not reproduce its artefact, or an unhandled artefact."""


@dataclass(frozen=True)
class Headline:
    """One audited headline inside an artefact, with its role and where its rows came from."""

    report: GroupwiseReport
    role: str
    role_reason: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        blob = self.report.as_dict()
        direction = next((t.direction for t in self.report.tables), 0)
        favours = "neither"
        if self.role == VS_RIVAL:
            favours = "argus" if direction > 0 else "rival" if direction < 0 else "neither"
        blob.update({
            "role": self.role, "role_reason": self.role_reason, "source": self.source,
            "split_ran": self.report.split is not None,
            "favours": favours,
        })
        return blob


@dataclass(frozen=True)
class Entry:
    """One artefact's state and, when checked, its audited headlines."""

    artefact: str
    status: str
    reason: str
    headlines: tuple[Headline, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise AuditError(f"{self.artefact}: unknown status {self.status!r}")
        if self.status == CHECKED and not self.headlines:
            raise AuditError(f"{self.artefact}: checked with no headline")
        for h in self.headlines:
            if h.role not in ROLES:
                raise AuditError(f"{self.artefact}: unknown role {h.role!r}")
        # A headline whose only declared group has one value is still a check that ran: its
        # answer is the flag ``single_group``. What is refused is a "checked" entry that neither
        # broke anything down, split anything, nor found anything.
        if self.status == CHECKED and not any(
                h.report.broken_down or h.report.split is not None or h.report.flagged
                for h in self.headlines):
            raise AuditError(f"{self.artefact}: 'checked' but nothing was broken down or split")

    def as_dict(self) -> dict[str, Any]:
        target = PACKAGE / self.artefact
        return {
            "status": self.status, "reason": self.reason,
            # The file as audited. `standing.groupwise_verdict` compares it with the file as it
            # is, so an artefact regenerated after this audit is not vouched for by it.
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest() if target.exists()
            else None,
            "flagged": any(h.report.flagged for h in self.headlines),
            "headlines": [h.as_dict() for h in self.headlines],
        }


def _load(path: str) -> Any:
    target = PACKAGE / path
    if target.suffix == ".jsonl":
        return [json.loads(line) for line in target.read_text("utf-8").splitlines()
                if line.strip()]
    return json.loads(target.read_text("utf-8"))


def _f(value: Any) -> float:
    return float(value)


def _headline(name: str, items: Sequence[Item], keys: Sequence[str], *, headline: str,
              orientation: str, role: str, role_reason: str, source: str,
              halves: Any = None, notes: Sequence[str] = ()) -> Headline:
    return Headline(
        report=audit(name, items, keys, headline=headline, orientation=orientation,
                     halves=halves, notes=notes),
        role=role, role_reason=role_reason, source=source)


def _parity(what: str, regenerated: float, published: float) -> None:
    if abs(regenerated - published) > PARITY_TOLERANCE:
        raise AuditError(
            f"{what}: regenerated {regenerated} against published {published}; the rows are "
            f"not the rows the artefact was computed from, so they are not audited")


# ---------------------------------------------------------------------------------------------
# Head-to-heads with per-item rows (recorded or regenerated)


def _overnight(path: str) -> Entry:
    from argus.eval import overnight_comparison as oc

    blob = _load(path)
    inputs = json.loads((oc.DATA / "inputs.json").read_text("utf-8"))
    rows = [r for r in oc.predict(oc.nights(inputs)) if r["fitted"]]
    best = str(blob["best_gloaming"])

    def mae(name: str) -> float:
        return float(sum(abs(r["estimates"][name] - r["gap"]) for r in rows) / len(rows) * 1e4)

    for name in ("argus_perp", best, "zero"):
        _parity(f"{path} {name} MAE", round(mae(name), 2), _f(blob["summary"][name]["mae_bps"]))

    def items(rival: str) -> list[Item]:
        return [Item(value=(abs(r["estimates"][rival] - r["gap"])
                            - abs(r["estimates"]["argus_perp"] - r["gap"])) * 1e4,
                     groups={"symbol": str(r["stock"]), "night": str(r["opened"])},
                     order=str(r["opened"])) for r in rows]

    source = ("regenerated: overnight_comparison.predict(nights(h2h_gloaming/inputs.json)), "
              "fitted rows; MAE of argus_perp, the best gloaming variant and zero reproduce the "
              "artefact's summary to the published two decimals")
    return Entry(path, CHECKED, "per-night rows regenerated from the saved inputs", (
        _headline("overnight gap: ARGUS vs gloaming's best variant", items(best),
                  ("symbol", "night"), headline=f"MAE argus_perp vs {best}, bps per night-stock",
                  orientation=f"|{best} error| - |argus_perp error|, bps: positive = ARGUS better",
                  role=VS_RIVAL, role_reason="the comparison's primary: the perpetual-implied "
                  "open against gloaming's best variant on the same nights", source=source),
        _headline("overnight gap: ARGUS vs no gap", items("zero"), ("symbol", "night"),
                  headline="MAE argus_perp vs the last close (no gap), bps",
                  orientation="|zero error| - |argus_perp error|, bps: positive = ARGUS better",
                  role=VS_RIVAL, role_reason="the comparison's floor", source=source),
    ))


def _void(path: str) -> Entry:
    from argus.eval import void_comparison as vc

    blob = _load(path)
    preds = [p for p in vc.predictions() if p["ARGUS"] is not None]
    for name in ("ARGUS", "MF", "M0"):
        regenerated = sum(abs(p[name] - p["actual"]) for p in preds) / len(preds) * 100
        _parity(f"{path} {name} MAE", round(regenerated, 3), _f(blob["summary"][name]["mae_pct"]))

    def items(rival: str) -> list[Item]:
        return [Item(value=(abs(p[rival] - p["actual"]) - abs(p["ARGUS"] - p["actual"])) * 100,
                     groups={"symbol": str(p["sym"]), "weekend": str(p["fri"])},
                     order=str(p["fri"])) for p in preds]

    source = ("regenerated: void_comparison.predictions() (nocturne's own walk-forward from its "
              "clone); ARGUS, MF and M0 MAE reproduce the artefact's summary")
    return Entry(path, CHECKED, "per-row walk-forward regenerated from nocturne's clone", (
        _headline("rToken Monday price: ARGUS vs nocturne's full fade", items("MF"),
                  ("symbol", "weekend"), headline="MAE ARGUS vs MF, percentage points",
                  orientation="|MF error| - |ARGUS error|, pp: positive = ARGUS better",
                  role=VS_RIVAL, role_reason="nocturne's best model on its own question",
                  source=source),
        _headline("rToken Monday price: ARGUS vs no change", items("M0"), ("symbol", "weekend"),
                  headline="MAE ARGUS vs the Sunday price", orientation="|M0 error| - |ARGUS "
                  "error|, pp: positive = ARGUS better", role=VS_RIVAL,
                  role_reason="nocturne's floor", source=source),
    ))


def _claimcheck(path: str) -> Entry:
    rows = [r for r in _load(path)["rows"] if r["graded"]]
    items = [Item(value=float(bool(r["argus_correct"])) - float(bool(r["mirrorline_correct"])),
                  groups={"kind": str(r["kind"])}) for r in rows]
    return Entry(path, CHECKED, "one row per graded claim", (
        _headline("claims about the tape: ARGUS vs MirrorLine", items, ("kind",),
                  headline="share of graded claims judged correctly",
                  orientation="ARGUS correct - MirrorLine correct per claim: positive = ARGUS",
                  role=VS_RIVAL, role_reason="the comparison's primary", source="rows[graded]",
                  notes=("one morning's claims: no chronological order to split",)),))


def _stopquality(path: str) -> Entry:
    rows = [r for r in _load(path)["rows"]
            if r["rook_noise_hit_rate_oos"] is not None
            and r["argus_noise_hit_rate_oos"] is not None]
    items = [Item(value=_f(r["rook_noise_hit_rate_oos"]) - _f(r["argus_noise_hit_rate_oos"]),
                  groups={"symbol": str(r["symbol"])}) for r in rows]
    return Entry(path, CHECKED, "one row per name", (
        _headline("stop hit by noise: ARGUS vs Rook", items, ("symbol",),
                  headline="share of held-out 24h windows whose low reached the stop",
                  orientation="Rook hit rate - ARGUS hit rate: positive = ARGUS's stop is hit "
                  "less often by ordinary movement", role=VS_RIVAL,
                  role_reason="the comparison's only metric", source="rows",
                  notes=("the ARGUS distance scored here is the harness's own _p90_adverse, not "
                         "desk/odds.py (data/harness_validity.json: desk/odds.py never "
                         "executes in this harness)",)),))


def _copilot_rivals(path: str) -> Entry:
    blob = _load(path)
    rows = blob["rows"]
    target = "bitget_daily"
    for arm in ("argus_open", "rival_qqq", "naive_one"):
        regenerated = sum(abs(r["pred"][arm] - r["real"][target]) for r in rows) / len(rows)
        _parity(f"{path} {arm} MAE", round(regenerated, 4),
                _f(blob["beta_error"][target]["mean_abs_error"][arm]))

    def items(rival: str) -> list[Item]:
        return [Item(value=abs(r["pred"][rival] - r["real"][target])
                     - abs(r["pred"]["argus_open"] - r["real"][target]),
                     groups={"origin": str(r["origin"]), "added": str(r["add"])},
                     order=str(r["origin"])) for r in rows]

    return Entry(path, CHECKED, "one row per book (1,800 books over nine origins)", (
        _headline("post-trade beta: ARGUS vs weekend-copilot (QQQ)", items("rival_qqq"),
                  ("origin", "added"), headline="mean absolute beta error, Bitget daily target",
                  orientation="|rival_qqq error| - |argus_open error|: positive = ARGUS better",
                  role=VS_RIVAL, role_reason="the pre-registered primary", source="rows"),
        _headline("post-trade beta: ARGUS vs beta = 1", items("naive_one"), ("origin", "added"),
                  headline="mean absolute beta error against calling every beta 1.0",
                  orientation="|naive error| - |argus_open error|: positive = ARGUS better",
                  role=VS_RIVAL, role_reason="the naive floor the register records ARGUS does "
                  "not beat significantly", source="rows"),
    ))


def _copilot_hedge(path: str) -> Entry:
    per_name = _load(path)["per_name"]
    items = [Item(value=_f(v["argus_same_name"]["held_out_variance_removed"])
                  - _f(v["ballast_same_name"]["held_out_variance_removed"]),
                  groups={"name": str(k)}, weight=_f(v["argus_same_name"]["held_out_nights"]))
             for k, v in per_name.items()]
    return Entry(path, CHECKED, "one row per name, weighted by its held-out nights", (
        _headline("overnight hedge: ARGUS same-name vs Ballast", items, ("name",),
                  headline="held-out share of overnight variance removed",
                  orientation="ARGUS - Ballast: positive = ARGUS removes more", role=VS_RIVAL,
                  role_reason="the comparison's primary", source="per_name",
                  notes=("each row stands for its name's held-out nights; the histogram counts "
                         "names, not nights",)),))


def _execution_arena(path: str) -> Entry:
    blob = _load(path)
    source = str(blob["referee"]["source"])
    symbol = re.search(r"\b([A-Z]{2,}USDT)\b", source)
    day = re.search(r"\d{4}-\d{2}-\d{2}", source)
    items = [Item(value=_f(v["cost_bps"]["bitget_twap_60s"]) - _f(v["cost_bps"]["argus_ac_60s"]),
                  groups={"size": str(k), "symbol": symbol.group(1) if symbol else "?",
                          "day": day.group(0) if day else "?"},
                  weight=_f(v["parents"])) for k, v in blob["by_size"].items()]
    return Entry(path, CHECKED, "one row per parent size (80 parents each); per-parent rows are "
                 "kept only as a 16-row sample", (
        _headline("order splitting: ARGUS one-minute AC vs Bitget TWAP", items,
                  ("size", "symbol", "day"), headline="cost in bps with fees, per parent",
                  orientation="Bitget TWAP cost - ARGUS AC(60s) cost: positive = ARGUS cheaper",
                  role=VS_RIVAL, role_reason="the schedule the console now prints against "
                  "Bitget's own TWAP", source="by_size"),))


def _risk_layer(path: str) -> Entry:
    from argus.eval import risk_layer_comparison as rl
    from argus.risk.circuit import REDUCE_ONLY_DRAWDOWN

    blob = _load(path)
    checkpoints, _, _, _ = rl._combined_book_checkpoints(frozen=True)
    truth_cut = float(REDUCE_ONLY_DRAWDOWN)
    truth = [c.true_drawdown_pct >= truth_cut for c in checkpoints]

    def precision(calls: Sequence[bool]) -> tuple[float, float]:
        tp = sum(1 for p, a in zip(calls, truth, strict=True) if p and a)
        predicted, actual = sum(calls), sum(truth)
        return (tp / predicted if predicted else 1.0), (tp / actual if actual else 1.0)

    argus_calls = [c.argus_locked for c in checkpoints]
    arp, _ = precision(argus_calls)
    published = blob["measurement_architecture"]
    _parity(f"{path} ARGUS precision", round(arp, 4), _f(published["argus_real_point"]
                                                         ["precision"]))
    best_t, best_p = None, -1.0
    for row in published["freqtrade_curve"]:
        t = _f(row["threshold"])
        p, r = precision([c.freqtrade_drawdown_measured >= t for c in checkpoints])
        _parity(f"{path} freqtrade precision at {t}", round(p, 4), round(_f(row["precision"]), 4))
        if r >= 0.999 and p > best_p:
            best_t, best_p = t, p
    if best_t is None:
        raise AuditError(f"{path}: no freqtrade threshold reaches full recall")
    ft_calls = [c.freqtrade_drawdown_measured >= best_t for c in checkpoints]
    items = [Item(value=float(a == t) - float(f == t),
                  groups={"symbol": str(c.symbol), "half": "oos" if c.is_oos else "in_sample"},
                  order=c.exit_ts.isoformat())
             for c, a, f, t in zip(checkpoints, argus_calls, ft_calls, truth, strict=True)]
    ins = [i for i, c in zip(items, checkpoints, strict=True) if not c.is_oos]
    oos = [i for i, c in zip(items, checkpoints, strict=True) if c.is_oos]
    halves = halves_from(sum(i.value for i in ins) / len(ins), sum(i.value for i in oos) / len(oos),
                         first_items=len(ins), second_items=len(oos),
                         label="the harness's own in-sample / out-of-sample boundary")
    return Entry(path, CHECKED, "per-checkpoint rows regenerated from the frozen candle fixture", (
        _headline("drawdown lock: ARGUS vs freqtrade at its best full-recall threshold", items,
                  ("symbol",), headline="lock called correctly against the shared ground truth "
                  f"(true drawdown >= {truth_cut:g})",
                  orientation=f"ARGUS correct - freqtrade (MaxDrawdown >= {best_t:g}) correct "
                  "per checkpoint: positive = ARGUS. Both have full recall, so this difference "
                  "counts exactly the false locks separating the two published precisions",
                  role=VS_RIVAL, role_reason="the decomposable form of the published "
                  "precision-at-full-recall comparison", halves=halves,
                  source="regenerated: risk_layer_comparison._combined_book_checkpoints"
                  "(frozen=True); ARGUS precision and every freqtrade curve point reproduce "
                  "measurement_architecture to four decimals"),))


def _schedule(path: str) -> Entry:
    per = _load(path)["out_of_sample"]["per_symbol"]
    items = [Item(value=_f(v["out_of_sample_savings_pct"]), groups={"symbol": str(k)})
             for k, v in per.items()]
    first = sum(_f(v["in_sample_savings_pct"]) for v in per.values()) / len(per)
    second = sum(_f(v["out_of_sample_savings_pct"]) for v in per.values()) / len(per)
    return Entry(path, CHECKED, "one row per symbol, in-sample and out-of-sample", (
        _headline("execution schedule: Almgren-Chriss vs TWAP", items, ("symbol",),
                  headline="expected-cost saving of AC over TWAP, out of sample, %",
                  orientation="saving %: positive = AC cheaper", role=VS_RIVAL,
                  role_reason="the comparison's out-of-sample result", source="out_of_sample",
                  halves=halves_from(first, second, first_items=len(per), second_items=len(per),
                                     label="in-sample vs out-of-sample saving by symbol")),))


def _queue_proof(path: str) -> Entry:
    blob = _load(path)
    oos = blob["out_of_sample"]
    ins = blob["in_sample"]
    items = [Item(value=_f(r["margin"]), groups={"regime": str(r["regime"])}) for r in oos]
    first = sum(_f(r["margin"]) for r in ins) / len(ins)
    second = sum(_f(r["margin"]) for r in oos) / len(oos)
    return Entry(path, CHECKED, "one row per simulated cancellation regime (4,000 episodes each)", (
        _headline("queue position: best model vs best naive rule", items, ("regime",),
                  headline="queue-ahead error margin, out of sample",
                  orientation="naive error - model error: positive = the model is closer",
                  role=VS_RIVAL, role_reason="the comparison's primary", source="out_of_sample",
                  halves=halves_from(first, second, first_items=len(ins), second_items=len(oos),
                                     label="in-sample vs out-of-sample episodes")),))


def _regime(path: str) -> Entry:
    trials = _load(path)["synthetic_groundtruth"]["trials"]

    def items(argus: str) -> list[Item]:
        return [Item(value=_f(t[argus]) - _f(t["ruptures_f1"]),
                     groups={"noise": str(t["noise_std"])}) for t in trials]

    return Entry(path, CHECKED, "100 seeded synthetic series with known boundaries", (
        _headline("boundaries: ARGUS exact partition vs ruptures", items("argus_dynp_f1"),
                  ("noise",), headline="boundary F1 on synthetic ground truth",
                  orientation="ARGUS F1 - ruptures F1: positive = ARGUS", role=VS_RIVAL,
                  role_reason="the tool that moved the capability to TIED", source="trials"),
        _headline("boundaries: FLUSS vs ruptures", items("argus_f1"), ("noise",),
                  headline="boundary F1 on synthetic ground truth",
                  orientation="FLUSS F1 - ruptures F1: positive = FLUSS", role=VS_RIVAL,
                  role_reason="ARGUS's original segmenter, the recorded loss", source="trials"),
    ))


def _rotation(path: str) -> Entry:
    blob = _load(path)
    rows = list(blob["baseline_reproduced"]["results"]) + list(
        blob["oos_wider_universe"]["results"])
    items = [Item(value=1.0 if r["agree"] else -1.0, groups={"symbol": str(r["symbol"])})
             for r in rows]
    return Entry(path, CHECKED, "one row per real instrument, design set and held-out set", (
        _headline("breadth score: ARGUS agrees with pytaa or refuses where it is NaN", items,
                  ("symbol",), headline="share of instruments where ARGUS reproduces the "
                  "reference or refuses exactly where it is undefined",
                  orientation="+1 agree, -1 disagree", role=RESULT,
                  role_reason="the parity-and-refusal claim over real instruments",
                  source="baseline_reproduced.results + oos_wider_universe.results"),))


def _earnings(path: str) -> Entry:
    rows = _load(path)["baseline_reproduced"]["results"]
    items = [Item(value=1.0 if r["agree"] else -1.0, groups={"symbol": str(r["symbol"])})
             for r in rows]
    return Entry(path, CHECKED, "one row per real anchor", (
        _headline("SUE: ARGUS reproduces QuantConnect's formula", items, ("symbol",),
                  headline="share of real anchors where the SUE values agree",
                  orientation="+1 agree, -1 disagree", role=RESULT,
                  role_reason="the reproduction over the real anchor universe",
                  source="baseline_reproduced.results"),))


def _explainability(path: str) -> Entry:
    cases = _load(path)["same_input_comparison"]["cases"]
    items = [Item(value=float(bool(c["argus_grounding_flags_it"]))
                  - float(bool(c["tradingagents_real_validator_raised"])),
                  groups={"symbol": str(c["symbol"])}) for c in cases]
    return Entry(path, CHECKED, "one row per fabricated figure on a live price", (
        _headline("fabricated entry price caught: ARGUS vs TraderProposal", items, ("symbol",),
                  headline="fabrications caught", orientation="ARGUS caught - TradingAgents "
                  "caught: positive = ARGUS", role=VS_RIVAL, role_reason="the same-input "
                  "comparison", source="same_input_comparison.cases",
                  notes=("four fabrication magnitudes per symbol are designed, so magnitude is "
                         "not declared as a population key",)),))


def _infoextract(path: str) -> Entry:
    cases = _load(path)["designed_cases"]["cases"]
    items = [Item(value=1.0 if c["resolved"] else 0.0, groups={"ticker": str(c["ticker"])})
             for c in cases]
    return Entry(path, CHECKED, "one row per (ticker, concept) fetched live", (
        _headline("filing extraction resolves the line item", items, ("ticker",),
                  headline="share of cases resolved to exactly one quarterly value",
                  orientation="1 resolved, 0 not", role=RESULT,
                  role_reason="ARGUS's own resolution on live SEC XBRL",
                  source="designed_cases.cases"),))


def _grammar(path: str) -> Entry:
    rows = _load(path)["swept_real_market_cases"]

    def symbol(name: str) -> str:
        found = re.search(r"real_([A-Z]+USDT)", name)
        return found.group(1) if found else "?"

    items = [Item(value=1.0 if r["agrees"] else -1.0,
                  groups={"symbol": symbol(str(r["name"])), "op": str(r["op"])}) for r in rows]
    return Entry(path, CHECKED, "one row per real-market lookback case", (
        _headline("rank on real market series: ARGUS vs qlib", items, ("symbol",),
                  headline="share of real-market windows where the two ranks agree",
                  orientation="+1 agree, -1 disagree", role=RESULT,
                  role_reason="the real-market half of the parity claim",
                  source="swept_real_market_cases"),))


def _sentiment(path: str) -> Entry:
    rows = _load(path)["narratives"]

    def symbol(text: str) -> str:
        found = re.search(r"\b([A-Z]{2,}USDT)\b", text)
        return found.group(1) if found else "?"

    items = [Item(value=float(bool(r["argus_discounts_coordination"]))
                  - float(not r["finbert_signal_scales_with_repetition"]),
                  groups={"symbol": symbol(str(r["narrative"]))}) for r in rows]
    return Entry(path, CHECKED, "one row per coordinated-posting narrative", (
        _headline("coordinated posting: ARGUS vs finBERT", items, ("symbol",),
                  headline="narratives on which repetition alone did not move the reader",
                  orientation="ARGUS resisted - finBERT resisted: positive = ARGUS",
                  role=VS_RIVAL, role_reason="the adversarial comparison",
                  source="narratives"),))


def _profile_divergence(path: str) -> Entry:
    blob = _load(path)
    by_symbol = [Item(value=_f(r["diverged"]) / _f(r["total"]), groups={"symbol": str(r["symbol"])},
                      weight=_f(r["total"])) for r in blob["by_symbol"]]
    rungs = [Item(value=_f(r["rate_pct"]) / 100.0,
                  groups={"rung": f"{r['notional']}/{r['loss_pct']}"}, weight=_f(r["frames"]))
             for r in blob["rungs"]]
    note = ("divergence is a rate, not a signed advantage: the breakdown asks whether one group "
            "supplies it, and the capability's own record says the rungs, not the symbols, do")
    return Entry(path, CHECKED, "proposal counts per symbol and per declared size/loss rung", (
        _headline("profiles diverge, by symbol", by_symbol, ("symbol",),
                  headline="share of proposals on which the two mandates diverge",
                  orientation="diverged / total", role=RESULT,
                  role_reason="the measured effect of the mandate over real frames",
                  source="by_symbol", notes=(note,)),
        _headline("profiles diverge, by rung", rungs, ("rung",),
                  headline="share of frames that diverge at each declared rung",
                  orientation="rate", role=RESULT, role_reason="the same effect by the "
                  "dimension the register names as its source", source="rungs",
                  notes=(note,)),
    ))


def _factor_divergence(path: str) -> Entry:
    blob = _load(path)
    oos = blob["oos_check"]
    first, second = oos["first_window"]["ic_difference"], oos["second_window"]["ic_difference"]
    items = [Item(value=_f(first["mean"]), groups={"window": "first"}, weight=_f(first["n"]),
                  order="1"),
             Item(value=_f(second["mean"]), groups={"window": "second"}, weight=_f(second["n"]),
                  order="2")]
    return Entry(path, CHECKED, "IC-difference means per chronological window; per-symbol and "
                 "per-reading rows are not kept", (
        _headline("market-vs-index IC difference, two windows", items, ("window",),
                  headline="mean IC difference (market - index), Alpha 23",
                  orientation="market IC - index IC", role=RESULT,
                  role_reason="the measured divergence the capability reports",
                  source="oos_check", halves=halves_from(
                      _f(first["mean"]), _f(second["mean"]), first_items=int(first["n"]),
                      second_items=int(second["n"]), label="first vs second 30-day window"),
                  notes=("per-symbol IC is pooled inside each reading, so no symbol breakdown "
                         "can run: the symbol dimension is recorded as missing, not passed",)),))


def _track1(path: str) -> Entry:
    rows = _load(path)["per_symbol"]
    beat = [Item(value=_f(r["best_net_sharpe"]) - _f(r["baseline_sharpe"]),
                 groups={"symbol": str(r["symbol"])}) for r in rows]
    best = [Item(value=_f(r["best_net_sharpe"]), groups={"symbol": str(r["symbol"])})
            for r in rows]
    ins = [_f(r["out_of_sample_decay"]["in_sample_sharpe"]) for r in rows]
    outs = [_f(r["out_of_sample_decay"]["out_of_sample_sharpe"]) for r in rows]
    gate = [Item(value=1.0 if _f(r["dsr_all_trials"]) > 0.95 else 0.0,
                 groups={"symbol": str(r["symbol"])}) for r in rows]
    return Entry(path, CHECKED, "one row per symbol", (
        _headline("deflated-Sharpe gate verdict over all trials", gate, ("symbol",),
                  headline="symbols whose best variant survives DSR over all trials (> 0.95)",
                  orientation="1 survives, 0 rejected", role=RESULT,
                  role_reason="the gate's own verdict, the thing the overfitting-gates "
                  "capability claims", source="per_symbol.dsr_all_trials"),
        _headline("best variant beats buy-and-hold", beat, ("symbol",),
                  headline="best net Sharpe - buy-and-hold Sharpe", orientation="positive = "
                  "the selected variant beat holding", role=CONTEXT,
                  role_reason="the strategies the gate judges, not the gate",
                  source="per_symbol"),
        _headline("best variant's net Sharpe, in and out of sample", best, ("symbol",),
                  headline="net Sharpe of each symbol's selected variant",
                  orientation="Sharpe", role=CONTEXT, role_reason="the strategies the gate "
                  "judges", source="per_symbol.out_of_sample_decay",
                  halves=halves_from(sum(ins) / len(ins), sum(outs) / len(outs),
                                     first_items=len(ins), second_items=len(outs),
                                     label="in-sample vs out-of-sample Sharpe of the "
                                           "selected variant")),
    ))


def _crosssection_study(path: str) -> Entry:
    rows = _load(path)["results"]
    groups = [{"factor": str(r["name"]), "rule": f"{r['rebalance_every']}h/{r['band']}"}
              for r in rows]
    gate = [Item(value=1.0 if isinstance(r["dsr_all_trials"], float)
                 and r["dsr_all_trials"] >= 0.95 else 0.0, groups=g)
            for r, g in zip(rows, groups, strict=True)]
    net = [Item(value=_f(r["mean_net_sharpe"]), groups=g) for r, g in zip(rows, groups,
                                                                           strict=True)]
    ins = [_f(r["phase_0"]["out_of_sample_decay"]["in_sample_sharpe"]) for r in rows]
    outs = [_f(r["phase_0"]["out_of_sample_decay"]["out_of_sample_sharpe"]) for r in rows]
    return Entry(path, CHECKED, "one row per (factor, trading rule) trial", (
        _headline("deflated-Sharpe verdict per trial", gate, ("factor", "rule"),
                  headline="trials surviving DSR over all 80", orientation="1 survives, 0 not",
                  role=RESULT, role_reason="the evaluation's own verdict",
                  source="results.dsr_all_trials"),
        _headline("cross-sectional factors' mean net Sharpe", net, ("factor", "rule"),
                  headline="mean net Sharpe across phases", orientation="Sharpe",
                  role=CONTEXT, role_reason="the factors the evaluation judges",
                  source="results.mean_net_sharpe",
                  halves=halves_from(sum(ins) / len(ins), sum(outs) / len(outs),
                                     first_items=len(ins), second_items=len(outs),
                                     label="phase-0 chronological in-sample vs out-of-sample "
                                           "Sharpe")),
    ))


def _overfit_gates(path: str) -> Entry:
    blob = _load(path)
    rejected = set(blob["rejected_as_noise"])
    items = [Item(value=1.0 if r["factor"] in rejected else 0.0,
                  groups={"factor": str(r["factor"]), "symbol": str(blob["instrument"])})
             for r in blob["detail"]]
    return Entry(path, CHECKED, "one row per primitive", (
        _headline("anti-overfit gates on the eight primitives", items, ("factor", "symbol"),
                  headline="primitives rejected as indistinguishable from shuffled data",
                  orientation="1 rejected, 0 not", role=RESULT,
                  role_reason="the gates' own verdicts on real market data", source="detail"),))


def _paper_ledger(path: str) -> Entry:
    from argus.eval.scorecard import score_ledger
    from argus.paper.ledger import PaperLedger

    ledger = PaperLedger(path=PACKAGE / path)
    by_seq = {str(e.seq): e for e in ledger.entries}
    scored = score_ledger(ledger)
    items = [Item(value=float(o.value_bps),
                  groups={"symbol": str(by_seq[o.decision_id].symbol),
                          "phase": str(by_seq[o.decision_id].session_phase)},
                  order=str(by_seq[o.decision_id].decided_at))
             for o in scored.abstention_outcomes]
    return Entry(path, CHECKED, "one row per graded abstention on the live paper ledger", (
        _headline("value of standing aside", items, ("symbol", "phase"),
                  headline="abstention value, bps net of the fee never paid "
                  "(observatory.AbstentionOutcome.value_bps via scorecard.score_ledger)",
                  orientation="positive = the abstention avoided a loss", role=CONTEXT,
                  role_reason="the desk's abstention record; the capability claims the "
                  "scoring is honest, not that abstaining paid", source="paper_ledger.jsonl"),))


def _afterhours(path: str) -> Entry:
    rows = _load(path)["base_case"]["blind_long_backtest"]["sessions"]
    items = [Item(value=_f(r["net_of_real_fee_bps"]), groups={"symbol": str(r["symbol"])},
                  order=str(r["start"])) for r in rows]
    return Entry(path, CHECKED, "one row per real holiday session", (
        _headline("the rival's blind pre-holiday long", items, ("symbol",),
                  headline="net bps of an unconditional long through each holiday closure",
                  orientation="positive = the rival's long made money", role=CONTEXT,
                  role_reason="the rival's claimed effect, measured; ARGUS's side is the "
                  "holiday phase existing, a property with no per-session measurement",
                  source="base_case.blind_long_backtest.sessions"),))


def _execution_comparison(path: str) -> Entry:
    rows = _load(path)["leg_pair_sweep"]["results"]
    items = [Item(value=1.0 if r["diverges_past_break_even"] else 0.0,
                  groups={"rtoken": str(r["rtoken"])}) for r in rows]
    return Entry(path, CHECKED, "one row per real (rToken, crypto) leg pair", (
        _headline("routing diverges once funding outweighs slippage", items, ("rtoken",),
                  headline="leg pairs where ARGUS's pick changes past break-even and the "
                  "fee-blind router's does not", orientation="1 diverges, 0 not", role=RESULT,
                  role_reason="the funding-aware routing property on real legs",
                  source="leg_pair_sweep.results"),))


def _review_oos(path: str) -> Entry:
    rows = _load(path)["detail"]
    items = [Item(value=0.0 if not r["gradeable_out_of_sample"] else
                  (1.0 if r["survived"] else -1.0), groups={"rule": str(r["rule"])})
             for r in rows]
    return Entry(path, CHECKED, "one row per standing rule, graded on each chronological half", (
        _headline("rule verdicts survive the held-out half", items, ("rule",),
                  headline="rules whose verdict is the same on the later half",
                  orientation="+1 same verdict, -1 flipped, 0 not gradeable", role=RESULT,
                  role_reason="the review's out-of-sample result", source="detail"),))


def _desk_skill_rows(path: str) -> Entry:
    rows = _load(path)["results"]
    items = [Item(value=_f(r["rate"]), groups={"skill": str(r["skill"])},
                  weight=_f(r["attempts"])) for r in rows]
    return Entry(path, CHECKED, "one row per Bitget Skill tool", (
        _headline("Bitget Skill tools that answer", items, ("skill",),
                  headline="share of attempts answered", orientation="answer rate",
                  role=CONTEXT, role_reason="the upstream service's reliability, not ARGUS's",
                  source="results"),))


# ---------------------------------------------------------------------------------------------
# Result artefacts outside the register (the families the recorded failures came from)


def _weekend(path: str) -> Entry:
    blob = _load(path)
    detail = blob["test_3_per_symbol"]["detail"]
    items = [Item(value=_f(v["rate_pct"]) - 50.0, groups={"symbol": str(k)},
                  weight=_f(v["trials"])) for k, v in detail.items()]
    split = blob["test_2_train_test_split"]
    return Entry(path, CHECKED, "per-symbol weekend continuation counts and the study's own split",
                 (_headline("weekend continuation above a coin flip", items, ("symbol",),
                            headline="continuation rate - 50%", orientation="points above 50%",
                            role=CONTEXT, role_reason="a research study, not a capability",
                            source="test_3_per_symbol + test_2_train_test_split",
                            halves=halves_from(
                                _f(split["train"]["rate_pct"]) - 50.0,
                                _f(split["out_of_sample"]["rate_pct"]) - 50.0,
                                first_items=int(split["train"]["trials"]),
                                second_items=int(split["out_of_sample"]["trials"]),
                                label="the study's own first vs second half")),))


def _shadow(path: str) -> Entry:
    blob = _load(path)
    even = _f(blob["break_even"])
    calls = [c for c in blob["calls"] if c["scored"]]
    items = [Item(value=(1.0 if c["correct"] else 0.0) - even,
                  groups={"symbol": str(c["symbol"])}, order=str(c["decided_on"]))
             for c in calls]
    return Entry(path, CHECKED, "one row per scored directional lean", (
        _headline("the desk's directional leans against break-even", items, ("symbol",),
                  headline="accuracy - break-even accuracy",
                  orientation="positive = the lean clears the fee", role=CONTEXT,
                  role_reason="the desk's shadow record", source="calls[scored]"),))


def _pead(path: str) -> Entry:
    trades = _load(path)["trades"]
    items = [Item(value=_f(t["net_return_bps"]),
                  groups={"rtoken": str(t["rtoken"]), "hold": str(t["hold_hours"])},
                  order=str(t["entry_at"])) for t in trades]
    return Entry(path, CHECKED, "one row per realised trade", (
        _headline("post-earnings drift on rTokens", items, ("rtoken", "hold"),
                  headline="net bps per trade", orientation="net return", role=CONTEXT,
                  role_reason="a research study", source="trades"),))


def _overfitting_grid(path: str) -> Entry:
    rows = _load(path)["grid"]["rows"]
    items = [Item(value=_f(r["net_sharpe"]), groups={"symbol": str(r["symbol"]),
                                                     "variant": str(r["variant"])})
             for r in rows]
    return Entry(path, CHECKED, "one row per (symbol, variant)", (
        _headline("the overfitting study's grid", items, ("symbol", "variant"),
                  headline="net Sharpe per (symbol, variant)", orientation="Sharpe",
                  role=CONTEXT, role_reason="a research study", source="grid.rows"),))


def _search_bakeoff(path: str) -> Entry:
    blob = _load(path)
    rows = blob["results"]
    random_oos = next(_f(r["picked"]["out_of_sample"]) for r in rows if r["name"] == "random")
    items = [Item(value=_f(r["picked"]["out_of_sample"]) - random_oos,
                  groups={"search": str(r["name"]), "symbol": str(blob["symbol"])})
             for r in rows]
    return Entry(path, CHECKED, "one row per search strategy", (
        _headline("search strategy against uniform random", items, ("search", "symbol"),
                  headline="picked candidate's out-of-sample Sharpe - random's",
                  orientation="positive = better than random", role=CONTEXT,
                  role_reason="a research bake-off", source="results"),))


def _carry(path: str) -> Entry:
    rows = _load(path)["profiles"]
    items = [Item(value=_f(r["mean_bps"]), groups={"symbol": str(r["symbol"])},
                  weight=_f(r["settlements"])) for r in rows]
    return Entry(path, CHECKED, "one row per instrument's funding settlements", (
        _headline("funding carry per settlement", items, ("symbol",),
                  headline="mean funding bps per settlement", orientation="bps",
                  role=CONTEXT, role_reason="a research study", source="profiles"),))


def _event_reactions(path: str) -> Entry:
    rows = [r for r in _load(path)["reactions"] if r["average_car_bps"] is not None]
    items = [Item(value=_f(r["average_car_bps"]), groups={"symbol": str(r["symbol"]),
                                                          "kind": str(r["kind"])},
                  weight=_f(r["events"])) for r in rows]
    return Entry(path, CHECKED, "one row per (symbol, event kind) with enough events for an "
                 "average; cells below the study's own minimum carry no average and are left "
                 "out, as the study leaves them out", (
        _headline("abnormal return around scheduled events", items, ("symbol", "kind"),
                  headline="average cumulative abnormal return, bps", orientation="bps",
                  role=CONTEXT, role_reason="a research study", source="reactions"),))


def _cointegration_study(path: str) -> Entry:
    blob = _load(path)
    pairs = blob["pairs"]
    survivors = set(blob["survivors_fdr"])
    gate = [Item(value=1.0 if p["pair"] in survivors else 0.0,
                 groups={"pair": str(p["pair"])}) for p in pairs]
    naive = [Item(value=(1.0 if p["in_sample"]["cointegrated_at_5pct"] else 0.0) - 0.05,
                  groups={"pair": str(p["pair"])}) for p in pairs]
    ins = sum(1.0 if p["in_sample"]["cointegrated_at_5pct"] else 0.0 for p in pairs) / len(pairs)
    outs = sum(1.0 if p["out_of_sample"]["stationary_at_5pct"] else 0.0
               for p in pairs) / len(pairs)
    return Entry(path, CHECKED, "one row per real pair", (
        _headline("pairs surviving Benjamini-Hochberg", gate, ("pair",),
                  headline="pairs whose cointegration survives the FDR correction",
                  orientation="1 survives, 0 not", role=RESULT,
                  role_reason="the corrected verdict the capability claims",
                  source="pairs + survivors_fdr"),
        _headline("naive 5% rejections in excess of chance", naive, ("pair",),
                  headline="in-sample rejection at 5% - 0.05", orientation="excess rejection",
                  role=CONTEXT, role_reason="the uncorrected selection the correction refuses",
                  source="pairs.in_sample",
                  halves=halves_from(ins - 0.05, outs - 0.05, first_items=len(pairs),
                                     second_items=len(pairs),
                                     label="in-sample cointegration vs out-of-sample "
                                           "stationarity, each against 5%")),
    ))


def _gap_study(path: str) -> Entry:
    rows = _load(path)["by_phase"]
    items = [Item(value=_f(v["continuation_rate_pct"]) - 50.0, groups={"phase": str(k)},
                  weight=_f(v["sessions"])) for k, v in rows.items()]
    return Entry(path, CHECKED, "per-phase session counts", (
        _headline("closed-session continuation above a coin flip", items, ("phase",),
                  headline="continuation rate - 50%", orientation="points above 50%",
                  role=CONTEXT, role_reason="a research study", source="by_phase"),))


def _rule_proposals(path: str) -> Entry:
    rows = _load(path)["checklist_comparison"]

    def items(arm: str) -> list[Item]:
        out: list[Item] = []
        for row in rows:
            cell = row[arm]
            if not cell["fired"]:
                continue
            out.append(Item(value=_f(cell["precision"]) - _f(cell["base_rate"]),
                            groups={"kind": str(row["kind"])}, weight=_f(cell["fired"])))
        return out

    induced = items("induction_all_pairs")
    hand = items("hand_written")
    heads = []
    if induced:
        heads.append(_headline(
            "induced rules on the held-out half", induced, ("kind",),
            headline="precision - base rate, weighted by firings", orientation="positive = "
            "a firing is likelier to be a real defect than a random decision", role=CONTEXT,
            role_reason="the rule-proposal experiment (not yet a capability row's claim)",
            source="checklist_comparison.induction_all_pairs"))
    if hand:
        heads.append(_headline(
            "hand-written rules on the held-out half", hand, ("kind",),
            headline="precision - base rate, weighted by firings", orientation="as above",
            role=CONTEXT, role_reason="the standing checklist, same record",
            source="checklist_comparison.hand_written"))
    return Entry(path, CHECKED, "per-kind held-out precision against the base rate", tuple(heads))


def _factor_split_half(path: str) -> Entry:
    libraries = _load(path)["libraries"]
    items = [Item(value=1.0 if f["split_half"]["outcome"] == "pass" else 0.0,
                  groups={"symbol": str(lib["symbol"]), "factor": str(f["factor"])})
             for lib in libraries for f in lib["factors"]]
    return Entry(path, CHECKED, "one row per (instrument, factor)", (
        _headline("factor payoffs reproduced across independent halves", items,
                  ("symbol", "factor"), headline="share of factor-instrument pairs passing the "
                  "split-half congruence test", orientation="1 pass, 0 fail or inconclusive",
                  role=CONTEXT, role_reason="the factor library the discovery agent searches",
                  source="libraries[].factors[].split_half"),))


def _document_qa(path: str) -> Entry:
    answers = _load(path)["answers"]
    items = [Item(value=1.0 if a["correct_with_enforcement"] else 0.0,
                  groups={"ticker": str(a["ticker"])})
             for a in answers.values() if a["answerable"]]
    return Entry(path, CHECKED, "one row per answerable question", (
        _headline("filing questions answered correctly with citations enforced", items,
                  ("ticker",), headline="share of answerable questions answered correctly",
                  orientation="1 correct, 0 not", role=RESULT,
                  role_reason="ARGUS's own answers over five issuers' filings",
                  source="answers[answerable]"),))


def _perturbed(path: str) -> Entry:
    blob = _load(path)
    baseline = _load("data/robustness_baseline.json")["baseline"]
    cells = blob["cells"] if "cells" in blob else blob["cases"]
    items: list[Item] = []
    for cell in cells:
        reference = baseline.get(cell["snapshot_id"], {}).get("reference_action")
        if reference is None:
            continue
        symbol, _, moment = str(cell["snapshot_id"]).partition("-")
        items.append(Item(value=1.0 if list(cell["decision"]["action"]) == list(reference)
                          else -1.0, groups={"symbol": symbol, "moment": moment,
                                             "condition": str(cell["condition"])}))
    return Entry(path, CHECKED, "one row per perturbed decision against its snapshot's "
                 "unanimous unperturbed action (data/robustness_baseline.json)", (
        _headline("decision unchanged under the perturbation", items, ("symbol", "moment"),
                  headline="share of perturbed decisions equal to the reference action",
                  orientation="+1 unchanged, -1 changed", role=RESULT,
                  role_reason="MetaPM.decide's invariance under a meaning-preserving edit",
                  source="cells"),))


def _skill_matrix(path: str) -> Entry:
    rows = _load(path)["rows"]
    items = [Item(value=_f(r["answer_rate"]), groups={"server": str(r["server"]),
                                                      "skill": str(r["skill"])},
                  weight=max(_f(r["rounds"]), 1.0)) for r in rows]
    return Entry(path, CHECKED, "one row per tool", (
        _headline("Bitget MCP tools that answer", items, ("server", "skill"),
                  headline="answer rate per tool", orientation="rate", role=CONTEXT,
                  role_reason="the upstream services' reliability", source="rows"),))


# ---------------------------------------------------------------------------------------------
# Artefacts the register began citing on 2026-09-26. The audit refused to run until each had a
# decision here, which is its design: an artefact the register leans on and this module has never
# read is exactly what it exists to catch.


def _nco_bakeoff(path: str) -> Entry:
    import math

    holdout = _load(path)["holdout"]
    results = holdout["results"]
    argus = results[holdout["argus_pick"]]
    origins = argus["origins"]
    middle = origins[len(origins) // 2]
    headlines = []
    for rival in holdout["decisive_rivals"]:
        theirs = dict(zip(results[rival]["origins"], results[rival]["oos_vol_bps"], strict=True))
        items = [
            Item(value=math.log(_f(theirs[o]) / _f(v)),
                 groups={"holdout_half": "first" if o < middle else "second"},
                 order=f"{o:08d}")
            for o, v in zip(origins, argus["oos_vol_bps"], strict=True) if o in theirs
        ]
        headlines.append(_headline(
            f"held-out volatility: ARGUS's pick vs {rival}", items, ("holdout_half",),
            headline="mean log(rival realised vol / ARGUS realised vol) over held-out windows",
            orientation="positive = ARGUS's allocation was calmer", role=VS_RIVAL,
            role_reason="the bake-off's decisive comparison, picked on the selection period",
            source="holdout.results.*.oos_vol_bps"))
    return Entry(path, CHECKED, "one row per held-out walk-forward window per allocator",
                 tuple(headlines))


def _financebench(path: str) -> Entry:
    answers = _load(path)["answers"]
    value = {"correct": 1.0, "wrong": -1.0, "disputed": 0.0}
    items = [Item(value=value[r["grade"]],
                  groups={"class": str(r["class"]), "formula": str(r.get("formula") or "none")},
                  order=str(r["id"]))
             for r in answers if r["status"] == "answered"]
    return Entry(path, CHECKED, "one row per FinanceBench question ARGUS answered", (
        _headline("filed-XBRL answers graded against FinanceBench's gold", items,
                  ("class", "formula"),
                  headline="+1 correct, 0 disputed, -1 wrong, over answered questions",
                  orientation="+1 correct, -1 wrong", role=RESULT,
                  role_reason="the engine's answers; abstentions are not scored here",
                  source="answers"),))


def _general_arb(path: str) -> Entry:
    blob = _load(path)
    real = blob["real_books"]["per_snapshot"]
    held = blob["out_of_sample"]["held_out"]["per_snapshot"]
    weekend = (blob["out_of_sample"].get("weekend") or {}).get("per_snapshot", [])
    held = [*held, *weekend]

    def right(row: dict[str, Any], arm: str) -> float:
        return 1.0 if bool(row[arm]) == bool(row["truth_monetizable"]) else 0.0

    def items(rows: Sequence[dict[str, Any]], arm: str, rival: str | None) -> list[Item]:
        return [Item(value=right(r, arm) - (right(r, rival) if rival else 1.0),
                     groups={"ticker": str(r["ticker"]), "phase": str(r["phase"])},
                     order=f"{int(r['round']):04d}") for r in rows]

    def halves(arm: str, rival: str | None) -> Any:
        a, b = items(real, arm, rival), items(held, arm, rival)
        return halves_from(sum(i.value for i in a) / len(a), sum(i.value for i in b) / len(b),
                           first_items=len(a), second_items=len(b),
                           label="design snapshots vs held-out and weekend snapshots")

    return Entry(path, CHECKED, "one row per two-sided book snapshot, design and held-out", (
        _headline("exact net walk vs NetworkX's Bellman-Ford cycle test", items(
            real + held, "argus_exact", "networkx_cycle"), ("ticker", "phase"),
            headline="(ARGUS right) - (rival right) against the monetisable truth",
            orientation="positive = ARGUS right where the rival was wrong", role=VS_RIVAL,
            role_reason="the general-purpose rival on the same books", source="per_snapshot",
            halves=halves("argus_exact", "networkx_cycle")),
        _headline("the deployed decomposition against the truth", items(
            real + held, "argus_deployed", None), ("ticker", "phase"),
            headline="(deployed path right) - 1 against the monetisable truth",
            orientation="0 = right, -1 = wrong", role=RESULT,
            role_reason="what the desk runs, which the artefact records false accepts for",
            source="per_snapshot", halves=halves("argus_deployed", None)),
    ))


def _eventdriven_agents(path: str) -> Entry:
    trades = _load(path)["walk_forward"]["test_trades"]
    headlines = []
    for rival in ("no_gate_follow", "ballast_pooled_t"):
        rows = trades[rival]
        items = [Item(value=-_f(t["net_bps"]),
                      groups={"symbol": str(t["symbol"]), "class": str(t["class"])},
                      order=str(t["at"])) for t in rows]
        headlines.append(_headline(
            f"ARGUS's significance gate (no trade taken) vs {rival}", items,
            ("symbol", "class"),
            headline="per rival trade: ARGUS's 0 bps minus the rival's net bps after costs",
            orientation="positive = the trade ARGUS declined lost money", role=VS_RIVAL,
            role_reason="the out-of-sample trades the gate declined and the rival took",
            source=f"walk_forward.test_trades.{rival}"))
    return Entry(path, CHECKED, "one row per out-of-sample trade the rivals took", tuple(headlines))


def _general_sue(path: str) -> Entry:
    rows = _load(path)["alignment"]["anchor_detail"]

    def answered(arm: dict[str, Any]) -> bool:
        return arm.get("sue") is not None

    items = [Item(value=(1.0 if answered(r["argus_dated_after"]) else 0.0)
                  - (1.0 if answered(r["pandas_period_lenient"]) else 0.0),
                  groups={"symbol": str(r["symbol"])}) for r in rows]
    return Entry(path, CHECKED, "one row per real anchor symbol", (
        _headline("SUE: dated alignment answers where pandas' lenient period index refuses",
                  items, ("symbol",),
                  headline="(ARGUS answers) - (pandas answers) per anchor",
                  orientation="+1 = only ARGUS aligned the filings, 0 = both did",
                  role=VS_RIVAL, role_reason="the general-purpose rival on the same filings",
                  source="alignment.anchor_detail"),))


def _copilot_stress(path: str) -> Entry:
    from argus.eval.copilot_stress import DOWN_DAY, SEVERE_DAY

    blob = _load(path)
    days = [d for d in blob["per_day"] if _f(d["shock"]) <= DOWN_DAY]
    headlines = []
    for rival, role in (("skfolio_vine", VS_RIVAL), ("skfolio_ep", VS_RIVAL)):
        both = [d for d in days if "argus_beta" in d["mae_pp"] and rival in d["mae_pp"]]
        items = [Item(value=_f(d["mae_pp"][rival]) - _f(d["mae_pp"]["argus_beta"]),
                      groups={"severity": "down_2pct" if _f(d["shock"]) <= SEVERE_DAY
                              else "down_1_to_2pct", "month": str(d["day"])[:7]},
                      order=str(d["day"])) for d in both]
        published = blob["down_1pct"]["comparisons"][f"argus_beta vs {rival}"]
        mean = -sum(i.value for i in items) / len(items)
        if len(items) != published["days"] or abs(mean - published["mean_difference_pp"]) > 1e-3:
            raise AuditError(f"{path}: per-day rows for {rival} do not reproduce the published "
                             f"comparison ({len(items)} days, {mean:.4f} pp)")
        headlines.append(_headline(
            f"stress-move error: ARGUS's open-hours beta vs {rival}", items,
            ("severity", "month"),
            headline="per down day: rival's mean absolute error minus ARGUS's, in pp",
            orientation="positive = ARGUS's stress sentence was closer", role=role,
            role_reason="the preregistered primary and its Entropy Pooling sibling",
            source="per_day"))
    return Entry(path, CHECKED, "one row per QQQUSDT down day, each a mean over 200 books",
                 tuple(headlines))


def _lui_comparison(path: str) -> Entry:
    from argus.eval.lui_comparison import sealed_rows

    published = _load(path)["sealed_comparison"]
    rows = sealed_rows()
    got = (sum(r["argus_ok"] for r in rows), sum(r["rasa_ok"] for r in rows),
           sum(r["argus_ok"] and not r["rasa_ok"] for r in rows),
           sum(r["rasa_ok"] and not r["argus_ok"] for r in rows))
    want = (published["argus_correct"], published["rasa_correct"],
            published["argus_only_correct"], published["rasa_only_correct"])
    if got != want:
        raise AuditError(f"{path}: the live cascade's sealed rows {got} do not reproduce the "
                         f"published counts {want}")
    items = [Item(value=float(r["argus_ok"]) - float(r["rasa_ok"]),
                  groups={"lang": str(r["lang"]), "intent": str(r["expect"])})
             for r in rows]
    return Entry(path, CHECKED, "one row per sealed question, regenerated from the deployed "
                 "cascade and Rasa's vendored predictions, parity-checked against the published "
                 "counts", (
        _headline("sealed intent routing: ARGUS's cascade vs Rasa's DIET", items,
                  ("lang", "intent"),
                  headline="(ARGUS right) - (Rasa right) per sealed question",
                  orientation="positive = ARGUS right where Rasa was not", role=VS_RIVAL,
                  role_reason="the preregistered sealed comparison", source="sealed_rows()"),))


def _lui_rematch(path: str) -> Entry:
    from argus.eval.lui_rematch_inputs import load

    blob = _load(path)
    rows = blob["paired_rows"]
    base = load()["confirmation_base"]
    if len(base) != len(rows["held_out"]):
        raise AuditError(f"{path}: {len(rows['held_out'])} held-out rows against "
                         f"{len(base)} inputs")
    # The rows are questions, not a time series: there is no chronological half to split on, so
    # no item carries an order.
    held = [Item(value=_f(v), groups={"lang": str(r["lang"]), "intent": str(r["expect"])})
            for v, r in zip(rows["held_out"], base, strict=True)]
    outside = [Item(value=_f(v), groups={"suite": suite})
               for suite in ("massive_oos", "clinc_oos", "hard_negatives") for v in rows[suite]]
    perturbed = [Item(value=_f(v), groups={"perturbation": suite.split(":", 1)[1]})
                 for suite in rows if suite.startswith("confirmation:") for v in rows[suite]]
    common = {"headline": "per row: (ARGUS right) - (Rasa right, mean over five seeds)",
              "orientation": "positive = ARGUS right where Rasa was not", "role": VS_RIVAL,
              "role_reason": "the rematch's paired comparison", "source": "paired_rows"}
    return Entry(path, CHECKED, "one paired row per question in every suite", (
        _headline("ARGUS's cascade vs Rasa's default pipeline, held out", held,
                  ("lang", "intent"), **common),
        _headline("ARGUS's cascade vs Rasa's default pipeline, out of scope", outside,
                  ("suite",), **common),
        _headline("ARGUS's cascade vs Rasa's default pipeline, perturbed held-out", perturbed,
                  ("perturbation",), **common),
    ))


def _pit_rivals(path: str) -> Entry:
    blob = _load(path)
    mine = {(r["accn"], r["side"]): 1.0 if r["right"] else 0.0 for r in blob["rows_argus"]}
    headlines = [_headline(
        "ARGUS's latest-quarter revenue against SEC's acceptance record",
        [Item(value=mine[(r["accn"], r["side"])] - 1.0,
              groups={"ticker": str(r["ticker"]), "side": str(r["side"])},
              order=str(r["as_of"])) for r in blob["rows_argus"]],
        ("ticker", "side"),
        headline="(ARGUS right) - 1 per question", orientation="0 = right, -1 = wrong",
        role=RESULT, role_reason="the gate's own answer on every question", source="rows_argus")]
    for rival, rows in blob["rows_rivals"].items():
        items = [Item(value=mine[(r["accn"], r["side"])]
                      - (1.0 if r["outcome"] == "right" else 0.0),
                      groups={"ticker": str(r["ticker"]), "side": str(r["side"])},
                      order=str(r["as_of"]))
                 for r in rows if (r["accn"], r["side"]) in mine]
        headlines.append(_headline(
            f"ARGUS vs {rival} on the same questions", items, ("ticker", "side"),
            headline="(ARGUS right) - (rival right) per question",
            orientation="positive = ARGUS right where the rival was not", role=VS_RIVAL,
            role_reason="the rival scored from its own output on the same questions",
            source=f"rows_rivals.{rival}"))
    return Entry(path, CHECKED, "one row per question: ten tickers' filings since 2017, one "
                 "hour before and after each acceptance", tuple(headlines))


# ---------------------------------------------------------------------------------------------
# Artefacts with no population to break down, with the reason read from each


def _not_checkable(status: str, reason: str) -> Callable[[str], Entry]:
    def build(path: str) -> Entry:
        return Entry(path, status, reason)
    return build


ADAPTERS: dict[str, Callable[[str], Entry]] = {
    # head-to-heads and results with rows
    "data/overnight_comparison.json": _overnight,
    "data/void_comparison.json": _void,
    "data/claimcheck_comparison.json": _claimcheck,
    "data/stopquality_comparison.json": _stopquality,
    "data/copilot_rivals.json": _copilot_rivals,
    "data/copilot_hedge.json": _copilot_hedge,
    "data/execution_arena.json": _execution_arena,
    "data/risk_layer_comparison.json": _risk_layer,
    "data/schedule_comparison.json": _schedule,
    "data/queue_proof.json": _queue_proof,
    "data/regime_comparison.json": _regime,
    "data/rotation_comparison.json": _rotation,
    "data/earnings_comparison.json": _earnings,
    "data/explainability_comparison.json": _explainability,
    "data/infoextract_comparison.json": _infoextract,
    "data/grammar_comparison.json": _grammar,
    "data/sentiment_comparison.json": _sentiment,
    "data/profile_divergence.json": _profile_divergence,
    "data/factor_divergence_comparison.json": _factor_divergence,
    "data/track1_study.json": _track1,
    "data/crosssection_study.json": _crosssection_study,
    "data/overfit_gates.json": _overfit_gates,
    "data/paper_ledger.jsonl": _paper_ledger,
    "data/afterhours_comparison.json": _afterhours,
    "data/execution_comparison.json": _execution_comparison,
    "data/review_oos.json": _review_oos,
    "data/skill_reliability.json": _desk_skill_rows,
    "data/cointegration.json": _cointegration_study,
    # cited from 2026-09-26
    "data/nco_bakeoff.json": _nco_bakeoff,
    "data/pit_rivals.json": _pit_rivals,
    "data/lui_comparison.json": _lui_comparison,
    "data/financebench_xbrl.json": _financebench,
    "data/general_arb_comparison.json": _general_arb,
    "data/eventdriven_agents.json": _eventdriven_agents,
    "data/general_sue_comparison.json": _general_sue,
    "data/eventdriven_rivals.json": _not_checkable(
        WITHOUT_ROWS, "seven null processes and power draws reduced to rejection rates; "
        "eval/eventdriven_rivals.py would need to record each draw's verdict per test"),
    "data/general_abstention_comparison.json": _not_checkable(
        WITHOUT_ROWS, "470 settled leans reduced to risk-coverage curves and AURC per scorer; "
        "eval/general_abstention_comparison.py would need to record each lean's score per arm"),
    "data/general_coint_comparison.json": _not_checkable(
        WITHOUT_ROWS, "160 simulated universes reduced to FDR, FWER and power per pipeline; "
        "eval/general_coint_comparison.py would need to record each universe's discoveries"),
    "data/review_rivals.json": _not_checkable(
        WITHOUT_ROWS, "40 seeds a cell reduced to false-admission and power rates per rule "
        "learner; eval/review_rivals.py would need to record each seed's outcome"),
    "data/xa_arena.json": _not_checkable(
        WITHOUT_ROWS, "hourly net asset values reduced to per-period scores per arm; "
        "eval/xa_arena.py would need to record each arm's hourly return path"),
    "data/lui_rematch.json": _lui_rematch,
    "data/general_grammar_comparison.json": _not_checkable(
        DESIGNED, "15 designed ill-typed expressions, 39 operator checks on one symbol's series "
        "and eight designed factors: properties of the grammar on chosen inputs"),
    "data/general_overfitgates_comparison.json": _not_checkable(
        WITHOUT_ROWS, "1,332 gate calls reduced to grade counts per arm and tier; "
        "eval/general_overfitgates_comparison.py would need to record each call's grade"),
    "data/general_rotation_comparison.json": _not_checkable(
        DESIGNED, "36 designed contract cases for the breadth-rotation inputs"),
    "data/lui_rematch_inputs.json": _not_checkable(
        NO_ROWS, "the rematch's frozen inputs: questions and labels, no result"),
    # outside the register
    "data/weekend_significance.json": _weekend,
    "data/shadow_record.json": _shadow,
    "data/pead_study.json": _pead,
    "data/overfitting_study.json": _overfitting_grid,
    "data/search_bakeoff.json": _search_bakeoff,
    "data/carry_study.json": _carry,
    "data/event_reactions.json": _event_reactions,
    "data/gap_study.json": _gap_study,
    "data/rule_proposals.json": _rule_proposals,
    "data/factor_split_half.json": _factor_split_half,
    "data/document_qa_eval.json": _document_qa,
    "data/perturbation_robustness.json": _perturbed,
    "data/vocab_stress.json": _perturbed,
    "data/feedbugged.json": _perturbed,
    "data/skill_matrix.json": _skill_matrix,
    # designed demonstrations: no population to break down
    "data/abstention_comparison.json": _not_checkable(
        DESIGNED, "three designed netting cases and a twelve-point avoided-loss/blocked-upside "
        "ratio sweep; no population of real abstentions is scored against the rival"),
    "data/deliberation_comparison.json": _not_checkable(
        DESIGNED, "a 17-point delay sweep and three thinking-budget tiers through two pure "
        "functions: a property of the formulas, not a measurement over decisions"),
    "data/dsr_comparison.json": _not_checkable(
        DESIGNED, "five designed cases and a 1,215-case parameter sweep of two implementations "
        "of one formula; only the agreement count is kept"),
    "data/rdagent_comparison.json": _not_checkable(
        DESIGNED, "an injection proof, a 400-trial pool on one symbol reduced to one deflated "
        "probability, and a five-point trial-count sweep"),
    "data/crosssection_comparison.json": _not_checkable(
        DESIGNED, "seven designed panels and 54 swept panels reduced to one all-agree flag"),
    "data/cointegration_comparison.json": _not_checkable(
        DESIGNED, "three designed ADF series, two self-inclusion cases and a synthetic "
        "190-pair noise panel reduced to counts"),
    "data/arbitrage_comparison.json": _not_checkable(
        DESIGNED, "three designed spreads and 60 constructed spreads reduced to counts; the "
        "real per-symbol basis lives in data/arbitrage_study.json, which no proof cites"),
    "data/recall_comparison.json": _not_checkable(
        DESIGNED, "three designed boundary cases for the point-in-time guard"),
    "data/journal_comparison.json": _not_checkable(
        DESIGNED, "a designed CRLF case, one tamper and one truncation, and a sweep of seven "
        "entry counts; the live ledger check is one boolean"),
    "data/protocol_commitments.jsonl": _not_checkable(
        DESIGNED, "hash commitments of the protocol: a record, not a measurement"),
    "data/mandate_comparison.json": _not_checkable(
        DESIGNED, "eleven designed boundary scenarios and a 19,440-scenario grid reduced to "
        "counts"),
    "data/feedlist_comparison.json": _not_checkable(
        DESIGNED, "designed vendor-failure scenarios over the rival's own category list"),
    "data/desk_notes.jsonl": _not_checkable(
        NO_ROWS, "free-text notes per live cycle; the feedlist harness reduces them to one "
        "distinct-source count and no per-cycle value is recorded"),
    "data/risk_proof.json": _not_checkable(
        DESIGNED, "an exhaustive sweep of 2,177,280 constructed risk states reduced to "
        "binding counts: a property of the gates, not a population measurement"),
    "data/shapematch_comparison.json": _not_checkable(
        DESIGNED, "three designed distance cases and query sweeps over one constructed series, "
        "kept as counts"),
    "data/review_comparison.json": _not_checkable(
        DESIGNED, "seven designed lifecycle fixtures and four threshold ablations"),
    "data/workbench_comparison.json": _not_checkable(
        DESIGNED, "a source-breadth count and one point-in-time boundary on one fact"),
    "data/execassist_comparison.json": _not_checkable(
        DESIGNED, "three thinking budgets through one cost function"),
    "data/eventdriven_comparison.json": _not_checkable(
        WITHOUT_ROWS, "the false-positive sweep draws 40 seeds of 150 events, but only the "
        "rates are kept (eval/eventdriven_comparison.py would need to record per-seed "
        "verdicts)"),
    "data/sentiment_tweeteval.json": _not_checkable(
        WITHOUT_ROWS, "12,284 TweetEval test tweets scored, but only per-class recall, confusion "
        "counts and paired-bootstrap differences are kept: TweetEval publishes no licence, so no "
        "per-tweet row is written (eval/sentiment_tweeteval.py would need to keep per-tweet "
        "labels by id to allow a breakdown by class and by topic)"),
    "data/analogstress_comparison.json": _not_checkable(
        WITHOUT_ROWS, "2,698 test queries scored, but only per-predictor aggregates and "
        "date-clustered Diebold-Mariano tests are kept (eval/analogstress_comparison.py would "
        "need to record per-query Winkler scores with name and date)"),
    "data/copilot_stress.json": _copilot_stress,
    "data/allocation_comparison.json": _not_checkable(
        WITHOUT_ROWS, "nine live-fetched walk-forward windows reduced to per-allocator means and "
        "windows-beating counts; the harness records per-window volatilities from 2026-09-26 "
        "(oos_vol_bps_per_window), but this artefact predates that and its candles were fetched "
        "live, so it cannot be regenerated. The capability's OOS and significance proofs cite "
        "data/nco_bakeoff.json, which has the rows"),
    "data/review_report.json": _not_checkable(
        WITHOUT_ROWS, "each rule's precision is kept as an aggregate; which decisions it fired "
        "on is not"),
    "data/h2h_optic/summary.json": _not_checkable(
        NO_ROWS, "three theses through two desks, compared in prose; nothing numeric per "
        "thesis is scored"),
    "data/h2h_baserate/summary.json": _not_checkable(
        NO_ROWS, "one sentence through two desks"),
    "data/retrieval_diversity.json": _not_checkable(
        WITHOUT_ROWS, "the same 2,698-query grid as analogstress, kept as per-arm aggregates"),
    "data/bakeoff.json": _not_checkable(
        DESIGNED, "two models, three replays each, of one frame on one symbol"),
    "data/policy_forecasts.json": _not_checkable(
        WITHOUT_ROWS, "49,142 forecasts reduced to per-symbol counts and reliability bins; "
        "per-symbol hit rates are not kept"),
    "data/forecastbench_restatement.json": _not_checkable(
        WITHOUT_ROWS, "the same forecasts as reliability bins only"),
    "data/refusal_alpha.json": _not_checkable(
        WITHOUT_ROWS, "per-horizon aggregates bootstrapped over decision cycles; per-mark rows "
        "are not kept"),
    "data/incremental_value.json": _not_checkable(
        WITHOUT_ROWS, "42 instants reduced to per-baseline win/loss counts"),
    "data/pause_drill.json": _not_checkable(
        DESIGNED, "nine designed kill-and-resume scenarios with a scripted model"),
    "data/guard_selfcheck.json": _not_checkable(
        DESIGNED, "an exhaustive sweep of 190,944 constructed orders reduced to an identity "
        "count"),
    "data/mcp_fuzz.json": _not_checkable(
        DESIGNED, "an authored 145-case fuzz corpus in five families"),
    "data/mcp_sdk_comparison.json": _not_checkable(
        DESIGNED, "the same authored fuzz corpus through two servers"),
}
"""Every artefact the audit knows how to read, and how. A register artefact missing from this map
is an :class:`AuditError`: a new capability row must come with a decision about its evidence."""

EXTRA = (
    "data/bakeoff.json", "data/carry_study.json", "data/document_qa_eval.json",
    "data/event_reactions.json", "data/factor_split_half.json", "data/feedbugged.json",
    "data/forecastbench_restatement.json", "data/gap_study.json", "data/guard_selfcheck.json",
    "data/incremental_value.json", "data/mcp_fuzz.json", "data/mcp_sdk_comparison.json",
    "data/overfitting_study.json", "data/pause_drill.json", "data/pead_study.json",
    "data/perturbation_robustness.json", "data/policy_forecasts.json",
    "data/refusal_alpha.json", "data/retrieval_diversity.json", "data/rule_proposals.json",
    "data/search_bakeoff.json", "data/shadow_record.json", "data/skill_matrix.json",
    "data/vocab_stress.json", "data/weekend_significance.json",
)
"""Result artefacts audited whether or not a register row cites them: the families the recorded
failures came from (single-symbol bake-offs, split studies, grids), and the artefacts written on
2026-09-25 that the register now cites."""


def register_artefacts() -> tuple[str, ...]:
    """Every data artefact the register names, and every statistical/OOS proof's scope."""
    from argus.eval.standing import REGISTER, capability_artefacts, proof_scope

    wanted: set[str] = set()
    for cap in REGISTER:
        wanted.update(capability_artefacts(cap))
        for proof in cap.proofs:
            if proof.condition in ("statistically_valid_evaluation", "out_of_sample_test"):
                wanted.update(proof_scope(cap, proof))
    return tuple(sorted(wanted))


def run(paths: Iterable[str] | None = None) -> dict[str, Any]:
    """Audit every register artefact (and :data:`EXTRA`), returning the report."""
    register = register_artefacts()
    targets = sorted(set(paths) if paths is not None else set(register) | set(EXTRA))
    unhandled = [p for p in targets if p not in ADAPTERS]
    if unhandled:
        raise AuditError(f"no groupwise decision for: {', '.join(unhandled)}")
    entries = {path: ADAPTERS[path](path) for path in targets}
    flagged: list[dict[str, Any]] = []
    for path, entry in entries.items():
        for h in entry.headlines:
            if h.report.flagged:
                flagged.append({"artefact": path, "headline": h.report.name, "role": h.role,
                                "flags": list(h.report.flags),
                                "readings": [t.reading for t in h.report.tables
                                             if t.carried or t.single_group
                                             or t.macro_disagrees]
                                + ([h.report.split.reading] if h.report.split is not None
                                   and h.report.split.flagged else [])})
    counts = {status: sum(1 for e in entries.values() if e.status == status)
              for status in STATUSES}
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "Mind2Web metric.py:236-259 (MIT): per-group mean and count, macro average, "
                  "0/1/2/3/>3 histogram; read by argus.eval.groupwise into four flags "
                  "(carried_by_one_group, single_group, macro_disagrees, flips_across_halves)",
        "register_artefacts": list(register),
        "counts": counts,
        "flagged": flagged,
        "flagged_gating": [f for f in flagged if f["role"] in GATING_ROLES],
        "artefacts": {path: entry.as_dict() for path, entry in entries.items()},
        "qwen_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="groupwise audit of the register's artefacts")
    parser.add_argument("--only", action="append", default=None)
    args = parser.parse_args(argv)
    report = run(args.only)
    artefact.write(REPORT_PATH, report)
    for item in report["flagged"]:
        print(f"FLAG {item['role']:15s} {item['artefact']} :: {item['headline']} "
              f"{item['flags']}")
    print(f"counts {report['counts']}; flagged {len(report['flagged'])} "
          f"({len(report['flagged_gating'])} on a capability's own claim)")
    print(f"written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ADAPTERS",
    "CHECKED",
    "CONTEXT",
    "DESIGNED",
    "GATING_ROLES",
    "NO_ROWS",
    "REPORT_PATH",
    "RESULT",
    "STATUSES",
    "VS_RIVAL",
    "WITHOUT_ROWS",
    "AuditError",
    "Entry",
    "Headline",
    "register_artefacts",
    "run",
]
