"""Thesis quality — four mechanisms from the 40-repo study, each measured on a desk replay.

`research/mypr-teardowns/_SYNTHESIS.md` names four STRONG items that all act on the quality of the
reasoning behind a decision rather than on the decision rule itself:

* **S5 — candidates ranked before commit** (`agents/meta_pm.py`, ``candidates``): draw a second,
  independently derived answer and commit the one a code-side checker ranks higher.
* **S6 — checker-triggered bounded repair** (`agents/meta_pm.py`, ``repair_rounds``): a thesis
  with a named defect is sent back once or twice with the defect stated; the best round is kept.
* **S7 — a debate stall detector** (`agents/debate.py`, ``stall_detection``): stop paying for
  rounds in which neither side moved, changed direction or cited anything new.
* **S4 — the unused-evidence stress question** (`desk/stress.py`, :func:`unused_evidence_stress`):
  of the evidence the thesis did not use, surface the item most worth answering.

None of them is allowed to become a default on the strength of its mechanism. This module runs
each against the alternative it replaces, on the same frames, and writes what it found to
``data/thesis_quality.json`` — losses and ties included, because the defaults in `meta_pm.py` and
`debate.py` are set from this file and a flattering reading would switch on something that does
not pay.

**The frames.** ``build_frames`` reconstructs desk decision instants the way `paper/replay.py`
does — Bitget hourly candles strictly up to the instant, and the sources that can honestly be
rebuilt as of it (SEC filings by acceptance time, the dated Treasury curve, the VIX close, FINRA
short volume for the session before) — plus one line of point-in-time technicals computed here
from the same candles, because a frame with only a price line biases the desk toward abstention by
construction (`paper/replay.py`'s own finding). Each frame carries the realised 24-hour move that
followed it, read from candles after the instant. The frames are written to
``data/thesis_quality_frames.json`` *before* any model sees them, and every stage reads that file:
the replay is over recorded frames, not over whatever the network returned on the day of a rerun.

**The model calls, and the cap.** Every Qwen response is recorded in
``data/thesis_quality_recording.json``, keyed by the hash of the exact request. A rerun replays
from the recording at zero cost, and :class:`CappedRecordingClient` refuses any call beyond
:data:`QWEN_CAP` in total and beyond each stage's share (:data:`STAGE_CAPS`), counting every HTTP
attempt — a retried request is two calls, not one. A stage that reaches its cap stops and says how
far it got; it does not borrow from the next stage.

**What "better" means per stage**, fixed before the run:

* S5: grounding pass rate and defect-free rate of the committed answer, P&L net of the round trip
  *and* of the extra call's deliberation cost (`meta_pm.deliberation_cost_bps`), lean hit rate,
  and how often ranking changed the committed answer at all.
* S6: share of defective single-shot answers repaired to zero defects **without acquiring a
  defect they did not have**, after one round and after two, against no revision (where the
  defect stays by definition).
* S7: rounds saved against detections that were wrong — a stop is wrong when a later round in the
  full-length debate changed a direction, moved the gap by more than `CONVERGENCE_BPS`, or
  converged — and agreement with an LLM judge asked MAF's own two progress questions. Also the
  recorded live debate log, read for how many debates could have been stopped at all.
* S4: how often the surfaced item is judged relevant to the decision, against a uniformly random
  unused item, by a judge that sees both blind and in shuffled order.

    python -m argus.eval.thesis_quality --build-frames      # network, no model calls
    python -m argus.eval.thesis_quality --run               # Qwen, capped at QWEN_CAP
    python -m argus.eval.thesis_quality                     # recompute from the recording only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from argus.agents import debate as debate_mod
from argus.agents.meta_pm import (
    Deliberation,
    MarketFrame,
    MetaPM,
    ThesisCheck,
    deliberation_cost_bps,
)
from argus.decision.verdicts import Verdict
from argus.desk.stress import (
    cited_evidence,
    default_encoder,
    lexical_encoder,
    split_evidence_line,
    unused_evidence,
)
from argus.eval.artefact import write as write_artefact
from argus.llm.qwen import Completion, QwenClient, QwenError, Thinking, TokenBudget, Usage
from argus.truth.clocks import DualClock
from argus.truth.evidence import Evidence

DATA = Path(__file__).resolve().parents[3] / "data"
FRAMES_PATH = DATA / "thesis_quality_frames.json"
RECORDING_PATH = DATA / "thesis_quality_recording.json"
REPORT_PATH = DATA / "thesis_quality.json"
DESK_NOTES_PATH = DATA / "desk_notes.jsonl"

QWEN_CAP = 100
"""Total Qwen calls this evaluation may make, set by the task that commissioned it."""

STAGE_CAPS: dict[str, int] = {"s5": 30, "s6": 16, "s4": 30, "s7": 24}
"""Each stage's share of :data:`QWEN_CAP`, in the order the stages run. S5 first because S6 and S4
reuse its answers at no cost; S7 last and largest-per-sample, so a short run still measures the
three stages that decide defaults on the live decision path."""

SYMBOLS = (
    "NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "METAUSDT", "COINUSDT", "MSTRUSDT", "AMZNUSDT",
    "MSFTUSDT",
)
PM_THINKING = Thinking.LOW
"""The budget `paper/replay.py` and the routine paper cycle decide at."""

PM_MAX_TOKENS = 900
"""What `agents/desk.py` gives the Meta-PM."""

ANNUALISED_VOL = Decimal("0.45")
"""`TradingDesk`'s default, so the deliberation charge matches the desk's own."""

JUDGE_PROMPT = """You audit a trading decision. Below are pieces of evidence the decision was given
but did NOT use. For EACH item, rate how much it bears on whether this decision is right:
0 = irrelevant to this decision, 1 = somewhat relevant, 2 = directly relevant and should have
been addressed. Judge each item on its own; do not rank them against each other.

Reply with a single json object and nothing else:
{"ratings": {"<label>": 0 | 1 | 2, ...}, "reasons": {"<label>": "<one short sentence>", ...}}"""

PROGRESS_PROMPT = """You supervise a two-sided debate (a bull seat and a bear seat) about one
instrument. You are shown the transcript up to and including the latest round. Answer two
questions about the LATEST round compared with the rounds before it:
- is_in_loop: are the sides repeating the same arguments rather than adding anything?
- is_progress_being_made: did the latest round add a new argument, new evidence, or a real change
  of position?

Reply with a single json object and nothing else:
{"is_in_loop": {"reason": "<short>", "answer": true | false},
 "is_progress_being_made": {"reason": "<short>", "answer": true | false}}"""
"""Two of the five fields of MAF's progress ledger (`_magentic.py:215-245`), the two its stall
counter reads (`:1119`). Used here only as the reference the non-LLM proxy is compared with."""


# --- the frames ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedFrame:
    """One reconstructed decision instant, frozen to disk before any model saw it."""

    symbol: str
    at: datetime
    price: Decimal
    evidence_lines: tuple[str, ...]
    absent: tuple[str, ...]
    realised_bps: float
    """The move from the instant to 24 hours later, from candles that postdate the instant."""

    def market_frame(self) -> MarketFrame:
        """The frame the desk's Meta-PM would have been given, less what a replay cannot rebuild.

        No hedge menu (the replay has no broker state), no memory (the replay is not a ledger),
        no debate block (S7 measures the debate separately) and no mandate — each rendered by the
        frame itself as absent, never as empty evidence.
        """
        session = DualClock().state(self.at, nav_age_seconds=5.0)
        return MarketFrame(
            symbol=self.symbol,
            as_of=self.at,
            session=session,
            token_price=self.price,
            position_quantity=Decimal("0"),
            round_trip_bps=Decimal("12"),
            evidence=self.evidence_lines,
            deliberation_bps=deliberation_cost_bps(
                session, thinking=PM_THINKING, annualised_vol=ANNUALISED_VOL
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "at": self.at.isoformat(), "price": str(self.price),
            "evidence_lines": list(self.evidence_lines), "absent": list(self.absent),
            "realised_bps": round(self.realised_bps, 4),
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> RecordedFrame:
        return cls(
            symbol=str(blob["symbol"]), at=datetime.fromisoformat(str(blob["at"])),
            price=Decimal(str(blob["price"])),
            evidence_lines=tuple(str(x) for x in blob["evidence_lines"]),
            absent=tuple(str(x) for x in blob["absent"]),
            realised_bps=float(blob["realised_bps"]),
        )


def technical_evidence(symbol: str, at: datetime, closes: Sequence[tuple[datetime, Decimal]]
                       ) -> Evidence | None:
    """Point-in-time technicals from hourly closes at or before ``at``. ``None`` below 50 bars.

    Every figure is computed here and stated with its window, so the model interprets numbers it
    was given rather than producing them: the 24-hour and 5-day change, the distance from the
    20- and 50-hour means, a 14-period RSI (Wilder's form, simple-average seed), and realised
    volatility of hourly returns annualised over 8,760 hours.
    """
    usable = [float(c) for ts, c in closes if ts <= at]
    if len(usable) < 50:
        return None
    last = usable[-1]
    change_24h = (last / usable[-25] - 1.0) * 100 if len(usable) > 24 else 0.0
    change_5d = (last / usable[-121] - 1.0) * 100 if len(usable) > 120 else None
    ma20 = sum(usable[-20:]) / 20
    ma50 = sum(usable[-50:]) / 50
    gains, losses = [], []
    for a, b in zip(usable[-15:-1], usable[-14:], strict=True):
        gains.append(max(0.0, b - a))
        losses.append(max(0.0, a - b))
    avg_gain, avg_loss = sum(gains) / 14, sum(losses) / 14
    rsi = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    rets = [math.log(b / a) for a, b in zip(usable[-49:-1], usable[-48:], strict=True) if a > 0]
    mean = sum(rets) / len(rets)
    vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) * math.sqrt(8760) * 100
    parts = [
        f"24h change {change_24h:+.2f}%",
        f"5-day change {change_5d:+.2f}%" if change_5d is not None else "5-day change unavailable",
        f"price {(last / ma20 - 1) * 100:+.2f}% vs 20-hour mean",
        f"{(last / ma50 - 1) * 100:+.2f}% vs 50-hour mean",
        f"RSI(14, hourly) {rsi:.1f}",
        f"realised volatility (48 hourly returns, annualised) {vol:.1f}%",
    ]
    return Evidence(
        id=f"ta-{symbol}-{int(at.timestamp())}",
        claim=(
            f"{symbol} technicals computed from hourly closes to this instant: "
            + "; ".join(parts)
        ),
        source="technical",
        available_at=at,
        credibility=1.0,
    )


def build_frames(
    *, symbols: Sequence[str] = SYMBOLS, days: int = 21, per_symbol: int = 2,
    every_hours: int = 24,
) -> list[RecordedFrame]:
    """Reconstruct decision instants from the network. Makes no model call."""
    from argus.backtest.engine import Bar
    from argus.market.history import CandleType, fetch_range
    from argus.paper.replay import frame_at, instants, realised_bps, reconstruct_sources

    out: list[RecordedFrame] = []
    for position, symbol in enumerate(symbols):
        try:
            candles = fetch_range(symbol, days=days, interval="1H",
                                  candle_type=CandleType.MARKET)
        except Exception as exc:
            print(f"{symbol}: candles unavailable ({exc})", file=sys.stderr)
            continue
        bars = [Bar(ts=c.ts, close=c.close, extra={}) for c in candles]
        closes = [(c.ts, c.close) for c in candles]
        picked = 0
        # Staggered by symbol so the frames do not all share two calendar days: sixteen
        # decisions on the same two dates are two market days observed eight times each, and a
        # P&L comparison over them would count one day's direction as sixteen outcomes.
        ordered = list(reversed(instants(bars, every_hours=every_hours)))
        stride = 5
        staggered = ordered[position % stride::stride] + ordered
        for at in dict.fromkeys(staggered):
            if picked >= per_symbol:
                break
            move = realised_bps(bars, at)
            if move is None:
                continue
            rebuilt, absent = reconstruct_sources(symbol, at)
            tech = technical_evidence(symbol, at, closes)
            macro = (*rebuilt, tech) if tech is not None else rebuilt
            frame = frame_at(symbol, at, bars, macro=macro)
            if frame is None:
                continue
            out.append(RecordedFrame(
                symbol=symbol, at=at, price=frame.price,
                evidence_lines=tuple(e.render() for e in frame.evidence),
                absent=tuple(sorted(set(absent) | set(frame.absent))),
                realised_bps=move,
            ))
            picked += 1
    return out


def save_frames(frames: Sequence[RecordedFrame], path: Path = FRAMES_PATH) -> None:
    write_artefact(path, {
        "generated_at": datetime.now(UTC).isoformat(),
        "note": "Recorded before any model call; every stage of eval/thesis_quality.py reads this.",
        "frames": [f.as_dict() for f in frames],
    })


def load_frames(path: Path = FRAMES_PATH) -> list[RecordedFrame]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    return [RecordedFrame.from_dict(f) for f in blob["frames"]]


# --- the capped, recording client ---------------------------------------------------------------


def _completion_to(c: Completion) -> dict[str, Any]:
    return {
        "content": c.content, "reasoning": c.reasoning, "finish_reason": c.finish_reason,
        "usage": {
            "prompt_tokens": c.usage.prompt_tokens, "completion_tokens": c.usage.completion_tokens,
            "reasoning_tokens": c.usage.reasoning_tokens, "total_tokens": c.usage.total_tokens,
            "cached_tokens": c.usage.cached_tokens, "reported": c.usage.reported,
        },
    }


def _completion_from(blob: dict[str, Any]) -> Completion:
    u = blob["usage"]
    return Completion(
        content=str(blob["content"]), reasoning=str(blob["reasoning"]),
        finish_reason=str(blob["finish_reason"]),
        usage=Usage(
            prompt_tokens=int(u["prompt_tokens"]), completion_tokens=int(u["completion_tokens"]),
            reasoning_tokens=int(u["reasoning_tokens"]), total_tokens=int(u["total_tokens"]),
            cached_tokens=int(u.get("cached_tokens", 0)), reported=bool(u.get("reported", True)),
        ),
    )


class CappedRecordingClient(QwenClient):
    """The real client, with a hard call cap per stage and every response recorded.

    ``_post`` is the one place an HTTP request is made, so the cap is enforced there and nowhere
    else: `complete_json`'s own validation retries pass through it and are counted. ``max_retries``
    is forced to 1 so one ``_post`` is exactly one HTTP attempt — the parent's transport retries
    would otherwise spend calls the counter never saw.

    ``live=False`` answers only from the recording and raises on a miss, which is how the offline
    rerun and the tests use it.
    """

    def __init__(
        self, recording: dict[str, dict[str, Any]], *, live: bool, cap: int = QWEN_CAP,
        stage_caps: dict[str, int] | None = None,
    ) -> None:
        if live:
            super().__init__(max_retries=1, budget=TokenBudget(limit=3_000_000))
        else:
            super().__init__(api_key="offline-replay", base_url="https://offline.invalid",
                             max_retries=1)
        self.recording = recording
        self.live = live
        self.cap = cap
        self.stage_caps = dict(stage_caps or STAGE_CAPS)
        self.stage = "s5"
        self.posts: dict[str, int] = {k: 0 for k in self.stage_caps}
        self.replayed = 0
        self.save_to: Path | None = None

    @property
    def total_posts(self) -> int:
        return sum(self.posts.values())

    def remaining(self, stage: str | None = None) -> int:
        stage = stage or self.stage
        return max(0, min(self.cap - self.total_posts,
                          self.stage_caps.get(stage, 0) - self.posts.get(stage, 0)))

    def _post(self, payload: dict[str, Any]) -> Completion:
        key = self._cache_key(payload)
        if key in self.recording:
            self.replayed += 1
            return _completion_from(self.recording[key])
        if not self.live:
            raise QwenError("offline replay: this request is not in the recording")
        if self.remaining() <= 0:
            raise QwenError(f"stage {self.stage} reached its call cap")
        self.posts[self.stage] = self.posts.get(self.stage, 0) + 1
        result = super()._post(payload)
        self.recording[key] = _completion_to(result)
        if self.save_to is not None:
            # written through on every paid response: the first live run was killed before its
            # end-of-run save, and the calls it had paid for were lost unrecorded
            save_recording(self.recording, self.save_to)
        return result


def load_recording(path: Path = RECORDING_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    return dict(blob.get("responses", {}))


def save_recording(recording: dict[str, dict[str, Any]], path: Path = RECORDING_PATH) -> None:
    write_artefact(path, {
        "note": "Every Qwen response eval/thesis_quality.py received, keyed by the sha256 of the "
                "exact request payload. A rerun replays from here at zero cost.",
        "responses": dict(sorted(recording.items())),
    })


# --- scoring helpers ----------------------------------------------------------------------------


def _net_bps(response: dict[str, Any] | None, realised: float, *, extra_cost: float) -> float:
    """Unit-notional P&L of an answer: zero unless it opened exposure. Round trip 12bps."""
    if response is None:
        return 0.0
    try:
        verdict = Verdict(str(response.get("verdict", "")).strip().lower())
        quantity = float(response.get("quantity", 0) or 0)
    except (ValueError, TypeError):
        return 0.0
    if not verdict.opens_exposure or quantity <= 0 or not response.get("invalidation"):
        return 0.0
    direction = -1.0 if str(response.get("side", "buy")).strip().lower() == "sell" else 1.0
    return direction * realised - 12.0 - extra_cost


def _lean_hit(response: dict[str, Any] | None, realised: float) -> bool | None:
    if response is None:
        return None
    lean = str(response.get("lean", "none")).strip().lower()
    if lean not in ("up", "down") or realised == 0:
        return None
    return (lean == "up") == (realised > 0)


def _arm_summary(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    checks = [r[f"{key}_check"] for r in rows if r.get(f"{key}_check") is not None]
    hits = [r[f"{key}_lean_hit"] for r in rows if r.get(f"{key}_lean_hit") is not None]
    nets = [float(r[f"{key}_net_bps"]) for r in rows]
    opened = sum(1 for r in rows if r[f"{key}_net_bps"] != 0.0)
    return {
        "decisions": len(rows),
        "defect_free": sum(1 for c in checks if not c["defects"]),
        "grounding_pass": sum(1 for c in checks if "ungrounded_figure" not in c["defects"]),
        "mean_score": round(sum(c["score"] for c in checks) / len(checks), 4) if checks else None,
        "opened": opened,
        "net_bps_total": round(sum(nets), 3),
        "lean_calls": len(hits),
        "lean_hits": sum(1 for h in hits if h),
        "verdicts": sorted({str(r[f"{key}_verdict"]) for r in rows}),
    }


# --- S5 and S6 ----------------------------------------------------------------------------------


def run_s5(frames: Sequence[RecordedFrame], client: CappedRecordingClient
           ) -> tuple[dict[str, Any], list[tuple[RecordedFrame, Deliberation]]]:
    """Single-shot vs rank-before-commit on the same frames. The first candidate IS single-shot."""
    client.stage = "s5"
    pm = MetaPM(client, max_tokens=PM_MAX_TOKENS, thinking=PM_THINKING,
                candidates=2, repair_rounds=0)
    rows: list[dict[str, Any]] = []
    kept: list[tuple[RecordedFrame, Deliberation]] = []
    stopped = ""
    for rf in frames:
        frame = rf.market_frame()
        if client.live and client.remaining() < 2 and not _recorded(client, pm, frame):
            stopped = f"stopped before {rf.symbol} {rf.at.isoformat()}: stage cap reached"
            break
        try:
            d = pm.deliberate(frame)
        except QwenError as exc:
            stopped = f"stopped at {rf.symbol} {rf.at.isoformat()}: {str(exc)[:120]}"
            break
        kept.append((rf, d))
        per_call = float(frame.deliberation_bps)
        single = d.attempts[0]
        committed = d.committed_attempt
        extra = per_call * (d.calls - 1)
        rows.append({
            "symbol": rf.symbol, "at": rf.at.isoformat(), "realised_bps": rf.realised_bps,
            "evidence_items": len(rf.evidence_lines),
            "single_verdict": (single.response or {}).get("verdict"),
            "ranked_verdict": (committed.response or {}).get("verdict"),
            "ranked_chose": committed.index,
            "second_available": len(d.attempts) > 1 and d.attempts[1].response is not None,
            "candidates_disagree_on_verdict": (
                len(d.attempts) > 1 and d.attempts[1].response is not None
                and str((d.attempts[1].response or {}).get("verdict", "")).lower()
                != str((single.response or {}).get("verdict", "")).lower()
            ),
            "single_check": single.check.as_dict() if single.check else None,
            "ranked_check": committed.check.as_dict() if committed.check else None,
            "single_net_bps": round(_net_bps(single.response, rf.realised_bps, extra_cost=0.0), 3),
            "ranked_net_bps": round(
                _net_bps(committed.response, rf.realised_bps, extra_cost=extra), 3
            ),
            "extra_deliberation_bps": round(extra, 4),
            "single_lean_hit": _lean_hit(single.response, rf.realised_bps),
            "ranked_lean_hit": _lean_hit(committed.response, rf.realised_bps),
            "calls": d.calls,
        })
    changed = sum(1 for r in rows if r["ranked_chose"] != 0)
    return {
        "question": "Does drawing a second, independently derived answer and committing the one "
                    "the code-side checker ranks higher beat committing the first answer?",
        "arms": {"single_shot": _arm_summary(rows, "single"),
                 "rank_before_commit": _arm_summary(rows, "ranked")},
        "ranking_changed_the_committed_answer": changed,
        "candidates_disagreed_on_verdict": sum(
            1 for r in rows if r["candidates_disagree_on_verdict"]
        ),
        "extra_deliberation_bps_per_decision": (
            round(sum(r["extra_deliberation_bps"] for r in rows) / len(rows), 4) if rows else None
        ),
        "rows": rows,
        "stopped": stopped,
    }, kept


def _recorded(client: CappedRecordingClient, pm: MetaPM, frame: MarketFrame) -> bool:
    """Whether this frame's first call is already in the recording (a rerun costs nothing)."""
    from argus.agents.meta_pm import SYSTEM_PROMPT

    payload: dict[str, Any] = {
        "model": client._model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": frame.to_prompt_block()}],
        "temperature": 0.0, "max_tokens": PM_MAX_TOKENS,
    }
    client._apply_thinking(payload, PM_THINKING)
    payload["response_format"] = {"type": "json_object"}
    return client._cache_key(payload) in client.recording


def run_s6(kept: Sequence[tuple[RecordedFrame, Deliberation]], client: CappedRecordingClient
           ) -> dict[str, Any]:
    """Repair the defective single-shot answers, one round then a second, against no repair."""
    client.stage = "s6"
    pm = MetaPM(client, max_tokens=PM_MAX_TOKENS, thinking=PM_THINKING,
                candidates=1, repair_rounds=2)
    rows: list[dict[str, Any]] = []
    stopped = ""
    defective = [(rf, d) for rf, d in kept
                 if d.attempts[0].check is not None and d.attempts[0].check.defects]
    for rf, _d in defective:
        if client.live and client.remaining() < 1:
            stopped = f"stopped before {rf.symbol} {rf.at.isoformat()}: stage cap reached"
            break
        try:
            d = pm.deliberate(rf.market_frame())
        except QwenError as exc:
            stopped = f"stopped at {rf.symbol}: {str(exc)[:120]}"
            break
        first = d.attempts[0].check
        assert first is not None
        repairs = [a for a in d.attempts if a.stage == "repair"]

        def summary(check: ThesisCheck | None, before: ThesisCheck = first) -> dict[str, Any]:
            if check is None:
                return {"available": False}
            new = sorted(set(check.defects) - set(before.defects))
            return {"available": True, "defects": list(check.defects), "new_defects": new,
                    "repaired_clean": not check.defects}

        after_one = repairs[0].check if repairs else None
        best_after_one = min(
            (c for c in (first, after_one) if c is not None),
            key=lambda c: (len(c.defects), -c.score),
        )
        committed = d.committed_attempt.check
        rows.append({
            "symbol": rf.symbol, "at": rf.at.isoformat(),
            "defects_before": list(first.defects),
            "unresolved_before": list(first.unresolved_figures),
            "round_1": summary(after_one),
            "round_2": summary(repairs[1].check if len(repairs) > 1 else None),
            "committed_after_1": summary(best_after_one),
            "committed_after_2": summary(committed),
            "repair_calls": d.calls - 1,
            "verdict_before": (d.attempts[0].response or {}).get("verdict"),
            "verdict_committed": (d.committed_attempt.response or {}).get("verdict"),
        })

    def rate(key: str) -> dict[str, Any]:
        done = [r for r in rows if r[key].get("available")]
        clean = [r for r in done if r[key]["repaired_clean"] and not r[key]["new_defects"]]
        new = [r for r in done if r[key]["new_defects"]]
        return {"n": len(done), "repaired_without_new_defects": len(clean),
                "acquired_a_new_defect": len(new)}

    return {
        "question": "Of single-shot answers carrying a named defect, how many does one (or two) "
                    "checker-triggered repair calls fix without introducing a defect they did "
                    "not have? No revision leaves every one of them defective.",
        "defective_single_shot": len(defective),
        "attempted": len(rows),
        "after_one_round": rate("committed_after_1"),
        "after_two_rounds": rate("committed_after_2"),
        "no_revision": {"n": len(rows), "repaired_without_new_defects": 0},
        "verdict_changed_by_repair": sum(
            1 for r in rows
            if str(r["verdict_before"]).lower() != str(r["verdict_committed"]).lower()
        ),
        "repair_calls": sum(r["repair_calls"] for r in rows),
        "rows": rows,
        "stopped": stopped,
    }


# --- S4 -----------------------------------------------------------------------------------------


def run_s4(kept: Sequence[tuple[RecordedFrame, Deliberation]], client: CappedRecordingClient,
           *, seed: int = 20260925) -> dict[str, Any]:
    """Surfaced unused item vs a random unused item, rated blind by a judge."""
    client.stage = "s4"
    encoder, encoder_name = default_encoder()
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    stopped = ""
    theses: list[tuple[RecordedFrame, dict[str, Any]]] = [
        (rf, a.response) for rf, d in kept for a in d.attempts
        if a.stage == "candidate" and a.response is not None
    ]
    for rf, response in theses:
        thesis = str(response.get("thesis", ""))
        evidence = [split_evidence_line(x, index=i) for i, x in enumerate(rf.evidence_lines)]
        cited = cited_evidence(thesis, evidence)
        queries = [str(q) for q in [*(response.get("invalidation") or []),
                                    str(response.get("counter_case", ""))] if str(q).strip()]
        ranked = unused_evidence(claim=thesis, queries=queries, evidence=evidence, cited=cited,
                                 encoder=encoder)
        ranked_lex = unused_evidence(claim=thesis, queries=queries, evidence=evidence,
                                     cited=cited, encoder=lexical_encoder)
        if len(ranked) < 2 or not ranked[0].gated_in:
            rows.append({"symbol": rf.symbol, "at": rf.at.isoformat(), "skipped":
                         "fewer than two unused items, or none passed the relevance gate",
                         "unused": len(ranked), "cited": sorted(cited)})
            continue
        surfaced = ranked[0]
        lexical = ranked_lex[0] if ranked_lex and ranked_lex[0].gated_in else None
        pool = [u for u in ranked if u.evidence_id not in
                {surfaced.evidence_id, lexical.evidence_id if lexical else ""}]
        if not pool:
            pool = [u for u in ranked if u.evidence_id != surfaced.evidence_id]
        control = rng.choice(pool)
        items = {"surfaced": surfaced.evidence_id, "random": control.evidence_id}
        if lexical is not None:
            items["lexical"] = lexical.evidence_id
        distinct = sorted(set(items.values()))
        rng.shuffle(distinct)
        labels = {eid: chr(ord("A") + i) for i, eid in enumerate(distinct)}
        texts = dict(evidence)
        body = (
            f"DECISION: {response.get('verdict')} {rf.symbol}\n"
            f"THESIS: {thesis}\n\nUNUSED EVIDENCE\n"
            + "\n".join(f"  {labels[eid]}: {texts[eid]}" for eid in sorted(labels,
                                                                           key=labels.__getitem__))
        )
        if client.live and client.remaining() < 1 and not _judge_recorded(client, body):
            stopped = f"stopped at {rf.symbol} {rf.at.isoformat()}: stage cap reached"
            break
        try:
            verdict = client.complete_json(
                [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": body}],
                required_keys=("ratings",), max_tokens=700, thinking=Thinking.LOW,
            )
        except QwenError as exc:
            stopped = f"stopped at {rf.symbol}: {str(exc)[:120]}"
            break
        ratings = verdict.get("ratings", {}) if isinstance(verdict.get("ratings"), dict) else {}

        def rating_of(
            eid: str, ratings: dict[str, Any] = ratings, labels: dict[str, str] = labels
        ) -> int | None:
            raw = ratings.get(labels[eid])
            try:
                return int(str(raw))
            except (TypeError, ValueError):
                return None

        rows.append({
            "symbol": rf.symbol, "at": rf.at.isoformat(), "verdict": response.get("verdict"),
            "cited": sorted(cited), "unused": len(ranked),
            "surfaced": {**surfaced.as_dict(), "rating": rating_of(surfaced.evidence_id)},
            "lexical": None if lexical is None else {
                **lexical.as_dict(), "rating": rating_of(lexical.evidence_id)},
            "random": {"evidence_id": control.evidence_id, "text": control.text,
                       "rating": rating_of(control.evidence_id)},
            "labels": labels,
        })

    judged = [r for r in rows if "surfaced" in r and r["surfaced"]["rating"] is not None
              and r["random"]["rating"] is not None]

    def share(key: str) -> dict[str, Any]:
        vals = [r[key]["rating"] for r in judged if r.get(key) and r[key]["rating"] is not None]
        return {"n": len(vals), "judged_relevant": sum(1 for v in vals if v >= 1),
                "judged_directly_relevant": sum(1 for v in vals if v >= 2),
                "mean_rating": round(sum(vals) / len(vals), 3) if vals else None}

    wins = sum(1 for r in judged if r["surfaced"]["rating"] > r["random"]["rating"])
    losses = sum(1 for r in judged if r["surfaced"]["rating"] < r["random"]["rating"])
    same_item = sum(1 for r in judged if r["surfaced"]["evidence_id"] == r["random"]["evidence_id"])
    return {
        "question": "Is the unused item STORM's moderator score surfaces judged more relevant to "
                    "the decision than a uniformly random unused item?",
        "encoder": encoder_name,
        "theses": len(theses),
        "judged": len(judged),
        "skipped": sum(1 for r in rows if "skipped" in r),
        "surfaced": share("surfaced"),
        "lexical_surfaced": share("lexical"),
        "random": share("random"),
        "paired": {"surfaced_rated_higher": wins, "random_rated_higher": losses,
                   "tied": len(judged) - wins - losses, "same_item": same_item},
        "judge": "qwen3.8-max via the hackathon endpoint, blind to which item is which, items in "
                 "seeded random order. NOT metaeval-checked: no human-labelled agreement set "
                 "exists for this judge.",
        "rows": rows,
        "stopped": stopped,
    }


def _judge_recorded(client: CappedRecordingClient, body: str) -> bool:
    payload: dict[str, Any] = {
        "model": client._model,
        "messages": [{"role": "system", "content": JUDGE_PROMPT},
                     {"role": "user", "content": body}],
        "temperature": 0.0, "max_tokens": 700,
    }
    client._apply_thinking(payload, Thinking.LOW)
    payload["response_format"] = {"type": "json_object"}
    return client._cache_key(payload) in client.recording


# --- S7 -----------------------------------------------------------------------------------------


def recorded_debate_log(path: Path = DESK_NOTES_PATH) -> dict[str, Any]:
    """How many debates on the live record had a second round the detector could have saved."""
    if not path.exists():
        return {"available": False}
    rounds: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for note in row.get("notes", []):
            text = str(note)
            if text.startswith("[debate]") and " round(s), ended " in text:
                head = text.split(" round(s), ended ", 1)[0]
                try:
                    rounds.append(int(head.rsplit(" ", 1)[-1]))
                except ValueError:
                    continue
    return {
        "available": True,
        "debates_held": len(rounds),
        "with_two_or_more_rounds": sum(1 for r in rounds if r >= 2),
        "rounds_the_detector_could_have_saved": sum(max(0, r - 2) for r in rounds),
        "why": "the desk's default debate budget is 2.0bps = one round (agents/desk.py), so no "
               "live debate has ever reached a round the detector reads",
    }


def _stop_round(debate: debate_mod.Debate) -> int | None:
    """Replay the MAF counter over a full-length debate: the round it would have stopped after."""
    count = 0
    for index in range(1, debate.rounds):
        if debate_mod.round_stalled(debate.positions, index):
            count += 1
        else:
            count = max(0, count - 1)
        if count > debate_mod.MAX_STALL_COUNT:
            return index
    return None


def _signed_gap(positions: Sequence[debate_mod.Position], upto: int) -> float | None:
    bull = [p for p in positions if p.side is debate_mod.Side.BULL and p.round_index <= upto]
    bear = [p for p in positions if p.side is debate_mod.Side.BEAR and p.round_index <= upto]
    if not bull or not bear:
        return None
    left, right = debate_mod.signed(bull[-1]), debate_mod.signed(bear[-1])
    return None if left is None or right is None else abs(left - right)


def run_s7(kept: Sequence[tuple[RecordedFrame, Deliberation]], client: CappedRecordingClient,
           *, debates: int = 3) -> dict[str, Any]:
    """Full-length debates, the detector replayed over them, and an LLM progress judge."""
    client.stage = "s7"
    frames = sorted({rf.symbol: rf for rf, _ in kept}.values(),
                    key=lambda f: -len(f.evidence_lines))
    rows: list[dict[str, Any]] = []
    stopped = ""
    for rf in frames:
        if len(rows) >= debates:
            break
        if client.live and client.remaining() < 2 * debate_mod.MAX_ROUNDS + 2:
            stopped = f"stopped before {rf.symbol}: stage cap cannot fund a full debate"
            break
        held = debate_mod.hold(
            symbol=rf.symbol, horizon_hours=24.0, evidence=rf.evidence_lines,
            bull_seat=client, bear_seat=client,
            budget=debate_mod.DebateBudget(limit_bps=Decimal("100")),
            max_rounds=debate_mod.MAX_ROUNDS, stall_detection=False,
        )
        if held.ending is debate_mod.Ending.UNAVAILABLE:
            stopped = f"{rf.symbol}: {held.note}"
            break
        stop = _stop_round(held)
        saved = 0 if stop is None else held.rounds - (stop + 1)
        wrong: bool | None = None
        why = ""
        if stop is not None and saved > 0:
            final = held.final
            at_stop = {s: [p for p in held.positions if p.side is s and p.round_index <= stop][-1]
                       for s in debate_mod.Side}
            flips = [s.value for s in debate_mod.Side
                     if (last := final[s]) is not None
                     and last.direction != at_stop[s].direction]
            gap_then, gap_end = _signed_gap(held.positions, stop), held.gap_bps
            gap_moved = (gap_then is not None and gap_end is not None
                         and abs(gap_end - gap_then) > debate_mod.CONVERGENCE_BPS)
            wrong = bool(flips) or gap_moved or held.ending is debate_mod.Ending.CONVERGED
            why = ("direction changed after the stop: " + ", ".join(flips) if flips else
                   "gap moved after the stop" if gap_moved else
                   "converged after the stop" if held.ending is debate_mod.Ending.CONVERGED else
                   "nothing changed after the stop")
        labels: list[dict[str, Any]] = []
        for index in range(1, held.rounds):
            transcript = "\n".join(p.render() for p in held.positions if p.round_index <= index)
            body = (
                f"SYMBOL: {rf.symbol}\nTRANSCRIPT (latest round is round {index + 1})\n"
                + transcript
            )
            try:
                judged = client.complete_json(
                    [{"role": "system", "content": PROGRESS_PROMPT},
                     {"role": "user", "content": body}],
                    required_keys=("is_in_loop", "is_progress_being_made"), max_tokens=500,
                    thinking=Thinking.LOW,
                )
            except QwenError as exc:
                labels.append({"round": index, "llm": None, "error": str(exc)[:120]})
                continue
            in_loop = _answer(judged.get("is_in_loop"))
            progressing = _answer(judged.get("is_progress_being_made"))
            llm_stalled = None if in_loop is None or progressing is None else (
                in_loop or not progressing
            )
            labels.append({
                "round": index,
                "proxy_stalled": debate_mod.round_stalled(held.positions, index),
                "llm_stalled": llm_stalled,
                "readings": [r.as_dict() for r in debate_mod.round_progress(held.positions,
                                                                            index)],
            })
        rows.append({
            "symbol": rf.symbol, "at": rf.at.isoformat(), "rounds_held": held.rounds,
            "ending": held.ending.value, "final_gap_bps": held.gap_bps,
            "detector_stop_after_round": None if stop is None else stop + 1,
            "rounds_saved": saved, "detection_wrong": wrong, "why": why,
            "progress_labels": labels,
            "transcript": [p.as_dict() for p in held.positions],
        })
    agree = [lab for r in rows for lab in r["progress_labels"]
             if lab.get("llm_stalled") is not None]
    return {
        "question": "How many paid rounds does the non-LLM stall proxy save on full-length "
                    "debates, how many of its stops were wrong, and how often does it agree "
                    "with an LLM asked MAF's two progress questions?",
        "recorded_log": recorded_debate_log(),
        "debates": len(rows),
        "rounds_held": sum(r["rounds_held"] for r in rows),
        "detections": sum(1 for r in rows if r["detector_stop_after_round"] is not None),
        "rounds_saved": sum(r["rounds_saved"] for r in rows),
        "wrong_detections": sum(1 for r in rows if r["detection_wrong"]),
        "proxy_vs_llm": {
            "labelled_rounds": len(agree),
            "agree": sum(1 for lab in agree if lab["proxy_stalled"] == lab["llm_stalled"]),
            "proxy_stalled": sum(1 for lab in agree if lab["proxy_stalled"]),
            "llm_stalled": sum(1 for lab in agree if lab["llm_stalled"]),
        },
        "rows": rows,
        "stopped": stopped,
    }


def _answer(field_value: Any) -> bool | None:
    if isinstance(field_value, dict):
        field_value = field_value.get("answer")
    if isinstance(field_value, bool):
        return field_value
    if isinstance(field_value, str) and field_value.strip().lower() in ("true", "false"):
        return field_value.strip().lower() == "true"
    return None


# --- the run ------------------------------------------------------------------------------------


def evaluate(*, live: bool, frames_path: Path = FRAMES_PATH,
             recording_path: Path = RECORDING_PATH) -> dict[str, Any]:
    frames = load_frames(frames_path)
    recording = load_recording(recording_path)
    client = CappedRecordingClient(recording, live=live)
    client.save_to = recording_path if live else None
    try:
        s5, kept = run_s5(frames, client)
        s6 = run_s6(kept, client)
        s4 = run_s4(kept, client)
        s7 = run_s7(kept, client)
    finally:
        if live:
            save_recording(recording, recording_path)
    frames_digest = hashlib.sha256(frames_path.read_bytes()).hexdigest()[:16]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "frames": len(frames),
        "frames_sha256_16": frames_digest,
        "frame_span": {
            "first": min(f.at for f in frames).isoformat() if frames else None,
            "last": max(f.at for f in frames).isoformat() if frames else None,
        },
        "qwen_calls_made_this_run": client.total_posts,
        "qwen_calls_by_stage": dict(client.posts),
        "responses_replayed_from_recording": client.replayed,
        "qwen_cap": QWEN_CAP,
        "s5": s5,
        "s6": s6,
        "s4": s4,
        "s7": s7,
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import contextlib

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="ARGUS thesis-quality evaluation (S4-S7)")
    parser.add_argument("--build-frames", action="store_true")
    parser.add_argument("--run", action="store_true", help="call Qwen for anything unrecorded")
    args = parser.parse_args(argv)
    if args.build_frames:
        frames = build_frames()
        save_frames(frames)
        print(f"{len(frames)} frame(s) written to {FRAMES_PATH}")
        return 0
    report = evaluate(live=args.run)
    write_artefact(REPORT_PATH, report)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2))
    for stage in ("s5", "s6", "s4", "s7"):
        blob = {k: v for k, v in report[stage].items() if k != "rows"}
        print(stage, json.dumps(blob, indent=2, default=str))
    return 0


__all__ = [
    "FRAMES_PATH",
    "QWEN_CAP",
    "RECORDING_PATH",
    "REPORT_PATH",
    "STAGE_CAPS",
    "CappedRecordingClient",
    "RecordedFrame",
    "build_frames",
    "evaluate",
    "load_frames",
    "recorded_debate_log",
    "run_s4",
    "run_s5",
    "run_s6",
    "run_s7",
    "technical_evidence",
]


if __name__ == "__main__":
    raise SystemExit(main())
