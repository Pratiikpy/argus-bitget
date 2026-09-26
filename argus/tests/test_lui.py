"""Language-interface tests, driven by the judge questions.

The acceptance criteria are the 25 questions in ``research/subthemes/_CENSUS-lui.md``, and the ones
that matter most are the ones that are *supposed to be refused*: an instrument the venue does not
carry, a reference with nothing to bind to, a statistic the sample cannot support. A confident
answer to any of those is the failure this suite exists to catch.

The second property under test is grounding. Every non-refusal answer must carry at least one
source, because an assertion nobody can check is worth less than a refusal somebody can act on.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from argus.lui.answer import answer
from argus.lui.question import (
    TRADED_SYMBOLS,
    Conversation,
    Intent,
    Speed,
    Tense,
    classify,
    extract_symbols,
    resolve_window,
)
from argus.paper.ledger import Entry, PaperLedger

NOW = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)  # a Monday
WEEKEND = datetime(2026, 9, 12, 15, 0, tzinfo=UTC)  # the Saturday before


def _entry(
    seq: int,
    *,
    symbol: str = "NVDAUSDT",
    decided: datetime,
    verdict: str = "no_trade",
    settled: datetime | None = None,
    net_pnl: str | None = None,
    phase: str = "weekend",
) -> Entry:
    abstention = verdict == "no_trade"
    return Entry(
        seq=seq,
        decided_at=decided.isoformat(),
        symbol=symbol,
        verdict=verdict,
        side="flat" if abstention else "long",
        quantity="0" if abstention else "1",
        entry_price="100",
        stated_confidence=0.92,
        thesis=f"fixture thesis for {symbol}",
        invalidation=("price gaps 2% on the open",),
        market_state_hash="m" * 16,
        approved_intent_hash="a" * 16,
        session_phase=phase,
        hours_to_discovery=46.0,
        entry_cost_bps="6",
        prev_hash="0" * 16,
        settled_at=settled.isoformat() if settled else None,
        exit_price="101" if settled else None,
        gross_pnl=net_pnl if settled else None,
        net_pnl=net_pnl if settled else None,
        direction_correct=(Decimal(net_pnl) > 0) if (settled and net_pnl) else None,
    )


def _ledger(entries: list[Entry], directory: Path) -> PaperLedger:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "paper.jsonl"
    path.write_text(
        "\n".join(json.dumps(asdict(e), default=str) for e in entries) + "\n", encoding="utf-8"
    )
    return PaperLedger(path=path)


@pytest.fixture
def weekend_ledger(tmp_path: Path) -> PaperLedger:
    """The state the live ledger was actually in: all abstentions, nothing settled."""
    entries = [
        _entry(i, symbol=TRADED_SYMBOLS[i % 3], decided=WEEKEND + timedelta(minutes=i))
        for i in range(1, 10)
    ]
    return _ledger(entries, tmp_path)


class TestRefusalIsAnAnswer:
    """The judge questions that are supposed to be refused. Answering them anyway is the bug."""

    def test_an_instrument_the_desk_does_not_trade_is_named_not_guessed(self) -> None:
        """The ledger has no row on gold, so the record layer refuses — and says what is true:
        gold trades on Bitget, the desk does not trade it, and the research layer (which runs
        first in the console) answers it. The reason once said gold was "not listed on Bitget",
        which XAUUSDT on the live contract list disproves."""
        q = classify("what's gold trading at?", now=NOW)
        assert q.intent is Intent.UNSUPPORTED
        assert "trades on Bitget (XAUUSDT)" in q.reason
        assert "not listed" not in q.reason

    def test_the_refusal_lists_what_can_be_asked_about_instead(
        self, weekend_ledger: PaperLedger
    ) -> None:
        a = answer(weekend_ledger, classify("what's gold trading at?", now=NOW))
        assert a.refused
        assert "NVDAUSDT" in " ".join(a.lines)

    def test_a_vague_reference_with_nothing_to_bind_to_asks_rather_than_picks(self) -> None:
        """A question, not an instruction — an imperative is caught earlier, by the order guard."""
        q = classify("what evidence backed that?", now=NOW)
        assert q.intent is Intent.AMBIGUOUS
        assert q.candidates == TRADED_SYMBOLS

    def test_calibration_below_the_floor_is_refused_with_the_count(
        self, weekend_ledger: PaperLedger
    ) -> None:
        a = answer(weekend_ledger, classify("are you well calibrated?", now=NOW))
        joined = " ".join(a.lines)
        assert "floor of 5" in joined
        assert "refusal to compute, not a missing feature" in joined

    def test_a_quote_on_the_record_path_points_to_the_live_quote(
        self, weekend_ledger: PaperLedger
    ) -> None:
        """The research layer quotes live prices first; the record layer only sees a price
        question it did not recognise, and must not claim a listed name is unlisted."""
        a = answer(weekend_ledger, classify("what is NVDA trading at?", now=NOW))
        assert a.refused
        assert "where is NVDA trading right now" in a.reason
        assert "not a contract Bitget lists" not in a.reason

    def test_an_unrecognised_question_does_not_get_routed_to_the_nearest_answerer(
        self, weekend_ledger: PaperLedger
    ) -> None:
        q = classify("what is the airspeed velocity of an unladen swallow", now=NOW)
        assert q.intent is Intent.UNKNOWN
        a = answer(weekend_ledger, q)
        assert a.refused
        assert "Answerable today" in " ".join(a.lines)


class TestANegativeDecisionNumberIsRefusedNotSilentlyReplaced:
    """Found by the 2026-09-22 ADVERSARIAL LENS pass: "decision -1" used to answer with the LATEST
    decision, undisclosed, because `_SEQ` could not capture the sign, so the "-1" was silently
    dropped rather than parsed and refused — the exact silent-reinterpretation this project's
    other rules exist to catch, just reached through a regex gap instead of a routing one."""

    def test_a_negative_decision_number_is_refused_like_decision_zero_already_was(
        self, weekend_ledger: PaperLedger
    ) -> None:
        zero = answer(weekend_ledger, classify("show me decision 0", now=NOW))
        negative = answer(weekend_ledger, classify("show me decision -1", now=NOW))
        assert zero.refused
        assert negative.refused
        assert negative.reason == zero.reason

    def test_it_is_not_silently_answered_with_the_most_recent_decision(
        self, weekend_ledger: PaperLedger
    ) -> None:
        q = classify("show me decision -1", now=NOW)
        assert q.seq == -1
        a = answer(weekend_ledger, q)
        assert a.refused
        assert "seq 9" not in " ".join(a.lines)  # 9 is the latest row in weekend_ledger

    def test_a_genuinely_absent_seq_still_falls_back_to_the_most_recent_decision(
        self, weekend_ledger: PaperLedger
    ) -> None:
        """The fallback itself is correct and intentional when no number was ever given — only a
        malformed-but-present number (a negative one) must not be swallowed by it the same way."""
        q = classify("explain the reasoning", now=NOW)
        assert q.seq is None
        a = answer(weekend_ledger, q)
        assert not a.refused
        assert "seq 9" in " ".join(a.lines)

    def test_a_large_negative_number_beyond_the_six_digit_cap_is_not_captured_either(self) -> None:
        q = classify("show me decision -9999999", now=NOW)
        assert q.seq is None


class TestAnInstructionIsNeverReinterpretedAsAQuestion:
    """Found by driving the judge session: "sell half of that" returned a past decision's thesis.

    The word "sell" matched the decision-explanation pattern, the vague reference bound to the
    decision discussed a turn earlier, and an instruction came back answered as though it had been
    a question. An interface that silently converts an imperative into a query is worse than one
    that fails, because the user believes they were understood.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "sell half of that",
            "buy 100 NVDA",
            "cancel that order",
            "rebalance to 50/30/20",
            "undo the last trade",
            "close my position",
            "short TSLA",
        ],
    )
    def test_imperatives_are_refused_by_name(self, text: str) -> None:
        assert classify(text, now=NOW).intent is Intent.ORDER

    def test_the_refusal_says_where_orders_actually_come_from(
        self, weekend_ledger: PaperLedger
    ) -> None:
        a = answer(weekend_ledger, classify("sell half of that", now=NOW))
        assert a.refused
        assert "does not place" in a.reason
        assert "risk layer" in " ".join(a.lines)

    def test_an_imperative_is_refused_even_when_a_reference_would_resolve(self) -> None:
        """The prior turn made "that" resolvable. That must not rescue an instruction."""
        convo = Conversation()
        convo.remember(classify("show me decision 26", now=NOW))
        assert classify("sell half of that", now=NOW, conversation=convo).intent is Intent.ORDER

    @pytest.mark.parametrize(
        "text",
        [
            "why did you sell NVDA",
            "what did you buy today",
            "did you close that position",
            "show me decision 3",
        ],
    )
    def test_questions_containing_trade_verbs_still_answer(self, text: str) -> None:
        assert classify(text, now=NOW).intent is not Intent.ORDER


class TestTheQuestionNobodyElseAnswers:
    """Why did you do nothing — the most common correct answer on this venue."""

    def test_why_nothing_routes_to_abstention_not_to_a_generic_decision(self) -> None:
        q = classify("why did you do nothing all weekend?", now=NOW)
        assert q.intent is Intent.ABSTENTION_WHY
        assert q.window is not None and q.window.label == "the weekend"

    def test_the_answer_cites_real_rows_and_says_abstention_is_a_decision(
        self, weekend_ledger: PaperLedger
    ) -> None:
        a = answer(weekend_ledger, classify("why did you do nothing?", now=NOW))
        assert not a.refused
        assert a.sources and all(s.kind == "ledger" for s in a.sources)
        assert "recorded decision, not an absence of one" in " ".join(a.lines)

    def test_an_ungraded_abstention_is_not_claimed_to_be_correct(
        self, weekend_ledger: PaperLedger
    ) -> None:
        """Nothing has settled, so nothing may be called a good call."""
        a = answer(weekend_ledger, classify("why did you do nothing?", now=NOW))
        joined = " ".join(a.lines)
        assert "None has settled yet, so none is graded" in joined

    def test_no_abstentions_in_the_window_is_a_refusal_not_an_empty_answer(
        self, tmp_path: Path
    ) -> None:
        led = _ledger(
            [_entry(1, decided=NOW, verdict="open_long", phase="rth")], tmp_path
        )
        a = answer(led, classify("why did you do nothing today?", now=NOW))
        assert a.refused


class TestGrounding:
    def test_every_non_refusal_answer_carries_a_source(
        self, weekend_ledger: PaperLedger
    ) -> None:
        asked = [
            "why did you do nothing?",
            "what is the sharpe?",
            "is the log tamper-evident?",
            "what is my position?",
            "show me decision 3",
            "what decisions did you make?",
            "is the market open?",
        ]
        for text in asked:
            a = answer(weekend_ledger, classify(text, now=NOW))
            assert a.is_grounded, f"ungrounded answer to {text!r}"
            if not a.refused:
                assert a.sources, f"no source for {text!r}"

    def test_a_decision_answer_quotes_the_hash_so_it_can_be_checked(
        self, weekend_ledger: PaperLedger
    ) -> None:
        a = answer(weekend_ledger, classify("show me decision 3", now=NOW))
        assert any("hash" in line for line in a.lines)

    def test_evidence_reads_the_committed_state_and_does_not_refetch(
        self, weekend_ledger: PaperLedger
    ) -> None:
        """Re-fetching would answer what is visible now, not what was visible then."""
        a = answer(weekend_ledger, classify("what evidence did you have for decision 3?", now=NOW))
        joined = " ".join(a.lines)
        assert "market state hash" in joined.lower()
        assert "as-of" in joined or "cannot be revised" in joined


class TestMultiTurnReferences:
    """The census found no studied system resolves these. This is the differentiator."""

    def test_that_binds_to_the_symbol_from_the_previous_turn(self) -> None:
        convo = Conversation()
        convo.remember(classify("why did you buy TSLA?", now=NOW))
        follow = classify("and what evidence did you have for that?", now=NOW, conversation=convo)
        assert follow.intent is Intent.EVIDENCE
        assert follow.symbols == ("TSLAUSDT",)

    def test_a_named_decision_binds_tighter_than_a_symbol(self) -> None:
        """"That" after "decision 25" means row 25, not merely the symbol it was about."""
        convo = Conversation()
        convo.remember(classify("show me decision 25", now=NOW))
        follow = classify("what evidence backed that?", now=NOW, conversation=convo)
        assert follow.seq == 25

    def test_a_three_turn_chain_keeps_resolving(self, weekend_ledger: PaperLedger) -> None:
        convo = Conversation()
        for text in ("show me decision 3", "what evidence backed that?", "why that one?"):
            q = classify(text, now=NOW, conversation=convo)
            convo.remember(q)
            assert q.intent is not Intent.AMBIGUOUS
        assert len(convo.turns) == 3


class TestTemporalResolution:
    def test_a_naive_clock_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            resolve_window("today", now=datetime(2026, 9, 14, 15, 0))

    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("what did you decide today?", "today"),
            ("did we profit yesterday?", "yesterday"),
            ("what happened overnight?", "overnight"),
            ("why nothing over the weekend?", "the weekend"),
            ("decisions this week?", "this week"),
        ],
    )
    def test_phrases_resolve_to_named_windows(self, text: str, label: str) -> None:
        q = classify(text, now=NOW)
        assert q.window is not None and q.window.label == label

    def test_a_forward_looking_question_is_marked_future_not_past(self) -> None:
        """Confusing "did we" with "will we" is the difference between a report and an order."""
        assert classify("will NVDA go up tomorrow?", now=NOW).tense is Tense.FUTURE
        assert classify("did we profit yesterday?", now=NOW).tense is Tense.PAST

    def test_a_window_uses_the_injected_clock_not_the_wall_clock(self) -> None:
        a = classify("what did you decide today?", now=NOW).window
        b = classify("what did you decide today?", now=NOW + timedelta(days=5)).window
        assert a is not None and b is not None
        assert a.start != b.start

    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("今天做了什么决定", "today"),
            ("昨日有没有盈利", "yesterday"),
            ("隔夜发生了什么", "overnight"),
            ("为什么整个周末都没有交易", "the weekend"),
            ("為什麼整個週末都沒有交易", "the weekend"),
            ("上周末为什么没交易", "last weekend"),
            ("本周的决策", "this week"),
            ("這週的決策", "this week"),
            ("上周的决策", "last week"),
            ("上週的決策", "last week"),
            ("到目前为止表现如何", "all time"),
        ],
    )
    def test_chinese_time_words_name_the_same_windows(self, text: str, label: str) -> None:
        window = resolve_window(text, now=NOW)
        assert window is not None
        assert window.label == label

    def test_this_weekend_in_english_is_not_this_week(self) -> None:
        window = resolve_window("why nothing this weekend", now=NOW)
        assert window is not None
        assert window.label == "the weekend"

    def test_last_weekend_asked_on_a_weekend_is_the_one_before(self) -> None:
        saturday = datetime(2026, 9, 26, 1, 0, tzinfo=UTC)
        current = resolve_window("why nothing this weekend", now=saturday)
        previous = resolve_window("why nothing last weekend", now=saturday)
        assert current is not None
        assert previous is not None
        assert current.start == datetime(2026, 9, 26, tzinfo=UTC)
        assert previous.start == datetime(2026, 9, 19, tzinfo=UTC)
        on_monday = resolve_window("why nothing last weekend", now=NOW)
        assert on_monday is not None
        assert on_monday.start == datetime(2026, 9, 12, tzinfo=UTC)


SATURDAY_EARLY = datetime(2026, 9, 26, 1, 0, tzinfo=UTC)
"""An hour into a weekend: the moment the public CI asked "all weekend" and got a refusal."""


class TestTheWeekendAskedAbout:
    """ "The weekend" is the current one; when nothing is recorded in it yet, the answer is about
    the latest weekend that has a record, and its first line says so."""

    def _ask(self, ledger: PaperLedger, text: str) -> tuple[bool, list[str]]:
        reply = answer(ledger, classify(text, now=SATURDAY_EARLY))
        return reply.refused, reply.lines

    def test_an_empty_current_weekend_answers_for_the_last_recorded_one(
        self, tmp_path: Path
    ) -> None:
        saturday = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
        ledger = _ledger([_entry(1, decided=saturday), _entry(2, decided=saturday)], tmp_path)
        refused, lines = self._ask(ledger, "why did you do nothing all weekend")
        assert not refused
        assert lines[0] == (
            "Assumed: the weekend of 2026-09-19 to 2026-09-20, the latest weekend with a record; "
            "nothing is recorded for the weekend that began 2026-09-26."
        )
        assert lines[1].startswith("2 abstention(s) in the weekend of 2026-09-19 to 2026-09-20")

    def test_the_chinese_answer_names_the_weekend_in_chinese(self, tmp_path: Path) -> None:
        saturday = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
        ledger = _ledger([_entry(1, decided=saturday)], tmp_path)
        refused, lines = self._ask(ledger, "为什么整个周末都没有交易")
        assert not refused
        assert lines[0].startswith("假设\uff1a按 2026-09-19 至 2026-09-20 周末 回答")
        assert "\uff082026-09-19 至 2026-09-20 周末\uff09" in lines[1]
        assert "weekend of" not in lines[1]

    def test_a_weekend_with_any_decision_is_never_moved(self, tmp_path: Path) -> None:
        ledger = _ledger(
            [
                _entry(1, decided=datetime(2026, 9, 19, 12, 0, tzinfo=UTC)),
                _entry(2, symbol="TSLAUSDT", decided=SATURDAY_EARLY - timedelta(minutes=30)),
            ],
            tmp_path,
        )
        refused, lines = self._ask(ledger, "why did you do nothing on NVDA all weekend")
        assert refused, "the current weekend has a record, so NVDA's empty window is refused"
        assert not lines[0].startswith("Assumed:")

    def test_an_empty_today_answers_for_the_last_day_with_a_record(self, tmp_path: Path) -> None:
        thursday = datetime(2026, 9, 24, 15, 0, tzinfo=UTC)
        ledger = _ledger([_entry(1, decided=thursday, phase="rth")], tmp_path)
        refused, lines = self._ask(ledger, "why no trades today")
        assert not refused
        assert lines[0] == (
            "Assumed: 2026-09-24, the latest day with a record; nothing is recorded for today "
            "(2026-09-26)."
        )
        assert lines[1].startswith("1 abstention(s) in 2026-09-24")

    def test_an_empty_week_answers_for_the_last_week_with_a_record(self, tmp_path: Path) -> None:
        ledger = _ledger([_entry(1, decided=datetime(2026, 9, 9, 15, 0, tzinfo=UTC))], tmp_path)
        refused, lines = self._ask(ledger, "why nothing this week")
        assert not refused
        assert lines[0] == (
            "Assumed: the week of 2026-09-07 to 2026-09-13, the latest week with a record; "
            "nothing is recorded for this week (from 2026-09-21)."
        )

    def test_nothing_within_the_lookback_is_refused_not_invented(self, tmp_path: Path) -> None:
        long_ago = datetime(2026, 6, 6, 12, 0, tzinfo=UTC)
        ledger = _ledger([_entry(1, decided=long_ago)], tmp_path)
        refused, lines = self._ask(ledger, "why did you do nothing all weekend")
        assert refused
        assert lines[0] == "No abstentions in the weekend."


class TestAskingForTheListByWindow:
    """Found while testing the weekend fix (2026-09-26): a plural object and "last week" broke
    the plainest request for the record."""

    @pytest.mark.parametrize(
        "text",
        [
            "list decisions this week",
            "list decisions last week",
            "list all decisions",
            "show me the decisions from last week",
        ],
    )
    def test_the_list_is_asked_for(self, text: str) -> None:
        assert classify(text, now=SATURDAY_EARLY).intent is Intent.DECISION_LIST

    @pytest.mark.parametrize(
        "text", ["NVDA last", "what is the last price of NVDA", "how did NVDA move last week"]
    )
    def test_last_is_still_a_price_word(self, text: str) -> None:
        assert classify(text, now=SATURDAY_EARLY).intent is Intent.MARKET


class TestSymbolExtraction:
    def test_company_names_resolve_to_the_traded_symbol(self) -> None:
        assert extract_symbols("how did nvidia do")[0] == ("NVDAUSDT",)
        assert extract_symbols("what about alphabet")[0] == ("GOOGLUSDT",)

    def test_an_off_venue_instrument_returns_the_reason_not_an_empty_tuple(self) -> None:
        symbols, reason = extract_symbols("what's gold doing")
        assert symbols == ()
        assert "Bitget" in reason

    def test_several_symbols_are_all_captured_in_order(self) -> None:
        assert extract_symbols("compare NVDA and TSLA")[0] == ("NVDAUSDT", "TSLAUSDT")


class TestLatencyBudgets:
    """The budget is part of the contract: a right answer delivered slowly is still a miss."""

    @pytest.mark.parametrize(
        ("text", "speed"),
        [
            ("what is my position?", Speed.FAST),
            ("why did you do nothing?", Speed.FAST),
            ("is the log tamper-evident?", Speed.FAST),
            ("what is the sharpe?", Speed.FAST),
            ("what is NVDA trading at?", Speed.MEDIUM),
        ],
    )
    def test_questions_carry_their_budget(self, text: str, speed: Speed) -> None:
        assert classify(text, now=NOW).speed is speed

    def test_the_record_only_questions_are_all_on_the_fast_path(self) -> None:
        """Nothing answerable from the ledger alone should ever need a network call."""
        for text in ("what is my position?", "why did you do nothing?", "show me decision 3"):
            assert classify(text, now=NOW).speed is Speed.FAST


class TestPerformanceAnswer:
    def test_the_missing_numbers_are_explained_rather_than_zeroed(
        self, weekend_ledger: PaperLedger
    ) -> None:
        a = answer(weekend_ledger, classify("what is the sharpe?", now=NOW))
        joined = " ".join(a.lines)
        assert "not available" in joined
        assert "abstentions are scored separately" in joined

    def test_once_trades_settle_the_three_numbers_are_stated(self, tmp_path: Path) -> None:
        """Thirty-five days, not five: `MIN_DAYS_FOR_SHARPE` is 30 (see `eval/performance.py`),
        so a five-day fixture now correctly yields *"not available"* with a reason rather than a
        number. This test is named for the populated case, so it gets a sample that supports one.
        The refusal path is covered by
        :meth:`test_a_thin_sample_states_the_reason_rather_than_a_number` below."""
        cycle = ["100", "-40", "80", "-20", "60"]
        entries = [
            _entry(i + 1, decided=NOW - timedelta(days=36 - i),
                   settled=NOW - timedelta(days=35 - i),
                   verdict="open_long", net_pnl=cycle[i % len(cycle)], phase="rth")
            for i in range(35)
        ]
        a = answer(_ledger(entries, tmp_path), classify("what is the sharpe?", now=NOW))
        joined = " ".join(a.lines).lower()
        assert "sharpe" in joined and "win rate" in joined and "drawdown" in joined
        assert "not available" not in joined, "a 35-day sample must produce real figures"
        assert "largest contributor" in joined, "the one-symbol trap must stay visible"

    def test_a_thin_sample_states_the_reason_rather_than_a_number(self, tmp_path: Path) -> None:
        """The real 2026-09-20 shape: two settled trades. The console must say why each figure is
        withheld, not go silent and not invent one — the same discipline the cockpit enforces."""
        entries = [
            _entry(i, decided=NOW - timedelta(days=4 - i), settled=NOW - timedelta(days=3 - i),
                   verdict="open_long", net_pnl=p, phase="rth")
            for i, p in enumerate(["100", "60"], start=1)
        ]
        a = answer(_ledger(entries, tmp_path), classify("what is the sharpe?", now=NOW))
        joined = " ".join(a.lines).lower()
        assert "not available" in joined
        assert "needs at least" in joined, "the reason must be stated, not just the absence"


class TestTheQuestionsAPersonActuallyAsks:
    """Fluency, pinned to phrasing rather than to vocabulary.

    Found by executing this project's own README as a stranger would: the README's example question
    was **"how did we do this week?"**, and the interface refused it. Track 3 scores LUI fluency,
    and the three most natural ways to ask about performance all reached UNKNOWN while the refusal
    message listed performance as answerable.

    Worse, that refusal message advertised "open positions" while "what do we hold?" was also
    unrecognised — an interface contradicting its own help text. Both classes are pinned here.
    """

    @pytest.mark.parametrize("phrasing", [
        "how did we do this week?",
        "how did we do?",
        "how are we doing?",
        "how's the desk doing?",
        "how has the book been?",
        "how did we perform?",
        "how much did we make?",
        "how much money did we make?",
        "did we beat the fee?",
        "are we profitable after costs?",
    ])
    def test_performance_is_recognised_however_it_is_phrased(self, phrasing: str) -> None:
        assert classify(phrasing, now=NOW).intent is Intent.PERFORMANCE

    @pytest.mark.parametrize("phrasing", [
        "what do we hold?",
        "what are our positions?",
        "are we long NVDA?",
        "are we short anything?",
        "what's our exposure?",
    ])
    def test_positions_are_recognised_because_the_refusal_promises_them(
        self, phrasing: str
    ) -> None:
        assert classify(phrasing, now=NOW).intent is Intent.POSITION

    @pytest.mark.parametrize("phrasing", [
        "what happened yesterday?",
        "summarise the week",
        "give me a recap",
        "catch me up",
    ])
    def test_a_recap_lists_the_decisions_in_the_window(self, phrasing: str) -> None:
        assert classify(phrasing, now=NOW).intent is Intent.DECISION_LIST

    @pytest.mark.parametrize("phrasing", [
        "why did we not trade NVDA?",
        "why did you do nothing on NVDA?",
        "why did the desk stand aside?",
    ])
    def test_the_new_patterns_did_not_steal_abstention(self, phrasing: str) -> None:
        """The performance pattern contains bare verbs; "why" questions must still win."""
        assert classify(phrasing, now=NOW).intent is Intent.ABSTENTION_WHY

    def test_a_question_about_the_sharpe_is_not_swallowed_by_the_recap_pattern(self) -> None:
        assert classify("what happened to the Sharpe?", now=NOW).intent is Intent.PERFORMANCE

    def test_an_order_is_still_refused_before_anything_else(self) -> None:
        assert classify("buy 100 NVDA", now=NOW).intent is Intent.ORDER

    def test_a_greeting_shaped_question_is_read_as_asking_about_the_desk(self) -> None:
        """Reversed deliberately on 2026-09-13.

        This previously asserted AMBIGUOUS on the grounds that "how's it going?" is a greeting.
        In a console whose only subject is the trading record there is nothing else it can be
        asking about, and refusing it is pedantry that costs exactly the fluency this interface
        is judged on. The idiomatic "it" refers to nothing, which is why the vague-reference rule
        no longer fires for intents that need no referent.
        """
        assert classify("how's it going?", now=NOW).intent is Intent.PERFORMANCE

    def test_every_capability_the_refusal_advertises_is_reachable(self) -> None:
        """The help text is a promise, and each item in it is executed here."""
        promised = {
            "performance": "how did we do?",
            "why a decision was taken": "why did you buy NVDA?",
            "why the desk stood aside": "why did we not trade?",
            "the evidence behind a decision": "what evidence did you see?",
            "calibration": "how calibrated are we?",
            "chain integrity": "is the chain intact?",
            "open positions": "what do we hold?",
            "session state": "is the market open?",
        }
        unreachable = [
            label for label, question in promised.items()
            if classify(question, now=NOW).intent
            in (Intent.UNKNOWN, Intent.AMBIGUOUS, Intent.UNSUPPORTED)
        ]
        assert not unreachable, f"advertised but unrecognised: {unreachable}"


class TestRiskControlAnswers:
    """Track 2 scores risk-control effectiveness. The console must answer it from the records."""

    @pytest.fixture(autouse=True)
    def _records(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Real risk records in the shape `argus.paper.runner` writes, read through the same
        ARGUS_DATA_DIR override the hosted console uses."""
        rows = [
            {"seq": 1, "symbol": "NVDAUSDT", "at": WEEKEND.isoformat(), "verdict": "buy",
             "intervened": True, "binding_constraint": "position_limit",
             "reason": "narrowed to the mandate cap", "quantity_before": "10",
             "quantity_after": "4", "model_changed_its_mind": True,
             "constitution_only_reduced": True},
            {"seq": 2, "symbol": "TSLAUSDT", "at": WEEKEND.isoformat(), "verdict": "no_trade",
             "intervened": False, "binding_constraint": "none",
             "reason": "no exposure proposed; nothing to narrow", "quantity_before": "0",
             "quantity_after": "0", "model_changed_its_mind": False,
             "constitution_only_reduced": True},
        ]
        (tmp_path / "risk_records.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        monkeypatch.setenv("ARGUS_DATA_DIR", str(tmp_path))

    def _answer(self, text: str, ledger: PaperLedger):  # type: ignore[no-untyped-def]
        from argus.lui.answer import answer as _answer_fn
        from argus.lui.question import classify as _classify
        return _answer_fn(ledger, _classify(text, now=NOW))

    def test_the_question_is_answered_rather_than_refused(
        self, weekend_ledger: PaperLedger
    ) -> None:
        got = self._answer("what did the risk layer block?", weekend_ledger)
        assert not got.refused

    def test_it_names_the_denominator_it_used(self, weekend_ledger: PaperLedger) -> None:
        """An intervention rate over every decision would make an untested layer look restrained,
        so the answer has to say which decisions could have been intervened on."""
        got = self._answer("what did the risk layer block?", weekend_ledger)
        assert any("offered a position it could reduce" in line for line in got.lines)

    def test_an_uninvoked_layer_is_called_untested_not_restrained(
        self, weekend_ledger: PaperLedger
    ) -> None:
        got = self._answer("how often did the constitution bind?", weekend_ledger)
        text = " ".join(got.lines)
        if "intervened zero times" in text:
            assert "UNTESTED" in text

    def test_it_reports_the_asymmetry_check_every_time(
        self, weekend_ledger: PaperLedger
    ) -> None:
        """The layer may only reduce. That is a claim, and this is where it is checked."""
        got = self._answer("did the risk layer ever increase a position?", weekend_ledger)
        assert any("Asymmetry:" in line for line in got.lines)

    def test_it_is_grounded_in_a_named_artefact(self, weekend_ledger: PaperLedger) -> None:
        got = self._answer("was anything vetoed?", weekend_ledger)
        assert got.is_grounded
        assert any("risk_records" in s.ref for s in got.sources)

    def test_it_carries_the_full_audit_as_data(self, weekend_ledger: PaperLedger) -> None:
        got = self._answer("which constraint was binding most?", weekend_ledger)
        assert "intervention_rate" in got.data and "by_constraint" in got.data


class TestIntegrityAnswers:
    """`is the log tamper-evident` is a suggested chip on the console's own landing page — the
    single most direct thing a judge can click to test this project's core trust claim. Added
    2026-09-22 alongside settlement seals: a tampered settled outcome, or one settled with no
    seal at all, must be named specifically, not folded into an unexplained "BROKEN" with no
    reason a reader can act on.
    """

    def _answer(self, text: str, ledger: PaperLedger):  # type: ignore[no-untyped-def]
        from argus.lui.answer import answer as _answer_fn
        from argus.lui.question import classify as _classify
        return _answer_fn(ledger, _classify(text, now=NOW))

    def test_an_intact_chain_explains_the_seal_not_just_the_exclusion(
        self, tmp_path: Path
    ) -> None:
        """The explanation used to say settlement fields are excluded from the hash and stop
        there, which — after this fix — read as though they were unprotected. It must also say
        they are sealed separately."""
        led = PaperLedger(path=tmp_path / "p.jsonl")
        got = self._answer("is the log tamper-evident?", led)
        text = " ".join(got.lines)
        assert "settlement-seal" in text or "seal" in text.lower()

    def test_a_tampered_settlement_is_named_specifically(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "p.jsonl")
        led.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis="t", invalidation=(),
            market_state_hash="m", approved_intent_hash="a",
            session_phase="weekend", hours_to_discovery=1.0, decided_at=WEEKEND,
        )
        led.settle(1, exit_price=Decimal("230"))
        lines = led.path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["net_pnl"] = "-99999.00"
        lines[0] = json.dumps(row, separators=(",", ":"))
        led.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        tampered = PaperLedger(path=led.path)
        got = self._answer("is the log tamper-evident?", tampered)
        text = " ".join(got.lines)
        assert "1" in text
        assert "mismatch" in text.lower() or "tamper" in text.lower()
        assert got.data["chain_intact"] == "False"

    def test_an_unsealed_settlement_is_named_specifically(self, tmp_path: Path) -> None:
        led = PaperLedger(path=tmp_path / "p.jsonl")
        led.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis="t", invalidation=(),
            market_state_hash="m", approved_intent_hash="a",
            session_phase="weekend", hours_to_discovery=1.0, decided_at=WEEKEND,
        )
        lines = led.path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[0])
        row["settled_at"] = WEEKEND.isoformat()
        row["net_pnl"] = "500.00"
        lines[0] = json.dumps(row, separators=(",", ":"))
        led.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        bypassed = PaperLedger(path=led.path)
        got = self._answer("is the log tamper-evident?", bypassed)
        text = " ".join(got.lines)
        assert "1" in text
        lowered = text.lower()
        assert "no settlement seal" in lowered or "unsealed" in lowered or "bypass" in lowered

    def test_the_head_hash_shown_is_the_real_head_not_a_stale_decision_hash(
        self, tmp_path: Path
    ) -> None:
        """The bug caught before it shipped: reading `entries[-1].content_hash` after a
        settlement shows the last DECISION's hash, which is stale the instant that decision's
        own seal is appended after it — the real chain head has already moved."""
        led = PaperLedger(path=tmp_path / "p.jsonl")
        led.record(
            symbol="NVDAUSDT", verdict="trade", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("220"),
            stated_confidence=0.7, thesis="t", invalidation=(),
            market_state_hash="m", approved_intent_hash="a",
            session_phase="weekend", hours_to_discovery=1.0, decided_at=WEEKEND,
        )
        led.settle(1, exit_price=Decimal("230"))
        got = self._answer("is the log tamper-evident?", led)
        real_head = led.verify()["head_hash"]
        assert any(real_head in line for line in got.lines)
        stale_head = led.entries[-1].content_hash
        assert stale_head != real_head  # the precondition this test actually exercises


class TestQueryStringRepair:
    """Chinese classified correctly in process and was refused on the hosted deployment.

    The code was identical; the runtime's path decoding was not. This is the standard latin-1
    round-trip repair, with the guard that keeps it from damaging text that was never broken.
    """

    def test_ascii_is_returned_untouched(self) -> None:
        from argus.lui.server import repair_mojibake

        assert repair_mojibake("what is the sharpe") == "what is the sharpe"

    def test_a_latin1_round_trip_is_undone(self) -> None:
        from argus.lui.server import repair_mojibake

        original = "为什么你没有交易 NVDA"
        broken = original.encode("utf-8").decode("latin-1")
        assert broken != original
        assert repair_mojibake(broken) == original

    def test_genuine_european_text_is_not_damaged(self) -> None:
        """The guard that matters. "café" survives a latin-1 round trip unchanged, so a naive
        repair would mangle every accented word in the corpus."""
        from argus.lui.server import repair_mojibake

        for text in ("café", "naïve", "Ölpreis", "señor"):
            assert repair_mojibake(text) == text, text

    def test_text_that_cannot_be_latin1_encoded_is_returned_as_is(self) -> None:
        from argus.lui.server import repair_mojibake

        assert repair_mojibake("夏普比率") == "夏普比率"

    def test_the_repaired_question_classifies(self) -> None:
        """End to end: the broken form must reach the right intent after repair."""
        from argus.lui.question import classify
        from argus.lui.server import repair_mojibake

        broken = "为什么你没有交易".encode().decode("latin-1")
        assert classify(broken, now=NOW).intent is Intent.UNKNOWN
        assert classify(repair_mojibake(broken), now=NOW).intent is Intent.ABSTENTION_WHY


class TestVoidedRowsAreNotCountedAsTrades:
    """**The console contradicted every other surface, and both numbers came from one chain.**

    Two live rows booked positions the Constitution had explicitly refused
    (`paper/corrections.py`). `ledger.performance`, `eval/performance`, the cockpit and the
    scorecard all exclude them. `answer_decision_list` counted `entry.verdict` raw, so the console
    answered *"493 no_trade, 2 trade"* while the submission and README said zero trades.

    That is precisely how a tamper-evident log launders a mistake: the chain verifies, so the
    wrong number looks authenticated.
    """

    def test_the_breakdown_excludes_voided_rows(self) -> None:
        from datetime import UTC, datetime

        from argus.lui.answer import answer
        from argus.lui.question import classify
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import LEDGER_PATH

        ledger = PaperLedger(path=LEDGER_PATH)
        if not ledger.entries:
            import pytest

            pytest.skip("no live ledger on this machine")
        result = answer(ledger, classify("show me every decision", now=datetime.now(UTC)))
        assert "trade" not in result.data.get("verdicts", {}), (
            "a position the risk layer refused must not be counted as a trade"
        )

    def test_the_excluded_rows_are_named_not_silently_dropped(self) -> None:
        """A total that quietly shrinks is the same defect as one that quietly includes."""
        from datetime import UTC, datetime

        from argus.lui.answer import answer
        from argus.lui.question import classify
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import LEDGER_PATH

        ledger = PaperLedger(path=LEDGER_PATH)
        if not ledger.entries:
            import pytest

            pytest.skip("no live ledger on this machine")
        result = answer(ledger, classify("show me every decision", now=datetime.now(UTC)))
        voided = result.data.get("voided", 0)
        if voided:
            assert any("corrections.py" in line for line in result.lines), (
                "excluded rows must be explained where they are excluded"
            )

    def test_the_voided_rows_remain_in_the_chain(self) -> None:
        """`corrections.py` refuses deletion; the rows stay and are listed, just not counted."""
        from argus.paper.corrections import VOIDED
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import LEDGER_PATH

        ledger = PaperLedger(path=LEDGER_PATH)
        if not ledger.entries:
            import pytest

            pytest.skip("no live ledger on this machine")
        # VOIDED holds VoidedEntry records, not bare sequence numbers — read from the module
        # rather than assumed. The first version of this test iterated it as ints and compared
        # objects against a set of numbers, which fails loudly; the dangerous version of that
        # mistake is the one that fails silently.
        present = {e.seq for e in ledger.entries}
        for voided in VOIDED:
            assert voided.seq in present, (
                f"seq {voided.seq} was deleted; corrections.py forbids that"
            )


class TestReviewReachesTheReviewEngine:
    """Track 3's Review & Self-Evolution. These questions used to be answered with the latest
    decision or its evidence — the review engine existed and nothing typed reached it."""

    @pytest.mark.parametrize("text", [
        "what bad decision patterns do you have",
        "what have you learned from your mistakes",
        "give me a checklist from your reviews",
        "review the desk's decisions",
        "run a post-mortem on the desk",
    ])
    def test_review_questions_are_review(self, text: str) -> None:
        assert classify(text, now=NOW).intent is Intent.REVIEW

    @pytest.mark.parametrize("text", [
        "why did you pass on NVDA", "what is the sharpe", "what did the risk layer block",
    ])
    def test_record_questions_are_untouched(self, text: str) -> None:
        assert classify(text, now=NOW).intent is not Intent.REVIEW

    def test_the_answer_is_computed_from_the_desks_own_notes(
        self, weekend_ledger: PaperLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json
        notes = [{"seq": i, "grounding": {"ungrounded": ["9.9%"]} if i % 2 else {}}
                 for i in range(1, 10)]
        (tmp_path / "desk_notes.jsonl").write_text(
            "\n".join(json.dumps(n) for n in notes), encoding="utf-8")
        monkeypatch.setenv("ARGUS_DATA_DIR", str(tmp_path))
        a = answer(weekend_ledger, classify("what bad decision patterns do you have", now=NOW))
        assert not a.refused and a.is_grounded
        assert "reviewed" in " ".join(a.lines)
        assert any("Checklist" in line for line in a.lines)

    def test_no_notes_is_a_refusal_not_an_empty_review(
        self, weekend_ledger: PaperLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ARGUS_DATA_DIR", str(tmp_path))
        a = answer(weekend_ledger, classify("give me a checklist", now=NOW))
        assert a.refused
