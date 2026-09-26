"""The register must be unable to claim evidence it does not have."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from argus.eval import standing
from argus.eval.standing import (
    GROUPWISE_CONDITIONS,
    ORDER,
    OWNED_CONDITIONS,
    REGISTER,
    TRANSITIONS,
    Capability,
    IllegalTransition,
    Proof,
    Report,
    StandingError,
    State,
    audit,
    check_transition,
    comparison_outcomes,
    groupwise_verdict,
    render_outcome,
    summary,
    transitions_since,
    write_report,
)


def _proof(condition: str = "implementation_complete") -> Proof:
    return Proof(condition=condition, how="built and tested", test="test_standing.py")


def _all_thirteen() -> tuple[Proof, ...]:
    return tuple(_proof(c) for c in OWNED_CONDITIONS)


class TestTheBarCannotBeLowered:
    def test_owned_without_all_thirteen_is_refused(self) -> None:
        with pytest.raises(StandingError, match="claims OWNED"):
            Capability(
                name="x", subtheme="t", module="argus/eval/standing.py",
                state=State.OWNED, baseline="qlib", proofs=_all_thirteen()[:12],
            )

    def test_twelve_of_thirteen_is_not_rounded_up(self) -> None:
        """The conjunction is the whole point: the missing condition is the informative one."""
        cap = Capability(
            name="x", subtheme="t", module="argus/eval/standing.py",
            state=State.IMPLEMENTED, baseline="qlib", proofs=_all_thirteen()[:12],
        )
        assert len(cap.conditions_met) == 12 and cap.state is not State.OWNED

    def test_all_thirteen_permits_owned(self) -> None:
        cap = Capability(
            name="x", subtheme="t", module="argus/eval/standing.py",
            state=State.OWNED, baseline="qlib", proofs=_all_thirteen(),
        )
        assert cap.state is State.OWNED and not cap.conditions_missing

    def test_a_fourteenth_condition_cannot_be_invented(self) -> None:
        with pytest.raises(StandingError, match="not one of the thirteen"):
            Proof(condition="looks_good", how="it does", test="test_standing.py")

    def test_a_proof_with_no_artefact_and_no_test_is_prose(self) -> None:
        with pytest.raises(StandingError, match="neither an artefact nor a test"):
            Proof(condition="ablation", how="we are confident it helps")

    def test_a_comparative_state_needs_a_named_baseline(self) -> None:
        with pytest.raises(StandingError, match="without naming the system"):
            Capability(
                name="x", subtheme="t", module="argus/eval/standing.py",
                state=State.TIED, baseline="",
            )

    def test_implemented_does_not_need_a_baseline(self) -> None:
        """A mechanism nobody else has built has nothing to be compared against yet."""
        cap = Capability(
            name="x", subtheme="t", module="argus/eval/standing.py",
            state=State.IMPLEMENTED, baseline="",
        )
        assert cap.state is State.IMPLEMENTED

    def test_there_are_exactly_four_states(self) -> None:
        assert len(list(State)) == 4 and len(ORDER) == 4

    def test_there_are_exactly_thirteen_conditions(self) -> None:
        assert len(OWNED_CONDITIONS) == 13
        assert len(set(OWNED_CONDITIONS)) == 13


class TestTheAuditReadsTheTree:
    def test_a_missing_artefact_is_a_finding(self) -> None:
        cap = Capability(
            name="ghost", subtheme="t", module="argus/eval/standing.py",
            state=State.IMPLEMENTED, baseline="",
            proofs=(Proof(condition="ablation", how="ran it",
                          artefact="data/this_does_not_exist.json"),),
        )
        report = audit((cap,))
        assert not report.clean
        assert "artefact" in report.findings[0].problem

    def test_a_missing_test_file_is_a_finding(self) -> None:
        cap = Capability(
            name="ghost", subtheme="t", module="argus/eval/standing.py",
            state=State.IMPLEMENTED, baseline="",
            proofs=(Proof(condition="ablation", how="ran it", test="test_nothing_here.py"),),
        )
        assert "test file" in audit((cap,)).findings[0].problem

    def test_a_named_test_that_is_not_in_the_file_is_a_finding(self) -> None:
        # Pointed at a *different* file on purpose. Naming a missing test inside this file would
        # pass trivially, because writing the name here puts the string in the file being read.
        cap = Capability(
            name="ghost", subtheme="t", module="argus/eval/standing.py",
            state=State.IMPLEMENTED, baseline="",
            proofs=(Proof(condition="ablation", how="ran it",
                          test="test_ablation.py::test_nothing_by_this_name_exists_over_there"),),
        )
        assert "not in the file" in audit((cap,)).findings[0].problem

    def test_a_named_test_that_is_in_the_file_passes(self) -> None:
        cap = Capability(
            name="real", subtheme="t", module="argus/eval/standing.py",
            state=State.IMPLEMENTED, baseline="",
            proofs=(Proof(condition="ablation", how="ran it",
                          test="test_standing.py::test_the_live_register_is_clean"),),
        )
        assert audit((cap,)).clean

    def test_a_missing_module_is_a_finding(self) -> None:
        cap = Capability(
            name="ghost", subtheme="t", module="argus/nowhere/absent.py",
            state=State.IMPLEMENTED, baseline="",
        )
        assert "module not found" in audit((cap,)).findings[0].problem

    def test_every_module_in_a_comma_separated_list_is_checked(self) -> None:
        cap = Capability(
            name="ghost", subtheme="t",
            module="argus/eval/standing.py, argus/nowhere/absent.py",
            state=State.IMPLEMENTED, baseline="",
        )
        assert len(audit((cap,)).findings) == 1


class TestTheLiveRegisterIsHonest:
    def test_only_named_capabilities_are_owned_and_each_genuinely_clears_all_thirteen(
        self,
    ) -> None:
        """The governing rule's stated baseline used to be "nothing claims OWNED" — checked
        first, per this test's own original instruction, before changing it: 2026-09-15,
        "Session-aware execution that refuses to solve through a boundary" (t2-execution) became
        the first capability to clear all thirteen. The same day, "Per-profile mandate that
        changes the verdict" (t3-personalisation) became the second — vendored and actually ran
        Vibe-Trading's real check_mandate() (byte-verified against its upstream commit), a
        19,440-scenario grid stressing every dimension either system checks, every one of 10
        checks independently confirmed load-bearing via a tripped/cleared pair, and a
        `no_specialist_capability_superior` claim scoped in writing to what Mandate actually
        covers rather than claimed whole. The same day, "Cross-sectional factor evaluation"
        (t1-alphafactory) became the third — vendored and actually ran qlib's real CSRankNorm
        (byte-verified against its upstream commit), 61 real panels (hand-designed + a
        deterministic grid) all agreeing exactly on relative order with ARGUS's own
        CrossRank.combine(), both of crossrank's own design choices (midrank ties, the n<=1
        neutral guard) confirmed independently load-bearing, and a real, run-verified edge case
        found in qlib's own code (a single-instrument group produces an extreme value, not a
        neutral one — ARGUS's crossrank guards against exactly this, twice over). The same day
        again, "Overfitting gates that raise instead of returning NaN" (t1-validation) became the
        fourth — vendored and actually ran vectorbt's real `deflated_sharpe_ratio()` (byte-verified
        against its upstream commit), 1,220 real cases agreeing EXACTLY with ARGUS's own
        `deflated_sharpe` (max abs diff 0.0), and a genuine bug this comparison found in ARGUS's
        OWN code: the pre-existing `variance_of_trials < 0` guard did not catch NaN
        (`float('nan') < 0` is `False`), sharing the exact silent-failure shape vectorbt's own gate
        has — found, fixed, and re-verified the same session (`backtest/metrics.py`). The same day
        a fifth time, "Episodic memory across decisions" (t2-agentic) became the fifth — vendored
        and actually ran TradingAgents' real `TradingMemoryLog` (byte-verified against its
        upstream commit), both systems' point-in-time guard run across the exact resolution
        boundary and confirmed correct on both sides, a genuine structural divergence found by
        running both (ARGUS's strict `<` on datetime instants vs. TradingAgents' `<=` on date
        strings), and a self-caught bug in this comparison's OWN ablation test (read the
        monkeypatched result after the patch was already restored, silently got the real answer
        back) found and fixed before being counted as evidence. The same day a sixth time,
        "Self-evolving review rules" (t3-review) became the sixth — reusing the SAME vendored
        TradingMemoryLog (no new vendoring needed), showing its real code re-injects a reflection
        about a decision with a demonstrably WRONG outcome (a real -15% return) identically to
        one about a right call, with zero precision/track-record concept anywhere in the whole
        TradingAgents repository (grepped directly); ARGUS's own `desk.review.evaluate()` run
        across six designed fixtures, each landing on its intended one of seven named lifecycle
        states — including `Status.EARNING`, a state this module's own docstring already records
        as a real self-found bug once unreachable, re-verified reachable here rather than assumed
        fixed. The same day a seventh time, "Pre-registered trading protocol, hash-committed"
        (t2-agentic) became the seventh — vendored and actually ran serenity-guardrails' real
        `ChainedJournal` (byte-verified against its upstream commit), and found a genuine,
        reproducible, platform-specific bug by RUNNING the real code rather than reading it: on
        Windows, `append()` writes via Python's default text-mode file I/O, which silently
        converts every `\n` to `\r\n` on disk, while `verify()` re-hashes raw binary-mode-read
        line bytes stripped only of `\n` — so serenity's own `ChainedJournal`, written and
        verified by its own unmodified code with zero tampering, reports its own head anchor as
        mismatched at every entry count swept from 1 through 50. ARGUS's own `PaperLedger` was
        confirmed structurally immune by construction (it hashes re-serialized PARSED dataclass
        fields, never raw disk bytes) and empirically immune by running it: clean at every swept
        N, clean on the actual live production ledger (248 real entries, written before this
        comparison existed), and in agreement with serenity on every property both systems
        actually implement (genuine tampering, tail truncation). The same day an eighth time,
        "Perception layer: what the desk can see" (t3-datasources) became the eighth — vendored
        and actually ran TradingAgents' real `route_to_vendor()` (byte-verified against its
        upstream commit: `interface.py`, `errors.py`, `config.py`, `default_config.py`), and found
        a genuine structural difference by running both real dispatch layers on identical failure
        scenarios: TradingAgents' router RAISES — can abort the whole perception cycle — when
        every configured vendor for a "core" category fails with a real error, while ARGUS's own
        `gather()` never raises for any source's failure, core or opt-in, confirmed across every
        one of TradingAgents' own real category/method pairs swept (not a sample), an ablation
        showing `gather()`'s caught-exception scoping is load-bearing (an uncaught type genuinely
        propagates rather than being silently swallowed), reproducibility confirmed on both real
        systems, and an out-of-sample check against the real, live-growing desk-notes log this
        session's own decision cycles wrote (194 real cycles, 5 distinct sources). The next day a
        ninth time, "Risk layer proved by domain sweep" (t2-riskcontrol) became the ninth — not
        by vendoring anything new, but by finally closing the condition its own blockers had left
        genuinely open since 2026-09-15: whether freqtrade's near-constant MaxDrawdown lock was a
        defensible "conservative by design" choice a different threshold could fix, or a
        structural measurement gap no threshold could fix. Settled by measuring, not arguing:
        ARGUS's ladder thresholds directly on the SAME peak-to-trough equity quantity
        `true_drawdown_pct` IS (verified empirically by `argus_measures_ground_truth_directly`
        reconstructing `argus_locked` from only ground-truth fields and checking it against the
        real recorded value — this caught a real gap in the first version of that reconstruction,
        which forgot `consecutive_losses` was also a real trigger, fixed before being trusted),
        while freqtrade's `MaxDrawdown` estimates it from a windowed sum of closed trades'
        cumulative returns — a real, measured proxy whose Pearson correlation with the ground
        truth is only 0.2998 on a fresh live 320-checkpoint pull. Freqtrade's own real code swept
        across ten threshold values tops out at 21.99% precision at full recall; ARGUS's real,
        already-computed decision scores 63.27% precision at the identical 100% recall,
        Pareto-dominating every one of freqtrade's ten swept thresholds at once. Each addition was
        verified independently here rather than trusted from the register's own `state` field
        alone — `Capability.__post_init__` already refuses to construct an OWNED entry short of
        thirteen, and this test re-derives the same answer a second, independent way. A capability
        appearing here that is NOT in this exact set is the real regression this test exists to
        catch — add it to the set only after checking its thirteen the same way these nine were
        checked, never to make a test pass. The same day a tenth time, "Market sentiment"
        (t2-sentiment) became the tenth — no code vendored (a HuggingFace Hub inference model,
        not a source file, so `transformers.pipeline("ProsusAI/finbert")` was run for real
        exactly as its own README documents, nothing redistributed) — and scoped precisely to
        the one axis SentimentAnalyst's own docstring ever claimed: not classification accuracy
        against finBERT (explicitly disclaimed), but a coordination/duplicate-source manipulation
        defense finBERT's per-sentence-only architecture cannot have by construction. A real
        coordinated-posting scenario (symbol-matched, fixed after an earlier draft mismatched the
        evaluated symbol against the claim's own subject) was run on both real systems: finBERT's
        naive aggregate scaled with repetition on every narrative tested; ARGUS's real
        `SentimentAnalyst` never let repetition alone flip a non-actionable read to actionable, on
        any narrative. Two real ablations, not one, because the first taught something the module
        did not start out claiming: removing only the single named "five accounts" sentence
        changed nothing — the model's broader reasoning already reached the same conclusion
        unprompted, a real negative result reported as found rather than discarded. A second,
        coarser ablation stripping the ENTIRE source-independence framing to a bare classifier DID
        show the real defense degrade — directional confidence measurably rose under coordinated
        repetition where the real analyst's stayed at zero — confirming the defense is a property
        of the prompt's whole framing, not any one sentence. The same day an eleventh time,
        "Deliberation priced as a trading cost" (t2-agentic) became the eleventh — the original
        88-repo corpus survey found no baseline to compare against, so a fresh, targeted search
        (2026-09-16) found the one real exception: `HaoKang-Timmy/LatencySensitiveBench` (arxiv
        2505.19481), read in full. No license to vendor under, so an independent clean-room
        reimplementation of its real linear-decay idea was built and verified against reference
        output vectors computed by running their own real, unmodified method once locally — six
        (input, output) pairs matched exactly. The decisive finding: on ARGUS's own three real,
        bake-off-measured thinking-budget delays (3s/8s/40s), the baseline's real formula reports
        the IDENTICAL cost for all three because each exceeds its own stated 1.5s decay-window
        cap, while ARGUS's real sqrt(delay) model reports three distinct, correctly ordered costs
        on the identical delays — confirmed by running both, and an ablation confirming the
        collapse is caused by the SPECIFIC 1.5s value, not the existence of a cap (widening it to
        60s restores distinguishability). The same day a twelfth time, "Abstention scored as a
        decision" (t2-riskcontrol) became the twelfth — a fresh search found
        `insaneamogh/AutonomousTradeAgents`'s real "Ghost P&L" ledger, unlicensed, so an
        independent clean-room reimplementation was verified against reference output computed
        by running their own real, unmodified `_bucket_from_rows` once locally on their OWN
        documented production incident (a vetoed bucket holding $30,788 avoided losses and
        $32,967 blocked gains, netting +$2,179). Their real headline `saved_usd` computation
        (`max(0.0, -net)`) produces exactly $0.00 on that exact real scenario — not a fixed
        historical bug, still live in their current `build_ghost_summary` — while ARGUS's real
        `abstention_quality()` reports the true, unfloored net on the structurally identical
        scenario; a sweep across twelve loss/upside ratios confirmed the defect fires precisely
        where the net crosses near zero and nowhere else, and an ablation isolated the floor
        itself (not the underlying aggregation) as the exact mechanism. Each addition was
        verified independently here rather than trusted from the register's own `state` field
        alone — `Capability.__post_init__` already refuses to construct an OWNED entry short of
        thirteen, and this test re-derives the same answer a second, independent way. A
        capability appearing here that is NOT in this exact set is the real regression this test
        exists to catch — add it to the set only after checking its thirteen the same way these
        twenty-one were checked, never to make a test pass. The same day a twenty-second time,
        "Holiday-aware closed-session pricing vs. an unconditional pre-holiday long bias"
        (t1-afterhours) became the twenty-second — closing the last of the nine sub-theme gaps
        an earlier research fork found: `research/gap_study.py`'s own `study()` had always
        constructed `DualClock()` with no holidays argument, so the real `SessionPhase.HOLIDAY`
        branch `_closed_sessions()` already classifies for had never once fired in any real run.
        Wiring in Lean's own real US equity holiday calendar (293 real dates, verbatim from its
        real `market-hours-database.json`) unlocked 24 real holiday sessions on live data for the
        first time, and running QuantConnect/Tutorials' real, unconditional Pre-Holiday Effect
        decision snippet against them found a real 54.2% blind-long win rate, gross and net of
        the real 12bps fee — close to, not decisively above, the 50% coin flip. The real
        reference's own two published code blocks disagree with each other (a filtered
        `public_holidays` variable is computed and then never used), found by reading them
        together. Each addition was verified independently here rather than trusted from the
        register's own `state` field alone — `Capability.__post_init__` already refuses to
        construct an OWNED entry short of thirteen, and this test re-derives the same answer a
        second, independent way. A capability appearing here that is NOT in this exact set is the
        real regression this test exists to catch — add it to the set only after checking its
        thirteen the same way these twenty-two were checked, never to make a test pass. A fresh,
        full 15-named-sub-theme survey against the handbook (not the earlier fork's partial
        9-gap list) the same day found one real capability mislabeled ("Session-aware execution
        that refuses to solve through a boundary" was `t2-execution`; its own module docstring
        names Track 3's Execution Assistance sub-theme explicitly, corrected to `t3-execution`,
        `audit()` re-confirmed unaffected) and one real, confirmed-open gap: Track 1's "rToken
        Factor Strategies" had no capability testing its actual named claim — the two entries
        filed under `t1-alphafactory` are both real, both OWNED, and both about generic factor-
        engine infrastructure (qlib expression-injection safety, qlib CSRankNorm correctness),
        neither testing whether a factor behaves differently on the rToken's own market series
        than on its native-stock reference. Closed the same day a twenty-third time:
        "rToken factor divergence vs. Alphalens' real Information Coefficient" (`t1-rtokenfactor`)
        — WorldQuant's real, published Alpha#101 formula #23, run via ARGUS's own grammar on real
        MARKET and real INDEX candles for the full real rToken universe, scored by Alphalens-
        reloaded's real, vendored Information Coefficient. The decisive finding surfaced in the
        real ABLATION, not the base case (withdrawn 2026-09-26: the artefact's own intervals
        include zero on both arms, so what follows was never true of the published run):
        stripping Alpha 23's own real conditional gate
        (`mean(high,20) < high`) reveals the bare, unconditional `delta(high,2)` carries a real,
        bootstrap-confirmed market-vs-index divergence whose 95% CI excludes zero, while the
        real, published, GATED formula erases it into noise — confirmed by both ARGUS's own
        dependency-aware stationary bootstrap and Alphalens' own real, naive `ttest_1samp`
        agreeing the gated difference is not significant. Two real bugs were found and fixed
        building this, both by running the real thing on real data, not by inspection: Alphalens'
        own real `factor_information_coefficient` silently collapses genuinely irregular real
        timestamps onto pandas' inferred fallback grid via `ic.asfreq(None)` rather than raising,
        so the two independently-collapsed real panels' indices were not guaranteed to align,
        producing a real `KeyError` on a smaller real OOS window (fixed by pairing on raw
        timestamp values via plain dict lookups instead of `pandas.Index.intersection`/`.loc[]`);
        and this module's own first-draft bootstrap guard used a threshold of 8 observations,
        under `backtest.dependence.optimal_block_length`'s own real minimum of 30, crashing with
        an uncaught `MetricError` on a real 29-reading OOS half-window (fixed by importing and
        checking against that same real constant). A capability appearing here that is NOT in
        this exact set is the real regression this test exists to catch — add it to the set only
        after checking its thirteen the same way these twenty-three were checked, never to make a
        test pass. Closed the same day a twenty-fourth time: "Numeric decision grounding vs.
        TradingAgents' real, unchecked TraderProposal" (`t2-explainability`) — TradingAgents' real
        trader.py, schemas.py, portfolio_manager.py, all three risk_mgmt/*.py debators,
        reporting.py, and trading_graph.py read in full first: the Trader's own prompt asks the
        model to ground prices in the market report, but nothing downstream — not the schema's
        own real `field_validator`, not any real consumer of `trader_investment_plan` — ever
        checks that it did. Vendored the real `TraderProposal` (byte-verified against the same
        commit already pinned for `tradingagents_memory.py`/`tradingagents_rating.py`) and ran it
        for real: its own unmodified validator accepts an `entry_price` of 999,999.0, a negative
        price, with zero relationship to any real fact, because `_coerce_optional_float` only
        normalises string FORMAT (a placeholder, a trailing "%", a currency symbol), never the
        VALUE. Same-input comparison against ARGUS's own, already-existing `agents.grounding.
        check` on real, live Bitget prices across 3 real symbols and 4 fabrication magnitudes: 0
        of 12 caught by the real rival, 12 of 12 caught by ARGUS, with a positive control
        confirming the real current price still resolves cleanly on the same live data — not a
        blanket flag. The ablation isolated the exact mechanism rather than assuming it: a figure
        just inside `grounding.py`'s own `TOLERANCE` constant resolves, the identical figure moved
        just past it on the same real fact does not. A capability appearing here that is NOT in
        this exact set is the real regression this test exists to catch — add it to the set only
        after checking its thirteen the same way these twenty-four were checked, never to make a
        test pass. Closed the same day a twenty-fifth time: "Structured filing extraction vs.
        FinanceBench's real, published LLM measurement" (`t3-infoextract`) — FinanceBench's real
        `evaluation_playground.ipynb` read in full first (its own real six-mode retrieval
        taxonomy, and its own documented admission that no automated scorer ships with the repo).
        No LICENSE file on the repo, so nothing was vendored: recomputed real summary statistics
        directly from FinanceBench's own real `results/*.jsonl` transcripts for its 50-question
        `metrics-generated` subset — even GPT-4 under best-case `oracle` retrieval scores 46/50
        (92%), while every realistic condition collapses (`sharedStore` 6/50 with 39 refusals,
        `inContext` 6/50 correct but 20/50 CONFIDENTLY WRONG). Ran ARGUS's own, already-existing
        `FundamentalsSource` (real, keyless, live SEC XBRL) on a freshly-designed set of 6 real
        (ticker, concept) cases — not FinanceBench's own unlicensed question set — resolving 5 of
        6, with the one non-resolution (GOOGL's real filings genuinely carry no `GrossProfit`
        XBRL tag) reported as a named reason rather than a fabrication. The decisive finding
        surfaced in the ablation, run on real live data: NVDA's real Q2 FY2009 net income carries
        two real rows under the identical fiscal end-date — a true ~91-day quarterly duration
        (-$120,929,000) and a sign-flipped ~181-day cumulative one (+$55,876,000) — a naive fetch
        with no quarterly filter is genuinely ambiguous between them, while ARGUS's real
        `quarterly_only=True` path resolves to exactly the correct quarterly value. A capability
        appearing here that is NOT in this exact set is the real regression this test exists to
        catch — add it to the set only after checking its thirteen the same way these twenty-five
        were checked, never to make a test pass. Restored to OWNED the same day a twenty-sixth
        time: "Self-evolving review rules" (`t3-review`), demoted on 2026-09-20 when `verify()`
        began opening artefacts instead of trusting filenames — `out_of_sample_test` was already
        fixed earlier (`data/review_oos.json`, a real 216/217 held-out split); the one condition
        still genuinely UNPROVEN was `failure_cases_documented`: `data/review_report.json` carried
        the rejected rules' own numbers under a `rejected` key `SIGNATURES` does not match, so the
        finding "none of the five standing rules earned a place" was true but not machine-checkable.
        Fixed by adding `ReviewReport.failure_cases` to `desk/review.py` — a real, structured
        summary of every rejected rule's own measured failure mode (rule, status, precision, why),
        additive alongside the existing `rejected` field rather than a rename of it, since other
        consumers of `rejected` were grepped for and found unrelated to this capability. Confirmed
        by regenerating the artefact against the live record (now 481 decisions, up from a stale
        40-decision snapshot) and re-running `standing.py`: this capability's `conditions_missing`
        is empty, all thirteen VERIFIED or ATTESTED, register-wide UNPROVEN count fell from 4 to 3.
        A capability appearing here that is NOT in this exact set is the real regression this test
        exists to catch — add it to the set only after checking its thirteen the same way these
        twenty-six were checked, never to make a test pass. Closed the same day a twenty-seventh
        time: "Point-in-time correctness vs. OpenBB's real, ungated live-API agent"
        (`t3-workbench`) — OpenBB's real `openbb-agents` (no LICENSE file, so grepped not
        vendored) and OpenBB Platform's real 32-provider directory both read in full. Exhaustive
        grep of the real agent source (`agent.py`/`tools.py`/`chains.py`/`prompts.py`) for
        `as_of`/point-in-time/look-ahead/historical_date/backtest/cutoff: zero matches. ARGUS's
        real `as_of` gating (already existing on every source, reused here) verified on real live
        SEC XBRL data at the sharpest possible boundary: NVDA's real EPS fact filed 2026-08-26 is
        visible when the cutoff is midnight of that day and withheld entirely when the cutoff is
        one day earlier — the same real fact, one day apart, a genuinely sharp transition, not
        assumed. The honest cost of this win, stated rather than hidden: OpenBB genuinely wins on
        raw source breadth (32 real providers vs ARGUS's 12 live-verified), reported in the same
        capability rather than omitted. A capability appearing here that is NOT in this exact set
        is the real regression this test exists to catch — add it to the set only after checking
        its thirteen the same way these twenty-seven were checked, never to make a test pass.
        Closed the same day a twenty-eighth time: "Decision-latency pricing vs. hftbacktest's
        real, network-only LatencyModel" (`t3-execassist`) — this project's own prior research
        note (`research/subthemes/t3-5-execution-assistance.md`) already named hftbacktest as
        the authority on latency modelling among 8 real systems source-read; its real
        `backtest/models/latency.rs` (commit 5f3ec40b2afb764e0fea112f941ed85523ef4e88, MIT) read
        in full and grepped exhaustively for decision/deliberation/LLM/reasoning latency: zero
        matches. Its real `LatencyModel` trait carries exactly `entry()`/`response()`; its real
        `ConstantLatency::new()` takes exactly two parameters — confirmed by literal signature
        match, not just absence of a keyword. Not run (a compiled Rust crate via PyO3 with no
        prebuilt wheel on this machine), read instead, which is sufficient to establish the
        structural absence. ARGUS's own real `thinking_budget_cost_bps` — already existing,
        already live-used by `agents/meta_pm.py` for real desk decisions — was run on ARGUS's
        own three real bake-off-measured thinking budgets (3s/8s/40s) against a real, freshly-
        fetched live VIX reading, producing three strictly increasing real costs whose ratio
        matches the formula's own sqrt(t) scaling to within 1%, verified numerically rather than
        assumed from reading the formula. A capability appearing here that is NOT in this exact
        set is the real regression this test exists to catch — add it to the set only after
        checking its thirteen the same way these twenty-eight were checked, never to make a test
        pass. Restored to OWNED the same day three more times, all found by the same AUDIT
        sweep: rather than trusting `conditions_missing` (which only checks a `Proof` object
        exists per condition, never that it actually verifies), every remaining IMPLEMENTED
        capability's thirteen proofs were independently re-run through the real `verify()`
        function. Three — "Overfitting gates that raise instead of returning NaN"
        (`t1-validation`), "Risk layer proved by domain sweep" (`t2-riskcontrol`), and
        "Session-aware execution that refuses to solve through a boundary" (`t3-execution`) —
        each still carried a `state=State.IMPLEMENTED` line and a demotion comment dated
        2026-09-20 naming one specific unproven condition, but an earlier pass had already fixed
        each named artefact (adding `failure_cases` to `overfit_gates.json`, wiring real evidence
        into `risk_layer_comparison.json`, filling in the ablation/same_input_comparison
        artefacts for the execution guard) without ever flipping the `state` line back — the
        exact same class of miss "Self-evolving review rules" turned out to be two iterations
        earlier, just undiscovered until this sweep checked every remaining entry rather than
        one at a time as each ledger row happened to need it. All three: thirteen VERIFIED or
        ATTESTED, zero UNPROVEN, confirmed by direct `verify()` calls before editing anything. A
        fourth, "Queue-position modelling ported from hftbacktest and measured against it"
        (`t2-execution`), was checked the same way and correctly NOT promoted — its own
        `blockers` already document in detail why `no_specialist_capability_superior` is
        genuinely unestablished (hftbacktest's real ground truth is paid, real CME
        market-by-order data; ARGUS's is a constructed simulator, and the gap cannot be closed
        with crypto data by either party) — a real, substantive, already-honest blocker, not a
        stale comment, and left exactly as it was. Restored to OWNED a thirty-second time,
        2026-09-22: 'Market sentiment' (`t2-sentiment`) carried the same conditions_missing-vs-
        verify() gap as the three above — adversarial_test and out_of_sample_test both claimed,
        both genuinely true (the coordinated-posting scenario IS the adversarial test; the
        reproducibility re-run on a fresh Qwen call IS an out-of-sample check), neither exposed
        under vocabulary the verifier's SIGNATURES dict looks for. Fixed in
        `sentiment_comparison.py`'s own source (two derived keys added to its report dict, naming
        facts already true rather than inventing new ones), the fix verified by simulation against
        the existing artefact BEFORE spending anything, then the real comparison re-run for real
        with real Qwen credits (8 calls, about 170s, the project owner's approval) rather than
        hand-editing the artefact — this project never hand-edits an artefact to match a claim.
        The fresh run reconfirms the original finding on both real narratives: finBERT's naive
        aggregate scales with repetition every time, ARGUS's real analyst never does,
        reproducibility holds (signal_stable=True on a genuine re-sample). Restored to OWNED a
        thirty-third time, same day, at zero cost: "Deliberation priced as a trading cost"
        (`t2-agentic`) had the identical conditions_missing-vs-verify() gap — reproducibility_proven
        claimed, artefact silent on it — but unlike the others, this module makes no LLM call at
        all (an earlier note in project memory had wrongly filed it as Qwen-key-blocked). Both
        compared cost functions are pure and deterministic, so `check_reproducibility()` runs each
        comparison twice and JSON-compares them for byte identity — reproducibility was not a
        plausible assumption to leave unproven, it was simply never run. A capability appearing
        here that is NOT in
        this exact set is the real regression this test exists to catch — add it to the set only
        after checking its thirteen the same way these thirty-three were checked, never to make a
        test pass."""
        # 2026-09-24: seven grades withdrawn because the rival they beat does not lead the
        # sub-theme (the rival review of 2026-09-24), and sentiment and factor
        # discovery narrowed to what was proven. Each withdrawn row stays in the register as
        # IMPLEMENTED with the reason as its first blocker.
        # 2026-09-25 (S18): twelve more withdrawn by the groupwise gate. Their statistical or
        # out-of-sample proof rests on designed cases, a parameter sweep or an aggregate whose rows
        # were not kept, so no breakdown by symbol, date or regime could run on it
        # (data/groupwise_audit.json). Each carries the reason and the route back as its first
        # blocker; `TestDemotionsCarryTheirRouteBack` pins that.
        # 2026-09-26: breadth rotation to TIED. A general validator (pandera with pydantic)
        # configured to the same contract handles the same 36 cases it does
        # (data/general_rotation_comparison.json); beating pytaa was beating a weaker rival.
        # 2026-09-26: rToken factor divergence to IMPLEMENTED. Its ablation proof said the ungated
        # divergence's CI excludes zero; its own artefact has it at [-0.0387, +0.0304].
        owned_names = {c.name for c in audit().owned}
        assert owned_names == {
            "Cross-sectional factor evaluation",
            "Sentiment integrity: resistance to coordinated posting, vs. finBERT",
            "Numeric decision grounding vs. TradingAgents' real, unchecked TraderProposal",
            "Per-profile mandate that changes the verdict",
            "Refusal-first earnings surprise ranking vs. a silently-exploding factor",
            "Risk layer proved by domain sweep",
        }
        for cap in audit().owned:
            assert cap.conditions_missing == (), cap.name

    def test_portfolio_allocation_moved_from_lost_to_tied(self) -> None:
        """Pinned so a future edit cannot silently move this back to LOST (or claim OWNED, which
        would overstate a tie as a win): ARGUS's own `nco_weights` was built from Riskfolio's
        real NCO source and verified 2026-09-22 to match it exactly (8.203bps both, ratio 1.0000)
        on the real walk-forward book — a genuine tie, not a win, and TIED is what this
        register's own vocabulary calls that."""
        cap = next(
            c for c in REGISTER
            if c.name == "Portfolio allocation, measured against Riskfolio-Lib's NCO"
        )
        assert cap.state is State.TIED

    def test_lui_routing_moved_from_lost_to_tied(self) -> None:
        """Pinned so a future edit cannot silently move this back to LOST or claim OWNED.

        The 2026-09-22 rebuild (logistic head -> linear SVM head, re-derived threshold) closed
        the sealed-accuracy gap from significant (p=0.0014) to not-significant (p=0.0576) against
        this register's own p<0.05 bar. Rasa is still numerically ahead (81.91% vs 77.82%) — this
        is 'not proven to be a loss', a real but fragile tie, not a claim that ARGUS matches or
        beats Rasa's accuracy. TIED, not OWNED, is what this register's vocabulary calls that."""
        cap = next(
            c for c in REGISTER
            if c.name == "LUI intent routing, measured against Rasa's real DIET classifier"
        )
        assert cap.state is State.TIED

    def test_regime_boundary_detection_moved_from_lost_to_tied(self) -> None:
        """Pinned so a future edit cannot silently move this back to LOST or claim OWNED.

        FLUSS, ARGUS's original segmenter, still loses decisively to ruptures (F1 0.443 vs
        0.975, p=1.5e-25) — unchanged and not softened. What moved the CAPABILITY to TIED is a
        second, different ARGUS tool (`desk/regime.py::exact_partition`, an exact L2 dynamic
        program read from ruptures' own real source) that ties ruptures on the same 100-trial
        synthetic ground truth (1 win/0 losses/99 ties, sign-test p=1.0 — far from significant,
        so this is reported as a tie from one discordant trial, not a proven win)."""
        cap = next(
            c for c in REGISTER
            if c.name == "Regime-boundary detection, measured against stumpy FLUSS and ruptures"
        )
        assert cap.state is State.TIED

    def test_every_capability_names_where_it_lives(self) -> None:
        for cap in REGISTER:
            assert cap.module, cap.name

    def test_every_capability_names_at_least_one_gap(self) -> None:
        for cap in REGISTER:
            assert cap.conditions_missing or cap.blockers, cap.name

    def test_sentiment_records_the_real_feed_fix_not_the_stale_demotion(self) -> None:
        """This used to assert the blockers still called the analyst 'DEMOTED' and cited the
        original 93%-empty figure as a live fact. It no longer does, correctly: re-checking that
        finding on 2026-09-16 found it stale (real Twitter/Reddit access opened up on this
        machine), and `market.evidence.TwitterSource`/`RedditSource` were built, wired into the
        live cycle, and swept at 0 of 12 empty on both — a genuine fix, not a claim to leave
        standing unexamined. The blockers still quote the original 93% figure as history, and the
        real, still-open limitation (no in-house scoring model) is still disclosed."""
        cap = next(c for c in REGISTER if c.name.startswith("Sentiment integrity"))
        assert any("93%" in b for b in cap.blockers)
        assert any("TwitterSource" in b for b in cap.blockers)
        assert any("RedditSource" in b for b in cap.blockers)
        assert not any("DEMOTED" in b for b in cap.blockers)

    def test_sentiment_carries_the_experiment_that_would_promote_it(self) -> None:
        """A demotion without a route back is a deletion with extra steps."""
        cap = next(c for c in REGISTER if c.name.startswith("Sentiment integrity"))
        assert "ablation" in cap.note and "30" in cap.note

    def test_the_review_checklist_is_recorded_as_not_yet_earned(self) -> None:
        """Restored to OWNED 2026-09-22 (all thirteen conditions now VERIFIED/ATTESTED), but the
        underlying fact this test protects is unchanged: the checklist itself is still honestly
        empty, and OWNED does not launder that away."""
        cap = next(c for c in REGISTER if c.name == "Self-evolving review rules")
        assert cap.state is State.IMPLEMENTED
        assert cap.blockers[0].startswith("RE-GRADED 2026-09-24 from OWNED to IMPLEMENTED")
        assert any("no rule has earned promotion" in b for b in cap.blockers)

    def test_the_grammar_records_its_remaining_breadth_gap(self) -> None:
        """The cross-sectional gap was closed and the register moved on to the next one. A
        register that keeps a solved blocker is as dishonest as one that hides an open one."""
        cap = next(c for c in REGISTER if "grammar" in c.name)
        assert any("open" in b and "101" in b for b in cap.blockers)
        assert not any("46 of the 101" in b for b in cap.blockers)

    def test_the_cross_sectional_result_is_recorded_as_negative(self) -> None:
        """Closing a capability gap is not the same as the capability paying off, and the
        register must not let the first read as the second."""
        cap = next(c for c in REGISTER if c.name == "Cross-sectional factor evaluation")
        assert any("NOT ONE" in b or "0 survive" in b for b in cap.blockers)
        assert any("1,336 backtests" in b for b in cap.blockers)

    def test_capability_names_are_unique(self) -> None:
        names = [c.name for c in REGISTER]
        assert len(names) == len(set(names))

    def test_the_report_names_the_owned_capability_explicitly_not_by_omission(self) -> None:
        """`render()` used to simply omit the "Nothing is OWNED" paragraph once something
        cleared all thirteen, leaving a genuinely earned claim stated nowhere as clearly as the
        "nothing yet" case had been — fixed the same day this first became true. The closing
        paragraph names every OWNED capability, and after the 2026-09-24 re-grade none of the
        withdrawn ones."""
        rendered = audit().render()
        assert "Nothing is OWNED" not in rendered
        assert f"{len(audit().earned)} capability(ies) OWNED" in rendered
        closing = rendered.split(" capability(ies) OWNED", 1)[1]
        for cap in audit().owned:
            assert cap.name in closing, cap.name
        for name in (
            "Clustering-corrected, base-rate-honest event significance vs. a fixed-null test",
            "Self-evolving review rules",
            "Structured filing extraction vs. FinanceBench's real, published LLM measurement",
        ):
            assert name in rendered and name not in closing, name

    def test_the_summary_is_one_line(self) -> None:
        assert "\n" not in summary(audit())

    def test_it_serialises_whole(self) -> None:
        got = audit().as_dict()
        assert got["by_state"]["owned"] == len(audit().earned)
        assert len(got["owned_conditions"]) == 13
        assert all("conditions_missing" in c for c in got["capabilities"])


class TestNextState:
    def test_the_next_state_above_implemented_is_owned(self) -> None:
        cap = Capability(name="x", subtheme="t", module="argus/eval/standing.py",
                         state=State.IMPLEMENTED, baseline="")
        assert cap.next_state is State.OWNED

    def test_owned_has_nothing_above_it(self) -> None:
        cap = Capability(name="x", subtheme="t", module="argus/eval/standing.py",
                         state=State.OWNED, baseline="q", proofs=_all_thirteen())
        assert cap.next_state is None

    def test_states_are_never_averaged(self) -> None:
        """by_state counts; it does not produce a mean. A 'mostly owned' register is the failure
        this module is built to prevent."""
        report = Report(capabilities=REGISTER)
        assert isinstance(report.by_state, dict)
        assert sum(report.by_state.values()) == len(REGISTER)


def test_the_live_register_does_not_overstate() -> None:
    """**"Clean" was redefined on 2026-09-20, and the redefinition is the point.**

    It used to mean "every artefact, test and module the register names really exists" — presence,
    which is what an adversarial review correctly called a rubber stamp. `verify()` now opens the
    artefacts, and twelve conditions turned out to be claimed with nothing behind them, so `clean`
    is permanently false and this assertion would be permanently red.

    Deleting those twelve findings to get a green test is exactly the dishonesty this module exists
    to prevent, so the bar moved to the thing that actually matters instead: **no capability may
    declare OWNED beyond what its artefacts evidence.** The twelve remaining gaps sit under
    IMPLEMENTED entries, are printed by `python -m argus.eval.standing`, and are a known state
    rather than a hidden one.
    """
    report = audit()
    overstated = sorted({c.name for c in report.owned} - {c.name for c in report.earned})
    assert not overstated, "declared OWNED without evidence: " + ", ".join(overstated)


# --- S18: the legal-transition table ------------------------------------------------------------


def _cap(name: str, state: State) -> Capability:
    proofs = _all_thirteen() if state is State.OWNED else ()
    baseline = "" if state in (State.LOST, State.IMPLEMENTED) else "a named rival"
    return Capability(name=name, subtheme="t", module="argus/eval/standing.py", state=state,
                      baseline=baseline, proofs=proofs)


class TestStateChangesAreLegalMovesOrNothing:
    """LOST -> TIED -> IMPLEMENTED -> OWNED one rung at a time; down any number; nothing else."""

    def test_the_table_climbs_exactly_one_rung_from_every_state(self) -> None:
        for rank, state in enumerate(ORDER):
            up = {s for s in TRANSITIONS[state] if ORDER.index(s) > rank}
            assert up == ({ORDER[rank + 1]} if rank + 1 < len(ORDER) else set()), state

    @pytest.mark.parametrize(("before", "after"), [
        (State.LOST, State.IMPLEMENTED), (State.LOST, State.OWNED), (State.TIED, State.OWNED),
    ])
    def test_a_skipped_rung_is_refused_with_the_legal_targets_named(
            self, before: State, after: State) -> None:
        with pytest.raises(IllegalTransition, match="may move only to") as err:
            check_transition("x", before, after)
        for legal in TRANSITIONS[before] - {before}:
            assert legal.value in str(err.value)

    @pytest.mark.parametrize("before", list(State))
    def test_every_fall_and_every_stay_is_legal(self, before: State) -> None:
        for after in ORDER[: ORDER.index(before) + 1]:
            check_transition("x", before, after)

    @pytest.mark.parametrize("after", list(State))
    def test_a_new_capability_may_enter_at_any_state(self, after: State) -> None:
        check_transition("x", None, after)

    def test_a_table_that_lets_a_state_skip_a_rung_fails_its_own_check(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        widened = {**TRANSITIONS, State.LOST: TRANSITIONS[State.LOST] | {State.OWNED}}
        monkeypatch.setattr(standing, "TRANSITIONS", widened)
        with pytest.raises(IllegalTransition, match="climb exactly one rung"):
            standing._check_transition_table()

    def test_a_table_that_forbids_a_fall_fails_its_own_check(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        narrowed = {**TRANSITIONS, State.OWNED: frozenset({State.OWNED, State.IMPLEMENTED})}
        monkeypatch.setattr(standing, "TRANSITIONS", narrowed)
        with pytest.raises(IllegalTransition, match="fall to every lower rung"):
            standing._check_transition_table()

    def test_changes_and_disappearances_are_both_recorded(self) -> None:
        previous = {"kept": "implemented", "fell": "owned", "renamed away": "tied"}
        register = (_cap("kept", State.IMPLEMENTED), _cap("fell", State.TIED),
                    _cap("new", State.IMPLEMENTED))
        changes = transitions_since(previous, register)
        assert {"capability": "fell", "from": "owned", "to": "tied"} in changes
        assert {"capability": "new", "from": None, "to": "implemented"} in changes
        assert {"capability": "renamed away", "from": "tied", "to": None} in changes
        assert not any(c["capability"] == "kept" for c in changes)

    def test_an_illegal_jump_raises_before_anything_is_written(self, tmp_path: Path) -> None:
        path = tmp_path / "standing.json"
        before = json.dumps({"capabilities": [{"name": "x", "state": "lost"}]})
        path.write_text(before, encoding="utf-8")
        with pytest.raises(IllegalTransition, match="lost -> implemented"):
            write_report(path, report=Report(capabilities=(_cap("x", State.IMPLEMENTED),)))
        assert path.read_text(encoding="utf-8") == before

    def test_a_legal_move_is_appended_to_the_persisted_log(self, tmp_path: Path) -> None:
        path = tmp_path / "standing.json"
        path.write_text(json.dumps({
            "capabilities": [{"name": "x", "state": "lost"}],
            "transition_log": [{"capability": "x", "from": None, "to": "lost", "at": "t0"}],
        }), encoding="utf-8")
        write_report(path, report=Report(capabilities=(_cap("x", State.TIED),)))
        blob = json.loads(path.read_text(encoding="utf-8"))
        assert blob["transitions_this_write"] == [{"capability": "x", "from": "lost", "to": "tied"}]
        assert [e["to"] for e in blob["transition_log"]] == ["lost", "tied"]
        assert blob["transition_log"][-1]["at"]
        # Written again with no change: nothing new is logged.
        write_report(path, report=Report(capabilities=(_cap("x", State.TIED),)))
        assert len(json.loads(path.read_text(encoding="utf-8"))["transition_log"]) == 2

    def test_an_unreadable_previous_register_is_not_silently_replaced(
            self, tmp_path: Path) -> None:
        path = tmp_path / "standing.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(StandingError, match="not readable JSON"):
            write_report(path, report=Report(capabilities=(_cap("x", State.TIED),)))
        assert path.read_text(encoding="utf-8") == "{not json"

    def test_the_live_register_is_a_legal_successor_of_the_persisted_one(self) -> None:
        """Whatever ``data/standing.json`` holds, the register as it stands now must be reachable
        from it by legal moves — otherwise the next ``python -m argus.eval.standing`` raises."""
        blob = json.loads(standing.REPORT_PATH.read_text(encoding="utf-8"))
        previous = {str(c["name"]): str(c["state"]) for c in blob["capabilities"]}
        transitions_since(previous, REGISTER)


# --- S18: no statistical or out-of-sample condition without its breakdown -----------------------

_ARTEFACT = "data/overnight_comparison.json"


def _stat_cap() -> Capability:
    return Capability(
        name="gated", subtheme="t", module="argus/eval/standing.py", state=State.IMPLEMENTED,
        baseline="", proofs=(Proof("statistically_valid_evaluation", "bootstrap",
                                   artefact=_ARTEFACT),))


def _groupwise(**headline: Any) -> dict[str, Any]:
    head = {"name": "h", "role": "argus_vs_rival", "flags": [], "favours": "argus", **headline}
    return {"artefacts": {_ARTEFACT: {
        "status": "checked", "sha256": standing._sha256(standing.PACKAGE / _ARTEFACT),
        "headlines": [head]}}}


class TestTheGroupwiseGate:
    @staticmethod
    def _verdict(groupwise: dict[str, Any] | None) -> tuple[bool, str]:
        cap = _stat_cap()
        return groupwise_verdict(cap, cap.proofs[0], groupwise)

    def test_a_clean_breakdown_on_the_current_artefact_passes(self) -> None:
        ok, why = self._verdict(_groupwise())
        assert ok and why.startswith(f"groupwise-checked on {_ARTEFACT}")

    def test_no_audit_at_all_fails(self) -> None:
        ok, why = self._verdict(None)
        assert not ok and "no groupwise check has run" in why

    def test_an_artefact_the_audit_never_reached_fails(self) -> None:
        ok, why = self._verdict({"artefacts": {}})
        assert not ok and "not in the audit" in why

    def test_designed_cases_cannot_pass(self) -> None:
        ok, why = self._verdict({"artefacts": {_ARTEFACT: {
            "status": "designed_cases", "reason": "cases an author chose"}}})
        assert not ok and "designed_cases" in why

    def test_a_context_headline_alone_does_not_count(self) -> None:
        ok, why = self._verdict(_groupwise(role="context"))
        assert not ok and "only context headlines" in why

    def test_a_headline_carried_by_one_group_fails_and_names_it(self) -> None:
        ok, why = self._verdict(_groupwise(flags=["carried_by_one_group:symbol=NVDAUSDT"]))
        assert not ok and "carried_by_one_group:symbol=NVDAUSDT" in why

    def test_halves_that_flip_fail(self) -> None:
        ok, why = self._verdict(_groupwise(flags=["flips_across_halves:chronological halves"]))
        assert not ok and "flips_across_halves" in why

    def test_a_headline_that_favours_the_rival_fails(self) -> None:
        ok, why = self._verdict(_groupwise(favours="rival"))
        assert not ok and "favours the rival" in why

    def test_an_audit_of_an_older_artefact_is_stale(self) -> None:
        blob = _groupwise()
        blob["artefacts"][_ARTEFACT]["sha256"] = "0" * 64
        ok, why = self._verdict(blob)
        assert not ok and "predates" in why

    def test_the_audit_demotes_an_owned_claim_whose_breakdown_is_missing(
            self, tmp_path: Path) -> None:
        proofs = tuple(
            Proof(c, "bootstrap", artefact=_ARTEFACT) if c in GROUPWISE_CONDITIONS else _proof(c)
            for c in OWNED_CONDITIONS)
        cap = Capability(name="gated", subtheme="t", module="argus/eval/standing.py",
                         state=State.OWNED, baseline="a rival", proofs=proofs)
        empty = tmp_path / "groupwise.json"
        empty.write_text(json.dumps({"artefacts": {}}), encoding="utf-8")
        report = audit((cap,), groupwise_path=empty)
        assert report.owned and not report.earned
        gated = [v for v in report.verifications if v.condition in GROUPWISE_CONDITIONS]
        assert len(gated) == 2 and all(v.status == "UNPROVEN" for v in gated)

        passing = tmp_path / "groupwise_ok.json"
        passing.write_text(json.dumps(_groupwise()), encoding="utf-8")
        report = audit((cap,), groupwise_path=passing)
        gated = [v for v in report.verifications if v.condition in GROUPWISE_CONDITIONS]
        assert all("groupwise-checked on" in v.detail for v in gated)

    def test_every_owned_row_was_broken_down_on_the_live_audit(self) -> None:
        report = audit()
        owned = {c.name for c in report.owned}
        gated = [v for v in report.verifications
                 if v.capability in owned and v.condition in GROUPWISE_CONDITIONS]
        assert gated
        for v in gated:
            assert v.status != "UNPROVEN" and "groupwise-checked on" in v.detail, v.render()


class TestDemotionsCarryTheirRouteBack:
    def test_every_row_the_gate_demoted_says_so_first_and_names_the_route_back(self) -> None:
        demoted = [c for c in REGISTER
                   if c.blockers and "by the groupwise gate" in c.blockers[0]]
        assert len(demoted) == 12
        for cap in demoted:
            assert cap.state is State.IMPLEMENTED, cap.name
            assert cap.blockers[0].startswith(
                "RE-GRADED 2026-09-25 from OWNED to IMPLEMENTED by the groupwise gate"), cap.name
            assert "oute back" in cap.blockers[0], cap.name


# --- verdicts read from the harness, not restated ------------------------------------------------


class TestVerdictsAreRead:
    def test_outcomes_are_the_artefacts_comparison_reports(self) -> None:
        blob = json.loads((standing.PACKAGE / _ARTEFACT).read_text(encoding="utf-8"))
        got = comparison_outcomes(_stat_cap())
        assert [o["outcome"] for o in got] == [r["outcome"] for r in blob["comparison_reports"]]
        assert all(o["artefact"] for o in got)

    def test_an_invalid_report_is_marked_as_not_counted(self) -> None:
        row = {"outcome": "argus_better", "valid": False, "question": "q", "rival": "r",
               "metric": "m", "argus_score": 1, "rival_score": 2, "n": 3, "unit": "night",
               "ci95": None, "p_value": 0.5, "artefact": _ARTEFACT}
        assert "(INVALID: not counted)" in render_outcome(row)
        assert "p = 0.5" in render_outcome(row)

    def test_an_owned_row_contradicted_by_its_own_report_is_not_earned(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        against = ({"outcome": "rival_better", "valid": True, "question": "q", "rival": "r"},)
        monkeypatch.setattr(standing, "comparison_outcomes", lambda cap: against)
        cap = Capability(name="x", subtheme="t", module="argus/eval/standing.py",
                         state=State.OWNED, baseline="r", proofs=_all_thirteen())
        report = audit((cap,))
        assert not report.earned
        assert any(f.problem == "a comparison report records the rival ahead"
                   for f in report.findings)

    def test_the_live_report_shows_each_rows_measured_outcomes(self) -> None:
        report = audit()
        overnight = next(c for c in REGISTER if c.name.startswith("Where a shut stock"))
        assert report.outcomes[overnight.name]
        assert "measured: argus_better" in report.render()


class TestTheEvaluatorSpineFindingsAreApplied:
    @staticmethod
    def _row(prefix: str) -> Capability:
        return next(c for c in REGISTER if c.name.startswith(prefix))

    def test_the_qqq_overnight_sentence_says_argus_leads(self) -> None:
        text = " ".join(self._row("Where a shut stock").blockers)
        assert "1.84bps ahead" in text
        assert "ARGUS is ahead on all eight stocks" in text

    def test_a_row_credits_console_code_only_when_its_harness_runs_it(self) -> None:
        """The credits returned on 2026-09-26 because the harnesses now call the code they name;
        a credit whose harness stops calling it fails here."""
        harnesses = standing.PACKAGE / "src" / "argus" / "eval"
        stop = (harnesses / "stopquality_comparison.py").read_text(encoding="utf-8")
        claims = (harnesses / "claimcheck_comparison.py").read_text(encoding="utf-8")
        if "desk/odds.py" in self._row("Where a stop sits").module:
            assert "directional_odds" in stop
        if "lui/research.py" in self._row("A trader's claims").module:
            assert "_claim_check" in claims

    @pytest.mark.parametrize("artefact", [
        "rule_proposals", "retrieval_diversity", "factor_split_half", "perturbation_robustness",
        "vocab_stress", "feedbugged", "document_qa_eval", "pause_drill", "skill_matrix",
        "guard_selfcheck", "mcp_fuzz", "mcp_sdk_comparison",
    ])
    def test_todays_artefacts_are_cited_on_a_row(self, artefact: str) -> None:
        ref = f"data/{artefact}.json"
        assert (standing.PACKAGE / ref).exists(), ref
        assert any(ref in " ".join((*c.blockers, c.note, *(p.artefact for p in c.proofs)))
                   for c in REGISTER), ref

    def test_the_corrupted_feed_result_stays_recorded_beside_the_gate_that_fixed_it(self) -> None:
        text = " ".join(self._row("Perception layer").blockers)
        assert "detection F1 0.0" in text and "first measurement stays recorded" in text
        assert "12 of 12 planted faults caught" in text and "data/feed_sanity_gate.json" in text

    def test_the_qwen_proposer_is_recorded_as_beating_neither_baseline(self) -> None:
        text = " ".join(self._row("Self-evolving review rules").blockers)
        assert "the model beat neither" in text
