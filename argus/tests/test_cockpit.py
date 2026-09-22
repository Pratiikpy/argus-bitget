"""Cockpit tests — the page a judge opens must not be able to lie, and it already did.

The hand-written `cockpit.html` this replaces said "The paper ledger is 2 decisions old" while the
ledger held 178. It was the most visible artefact in the project and the only one outside
`eval/docclaims.py`, the gate built to stop precisely that.

Two properties are defended here. **No figure may disagree with its artefact** — checked by
regenerating and comparing the committed file against freshly built values, which neither quantstats
nor agent-backtest-lab does. And **an unavailable figure must say so** — quantstats calls
`.fillna(0)` before rendering (`quantstats/reports.py:1289`), so a strategy with no trades reports a
Sharpe of 0.0; agent-backtest-lab guards with `np.isfinite` and prints an em-dash
(`abl/scorecard/html_render.py:160`), honest but silent about why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.demo.cockpit import (
    OUTPUT_PATH,
    PANELS,
    UNAVAILABLE,
    Cockpit,
    Metric,
    Panel,
    build,
    render,
)


def _page(*panels: Panel) -> str:
    from datetime import UTC, datetime

    return render(Cockpit(generated_at=datetime(2026, 9, 14, tzinfo=UTC), panels=panels))


class TestAnUnavailableFigureSaysSo:
    def test_a_none_value_renders_as_not_available(self) -> None:
        page = _page(Panel("P", "t", (Metric("Sharpe", None, "src"),)))
        assert UNAVAILABLE in page

    def test_it_never_renders_as_a_zero(self) -> None:
        """quantstats fillna(0)s before rendering, so a strategy with no trades reports Sharpe 0.0.
        On this project's most-read page that would be the strongest false claim in it."""
        page = _page(Panel("P", "t", (Metric("Sharpe", None, "src"),)))
        assert ">0<" not in page and ">0.0<" not in page and ">0.00<" not in page

    def test_it_never_renders_as_a_blank_or_a_dash(self) -> None:
        """An em-dash is honest but reads as a formatting choice. A reader must be told."""
        page = _page(Panel("P", "t", (Metric("Sharpe", None, "src"),)))
        assert "<dd></dd>" not in page
        assert "<dd>—</dd>" not in page

    def test_an_available_zero_is_still_shown_as_zero(self) -> None:
        """A measured zero and an unmeasurable figure are different, and both must render truly."""
        page = _page(Panel("P", "t", (Metric("violations", "0", "src"),)))
        assert ">0<" in page
        assert UNAVAILABLE not in page.split("</header>")[1]

    def test_undefined_figures_are_listed_by_name(self) -> None:
        cockpit = Cockpit(
            generated_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            panels=(Panel("P", "t", (Metric("Sharpe", None, "s"), Metric("n", "3", "s"))),),
        )
        assert cockpit.undefined == ("P · Sharpe",)


class TestAMissingArtefactDegradesRatherThanFails:
    def test_a_panel_with_no_artefact_states_the_reason(self) -> None:
        page = _page(Panel("P", "t", (), missing_reason="data/x.json is not on disk"))
        assert "data/x.json is not on disk" in page

    def test_a_panel_whose_builder_raises_becomes_an_absent_panel(self) -> None:
        """One broken panel must not take the page down: a cockpit that fails to render shows a
        judge nothing at all, and one missing card shows them eleven."""
        def explode() -> Panel:
            raise RuntimeError("artefact moved")

        got = build([explode])
        assert len(got.panels) == 1
        assert "RuntimeError" in got.panels[0].missing_reason
        assert "artefact moved" in got.panels[0].missing_reason

    def test_the_page_still_renders_with_every_panel_broken(self) -> None:
        def explode() -> Panel:
            raise RuntimeError("gone")

        page = render(build([explode, explode]))
        assert "<html" in page and "ARGUS Decision Cockpit" in page


class TestTheCommittedPageAgreesWithTheArtefacts:
    """The check neither prior implementation has: regenerate, then compare."""

    @pytest.fixture(scope="class")
    def committed(self) -> str:
        if not OUTPUT_PATH.exists():
            pytest.skip("cockpit.html has not been generated")
        return OUTPUT_PATH.read_text(encoding="utf-8")

    def test_every_live_figure_appears_on_the_committed_page(self, committed: str) -> None:
        """A value that has moved since the page was written fails here by name."""
        stale: list[str] = []
        for panel in build().panels:
            for metric in panel.metrics:
                if metric.available and str(metric.value) not in committed:
                    stale.append(f"{panel.title} · {metric.label} = {metric.value}")
        assert not stale, f"cockpit.html disagrees with the artefacts: {stale}"

    def test_the_ledger_count_on_the_page_is_the_live_one(self, committed: str) -> None:
        """The exact defect that motivated this module."""
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import LEDGER_PATH

        if not LEDGER_PATH.exists():
            pytest.skip("no live ledger")
        live = len(PaperLedger(path=LEDGER_PATH).entries)
        assert f"<dd>{live}</dd>" in committed

    def test_the_page_never_claims_a_settled_performance_figure(self, committed: str) -> None:
        from argus.paper.ledger import PaperLedger
        from argus.paper.runner import LEDGER_PATH

        if not LEDGER_PATH.exists():
            pytest.skip("no live ledger")
        settled = PaperLedger(path=LEDGER_PATH).performance().get("settled_trades", 0)
        if not settled:
            assert UNAVAILABLE in committed


class TestThePanelCanActuallyPopulate:
    """The assertion whose absence let the Track 2 panel be permanently disconnected.

    `performance_panel` read `PaperLedger.performance()`, which returns neither `sharpe` nor
    `max_drawdown` under any branch and returns `win_rate_pct` where the panel asked for
    `win_rate`. All three resolved to `None` whatever the ledger held.

    **Nothing failed and nothing would have.** With zero settled trades "unavailable" is the
    correct output, so the page read perfectly while being disconnected from its own data, and the
    first settled position would not have changed a character. Every test here was written against
    the *empty* record and so could not tell the two states apart. This class asserts the one thing
    that separates them: **that the numbers appear when the data exists.**
    """

    def _panel_with_one_settled_trade(self, tmp_path, monkeypatch, pnl_sequence=None):
        from datetime import UTC, datetime, timedelta

        from argus.demo import cockpit as mod
        from argus.paper.ledger import Entry, PaperLedger

        day = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)

        def _entry(seq, decided, settled, net):
            return Entry(
                seq=seq, decided_at=decided.isoformat(), symbol="NVDAUSDT",
                verdict="open_long", side="long", quantity="1", entry_price="100",
                stated_confidence=0.7, thesis="fixture", invalidation=("below 95",),
                market_state_hash="m" * 16, approved_intent_hash="a" * 16,
                session_phase="rth", hours_to_discovery=0.0, entry_cost_bps="6",
                prev_hash="0" * 16, settled_at=settled.isoformat(), exit_price="101",
                gross_pnl=net, net_pnl=net,
                direction_correct=True,
            )

        path = tmp_path / "ledger.jsonl"
        ledger = PaperLedger(path=path)
        pnls = pnl_sequence or ["120", "-40", "65"]
        # `entries` is a read-only, decisions-only *view* over `_raw_entries` (2026-09-22,
        # settlement seals) — injecting a synthetic fixture writes the real backing field instead.
        ledger._raw_entries = [
            _entry(i + 1, day + timedelta(days=i), day + timedelta(days=i + 1), p)
            for i, p in enumerate(pnls)
        ]
        monkeypatch.setattr(mod, "_absent", mod._absent)
        monkeypatch.setattr("argus.paper.runner.LEDGER_PATH", path)
        path.write_text("", encoding="utf-8")
        monkeypatch.setattr("argus.paper.ledger.PaperLedger", lambda **kw: ledger)
        return mod.performance_panel()

    def test_the_three_scored_numbers_appear_once_trades_have_settled(
        self, tmp_path, monkeypatch
    ) -> None:
        panel = self._panel_with_one_settled_trade(tmp_path, monkeypatch)
        by_label = {m.label: m for m in panel.metrics}
        assert by_label["settled trades"].value == "3"
        # The point of the test: at least one of the three scored figures must now carry a value.
        # Before the fix every one of them was None regardless of the ledger.
        scored = [by_label["Sharpe"], by_label["max drawdown"], by_label["win rate"]]
        assert any(m.available for m in scored), (
            "none of Sharpe / max drawdown / win rate populated with three settled trades — "
            "the panel is disconnected from its data source"
        )

    def test_win_rate_populates_rather_than_hitting_a_key_mismatch(
        self, tmp_path, monkeypatch
    ) -> None:
        """`win_rate` vs `win_rate_pct` — the mismatch that made this permanently blank."""
        panel = self._panel_with_one_settled_trade(tmp_path, monkeypatch)
        win_rate = next(m for m in panel.metrics if m.label == "win rate")
        assert win_rate.available, "win rate is blank with three settled trades"

    def test_max_drawdown_populates_once_the_sample_supports_it(
        self, tmp_path, monkeypatch
    ) -> None:
        """Undefined on a thin sample by design; with enough trades it must be a real figure.

        Six trades, not three: `MIN_TRADES_FOR_DRAWDOWN` is 5, because two or three winners
        produce a 0.00% drawdown that reads as *"took risk, never lost"*. The mechanic under test
        is unchanged — the panel must render a real number once the record can carry one.
        """
        panel = self._panel_with_one_settled_trade(
            tmp_path, monkeypatch, pnl_sequence=["120", "-40", "65", "-25", "80", "-15"],
        )
        drawdown = next(m for m in panel.metrics if m.label == "max drawdown")
        assert drawdown.available, "max drawdown is blank with six settled trades"

    def test_a_thin_sample_shows_win_rate_but_withholds_sharpe(self, tmp_path, monkeypatch) -> None:
        """The real 2026-09-20 shape, pinned on the page itself: the panel must not print a
        Sharpe off two settled trades. It briefly did — 6.75 — beside prose calling it undefined."""
        panel = self._panel_with_one_settled_trade(
            tmp_path, monkeypatch, pnl_sequence=["120", "90"],
        )
        by_label = {m.label: m for m in panel.metrics}
        assert by_label["win rate"].available
        assert by_label["win rate"].value == "100.0%", "a fraction rendered with a % suffix"
        assert not by_label["Sharpe"].available
        assert not by_label["max drawdown"].available

    def test_a_withheld_figure_carries_its_reason_not_just_a_blank(self) -> None:
        """A judge learns more from why a number is missing than from the gap where it was."""
        from argus.demo.cockpit import performance_panel
        from argus.paper.runner import LEDGER_PATH

        if not LEDGER_PATH.exists():
            pytest.skip("no live ledger")
        panel = performance_panel()
        blanks = [m for m in panel.metrics if not m.available]
        if blanks:
            assert any("evaluate_ledger" in m.source for m in blanks)


class TestItIsSelfContained:
    def test_the_page_makes_no_network_request(self, tmp_path: Path) -> None:
        """The previous cockpit pulled two webfonts from Google, so the "accessible demo" rendered
        differently, or not at all, behind a firewall."""
        page = _page(Panel("P", "t", (Metric("a", "1", "s"),)))
        for marker in ("http://", "https://", "//fonts", "src=\"/", "<script"):
            assert marker not in page, marker

    def test_the_styles_are_inline(self) -> None:
        page = _page(Panel("P", "t", (Metric("a", "1", "s"),)))
        assert "<style>" in page and "<link" not in page

    def test_it_declares_a_viewport_and_survives_a_phone_width(self) -> None:
        page = _page(Panel("P", "t", (Metric("a", "1", "s"),)))
        assert 'name="viewport"' in page
        assert "@media (max-width:420px)" in page

    def test_it_carries_both_light_and_dark_palettes(self) -> None:
        page = _page(Panel("P", "t", (Metric("a", "1", "s"),)))
        assert ":root{" in page
        assert "prefers-color-scheme:dark" in page


class TestContentIsEscaped:
    def test_a_value_containing_markup_cannot_inject(self) -> None:
        page = _page(Panel("P", "t", (Metric("x", "<script>alert(1)</script>", "s"),)))
        assert "<script>" not in page
        assert "&lt;script&gt;" in page

    def test_a_panel_title_is_escaped(self) -> None:
        page = _page(Panel("<b>P</b>", "t", ()))
        assert "<b>P</b>" not in page

    def test_terminal_emphasis_renders_as_bold_not_as_asterisks(self) -> None:
        """Several verdicts come from modules whose primary surface is a terminal, where `**x**` is
        this codebase's emphasis convention. `eval/autopsy.py`'s falsifier says `**FIRED.**`, and it
        reached the public page as four literal asterisks."""
        page = _page(Panel("P", "t", (), verdict="the falsifier **FIRED.** on 216 decisions"))
        assert "<strong>FIRED.</strong>" in page
        assert "**" not in page

    def test_emphasis_never_becomes_an_injection_vector(self) -> None:
        """Escaping happens before the asterisks are paired, so markup inside an emphasised span is
        still inert — otherwise this would be a way to smuggle tags in through an artefact."""
        page = _page(Panel("P", "t", (), verdict="**<script>alert(1)</script>**"))
        assert "<script>" not in page
        assert "&lt;script&gt;" in page

    def test_an_unbalanced_asterisk_run_is_left_exactly_as_written(self) -> None:
        """Guessing where the author meant to close emphasis would silently rewrite a verdict."""
        page = _page(Panel("P", "t", (), verdict="a **dangling run of emphasis"))
        assert "a **dangling run of emphasis" in page
        assert "<strong>" not in page

    def test_the_published_page_carries_no_literal_asterisk_pairs(self) -> None:
        """The real page, not a fixture — this is the surface a judge opens."""
        assert "**" not in render(build())


class TestTheEvidenceIsTraceable:
    def test_every_metric_names_the_file_it_came_from(self) -> None:
        for panel in build().panels:
            for metric in panel.metrics:
                assert metric.source.strip(), f"{panel.title} · {metric.label}"

    def test_the_sources_are_rendered_for_the_reader(self) -> None:
        page = _page(Panel("P", "t", (Metric("a", "1", "data/x.json"),)))
        assert "data/x.json" in page

    def test_every_registered_panel_builds(self) -> None:
        got = build()
        assert len(got.panels) == len(PANELS)
        assert not got.missing, f"artefacts absent: {got.missing}"

    def test_the_required_track_materials_are_present(self) -> None:
        """Track 2's event->decision->execution flow and Track 3's research task are both
        submission requirements, so the demo must evidence both."""
        titles = {p.title for p in build().panels}
        assert "event to decision to execution" in titles
        assert "One complete research task" in titles

    def test_the_report_serialises(self) -> None:
        import json

        blob = json.loads(json.dumps(build().as_dict()))
        assert blob["panels"]
        assert "undefined_metrics" in blob
