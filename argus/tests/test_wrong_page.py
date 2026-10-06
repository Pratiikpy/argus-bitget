"""``/wrong`` carries every loss the register records, not the one it happened to be written with.

On 2026-09-26 the page listed one comparison lost (regime detection) while the register held
fourteen, and the README and the form both said the Bitget TWAP, Ballast and TweetEval losses were
on it. The entries are now read from the register's own text at request time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from argus.lui.corrections_page import collect, register_losses, register_regrades, render

DATA = Path(__file__).resolve().parents[1] / "data"


def _cap(name: str, state: str, *blockers: str, note: str = "") -> dict[str, Any]:
    return {"name": name, "state": state, "blockers": list(blockers), "note": note, "proofs": []}


class TestLossesFromTheRegister:
    def test_a_capitalised_loss_is_an_entry_and_a_lower_case_one_is_not(self) -> None:
        standing = {"capabilities": [
            _cap("Order splitting", "tied", "LOST first, then TIED. The schedule cost 12.2bps."),
            _cap("Abstention", "implemented", "a large avoided loss and a blocked upside offset"),
        ]}
        found = register_losses(standing)
        assert [c.headline for c in found] == ["Lost, then rebuilt to a tie: Order splitting"]
        assert "12.2bps" in found[0].detail and found[0].kind == "loss"

    def test_an_open_loss_says_it_is_open(self) -> None:
        found = register_losses({"capabilities": [
            _cap("Arbitrage", "implemented", "LOSS for the deployed decompose(): 49 of 52.")]})
        assert found[0].headline == "Lost to a rival, still open: Arbitrage"

    def test_the_regrade_line_is_not_counted_as_a_loss(self) -> None:
        found = register_losses({"capabilities": [
            _cap("X", "implemented", "RE-GRADED 2026-09-25 from OWNED to IMPLEMENTED; LOST?")]})
        assert found == []

    def test_regrades_are_grouped_by_the_review_that_made_them(self) -> None:
        standing = {"capabilities": [
            _cap("A", "implemented", "RE-GRADED 2026-09-24 from OWNED to IMPLEMENTED: weak rival"),
            _cap("B", "implemented", "RE-GRADED 2026-09-25 from OWNED to IMPLEMENTED by the gate"),
            _cap("C", "implemented", "RE-GRADED 2026-09-25 from OWNED to IMPLEMENTED by the gate"),
            _cap("D", "tied", "RE-GRADED 2026-09-26 from OWNED to TIED: a validator ties it"),
            _cap("E", "owned", "no re-grade"),
        ]}
        heads = [c.headline for c in register_regrades(standing)]
        assert heads[0].startswith("1 OWNED grade withdrawn on 2026-09-24")
        assert heads[1].startswith("2 OWNED grades withdrawn on 2026-09-25")
        assert heads[2].startswith("1 OWNED grade withdrawn on 2026-09-26")


class TestTheLivePageKeepsItsPromises:
    def test_every_loss_the_readme_names_is_on_the_page(self) -> None:
        text = " ".join(f"{c.headline} {c.detail}" for c in collect(DATA))
        # README: the TWAP and Ballast losses "stay on /wrong"; the form cites TweetEval there
        for marker in ("Bitget's own TWAP", "Ballast", "TweetEval"):
            assert marker in text, marker

    def test_every_register_loss_has_an_entry(self) -> None:
        standing = json.loads((DATA / "standing.json").read_text(encoding="utf-8"))
        names = {c.headline.split(": ", 1)[1] for c in register_losses(standing)}
        on_page = " ".join(c.headline for c in collect(DATA))
        assert names and all(name in on_page for name in names)

    def test_the_regrade_counts_add_up_to_the_register(self) -> None:
        standing = json.loads((DATA / "standing.json").read_text(encoding="utf-8"))
        regraded = sum(
            1 for c in standing["capabilities"]
            if any(str(b).startswith("RE-GRADED") and "from OWNED" in str(b)
                   for b in c.get("blockers", [])))
        counted = sum(int(c.headline.split(" ", 1)[0]) for c in register_regrades(standing))
        assert counted == regraded

    def test_it_renders(self) -> None:
        page = render(collect(DATA))
        assert page.count("a baseline beat us") >= 10

    def test_a_plain_summary_sits_directly_above_the_first_entry(self) -> None:
        """First-user audit, 2026-09-29: the first thing a retail trader read on this page, cold,
        was "N ledger rows booked P&L on positions the risk layer refused" — a bug report with no
        framing. A plain-language box now sits directly above the first entry, saying why a page
        of losses exists at all; its count is read from the same `corrections` list the rows are
        built from, never typed separately."""
        found = collect(DATA)
        page = render(found)
        plain_at = page.index("<div class='plain'>")
        first_entry_at = page.index("<details class='c ")
        assert plain_at < first_entry_at
        box = page[plain_at:page.index("</div>", plain_at)]
        assert "trading tool that only tells you what went right" in box
        assert f"{len(found)} entries" in box
        lost = sum(1 for c in found if c.kind == "loss")
        assert f"{lost} of them" in box
        # The plain summary leads the page, ahead of the intro on how entries are read: a
        # first-time user found the page dense before reaching it (round 10, 2026-09-30). The
        # intro stays, after it.
        assert plain_at < page.index("Every entry is read out of the artefact") < first_entry_at

    def test_an_empty_record_still_carries_the_plain_summary(self) -> None:
        page = render([])
        assert "<div class='plain'>" in page
        assert "0 entries follow, 0 of them" in page


def test_a_removed_capability_is_listed_with_its_reason() -> None:
    from argus.lui.corrections import register_removals

    out = register_removals({"removed": [{
        "name": "A row", "removed_on": "2026-09-29", "last_state": "implemented",
        "reason": "It serves no flow.", "revive_by": "a flow that uses it"}]})
    assert len(out) == 1
    assert out[0].headline == "Removed from the register on 2026-09-29 (IMPLEMENTED): A row"
    assert "It serves no flow." in out[0].detail and "a flow that uses it" in out[0].detail
    assert out[0].kind == "withdrawn"


class TestRound45Minors:
    """Round 45 visual audit, minors 4 and 7."""

    def test_backticks_become_code_and_none_is_left_literal(self) -> None:
        from argus.lui.corrections_page import _prose

        said = _prose("`paper/runner.py` recorded <b>")
        assert said == "<code>paper/runner.py</code> recorded &lt;b&gt;"

    def test_a_cut_tail_is_taken_back_to_a_whole_word_or_sentence(self) -> None:
        from argus.lui.corrections_page import _whole

        assert _whole("A day-clustere...") == "A…"
        assert _whole("Whole text.") == "Whole text."
        long = "First sentence ends here. second sentence runs on until it is cut mid-wor..."
        assert _whole(long).endswith("cut…")
        sentence = "One full sentence of some length that matters. Tail frag..."
        assert _whole(sentence) == "One full sentence of some length that matters. …"
        assert "..." not in _whole(long)

    def test_every_entry_folds_behind_its_headline(self) -> None:
        page = render(collect(DATA))
        assert page.count("<details class='c ") == len(collect(DATA))
        assert "<article" not in page
