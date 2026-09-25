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
