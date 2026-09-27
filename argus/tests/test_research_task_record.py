"""The worked research task kept as an artefact (`eval/research_task_record.py`).

The run itself needs the network, so these read the committed artefact: its headline figures must
be the figures its own verdict states, and the verdict must be the one the page would compose from
the data it kept. Nothing here reaches a market.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from argus.eval import research_task_record as rec
from argus.lui.task import EXECUTION_TITLE, IMPACT_TITLE


@pytest.fixture(scope="module")
def blob() -> dict[str, Any]:
    return dict(json.loads(rec.REPORT_PATH.read_text(encoding="utf-8")))


def test_it_is_the_question_the_submission_quotes(blob: dict[str, Any]) -> None:
    assert blob["task"]["question"] == rec.QUESTION
    assert blob["task"]["read_as"]["summary"] == "add 15% TSLA to 40% NVDA / 30% MSFT / 30% AAPL"


def test_every_engine_answered_or_said_why_not(blob: dict[str, Any]) -> None:
    steps = blob["task"]["steps"]
    assert len(steps) == 8
    assert all(step["lines"] for step in steps)
    assert {IMPACT_TITLE, EXECUTION_TITLE} <= set(blob["data"])


def test_the_headline_is_what_the_verdict_says(blob: dict[str, Any]) -> None:
    h = rec.headline(blob)
    verdict = blob["task"]["verdict"]
    assert verdict["call"] == f"Add, at {h['proposed']:.0f}%"
    first, crowded, fill, *_later = verdict["lines"]
    assert (f"carry {h['share_after']:.0f}% of the book's risk, inside the {h['budget']:.0f}% "
            f"budget; {h['ceiling']:.0f}% is the most") in first
    assert (f"NVDA carries {h['crowded_share']:.0f}% of the risk at "
            f"{h['crowded_weight']:.0f}% of the money") in crowded
    assert f"trimming NVDA to {h['crowded_trim_to']:.0f}%" in crowded
    assert (f"about {h['sliced_bps']:.1f} bps all in on the live book (one market order: "
            f"{h['single_order_bps']:.1f} bps)") in fill


def test_the_execution_step_states_the_single_order_figure(blob: dict[str, Any]) -> None:
    """The verdict's one-order cost is the execution step's own first line, not a third figure."""
    h = rec.headline(blob)
    execution = next(s for s in blob["task"]["steps"] if s["title"] == EXECUTION_TITLE)
    assert f"costs about {h['single_order_bps']:.1f}bps" in execution["lines"][0]
