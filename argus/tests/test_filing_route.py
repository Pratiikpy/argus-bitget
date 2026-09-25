"""A filing question in the console is answered from the filing, citations enforced (offline)."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pytest

from argus.llm.qwen import Completion, Usage
from argus.lui import server
from argus.lui.provenance import labels
from argus.research import document_qa as dq

FILING = ("Data Center revenue was $89.0 billion, up 117% from a year ago, driven by Blackwell "
          "Ultra.\nWe had approximately 42,000 employees in 38 countries.\n")


class _Cites:
    """Cites the first passage it was shown, once truthfully and once with an invented id."""

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        shown = re.findall(r"dqa-[0-9a-f]{8}", messages[-1]["content"])
        text = (f"Data Center revenue was $89.0 billion, up 117%, driven by Blackwell Ultra "
                f"({shown[0]}). Margins will double next year (dqa-deadbeef).")
        return Completion(content=text, reasoning="", finish_reason="stop",
                          usage=Usage(10, 10, 0, 20))


@pytest.fixture
def filings(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = dq.Document(doc_id="NVDA-10-Q-1", ticker="NVDA", form="10-Q", filed=date(2026, 8, 26),
                      url="https://www.sec.gov/x.htm", title="10-Q", text=FILING)
    monkeypatch.setattr(dq.EdgarDocuments, "latest",
                        lambda self, ticker: [doc] if ticker == "NVDA" else [])
    monkeypatch.setattr(server, "_filing_model", lambda: _Cites())
    monkeypatch.setattr(server, "_model_for", lambda visitor: object())
    server._FILINGS.clear()


def test_a_filing_question_is_answered_from_the_filing_and_an_invented_cite_is_removed(
        filings: None) -> None:
    payload = server.handle_ask("what did NVDA's latest 10-Q say drove data center revenue?", [],
                                visitor="t")
    assert payload["classified_by"] == "research-filing"
    text = "\n".join(payload["lines"])
    assert "$89.0 billion" in text and "Margins will double" not in text
    assert len(payload["sources"]) == 1
    tags = labels(payload["lines"])
    assert tags[payload["lines"].index(next(x for x in payload["lines"] if "$89.0" in x))] == "live"


def test_a_coin_with_no_filings_takes_the_usual_route(filings: None) -> None:
    payload = server.handle_ask("what does the BTC annual report disclose", [], visitor="t")
    assert payload["classified_by"] != "research-filing"
