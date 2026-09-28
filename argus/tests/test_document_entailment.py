"""The whole-claim entailment check and ALCE's leave-one-out citation precision in
`research/document_qa.enforce` (research/harvest/07-alce.md), with a scripted entailer so the
logic is tested without a model."""

from __future__ import annotations

from datetime import date
from typing import Any

from argus.research import document_qa as dq


def chunks() -> list[dq.Chunk]:
    def doc(text: str, doc_id: str) -> dq.Document:
        return dq.Document(doc_id=doc_id, ticker="NVDA", form="10-Q", filed=date(2026, 8, 27),
                           url=f"https://example.test/{doc_id}", title=doc_id, text=text)

    return [*dq.chunk(doc("Revenue was $46.7 billion in the quarter.", "d-rev")),
            *dq.chunk(doc("Demand for AI systems drove the increase.", "d-why")),
            *dq.chunk(doc("The company leases offices in Santa Clara.", "d-lease"))]


def entailer(supports: dict[str, set[str]]) -> tuple[dq.Entailer, list[str]]:
    """Entailed when every fact the claim needs is in the premise text."""
    calls: list[str] = []

    def entail(premise: str, hypothesis: str) -> bool:
        calls.append(hypothesis)
        return all(fact in premise for fact in supports.get(hypothesis, {"__never__"}))
    return entail, calls


def test_a_claim_the_passages_do_not_support_as_a_whole_is_dropped() -> None:
    rev, why, lease = chunks()
    raw = (f"Revenue was $46.7 billion ({rev.id}). "
           f"Revenue was $46.7 billion because of AI demand ({rev.id}).")
    entail, _ = entailer({"Revenue was $46.7 billion.": {"$46.7 billion"},
                          "Revenue was $46.7 billion because of AI demand.":
                              {"$46.7 billion", "AI systems"}})
    out = dq.enforce("q", raw, [rev, why, lease], entail=entail)
    assert [c.text for c in out.claims] == ["Revenue was $46.7 billion."]
    assert [(d.reason, d.sentence) for d in out.dropped] == [
        ("not_entailed", "Revenue was $46.7 billion because of AI demand.")]


def test_an_unneeded_citation_is_removed_and_a_needed_pair_kept() -> None:
    rev, why, lease = chunks()
    claim = "Revenue was $46.7 billion, driven by AI demand."
    raw = f"{claim} ({rev.id}, {why.id}, {lease.id})"
    entail, _ = entailer({claim: {"$46.7 billion", "AI systems"}})
    out = dq.enforce("q", raw, [rev, why, lease], entail=entail)
    assert out.claims[0].cited == (rev.id, why.id)
    assert out.uncited_by_precision == [lease.id]


def test_the_call_cap_leaves_claims_marked_unverified(monkeypatch: Any) -> None:
    rev, why, lease = chunks()
    monkeypatch.setattr(dq, "MAX_ENTAIL_CALLS", 1)
    raw = f"Revenue was $46.7 billion ({rev.id}). Offices are in Santa Clara ({lease.id})."
    entail, calls = entailer({"Revenue was $46.7 billion.": {"$46.7 billion"}})
    out = dq.enforce("q", raw, [rev, why, lease], entail=entail)
    assert len(calls) == 1 and out.unverified == ["Offices are in Santa Clara."]
    assert len(out.claims) == 2


def test_without_an_entailer_nothing_changes() -> None:
    rev, why, lease = chunks()
    raw = f"Revenue was $46.7 billion because of AI demand ({rev.id}, {lease.id})."
    out = dq.enforce("q", raw, [rev, why, lease])
    assert out.claims[0].cited == (rev.id, lease.id) and out.entail_calls == 0
